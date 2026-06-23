"""Play an action chunk on a robot backend at the control rate.

The denormalization contract is the load-bearing detail (the *denorm-must-match-training* rule —
fudge factors like ``*5000`` are symptoms of getting THIS wrong, not real tuning):

  * ``actions_already_metric=True`` (e.g. MotionPolicyDroidNative — its ``get_final_action`` is a
    pass-through because the Jacobian was denormalized server-side, so the solve already yields
    physical du): the client MUST NOT re-scale. Apply as-is.
  * ``actions_already_metric=False``: multiply each dim by ``action_abs_scale`` to recover physical du.

Gripper: when ``gripper_is_raw`` the server sends a raw float; binarize ``>0.5 -> close``. The
gripper dim is never multiplied by an arm scale. See CONTROLLER_SPEC §4.
"""
from __future__ import annotations

import logging
import time
from typing import List, Optional

import numpy as np

from vera.controller.robot_iface import RobotBackend
from vera.server.protocol.server_config import VeraServerConfig

logger = logging.getLogger("vera.controller.action")


class GripperZGate:
    """Adaptive gripper trigger for weak plan_chunk signals (DEPLOY_LOG entry #2 → operator spec).

    Welford running stats over the raw gripper channel, seeded with a prior (μ₀, σ₀, pseudo-count
    n₀) so early steps aren't self-referential. Close when z > +k, open when z < −k, otherwise
    HOLD the previous state — state persists across chunks within an episode (reset per episode).
    Replaces the fixed >0.5 binarize, which weak raw signals (|max| ≈ 0.07) can never reach.
    """

    def __init__(self, *, mu0: float = 0.0, sigma0: float = 0.05, n0: int = 70, k: float = 2.0,
                 arrival_dz: Optional[float] = None,
                 open_k: Optional[float] = None, open_steps: int = 1,
                 descend_thresh: float = 0.003) -> None:
        self._mu0, self._sigma0, self._n0, self.k = float(mu0), float(sigma0), int(n0), float(k)
        # Asymmetric open hysteresis (anti mid-carry drop): while CLOSED, an open commits only
        # after `open_steps` consecutive steps below -open_k. Closing stays single-step. A lone
        # negative flicker in the dream's carry phase then cannot drop the payload.
        self.open_k = float(open_k) if open_k is not None else float(k)
        self.open_steps = max(1, int(open_steps))
        # state-conditioned close (operator entry-#4 reply): the dream's close intent fires a
        # few cm ABOVE the cube (monocular wrist-view depth ambiguity), so committing on intent
        # alone closes mid-air. With arrival_dz set, a close commits only once the descent has
        # decayed (dz > -arrival_dz, m/step): "close when the dream says so AND we've arrived".
        # NOTE the dz fed here is the dream's PRE-mps descent intent (caller divides the
        # mps-amplified command by motion_plan_scale, Bug #2 fix 2026-06-12), so the arrival
        # threshold means the same thing regardless of the mps speed knob.
        self.arrival_dz = float(arrival_dz) if arrival_dz is not None else None
        # Bug #3 fix (2026-06-12): a latched close intent that sees the arm clearly ASCENDING
        # (dz > descend_thresh) is dropped — the dream gave up this grasp point and the stale
        # intent must not fire a mid-air close on a later non-descending step (run #6 chunk 4
        # closed at dz=+0.038 while lifting). descend_thresh = "genuinely ascending" margin
        # (m/step, pre-mps); a fresh close spike + real descent must re-arm.
        self._descend_thresh = float(descend_thresh)
        self.reset()

    def reset(self) -> None:
        self._n = float(self._n0)
        self._mean = self._mu0
        self._m2 = (self._sigma0 ** 2) * self._n0
        self.closed = False
        self.close_intent = False
        self._open_run = 0

    def update(self, raw: float, dz: Optional[float] = None) -> tuple:
        """Feed one raw gripper value (and the step's PRE-mps commanded z delta, m/step);
        returns (z, closed)."""
        self._n += 1.0
        d = raw - self._mean
        self._mean += d / self._n
        self._m2 += d * (raw - self._mean)
        sigma = max((self._m2 / self._n) ** 0.5, 1e-6)
        z = (raw - self._mean) / sigma
        if z > self.k:
            self.close_intent = True
            self._open_run = 0
        elif z < -self.open_k:
            self._open_run += 1
            if not self.closed or self._open_run >= self.open_steps:
                self.close_intent = False
                self.closed = False
        else:
            self._open_run = 0
        if self.close_intent and not self.closed:
            if self.arrival_dz is not None and dz is not None and dz > self._descend_thresh:
                # Bug #3 (2026-06-12): the arm is clearly ASCENDING — the dream gave up this
                # grasp point (judged it arrived too high and moved on). Drop the stale latched
                # intent so it can't fire as a mid-air close later (run #6 chunk 4 closed at
                # dz=+0.038 while lifting). A fresh close spike + real descent must re-arm it.
                self.close_intent = False
            else:
                # arrival: descent has decayed (dz > -arrival_dz). Descending steps (dz <=
                # -arrival_dz) hold the intent pending; the ascending case is handled above.
                arrived = (self.arrival_dz is None or dz is None or dz > -self.arrival_dz)
                if arrived:
                    self.closed = True
        return z, self.closed


class ActionPlayer:
    def __init__(self, config: VeraServerConfig) -> None:
        self.action_space = config.action_space
        self.control_dt = float(config.control_dt)
        self.already_metric = bool(config.actions_already_metric)
        self.abs_scale = np.asarray(config.action_abs_scale, dtype=np.float32) if config.action_abs_scale else None
        self.gripper_is_raw = bool(config.gripper_is_raw)
        self.gripper_dim = int(config.gripper_dim_index)
        self.z_gate: Optional[GripperZGate] = None   # off by default; enable_z_gate() to opt in
        # Chunk-level translation budget (meters of total path per chunk). The per-step clamp
        # alone cannot bound sustained speed: 10 consecutive clamped 3 cm steps = 0.45 m/s for a
        # whole chunk (the "too fast" dive of 2026-06-11 21:39, mps=1.5 hand-only). If the chunk's
        # summed |xyz| exceeds this, translation is scaled down uniformly (direction/shape kept) —
        # the old droid stack's max-trans guard (validated 0.15-0.21 in the May recipes).
        self.max_chunk_trans: Optional[float] = None
        # Server motion_plan_scale (mps), mirrored here so the z-gate arrival test can divide
        # the mps-amplified commanded descent back to the dream's PRE-mps intent (Bug #2 fix
        # 2026-06-12) — otherwise the close-commit moment silently shifts as mps changes.
        self.gripper_mps: float = 1.0
        # Per-step translation clamp applied downstream by the backend (m). Used only to compute
        # honest rails telemetry (rails_bound / executed_trans) for the server's adaptive gain.
        self.step_trans_clamp: float = 0.03
        # Rails telemetry from the LAST played chunk (anti-windup feedback, operator 21a8dc1):
        # whether any rail clipped translation, and the net post-clip translation path actually
        # commanded. Sent with the NEXT infer so the gain estimator skips rails-bound evidence.
        self.last_rails_bound: Optional[bool] = None
        self.last_executed_trans: Optional[float] = None
        if not self.already_metric and self.abs_scale is None:
            logger.warning("actions not metric AND no action_abs_scale advertised — applying raw (likely wrong)")

    def enable_z_gate(self, k: float = 2.0, arrival_dz: Optional[float] = None) -> None:
        self.z_gate = GripperZGate(k=k, arrival_dz=arrival_dz)
        logger.info("[z-gate] enabled: k=%.2f prior(mu=0.0, sigma=0.05, n0=70) arrival_dz=%s",
                    k, arrival_dz)

    def reset_episode_state(self) -> None:
        if self.z_gate is not None:
            self.z_gate.reset()

    def denormalize(self, action: np.ndarray) -> np.ndarray:
        """(H,D) wire action -> (H,D) physical command. No-op scale when already metric.

        The gripper dim is NEVER multiplied by an arm scale and is binarized from the RAW wire
        value (scaling would corrupt the >0.5 threshold)."""
        raw = np.asarray(action, dtype=np.float32)
        out = raw.copy()
        D = out.shape[1] if out.ndim == 2 else 0
        gi = (self.gripper_dim if self.gripper_dim >= 0 else D + self.gripper_dim) if (self.gripper_is_raw and D) else None

        if not self.already_metric and self.abs_scale is not None:
            scale = self.abs_scale.reshape(1, -1).astype(np.float32)
            if scale.shape[1] == D:
                scale = scale.copy()
                if gi is not None and 0 <= gi < D:
                    scale[0, gi] = 1.0                      # never scale the gripper dim
                out = out * scale
            else:
                logger.warning("abs_scale dim %d != action dim %d; skipping scale", scale.shape[1], D)

        if gi is not None and 0 <= gi < D:
            if self.z_gate is not None:
                # adaptive z-score gate: stateful close/open/hold from the RAW signal stream.
                # dz (commanded z delta, metric m/step) feeds the arrival condition.
                # z logged per step so the operator can calibrate k (DEPLOY_LOG entry #2 §2).
                states = np.empty(out.shape[0], dtype=np.float32)
                for i, r in enumerate(raw[:, gi]):
                    # Bug #2: feed the PRE-mps descent (÷ mps) so arrival_dz is mps-invariant.
                    dz = (float(out[i, 2]) / self.gripper_mps) if (D > 2 and self.gripper_mps) else None
                    z, closed = self.z_gate.update(float(r), dz=dz)
                    states[i] = 1.0 if closed else 0.0
                    state_s = ("CLOSE" if closed
                               else "pending-arrival" if self.z_gate.close_intent else "open")
                    logger.info("[z-gate] raw=%+.4f z=%+.2f dz=%+.4f -> %s",
                                r, z, dz if dz is not None else float("nan"), state_s)
                out[:, gi] = states
            else:
                out[:, gi] = (raw[:, gi] > 0.5).astype(np.float32)   # binarize from RAW: >0.5 -> close
        return out

    def play(
        self,
        action_chunk: np.ndarray,
        backend: RobotBackend,
        *,
        horizon: Optional[int] = None,
        pace: bool = True,
        frame_callback: Optional[Any] = None,
    ) -> np.ndarray:
        """Apply the first ``horizon`` actions at ``control_dt``. Returns the physical chunk played.

        ``frame_callback``: optional zero-argument callable invoked after each action step (before
        the sleep). Used by the controller to capture mid-chunk frames into the rolling context
        window at the control rate, so the obs_builder window fills toward context_frames during
        playback rather than incrementing only once per chunk.
        """
        phys = self.denormalize(action_chunk)
        H = phys.shape[0] if horizon is None else min(horizon, phys.shape[0])
        budget_bound = False
        if self.max_chunk_trans is not None and phys.shape[1] >= 3:
            path = float(np.linalg.norm(phys[:H, :3], axis=1).sum())
            if path > self.max_chunk_trans:
                scale = self.max_chunk_trans / path
                phys = phys.copy()
                phys[:H, :3] *= scale
                budget_bound = True
                logger.warning("[max-trans] chunk path %.3fm > budget %.3fm — translation scaled x%.2f",
                               path, self.max_chunk_trans, scale)
        if phys.shape[1] >= 3:
            # honest rails telemetry: replicate the backend's per-step clamp to compute the net
            # translation actually commanded; flag bound if either rail clipped anything.
            c = self.step_trans_clamp
            clipped = np.clip(phys[:H, :3], -c, c)
            self.last_rails_bound = bool(budget_bound or not np.allclose(clipped, phys[:H, :3]))
            self.last_executed_trans = float(np.linalg.norm(clipped, axis=1).sum())
        t_chunk = time.monotonic()
        overruns = 0
        for i in range(H):
            t0 = time.monotonic()
            backend.apply_action(phys[i], self.action_space)
            if frame_callback is not None:
                frame_callback()
            dt = self.control_dt - (time.monotonic() - t0)
            if dt > 0:
                if pace:
                    time.sleep(dt)
            else:
                overruns += 1
        elapsed = time.monotonic() - t_chunk
        if overruns:
            # apply_action+frame_callback exceeded control_dt — playback ran below the control
            # rate, so executed motion is slower than the plan and context frames are spaced
            # wider than the training 15 Hz. Visibility only; pacing self-corrects next step.
            logger.warning("[play] %d/%d steps overran control_dt (chunk took %.2fs, nominal %.2fs)",
                           overruns, H, elapsed, H * self.control_dt)
        return phys[:H]
