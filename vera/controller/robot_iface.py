"""Robot backends the controller drives.

A ``RobotBackend`` abstracts the robot so the control loop is identical for the real FR3 and a
hardware-free replay. Contract:
  * ``get_state()`` -> dict with per-view camera frames (uint8 HxWx3, keyed by view name) plus
    proprio arrays (``joint_position``/``cartesian_position``/``gripper_position``).
  * ``apply_action(action, action_space)`` -> command one denormalized action (metric units).
  * ``reset()`` -> start a fresh episode.

``ReplayBackend`` (hardware-free) plays recorded multiview videos — mirrors DreamZero's
``test_client_AR`` replay client — so the whole obs->infer->play path is validated against the
live server before touching hardware. ``RobotEnvBackend`` (real) wraps DROID ``RobotEnv``/polymetis.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger("vera.controller.robot")


@runtime_checkable
class RobotBackend(Protocol):
    view_keys: List[str]

    def get_state(self) -> Dict[str, Any]:
        """Per-view frames (uint8 HxWx3) + proprio arrays. Advances replay; reads hardware."""
        ...

    def apply_action(self, action: np.ndarray, action_space: str) -> None:
        """Command ONE denormalized action (metric units) at the current control tick."""
        ...

    def reset(self) -> None:
        ...


class ReplayBackend:
    """Hardware-free backend: serves frames from recorded multiview videos.

    ``apply_action`` advances the playhead by one tick (so the camera "moves forward" in the
    recording as actions are consumed) and records the commanded action for inspection. Proprio
    is zero (a recording has no live robot state) — fine for validating the obs/infer/play path.
    """

    def __init__(self, video_paths: List[str], view_keys: List[str], *,
                 frames_per_action: int = 1, proprio_dims: Dict[str, int] | None = None) -> None:
        import av  # decode lazily
        assert len(video_paths) == len(view_keys), "one video per view"
        self.view_keys = list(view_keys)
        self._frames_per_action = int(frames_per_action)
        self._proprio_dims = proprio_dims or {
            "joint_position": 7, "cartesian_position": 6, "gripper_position": 1,
        }
        self._views: List[np.ndarray] = []
        for p in video_paths:
            c = av.open(str(p))
            frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
            self._views.append(np.stack(frames))  # (N,H,W,3) uint8
        self._n = min(v.shape[0] for v in self._views)
        self._t = 0
        self.applied_actions: List[np.ndarray] = []
        logger.info("ReplayBackend: %d views, %d frames each", len(self._views), self._n)

    def get_state(self) -> Dict[str, Any]:
        t = min(self._t, self._n - 1)
        state: Dict[str, Any] = {k: self._views[i][t] for i, k in enumerate(self.view_keys)}
        for name, dim in self._proprio_dims.items():
            state[name] = np.zeros(dim, dtype=np.float32)
        state["_done"] = self._t >= self._n - 1
        return state

    def apply_action(self, action: np.ndarray, action_space: str) -> None:
        self.applied_actions.append(np.asarray(action).copy())
        self._t = min(self._t + self._frames_per_action, self._n - 1)

    def reset(self) -> None:
        self._t = 0
        self.applied_actions = []


class RobotEnvBackend:
    """Real DROID FR3 backend over polymetis. Wraps ``droid.robot_env.RobotEnv``.

    Thin by design: ``get_state`` reads the robot_state + cameras; ``apply_action`` forwards a
    denormalized action to ``update_robot(..., blocking=False)`` (polymetis 1 kHz interpolates
    between the 15 Hz setpoints). Imported lazily so the controller package stays importable on a
    host without the droid stack.

    ``control_dt``: the server's control timestep (seconds).  Required to convert ``se3_delta``
    actions (meters/step) to ``cartesian_velocity`` (m/s) that DROID's IK solver expects — DROID
    multiplies by control_dt internally, so we divide first to cancel that factor.
    """

    # XINGJIAN_HOME_LOWER — episode-start home (default since 2026-06-12): the old xingjian/
    # LESTER_HOME_HIGHER pose lowered exactly 0.05 m in z (same xy + orientation, IK-solved).
    # Installed as the env's reset_joints so RobotEnv.reset() (open gripper -> blocking
    # update_joints) lands here instead of the DROID default pose. Keep tools/go_home.py in sync.
    HOME_JOINTS = np.array([0.3036, -0.3578, 0.2733, -2.5927, 0.1284, 2.2284, 0.0469],
                           dtype=np.float64)

    def __init__(self, view_keys: List[str], *, control_dt: float = 1 / 15.0,
                 gripper_action_space: Optional[str] = None,
                 camera_reader: Optional[Any] = None) -> None:
        from droid.robot_env import RobotEnv  # lazy; only on the robot host
        self.view_keys = list(view_keys)
        # do_reset=False: the constructor would otherwise home to the DROID default pose before
        # we install HOME_JOINTS; _new_episode -> reset() does the single homing move instead.
        self._env = RobotEnv(do_reset=False)
        self._env.reset_joints = self.HOME_JOINTS.copy()
        self._control_dt = float(control_dt)
        self._gripper_action_space = gripper_action_space
        self._camera_reader = camera_reader  # supplies {view_key: uint8 HxWx3}
        # Online rotation-drift lock (port of droid_ws_runner.py:993, the old stack's
        # "rot-lock K=3"): polymetis' cartesian_velocity impedance controller slowly drifts the
        # EEF rotation; dims zeroed in the jacobian solve (jzd 3,4) get NO policy command, so the
        # drift is unrecoverable open-loop — FK on droidspec_ab2 measured +20° pitch over 19
        # chunks. Per-step P-control pulls each locked dim back to the episode-start snapshot:
        # cmd[d] = -gain * drift * control_dt  (gain in rad/s per rad, matching the old CLI).
        self.rot_lock_dims: List[int] = []
        self.rot_lock_gain: float = 3.0
        self._locked_euler: Optional[np.ndarray] = None
        self._last_euler: Optional[np.ndarray] = None
        self._rot_lock_step = 0

    # ZED cameras deliver their first frames asynchronously after open — the first few
    # get_observation() calls can be missing camera keys. Retry briefly instead of crashing.
    _CAMERA_WARMUP_TRIES = 50
    _CAMERA_WARMUP_SLEEP_S = 0.1

    def _read_obs_with_frames(self) -> tuple:
        import time as _time
        last_err: Optional[Exception] = None
        for attempt in range(self._CAMERA_WARMUP_TRIES):
            obs = self._env.get_observation()
            if not self._camera_reader:
                return obs, {}
            try:
                return obs, self._camera_reader(obs)
            except KeyError as e:
                last_err = e
                if attempt == 0:
                    logger.info("camera frames not ready (%s); warming up ...", e)
                _time.sleep(self._CAMERA_WARMUP_SLEEP_S)
        raise RuntimeError(
            f"camera frames still missing after "
            f"{self._CAMERA_WARMUP_TRIES * self._CAMERA_WARMUP_SLEEP_S:.0f}s warmup"
        ) from last_err

    def get_state(self) -> Dict[str, Any]:
        obs, frames = self._read_obs_with_frames()
        rs = obs.get("robot_state", {})
        state: Dict[str, Any] = {
            "joint_position": np.asarray(rs.get("joint_positions", rs.get("joint_position")), np.float32),
            "cartesian_position": np.asarray(rs.get("cartesian_position"), np.float32),
            "gripper_position": np.asarray([rs.get("gripper_position", 0.0)], np.float32),
        }
        for k in self.view_keys:
            if k in frames:
                state[k] = frames[k]
        # cache euler for the rot-lock loop (apply_action runs between get_state calls; one
        # control step of staleness is fine for a slow-drift P correction)
        cart = state["cartesian_position"]
        if cart is not None and cart.shape[0] >= 6:
            self._last_euler = np.asarray(cart[3:6], dtype=np.float64)
        state["_done"] = False
        return state

    # Map vera action_space names -> DROID action_space names.
    _ACTION_SPACE_MAP = {
        "se3_delta": "cartesian_velocity",
        "joint_delta": "joint_velocity",
    }

    # Hard safety limits per control step (applied BEFORE the ÷control_dt conversion).
    # Prevents runaway server output (e.g. stale context → |mean|=477) from reaching the arm.
    # 3 cm/step: with motion_plan_scale=1.0 server-side the policy peaks ≈0.038 m/step, so this
    # barely clips in normal operation but still catches runaway output (DEPLOY_LOG entry #2/#3).
    _TRANS_CLAMP_M = 0.03       # ≤ 3 cm per step → 0.45 m/s max
    _ROT_CLAMP_RAD = 0.0524     # ≤ 3° per step
    # Optional tighter limit on DOWNWARD z only (violent descent dives, 2026-06-11 rerun);
    # lateral and upward motion keep the general clamp. None = off; set via max_descent_step.
    max_descent_step: Optional[float] = None

    def apply_action(self, action: np.ndarray, action_space: str) -> None:
        droid_action_space = self._ACTION_SPACE_MAP.get(action_space, action_space)
        cmd = np.asarray(action, dtype=np.float32)
        # CRITICAL safety guard (audit 2026-06-12): a non-finite action (planner collapse,
        # IDM divide-by-zero, msgpack/network glitch) sails THROUGH np.clip (clip does not
        # filter NaN/Inf) and through DROID's norm-check (NaN>1 is False), reaching the
        # impedance controller as an undefined command — a fault or a wild jerk. The
        # collapse-canary only logs and would not even fire (NaN<1e-3 is False). Reject and
        # hold the last setpoint (polymetis' documented underrun-safe behavior).
        if not np.all(np.isfinite(cmd)):
            logger.error("[safety] non-finite action rejected, holding pose: %s", cmd)
            return
        if action_space == "se3_delta":
            cmd = cmd.copy()
            if self.rot_lock_dims and self._last_euler is not None:
                if self._locked_euler is None:
                    self._locked_euler = self._last_euler.copy()
                    logger.info("[rot-lock] reference euler locked: %s (dims=%s, gain=%.1f)",
                                np.round(self._locked_euler, 4).tolist(),
                                self.rot_lock_dims, self.rot_lock_gain)
                for d in self.rot_lock_dims:
                    if 3 <= d <= 5:
                        # shortest-angle drift in [-pi, pi] vs the episode-start snapshot
                        drift = ((self._last_euler[d - 3] - self._locked_euler[d - 3] + np.pi)
                                 % (2 * np.pi)) - np.pi
                        cmd[d] = -self.rot_lock_gain * drift * self._control_dt
                self._rot_lock_step += 1
                if self._rot_lock_step % 32 == 1:
                    drifts = ((self._last_euler - self._locked_euler + np.pi) % (2 * np.pi)) - np.pi
                    logger.info("[rot-lock] drift deg rx=%+.2f ry=%+.2f rz=%+.2f",
                                *np.degrees(drifts))
            # Safety clamp before unit conversion so limits are in physical units.
            cmd[:3] = np.clip(cmd[:3], -self._TRANS_CLAMP_M, self._TRANS_CLAMP_M)
            if self.max_descent_step is not None and cmd[2] < -self.max_descent_step:
                cmd[2] = -self.max_descent_step
            cmd[3:6] = np.clip(cmd[3:6], -self._ROT_CLAMP_RAD, self._ROT_CLAMP_RAD)
            # se3_delta is in meters/step; DROID cartesian_velocity expects m/s and multiplies by
            # control_dt internally.  Divide the 6-DOF arm dims so the net motion equals the delta.
            # Gripper (dim −1) is already binarized by ActionPlayer and must not scale.
            cmd[:6] /= self._control_dt
        self._env.update_robot(
            cmd, action_space=droid_action_space,
            gripper_action_space=self._gripper_action_space, blocking=False,
        )

    def reset(self) -> None:
        self._env.reset()
        # re-snapshot the rot-lock reference after homing (next apply_action re-locks)
        self._locked_euler = None
        self._rot_lock_step = 0
