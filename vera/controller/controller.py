"""The control loop: drive a robot backend from the policy server.

Two modes (CONTROLLER_SPEC §2), both reusing one long-lived ws connection:
  * ``sync_hold`` — build obs -> infer (blocks) -> play H -> repeat. Robot stop-and-goes. Default
    for bring-up; deterministic and safe.
  * ``async_pipeline`` — a background thread keeps inferring the next chunk on the obs snapshot from
    the start of the current chunk (1-chunk stale); the exec loop plays the latest ready chunk. On
    underrun (next chunk not ready at chunk end) the robot HOLDS its last pose (no setpoint sent —
    polymetis holds), per the chosen underrun policy.

Session is a uuid per episode; a changed session_id also auto-resets the server adapter (defense in
depth). On stop, a final reset() lets the server flush artifacts (the DreamZero kill-loses-data gap).
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

import numpy as np

from vera.controller.action_player import ActionPlayer
from vera.controller.obs_builder import ObsBuilder
from vera.controller.robot_iface import RobotBackend
from vera.server.protocol.server_config import VeraServerConfig

logger = logging.getLogger("vera.controller")

_COLLAPSE_ABSMEAN = 1e-3


def _local_git_head() -> str:
    """This client's git HEAD (the vera repo containing this file). Empty string on failure."""
    import subprocess
    from pathlib import Path
    try:
        return subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except Exception:
        return ""


class VeraController:
    def __init__(
        self,
        client: Any,                       # WebsocketClientPolicy (or any infer/reset object)
        backend: RobotBackend,
        *,
        action_horizon: int = 10,
        context_frames: int | None = None,   # None -> use the server-advertised context length
        image_hw: tuple[int, int] = (128, 192),
        settle_steps: int = 5,
    ) -> None:
        self.client = client
        self.backend = backend
        self._action_horizon = action_horizon
        self._context_frames = context_frames
        self._image_hw = image_hw
        # Extra fields merged into every infer request (e.g. lang_guidance/hist_guidance —
        # the server applies them on change, same contract as the per-request prompt).
        self.infer_extras: Dict[str, Any] = {}
        # Phase-dependent motion_plan_scale (operator entry-#6 reply): the jacobian's linear
        # regime saturates on large dream flows (~2.2x under-delivery at 6-10px vs 0.3-3px), so
        # the approach phase needs a higher gain than fine manipulation. (approach, fine) scales
        # keyed to the z-gate state: gripper open = approach, closed = fine; reopen (retry)
        # switches back. Sent live via configure between chunks. None = static mps.
        self._phase_mps: Optional[tuple] = None
        self._current_mps: Optional[float] = None
        # Post-chunk settle: 15 Hz position-target streaming accumulates tracking lag (the arm
        # can't complete a ~3 cm step in 67 ms), and the residual releases all at once when the
        # command stream stops at the chunk boundary — a large visual jump between the last
        # playback frame and the next infer's obs. Sampling settle_steps extra frames at the
        # control rate keeps that catch-up motion uniformly sampled in the context.
        self._settle_steps = int(settle_steps)
        # Cold-start warmup (operator entry-#15 "settle-capture", arm B): number of REAL camera
        # frames to accumulate in the obs window before infer #1. 1 = true single-frame cold
        # start (validated path). >1 lets the camera capture held frames at the control rate
        # while the arm is stationary — real sensor noise breaks the static prior, so unlike
        # COPIED frames (which froze the dream twice on the robot) this does not freeze.
        self._warmup_frames = 1
        self._apply_config(client.get_server_metadata())
        self._check_provenance()

    def _apply_config(self, metadata: Dict[str, Any]) -> None:
        """(Re)build everything that depends on the server config — called on init and after a
        hot-swap (switch). obs_builder/player/H all follow the (possibly new) embodiment + ckpt."""
        self.config = VeraServerConfig(**metadata)
        self.H = int(self._action_horizon or self.config.action_horizon)
        # default to the WAN's real context length advertised by the server (1+(N-1)*stride)
        ctx_frames = int(self._context_frames or self.config.context_frames)
        self.obs_builder = ObsBuilder(
            self.config.view_keys, image_hw=self._image_hw,
            context_frames=ctx_frames, proprio_keys=self.config.proprio_keys,
        )
        self.player = ActionPlayer(self.config)
        logger.info(
            "controller ready: embodiment=%s H=%d dt=%.4f already_metric=%s views=%s | %s + %s",
            self.config.embodiment, self.H, self.config.control_dt,
            self.config.actions_already_metric, self.config.view_keys,
            self.config.planner_model, self.config.idm_model,
        )

    def switch(self, **params: Any) -> None:
        """Hot-swap the server's model and re-init this controller around the new config. ``params``:
        ``embodiment`` (e.g. droid/allegro), ``algo_config`` (WAN ckpt yaml), ``dynamics_run_id`` (IDM),
        ``sample_steps``, ``action_horizon``. NOTE: switching embodiment changes view_keys — the
        ``backend`` must be able to supply the new views (swap the backend too for a real embodiment
        change; a same-embodiment ckpt swap keeps the current backend valid)."""
        logger.info("[switch] requesting reload: %s", params)
        new_meta = self.client.reload(params)
        self._apply_config(new_meta)
        self._check_provenance()
        logger.info("[switch] now serving %s + %s (embodiment=%s)",
                    self.config.planner_model, self.config.idm_model, self.config.embodiment)

    def _check_provenance(self) -> None:
        """Compare the server's git commit (advertised in the handshake) to this client's local
        commit, so a two-machine code-drift shows up the instant the client connects — not after a
        confusing debug session. The server stamps git_head/git_dirty in VeraServerConfig.from_runtime.
        """
        server_head = (self.config.git_head or "")[:12]
        server_dirty = self.config.git_dirty
        local_head = _local_git_head()[:12]
        logger.info(
            "[provenance] server %s@%s%s | client @%s",
            self.config.hostname or "?", server_head or "unknown",
            " (dirty)" if server_dirty else "", local_head or "unknown",
        )
        if server_head and local_head and server_head != local_head:
            logger.warning(
                "[provenance] CODE DRIFT: server is on %s but this client is on %s. The two "
                "machines are NOT on the same commit — pull/push to sync before trusting a rollout.",
                server_head, local_head,
            )
        elif server_dirty:
            logger.warning(
                "[provenance] server has UNCOMMITTED changes (git_dirty) — its exact code is not in "
                "any commit; results may not be reproducible. Commit on the server before a real run.",
            )

    # -- helpers ------------------------------------------------------------- #
    def _new_episode(self, prefill: bool = True) -> str:
        sid = str(uuid.uuid4())
        self.obs_builder.reset()
        self.player.reset_episode_state()
        self.backend.reset()
        self.client.reset({"session_id": sid, "reason": "new_episode"})
        if prefill:
            self._prefill_context()
        return sid

    def _prefill_context(self) -> None:
        """Capture context_frames at the control rate while the robot holds.

        DISABLED BY DEFAULT (prefill=False): 21 near-identical frames of a holding robot are
        exactly the "repeated frames collapse the planner" failure (Bug B) — every prefilled
        episode on 2026-06-11 froze on chunk 1 (|mean| ~1e-04). The correct cold start is ONE
        frame (the policy pads at call time, option-b); the window then grows naturally via
        the per-step playback capture and is full by chunk ~3.
        """
        n = self.obs_builder.context_frames
        dt = float(self.config.control_dt)
        logger.info("[prefill] capturing %d frames at %.1f Hz (%.1fs) ...", n, 1.0 / dt, n * dt)
        for _ in range(n):
            t0 = time.monotonic()
            state = self.backend.get_state()
            self.obs_builder.append_frame(state)
            remaining = dt - (time.monotonic() - t0)
            if remaining > 0:
                time.sleep(remaining)
        logger.info("[prefill] done ctx_window=%d/%d", len(self.obs_builder._window),
                    self.obs_builder.context_frames)

    def set_phase_mps(self, approach: float, fine: float) -> None:
        self._phase_mps = (float(approach), float(fine))

    def _maybe_switch_phase_mps(self) -> None:
        """Live-switch the server's motion_plan_scale on gripper phase transitions."""
        if self._phase_mps is None:
            return
        gate = getattr(self.player, "z_gate", None)
        closed = bool(gate.closed) if gate is not None else False
        target = self._phase_mps[1] if closed else self._phase_mps[0]
        if target == self._current_mps:
            return
        try:
            self.client.configure({"motion_plan_scale": float(target)})
            self._current_mps = target
            logger.info("[phase-mps] %s phase -> motion_plan_scale=%.2f",
                        "fine (gripper closed)" if closed else "approach (gripper open)", target)
        except Exception as e:
            logger.warning("[phase-mps] configure failed (continuing): %s", e)

    def _post_chunk_observe(self, sid: str, played_frames: Optional[list] = None) -> None:
        """Push the just-executed frames to the server's viewer (observe endpoint) so the
        retro-render appears immediately instead of with the next infer.

        New contract (cf5ef05): send ``executed_rgb`` = exactly the frames played this chunk,
        in playback order — the server prepends the plan-time obs, so dream[i] pairs with
        executed[i-1] by construction (immune to cold start / settle / top-up frames). Falls
        back to the legacy context-tail form when no played frames are available.
        Best-effort: viewer-only, never aborts the control loop."""
        if not hasattr(self.client, "observe"):
            return
        msg: Dict[str, Any] = {
            "view_keys": list(self.obs_builder.view_keys),
            "view_widths": self.obs_builder.view_widths,
            "session_id": sid,
        }
        if played_frames:
            msg["executed_rgb"] = np.stack(played_frames, axis=0)
        else:
            window = self.obs_builder.current_window()
            if window is None:
                return
            msg["context_rgb"] = window
        try:
            self.client.observe(msg)
        except Exception as e:
            logger.warning("[observe] viewer update failed (continuing): %s", e)

    def _infer(self, state: Dict[str, Any], sid: str, prompt: Optional[str]) -> np.ndarray:
        obs = self.obs_builder.build(state, session_id=sid, prompt=prompt)
        if self.infer_extras:
            obs.update(self.infer_extras)
        # rails telemetry from the previous chunk (anti-windup feedback for the server's
        # adaptive gain, operator 21a8dc1): bound chunks must not update the gain.
        if getattr(self.player, "last_rails_bound", None) is not None:
            obs["rails_bound"] = bool(self.player.last_rails_bound)
            obs["executed_trans"] = float(self.player.last_executed_trans)
        t0 = time.monotonic()
        out = self.client.infer(obs)
        infer_s = time.monotonic() - t0
        action = np.asarray(out["action"], dtype=np.float32)
        absmean = float(np.abs(action).mean()) if action.size else 0.0
        if absmean < _COLLAPSE_ABSMEAN:
            logger.warning("[collapse-canary] action |mean|=%.2e (frozen?)", absmean)
        logger.info("infer %.2fs action=%s |mean|=%.4f", infer_s, tuple(action.shape), absmean)
        return action

    # -- mode (a): sync + hold ---------------------------------------------- #
    def run_sync(self, *, prompt: Optional[str] = None, max_steps: int = 50,
                 prefill: bool = False) -> Dict[str, Any]:
        sid = self._new_episode(prefill=prefill)
        logger.info("[sync_hold] episode %s start (prefill=%s)", sid, prefill)
        self._current_mps = None
        self._maybe_switch_phase_mps()   # start in approach-phase gain
        n = 0

        # Local dump of the REAL executed frames per chunk. The server's debug dump
        # has NO real video — its 'rgb' field is the dream's flow-source frames
        # (motion_policy.py:2576) and 'context_rgb' is a single plan-time obs. The
        # only true camera record of an episode lives right here in played_frames,
        # so persist it for the archive renderer (real.mp4 / policy_vis current panel).
        import os as _os, datetime as _dt
        exec_dir = _os.path.join(
            "outputs", "client_runs",
            f"{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{sid[:8]}")
        _os.makedirs(exec_dir, exist_ok=True)
        logger.info("[sync_hold] executed-frame dump dir: %s", exec_dir)

        played_frames: list = []

        def _capture_frame() -> None:
            """Append one camera frame to the rolling context window mid-chunk and record it
            as an EXECUTED frame for the post-chunk observe (exact retro alignment)."""
            mid = self.backend.get_state()
            played_frames.append(self.obs_builder.append_frame(mid))

        def _settle_frame() -> None:
            """Settle-window: let the arm finish its residual tracking lag at the control
            rate WITHOUT appending to the planner context. Settle frames are near-static
            (the arm is decelerating onto the last setpoint, no new command is sent), so
            appending them put up to 5/9 static frames in the freshest TAIL of every
            context window — exactly the frames the AR planner weights most — which
            reliably collapsed the dream (static-tail freeze; Bug #1 audit 2026-06-12,
            confirmed on the 22:09 mps=1.25 run: 2 chunks frozen right after a settle).
            We still read state (the arm settles, and _last_euler refreshes for rot-lock),
            but the obs window now holds ONLY executed (moving) frames."""
            self.backend.get_state()

        # Cold-start warmup (arm B): before the first infer, capture warmup_frames-1 REAL held
        # frames at the control rate (the arm is stationary; the camera re-shoots each frame so
        # sensor noise breaks the static prior — NOT copied pixels, which freeze the dream). The
        # first infer's build() appends the last frame, so the window enters infer #1 with
        # warmup_frames real frames instead of a single cold-start frame.
        if self._warmup_frames > 1:
            dt = float(self.config.control_dt)
            for _ in range(self._warmup_frames - 1):
                t0 = time.monotonic()
                self.obs_builder.append_frame(self.backend.get_state())
                rem = dt - (time.monotonic() - t0)
                if rem > 0:
                    time.sleep(rem)
            logger.info("[sync_hold] cold-start warmup: %d real held frames before infer #1",
                        self._warmup_frames - 1)

        try:
            for step in range(max_steps):
                state = self.backend.get_state()
                action = self._infer(state, sid, prompt)
                played_frames.clear()
                self.player.play(action, self.backend, horizon=self.H,
                                 frame_callback=_capture_frame)
                # settle window: pace the control loop while the arm completes its residual
                # tracking lag, but do NOT feed these frames to the planner (see _settle_frame).
                # Bug #1 fix (2026-06-12): appending settle frames made the context tail static
                # and froze the dream; the window now holds only executed frames.
                dt = float(self.config.control_dt)
                for _ in range(self._settle_steps):
                    t0 = time.monotonic()
                    _settle_frame()
                    rem = dt - (time.monotonic() - t0)
                    if rem > 0:
                        time.sleep(rem)
                if played_frames:
                    np.savez_compressed(
                        _os.path.join(exec_dir, f"executed_chunk_{n:03d}.npz"),
                        executed_rgb=np.stack(played_frames, axis=0))  # (T,H,ΣW,3) uint8
                n += 1
                self._maybe_switch_phase_mps()
                self._post_chunk_observe(sid, played_frames)
                if state.get("_done"):
                    logger.info("[sync_hold] backend signaled done at step %d", step)
                    break
        finally:
            try:
                self.client.reset({"session_id": sid, "reason": "episode_end"})
            except Exception as e:
                # e.g. SIGINT mid-infer leaves the ws recv busy (ConcurrencyError). Harmless:
                # the server's session auto-reset covers the next episode.
                logger.warning("[sync_hold] episode_end reset failed (%s) — server auto-reset covers it", e)
        logger.info("[sync_hold] episode %s done: %d chunks", sid, n)
        return {"session_id": sid, "chunks": n}

    # -- mode (b): async pipeline (hold-last-pose on underrun) --------------- #
    def run_async(self, *, prompt: Optional[str] = None, max_steps: int = 50,
                  prefill: bool = False) -> Dict[str, Any]:
        sid = self._new_episode(prefill=prefill)
        logger.info("[async_pipeline] episode %s start", sid)
        slot: Dict[str, Any] = {"chunk": None, "seq": 0}
        lock = threading.Lock()
        stop = threading.Event()
        underruns = {"n": 0}

        # prime: first chunk synchronously (cold start) so we have something to play.
        first = self._infer(self.backend.get_state(), sid, prompt)
        with lock:
            slot["chunk"], slot["seq"] = first, 1

        def infer_worker():
            produced = 1
            while not stop.is_set() and produced < max_steps:
                state = self.backend.get_state()       # snapshot at (roughly) current chunk start
                try:
                    action = self._infer(state, sid, prompt)
                except Exception:
                    logger.exception("[async] infer failed; stopping worker")
                    break
                with lock:
                    slot["chunk"], slot["seq"] = action, slot["seq"] + 1
                produced += 1
            stop.set()

        worker = threading.Thread(target=infer_worker, daemon=True)
        worker.start()

        played_seq = 0
        chunks_played = 0
        try:
            while not (stop.is_set() and slot["seq"] == played_seq):
                with lock:
                    chunk, seq = slot["chunk"], slot["seq"]
                if chunk is not None and seq != played_seq:
                    # NOTE: frame capture during async playback would race with infer_worker's
                    # get_state(); use run_sync until DROID thread-safety is confirmed.
                    self.player.play(chunk, self.backend, horizon=self.H)
                    played_seq = seq
                    chunks_played += 1
                    if self.backend.get_state().get("_done"):
                        break
                else:
                    underruns["n"] += 1                # hold last pose: send nothing, wait a tick
                    time.sleep(self.config.control_dt)
        finally:
            stop.set()
            worker.join(timeout=2.0)
            self.client.reset({"session_id": sid, "reason": "episode_end"})
        logger.info("[async_pipeline] episode %s done: %d chunks, %d underrun ticks",
                    sid, chunks_played, underruns["n"])
        return {"session_id": sid, "chunks": chunks_played, "underrun_ticks": underruns["n"]}
