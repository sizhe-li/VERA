"""VeraPolicyAdapter — the seam between the wire protocol and a real MotionPolicy.

The transport (``WebsocketPolicyServer``) speaks the ``BasePolicy`` ABC: ``infer(obs)->dict``
and ``reset(reset_info)``. The real two-stage policy (``MotionPolicy``: WAN planner -> jacobian
IDM) speaks ``predict_action_chunk(PolicyObservation, context_rgb, ...)->PolicyOutput``. This
adapter is the thin, algorithm-free translation layer between them. It owns NO model logic —
only obs/format translation + the deploy-time state hygiene that the old DreamZero rollouts
got wrong (see SERVER_BUG_ANALYSIS.md):

  * obs translation     wire dict  -> (context_rgb [T,H,W,3] float[0,1], PolicyObservation)
  * action translation  PolicyOutput.action (B,H,D) -> {"action": (H,D)} the controller plays
  * cold-start (opt-b)  first infer after reset does NOT pre-repeat the single seed frame into
                        a full context (repeated frames -> frozen/static attractor); it passes
                        the short context through and lets the policy pad at call time.
  * AR-cache reset      ``reset()`` is the single source of truth — it clears the context queue,
                        controller, dream index AND any AR KV-cache inside the planner. The
                        adapter never reaches into model internals; it just calls reset().
  * session auto-reset  a changed ``session_id`` auto-triggers reset (Bug C: a client that forgot
                        to reset would otherwise plan on the previous episode's stale history).
  * observability       per-infer latency + action |mean| (collapse canary: a healthy chunk has
                        absmean well above ~1e-3; the frozen-collapse bug drove it to ~3e-4).

Obs naming follows the OLD vera contract (width-concat ``rgb``/``context_rgb`` + ``view_keys``/
``view_widths`` + proprio ``q_robot``/``eef_pos``/...), NOT roboarena — but the buggy behaviors
above are fixed here rather than inherited. See SERVER_PROTOCOL_SPEC.md §4/§5/§6.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import numpy as np

from vera.policy.base_policy import PolicyObservation, PolicyOutput
from vera.server.protocol.base_policy import BasePolicy
from vera.server.protocol.server_config import VeraServerConfig

logger = logging.getLogger("vera.adapter")

# action |mean| below this for a full chunk is the frozen-collapse signature (Bug B).
_COLLAPSE_ABSMEAN = 1e-3


# --------------------------------------------------------------------------- #
# obs translation (vendored clean — no dependency on the old policy_server)    #
# --------------------------------------------------------------------------- #
def _numpy_or_none(value: Any) -> Optional[np.ndarray]:
    return None if value is None else np.asarray(value)


def _context_rgb_from_msg(msg: Dict[str, Any]) -> np.ndarray:
    """Extract client-owned context as float32 [0,1], shape [T,H,W,3] or [B,T,H,W,3]."""
    if "context_rgb" in msg:
        ctx = np.asarray(msg["context_rgb"])
    elif "rgb_context" in msg:  # legacy alias
        ctx = np.asarray(msg["rgb_context"])
    else:
        raise KeyError("infer requests require 'context_rgb' ([T,H,W,3] or [B,T,H,W,3])")
    if ctx.ndim not in (4, 5):
        raise ValueError(f"context_rgb must be [T,H,W,3] or [B,T,H,W,3], got {tuple(ctx.shape)}")
    if ctx.shape[-1] != 3:
        raise ValueError(f"context_rgb must have RGB last, got {tuple(ctx.shape)}")
    if ctx.dtype == np.uint8:
        ctx = ctx.astype(np.float32) / 255.0
    elif ctx.dtype != np.float32:
        ctx = ctx.astype(np.float32)
    return ctx


def _build_policy_observation(msg: Dict[str, Any]) -> PolicyObservation:
    """Wire dict -> PolicyObservation. RGB normalized to float[0,1]; proprio passed through."""
    rgb = np.asarray(msg["rgb"])
    if rgb.ndim == 3:
        rgb = rgb[None, ...]  # add batch dim
    if rgb.dtype == np.uint8:
        rgb = rgb.astype(np.float32) / 255.0
    return PolicyObservation(
        rgb=rgb,
        q_robot=_numpy_or_none(msg.get("q_robot")),
        rgb_vis=None,  # vis off on the hot path; opt-in elsewhere
        view_keys=msg.get("view_keys"),
        view_widths=msg.get("view_widths"),
        concat_rgb_key=msg.get("concat_rgb_key"),
        step_index=msg.get("step_index"),
        eef_pos=_numpy_or_none(msg.get("eef_pos")),
        eef_quat=_numpy_or_none(msg.get("eef_quat")),
        gripper_qpos=_numpy_or_none(msg.get("gripper_qpos")),
        dt=float(msg.get("dt", 0.1)),
        action_mode=msg.get("action_mode", "velocity"),
    )


def _build_chunk_observation(msg: Dict[str, Any], context_rgb: np.ndarray) -> PolicyObservation:
    """Build a PolicyObservation for a chunk request; default ``rgb`` to the last context frame."""
    m = dict(msg)
    if "rgb" not in m:
        m["rgb"] = context_rgb[-1] if context_rgb.ndim == 4 else context_rgb[:, -1]
    return _build_policy_observation(m)


# --------------------------------------------------------------------------- #
# the adapter                                                                  #
# --------------------------------------------------------------------------- #
class VeraPolicyAdapter(BasePolicy):
    """Wrap a real ``MotionPolicy`` (DROID / Allegro) behind the wire ``BasePolicy``."""

    def __init__(
        self,
        policy: Any,
        config: VeraServerConfig,
        *,
        default_execute_horizon: Optional[int] = None,
    ) -> None:
        if not hasattr(policy, "predict_action_chunk"):
            raise TypeError(
                f"{type(policy).__name__} has no predict_action_chunk; the chunked open-loop "
                "protocol requires a joint MotionPolicy (DROID native / Allegro IDM)."
            )
        self._policy = policy
        self.config = config
        # default chunk length: config.action_horizon (H) unless the call overrides it.
        self._default_H = int(default_execute_horizon or config.action_horizon)
        # episode state owned by the adapter (NOT the model — the model owns its own queues/cache)
        self._session_id: Optional[Any] = None
        self._first_infer_since_reset: bool = True
        self._infer_count: int = 0
        self._last_prompt: Optional[str] = None   # last per-request prompt applied to the WAN
        # Holds the PREVIOUS chunk's vis frames so the next infer can re-render them against the
        # now-executed observations (retro-render) — that is what makes the "current/executed" panel
        # evolve frame-by-frame alongside the dream instead of showing one frozen plan-time obs.
        self._pending_chunk_vis: Optional[dict] = None
        # ── grouped adaptive gains (achieved-vs-desired, port of the validated
        # mimicgen rule). OFF by default; enable via configure adaptive_gains=true.
        # Needs the client to send eef_pos (+optional eef_quat, gripper_width)
        # with each infer. Gains clamp to [0.5, 3.0], EMA alpha 0.3.
        self._adaptive_enabled: bool = False
        self._adaptive_gains = {"translation": 1.0, "rotation": 1.0}
        self._adaptive_prev: Optional[dict] = None   # {pose..., commanded per-group}
        # v2 feedforward flow-saturation compensation (stateless, no windup):
        # expected |trans| = k_lin * mean dream-flow px (k_lin from the measured
        # small-flow linear regime); boost the solved translation up to x2.5 when
        # it under-delivers. Toggle: flow_comp=true; k: flow_comp_klin.
        self._flow_comp_enabled: bool = False
        self._flow_comp_klin: float = 0.005
        # Optional live viewer. When set (a vis_server.VisHub), each infer renders the policy_vis
        # and pushes it + flow/track/action stats to the MJPEG dashboard. Set externally so the
        # reload path can re-attach it after a model swap.
        self.vis_hub: Any = None

    # -- BasePolicy.infer ---------------------------------------------------- #
    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        # (1) session auto-reset — a new episode/session clears stale history (Bug C guard).
        if self.config.needs_session_id:
            sid = obs.get("session_id")
            if sid is not None and sid != self._session_id:
                self.reset({"session_id": sid, "reason": "session_change"})
            elif sid is None and not getattr(self, "_warned_no_session_id", False):
                # needs_session_id is advertised but the client isn't sending one. The
                # explicit reset RPC is still the primary per-episode contract; warn once
                # so that a client which ALSO never resets doesn't silently leak gripper /
                # dream / adaptive state across episodes (N1).
                logger.warning(
                    "[session] needs_session_id=True but infer has no 'session_id'; "
                    "relying on the explicit reset RPC for per-episode state. If the "
                    "client neither sends session_id NOR calls reset between episodes, "
                    "gripper/dream/adaptive state leaks across episodes."
                )
                self._warned_no_session_id = True

        # (2) translate wire -> context + observation
        context_rgb = _context_rgb_from_msg(obs)
        pobs = _build_chunk_observation(obs, context_rgb)
        if self.vis_hub is not None:
            pobs.rgb_vis = pobs.rgb.copy()      # makes the policy render policy_vis into out.info

        # (2b) per-request prompt override. The launch-time --text is only a DEFAULT; a client may
        #      send "prompt" with any infer to re-condition the WAN dream WITHOUT a server restart
        #      (the WAN encodes text live each forward when load_prompt_embed=false). Only re-apply
        #      on change to avoid the ~50ms text-encode every call.
        prompt = obs.get("prompt")
        if prompt is not None and prompt != self._last_prompt:
            try:
                self._policy.configure_runtime(text_conditioning=prompt)
                self._last_prompt = prompt
                logger.info("[prompt] re-conditioned WAN: %r", str(prompt)[:80])
            except Exception:
                logger.exception("[prompt] configure_runtime failed (continuing)")

        # (2c) per-request guidance overrides — same pattern as the prompt: send
        #      "lang_guidance" / "hist_guidance" floats with any infer to retune the
        #      WAN's CFG scales live (prompt adherence vs context adherence). Only
        #      re-applied on change.
        if obs.get("flow_comp") is not None:
            self._flow_comp_enabled = bool(obs.get("flow_comp"))
            if obs.get("flow_comp_klin") is not None:
                self._flow_comp_klin = float(obs.get("flow_comp_klin"))
            logger.info("[flow-comp] enabled=%s k_lin=%.4f", self._flow_comp_enabled, self._flow_comp_klin)
        if obs.get("adaptive_gains") is not None:
            self._adaptive_enabled = bool(obs.get("adaptive_gains"))
            logger.info("[adaptive] enabled=%s gains=%s", self._adaptive_enabled, self._adaptive_gains)

        guidance = {k: obs.get(k) for k in ("lang_guidance", "hist_guidance") if obs.get(k) is not None}
        if guidance and guidance != getattr(self, "_last_guidance", None):
            try:
                self._policy.configure_runtime(**{k: float(v) for k, v in guidance.items()})
                self._last_guidance = guidance
                logger.info("[guidance] applied: %s", guidance)
            except Exception:
                logger.exception("[guidance] configure_runtime failed (continuing)")

        # (3) cold-start (option-b): on the first infer of an episode we pass the short context
        #     through as-is. We do NOT pre-repeat a single seed frame to fill the context window
        #     (repeated frames collapse the planner into a static/frozen prediction); the policy
        #     pads at call time. Reset clears _first_infer_since_reset.
        cold_start = self._first_infer_since_reset
        ctx_len = context_rgb.shape[0] if context_rgb.ndim == 4 else context_rgb.shape[1]

        # (4) run the two-stage policy (WAN planner -> jacobian IDM), client-owned context.
        H = int(obs.get("execute_horizon") or self._default_H)
        mode = str(obs.get("context_update_mode", "replace")).lower()
        t0 = time.monotonic()
        out: PolicyOutput = self._policy.predict_action_chunk(
            pobs, context_rgb=context_rgb, execute_horizon=H, context_update_mode=mode,
        )
        infer_s = time.monotonic() - t0

        # (5) action (B,H,D) -> (H,D); the controller plays all H open-loop.
        action = np.asarray(out.action, dtype=np.float32)
        if action.ndim == 3:
            action = action[0]
        chunk_len = int(action.shape[0]) if action.ndim >= 1 else 0

        # ── v2 feedforward: compensate flow-magnitude saturation (stateless) ──
        if self._flow_comp_enabled and chunk_len > 0:
            try:
                df = (out.info or {}).get("desired_flow")
                if df is not None:
                    flow_px = float(np.mean(np.linalg.norm(
                        np.asarray(df, dtype=np.float32).reshape(-1, 2), axis=-1)))
                    solved_t = float(np.linalg.norm(action[:, :3].sum(axis=0)))
                    expected_t = self._flow_comp_klin * flow_px
                    if solved_t > 1e-4 and expected_t > solved_t:
                        boost = float(np.clip(expected_t / solved_t, 1.0, 2.5))
                        action = action.copy(); action[:, :3] *= boost
                        logger.info("[flow-comp] flow=%.2fpx solved=%.4f expected=%.4f boost=%.2f",
                                    flow_px, solved_t, expected_t, boost)
            except Exception:
                logger.exception("[flow-comp] failed (continuing)")

        # ── adaptive gain update + apply (uses this request's proprio vs the
        # previous chunk's commanded deltas) ──
        if self._adaptive_enabled and chunk_len > 0:
            try:
                eef_pos = obs.get("eef_pos")
                prev = self._adaptive_prev
                # ANTI-WINDUP (DEPLOY_LOG #7): when the client's rails (chunk budget /
                # per-step clamp) clipped the last chunk, achieved-vs-commanded is
                # measuring the RAILS, not the plant — updating would integrate the
                # gain into the cap forever. Client sends rails_bound per infer.
                rails_bound = bool(obs.get("rails_bound") or False)
                if eef_pos is not None and prev is not None and prev.get("eef_pos") is not None and not rails_bound:
                    realized_t = float(np.linalg.norm(
                        np.asarray(eef_pos, dtype=np.float32).reshape(-1)[:3]
                        - np.asarray(prev["eef_pos"], dtype=np.float32).reshape(-1)[:3]))
                    # option (b): client may report the post-clip command it ACTUALLY
                    # played — the honest denominator when rails partially bind.
                    cmd_t = float(obs.get("executed_trans") or prev.get("cmd_translation") or 0.0)
                    if cmd_t > 5e-3:                       # stationary chunks teach nothing
                        ratio = realized_t / cmd_t
                        g = self._adaptive_gains["translation"]
                        # cap 2.2 = the offline-measured saturation ratio; alpha 0.15
                        target = float(np.clip(g / max(ratio, 1e-3), 0.5, 2.2))
                        self._adaptive_gains["translation"] = float(
                            np.clip(0.85 * g + 0.15 * target, 0.5, 2.2))
                        logger.info("[adaptive] ratio=%.3f gain_t=%.3f (cmd=%.4f realized=%.4f)",
                                    ratio, self._adaptive_gains["translation"], cmd_t, realized_t)
                elif rails_bound:
                    logger.info("[adaptive] rails_bound: skipping gain update (gain_t=%.3f)",
                                self._adaptive_gains["translation"])
                # apply current gains to the outgoing chunk (translation dims 0:3)
                action = action.copy()
                action[:, :3] *= self._adaptive_gains["translation"]
                # stash THIS chunk's command + start pose for the next update
                self._adaptive_prev = {
                    "eef_pos": None if eef_pos is None else np.asarray(eef_pos, dtype=np.float32).copy(),
                    # net commanded translation of the (gained) chunk we are sending
                    "cmd_translation": float(np.linalg.norm(action[:, :3].sum(axis=0))),
                }
            except Exception:
                logger.exception("[adaptive] update failed (continuing)")

        # (6) observability — latency + collapse canary.
        absmean = float(np.abs(action).mean()) if action.size else 0.0
        self._first_infer_since_reset = False
        self._infer_count += 1
        if absmean < _COLLAPSE_ABSMEAN and chunk_len > 0:
            logger.warning(
                "[collapse-canary] infer #%d action |mean|=%.2e < %.0e (chunk frozen?)",
                self._infer_count, absmean, _COLLAPSE_ABSMEAN,
            )
        logger.info(
            "[infer #%d] %.2fs H=%d chunk=%d ctx=%d |mean|=%.4f%s",
            self._infer_count, infer_s, H, chunk_len, ctx_len, absmean,
            " cold_start" if cold_start else "",
        )

        # info MUST be msgpack-safe: the policy's out.info carries torch Tensors (vis frames, flow)
        # which the wire codec cannot serialize and which the controller does not need (vis is saved
        # server-side via debug_dump). Keep only wire-safe scalars from it + our observability fields.
        info: Dict[str, Any] = {
            "infer_s": infer_s, "action_absmean": absmean, "chunk_len": chunk_len,
            "cold_start": cold_start, "context_len": int(ctx_len), "session_id": self._session_id,
        }
        for k, v in (out.info or {}).items():
            if isinstance(v, (int, float, bool, str)) or v is None:
                info.setdefault(k, v)

        # (7) live viewer: push this chunk's visualization to the MJPEG dashboard (best-effort).
        #     retro_vis re-renders the PREVIOUS chunk's dream against the now-executed frames in this
        #     call's context_rgb, so the "current/executed" panel evolves alongside the dream. Then we
        #     stash THIS chunk's frames for the next call to retro-render.
        if self.vis_hub is not None:
            retro_vis = self._make_retro_chunk_policy_vis(context_rgb)
            self._push_vis(context_rgb, action, infer_s, out.info or {}, retro_vis=retro_vis)
            self._store_pending_chunk_vis(pobs, out, exec_n=chunk_len)
        return {"action": action, "info": info}

    def observe(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Executed-frame report (no inference): the client posts its current rolling
        context right after PLAYING a chunk, and we retro-render that chunk's
        executed-vs-dream panel IMMEDIATELY instead of waiting for the next infer.
        Kills the one-chunk viewer latency. The payload reuses the infer context
        format ("context_rgb" [T,H,W,3], optional session_id) so the client sends
        the exact array it already maintains. Idempotent: pending vis is consumed,
        so the next infer pushes its own plan-time panel instead of re-retroing."""
        if self.vis_hub is None or self._pending_chunk_vis is None:
            return {"ok": True, "retro_rendered": False}
        try:
            if msg.get("executed_rgb") is not None:
                # EXPLICIT contract (preferred): exactly the frames played this chunk,
                # in order. We prepend the plan-time obs so dream[0] (the boundary)
                # aligns at index 0 and dream[i] pairs with executed[i-1] exactly —
                # immune to cold-start (ctx=1) and client window top-up schemes.
                ex = np.asarray(msg["executed_rgb"])
                if ex.ndim == 3:
                    ex = ex[None]
                if ex.dtype == np.uint8:
                    ex = ex.astype(np.float32) / 255.0
                pobs = self._pending_chunk_vis.get("obs")
                plan = np.asarray(getattr(pobs, "rgb", ex[:1]))
                if plan.ndim == 4:           # (B,H,W,3) -> (H,W,3)
                    plan = plan[0]
                context_rgb = np.concatenate([plan[None], ex], axis=0)
                self._pending_chunk_vis["exec_n"] = int(ex.shape[0])
            else:
                context_rgb = _context_rgb_from_msg(msg)
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": f"bad executed/context rgb: {e!r}"}
        retro = self._make_retro_chunk_policy_vis(context_rgb)
        if retro is None:
            return {"ok": True, "retro_rendered": False}
        self._pending_chunk_vis = None
        try:
            ctx = context_rgb[0] if context_rgb.ndim == 5 else context_rgb
            input_rgb = (np.asarray(ctx[-1]) * 255).astype(np.uint8)
            self.vis_hub.record_request(
                kind="enqueue",            # obs-only push (existing stats bucket)
                input_rgb=input_rgb,
                policy_vis=retro,
                infer_ms=None,
                action=None,
            )
        except Exception:
            logger.exception("[vis] observe push failed (continuing)")
            return {"ok": False, "error": "vis push failed"}
        return {"ok": True, "retro_rendered": True}

    def _push_vis(self, context_rgb, action, infer_s, oinfo, retro_vis=None) -> None:
        """Feed one chunk's vis to the VisHub. Best-effort: a viewer hiccup never breaks inference.

        When ``retro_vis`` is available (the previous chunk re-rendered against the now-executed
        frames), it is the policy_vis we stream — its "current/executed" panel evolves alongside the
        dream. On the first chunk (no pending) we fall back to the plan-time policy_vis (static
        current panel — unavoidable, the real frames don't exist yet)."""
        try:
            ctx = context_rgb[0] if context_rgb.ndim == 5 else context_rgb   # (T,H,W,3) in [0,1]
            input_rgb = (np.asarray(ctx[-1]) * 255).astype(np.uint8)
            policy_vis = retro_vis if retro_vis is not None else oinfo.get("policy_vis")
            self.vis_hub.record_request(
                kind="predict",
                input_rgb=input_rgb,
                policy_vis=policy_vis,
                infer_ms=infer_s * 1000.0,
                action=action,
                desired_flow=oinfo.get("desired_flow"),
                desired_source_rgb=oinfo.get("desired_source_rgb"),
                track_stats=oinfo.get("track_stats"),
                control_track_stats=oinfo.get("control_track_stats"),
                outlier_stats=oinfo.get("outlier_stats"),
                jacobian_abs_stats=oinfo.get("jacobian_abs_stats"),
                action_debug=oinfo.get("action_debug"),
                dream_rollout=oinfo.get("dream_rollout"),   # context/executed/lookahead strip
            )
        except Exception:
            logger.exception("[vis] record_request failed (continuing)")

    def _make_retro_chunk_policy_vis(self, context_rgb):
        """Re-render the PREVIOUS chunk's dream frames with ``current_obs_rgb`` set to the frames the
        robot ACTUALLY executed (the tail of this call's context_rgb). The plan-time vis only has one
        real observation, so its "current" panel is frozen; this pairs each dream frame with the real
        frame that followed it. Returns a (T,H,W,3) float[0,1] vis stack, or None on the first chunk.
        Ported from the old MotionPolicyServer._make_retro_chunk_policy_vis."""
        pending = self._pending_chunk_vis
        if pending is None:
            return None
        obs = pending.get("obs")
        frames = pending.get("frames")
        render = getattr(self._policy, "_make_policy_vis_joint", None)
        if obs is None or getattr(obs, "rgb_vis", None) is None or not frames or render is None:
            return None
        uploaded = np.asarray(context_rgb)
        if uploaded.ndim == 4:
            uploaded = uploaded[None, ...]
        if uploaded.ndim != 5 or uploaded.shape[-1] != 3:
            return None
        ctx_len = int(uploaded.shape[1])
        if ctx_len <= 0 or not frames:
            return None
        # ALIGNMENT (the subtle part): dream frame 0 is the boundary == the plan-time obs; dream
        # frames 1..exec_n are the steps we ACTUALLY executed; frames exec_n+1..N are lookahead that
        # never ran. The executed obs are the last exec_n frames of this call's context, so the plan
        # obs (dream[0]) sits at context index (ctx_len-1-exec_n). Pair dream[i] -> context[that+i].
        # Lookahead frames (i>exec_n, or index past the buffer) get NO real frame -> they keep the
        # static plan-obs fallback, which is honest (there is no executed reality to show for them).
        exec_n = int(pending.get("exec_n") or 0)
        if exec_n <= 0 or exec_n > ctx_len - 1:
            # Unknown / unusable exec count -> fall back to end-alignment over the shared tail.
            exec_n = min(len(frames), ctx_len) - 1
        base = ctx_len - 1 - exec_n            # context index of dream[0] (the plan-time obs)
        retro_frames = []
        for i in range(len(frames)):
            f = dict(frames[i])
            ctx_idx = base + i
            if i <= exec_n and 0 <= ctx_idx <= ctx_len - 1:
                f["current_obs_rgb"] = uploaded[:, ctx_idx]
            retro_frames.append(f)
        try:
            return render(obs, retro_frames, dream_index=pending.get("dream_index"), action=None)
        except Exception:
            logger.exception("[vis] retro chunk render failed (continuing)")
            return None

    def _store_pending_chunk_vis(
        self, obs: PolicyObservation, out: PolicyOutput, *, exec_n: int
    ) -> None:
        """Stash this chunk's dream frames so the NEXT infer can retro-render them against the
        executed observations. ``exec_n`` is how many actions the controller will play before the
        next infer (== the number of executed obs that will appear in the next context tail) — the
        retro pairing uses it for alignment. Cleared if vis is off or no chunk_story frames."""
        info = out.info or {}
        frames = info.get("chunk_story_vis_frames")
        if getattr(obs, "rgb_vis", None) is None or not frames:
            self._pending_chunk_vis = None
            return
        self._pending_chunk_vis = {
            "obs": obs,
            "frames": frames,
            "dream_index": info.get("chunk_story_index"),
            "exec_n": int(exec_n),
        }

    # -- shutdown flush ------------------------------------------------------ #
    def flush(self) -> None:
        """Forward the transport's shutdown flush to the policy so the async debug-dump
        writer drains its queue before the process exits (no lost final chunks)."""
        f = getattr(self._policy, "flush", None)
        if callable(f):
            f()

    # -- BasePolicy.reset ---------------------------------------------------- #
    def reset(self, reset_info: Dict[str, Any]) -> None:
        # Single source of truth for ALL episode state: context queue, controller, dream index,
        # AND any AR KV-cache living inside the planner. The adapter never pokes model internals.
        self._policy.reset()
        self._first_infer_since_reset = True
        self._pending_chunk_vis = None   # don't retro-render a new episode against the old chunk
        self._adaptive_prev = None       # poses don't carry across episodes; learned gains do
        sid = (reset_info or {}).get("session_id")
        if sid is not None:
            self._session_id = sid
        reason = (reset_info or {}).get("reason")
        # Clear the viewer on reset EXCEPT on "episode_end": a kill/stop sends
        # reset(reason="episode_end") purely to flush artifacts, but we keep the last
        # episode's viewer timeline ALIVE so the operator can study it after a kill.
        # The next run's reset(reason="new_episode") — or a session_change — wipes it
        # fresh at the START of the next run. (Previously cleared on EVERY reset incl.
        # episode_end, so the dashboard went blank the instant the run was killed.)
        # Stale-frame guard from past episodes still holds: new_episode/session_change clear.
        if self.vis_hub is not None and reason != "episode_end":
            try:
                self.vis_hub.clear_history()
            except Exception:
                logger.exception("[reset] vis_hub.clear_history failed (continuing)")
        logger.info("[reset] session=%s reason=%s", sid, reason)
