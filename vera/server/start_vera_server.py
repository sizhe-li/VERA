"""Launch the Vera policy server (new DreamZero-derived protocol).

Builds the two-stage policy for one embodiment via the factory, wraps it in a
``VeraPolicyAdapter``, optionally enables the measured speedups (DiT teacache, default ON at
the near-lossless threshold), and serves it over the websocket transport.

    # DROID FR3, per-embodiment WAN + jacobian, teacache on (default)
    python -m vera.server.start_vera_server --embodiment droid --port 8000

    # Allegro hand
    python -m vera.server.start_vera_server --embodiment allegro --port 8001

    # Swap the planner to the omni model (one flag) + a text prompt
    python -m vera.server.start_vera_server --embodiment droid \
        --algo-config /path/to/omni/algo_config.yaml --text "pick up the red block"

Speedups (measured: docs/SPEEDUP_REPORT.md). flash_attn is a property of the conda env (use
sizheli-okto). DiT teacache is enabled here by default at rel_l1_thresh=0.10 — the near-lossless
operating point (~1.4x, PSNR 46 / LPIPS 0.09). DO NOT raise much past 0.15 (quality cliff).
"""
from __future__ import annotations

import argparse
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("vera.start")

# near-lossless teacache threshold (see docs/SPEEDUP_REPORT.md — knee is ~0.15).
DEFAULT_TEACACHE_THRESH = 0.10


def _enable_teacache(policy, thresh: float) -> bool:
    """Enable DiT teacache on the policy's WAN planner. Returns True if applied.

    The WAN algo is ``policy.motion_planner``; its DiT is ``.model`` (``._model.model`` for a
    pipeline-wrapped planner). Non-WAN / AR planners that lack a WanModel are skipped (logged).
    """
    from vera.video_model.algorithms.wan.dit_cache import enable_dit_cache

    planner = getattr(policy, "motion_planner", None)
    dit = getattr(planner, "model", None)
    if dit is None and planner is not None:
        dit = getattr(getattr(planner, "_model", None), "model", None)
    if dit is None:
        logger.warning("[teacache] no WAN DiT found on policy.motion_planner; skipping")
        return False
    enable_dit_cache(dit, rel_l1_thresh=thresh)
    logger.info("[teacache] enabled rel_l1_thresh=%.3f on %s", thresh, type(dit).__name__)
    return True


def main():
    ap = argparse.ArgumentParser(description="Vera policy server (new protocol)")
    ap.add_argument("--embodiment", required=True, choices=["droid", "allegro", "mimicgen", "pusht"])
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--algo-config", default=None,
                    help="WAN planner algo_config.yaml. Point at the omni config to swap models.")
    ap.add_argument("--dynamics-run-id", default=None, help="jacobian/IDM wandb run id (per-embodiment default if unset)")
    ap.add_argument("--text", default=None, help="text conditioning prompt")
    ap.add_argument("--sample-steps", type=int, default=None, help="WAN denoise steps (default: yaml; deploy uses 10)")
    ap.add_argument("--no-teacache", action="store_true", help="disable DiT teacache speedup")
    ap.add_argument("--teacache-thresh", type=float, default=DEFAULT_TEACACHE_THRESH,
                    help=f"teacache rel_l1 threshold (default {DEFAULT_TEACACHE_THRESH}; >0.15 hits a quality cliff)")
    ap.add_argument("--debug-dump-dir", default=None,
                    help="per-chunk npz dump dir (default: $VERA_DEBUG_DUMP_DIR; droid defaults ON "
                         "to /path/to/data/jacobian/vera_rollouts/droid_live)")
    ap.add_argument("--no-debug-dump", action="store_true",
                    help="disable per-chunk debug dumps even for droid")
    ap.add_argument("--vis-port", type=int, default=0,
                    help="MJPEG live viewer port (0=off). Watch dream/tracks/actions at http://<host>:<vis-port>/")
    args = ap.parse_args()

    from vera.server.protocol.adapter_factory import make_adapter
    from vera.server.protocol.websocket_policy_server import WebsocketPolicyServer

    # Optional live viewer (MJPEG dashboard). One hub for the server's lifetime; re-attached to each
    # adapter (initial + after a reload) so switching ckpts keeps the viewer alive.
    vis_hub = None
    if args.vis_port:
        from vera.server.vis_server import VisHub, start_vis_server
        vis_hub = VisHub()
        vis_hub.set_metadata({"embodiment": args.embodiment, "policy_port": args.port})
        start_vis_server(vis_hub, host=args.host, port=args.vis_port)
        logger.info("[vis] live viewer at http://%s:%d/", args.host if args.host != "0.0.0.0" else "localhost", args.vis_port)

    def _build(embodiment, algo_config, dynamics_run_id, sample_steps, action_horizon):
        logger.info("building %s policy (algo_config=%s dynamics=%s steps=%s H=%s)...",
                    embodiment, algo_config, dynamics_run_id, sample_steps, action_horizon)
        ad = make_adapter(
            embodiment, algo_config_path=algo_config, dynamics_run_id=dynamics_run_id,
            text=args.text, sample_steps=sample_steps, action_horizon=action_horizon,
            run_dir=f"vera_server_{embodiment}_{args.port}",
        )
        if not args.no_teacache:
            _enable_teacache(ad._policy, args.teacache_thresh)
        else:
            logger.info("[teacache] disabled by --no-teacache")
        # Full-reproducibility rollout dumps (dreams/jacobians/flow per chunk) for
        # live robot sessions. Precedence: --no-debug-dump > --debug-dump-dir >
        # $VERA_DEBUG_DUMP_DIR > droid default-ON (live robot data is too precious
        # to lose to a forgotten env var — NORA request, DEPLOY_LOG #11).
        _dump_dir = args.debug_dump_dir or os.environ.get("VERA_DEBUG_DUMP_DIR")
        if _dump_dir is None and embodiment == "droid":
            _dump_dir = "/path/to/data/jacobian/vera_rollouts/droid_live"
        if args.no_debug_dump:
            _dump_dir = None
        if _dump_dir:
            try:
                ad._policy.configure_runtime(debug_dump_enabled=True, debug_dump_dir=_dump_dir)
                logger.info("[debug-dump] enabled -> %s", _dump_dir)
            except Exception:
                logger.exception("[debug-dump] configure failed (continuing)")
        ad.vis_hub = vis_hub                         # attach viewer (None when --vis-port off)
        if vis_hub is not None:                      # params shown in the viewer's top bar
            c = ad.config
            eff_steps = sample_steps                  # None -> read the actual value from the built policy
            if eff_steps is None:
                try:
                    eff_steps = int(ad._policy.cfg.motion_planner.diffusion_sampling_timesteps)
                except Exception:
                    eff_steps = None
            vis_hub.set_metadata({
                "embodiment": c.embodiment,
                "planner_model": c.planner_model,
                "idm_model": c.idm_model,
                "action_horizon": int(c.action_horizon),
                "control_hz": round(1.0 / c.control_dt, 1) if c.control_dt else None,
                "action_space": c.action_space,
                "sample_steps": int(eff_steps) if eff_steps is not None else "yaml",
                "teacache_thresh": (None if args.no_teacache else float(args.teacache_thresh)),
                "git": (c.git_head or "")[:8] + (" (dirty)" if c.git_dirty else ""),
            })
        return ad

    def _reload_fn(params, old_policy):
        """Hot-swap: free the old model's GPU memory, then build a new adapter from `params`
        (falling back to the server's launch defaults for anything unspecified)."""
        import gc
        import torch
        if old_policy is not None:                      # drop old model + reclaim VRAM first
            # Stop the old policy's async debug-dump writer thread FIRST. Its loop target is
            # a bound method holding a strong ref to the old MotionPolicy (-> its GPU model),
            # so a still-running writer pins the old model in VRAM and defeats the reclaim
            # below (it would leak one thread + a full model per reload). flush() drains the
            # queue and joins the thread, releasing that reference.
            try:
                f = getattr(old_policy, "flush", None)
                if callable(f):
                    f()
            except Exception:
                logger.exception("[reload] old-policy flush failed (continuing)")
            try:
                old_policy._policy = None
            except Exception:
                pass
            del old_policy
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("[reload] freed old model; GPU mem now %.1f/%.1f GB free",
                            *(x / 1024**3 for x in torch.cuda.mem_get_info()))
        ad = _build(
            params.get("embodiment", args.embodiment),
            params.get("algo_config", args.algo_config),
            params.get("dynamics_run_id", args.dynamics_run_id),
            int(params.get("sample_steps", args.sample_steps or 10)),
            int(params.get("action_horizon", 10)),
        )
        return ad, ad.config

    adapter = _build(args.embodiment, args.algo_config, args.dynamics_run_id, args.sample_steps, 10)
    server = WebsocketPolicyServer(adapter, adapter.config, host=args.host, port=args.port,
                                   reload_fn=_reload_fn)
    server.serve_forever()


if __name__ == "__main__":
    main()
