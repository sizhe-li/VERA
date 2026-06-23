"""
Motion planner factory: builds inference-only world model adapters (e.g. WAN)
without putting them in the training algorithm registry. Trainable planners
(e.g. dfot_motion_policy_joint) are still built via the algorithm registry.
"""

from typing import Any
import torch


_WAN_BIDIR_NAMES = ("wan_t2v", "wan_i2v")
_WAN_AR_NAMES = ("wan_ar_df", "wan_ar_tf")


def _maybe_autodetect_ar_recipe(name: str, algo_cfg: Any) -> str | None:
    """If ``algo_cfg`` has an ``ar.recipe`` block and the name still says
    ``wan_t2v`` / ``wan_i2v`` (because the AR base recipe inherits from those
    via hydra ``defaults``), upgrade to ``wan_ar_df`` / ``wan_ar_tf``.

    Lets users hand the registry a training-run yaml unmodified.
    """
    if name in _WAN_AR_NAMES:
        return name
    if not hasattr(algo_cfg, "get"):
        return None
    ar_block = algo_cfg.get("ar", None)
    if ar_block is None:
        return None
    recipe = ar_block.get("recipe", None) if hasattr(ar_block, "get") else None
    if recipe in ("df", "tf"):
        return f"wan_ar_{recipe}"
    return None


def _build_motion_track_config(algo_cfg: Any):
    """Shared MotionTrackConfig builder used by both bidirectional and AR paths."""
    from vera.video_model.link.wan_pipeline import MotionTrackConfig

    tracker_cfg = algo_cfg.get("tracker", {}) if hasattr(algo_cfg, "get") else {}
    alltracker_cfg = (
        algo_cfg.get("alltracker", {}) if hasattr(algo_cfg, "get") else {}
    )
    cotracker_cfg = (
        algo_cfg.get("cotracker", {}) if hasattr(algo_cfg, "get") else {}
    )
    return MotionTrackConfig(
        backend=str(tracker_cfg.get("backend", "alltracker")),
        enabled=bool(
            tracker_cfg.get("enabled", alltracker_cfg.get("enabled", True))
        ),
        return_visualization=bool(
            tracker_cfg.get(
                "return_visualization",
                alltracker_cfg.get("return_visualization", True),
            )
        ),
        chunk_size=alltracker_cfg.get("chunk_size", None),
        rate=int(alltracker_cfg.get("rate", 2)),
        query_frame=int(alltracker_cfg.get("query_frame", 0)),
        inference_iters=int(alltracker_cfg.get("inference_iters", 4)),
        conf_thr=float(alltracker_cfg.get("conf_thr", 0.60)),
        bkg_opacity=float(alltracker_cfg.get("bkg_opacity", 0.0)),
        temporal_stride=int(alltracker_cfg.get("temporal_stride", 1)),
        cotracker_model_name=str(
            cotracker_cfg.get("model_name", "cotracker3_offline")
        ),
        cotracker_grid_size=int(cotracker_cfg.get("grid_size", 15)),
    )


def build_motion_planner(
    name: str,
    algo_cfg: Any,
    device: str | torch.device = "cuda:0",
    *,
    config_path: str | None = None,
    ckpt_path: str | None = None,
):
    """
    Build a motion planner instance from config.

    - ``"wan_t2v"`` / ``"wan_i2v"``: bidirectional ``WanAllTrackerPipeline``
      (existing 14B / 1.3B path), optionally with runtime AllTracker motion
      tracks.
    - ``"wan_ar_df"`` / ``"wan_ar_tf"``: causal ``WanARTrackerPipeline``
      built around the wan_ar AR sampler (block-causal, KV-cache, ctx cap
      from training-time ``max_train_latent_frames``). Same MotionTrackConfig
      surface as the bidirectional path so downstream policy code is unchanged.
    - Any other name: return ``None``; caller should use
      ``resolve_algorithm_instance(algo_cfg)``.
    """
    # Auto-upgrade name when caller passed a generic bidirectional name but
    # cfg is shaped like AR (cfg.ar.recipe present). This lets training yamls
    # work as-is without manually overriding ``algorithm.name``.
    autodetected = _maybe_autodetect_ar_recipe(name, algo_cfg)
    if autodetected is not None:
        name = autodetected

    if name in _WAN_BIDIR_NAMES:
        from vera.video_model.link.wan_pipeline import WanAllTrackerPipeline

        motion_track_config = _build_motion_track_config(algo_cfg)
        if config_path is not None:
            return WanAllTrackerPipeline.from_config(
                config_path,
                ckpt_path=ckpt_path,
                device=device,
                motion_track_config=motion_track_config,
            )
        return WanAllTrackerPipeline(
            algo_cfg,
            ckpt_path=ckpt_path,
            device=device,
            motion_track_config=motion_track_config,
        )
    if name in _WAN_AR_NAMES:
        from vera.policy.wan_ar_planner import WanARTrackerPipeline

        motion_track_config = _build_motion_track_config(algo_cfg)
        return WanARTrackerPipeline(
            algo_cfg,
            ckpt_path=ckpt_path,
            device=device,
            motion_track_config=motion_track_config,
        )
    return None
