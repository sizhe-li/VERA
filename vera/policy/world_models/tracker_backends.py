from __future__ import annotations

from typing import Any

import torch

from .alltracker_inference import AllTrackerConfig, AllTrackerInference
from .cotracker_inference import CoTrackerConfig, CoTrackerInference
from .runtime_motion_tracks import (
    TrackerInferenceOutput,
    merge_runtime_tracks_across_views,
    stitch_view_visualizations,
)


def tracker_backend_from_cfg(cfg: Any) -> str:
    backend = getattr(cfg, "tracker_backend", None)
    if backend is None:
        return "alltracker"
    return str(backend)


def tracker_enabled_from_cfg(cfg: Any) -> bool:
    if hasattr(cfg, "tracker_enabled"):
        return bool(getattr(cfg, "tracker_enabled"))
    return bool(getattr(cfg, "alltracker_enabled", True))


def tracker_return_visualization_from_cfg(cfg: Any) -> bool:
    if hasattr(cfg, "tracker_return_visualization"):
        return bool(getattr(cfg, "tracker_return_visualization"))
    return bool(getattr(cfg, "alltracker_return_visualization", True))


def build_motion_tracker(cfg: Any, device: str | torch.device):
    backend = tracker_backend_from_cfg(cfg)
    if backend == "alltracker":
        track_cfg = AllTrackerConfig(
            chunk_size=getattr(cfg, "alltracker_chunk_size", None),
            rate=int(getattr(cfg, "alltracker_rate", 2)),
            query_frame=int(getattr(cfg, "alltracker_query_frame", 0)),
            inference_iters=int(getattr(cfg, "alltracker_inference_iters", 4)),
            conf_thr=float(getattr(cfg, "alltracker_conf_thr", 0.60)),
            bkg_opacity=float(getattr(cfg, "alltracker_bkg_opacity", 0.0)),
        )
        return AllTrackerInference(config=track_cfg, device=device)
    if backend == "cotracker":
        cotracker_cfg = CoTrackerConfig(
            model_name=str(
                getattr(cfg, "cotracker_model_name", "cotracker3_offline")
            ),
            grid_size=int(getattr(cfg, "cotracker_grid_size", 15)),
        )
        return CoTrackerInference(config=cotracker_cfg, device=device)
    if backend == "megaflow":
        from .megaflow_inference import MegaFlowConfig, MegaFlowInference
        megaflow_cfg = MegaFlowConfig(
            model_name=str(getattr(cfg, "megaflow_model_name", "megaflow-track")),
            num_reg_refine=int(getattr(cfg, "megaflow_num_reg_refine", 8)),
            query_frame=int(getattr(cfg, "megaflow_query_frame", 0)),
            rate=int(getattr(cfg, "megaflow_rate", 4)),
            autocast_dtype=str(getattr(cfg, "megaflow_autocast_dtype", "bfloat16")),
            vis_from_flow_mag=bool(getattr(cfg, "megaflow_vis_from_flow_mag", False)),
            vis_flow_mag_thresh=float(getattr(cfg, "megaflow_vis_flow_mag_thresh", 96.0)),
            bkg_opacity=float(getattr(cfg, "megaflow_bkg_opacity", 0.0)),
        )
        return MegaFlowInference(config=megaflow_cfg, device=device)
    raise ValueError(f"Unsupported tracker backend: {backend}")


def infer_multiview_tracks(
    tracker: Any,
    rgb: torch.Tensor,
    *,
    return_visualization: bool,
    view_keys: list[str] | None = None,
    view_widths: list[int] | None = None,
) -> TrackerInferenceOutput:
    if view_widths is None or len(view_widths) <= 1:
        print(
            "[tracker_backends] single-view tracker input "
            f"rgb_shape={tuple(rgb.shape)} view_keys={view_keys} "
            f"view_widths={view_widths}",
            flush=True,
        )
        return tracker.infer(rgb, return_visualization=return_visualization)

    if sum(int(width) for width in view_widths) != int(rgb.shape[-1]):
        raise ValueError(
            f"view_widths do not match rgb width: {view_widths} vs {int(rgb.shape[-1])}"
        )
    if view_keys is not None and len(view_keys) != len(view_widths):
        raise ValueError(
            f"view_keys/view_widths mismatch: {len(view_keys)} vs {len(view_widths)}"
        )

    print(
        "[tracker_backends] multiview tracker input "
        f"rgb_shape={tuple(rgb.shape)} view_keys={view_keys} "
        f"view_widths={[int(width) for width in view_widths]}",
        flush=True,
    )
    per_view_outputs = []
    start = 0
    for view_index, view_width in enumerate(view_widths):
        end = start + int(view_width)
        per_view_rgb = rgb[..., start:end]
        print(
            "[tracker_backends] tracker per-view slice "
            f"view_index={view_index} "
            f"view_key={None if view_keys is None else view_keys[view_index]} "
            f"x_range=({start},{end}) shape={tuple(per_view_rgb.shape)}",
            flush=True,
        )
        per_view_outputs.append(
            tracker.infer(
                per_view_rgb,
                return_visualization=return_visualization,
            )
        )
        start = end

    merged_tracks = merge_runtime_tracks_across_views(
        [output.motion_tracks for output in per_view_outputs],
        view_widths=[int(width) for width in view_widths],
        view_keys=view_keys,
    )
    merged_vis = stitch_view_visualizations(
        [
            output.visualization
            for output in per_view_outputs
            if output.visualization is not None
        ]
    )
    merged_tracks.meta["per_view_outputs"] = [
        {
            "view_key": None if view_keys is None else view_keys[view_index],
            "view_index": view_index,
        }
        for view_index in range(len(view_widths))
    ]
    print(
        "[tracker_backends] merged tracker output "
        f"image_size={merged_tracks.image_size} "
        f"xy_src_shape={tuple(merged_tracks.xy_src.shape)} "
        f"meta_view_widths={merged_tracks.meta.get('view_widths')} "
        f"meta_view_keys={merged_tracks.meta.get('view_keys')}",
        flush=True,
    )
    return TrackerInferenceOutput(
        motion_tracks=merged_tracks,
        visualization=merged_vis,
        per_view_outputs=per_view_outputs,
    )
