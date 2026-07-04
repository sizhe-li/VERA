"""MegaFlow tracker backend for okto MotionPolicy.

Mirrors the contract of `AllTrackerInference` and `CoTrackerInference` so it
plugs into `tracker_backends.build_motion_tracker` with no other code changes.

MegaFlow's `forward_track` produces flow from a query frame (default: frame 0)
to every frame in a clip; we convert that into the pairwise (frame i → i+1)
`RuntimeMotionTracks` layout the rest of the pipeline expects.

Install:
    pip install -e third_party/megaflow
The pyproject pins python>=3.12 but the package itself works on 3.11 — pass
`--ignore-requires-python` if your env (e.g. okto) is on 3.11. Weights are
auto-downloaded from HuggingFace on first call.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from einops import rearrange

from vera.utils.alltracker_utils import draw_pts_gpu

from .runtime_motion_tracks import (
    RuntimeMotionTracks,
    TrackerInferenceOutput,
    xy_to_idx_tensor,
)


def _get_2d_colors(xys: np.ndarray, height: int, width: int) -> np.ndarray:
    if xys.ndim != 2 or xys.shape[1] != 2:
        raise ValueError(f"Expected [N, 2] coordinates, got {xys.shape}")
    x_norm = np.clip(xys[:, 0] / max(width - 1, 1), 0.0, 1.0)
    y_norm = np.clip(xys[:, 1] / max(height - 1, 1), 0.0, 1.0)
    colors = np.stack(
        [x_norm, y_norm, 1.0 - 0.5 * (x_norm + y_norm)],
        axis=-1,
    )
    return np.clip(colors * 255.0, 0.0, 255.0).astype(np.uint8)


@dataclass
class MegaFlowConfig:
    model_name: str = "megaflow-track"
    num_reg_refine: int = 8
    query_frame: int = 0
    rate: int = 4  # vis-only: subsample factor when drawing tracks
    autocast_dtype: str = "bfloat16"
    # MegaFlow gives no native visibility flag. If True, mark a track as
    # invalid when its absolute displacement exceeds `vis_flow_mag_thresh`
    # pixels (typically out-of-frame / large occlusion). If False, all valid.
    vis_from_flow_mag: bool = False
    vis_flow_mag_thresh: float = 96.0
    bkg_opacity: float = 0.0


class MegaFlowInference:
    """Thin wrapper around MegaFlow.forward_track for in-memory videos."""

    def __init__(
        self,
        config: MegaFlowConfig | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.config = config or MegaFlowConfig()
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self._model: torch.nn.Module | None = None

    def _load_model(self) -> torch.nn.Module:
        if self._model is not None:
            return self._model
        try:
            from megaflow import MegaFlow
        except ImportError as exc:
            raise RuntimeError(
                "megaflow tracker backend selected but the `megaflow` package is "
                "not importable. Install with:\n"
                "    pip install -e /path/to/megaflow --ignore-requires-python"
            ) from exc
        model = MegaFlow.from_pretrained(self.config.model_name, device=str(self.device))
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.eval()
        self._model = model
        return model

    def _preprocess_video(self, video: torch.Tensor) -> torch.Tensor:
        """Coerce video to [B, T, 3, H, W] float32 in [0, 255] on self.device."""
        if video.ndim != 5 or video.shape[2] != 3:
            raise ValueError(
                "Expected video with shape [B, T, 3, H, W] in [0, 1], [-1, 1], "
                f"or [0, 255], got {tuple(video.shape)}"
            )
        video = video.to(device=self.device, dtype=torch.float32)
        min_val = float(video.min().item())
        max_val = float(video.max().item())
        if min_val >= 0.0 and max_val <= 1.0 + 1e-6:
            return video * 255.0
        if min_val >= -1.0 - 1e-6 and max_val <= 1.0 + 1e-6:
            return (video.clamp(-1.0, 1.0) + 1.0) * 127.5
        return video.clamp(0.0, 255.0)

    def _autocast_dtype(self) -> torch.dtype:
        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(self.config.autocast_dtype.lower(), torch.bfloat16)

    @torch.no_grad()
    def _forward(self, video: torch.Tensor) -> torch.Tensor:
        """Run MegaFlow.forward_track and return absolute tracks `[B, T, H*W, 2]`."""
        model = self._load_model()
        b, t, _, h, w = video.shape

        use_autocast = self.device.type == "cuda" and self._autocast_dtype() != torch.float32
        ctx = (
            torch.autocast(device_type="cuda", dtype=self._autocast_dtype())
            if use_autocast
            else _NullCtx()
        )
        with ctx:
            results = model.forward_track(
                video,
                num_reg_refine=int(self.config.num_reg_refine),
            )
        flow_final = results["flow_final"].to(dtype=torch.float32, device=self.device)
        # When T==2 the model returns flow_preds[-1] which is [B, T-1, 2, H, W];
        # otherwise [B, T, 2, H, W] from the query frame. Normalize to
        # [B, T, 2, H, W] so the rest of this method has one shape.
        if flow_final.shape[1] == t - 1:
            zero_flow = torch.zeros(b, 1, 2, h, w, dtype=flow_final.dtype, device=flow_final.device)
            flow_final = torch.cat([zero_flow, flow_final], dim=1)
        elif flow_final.shape[1] != t:
            raise RuntimeError(
                f"Unexpected MegaFlow flow_final shape {tuple(flow_final.shape)} "
                f"for video shape {tuple(video.shape)}"
            )

        # Build a per-pixel coordinate grid in (x, y) order: [1, 1, 2, H, W].
        ys, xs = torch.meshgrid(
            torch.arange(h, device=self.device, dtype=torch.float32),
            torch.arange(w, device=self.device, dtype=torch.float32),
            indexing="ij",
        )
        grid_xy = torch.stack([xs, ys], dim=0).unsqueeze(0).unsqueeze(0)  # [1,1,2,H,W]
        abs_xy = grid_xy + flow_final  # [B, T, 2, H, W]
        return rearrange(abs_xy, "b t c h w -> b t (h w) c")

    def _build_visibility(self, tracks: torch.Tensor) -> torch.Tensor:
        """Return [B, T, N] visibility tensor in [0, 1]."""
        b, t, n, _ = tracks.shape
        if not self.config.vis_from_flow_mag:
            return torch.ones(b, t, n, dtype=torch.float32, device=tracks.device)
        # Displacement from the query frame.
        query = tracks[:, self.config.query_frame : self.config.query_frame + 1]
        disp_mag = (tracks - query).norm(dim=-1)  # [B, T, N]
        return (disp_mag <= float(self.config.vis_flow_mag_thresh)).float()

    def _to_runtime_tracks(
        self,
        tracks: torch.Tensor,
        visibility: torch.Tensor,
        image_size: tuple[int, int],
    ) -> RuntimeMotionTracks:
        height, width = image_size
        tracks = tracks.detach().cpu().float()
        visibility = visibility.detach().cpu().float()
        xy_src = tracks[:, :-1]
        xy_tgt = tracks[:, 1:]
        vis_src = visibility[:, :-1]
        vis_tgt = visibility[:, 1:]
        return RuntimeMotionTracks(
            xy_src=xy_src,
            xy_tgt=xy_tgt,
            vis_src=vis_src,
            vis_tgt=vis_tgt,
            idx_src=xy_to_idx_tensor(xy_src, height, width),
            idx_tgt=xy_to_idx_tensor(xy_tgt, height, width),
            image_size=image_size,
            meta={
                "tracker_backend": "megaflow",
                "model_name": self.config.model_name,
                "num_reg_refine": int(self.config.num_reg_refine),
                "query_frame": int(self.config.query_frame),
                "vis_from_flow_mag": bool(self.config.vis_from_flow_mag),
            },
        )

    @torch.no_grad()
    def infer(
        self,
        video: torch.Tensor,
        return_visualization: bool = True,
    ) -> TrackerInferenceOutput:
        if video.shape[1] < 2:
            raise ValueError("MegaFlow tracker requires at least 2 frames")

        pixel_video = self._preprocess_video(video)
        image_size = (int(pixel_video.shape[-2]), int(pixel_video.shape[-1]))
        tracks = self._forward(pixel_video)              # [B, T, H*W, 2]
        visibility = self._build_visibility(tracks)      # [B, T, H*W]
        runtime_tracks = self._to_runtime_tracks(
            tracks=tracks, visibility=visibility, image_size=image_size
        )

        visualization = None
        if return_visualization:
            vis_videos = []
            tracks_cpu = tracks.detach().cpu().float()
            vis_cpu = visibility.detach().cpu().float()
            for batch_idx in range(pixel_video.shape[0]):
                xy0 = tracks_cpu[batch_idx, 0].numpy()
                colors = _get_2d_colors(xy0, image_size[0], image_size[1])
                vis_np = draw_pts_gpu(
                    pixel_video[batch_idx],
                    tracks_cpu[batch_idx],
                    vis_cpu[batch_idx],
                    colors,
                    rate=int(self.config.rate),
                    bkg_opacity=float(self.config.bkg_opacity),
                )
                vis_videos.append(torch.from_numpy(vis_np))
            visualization = torch.stack(vis_videos, dim=0)

        return TrackerInferenceOutput(
            motion_tracks=runtime_tracks,
            visualization=visualization,
        )


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False
