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
class CoTrackerConfig:
    model_name: str = "cotracker3_offline"
    grid_size: int = 15


class CoTrackerInference:
    """Thin wrapper around CoTracker for in-memory videos."""

    def __init__(
        self,
        config: CoTrackerConfig | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.config = config or CoTrackerConfig()
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        model = torch.hub.load(
            "facebookresearch/co-tracker",
            self.config.model_name,
        ).to(self.device)
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.eval()
        self._model = model
        return model

    def _preprocess_video(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5 or video.shape[2] != 3:
            raise ValueError(
                "Expected video with shape [B, T, 3, H, W] in [0, 1], [-1, 1], or [0, 255], "
                f"got {tuple(video.shape)}"
            )
        video = video.to(device=self.device, dtype=torch.float32)
        min_val = float(video.min().item())
        max_val = float(video.max().item())
        if min_val >= 0.0 and max_val <= 1.0 + 1e-6:
            return video * 255.0
        if min_val >= -1.0 - 1e-6 and max_val <= 1.0 + 1e-6:
            return (video.clamp(-1.0, 1.0) + 1.0) * 127.5
        return video.clamp(0.0, 255.0)

    def _to_runtime_tracks(
        self,
        tracks: torch.Tensor,
        visibility: torch.Tensor,
        image_size: tuple[int, int],
    ) -> RuntimeMotionTracks:
        height, width = image_size
        tracks = tracks.float().detach().cpu()
        visibility = visibility.float().detach().cpu()
        if visibility.ndim == 4 and visibility.shape[-1] == 1:
            visibility = visibility[..., 0]
        xy_src = tracks[:, :-1].float()
        xy_tgt = tracks[:, 1:].float()
        vis_src = visibility[:, :-1].float()
        vis_tgt = visibility[:, 1:].float()
        return RuntimeMotionTracks(
            xy_src=xy_src,
            xy_tgt=xy_tgt,
            vis_src=vis_src,
            vis_tgt=vis_tgt,
            idx_src=xy_to_idx_tensor(xy_src, height, width),
            idx_tgt=xy_to_idx_tensor(xy_tgt, height, width),
            image_size=image_size,
            meta={
                "tracker_backend": "cotracker",
                "model_name": self.config.model_name,
                "grid_size": int(self.config.grid_size),
            },
        )

    @torch.no_grad()
    def infer(
        self,
        video: torch.Tensor,
        return_visualization: bool = True,
    ) -> TrackerInferenceOutput:
        if video.shape[1] < 2:
            raise ValueError("CoTracker requires at least 2 frames")
        pixel_video = self._preprocess_video(video)
        image_size = (int(pixel_video.shape[-2]), int(pixel_video.shape[-1]))
        model = self._load_model()
        pred_tracks, pred_visibility = model(
            pixel_video,
            grid_size=int(self.config.grid_size),
        )
        runtime_tracks = self._to_runtime_tracks(
            tracks=pred_tracks,
            visibility=pred_visibility,
            image_size=image_size,
        )
        visualization = None
        if return_visualization:
            vis_videos = []
            visibility = pred_visibility
            if visibility.ndim == 4 and visibility.shape[-1] == 1:
                visibility = visibility[..., 0]
            visibility = visibility.detach().cpu().float()
            pred_tracks_cpu = pred_tracks.detach().cpu().float()
            for batch_idx in range(pixel_video.shape[0]):
                xy0 = pred_tracks_cpu[batch_idx, 0].numpy()
                colors = _get_2d_colors(xy0, image_size[0], image_size[1])
                vis_np = draw_pts_gpu(
                    pixel_video[batch_idx],
                    pred_tracks_cpu[batch_idx],
                    visibility[batch_idx],
                    colors,
                    rate=2,
                    bkg_opacity=0.0,
                )
                visualization = torch.from_numpy(vis_np)
                vis_videos.append(visualization)
            visualization = torch.stack(vis_videos, dim=0)
        return TrackerInferenceOutput(
            motion_tracks=runtime_tracks,
            visualization=visualization,
        )
