from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from einops import rearrange

from vera.utils.alltracker_utils import draw_pts_gpu

from .runtime_motion_tracks import (
    RuntimeMotionTracks,
    TrackerInferenceOutput,
    merge_runtime_tracks_across_views,
    stitch_view_visualizations,
    xy_to_idx_tensor,
)


def _ensure_alltracker_path() -> Path:
    """Add the local ``alltracker`` checkout to sys.path for local imports.

    Robust to layout: vera lives at ``…/third_party/vera`` (so ``parents[4]`` is already
    ``…/third_party``), but a future split-out may put it elsewhere. Try, in order:
      1. ``$VERA_ALLTRACKER_ROOT`` (explicit override),
      2. ``…/third_party/alltracker`` (current nested layout — parents[4] == third_party),
      3. ``…/third_party/alltracker`` via the repo root (parents[5]) — same dir, split-safe.
    The previous code used ``parents[4]/third_party/alltracker`` → ``third_party/third_party/
    alltracker`` (a non-existent path) because the okto→vera port added one nesting level.
    """
    import os
    import sys

    here = Path(__file__).resolve()
    candidates = []
    env = os.environ.get("VERA_ALLTRACKER_ROOT")
    if env:
        candidates.append(Path(env))
    candidates.append(here.parents[4] / "alltracker")            # …/third_party/alltracker
    candidates.append(here.parents[5] / "third_party" / "alltracker")  # split-out safety
    for root in candidates:
        if root.is_dir():
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            return root
    # none found: return the canonical expected path so the ImportError names a real location
    fallback = here.parents[4] / "alltracker"
    return fallback


def _import_alltracker():
    _ensure_alltracker_path()
    import alltracker.utils.basic as basic
    from alltracker.nets.alltracker import Net

    return basic, Net


def _get_2d_colors(xys: np.ndarray, height: int, width: int) -> np.ndarray:
    """Generate deterministic RGB colors from 2D pixel coordinates.

    This avoids importing AllTracker's visualization helpers, which require
    `skimage` even though inference itself does not.
    """
    if xys.ndim != 2 or xys.shape[1] != 2:
        raise ValueError(f"Expected [N, 2] coordinates, got {xys.shape}")

    x_norm = np.clip(xys[:, 0] / max(width - 1, 1), 0.0, 1.0)
    y_norm = np.clip(xys[:, 1] / max(height - 1, 1), 0.0, 1.0)
    colors = np.stack(
        [
            x_norm,
            y_norm,
            1.0 - 0.5 * (x_norm + y_norm),
        ],
        axis=-1,
    )
    return np.clip(colors * 255.0, 0.0, 255.0).astype(np.uint8)


@dataclass
class AllTrackerConfig:
    """Inference options for in-memory AllTracker runs."""

    window_len: int = 16
    rate: int = 2
    query_frame: int = 0
    inference_iters: int = 4
    conf_thr: float = 0.60
    bkg_opacity: float = 0.0
    chunk_size: int | None = None
    min_frames_per_chunk: int = 2
    checkpoint_url: str = (
        "https://huggingface.co/aharley/alltracker/resolve/main/alltracker.pth"
    )


AllTrackerInferenceOutput = TrackerInferenceOutput


class AllTrackerInference:
    """Thin wrapper around AllTracker for in-memory videos in [-1, 1]."""

    def __init__(
        self,
        config: AllTrackerConfig | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.config = config or AllTrackerConfig()
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self._model: torch.nn.Module | None = None
        self._basic = None

    def _load_model(self) -> torch.nn.Module:
        if self._model is not None:
            return self._model

        basic, Net = _import_alltracker()
        model = Net(self.config.window_len)
        state_dict = torch.hub.load_state_dict_from_url(
            self.config.checkpoint_url,
            map_location="cpu",
        )
        model.load_state_dict(state_dict["model"], strict=True)
        model.to(self.device)
        for _, parameter in model.named_parameters():
            parameter.requires_grad = False
        model.eval()

        self._basic = basic
        self._model = model
        return model

    def _xy_to_idx(self, xy: torch.Tensor, height: int, width: int) -> torch.Tensor:
        return xy_to_idx_tensor(xy, height, width)

    def _chunk_bounds(self, num_frames: int) -> list[tuple[int, int]]:
        chunk_size = self.config.chunk_size
        if chunk_size is None or num_frames <= chunk_size:
            return [(0, num_frames)]
        if chunk_size < 2:
            raise ValueError("AllTracker chunk_size must be at least 2")

        bounds: list[tuple[int, int]] = []
        start = 0
        while True:
            end = min(start + chunk_size, num_frames)
            bounds.append((start, end))
            if end >= num_frames:
                break
            start = end - 1

        if bounds and (bounds[-1][1] - bounds[-1][0]) < self.config.min_frames_per_chunk:
            prev_start, _ = bounds[-2]
            bounds[-2] = (prev_start, bounds[-1][1])
            bounds.pop()
        return bounds

    def _preprocess_video(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5 or video.shape[2] != 3:
            raise ValueError(
                "Expected video with shape [B, T, 3, H, W] in [0, 1] or [-1, 1], "
                f"got {tuple(video.shape)}"
            )
        video = video.to(dtype=torch.float32)
        if float(video.min().item()) >= 0.0:
            video = video.clamp(0.0, 1.0) * 2.0 - 1.0
        return ((video.clamp(-1.0, 1.0) + 1.0) * 127.5).to(
            device=self.device,
            dtype=torch.float32,
        )

    @torch.no_grad()
    def _forward_video(
        self,
        video: torch.Tensor,
        return_visualization: bool,
    ) -> dict[str, torch.Tensor]:
        model = self._load_model()
        assert self._basic is not None

        pixel_video = self._preprocess_video(video)
        _, t, _, height, width = pixel_video.shape
        grid_xy = self._basic.gridcloud2d(
            1, height, width, norm=False, device=str(pixel_video.device)
        ).float()
        grid_xy = grid_xy.permute(0, 2, 1).reshape(1, 1, 2, height, width)

        flows_e, visconf_maps_e, _, _ = model.forward_sliding(
            pixel_video[:, self.config.query_frame :],
            iters=self.config.inference_iters,
            sw=None,
            is_training=False,
        )

        # Upstream AllTracker returns a special pairwise format for 2-frame clips:
        # `flows_e` and `visconf_maps_e` are [B, 2, H, W] instead of
        # [B, T, 2, H, W]. Normalize that edge case so downstream code can keep
        # treating tracks as a time-indexed sequence.
        if flows_e.ndim == 4 and visconf_maps_e.ndim == 4:
            flow_map = flows_e.to(pixel_video.device).unsqueeze(1)
            target_xy = grid_xy.to(pixel_video.device) + flow_map
            traj_maps_e = torch.cat(
                [
                    grid_xy.to(pixel_video.device).expand(flow_map.shape[0], -1, -1, -1, -1),
                    target_xy,
                ],
                dim=1,
            )
            target_visibility = (
                visconf_maps_e[:, 1].to(pixel_video.device) > self.config.conf_thr
            ).unsqueeze(1)
            visibility_maps = torch.cat(
                [
                    torch.ones_like(target_visibility, dtype=torch.bool),
                    target_visibility,
                ],
                dim=1,
            )
        else:
            traj_maps_e = flows_e.to(pixel_video.device) + grid_xy.to(pixel_video.device)
            visibility_maps = (
                visconf_maps_e[:, :, 1].to(pixel_video.device) > self.config.conf_thr
            )

        tracks = rearrange(traj_maps_e, "b t c h w -> b t (h w) c")
        visibility = rearrange(visibility_maps, "b t h w -> b t (h w)")

        result = {
            "tracks": tracks.detach().cpu(),
            "visibility": visibility.detach().cpu().float(),
        }
        if not return_visualization:
            return result

        vis_videos = []
        for batch_idx in range(pixel_video.shape[0]):
            xy0 = tracks[batch_idx, 0].detach().cpu().numpy()
            colors = _get_2d_colors(xy0, height, width)
            vis_np = draw_pts_gpu(
                pixel_video[batch_idx],
                tracks[batch_idx],
                visibility[batch_idx],
                colors,
                rate=self.config.rate,
                bkg_opacity=self.config.bkg_opacity,
            )
            vis_videos.append(torch.from_numpy(vis_np))
        result["vis"] = torch.stack(vis_videos, dim=0)
        return result

    def _to_runtime_tracks(
        self,
        tracks: torch.Tensor,
        visibility: torch.Tensor,
        image_size: tuple[int, int],
    ) -> RuntimeMotionTracks:
        height, width = image_size
        xy_src = tracks[:, :-1].float()
        xy_tgt = tracks[:, 1:].float()
        vis_src = visibility[:, :-1].float()
        vis_tgt = visibility[:, 1:].float()
        return RuntimeMotionTracks(
            xy_src=xy_src,
            xy_tgt=xy_tgt,
            vis_src=vis_src,
            vis_tgt=vis_tgt,
            idx_src=self._xy_to_idx(xy_src, height, width),
            idx_tgt=self._xy_to_idx(xy_tgt, height, width),
            image_size=image_size,
            meta={
                "tracker_backend": "alltracker",
                "tracker_rate": self.config.rate,
                "conf_thr": self.config.conf_thr,
                "query_frame": self.config.query_frame,
            },
        )

    @torch.no_grad()
    def infer(
        self,
        video: torch.Tensor,
        return_visualization: bool = True,
    ) -> AllTrackerInferenceOutput:
        if video.shape[1] < 2:
            raise ValueError("AllTracker requires at least 2 frames")

        image_size = (int(video.shape[-2]), int(video.shape[-1]))
        bounds = self._chunk_bounds(int(video.shape[1]))
        runtime_tracks: list[RuntimeMotionTracks] = []
        visualizations: list[torch.Tensor] = []

        for start, end in bounds:
            chunk_outputs = self._forward_video(
                video[:, start:end],
                return_visualization=return_visualization,
            )
            runtime_tracks.append(
                self._to_runtime_tracks(
                    tracks=chunk_outputs["tracks"],
                    visibility=chunk_outputs["visibility"],
                    image_size=image_size,
                )
            )
            if return_visualization and "vis" in chunk_outputs:
                vis_chunk = chunk_outputs["vis"]
                if visualizations:
                    vis_chunk = vis_chunk[:, 1:]
                visualizations.append(vis_chunk)

        if len(runtime_tracks) == 1:
            merged_tracks = runtime_tracks[0]
        else:
            merged_tracks = RuntimeMotionTracks(
                xy_src=torch.cat([chunk.xy_src for chunk in runtime_tracks], dim=1),
                xy_tgt=torch.cat([chunk.xy_tgt for chunk in runtime_tracks], dim=1),
                vis_src=torch.cat([chunk.vis_src for chunk in runtime_tracks], dim=1),
                vis_tgt=torch.cat([chunk.vis_tgt for chunk in runtime_tracks], dim=1),
                idx_src=torch.cat([chunk.idx_src for chunk in runtime_tracks], dim=1),
                idx_tgt=torch.cat([chunk.idx_tgt for chunk in runtime_tracks], dim=1),
                image_size=image_size,
                meta=dict(runtime_tracks[0].meta),
            )

        visualization = None
        if visualizations:
            visualization = torch.cat(visualizations, dim=1)

        return AllTrackerInferenceOutput(
            motion_tracks=merged_tracks,
            visualization=visualization,
        )
