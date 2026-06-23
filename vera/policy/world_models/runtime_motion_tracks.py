from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


def xy_to_idx_tensor(xy: torch.Tensor, height: int, width: int) -> torch.Tensor:
    xy_round = torch.round(xy).long()
    x = xy_round[..., 0].clamp(0, width - 1)
    y = xy_round[..., 1].clamp(0, height - 1)
    return y * width + x


@dataclass
class RuntimeMotionTracks:
    """Backend-neutral sparse motion tracks for policy inference."""

    xy_src: torch.Tensor
    xy_tgt: torch.Tensor
    vis_src: torch.Tensor
    vis_tgt: torch.Tensor
    idx_src: torch.Tensor
    idx_tgt: torch.Tensor
    image_size: tuple[int, int]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def disp(self) -> torch.Tensor:
        return (self.xy_tgt - self.xy_src).float()

    @property
    def valid(self) -> torch.Tensor:
        return ((self.vis_src > 0.5) & (self.vis_tgt > 0.5)).float()

    def slice_time(self, start: int = 0, end: int | None = None) -> "RuntimeMotionTracks":
        return RuntimeMotionTracks(
            xy_src=self.xy_src[:, start:end],
            xy_tgt=self.xy_tgt[:, start:end],
            vis_src=self.vis_src[:, start:end],
            vis_tgt=self.vis_tgt[:, start:end],
            idx_src=self.idx_src[:, start:end],
            idx_tgt=self.idx_tgt[:, start:end],
            image_size=self.image_size,
            meta=dict(self.meta),
        )

    def as_policy_dict(self) -> dict[str, Any]:
        return {
            "xy_src": self.xy_src,
            "xy_tgt": self.xy_tgt,
            "disp": self.disp,
            "valid": self.valid,
            "idx_src": self.idx_src,
            "idx_tgt": self.idx_tgt,
            "image_size": self.image_size,
            "meta": dict(self.meta),
        }

    def with_x_offset(
        self,
        x_offset: int,
        image_size: tuple[int, int],
        *,
        view_key: str | None = None,
        view_index: int | None = None,
    ) -> "RuntimeMotionTracks":
        height, width = image_size
        xy_src = self.xy_src.clone()
        xy_tgt = self.xy_tgt.clone()
        xy_src[..., 0] += float(x_offset)
        xy_tgt[..., 0] += float(x_offset)
        meta = dict(self.meta)
        if view_key is not None:
            meta["view_key"] = view_key
        if view_index is not None:
            meta["view_index"] = int(view_index)
        meta["x_offset"] = int(x_offset)
        return RuntimeMotionTracks(
            xy_src=xy_src,
            xy_tgt=xy_tgt,
            vis_src=self.vis_src.clone(),
            vis_tgt=self.vis_tgt.clone(),
            idx_src=xy_to_idx_tensor(xy_src, height, width),
            idx_tgt=xy_to_idx_tensor(xy_tgt, height, width),
            image_size=image_size,
            meta=meta,
        )


@dataclass
class TrackerInferenceOutput:
    motion_tracks: RuntimeMotionTracks
    visualization: torch.Tensor | None = None
    per_view_outputs: list["TrackerInferenceOutput"] | None = None


def merge_runtime_tracks_across_views(
    tracks: list[RuntimeMotionTracks],
    *,
    view_widths: list[int],
    view_keys: list[str] | None = None,
) -> RuntimeMotionTracks:
    if not tracks:
        raise ValueError("Expected at least one RuntimeMotionTracks to merge")
    if len(tracks) != len(view_widths):
        raise ValueError(
            f"tracks/view_widths mismatch: {len(tracks)} vs {len(view_widths)}"
        )
    if view_keys is not None and len(view_keys) != len(tracks):
        raise ValueError(
            f"tracks/view_keys mismatch: {len(tracks)} vs {len(view_keys)}"
        )

    height = int(tracks[0].image_size[0])
    total_width = int(sum(int(width) for width in view_widths))
    adjusted_tracks: list[RuntimeMotionTracks] = []
    x_offset = 0
    for view_index, track in enumerate(tracks):
        if int(track.image_size[0]) != height:
            raise ValueError(
                "All per-view track tensors must share the same height for stitching"
            )
        expected_width = int(view_widths[view_index])
        if int(track.image_size[1]) != expected_width:
            raise ValueError(
                f"Per-view track width mismatch at index {view_index}: "
                f"expected {expected_width}, got {int(track.image_size[1])}"
            )
        adjusted_tracks.append(
            track.with_x_offset(
                x_offset,
                (height, total_width),
                view_key=None if view_keys is None else view_keys[view_index],
                view_index=view_index,
            )
        )
        x_offset += expected_width

    meta = dict(adjusted_tracks[0].meta)
    meta["view_widths"] = [int(width) for width in view_widths]
    if view_keys is not None:
        meta["view_keys"] = list(view_keys)
    meta["per_view_track_counts"] = [
        int(track.xy_src.shape[2]) for track in adjusted_tracks
    ]
    return RuntimeMotionTracks(
        xy_src=torch.cat([track.xy_src for track in adjusted_tracks], dim=2),
        xy_tgt=torch.cat([track.xy_tgt for track in adjusted_tracks], dim=2),
        vis_src=torch.cat([track.vis_src for track in adjusted_tracks], dim=2),
        vis_tgt=torch.cat([track.vis_tgt for track in adjusted_tracks], dim=2),
        idx_src=torch.cat([track.idx_src for track in adjusted_tracks], dim=2),
        idx_tgt=torch.cat([track.idx_tgt for track in adjusted_tracks], dim=2),
        image_size=(height, total_width),
        meta=meta,
    )


def stitch_view_visualizations(visualizations: list[torch.Tensor]) -> torch.Tensor | None:
    if not visualizations:
        return None
    return torch.cat(visualizations, dim=3)
