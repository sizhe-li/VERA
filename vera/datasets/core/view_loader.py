"""Per-view frame loading -> a list of lossless ``[T, C, H, W]`` tensors (one per view).

This is the single place that touches raw pixels. It loads EVERY view separately (never tiled),
resizing/center-cropping each to a common ``(H, per_view_w)``. Downstream ``layout.apply_layout``
then either stacks (separate) or concatenates (tiled). Keeping load and layout separate is what makes
the two stages provably consistent.

Phase 0 ships:
  * ``uniform_resize_center_crop`` — the pure geometry op (port of the helper shared by
    okto ``loaders/rgb_loader`` and flow-planner ``datasets/utils/img_utils``), unit-testable.
  * ``ViewLoader`` — the abstract contract.
  * ``DroidViewLoader`` — DROID skeleton; the decord/packed-NPZ decode body is ported in Phase 3 from
    ``third_party/flow-planner/datasets/droid_flow.py::_load_video_concat`` (per-view decord reads) and
    ``project/okto/datasets/action/loaders/rgb_loader.py`` (separate-view stack + packed reads).
"""

from __future__ import annotations

import abc
from typing import List, Sequence

import numpy as np
import torch


def uniform_resize_center_crop(
    frames: torch.Tensor, height: int, width: int, rounding: str = "ceil"
) -> torch.Tensor:
    """Resize ``[T, C, H, W]`` to cover ``(height, width)`` then center-crop to exactly that size.

    Pure geometry; no I/O. BYTE-FOR-BYTE port of flow-planner
    ``datasets/utils/img_utils.py::uniform_resize_center_crop`` so per-view pixels match
    the video model's training distribution: ``scale = max(H/h, W/w)``, integer new size by
    ``rounding`` (``ceil``/``round``/``floor``), then ``F.interpolate(mode="bilinear",
    align_corners=False)`` (NO antialias — matching flow-planner) and a center crop. The
    early-out for an already-correct size also mirrors flow-planner (returns unchanged).
    """
    import math

    if frames.ndim != 4:
        raise ValueError(f"expected [T, C, H, W], got {tuple(frames.shape)}")
    _, _, h, w = frames.shape
    if h == height and w == width:
        return frames
    scale = max(float(height) / float(h), float(width) / float(w))
    if rounding == "ceil":
        new_h, new_w = math.ceil(h * scale), math.ceil(w * scale)
    elif rounding == "round":
        new_h, new_w = round(h * scale), round(w * scale)
    elif rounding == "floor":
        new_h, new_w = int(h * scale), int(w * scale)
    else:
        raise ValueError(f"Unsupported rounding mode: {rounding}")
    if new_h < height or new_w < width:
        raise RuntimeError(
            f"Resized {(new_h, new_w)} does not cover target {(height, width)}"
        )
    frames = torch.nn.functional.interpolate(
        frames, size=(new_h, new_w), mode="bilinear", align_corners=False
    )
    top = (new_h - height) // 2
    left = (new_w - width) // 2
    return frames[:, :, top : top + height, left : left + width].contiguous()


class ViewLoader(abc.ABC):
    """Loads a single episode's frames as a list of per-view ``[T, C, H, W]`` tensors.

    ``height`` and ``per_view_w`` are the per-view target size (NOT the tiled canvas width). The tiled
    canvas width is ``per_view_w * n_view_slots`` and is assembled by ``layout``, not here.
    """

    def __init__(self, height: int, per_view_w: int):
        self.height = int(height)
        self.per_view_w = int(per_view_w)

    @abc.abstractmethod
    def load_rgb(self, episode, frame_indices: np.ndarray) -> List[torch.Tensor]:
        """Return one ``[T, C, H, per_view_w]`` float tensor per view, in canonical view order."""

    def load_flow(self, episode, frame_indices: np.ndarray) -> List[torch.Tensor] | None:
        """Optional per-view optical flow, one ``[T, 2, H, per_view_w]`` tensor per view."""
        return None


class PackedViewLoader(ViewLoader):
    """Per-view loader for okto's packed-NPZ format (JPEG RGB + qint8 flow).

    Ports the decode + resize policy from okto so per-view pixels/flow match:
      * ``load_rgb``  <- okto ``rgb_loader.load_rgb_frames_from_packed_npz`` + the
        ``EpisodeAssembler`` resize (``center_crop_square=False`` -> plain bilinear
        ``interpolate`` to the cfg ``image_size``, NOT center-crop).
      * ``load_flow`` <- okto ``optical_flow_loader._load_packed_optical_flow_impl``
        (qint8 decode -> ``(2,H,W)``), the ``eye_in_hand`` bottom-N-px mask applied
        in NATIVE resolution, then ``geometry_utils.resize_flow`` to the cfg size.
      * ``load_trajectory`` <- okto ``action_loader._load_packed_dataset`` (low-dim
        arrays for the SE3-finite-diff action derivation).

    Both RGB and flow are returned in canonical (cfg) view order, one ``[T, C, H, W]``
    tensor per view, so ``layout.apply_layout`` stacks them into ``[T, V, C, H, W]``
    (IDM) or tiles them. ``eye_in_hand_mask_bottom_px`` is read from the cfg (None =
    no masking) so this stays general across embodiments.
    """

    def __init__(
        self,
        height: int,
        per_view_w: int,
        *,
        eye_in_hand_mask_bottom_px: int | None = None,
        pusht_zarr_cfg=None,
    ):
        super().__init__(height, per_view_w)
        self.eye_in_hand_mask_bottom_px = eye_in_hand_mask_bottom_px
        # PushT actions live in a zarr (not the packed NPZ). When the dataset is
        # PushT, the registry passes the dataset cfg here so the action model can
        # read actions/state via load_pusht_zarr(). None for every other embodiment.
        self._pusht_zarr_cfg = pusht_zarr_cfg
        self._pusht_zarr = None

    def load_pusht_zarr(self):
        """Lazily open + cache the PushT replay zarr (provider with action/state/
        episode_ends/action_abs_max/q_min/q_max). Consumed by
        ``PushTPosCmdDeltaAction.compute``; raises a clear error if no zarr cfg was
        wired (so the action model never silently emits wrong du)."""
        if self._pusht_zarr_cfg is None:
            raise NotImplementedError(
                "PushTPosCmdDeltaAction selected but no pusht_zarr_cfg was wired into "
                "PackedViewLoader. Set cfg.pusht_zarr_root and route name='pusht' "
                "through registry.build_dataset."
            )
        if self._pusht_zarr is None:
            from vera.datasets.core.pusht_zarr import build_pusht_zarr

            self._pusht_zarr = build_pusht_zarr(self._pusht_zarr_cfg)
        return self._pusht_zarr

    def _packed_summary(self, episode):
        packed = episode.paths.get("packed") if isinstance(episode.paths, dict) else None
        if packed is None:
            raise RuntimeError(
                "PackedViewLoader requires a resolved episode "
                "(call PackedSource.resolve_episode first)."
            )
        return packed

    def load_rgb(self, episode, frame_indices: np.ndarray):  # noqa: D401
        from vera.datasets.core.packed import decode_packed_rgb_frame, open_packed_npz

        idx = np.asarray(frame_indices).astype(np.int64)
        npz = open_packed_npz(episode.paths["packed_npz"])
        out: List[torch.Tensor] = []
        for view in episode.views:
            frames = []
            for t in idx.tolist():
                arr = decode_packed_rgb_frame(npz, int(t), view)  # (H, W, 3) uint8
                frames.append(torch.from_numpy(arr).permute(2, 0, 1))  # (3, H, W)
            rgb = torch.stack(frames, dim=0).float().div_(255.0)  # (T, 3, H0, W0)
            # okto EpisodeAssembler: center_crop_square=False -> plain bilinear resize.
            if rgb.shape[-2:] != (self.height, self.per_view_w):
                rgb = torch.nn.functional.interpolate(
                    rgb,
                    size=(self.height, self.per_view_w),
                    mode="bilinear",
                    align_corners=False,
                )
            out.append(rgb.contiguous())
        return out

    def load_flow(self, episode, frame_indices: np.ndarray):  # noqa: D401
        from vera.utils import geometry_utils
        from vera.datasets.core.packed import decode_packed_flow_frame, open_packed_npz

        packed = self._packed_summary(episode)
        flow_entries = packed.get("flow_entries", {})
        if not any(flow_entries.get(v) for v in episode.views):
            return None  # no flow packed for this episode

        idx = np.asarray(frame_indices).astype(np.int64)
        npz = open_packed_npz(episode.paths["packed_npz"])
        out: List[torch.Tensor] = []
        ref: torch.Tensor | None = None
        flows_raw: List[torch.Tensor | None] = []
        for view in episode.views:
            entry = flow_entries.get(view) or {}
            if not entry:
                flows_raw.append(None)
                continue
            codec = str(entry.get("codec", "qint8_zstd_npz"))
            frames = []
            for t in idx.tolist():
                arr = decode_packed_flow_frame(npz, int(t), view, codec)  # (2,H,W)|(H,W,2)
                frames.append(np.asarray(arr, dtype=np.float32))
            flow_np = np.stack(frames, axis=0)  # (T, 2, H0, W0) | (T, H0, W0, 2)
            flow = torch.from_numpy(flow_np).float()
            if flow.shape[-1] == 2:  # (T, H, W, 2) -> (T, 2, H, W)
                flow = flow.permute(0, 3, 1, 2)
            # eye_in_hand mask: zero bottom-N rows in NATIVE resolution (okto order).
            if (
                self.eye_in_hand_mask_bottom_px is not None
                and "eye_in_hand" in view
            ):
                flow[:, :, -int(self.eye_in_hand_mask_bottom_px) :, :] = 0
            if flow.shape[-2:] != (self.height, self.per_view_w):
                flow = geometry_utils.resize_flow(flow, self.height, self.per_view_w)
            flow = flow.contiguous()
            ref = flow if ref is None else ref
            flows_raw.append(flow)
        # Replace missing views with zeros matching a valid view (okto OpticalFlowLoader).
        if ref is None:
            return None
        for f in flows_raw:
            out.append(f if f is not None else torch.zeros_like(ref))
        return out

    def load_trajectory(self, episode, key: str, row_indices: np.ndarray | None = None):
        """Load a low-dim trajectory array (e.g. ``low_dim/robot0_eef_pos``).

        Ported from okto ``action_loader._load_packed_dataset``; used by the action
        ``du`` derivation in ``UnifiedDataset.__getitem__``.
        """
        from vera.datasets.core.packed import load_packed_array

        return load_packed_array(episode.paths["packed_npz"], key, row_indices)


class DroidViewLoader(ViewLoader):
    """DROID per-view loader (ext1/ext2/wrist).

    Phase 0 = contract + view ordering. Phase 3 ports the body from
    ``flow-planner/datasets/droid_flow.py::_load_video_concat`` (decord per-view reads, normalize,
    ``uniform_resize_center_crop``) and ``okto/.../rgb_loader.py`` (packed-NPZ path).
    """

    CANONICAL_VIEWS: Sequence[str] = ("ext1", "ext2", "wrist")

    def effective_num_frames(self, episode) -> int:
        """Min decord frame count across the episode's views (flow-planner
        ``_load_video_concat``'s ``min_n_frames``).

        flow-planner samples frame indices from the SHORTEST view so no view reads
        out of bounds; the CSV ``n_frames`` is the anchor view only and can exceed a
        shorter sibling view. The dataset (``_getitem_video``) calls this to size the
        sampling window so the sampled indices match flow-planner byte-for-byte. Pure
        metadata read (len(VideoReader)), no pixel decode. Cached per-episode."""
        import decord

        cached = episode.paths.get("_min_view_frames") if isinstance(episode.paths, dict) else None
        if cached is not None:
            return int(cached)
        videos = episode.paths["videos"]
        view_order = list(episode.views) if episode.views else list(self.CANONICAL_VIEWS)
        min_n = None
        for view in view_order:
            n = len(decord.VideoReader(videos[view]))
            min_n = n if min_n is None else min(min_n, n)
        min_n = int(min_n if min_n is not None else episode.num_frames)
        if isinstance(episode.paths, dict):
            episode.paths["_min_view_frames"] = min_n
        return min_n

    def load_rgb(self, episode, frame_indices: np.ndarray) -> List[torch.Tensor]:  # noqa: D401
        """Decode the sampled frames for each view → ``[T, 3, H, per_view_w]`` in [0,1].

        Mirrors flow-planner droid_flow._load_video_concat: per-view decord read of the
        given frame indices, uint8→float[0,1], then ``uniform_resize_center_crop`` to the
        per-view target size. Views are returned in canonical order; ``layout`` tiles/stacks.
        """
        import decord

        decord.bridge.set_bridge("torch")
        idx = np.asarray(frame_indices).astype(np.int64)
        videos = episode.paths["videos"]
        # iterate the episode's own views (canonical order) so one loader serves any
        # embodiment (DROID 3-view, mimicgen/allegro N-view) — not just ext1/ext2/wrist.
        view_order = list(episode.views) if episode.views else list(self.CANONICAL_VIEWS)
        out: List[torch.Tensor] = []
        for view in view_order:
            vr = decord.VideoReader(videos[view])
            clamped = np.clip(idx, 0, len(vr) - 1)
            frames = vr.get_batch(clamped.tolist())            # [T, H, W, C] uint8
            frames = frames.permute(0, 3, 1, 2).float() / 255.0  # [T, C, H, W] in [0,1]
            frames = uniform_resize_center_crop(frames, self.height, self.per_view_w)
            out.append(frames.contiguous())
        return out
