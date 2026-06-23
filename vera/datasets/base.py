"""Base unified dataset: sample frames -> load views separately -> adapt layout.

The two consumer datasets differ ONLY in the final ``layout`` and output dict keys. Everything else
(episode selection, fps-aware frame sampling, per-view loading) is shared, which is the whole point of
the unification.

For the packed (action) datasets (robomimic/mimicgen/iiwa/pusht) this module also derives the
inverse-dynamics target ``du`` and applies action + flow normalization, ported from okto
``datasets/action/robomimic_dataset.py`` and ``datasets/action/loaders/action_loader.py`` — so the
self-contained core emits ``du`` / ``rgb`` / ``flow`` in the SAME keys/shapes/normalization the okto
``RobomimicDataset`` produced, with NO ``import okto``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:  # torch.utils.data is optional at import time for the Phase-0 pure-logic tests
    from torch.utils.data import Dataset as _TorchDataset
except Exception:  # pragma: no cover
    _TorchDataset = object  # type: ignore

from vera.datasets.core.frame_sampler import FrameSamplerConfig, sample_frame_indices
from vera.datasets.core.layout import apply_layout
from vera.datasets.core.sources import Episode, Source
from vera.datasets.core.view_loader import ViewLoader


@dataclass
class DatasetConfig:
    layout: str  # "separate" (IDM) | "tiled" (video model)
    n_frames: int
    height: int
    per_view_w: int
    target_fps: Optional[int] = None
    temporal_stride: Optional[int] = None
    trim_mode: str = "random_cut"
    pad_mode: str = "pad_last"
    pad_views_to: Optional[int] = None
    pad_position: str = "right"
    load_flow: bool = False

    # --- video-model (WAN/OMNI) extras ---
    # Black-pad the post-tile canvas width to this (combined_4env: 576). None = no pad.
    pad_to_width: Optional[int] = None
    # Prepended to every prompt (flow-planner cfg.id_token); usually "".
    id_token: str = ""
    # i2v vs t2v — only affects which keys the contract advertises (the first frame is
    # always videos[:1]); kept for cfg fidelity.
    image_to_video: bool = True

    # --- packed / action (IDM) extras — all driven by the dataset cfg, not hardcoded ---
    # When True, derive ``du`` (SE3 finite-diff twist + gripper) and apply action/flow
    # normalization in __getitem__ (the okto RobomimicDataset behavior).
    derive_action: bool = False
    linearize: int = 1                  # t -> t+linearize transition for du / flow
    overfit_idx: Optional[int] = None   # pin to one episode (validation viz)
    # SE3 finite-diff action raw scales (okto RobomimicLiftSE3DeltaAction defaults).
    se3_scale: float = 50.0
    gripper_scale: float = 80.0
    du_scale: float = 1.0
    # Action normalization (okto RobomimicDatasetCfg / normalization.py).
    action_normalization_mode: str = "none"  # none|symmetric_percentile|minmax|zscore
    action_abs_scale: Optional[Sequence[float]] = None
    action_min: Optional[Sequence[float]] = None
    action_max: Optional[Sequence[float]] = None
    action_mean: Optional[Sequence[float]] = None
    action_std: Optional[Sequence[float]] = None
    action_percentile: Optional[float] = None
    # Flow normalization (okto normalization.py).
    flow_normalization_mode: str = "scale"   # scale|symmetric_percentile|percentile_minmax
    flow_normalization_space: str = "raw_fullres"
    oflow_scale: Optional[float] = None
    oflow_std: Optional[Sequence[float]] = None
    flow_scale_factor: float = 1.0
    oflow_abs_scale: Optional[Sequence[float]] = None
    oflow_percentile: Optional[float] = None
    oflow_percentile_min: Optional[Sequence[float]] = None
    oflow_percentile_max: Optional[Sequence[float]] = None
    eye_in_hand_mask_bottom_px: Optional[int] = None
    # Action model selection (mirrors okto per-dataset action_model choice). Driven by
    # ``action_mode``; default se3_quat preserves the old hardcoded behavior for
    # robomimic/mimicgen. See vera/datasets/core/actions.py::resolve_action_model.
    action_mode: str = "se3_quat"  # se3_quat | droid_se3 | qpos_delta | pusht_pos_cmd
    # Low-dim trajectory keys for the SE3-quat action (general over embodiment).
    action_pos_key: str = "low_dim/robot0_eef_pos"
    action_quat_key: str = "low_dim/robot0_eef_quat"
    action_gripper_key: str = "low_dim/robot0_gripper_qpos"
    # DROID SE3 (euler-6D) trajectory keys + time-aware delta knobs.
    droid_cartesian_key: str = "observation/robot_state/cartesian_position"
    droid_gripper_key: str = "observation/robot_state/gripper_position"
    use_time_aware_delta: bool = False
    time_delta_source: str = "robot_state"
    target_delta_sec: Optional[float] = None
    # Allegro / qpos-delta knobs.
    qpos_key: str = "qpos"
    qpos_indices: Optional[Sequence[int]] = None
    # Shared raw scale (DROID joint/qpos/pusht use action_scale; SE3 uses se3/gripper).
    action_scale: float = 1.0
    action_dim: int = 7
    views: Optional[Sequence[str]] = None
    robot_name: str = "eef_gripper"
    # Length reported for a training dataset (okto: a large fixed number so the
    # sampler draws fresh random windows; index % num_episodes selects the episode).
    train_len: int = 10_000_000


# --------------------------------------------------------------------------
# Action math now lives with the ActionModel registry in core/actions.py
# (de-duplicated). Re-exported here for backward-compat with existing imports.
# --------------------------------------------------------------------------
from vera.datasets.core.actions import (  # noqa: E402
    _matrix_to_twist,
    _pose_to_matrix,
    resolve_action_model,
)


class UnifiedDataset(_TorchDataset):
    """Composes a Source + ViewLoader + FrameSamplerConfig + layout into a torch Dataset."""

    def __init__(
        self,
        source: Source,
        view_loader: ViewLoader,
        cfg: DatasetConfig,
        seed: int = 0,
    ):
        self.source = source
        self.view_loader = view_loader
        self.cfg = cfg
        self._episodes: List[Episode] = source.list_episodes()
        self._rng = np.random.default_rng(seed)
        # Packed sources resolve per-episode metadata lazily (num_frames/views).
        self._is_packed = hasattr(source, "resolve_episode")
        # Per-embodiment action model (built once; mirrors okto's per-dataset
        # self.action_model selection). Only needed for the action/IDM path.
        self.action_model = resolve_action_model(cfg) if cfg.derive_action else None

    def __len__(self) -> int:
        if self.cfg.derive_action and self._is_train():
            return int(self.cfg.train_len)
        return len(self._episodes)

    def _is_train(self) -> bool:
        # The frame sampler is deterministic when rng is None; for action datasets the
        # large train_len + random window is okto's training behavior. We treat any
        # action dataset with no overfit pin as "train" for indexing purposes.
        return self.cfg.derive_action and self.cfg.overfit_idx is None

    # ------------------------------------------------------------------
    # Metadata consumed by the IDM algorithm (denormalization). Mirrors
    # okto RobomimicDataset.get_metadata() exactly.
    # ------------------------------------------------------------------
    def get_metadata(self) -> Dict[str, Any]:
        from vera.datasets.normalization import (
            compute_jacobian_action_scales,
            resolve_effective_oflow_scale,
        )

        c = self.cfg
        oflow_scale = resolve_effective_oflow_scale(
            c.oflow_scale, c.oflow_std, c.flow_scale_factor
        )
        jac_scales = compute_jacobian_action_scales(
            action_dim=c.action_dim,
            du_scale=c.du_scale,
            action_mean=c.action_mean,
            action_std=c.action_std,
            action_min=c.action_min,
            action_max=c.action_max,
            action_abs_scale=c.action_abs_scale,
        )
        out: Dict[str, Any] = {
            "jacobian_action_scales": jac_scales,
            "oflow_scale": oflow_scale,
            "views": list(c.views) if c.views else [],
            "robot_name": c.robot_name,
            "flow_scale_factor": c.flow_scale_factor,
            "du_scale": c.du_scale,
            "action_pre_scale": c.du_scale,
            "action_normalization_mode": c.action_normalization_mode,
            "flow_normalization_mode": c.flow_normalization_mode,
            "flow_normalization_space": c.flow_normalization_space,
        }
        if c.action_normalization_mode == "symmetric_percentile":
            if c.action_percentile is not None:
                out["action_percentile"] = c.action_percentile
            if c.action_abs_scale is not None:
                out["action_abs_scale"] = list(c.action_abs_scale)
        if c.action_mean is not None:
            out["action_mean"] = list(c.action_mean)
        if c.action_std is not None:
            out["action_std"] = list(c.action_std)
        if c.action_min is not None:
            out["action_min"] = list(c.action_min)
        if c.action_max is not None:
            out["action_max"] = list(c.action_max)
        if c.flow_normalization_mode == "scale":
            out["oflow_scale"] = oflow_scale
            if c.oflow_std is not None:
                out["oflow_std"] = list(c.oflow_std)
        if c.flow_normalization_mode == "symmetric_percentile":
            if c.oflow_percentile is not None:
                out["oflow_percentile"] = c.oflow_percentile
            if c.oflow_abs_scale is not None:
                out["oflow_abs_scale"] = list(c.oflow_abs_scale)
        if c.flow_normalization_mode == "percentile_minmax":
            if c.oflow_percentile_min is not None:
                out["oflow_percentile_min"] = list(c.oflow_percentile_min)
            if c.oflow_percentile_max is not None:
                out["oflow_percentile_max"] = list(c.oflow_percentile_max)
        return out

    def _sampler_cfg(self, episode: Episode) -> FrameSamplerConfig:
        return FrameSamplerConfig(
            n_frames=self.cfg.n_frames,
            source_fps=episode.native_fps,
            target_fps=self.cfg.target_fps,
            temporal_stride=self.cfg.temporal_stride,
            trim_mode=self.cfg.trim_mode,
            pad_mode=self.cfg.pad_mode,
        )

    # ------------------------------------------------------------------
    # Action du derivation — the per-embodiment ActionModel computes the raw
    # du (SE3-quat / DROID-SE3 / qpos / pusht); the dataset then applies du_scale
    # + action normalization (mirrors okto RobomimicDataset._getitem_once steps 2-3,
    # where models return raw du and the dataset post-scales/normalizes).
    # ------------------------------------------------------------------
    def _derive_du(self, episode: Episode, t_src: np.ndarray) -> "Any":
        import torch

        c = self.cfg
        ts = np.asarray(t_src, dtype=np.int64)
        du = self.action_model.compute(self.view_loader, episode, ts, c)  # (N, action_dim)
        du = torch.as_tensor(np.asarray(du), dtype=torch.float32)
        # du_scale pre-scale (okto applies it again in __getitem__).
        du = float(c.du_scale) * du
        # Action normalization (okto RobomimicDataset._getitem_once step 3).
        mode = c.action_normalization_mode
        if mode == "symmetric_percentile" and c.action_abs_scale is not None:
            scale = torch.tensor(list(c.action_abs_scale), dtype=du.dtype)
            du = du / (scale + 1e-8)
        elif mode == "minmax" and c.action_min is not None and c.action_max is not None:
            amin = torch.tensor(list(c.action_min), dtype=du.dtype)
            amax = torch.tensor(list(c.action_max), dtype=du.dtype)
            du = (du - amin) / (amax - amin + 1e-8)
            du = du * 2.0 - 1.0
        elif mode == "zscore" and c.action_mean is not None and c.action_std is not None:
            mean = torch.tensor(list(c.action_mean), dtype=du.dtype)
            std = torch.tensor(list(c.action_std), dtype=du.dtype)
            du = (du - mean) / (std + 1e-8)
        return du

    def _normalize_flow(self, flow: "Any") -> "Any":
        import torch

        from vera.datasets.normalization import resolve_effective_oflow_scale

        c = self.cfg

        def _chan_view(values):
            stats = torch.tensor(list(values), dtype=flow.dtype, device=flow.device)
            if flow.ndim == 5:  # (T, V, C, H, W)
                return stats.view(1, 1, -1, 1, 1)
            return stats.view(1, -1, 1, 1)  # (T, C, H, W)

        mode = c.flow_normalization_mode
        if mode == "symmetric_percentile" and c.oflow_abs_scale is not None:
            return flow / (_chan_view(c.oflow_abs_scale) + 1e-8)
        if (
            mode == "percentile_minmax"
            and c.oflow_percentile_min is not None
            and c.oflow_percentile_max is not None
        ):
            fmin = _chan_view(c.oflow_percentile_min)
            fmax = _chan_view(c.oflow_percentile_max)
            flow = (flow - fmin) / (fmax - fmin + 1e-8)
            return flow * 2.0 - 1.0
        scale = resolve_effective_oflow_scale(
            c.oflow_scale, c.oflow_std, c.flow_scale_factor
        )
        return scale * flow

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------
    def _select_episode(self, idx: int) -> Episode:
        if self.cfg.overfit_idx is not None:
            ep = self._episodes[int(self.cfg.overfit_idx) % len(self._episodes)]
        else:
            ep = self._episodes[idx % len(self._episodes)]
        if self._is_packed:
            ep = self.source.resolve_episode(ep)
        return ep

    def _sample_window(self, episode: Episode) -> np.ndarray:
        """okto ``_sample_uniform_window``: contiguous t in [start, start+n).

        Valid range: start + (n-1) + linearize <= num_frames - 1, i.e.
        max_start = num_frames - linearize - 1 - (n - 1). Deterministic (start=0)
        when overfit-pinned, random otherwise (matches okto training).
        """
        c = self.cfg
        n = c.n_frames
        ep_len = int(episode.num_frames)
        max_start = (ep_len - 1) - int(c.linearize) - (n - 1)
        if max_start < 0:
            raise RuntimeError(
                f"Episode too short: ep_len={ep_len}, n_frames={n}, linearize={c.linearize}"
            )
        if c.overfit_idx is not None:
            start = 0
        else:
            start = int(self._rng.integers(0, max_start + 1))
        return np.arange(start, start + n, dtype=np.int64)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.cfg.derive_action:
            return self._getitem_action(idx)
        return self._getitem_video(idx)

    # ---- action / IDM path (packed) ----------------------------------
    def _getitem_action(self, idx: int) -> Dict[str, Any]:
        import torch

        max_tries = 20
        last_exc: Exception | None = None
        for attempt in range(max_tries):
            try:
                use_idx = idx if attempt == 0 else int(self._rng.integers(0, len(self._episodes)))
                episode = self._select_episode(use_idx)
                t_src = self._sample_window(episode)
                rgb_views = self.view_loader.load_rgb(episode, t_src)  # list [T,C,H,W]
                rgb = apply_layout(
                    rgb_views,
                    layout=self.cfg.layout,
                    pad_views_to=self.cfg.pad_views_to,
                    pad_position=self.cfg.pad_position,
                )
                du = self._derive_du(episode, t_src)
                out: Dict[str, Any] = {"rgb": rgb, "du": du}
                if self.cfg.load_flow:
                    flow_views = self.view_loader.load_flow(episode, t_src)
                    if flow_views is not None:
                        flow = apply_layout(
                            flow_views,
                            layout=self.cfg.layout,
                            pad_views_to=self.cfg.pad_views_to,
                            pad_position=self.cfg.pad_position,
                        )
                        out["flow"] = self._normalize_flow(flow)
                return out
            except Exception as e:  # okto retries on failure with a fresh index
                last_exc = e
                continue
        raise RuntimeError(
            f"{type(self).__name__} __getitem__ failed after {max_tries} attempts"
        ) from last_exc

    # ---- video-model path (WAN/OMNI contract) ------------------------
    # Reproduces flow-planner VideoFlowDataset/DroidFlowDataset.__getitem__ for the
    # combined_4env loader: tile per-view rgb along width, black-pad views + canvas to
    # pad_to_width BEFORE normalize, map [0,1]->[-1,1] (img_normalize([0.5]*3,[0.5]*3)),
    # and emit the keys wan_t2v/wan_i2v.prepare_embeds read.
    def _getitem_video(self, idx: int) -> Dict[str, Any]:
        import torch

        from vera.datasets.core.layout import pad_to_width as _pad_canvas

        episode = self._episodes[idx]
        if self._is_packed:
            episode = self.source.resolve_episode(episode)
        # flow-planner samples from the SHORTEST view (min decord frame count) for
        # multi-view raw-video sources (DROID), not the anchor-view CSV n_frames. The
        # view loader exposes this via effective_num_frames(); fall back to
        # episode.num_frames for sources where every view has equal length
        # (mimicgen/allegro one-clip-per-view, where the two agree).
        eff = getattr(self.view_loader, "effective_num_frames", None)
        num_src_frames = int(eff(episode)) if eff is not None else int(episode.num_frames)
        frame_indices = sample_frame_indices(
            num_src_frames, self._sampler_cfg(episode), rng=self._rng
        )
        rgb_views = self.view_loader.load_rgb(episode, frame_indices)  # list [T,3,H,Wv] in [0,1]
        # Tile views along width (+ black-pad missing view slots), then canvas-pad to
        # pad_to_width. Both pads happen in [0,1] so the padded region == 0 -> -1 after
        # normalize (same as flow-planner's pad-before-img_normalize order).
        rgb = apply_layout(
            rgb_views,
            layout=self.cfg.layout,  # "tiled" for the video model
            pad_views_to=self.cfg.pad_views_to,
            pad_position=self.cfg.pad_position,
        )  # [T, 3, H, W_tile] in [0,1]
        rgb = _pad_canvas(rgb, self.cfg.pad_to_width, pad_position=self.cfg.pad_position)

        # Separate-layout video cfg (RGB-only IDM-style consumer, no WAN canvas): emit
        # the legacy [0,1] rgb + frame keys and skip the [-1,1]/bbox/prompt contract,
        # which only applies to the tiled WAN canvas. Keeps the layout-parity test green.
        if self.cfg.layout != "tiled":
            out_sep: Dict[str, Any] = {
                "rgb": rgb,
                "frame_indices": frame_indices,
                "video_frame_indices": torch.as_tensor(frame_indices, dtype=torch.long),
                "episode_id": episode.episode_id,
            }
            if self.cfg.load_flow:
                flow_views = self.view_loader.load_flow(episode, frame_indices)
                if flow_views is not None:
                    out_sep["flow"] = apply_layout(
                        flow_views, layout=self.cfg.layout,
                        pad_views_to=self.cfg.pad_views_to, pad_position=self.cfg.pad_position,
                    )
            return out_sep

        videos = rgb.mul(2.0).sub(1.0)  # [0,1] -> [-1,1] (img_normalize mean/std 0.5)

        T, _, H, W = videos.shape
        meta = episode.meta or {}
        caption = str(meta.get("caption", ""))
        prompts = (self.cfg.id_token or "") + caption
        # bbox: combined_4env carries no bbox metadata, so zeros (flow-planner
        # video_base._render_bbox returns zeros when the bbox cols are commented out).
        # bbox_render is canvas-width so default_collate stacks across the batch.
        bbox_render = torch.zeros(2, H, W, dtype=videos.dtype)
        has_bbox = torch.zeros(2, dtype=torch.bool)

        out: Dict[str, Any] = {
            "videos": videos,
            "video_metadata": {"num_frames": int(T), "height": int(H), "width": int(W)},
            "bbox_render": bbox_render,
            "has_bbox": has_bbox,
            "src_n_frames": int(episode.num_frames),
            "video_frame_indices": torch.as_tensor(frame_indices, dtype=torch.long),
            "prompts": prompts,
            "task_class": str(meta.get("task_class", "")),
            "video_path": str(meta.get("video_path", episode.episode_id)),
            # legacy keys kept for the layout-parity test / IDM-style consumers.
            "rgb": rgb,
            "frame_indices": frame_indices,
            "episode_id": episode.episode_id,
        }
        if self.cfg.load_flow:
            flow_views = self.view_loader.load_flow(episode, frame_indices)
            if flow_views is not None:
                flow = apply_layout(
                    flow_views,
                    layout=self.cfg.layout,
                    pad_views_to=self.cfg.pad_views_to,
                    pad_position=self.cfg.pad_position,
                )
                out["optical_flow"] = _pad_canvas(
                    flow, self.cfg.pad_to_width, pad_position=self.cfg.pad_position
                )
        return out
