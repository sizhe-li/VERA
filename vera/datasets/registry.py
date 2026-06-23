"""Unified dataset registry — the Phase-3 *unify* seam.

Replaces ``okto.datasets.registry``. Where okto registered many concrete dataset
classes keyed by name and instantiated ``cls(cfg, stage=stage)``, vera has ONE
dataset system (``vera.datasets.core`` + :class:`IDMDataset` / :class:`VideoModelDataset`).
``build_dataset`` is the single factory that bridges a Hydra dataset cfg node onto
that composed core: it picks a :class:`Source` / :class:`ViewLoader` by ``cfg.name``,
maps the cfg fields onto a :class:`DatasetConfig`, and selects the dataset class by
layout (``"tiled"`` -> video model, else separate-view IDM).

To support a new embodiment, add its ``Source`` + ``ViewLoader`` in
``vera/datasets/core`` and register it in ``_SOURCES`` below — no per-dataset class.
"""
from __future__ import annotations

from typing import Any, Dict, Literal, Tuple, Type

from vera.datasets.base import DatasetConfig, UnifiedDataset
from vera.datasets.idm_dataset import IDMDataset
from vera.datasets.video_dataset import VideoModelDataset
from vera.datasets.core.sources import (
    Source,
    DroidSource,
    AllegroSimSource,
    AllegroRealSource,
    MimicgenSource,
    PackedSource,
)
from vera.datasets.core.view_loader import ViewLoader, DroidViewLoader, PackedViewLoader

# Optional keyed registry kept for API-compat with code that imported
# ``register_dataset`` / ``DATASET_REGISTRY`` from okto. The unified path does not
# need it, but exposing it avoids breaking incidental imports.
try:
    from vera.utils.registry import Registry  # ported in Phase 2
    DATASET_REGISTRY = Registry("dataset")  # type: ignore[call-arg]
    register_dataset = DATASET_REGISTRY.register
except Exception:  # pragma: no cover - registry is optional for the unified path
    DATASET_REGISTRY = None

    def register_dataset(*_a, **_k):  # type: ignore[misc]
        def _wrap(x):
            return x
        return _wrap

# name -> (Source class, ViewLoader class). The DroidViewLoader decodes any episode's
# views (it iterates episode.views), so all video-CSV embodiments share it; only the
# discovery (Source) differs (wide DROID CSV vs long allegro/mimicgen CSV).
# pusht uses a packed/zarr format (not video-CSV) — add a PushtSource in core/ to enable.
_SOURCES: Dict[str, Tuple[Type[Source], Type[ViewLoader]]] = {
    "droid": (DroidSource, DroidViewLoader),
    "droid_flow": (DroidSource, DroidViewLoader),
    "allegro_sim": (AllegroSimSource, DroidViewLoader),
    "allegro_real": (AllegroRealSource, DroidViewLoader),
    "mimicgen": (MimicgenSource, DroidViewLoader),
}


def resolve_dataset_cfg(cfg: Any) -> Any:
    """Normalize a dataset cfg node.

    okto resolved a Hydra node into a typed dataclass here; the unified path reads
    the node directly (OmegaConf attribute access), so this is an identity hook kept
    for call-site compatibility and future typed resolution.
    """
    return cfg


def _get(cfg: Any, key: str, default: Any = None) -> Any:
    """Tolerant attr/item access over OmegaConf / dataclass / dict cfg nodes."""
    val = None
    try:
        val = getattr(cfg, key)
    except Exception:
        try:
            val = cfg[key]
        except Exception:
            val = None
    return default if val is None else val


# Packed (action / IDM) datasets — JPEG RGB + qint8_zstd flow + SE3-finite-diff du,
# stored as one .npz per episode (okto_packed format). These are now served by the
# SELF-CONTAINED unified core (PackedSource + PackedViewLoader + UnifiedDataset action
# path) with NO ``import okto`` — the okto bridge was removed once every embodiment
# reached byte-for-byte parity with the legacy loaders (du vs okto, videos vs flow-planner).
_PACKED_NAMES = {"robomimic", "mimicgen", "iiwa", "pusht", "drake_allegro", "droid"}

# Default action_mode per packed dataset name (mirrors okto's per-dataset
# self.action_model selection). A cfg may override via an explicit ``action_mode``.
_DEFAULT_ACTION_MODE = {
    "robomimic": "se3_quat",
    "mimicgen": "se3_quat",
    "iiwa": "se3_quat",
    "drake_allegro": "qpos_delta",
    "droid": "droid_se3",
    "pusht": "pusht_pos_cmd",
}

def _cfg_keys(cfg: Any):
    """Best-effort key listing over OmegaConf / dict / dataclass cfg nodes."""
    try:
        if hasattr(cfg, "keys"):
            return list(cfg.keys())
    except Exception:
        pass
    try:
        return list(vars(cfg).keys())
    except Exception:
        return []


def _is_mixture_cfg(cfg: Any) -> bool:
    """A mixture cfg carries one or more ``subset/<name>`` child nodes (flow-planner
    combined_4env layout). MixtureDataset reads ``subset/*`` + per-split ``weight``.

    Detected structurally so a mixture is routed regardless of its ``name`` and never
    collides with a concrete embodiment Source. A cfg with no ``subset/*`` key is a
    normal single-embodiment cfg and falls through to the packed/video routes."""
    return any(str(k).startswith("subset/") for k in _cfg_keys(cfg))


def _getsub(cfg: Any, *path: str, default: Any = None) -> Any:
    """Nested tolerant access, e.g. ``_getsub(cfg, 'sampling', 'num_frames')``."""
    node = cfg
    for key in path:
        node = _get(node, key, None)
        if node is None:
            return default
    return node


def _as_list(val: Any) -> Any:
    """OmegaConf ListConfig / tuple -> plain python list (or None)."""
    if val is None:
        return None
    try:
        return [float(x) for x in val]
    except (TypeError, ValueError):
        try:
            return list(val)
        except TypeError:
            return val


def _build_packed_core_dataset(
    cfg: Any, *, stage: str
) -> UnifiedDataset:
    """Self-contained packed (IDM) dataset — no okto import.

    Maps an okto-style packed dataset cfg node (robomimic/mimicgen/iiwa/pusht) onto
    PackedSource + PackedViewLoader + UnifiedDataset's action path, deriving ``du``
    (SE3 finite-diff) and applying action/flow normalization. All numeric specifics
    (views, image size, normalization scales) come from the cfg, not hardcoded.
    """
    name = str(_get(cfg, "name") or _get(cfg, "_name"))
    views = _getsub(cfg, "camera", "views")
    views = list(views) if views is not None else None
    image_size = _getsub(cfg, "camera", "image_size")
    if image_size is not None:
        height = int(image_size[0])
        per_view_w = int(image_size[1])
    else:
        res = int(_get(cfg, "resolution", 128))
        height = per_view_w = res

    num_frames = int(_getsub(cfg, "sampling", "num_frames", default=_get(cfg, "max_frames", 8)))
    linearize = int(_getsub(cfg, "sampling", "linearize_time_length", default=1))
    eih_mask = _get(cfg, "eye_in_hand_mask_bottom_px")
    eih_mask = int(eih_mask) if eih_mask is not None else None

    # Layout: IDM => separate. (Packed video-tiled is not a current consumer.)
    layout = str(_get(cfg, "layout", "separate"))

    # Resolve action_mode: explicit cfg field wins; else the per-name default.
    # okto's droid cfg uses {joint, joint_delta, se3_delta}; map se3_delta -> droid_se3
    # (joint/joint_delta are not ported in this reduced registry — they raise clearly).
    raw_action_mode = _get(cfg, "action_mode")
    if raw_action_mode is not None:
        _droid_map = {"se3_delta": "droid_se3", "se3_quat": "se3_quat",
                      "qpos_delta": "qpos_delta", "pusht_pos_cmd": "pusht_pos_cmd",
                      "droid_se3": "droid_se3"}
        action_mode = _droid_map.get(str(raw_action_mode), str(raw_action_mode))
    else:
        action_mode = _DEFAULT_ACTION_MODE.get(name, "se3_quat")

    qpos_indices = _get(cfg, "qpos_indices")
    qpos_indices = [int(x) for x in qpos_indices] if qpos_indices is not None else None

    dcfg = DatasetConfig(
        layout=layout,
        n_frames=num_frames,
        height=height,
        per_view_w=per_view_w,
        load_flow=True,
        derive_action=True,
        linearize=linearize,
        overfit_idx=_get(cfg, "overfit_idx"),
        du_scale=float(_get(cfg, "du_scale", 1.0)),
        action_normalization_mode=str(_get(cfg, "action_normalization_mode", "none")),
        action_abs_scale=_as_list(_get(cfg, "action_abs_scale")),
        action_min=_as_list(_get(cfg, "action_min")),
        action_max=_as_list(_get(cfg, "action_max")),
        action_mean=_as_list(_get(cfg, "action_mean")),
        action_std=_as_list(_get(cfg, "action_std")),
        action_percentile=_get(cfg, "action_percentile"),
        flow_normalization_mode=str(_get(cfg, "flow_normalization_mode", "scale")),
        flow_normalization_space=str(_get(cfg, "flow_normalization_space", "raw_fullres")),
        oflow_scale=_get(cfg, "oflow_scale"),
        oflow_std=_as_list(_get(cfg, "oflow_std")),
        flow_scale_factor=float(_get(cfg, "flow_scale_factor", 1.0)),
        oflow_abs_scale=_as_list(_get(cfg, "oflow_abs_scale")),
        oflow_percentile=_get(cfg, "oflow_percentile"),
        oflow_percentile_min=_as_list(_get(cfg, "oflow_percentile_min")),
        oflow_percentile_max=_as_list(_get(cfg, "oflow_percentile_max")),
        eye_in_hand_mask_bottom_px=eih_mask,
        views=views,
        robot_name=str(_get(cfg, "robot_name", "eef_gripper")),
        # --- per-embodiment action model selection + keys/scales ---
        action_mode=action_mode,
        se3_scale=float(_get(cfg, "se3_scale", 50.0)),
        gripper_scale=float(_get(cfg, "gripper_scale", 80.0)),
        action_scale=float(_get(cfg, "action_scale", 1.0)),
        droid_cartesian_key=str(
            _get(cfg, "droid_cartesian_key",
                 "observation/robot_state/cartesian_position")
        ),
        droid_gripper_key=str(
            _get(cfg, "droid_gripper_key", "observation/robot_state/gripper_position")
        ),
        use_time_aware_delta=bool(_get(cfg, "use_time_aware_delta", False)),
        time_delta_source=str(_get(cfg, "time_delta_source", "robot_state")),
        target_delta_sec=_get(cfg, "target_delta_sec"),
        qpos_key=str(_get(cfg, "qpos_key", "qpos")),
        qpos_indices=qpos_indices,
    )

    data_root = _get(cfg, "data_root", _get(cfg, "root_dir", _get(cfg, "root", ".")))
    source = PackedSource(data_root, name=name, views=views)
    # PushT actions live in a zarr replay buffer (not the packed NPZ). When the
    # action model is the PushT pos-cmd delta, pass a small zarr cfg to the loader so
    # load_pusht_zarr() can open it. The zarr provider reads pusht_zarr_root +
    # state_q_min/state_q_max + qpos_indices from this namespace.
    pusht_zarr_cfg = None
    if action_mode == "pusht_pos_cmd":
        from types import SimpleNamespace

        pusht_zarr_cfg = SimpleNamespace(
            pusht_zarr_root=_get(cfg, "pusht_zarr_root"),
            qpos_indices=qpos_indices or [0, 1],
            state_q_min=_as_list(_get(cfg, "state_q_min")),
            state_q_max=_as_list(_get(cfg, "state_q_max")),
        )
    view_loader = PackedViewLoader(
        height=height,
        per_view_w=per_view_w,
        eye_in_hand_mask_bottom_px=eih_mask,
        pusht_zarr_cfg=pusht_zarr_cfg,
    )
    dataset_cls: Type[UnifiedDataset] = (
        VideoModelDataset if layout == "tiled" else IDMDataset
    )
    return dataset_cls(source, view_loader, dcfg, seed=int(_get(cfg, "seed", 0)))


def build_dataset(
    cfg: Any,
    *,
    stage: Literal["training", "validation", "test", "train", "val", "test"],
) -> UnifiedDataset:
    """Bridge a dataset cfg node onto the unified core datasets.

    stage is accepted for API-compat (okto used it for split-specific behavior);
    split handling now lives in the cfg / frame sampler, so it is currently advisory.
    """
    import os

    cfg = resolve_dataset_cfg(cfg)
    name = _get(cfg, "name") or _get(cfg, "_name")

    # Mixture route (combined_4env / any cfg carrying ``subset/*`` nodes + per-split
    # ``weight``). Dispatch to MixtureDataset so the WAN data-module shim is optional:
    # build_dataset(combined_4env_cfg, stage="training") -> a fault-tolerant weighted
    # mixture over the per-subset VideoModelDatasets. Detected structurally (a node
    # with ``subset/<name>`` keys + a ``training``/``validation`` weight block), so it
    # never shadows a concrete embodiment name.
    if _is_mixture_cfg(cfg):
        from vera.datasets.mixture import MixtureDataset

        # stage aliases -> the split key the mixture cfg uses.
        split = "validation" if str(stage) in ("validation", "val", "test") else "training"
        return MixtureDataset(cfg, split=split)  # type: ignore[return-value]

    # Layout-first routing: a ``tiled`` cfg is always the video-model (WAN/OMNI)
    # raw-video path, even for a name (mimicgen/droid) that ALSO has a packed/IDM
    # variant. The packed path below is the ``separate``-layout default for those
    # names. This lets the same embodiment name serve both consumers by cfg.layout.
    layout_hint = str(_get(cfg, "layout", "")).lower()
    is_video_cfg = layout_hint == "tiled" and name in _SOURCES

    # Packed (IDM) datasets: the self-contained core path (no okto).
    if name in _PACKED_NAMES and not is_video_cfg:
        return _build_packed_core_dataset(cfg, stage=str(stage))

    if name not in _SOURCES:
        raise NotImplementedError(
            f"vera.datasets has no unified core Source/ViewLoader for dataset "
            f"'{name}'. Available: {sorted(_SOURCES)}. Add one in vera/datasets/core "
            f"and register it in registry._SOURCES (Phase-3 continuation)."
        )
    source_cls, view_loader_cls = _SOURCES[name]

    layout = str(_get(cfg, "layout", "tiled"))
    height = int(_get(cfg, "height", 128))
    # Video model: per_view_w = canvas_width // n_view_slots (flow-planner
    # _load_video_concat: per_view_w = self.width // (pad_views_to or len(concat_views))).
    # The OUTER `width` may be the padded canvas (e.g. 576) so it must NOT be used as
    # per_view_w directly; use the subset's native `width` and its view count.
    concat_views = _get(cfg, "concat_views")
    concat_views = list(concat_views) if concat_views is not None else None
    native_width = int(_get(cfg, "width", height))
    pad_views_to = _get(cfg, "pad_views_to")
    explicit_pvw = _get(cfg, "per_view_w")
    if explicit_pvw is not None:
        per_view_w = int(explicit_pvw)
    elif layout == "tiled":
        n_view_slots = int(pad_views_to) if pad_views_to else (
            len(concat_views) if concat_views else 1
        )
        per_view_w = native_width // n_view_slots
    else:
        per_view_w = int(_get(cfg, "width", 192))

    # Source/target fps: flow-planner stride = download.override_fps // cfg.fps, with
    # the SOURCE fps stamped into each record (record["fps"] = override_fps). vera's
    # sampler derives stride from episode.native_fps (the CSV `fps` column) // target_fps.
    # For mimicgen/allegro_sim CSV fps == override_fps so the two agree; for
    # allegro_real/droid where the CSV fps may differ from override_fps, set
    # `temporal_stride` explicitly in the cfg (see configs) to match exactly.
    target_fps = _get(cfg, "fps") if layout == "tiled" else _get(cfg, "target_fps", _get(cfg, "fps"))

    dcfg = DatasetConfig(
        layout=layout,
        n_frames=int(_get(cfg, "n_frames", 37)),
        height=height,
        per_view_w=per_view_w,
        target_fps=target_fps,
        temporal_stride=_get(cfg, "temporal_stride"),
        trim_mode=str(_get(cfg, "trim_mode", "random_cut")),
        pad_mode=str(_get(cfg, "pad_mode", "pad_last")),
        pad_views_to=pad_views_to,
        pad_position=str(_get(cfg, "pad_position", _get(cfg, "view_pad_position", "right"))),
        load_flow=bool(_get(cfg, "load_flow", _get(cfg, "load_optical_flow", False))),
        pad_to_width=_get(cfg, "pad_to_width"),
        id_token=str(_get(cfg, "id_token", "") or ""),
        image_to_video=bool(_get(cfg, "image_to_video", True)),
    )

    data_root = _get(cfg, "data_root", _get(cfg, "root_dir", _get(cfg, "root", ".")))
    metadata_path = _get(cfg, "metadata_path")
    # LongFormatVideoSource (allegro/mimicgen) takes metadata_path + concat_views;
    # DroidSource (wide CSV) takes only data_root. Pass kwargs tolerantly.
    try:
        source = source_cls(
            data_root, metadata_path=metadata_path, concat_views=concat_views
        )
    except TypeError:
        source = source_cls(data_root)
    view_loader = view_loader_cls(height=height, per_view_w=per_view_w)

    dataset_cls: Type[UnifiedDataset] = (
        VideoModelDataset if layout == "tiled" else IDMDataset
    )
    return dataset_cls(source, view_loader, dcfg, seed=int(_get(cfg, "seed", 0)))


__all__ = [
    "build_dataset",
    "resolve_dataset_cfg",
    "register_dataset",
    "DATASET_REGISTRY",
]
