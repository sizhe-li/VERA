"""Shared fps-aware temporal frame sampling.

This unifies the two original frame-sampling strategies:

* flow-planner (video model) — ``datasets/video_base.py::_temporal_sample``: subsample by an
  integer fps ratio (``override_fps`` / ``data_fps``) and pick ``n_frames`` evenly inside a
  source window chosen by ``trim_mode`` (``random_cut`` / ``first_chunk`` / ``speedup``), with
  ``pad_mode`` (``pad_last`` / ``slowdown`` / ``discard``) when the clip is too short.
* okto (Jacobian IDM) — a uniform window over the trajectory. That is exactly this sampler with
  ``temporal_stride=1`` (i.e. ``target_fps == source_fps``) and ``trim_mode="window"``.

Pure NumPy, no torch / no I/O — unit-testable without data. Returns integer source-frame indices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_TRIM_MODES = ("random_cut", "first_chunk", "speedup", "window")
_PAD_MODES = ("pad_last", "slowdown", "discard")


@dataclass(frozen=True)
class FrameSamplerConfig:
    """Configuration for :func:`sample_frame_indices`.

    Args:
        n_frames: number of output frames to return.
        source_fps: native fps of the raw clip.
        target_fps: desired fps (<= source_fps). ``None`` -> no subsampling (= source_fps).
            ``source_fps`` must be an integer multiple of ``target_fps`` (matches flow-planner's
            assertion ``override_fps % data_fps == 0``).
        temporal_stride: explicit subsample stride; overrides the fps-derived stride when set.
        trim_mode: how to pick the source window when the clip is longer than needed.
            ``window`` is an alias for ``first_chunk`` (the okto uniform-window default).
        pad_mode: how to handle clips shorter than the requested window.
    """

    n_frames: int
    source_fps: int = 1
    target_fps: int | None = None
    temporal_stride: int | None = None
    trim_mode: str = "random_cut"
    pad_mode: str = "pad_last"

    def __post_init__(self) -> None:
        if self.n_frames < 1:
            raise ValueError(f"n_frames must be >= 1, got {self.n_frames}")
        if self.trim_mode not in _TRIM_MODES:
            raise ValueError(f"trim_mode must be one of {_TRIM_MODES}, got {self.trim_mode!r}")
        if self.pad_mode not in _PAD_MODES:
            raise ValueError(f"pad_mode must be one of {_PAD_MODES}, got {self.pad_mode!r}")

    def stride(self) -> int:
        """Integer subsample stride (>= 1)."""
        if self.temporal_stride is not None:
            return max(1, int(self.temporal_stride))
        tf = int(self.target_fps or self.source_fps)
        if tf <= 0:
            raise ValueError(f"target_fps must be > 0, got {tf}")
        if int(self.source_fps) % tf != 0:
            raise ValueError(
                f"source_fps ({self.source_fps}) must be an integer multiple of "
                f"target_fps ({tf})."
            )
        return max(1, int(self.source_fps) // tf)


def sample_frame_indices(
    num_src_frames: int,
    cfg: FrameSamplerConfig,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Pick ``cfg.n_frames`` source-frame indices from a clip of ``num_src_frames`` frames.

    With ``rng=None`` the sampling is deterministic (``random_cut`` starts at 0), which is what the
    parity test and reproducible eval rely on. Returns an int64 array of shape ``(n_frames,)``,
    each index in ``[0, num_src_frames - 1]``.
    """
    if num_src_frames < 1:
        raise ValueError(f"num_src_frames must be >= 1, got {num_src_frames}")

    n = cfg.n_frames
    stride = cfg.stride()
    target_len = n * stride  # source-frame span we want to cover at this stride

    if num_src_frames < target_len:
        # Too short -> pad.
        if cfg.pad_mode == "discard":
            raise ValueError(
                f"clip too short ({num_src_frames} < {target_len}) and pad_mode='discard'"
            )
        if cfg.pad_mode == "pad_last":
            # Walk past the end; clamp below makes the final frame repeat (pad with last).
            idx = np.linspace(0, target_len - 1, n)
        else:  # "slowdown": stretch the available frames across n outputs.
            idx = np.linspace(0, num_src_frames - 1, n)
    elif num_src_frames > target_len:
        if cfg.trim_mode == "random_cut":
            max_start = num_src_frames - target_len
            start = int(rng.integers(0, max_start + 1)) if rng is not None else 0
            idx = start + np.linspace(0, target_len - 1, n)
        elif cfg.trim_mode in ("first_chunk", "window"):
            idx = np.linspace(0, target_len - 1, n)
        else:  # "speedup": span the whole clip, skipping more frames than the stride implies.
            idx = np.linspace(0, num_src_frames - 1, n)
    else:  # exact fit
        idx = np.linspace(0, num_src_frames - 1, n)

    idx = np.round(idx).astype(np.int64)
    # pad_last (and any rounding overshoot) clamps to the last available frame.
    return np.clip(idx, 0, num_src_frames - 1)
