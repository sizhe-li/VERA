"""View-layout adapter — the seam between the two stages.

Both stages start from the SAME per-view tensors (loaded separately, lossless). The only
difference is the final layout:

* Jacobian IDM        -> ``separate``: ``[T, V, C, H, W]`` (views are their own dim).
* video model (WAN)   -> ``tiled``:    ``[T, C, H, W*V]`` (views concatenated along width).

Both layouts are produced from the **same** padded per-view list, so the invariant

    tiled == cat([separate[:, v] for v in range(V)], dim=-1)

holds by construction. This is what the ``combined_4env`` 14B run violated: DROID was routed to a
single-view loader (``DroidVideoDataset``) instead of the multiview one, silently producing one
stretched view. Here the layout is an explicit, tested argument — not an accident of class routing.

``pad_views_to`` + ``pad_position='right'`` reproduce flow-planner's black-pad-right for subsets with
fewer than the canvas's view slots (e.g. 2-view MimicGen padded into a 3-slot canvas).
"""

from __future__ import annotations

from typing import List, Sequence

import torch

SEPARATE = "separate"
TILED = "tiled"
_LAYOUTS = (SEPARATE, TILED)


def _check_views(views: Sequence[torch.Tensor]) -> None:
    if len(views) == 0:
        raise ValueError("need at least one view tensor")
    shape0 = views[0].shape  # (T, C, H, W)
    if len(shape0) != 4:
        raise ValueError(f"each view must be [T, C, H, W]; got {tuple(shape0)}")
    for i, v in enumerate(views):
        if v.shape != shape0:
            raise ValueError(
                f"all views must share shape; view 0 is {tuple(shape0)} but view {i} is "
                f"{tuple(v.shape)}. (per-view resize happens in view_loader before layout.)"
            )


def pad_views(
    views: Sequence[torch.Tensor],
    pad_views_to: int | None = None,
    pad_position: str = "right",
) -> List[torch.Tensor]:
    """Append/prepend black (zero) view panels until there are ``pad_views_to`` views.

    Each view is ``[T, C, H, W]``. Returns a new list (real views + black panels). When
    ``pad_views_to`` is ``None`` or <= len(views), returns the views unchanged.
    """
    _check_views(views)
    views = list(views)
    if pad_views_to is None or pad_views_to <= len(views):
        return views
    if pad_position not in ("right", "left"):
        raise ValueError(f"pad_position must be 'right'|'left', got {pad_position!r}")
    n_pad = pad_views_to - len(views)
    black = torch.zeros_like(views[0])
    pads = [black.clone() for _ in range(n_pad)]
    return views + pads if pad_position == "right" else pads + views


def to_separate(views: Sequence[torch.Tensor], squeeze_single: bool = True) -> torch.Tensor:
    """Stack per-view tensors into ``[T, V, C, H, W]`` (or ``[T, C, H, W]`` if V==1 and squeeze)."""
    _check_views(views)
    out = torch.stack(list(views), dim=1)  # [T, V, C, H, W]
    if squeeze_single and out.shape[1] == 1:
        out = out.squeeze(1)
    return out


def to_tiled(views: Sequence[torch.Tensor]) -> torch.Tensor:
    """Concatenate per-view tensors along width into ``[T, C, H, W*V]``."""
    _check_views(views)
    return torch.cat(list(views), dim=-1)


def pad_to_width(frame: torch.Tensor, pad_to_width: int | None, pad_position: str = "right") -> torch.Tensor:
    """Black-pad a tiled ``[..., H, W]`` tensor on the width axis up to ``pad_to_width``.

    Port of flow-planner ``droid_flow.py::_load_video_concat`` lines 822-837: a mixture
    that combines subsets of different native widths (combined_4env: DROID 576,
    allegro_sim 384, allegro_real/mimicgen 256) right-pads each subset's post-concat
    frame to a common canvas. Padding happens BEFORE [-1,1] normalize, so the padded
    region sits at the same value as the missing-view (``pad_views``) black panels.
    ``pad_position='center'`` splits the extra width evenly (matches flow-planner).
    """
    if pad_to_width is None or frame.shape[-1] >= int(pad_to_width):
        return frame
    extra_w = int(pad_to_width) - frame.shape[-1]
    lead = frame.shape[:-1]
    if pad_position == "center":
        left_w = extra_w // 2
        right_w = extra_w - left_w
        left = torch.zeros(*lead, left_w, dtype=frame.dtype, device=frame.device)
        right = torch.zeros(*lead, right_w, dtype=frame.dtype, device=frame.device)
        return torch.cat([left, frame, right], dim=-1)
    return torch.cat(
        [frame, torch.zeros(*lead, extra_w, dtype=frame.dtype, device=frame.device)], dim=-1
    )


def apply_layout(
    views: Sequence[torch.Tensor],
    layout: str,
    pad_views_to: int | None = None,
    pad_position: str = "right",
    squeeze_single: bool = True,
) -> torch.Tensor:
    """Produce the requested layout from a list of per-view ``[T, C, H, W]`` tensors.

    Args:
        layout: ``"separate"`` (IDM) or ``"tiled"`` (video model).
        pad_views_to: pad to this many view slots with black panels first (both layouts).
        pad_position: where the black panels go (``"right"`` matches flow-planner).
        squeeze_single: for ``separate`` only, drop the V dim when V==1.
    """
    if layout not in _LAYOUTS:
        raise ValueError(f"layout must be one of {_LAYOUTS}, got {layout!r}")
    padded = pad_views(views, pad_views_to=pad_views_to, pad_position=pad_position)
    if layout == SEPARATE:
        return to_separate(padded, squeeze_single=squeeze_single)
    return to_tiled(padded)
