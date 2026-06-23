# okto/core/geometry/flow_ops.py
from __future__ import annotations

import torch
import torch.nn.functional as F


def resize_flow(
    flow: torch.Tensor,  # (N, 2, H, W)
    new_H: int,
    new_W: int,
    inter_mode: str = "bilinear",
    align_corners: bool = True,
) -> torch.Tensor:
    """
    Resize a batch of optical flow fields (pixel displacements).

    Args:
        flow: (N, 2, H, W) tensor, where channel 0 = dx, channel 1 = dy
        new_H: target height
        new_W: target width
        inter_mode: interpolation mode ("bilinear" recommended)
        align_corners: align_corners flag for interpolation

    Returns:
        Flow tensor of shape (N, 2, new_H, new_W),
        with displacements rescaled to the new spatial size.
    """
    assert flow.ndim == 4, f"Expected 4D tensor (N, 2, H, W), got {flow.shape}"
    assert flow.shape[1] == 2, f"Flow must have 2 channels, got {flow.shape[1]}"

    curr_H, curr_W = flow.shape[-2:]

    # Scale displacement magnitudes before resizing
    scale_x = new_W / curr_W
    scale_y = new_H / curr_H

    flow = flow.clone()
    flow[:, 0] *= scale_x
    flow[:, 1] *= scale_y

    # Interpolate spatial grid
    flow = F.interpolate(
        flow, size=(new_H, new_W), mode=inter_mode, align_corners=align_corners
    )

    return flow
