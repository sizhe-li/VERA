"""Video visualization utilities for WAN training and validation."""

import torch
from einops import rearrange

from .optical_flow import flow_to_rgb  # re-exported for consumers


def add_red_border(frames, border_width=4):
    """Add red border to frames [T, C, H, W] in [0,1] range. Modifies in-place."""
    frames[:, 0, :border_width, :] = 1.0
    frames[:, 1:, :border_width, :] = 0.0
    frames[:, 0, -border_width:, :] = 1.0
    frames[:, 1:, -border_width:, :] = 0.0
    frames[:, 0, :, :border_width] = 1.0
    frames[:, 1:, :, :border_width] = 0.0
    frames[:, 0, :, -border_width:] = 1.0
    frames[:, 1:, :, -border_width:] = 0.0
    return frames


def pad_temporal(video, target_t):
    """Pad [B, T, C, H, W] to target_t by repeating the last frame."""
    if video.shape[1] < target_t:
        pad = video[:, -1:].expand(-1, target_t - video.shape[1], -1, -1, -1)
        return torch.cat([video, pad], dim=1)
    return video


def build_pred_flow_panel(pred_flow, total_T, ctx_pixel_frames, height, width):
    """Build a predicted flow RGB panel [B, T, 3, H, W].

    Args:
        pred_flow: [B, 2, T_flow, H, W] — raw 2-channel flow from flow_decode,
                   covering pixel frames 1 onward (context then prediction).
        total_T: total temporal length of the output panel.
        ctx_pixel_frames: number of context (input) pixel frames.
        height, width: spatial dimensions.
    """
    B, device = pred_flow.shape[0], pred_flow.device
    panel = torch.zeros(B, total_T, 3, height, width, device=device)
    n_ctx_flow = ctx_pixel_frames - 1

    if ctx_pixel_frames > 1:
        panel[:, 1:ctx_pixel_frames] = rearrange(
            flow_to_rgb(pred_flow[:, :, :n_ctx_flow]), "b c t h w -> b t c h w"
        )
        for bi in range(B):
            add_red_border(panel[bi, :ctx_pixel_frames])

    pred_src = pred_flow[:, :, n_ctx_flow:]
    pred_rgb = rearrange(flow_to_rgb(pred_src), "b c t h w -> b t c h w")
    end = min(ctx_pixel_frames + pred_rgb.shape[1], total_T)
    panel[:, ctx_pixel_frames:end] = pred_rgb[:, : end - ctx_pixel_frames]

    return panel


def build_gt_flow_panel(optical_flow_bt, total_T, height, width, start=0):
    """Build a GT flow RGB panel [B, total_T, 3, H, W].

    Args:
        optical_flow_bt: [B, T-1, 2, H, W] — raw GT optical flow from the batch.
        total_T: total temporal length of the output panel.
        height, width: spatial dimensions.
        start: temporal offset into optical_flow_bt.
    """
    gt = rearrange(optical_flow_bt, "b t c h w -> b c t h w")
    B, device = gt.shape[0], gt.device
    panel = torch.zeros(B, total_T, 3, height, width, device=device)
    n = min(total_T - 1, gt.shape[2] - start)
    if n > 0:
        panel[:, 1 : 1 + n] = rearrange(
            flow_to_rgb(gt[:, :, start : start + n]), "b c t h w -> b t c h w"
        )
    return panel
