from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch

_TRACK_COORD_ABS_CLIP = 1.0e6


def _tensor_points_to_numpy(points: torch.Tensor) -> np.ndarray:
    pts = points.detach().float()
    pts = torch.nan_to_num(
        pts,
        nan=0.0,
        posinf=_TRACK_COORD_ABS_CLIP,
        neginf=-_TRACK_COORD_ABS_CLIP,
    ).clamp(-_TRACK_COORD_ABS_CLIP, _TRACK_COORD_ABS_CLIP)
    return pts.cpu().numpy()


def _visible_to_numpy(visible: Optional[torch.Tensor], length: int) -> np.ndarray:
    if visible is None:
        return np.ones(length, dtype=bool)
    vis = visible.detach().cpu().numpy()
    out = np.asarray(vis).reshape(-1).astype(bool)
    if out.shape[0] == length:
        return out
    fixed = np.zeros(length, dtype=bool)
    fixed[: min(length, out.shape[0])] = out[: min(length, out.shape[0])]
    return fixed


def draw_curr_trgt_tracks(
    obs: np.ndarray,
    curr_track: torch.Tensor,
    trgt_track: Optional[torch.Tensor] = None,
    curr_visible: Optional[torch.Tensor] = None,
    trgt_visible: Optional[torch.Tensor] = None,
    point_color: Tuple[int, int, int] = (0, 255, 0),
    arrow_color: Tuple[int, int, int] = (255, 0, 0),
    point_radius: int = 3,
    arrow_thickness: int = 1,
    sparsity: int = 1,
) -> np.ndarray:
    """
    Draw arrows (if trgt_track provided) or points for curr_track on the image.

    Args:
        obs: H,W,C np.ndarray (uint8)
        curr_track: torch.Tensor [M, 2] (x, y)
        trgt_track: optional torch.Tensor [M, 2]
        curr_visible: optional torch.Tensor [M] (bool or 0/1)
        trgt_visible: optional torch.Tensor [M] (bool or 0/1)
        point_color: BGR tuple for drawing
        point_radius: radius of points
        arrow_thickness: thickness of arrows
        sparsity: int, only draw every `sparsity`-th point/arrow (>=1)

    Returns:
        canvas: H,W,C np.ndarray with drawings
    """
    canvas = obs.copy()
    H, W = obs.shape[0], obs.shape[1]
    curr = _tensor_points_to_numpy(curr_track)
    trgt = _tensor_points_to_numpy(trgt_track) if trgt_track is not None else None

    curr_vis = _visible_to_numpy(curr_visible, len(curr))
    trgt_vis = (
        _visible_to_numpy(trgt_visible, len(curr))
        if trgt_visible is not None
        else (np.ones(len(curr), dtype=bool) if trgt is not None else None)
    )

    sparsity = max(int(sparsity), 1)
    for i in range(0, len(curr), sparsity):
        if not curr_vis[i] or not np.isfinite(curr[i]).all():
            continue

        x_start = int(np.clip(round(curr[i, 0]), 0, W - 1))
        y_start = int(np.clip(round(curr[i, 1]), 0, H - 1))
        cv2.circle(canvas, (x_start, y_start), point_radius, point_color, -1)

        # if magnitude is too small, skip drawing arrow
        if (
            trgt is not None
            and trgt_vis is not None
            and trgt_vis[i]
            and np.isfinite(trgt[i]).all()
            and np.hypot(*(curr[i] - trgt[i])) > 0.0
        ):
            x_des = int(np.clip(round(trgt[i, 0]), 0, W - 1))
            y_des = int(np.clip(round(trgt[i, 1]), 0, H - 1))
            canvas = cv2.arrowedLine(
                canvas,
                (x_start, y_start),
                (x_des, y_des),
                color=arrow_color,
                thickness=arrow_thickness,
                tipLength=0.3,
            )

    return canvas


def draw_curr_trgt_tracks_dense(
    obs: np.ndarray,
    curr_track: torch.Tensor,
    trgt_track: Optional[torch.Tensor] = None,
    curr_visible: Optional[torch.Tensor] = None,
    trgt_visible: Optional[torch.Tensor] = None,
    point_color: Tuple[int, int, int] = (0, 255, 0),
    arrow_color: Tuple[int, int, int] = (255, 0, 0),
    point_radius: int = 3,
    arrow_thickness: int = 1,
    sparsity: int = 100,
    motion_thresh: float = 0.5,
    keep_indices: Optional[Union[torch.Tensor, np.ndarray]] = None,
    arrow_scale: float = 1.0,
) -> np.ndarray:
    """
    Draw curr->trgt motion arrows with grid-based sparsity and motion threshold.
    Use this for clearer, denser visualizations (inspired by image_jacobian.visualize_pixel_motion).

    Args:
        obs: H,W,C np.ndarray (uint8)
        curr_track: torch.Tensor [M, 2] (x, y) in pixel coords
        trgt_track: optional torch.Tensor [M, 2]
        curr_visible: optional torch.Tensor [M] (bool or 0/1)
        trgt_visible: optional torch.Tensor [M] (bool or 0/1)
        point_color: BGR tuple for drawing points
        arrow_color: BGR tuple for arrows
        point_radius: radius of points
        arrow_thickness: thickness of arrows
        sparsity: int, higher -> fewer arrows; controls grid cell count (one point per cell)
        motion_thresh: skip arrow when motion norm (pixels) <= this
        keep_indices: optional 1D indices to draw (e.g. for temporal consistency); when None, use grid sparsity
        arrow_scale: scale factor for arrow length (1.0 = true displacement; >1 draws longer arrows)

    Returns:
        canvas: H,W,C np.ndarray with drawings
    """
    canvas = obs.copy()
    H, W = obs.shape[0], obs.shape[1]

    curr = _tensor_points_to_numpy(curr_track)
    trgt = _tensor_points_to_numpy(trgt_track) if trgt_track is not None else None
    curr_vis = _visible_to_numpy(curr_visible, len(curr))
    trgt_vis = (
        _visible_to_numpy(trgt_visible, len(curr))
        if trgt_visible is not None
        else (np.ones(len(curr), dtype=bool) if trgt is not None else None)
    )
    finite = np.isfinite(curr).all(axis=-1)
    if trgt is not None:
        finite &= np.isfinite(trgt).all(axis=-1)
    curr_vis = curr_vis & finite
    if trgt_vis is not None:
        trgt_vis = trgt_vis & finite

    if keep_indices is not None:
        if isinstance(keep_indices, torch.Tensor):
            keep_indices = keep_indices.detach().cpu().numpy()
        keep_indices = np.asarray(keep_indices, dtype=np.int64)
        N_cur = curr.shape[0]
        valid = keep_indices < N_cur
        keep_indices = keep_indices[valid]
        curr = curr[keep_indices]
        trgt = trgt[keep_indices] if trgt is not None else None
        curr_vis = curr_vis[keep_indices]
        trgt_vis = trgt_vis[keep_indices] if trgt_vis is not None else None
    else:
        # Per-call sparsification via spatial grid (same logic as image_jacobian.visualize_pixel_motion)
        base_area = 8 * 8
        approx_arrows = max(int((H * W) / base_area / max(sparsity / 100.0, 1e-3)), 1)
        approx_arrows = min(approx_arrows, int(curr.shape[0]))
        grid_size = max(int(np.sqrt(approx_arrows)), 1)
        gx = max(W // grid_size, 1)
        gy = max(H // grid_size, 1)
        x = np.clip(curr[:, 0], 0, W - 1)
        y = np.clip(curr[:, 1], 0, H - 1)
        cell_x = np.floor(x / gx).astype(np.int64).clip(0, W // gx)
        cell_y = np.floor(y / gy).astype(np.int64).clip(0, H // gy)
        num_x = (W // gx) + 1
        cell_id = cell_y * num_x + cell_x
        N = cell_id.shape[0]
        max_cell = int(cell_id.max()) + 1
        first_idx = np.full(max_cell, N, dtype=np.int64)
        for i in range(N):
            cid = cell_id[i]
            if first_idx[cid] > i:
                first_idx[cid] = i
        unique_indices = first_idx[first_idx < N]
        keep = np.zeros(N, dtype=bool)
        keep[unique_indices] = True
        curr = curr[keep]
        trgt = trgt[keep] if trgt is not None else None
        curr_vis = curr_vis[keep]
        trgt_vis = trgt_vis[keep] if trgt_vis is not None else None

    for i in range(len(curr)):
        if not curr_vis[i]:
            continue
        x_start = int(np.clip(round(curr[i, 0]), 0, W - 1))
        y_start = int(np.clip(round(curr[i, 1]), 0, H - 1))
        if trgt is not None and trgt_vis is not None and trgt_vis[i]:
            dx = trgt[i, 0] - curr[i, 0]
            dy = trgt[i, 1] - curr[i, 1]
            motion_norm = np.hypot(dx, dy)
            if motion_norm <= motion_thresh:
                continue

            x_des = curr[i, 0] + arrow_scale * dx
            y_des = curr[i, 1] + arrow_scale * dy
            x_des = int(np.clip(round(x_des), 0, W - 1))
            y_des = int(np.clip(round(y_des), 0, H - 1))
            cv2.arrowedLine(
                canvas,
                (x_start, y_start),
                (x_des, y_des),
                color=arrow_color,
                thickness=arrow_thickness,
                tipLength=0.3,
            )
    return canvas
