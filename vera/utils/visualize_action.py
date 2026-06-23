"""
Visualization helpers for SE(3) / EEF delta actions (e.g. [vx, vy, vz, wx, wy, wz, gripper]).

Used to overlay action arrows and text on RGB frames in the style of
notebooks/test_robomimic_dataset/policy_testers/visualize_action_data.ipynb.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw


def to_uint8(frame: np.ndarray) -> np.ndarray:
    """Convert frame to uint8. Pass-through if already uint8."""
    if frame.dtype == np.uint8:
        return frame
    return np.clip(frame * 255.0, 0, 255).astype(np.uint8)


def draw_full_se3_on_frame(
    frame: np.ndarray,
    du_t: np.ndarray,
    trans_scale: float = 200.0,
    rot_scale: float = 200.0,
    grip_scale: float = 80.0,
    *,
    show_text: bool = True,
    show_rotation_label: bool = True,
) -> np.ndarray:
    """Draw full SE(3) delta action on a single RGB frame.

    Args:
        frame: (H, W, 3) uint8 RGB image.
        du_t: (7,) action [vx, vy, vz, wx, wy, wz, dg] (linear vel, angular vel, gripper).
        trans_scale: Scale for translation arrows (pixels per unit).
        rot_scale: Scale for rotation arc (degrees per unit).
        grip_scale: Scale for gripper bar (pixels per unit).

    Returns:
        (H, W, 3) uint8 RGB image with overlays.
    """
    du_t = np.asarray(du_t, dtype=np.float64)
    if du_t.size < 7:
        du_t = np.resize(du_t, 7)  # pad with zeros if needed
    vx, vy, vz, wx, wy, wz, dg = [float(x) for x in du_t.flat[:7]]

    if frame.dtype != np.uint8:
        frame = to_uint8(frame)
    frame = np.asarray(frame)
    H, W = frame.shape[:2]
    cx, cy = W // 2, H // 2

    im = Image.fromarray(frame)
    draw = ImageDraw.Draw(im)

    # --- Translation ---
    dx = int(vx * trans_scale)
    dy = int(-vy * trans_scale)
    dz = int(vz * trans_scale)

    draw.line((cx, cy, cx + dx, cy), fill=(255, 0, 0), width=3)  # X
    draw.line((cx, cy, cx, cy + dy), fill=(0, 255, 0), width=3)  # Y
    draw.line((cx - 8, cy, cx - 8, cy - dz), fill=(0, 0, 255), width=4)  # Z

    r = 4
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(255, 255, 255))

    # --- Rotation (wz) as directional curved arrow ---
    rot_r = 22
    rot_cx, rot_cy = W - 60, 60

    wz_vis = float(np.clip(wz, -1.0, 1.0))
    angle = abs(wz_vis) * rot_scale

    if wz_vis >= 0:
        start, end = 0, angle
        arrow_color = (255, 200, 100)  # CCW (positive)
    else:
        start, end = -angle, 0
        arrow_color = (255, 120, 120)  # CW (negative)

    draw.arc(
        [rot_cx - rot_r, rot_cy - rot_r, rot_cx + rot_r, rot_cy + rot_r],
        start=start,
        end=end,
        fill=arrow_color,
        width=3,
    )

    theta = np.deg2rad(end)
    tip_x = rot_cx + rot_r * np.cos(theta)
    tip_y = rot_cy + rot_r * np.sin(theta)
    arrow_size = 6
    draw.polygon(
        [
            (tip_x, tip_y),
            (tip_x - arrow_size, tip_y - arrow_size // 2),
            (tip_x - arrow_size, tip_y + arrow_size // 2),
        ],
        fill=arrow_color,
    )
    if show_rotation_label:
        draw.text((rot_cx - 18, rot_cy + 30), "wz", fill=arrow_color)

    # --- Gripper ---
    dg = float(np.clip(dg, -1.0, 1.0))
    bar_x = W - 20
    bar_center = H // 2
    bar_h = int(abs(dg) * grip_scale)
    if dg >= 0:
        top, bottom = bar_center - bar_h, bar_center
        color = (200, 255, 200)
    else:
        top, bottom = bar_center, bar_center + bar_h
        color = (255, 150, 150)
    draw.rectangle([bar_x, top, bar_x + 8, bottom], fill=color)

    if show_text:
        draw.text(
            (15, 10), f"v: [{vx:+.3f},{vy:+.3f},{vz:+.3f}]", fill=(255, 255, 255)
        )
        draw.text(
            (15, 30), f"w: [{wx:+.3f},{wy:+.3f},{wz:+.3f}]", fill=(255, 200, 100)
        )
        draw.text((15, 50), f"g: {dg:+.3f}", fill=(200, 255, 200))

    return np.array(im)


def annotate_dual_view_full_se3(
    agentview: np.ndarray,
    eih: np.ndarray,
    du: np.ndarray,
    trans_scale: float = 200.0,
    rot_scale: float = 200.0,
    grip_scale: float = 80.0,
) -> np.ndarray:
    """Annotate agentview and eye-in-hand with same action; return horizontal stack (T, H, 2*W, 3)."""
    agentview = np.asarray(agentview)
    eih = np.asarray(eih)
    du = np.asarray(du)
    T = min(agentview.shape[0], eih.shape[0], du.shape[0])
    frames = []
    for i in range(T):
        av = to_uint8(agentview[i])
        eh = to_uint8(eih[i])
        av_annot = draw_full_se3_on_frame(
            av, du[i], trans_scale=trans_scale, rot_scale=rot_scale, grip_scale=grip_scale
        )
        eh_annot = draw_full_se3_on_frame(
            eh, du[i], trans_scale=trans_scale, rot_scale=rot_scale, grip_scale=grip_scale
        )
        frames.append(np.concatenate([av_annot, eh_annot], axis=1))
    return np.stack(frames, axis=0)


def annotate_two_row_full_se3(
    agentview: np.ndarray,
    eih: np.ndarray,
    du_new: np.ndarray,
    du_old: np.ndarray,
    trans_scale: float = 200.0,
    rot_scale: float = 200.0,
    grip_scale: float = 80.0,
) -> np.ndarray:
    """Annotate two rows (new vs old param); return (T, 2*H, 2*W, 3)."""
    agentview = np.asarray(agentview)
    eih = np.asarray(eih)
    du_new = np.asarray(du_new)
    du_old = np.asarray(du_old)
    T = min(
        agentview.shape[0],
        eih.shape[0],
        du_new.shape[0],
        du_old.shape[0],
    )
    frames = []
    for i in range(T):
        av = to_uint8(agentview[i])
        eh = to_uint8(eih[i])
        av_new = draw_full_se3_on_frame(
            av, du_new[i], trans_scale=trans_scale, rot_scale=rot_scale, grip_scale=grip_scale
        )
        eh_new = draw_full_se3_on_frame(
            eh, du_new[i], trans_scale=trans_scale, rot_scale=rot_scale, grip_scale=grip_scale
        )
        row_new = np.concatenate([av_new, eh_new], axis=1)
        av_old = draw_full_se3_on_frame(
            av, du_old[i], trans_scale=trans_scale, rot_scale=rot_scale, grip_scale=grip_scale
        )
        eh_old = draw_full_se3_on_frame(
            eh, du_old[i], trans_scale=trans_scale, rot_scale=rot_scale, grip_scale=grip_scale
        )
        row_old = np.concatenate([av_old, eh_old], axis=1)
        frames.append(np.concatenate([row_new, row_old], axis=0))
    return np.stack(frames, axis=0)
