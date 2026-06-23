from __future__ import annotations

import cv2
import numpy as np
from torch import Tensor


def frame_to_uint8(frame: np.ndarray) -> np.ndarray:
    if frame.dtype == np.uint8:
        return frame
    if np.issubdtype(frame.dtype, np.floating):
        frame = np.clip(frame, 0.0, 1.0) * 255.0
    return np.clip(frame, 0, 255).astype(np.uint8)


def rgb_tensor_to_uint8_video(rgb_sequence: Tensor) -> np.ndarray:
    rgb_np = rgb_sequence.detach().cpu().float().numpy()
    if rgb_np.ndim != 4 or rgb_np.shape[1] != 3:
        raise ValueError(f"Expected RGB sequence [T,3,H,W], got {tuple(rgb_sequence.shape)}")
    rgb_np = np.transpose(rgb_np, (0, 2, 3, 1))
    return np.stack([frame_to_uint8(frame) for frame in rgb_np], axis=0)


def draw_action_arrow(
    frame: np.ndarray,
    action_xy: np.ndarray,
    *,
    color: tuple[int, int, int],
    label: str,
    origin: tuple[int, int],
    scale: float,
    text_y: int,
) -> np.ndarray:
    canvas = np.ascontiguousarray(frame.copy())
    x0, y0 = origin
    dx = int(float(action_xy[0]) * scale)
    dy = int(float(action_xy[1]) * scale)
    end = (x0 + dx, y0 - dy)
    cv2.arrowedLine(canvas, origin, end, (0, 0, 0), 4, tipLength=0.22)
    cv2.arrowedLine(canvas, origin, end, color, 2, tipLength=0.18)
    text = f"{label}: [{float(action_xy[0]):+.3f}, {float(action_xy[1]):+.3f}]"
    cv2.putText(
        canvas,
        text,
        (12, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        text,
        (12, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )
    return canvas


def build_pusht_action_overlay_video(
    *,
    rgb_sequence: Tensor,
    gt_action: Tensor,
    pred_action: Tensor,
) -> np.ndarray:
    frames = rgb_tensor_to_uint8_video(rgb_sequence)
    gt_np = gt_action.detach().cpu().float().numpy()
    pred_np = pred_action.detach().cpu().float().numpy()
    if gt_np.shape != (2,) or pred_np.shape != (2,):
        raise ValueError(
            f"Expected PushT 2D actions for visualization, got {gt_np.shape} and {pred_np.shape}"
        )

    height, width = int(frames.shape[1]), int(frames.shape[2])
    origin = (width // 2, height // 2)
    scale = 0.35 * float(min(height, width))
    rendered = []
    for idx, frame in enumerate(frames):
        canvas = draw_action_arrow(
            frame,
            gt_np,
            color=(90, 220, 90),
            label="GT",
            origin=origin,
            scale=scale,
            text_y=24,
        )
        canvas = draw_action_arrow(
            canvas,
            pred_np,
            color=(255, 191, 0),
            label="Pred",
            origin=origin,
            scale=scale,
            text_y=46,
        )
        cv2.putText(
            canvas,
            f"frame {idx}",
            (12, height - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        rendered.append(np.transpose(canvas, (2, 0, 1)))
    return np.stack(rendered, axis=0)
