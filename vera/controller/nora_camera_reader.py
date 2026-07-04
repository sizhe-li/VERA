"""NORA camera reader: maps DROID RobotEnv observation to vera view keys.

NORA camera serials (from droid_private/droid/misc/parameters.py):
  hand_camera_id      = "<ZED_SERIAL_HAND>"   ZED-M   (wrist)
  varied_camera_1_id  = "<ZED_SERIAL>"   ZED 2i  (exterior left  / varied_1)
  varied_camera_2_id  = "<ZED_SERIAL_2>"   ZED 2i  (exterior right / varied_2)

DROID ZED observation structure:
  obs["image"]["<serial>_left"]  -> uint8 HxWx4 BGRA  (left stereo lens)
  obs["image"]["<serial>_right"] -> uint8 HxWx4 BGRA  (right stereo lens; not used here)

Mapping mirrors the DROID training convention so model receives the expected views.
"""
from __future__ import annotations

import numpy as np


_VARIED_1 = "<ZED_SERIAL>_left"
_VARIED_2 = "<ZED_SERIAL_2>_left"
_HAND     = "<ZED_SERIAL_HAND>_left"


def _bgra_to_rgb(frame: np.ndarray) -> np.ndarray:
    """BGRA uint8 -> RGB uint8."""
    return frame[..., :3][..., ::-1].copy()


def camera_reader(droid_obs: dict) -> dict:
    """Map NORA's DROID observation -> vera view keys.

    Returns {view_key: uint8 (H, W, 3) RGB}.  The controller resizes internally.
    """
    imgs = droid_obs["image"]
    return {
        "varied_1": _bgra_to_rgb(imgs[_VARIED_1]),
        "varied_2": _bgra_to_rgb(imgs[_VARIED_2]),
        "hand":     _bgra_to_rgb(imgs[_HAND]),
    }
