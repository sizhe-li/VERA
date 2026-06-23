"""Build the wire observation from a robot backend's state.

Produces the OLD-vera obs contract the server adapter consumes (NOT roboarena): a width-concat
``context_rgb`` over ``view_keys`` plus proprio. Maintains the rolling context window and the
cold-start rule (the first obs of an episode carries a SINGLE frame — repeated frames collapse the
planner, Bug B; matches DreamZero's ``test_client_AR`` frame-[0] cold start). See CONTROLLER_SPEC §3.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("vera.controller.obs")


class ObsBuilder:
    def __init__(
        self,
        view_keys: List[str],
        *,
        image_hw: tuple[int, int] = (128, 192),     # (H, per_view_W) each view resized to before concat
        context_frames: int = 9,
        proprio_keys: Optional[List[str]] = None,
    ) -> None:
        self.view_keys = list(view_keys)
        self.image_h, self.per_view_w = int(image_hw[0]), int(image_hw[1])
        self.context_frames = int(context_frames)
        self.proprio_keys = list(proprio_keys or ["joint_position", "cartesian_position", "gripper_position"])
        self._window: deque = deque(maxlen=self.context_frames)

    def reset(self) -> None:
        self._window.clear()

    @property
    def view_widths(self) -> List[int]:
        return [self.per_view_w] * len(self.view_keys)

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        import cv2
        # frame: (H,W,3) uint8 -> (image_h, per_view_w, 3)
        return cv2.resize(frame, (self.per_view_w, self.image_h), interpolation=cv2.INTER_AREA)

    def _concat_views(self, state: Dict[str, Any]) -> np.ndarray:
        """Width-concat the per-view frames in view_keys order -> (H, ΣW, 3) uint8."""
        tiles = []
        for k in self.view_keys:
            if k not in state:
                raise KeyError(f"backend state missing view '{k}'; has {list(state)}")
            tiles.append(self._resize(np.asarray(state[k])))
        return np.concatenate(tiles, axis=1)

    def current_window(self) -> Optional[np.ndarray]:
        """Stacked (T,H,ΣW,3) uint8 of the current rolling window WITHOUT appending a frame.
        None when the window is empty. Used by the post-chunk 'observe' viewer update."""
        if not self._window:
            return None
        return np.stack(list(self._window), axis=0)

    def append_frame(self, state: Dict[str, Any]) -> np.ndarray:
        """Push one frame into the rolling window without building the full obs dict.

        Used by the playback loop to accumulate context at the control rate between
        infer calls, so the window grows toward context_frames during chunk execution.
        Returns the concatenated (H, ΣW, 3) frame so callers can also record it.
        """
        frame = self._concat_views(state)
        self._window.append(frame)
        return frame

    def build(
        self,
        state: Dict[str, Any],
        *,
        session_id: str,
        prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Append the current frame to the window and return the wire obs dict.

        First call after reset() (or after a prefill) yields a context_rgb with however
        many frames are already in the window + 1 (cold start = 1 frame if no prefill).
        Subsequent calls grow the window up to context_frames.
        """
        self._window.append(self._concat_views(state))
        context_rgb = np.stack(list(self._window), axis=0)   # (T, H, ΣW, 3) uint8
        logger.info("[obs] ctx_window=%d/%d (build)", len(self._window), self.context_frames)

        obs: Dict[str, Any] = {
            "context_rgb": context_rgb,
            "view_keys": list(self.view_keys),
            "view_widths": self.view_widths,
            "session_id": session_id,
        }
        # proprio under the old-vera names; also mirror joint_position to q_robot (policy obs key).
        for key in self.proprio_keys:
            if key in state:
                obs[key] = np.asarray(state[key], dtype=np.float32)
        if "joint_position" in state:
            obs["q_robot"] = np.asarray(state["joint_position"], dtype=np.float32)
        if "cartesian_position" in state:
            # eef_pos [x,y,z]: required by the server's grouped adaptive gain (achieved-vs-
            # desired translation, operator 33207a2). cartesian_position is [xyz, euler].
            obs["eef_pos"] = np.asarray(state["cartesian_position"][:3], dtype=np.float32)
        if prompt is not None:
            obs["prompt"] = prompt
        return obs
