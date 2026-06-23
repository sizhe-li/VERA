from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.spatial.transform import Rotation as R

from vera.utils.visualize_action import draw_full_se3_on_frame


@dataclass
class AdaptiveControllerCfg:
    # Disabled by default so existing configs and notebooks preserve behavior.
    enabled: bool = False
    mode: Literal["track", "state_delta_grouped"] = "track"
    ema_alpha: float = 0.2
    invalid_track_penalty: float = 0.25
    lam_max_scale: float = 4.0
    action_gain_min_scale: float = 0.35
    action_gain_max_scale: float = 1.0
    eps: float = 1e-6
    use_track_mismatch_for_lam: bool = True
    grouped_action_ema_alpha: float = 0.2
    grouped_action_eps: float = 1e-4
    grouped_action_dt_eps: float = 1e-6
    grouped_action_step_up: float = 0.25
    grouped_action_step_down: float = 0.15
    translation_gain_min_scale: float = 0.5
    translation_gain_max_scale: float = 3.0
    rotation_gain_min_scale: float = 0.5
    rotation_gain_max_scale: float = 3.0
    gripper_gain_min_scale: float = 0.5
    gripper_gain_max_scale: float = 3.0
    enable_gripper_channel_adaptation: bool = True
    # Synthetic warm-start: initial mismatch_ema applied at every reset.
    # Replaces the lost Mar-12 adaptive snapshot. mismatch_ema > 0 makes the
    # controller start in a defensive regime (lower gain, higher lam) instead
    # of full aggression at step 0.
    initial_mismatch_ema: float = 0.0


@dataclass
class AdaptiveChannelState:
    translation: float = 1.0
    rotation: float = 1.0
    gripper: float = 1.0

    def as_dict(self) -> dict[str, float]:
        return {
            "translation": float(self.translation),
            "rotation": float(self.rotation),
            "gripper": float(self.gripper),
        }


def quat_delta_to_rotvec(
    quat_start: np.ndarray,
    quat_end: np.ndarray,
) -> np.ndarray:
    rot_start = R.from_quat(quat_start)
    rot_end = R.from_quat(quat_end)
    return (rot_end * rot_start.inv()).as_rotvec()


def gripper_width_from_qpos(gripper_qpos: np.ndarray) -> np.ndarray:
    if gripper_qpos.shape[-1] >= 2:
        return 0.5 * np.abs(gripper_qpos[..., 1] - gripper_qpos[..., 0])
    return np.abs(gripper_qpos[..., 0])


def draw_cartesian_action_on_frame(
    frame: np.ndarray,
    action: np.ndarray | None,
) -> np.ndarray:
    if action is None:
        return frame
    action_np = np.asarray(action)
    if action_np.size < 7:
        return frame
    return draw_full_se3_on_frame(
        frame,
        action_np,
        show_text=False,
        show_rotation_label=False,
    )
