from __future__ import annotations
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from jaxtyping import Float


def populate_queues(
    queues: dict[str, deque],
    batch: dict[str, torch.Tensor],
    exclude_keys: list[str] | None = None,
):
    """Append one batch entry per key to its queue.

    Queue storage reflects real observation history — no front-filling on
    first call. Consumers that need a fixed-length context (e.g. the WAN
    motion planner, which requires exactly `required_pixel_frames` frames)
    are responsible for padding the deque contents at call time. See
    `motion_policy.MotionPolicy._compute_plan_joint` for the pad-at-call-time
    pattern.
    """
    if exclude_keys is None:
        exclude_keys = []
    for key in batch:
        if key not in queues or key in exclude_keys:
            continue
        queues[key].append(batch[key])
    return queues


@dataclass
class PolicyObservation:
    rgb: Float[np.ndarray, "batch height width 3"]
    q_robot: Float[np.ndarray, "batch dim_q"] | None
    rgb_vis: Float[np.ndarray, "batch height width 3"] | None = None
    # Optional multiview metadata when rgb is a width-wise concatenation of views.
    view_keys: list[str] | None = None
    view_widths: list[int] | None = None
    concat_rgb_key: str | None = None
    # When using action chunks: "dream" step index (number of action chunks executed so far). Used for policy_vis label.
    dream_index: int | None = None
    # Current env step index (e.g. episode step); drawn on left panel of policy_vis when set.
    step_index: int | None = None
    # State information for absolute position control
    eef_pos: Float[np.ndarray, "batch 3"] | None = None  # end-effector position
    eef_quat: Float[np.ndarray, "batch 4"] | None = (
        None  # end-effector quaternion (x,y,z,w)
    )
    gripper_qpos: Float[np.ndarray, "batch 2"] | None = None  # gripper joint positions
    dt: float = 0.1  # time step for velocity integration
    action_mode: Literal["velocity", "absolute"] = "velocity"
    pose_format: Literal["quat", "axis_angle"] = "quat"  # rotation format for actions


@dataclass
class PolicyOutput:
    action: Float[np.ndarray, "batch dim_u"]
    info: dict | None


@dataclass
class BasePolicyCfg:
    name: Literal[""]


class BasePolicy(ABC):
    cfg: BasePolicyCfg
    device: torch.device

    @abstractmethod
    def predict_action(self, obs: PolicyObservation) -> PolicyOutput:
        raise NotImplementedError()

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError()

    def observe_rollout_feedback(
        self,
        obs: PolicyObservation,
    ) -> dict | None:
        """Optional post-step feedback hook.

        Policies that do not use online adaptation can ignore rollout feedback by
        inheriting this default no-op implementation.
        """
        del obs
        return None

    def get_warm_start_state(self) -> dict | None:
        """Optional hook for persisting lightweight runtime state across episodes."""
        return None

    def set_warm_start_state(self, state: dict | None) -> None:
        """Optional hook for restoring lightweight runtime state across episodes."""
        del state
