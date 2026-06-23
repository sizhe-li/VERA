"""
MimicGen environment runner for policy evaluation and replay.

Uses mimicgen.utils.robomimic_utils.create_env and a gymnasium.Env-compatible
MimicGenImageWrapper (similar to RobomimicImageWrapper). Resets to initial_state
built from HDF5 (states[0], model_file, ep_meta) with use_stored_model=True.
"""

import collections
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Union

import cv2
import gymnasium as gym
import h5py
import mediapy as media
import numpy as np
import torch
import tqdm
from vera.env_runner.base_runner import BaseRunner, BaseRunnerCfg, rr
from vera.policy.base_policy import BasePolicy, PolicyObservation, PolicyOutput
from vera.utils.logging import cyan
from robomimic.utils import obs_utils as ObsUtils
from robomimic.utils.file_utils import get_env_metadata_from_dataset
import robosuite.utils.transform_utils as T
from scipy.spatial.transform import Rotation

# MimicGen env creation (optional import so runner can be used without third_party)
try:
    from mimicgen.utils.robomimic_utils import create_env as mimicgen_create_env
except ImportError:
    mimicgen_create_env = None

from vera.utils.mimicgen_playback_utils import build_initial_state


# ---------------------------------------------------------------------------
# MimicGenImageWrapper: gymnasium.Env wrapper for MimicGen (like RobomimicImageWrapper)
# ---------------------------------------------------------------------------
RenderObsKey = str | list[str]


def _resolve_render_obs(
    obs: dict[str, Any],
    render_obs_key: RenderObsKey,
    *,
    fallback_image_keys: Optional[list[str]] = None,
) -> np.ndarray:
    """Resolve one or more image observations, concatenating multi-view inputs side by side."""
    if isinstance(render_obs_key, str):
        keys = [render_obs_key]
    else:
        keys = list(render_obs_key)
        if not keys:
            raise ValueError("render_obs_key list must contain at least one image key")

    if len(keys) == 1:
        key = keys[0]
        if key not in obs:
            if fallback_image_keys:
                fallback_key = fallback_image_keys[0]
                if fallback_key in obs:
                    key = fallback_key
                else:
                    raise KeyError(f"Fallback image key not found: {fallback_key}")
            else:
                raise KeyError(f"Render observation key not found: {key}")
        value = np.asarray(obs[key])
        if value.ndim == 4 and value.shape[0] == 1:
            value = value[0]
        return value.copy()

    frames = []
    for key in keys:
        if key not in obs:
            raise KeyError(
                f"Render observation key not found for multiview path: {key}"
            )
        value = np.asarray(obs[key])
        if value.ndim == 4 and value.shape[0] == 1:
            value = value[0]
        frames.append(value)

    heights = {frame.shape[0] for frame in frames}
    channels = {frame.shape[2] for frame in frames}
    if len(heights) != 1 or len(channels) != 1:
        raise ValueError(
            "All multiview render observations must share the same height and channel count"
        )
    return np.concatenate(frames, axis=1).copy()


class MimicGenImageWrapper(gym.Env):
    """Wraps MimicGen/robosuite env to provide normalized image obs and gymnasium API.

    Supports full MimicGen initial_state dict (states + model + ep_meta) for
    reset_to() when init_state is a dict; otherwise init_state can be states array only.
    """

    def __init__(
        self,
        env,
        shape_meta: dict,
        init_state: Optional[Union[dict, np.ndarray]] = None,
        render_obs_key: RenderObsKey = "agentview_image",
    ):
        self.env = env
        self.render_obs_key = render_obs_key
        self.init_state = init_state
        self.has_reset_before = False
        self.shape_meta = shape_meta
        self.render_cache = None

        action_shape = shape_meta["action"]["shape"]
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=action_shape, dtype=np.float32
        )

        observation_space = gym.spaces.Dict()
        for key, value in shape_meta["obs"].items():
            shape = value["shape"]
            min_val, max_val = (0, 1) if key.endswith("image") else (-1, 1)
            observation_space[key] = gym.spaces.Box(
                low=min_val, high=max_val, shape=shape, dtype=np.float32
            )
        self.observation_space = observation_space

    def get_observation(self, raw_obs=None):
        if raw_obs is None:
            raw_obs = self.env.get_observation()
        self.render_cache = _resolve_render_obs(raw_obs, self.render_obs_key)
        filtered = {}
        for key, spec in self.shape_meta["obs"].items():
            if key not in raw_obs:
                continue
            val = raw_obs[key].copy()
            if spec.get("type") == "rgb":
                val = val.astype(np.float32) / 255.0
            else:
                val = val.astype(np.float32)
            filtered[key] = val
        return filtered

    def _raw_robosuite_env(self):
        candidate = getattr(self.env, "env", None)
        if candidate is not None and hasattr(candidate, "robots") and hasattr(candidate, "sim"):
            return candidate
        if hasattr(self.env, "robots") and hasattr(self.env, "sim"):
            return self.env
        return None

    @staticmethod
    def _iter_robot_grippers(raw_env):
        for robot in getattr(raw_env, "robots", []):
            gripper = getattr(robot, "gripper", None)
            actuator_idxs = getattr(robot, "_ref_joint_gripper_actuator_indexes", None)
            if gripper is None or actuator_idxs is None:
                continue
            if isinstance(gripper, dict):
                if not isinstance(actuator_idxs, dict):
                    continue
                for arm, arm_gripper in gripper.items():
                    arm_idxs = actuator_idxs.get(arm)
                    if arm_gripper is None or arm_idxs is None:
                        continue
                    yield arm_gripper, np.asarray(arm_idxs, dtype=np.int32).reshape(-1)
            else:
                yield gripper, np.asarray(actuator_idxs, dtype=np.int32).reshape(-1)

    def _clear_hidden_gripper_state(self) -> None:
        raw_env = self._raw_robosuite_env()
        if raw_env is None:
            return
        sim = raw_env.sim
        if hasattr(sim.data, "act") and getattr(sim.data.act, "size", 0) > 0:
            sim.data.act[:] = 0
        for gripper, actuator_idxs in self._iter_robot_grippers(raw_env):
            current_action = np.asarray(gripper.current_action, dtype=np.float64)
            gripper.current_action = np.zeros_like(current_action, dtype=np.float64)
            if actuator_idxs.size > 0:
                sim.data.ctrl[actuator_idxs] = 0.0
        sim.forward()

    def _realign_hidden_arm_controller_state(self) -> None:
        raw_env = self._raw_robosuite_env()
        if raw_env is None:
            return
        sim = raw_env.sim
        for robot in getattr(raw_env, "robots", []):
            root_body = getattr(robot.robot_model, "root_body", None)
            if root_body is not None:
                robot.base_pos = sim.data.get_body_xpos(root_body)
                robot.base_ori = T.mat2quat(
                    sim.data.get_body_xmat(root_body).reshape((3, 3))
                )

            controller = getattr(robot, "controller", None)
            joint_pos_indexes = getattr(robot, "_ref_joint_pos_indexes", None)
            if controller is None or joint_pos_indexes is None:
                continue

            current_joint_pos = np.asarray(
                sim.data.qpos[joint_pos_indexes], dtype=np.float64
            )

            if isinstance(controller, dict):
                split_idx = int(getattr(robot, "_joint_split_idx", 0))
                for arm_name, arm_controller in controller.items():
                    if arm_controller is None:
                        continue
                    if arm_name == "right":
                        arm_joint_pos = current_joint_pos[:split_idx]
                    else:
                        arm_joint_pos = current_joint_pos[split_idx:]
                    arm_controller.update_initial_joints(arm_joint_pos)
                    arm_controller.reset_goal()
                    arm_controller.update_base_pose(robot.base_pos, robot.base_ori)
            else:
                controller.update_initial_joints(current_joint_pos)
                controller.reset_goal()
                controller.update_base_pose(robot.base_pos, robot.base_ori)
        sim.forward()

    def _zero_arm_joint_velocities(self) -> None:
        raw_env = self._raw_robosuite_env()
        if raw_env is None:
            return
        sim = raw_env.sim
        for robot in getattr(raw_env, "robots", []):
            joint_vel_indexes = getattr(robot, "_ref_joint_vel_indexes", None)
            if joint_vel_indexes is None:
                continue
            sim.data.qvel[joint_vel_indexes] = 0.0
        sim.forward()

    def _prime_hidden_gripper_state(self, command_value: float) -> None:
        raw_env = self._raw_robosuite_env()
        if raw_env is None:
            return
        sim = raw_env.sim
        self._clear_hidden_gripper_state()
        for gripper, actuator_idxs in self._iter_robot_grippers(raw_env):
            action_dim = max(1, int(getattr(gripper, "dof", 0)))
            commanded = np.full((action_dim,), float(command_value), dtype=np.float64)
            actual = None
            for _ in range(256):
                actual = np.asarray(gripper.format_action(commanded), dtype=np.float64).reshape(-1)
            if actual is None or actuator_idxs.size == 0:
                continue
            gripper.current_action = actual.copy()
            if actual.size == 1 and actuator_idxs.size > 1:
                actual = np.full((actuator_idxs.size,), float(actual[0]), dtype=np.float64)
            elif actual.size != actuator_idxs.size:
                actual = np.resize(actual, actuator_idxs.size)
            ctrl_range = sim.model.actuator_ctrlrange[actuator_idxs]
            bias = 0.5 * (ctrl_range[:, 1] + ctrl_range[:, 0])
            weight = 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0])
            sim.data.ctrl[actuator_idxs] = bias + weight * actual
        sim.forward()

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        init_from_options = None
        gripper_reset_action = None
        if options is not None and isinstance(options, dict):
            init_from_options = options.get("init_state")
            gripper_reset_action = options.get("gripper_reset_action")
            zero_arm_joint_vel = bool(options.get("zero_arm_joint_vel", False))
        else:
            zero_arm_joint_vel = False

        if init_from_options is not None:
            to_use = init_from_options
        elif self.init_state is not None:
            to_use = self.init_state
        else:
            to_use = None

        if to_use is not None:
            if not self.has_reset_before:
                self.env.reset()
                self.has_reset_before = True
            if isinstance(to_use, dict):
                raw_obs = self.env.reset_to(to_use)
            else:
                raw_obs = self.env.reset_to({"states": to_use})
            if zero_arm_joint_vel:
                self._zero_arm_joint_velocities()
            self._realign_hidden_arm_controller_state()
            self._clear_hidden_gripper_state()
            if gripper_reset_action is not None:
                self._prime_hidden_gripper_state(float(gripper_reset_action))
                raw_obs = self.env.get_observation()
        else:
            raw_obs = self.env.reset()

        if raw_obs is None:
            raw_obs = self.env.get_observation()
        obs = self.get_observation(raw_obs)
        return obs, {}

    def step(self, action):
        # Support batched action (n_envs, 7) from VectorEnv; inner env expects (7,)
        if hasattr(action, "ndim") and action.ndim == 2 and action.shape[0] == 1:
            action = action[0]
        elif hasattr(action, "ndim") and action.ndim == 2:
            action = action[0]  # single env in vector: use first row
        raw_obs, reward, done, info = self.env.step(action)
        obs = self.get_observation(raw_obs)
        return obs, reward, done, False, info

    def render(self, mode="rgb_array"):
        if self.render_cache is None:
            raise RuntimeError("Must run reset or step before render.")
        return self.render_cache.copy()


# Re-use pose helpers from robomimic_runner for policy observation / action_mode
def quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    if quat.ndim == 1:
        return Rotation.from_quat(quat).as_rotvec()
    return Rotation.from_quat(quat).as_rotvec()


def axis_angle_to_quat(axis_angle: np.ndarray) -> np.ndarray:
    if axis_angle.ndim == 1:
        return Rotation.from_rotvec(axis_angle).as_quat()
    return Rotation.from_rotvec(axis_angle).as_quat()


def apply_vel_to_pose(
    pose_pos: np.ndarray,
    pose_rot: np.ndarray,
    vel_lin: np.ndarray,
    vel_ang: np.ndarray,
    dt: float,
    rot_format: Literal["quat", "axis_angle"] = "quat",
) -> tuple:
    next_pos = pose_pos + vel_lin * dt
    if rot_format == "axis_angle":
        pose_rot_quat = axis_angle_to_quat(pose_rot)
    else:
        pose_rot_quat = pose_rot
    vel_ang_quat = axis_angle_to_quat(vel_ang * dt)
    r_current = Rotation.from_quat(pose_rot_quat)
    r_vel = Rotation.from_quat(vel_ang_quat)
    next_rot_quat = (r_vel * r_current).as_quat()
    if rot_format == "axis_angle":
        next_rot = quat_to_axis_angle(next_rot_quat)
    else:
        next_rot = next_rot_quat
    return next_pos, next_rot


@dataclass
class MimicgenRunnerCfg(BaseRunnerCfg):
    env_name: Literal["mimicgen"]

    dataset_path: str = ""
    max_episode_steps: int = 400
    success_reward_threshold: Optional[float] = 0.9
    relaxed_stack_three_success: bool = True
    post_success_steps: int = 10
    num_envs: int = 1  # number of vectorized envs (batch dim in obs/action)
    num_demos_to_run: int = 3  # number of demos to evaluate per run() call
    render_size: int = 252
    n_repeat: int = 1
    action_scale: float = 1.0
    output_dir: str = "outputs/mimicgen_eval"
    save_videos: bool = True
    save_trajectory: bool = True
    save_rrd: bool = True
    video_fps: int = 10

    render_obs_key: RenderObsKey = "agentview_image"
    use_stored_model: bool = True
    texture_overrides: Optional[dict] = None

    action_mode: Literal["velocity", "absolute"] = "velocity"
    pose_format: Literal["quat", "axis_angle"] = "quat"
    dt: float = 0.1

    # Warmup: replay this many demo actions at the start of each episode to
    # build real temporal context for the policy (avoids filling the obs queue
    # with N copies of frame 0).  Set to context_frames - 1 for full context.
    demo_warmup_steps: int = 0
    # Advance the simulator for this many steps after reset / warmup before the
    # counted rollout begins. These steps do not consume the policy horizon.
    settle_reset_steps: int = 0
    # Optional fixed gripper command to use during reset settling. Arm channels
    # remain zero. When None, the full settle action is zero.
    settle_reset_gripper_action: float | None = None
    # If True, zero arm joint velocities before settle_reset_steps begin so the
    # arm does not drift just because the dataset frame was mid-motion.
    settle_reset_zero_arm_joint_vel: bool = False
    # Feed settle observations into the policy warmup queue so the planner sees
    # the settled context before the first counted action.
    settle_reset_feed_policy_context: bool = True
    log_step_debug: bool = True


def _normalize_obs_for_policy(obs: dict, image_keys: list) -> dict:
    """Convert obs to float32; scale image keys to [0, 1]."""
    out = {}
    for k, v in obs.items():
        v = np.asarray(v, dtype=np.float32)
        if k in image_keys:
            if v.dtype == np.uint8:
                v = v.astype(np.float32) / 255.0
        out[k] = v
    return out


class MimicgenRunner(BaseRunner):
    """Runner for MimicGen playback env: one env, reset to demo initial_state per episode."""

    cfg: MimicgenRunnerCfg

    def __init__(
        self,
        cfg: MimicgenRunnerCfg,
        device: Optional[torch.device] = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda:0")
        super().__init__(cfg, device)

    def _step_debug_enabled(self) -> bool:
        return bool(getattr(self.cfg, "log_step_debug", True))

    def _step_debug_write(self, message: str) -> None:
        if self._step_debug_enabled():
            tqdm.tqdm.write(message)

    @staticmethod
    def _sanitize_run_tag(tag: str) -> str:
        safe = []
        for ch in tag.strip():
            if ch.isalnum() or ch in ("-", "_", "."):
                safe.append(ch)
            else:
                safe.append("_")
        return "".join(safe).strip("._-") or "run"

    def _infer_obs_and_camera_from_dataset(self, dataset_path: str) -> tuple:
        """Infer obs keys, modality mapping, camera names, image size, and shape_meta from HDF5."""
        with h5py.File(dataset_path, "r") as f:
            first_ep = "demo_0" if "demo_0" in f["data"] else list(f["data"].keys())[0]
            ep_group = f["data"][first_ep]
            obs_keys = list(ep_group["obs"].keys()) if "obs" in ep_group else []
            camera_names = [
                k.replace("_image", "") for k in obs_keys if k.endswith("_image")
            ]
            if not camera_names:
                camera_names = ["agentview", "robot0_eye_in_hand"]
            first_img_key = next((k for k in obs_keys if k.endswith("_image")), None)
            if first_img_key is not None:
                img_shape = ep_group["obs"][first_img_key].shape
                camera_height = int(img_shape[1])
                camera_width = int(img_shape[2])
            else:
                camera_height = camera_width = self.cfg.render_size

            # Build shape_meta for MimicGenImageWrapper (single-step obs and action shapes)
            obs_shapes = {}
            for k in obs_keys:
                arr = ep_group["obs"][k]
                shp = arr.shape
                if len(shp) > 1:
                    one_step = tuple(int(x) for x in shp[1:])
                else:
                    one_step = (int(shp[0]),)
                obs_type = "rgb" if k.endswith("_image") else "low_dim"
                obs_shapes[k] = {"shape": list(one_step), "type": obs_type}
            act_arr = ep_group["actions"]
            action_shape = (
                list(act_arr.shape[1:]) if act_arr.ndim > 1 else [int(act_arr.shape[0])]
            )
            shape_meta = {"obs": obs_shapes, "action": {"shape": action_shape}}

        modality_mapping = collections.defaultdict(list)
        for k in obs_keys:
            modality_mapping["rgb" if k.endswith("_image") else "low_dim"].append(k)
        return (
            obs_keys,
            modality_mapping,
            camera_names,
            camera_height,
            camera_width,
            shape_meta,
        )

    def setup_env(self) -> None:
        if mimicgen_create_env is None:
            raise ImportError(
                "mimicgen.utils.robomimic_utils.create_env not available. "
                "Add third_party/mimicgen to PYTHONPATH."
            )
        dataset_path = Path(self.cfg.dataset_path).expanduser()
        env_meta = get_env_metadata_from_dataset(str(dataset_path))
        (
            obs_keys,
            modality_mapping,
            camera_names,
            camera_height,
            camera_width,
            shape_meta,
        ) = self._infer_obs_and_camera_from_dataset(str(dataset_path))

        # Override image size for rendering using cfg.render_size
        render_res = self.cfg.render_size
        if render_res != camera_height or render_res != camera_width:
            print(
                f"[MimicgenRunner] Overriding image size to {render_res}x{render_res} (dataset was {camera_height}x{camera_width})"
            )
            env_meta["env_kwargs"]["camera_heights"] = render_res
            env_meta["env_kwargs"]["camera_widths"] = render_res
            for _obs_key, spec in shape_meta["obs"].items():
                if spec.get("type") == "rgb":
                    # Support both (H, W, C) and (C, H, W) from dataset
                    shp = spec["shape"]
                    if len(shp) == 3 and shp[0] == 3:
                        spec["shape"] = [3, render_res, render_res]
                    else:
                        spec["shape"] = [render_res, render_res, 3]
        camera_height = render_res
        camera_width = render_res

        ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

        def env_fn():
            raw_env = mimicgen_create_env(
                env_meta=env_meta,
                camera_names=camera_names,
                camera_height=camera_height,
                camera_width=camera_width,
                render=False,
                render_offscreen=True,
                use_image_obs=True,
            )
            wrapped = MimicGenImageWrapper(
                env=raw_env,
                shape_meta=shape_meta,
                init_state=None,
                render_obs_key=self.cfg.render_obs_key,
            )
            wrapped = gym.wrappers.PassiveEnvChecker(wrapped)
            wrapped = gym.wrappers.OrderEnforcing(wrapped)
            wrapped = gym.wrappers.TimeLimit(
                wrapped, max_episode_steps=self.cfg.max_episode_steps
            )
            return wrapped

        num_envs = max(1, int(self.cfg.num_envs))
        self.env = gym.vector.SyncVectorEnv([env_fn for _ in range(num_envs)])
        self.env_meta = env_meta
        self._num_envs = num_envs
        self.dataset_path = str(dataset_path)
        self.obs_keys = obs_keys
        self.image_keys = [k for k in obs_keys if k.endswith("_image")]

    def _pad_frames_to_target(self, frames, target_h, target_w, dtype=np.uint8):
        out = []
        for f in frames:
            h, w = f.shape[:2]
            if h == target_h and w == target_w:
                out.append(f)
            else:
                pad = np.zeros((target_h, target_w, f.shape[2]), dtype=dtype)
                pad[:h, :w] = f
                out.append(pad)
        return out

    def _policy_rgb_from_obs(self, obs: dict[str, Any]) -> np.ndarray:
        return _resolve_render_obs(
            obs,
            self.cfg.render_obs_key,
            fallback_image_keys=self.image_keys,
        )

    @staticmethod
    def _frame_width(frame: Any) -> int:
        value = np.asarray(frame)
        if value.ndim == 4 and value.shape[0] == 1:
            value = value[0]
        return int(value.shape[1])

    def _policy_view_metadata(
        self,
        obs: dict[str, Any],
    ) -> tuple[list[str], list[int], str]:
        if isinstance(self.cfg.render_obs_key, str):
            key = self.cfg.render_obs_key
            if key not in obs:
                if self.image_keys and self.image_keys[0] in obs:
                    key = self.image_keys[0]
                else:
                    raise KeyError(f"Render observation key not found: {key}")
            return [key], [self._frame_width(obs[key])], key

        view_keys = list(self.cfg.render_obs_key)
        if not view_keys:
            raise ValueError("render_obs_key list must contain at least one key")
        for key in view_keys:
            if key not in obs:
                raise KeyError(
                    f"Render observation key not found for multiview path: {key}"
                )
        view_widths = [self._frame_width(obs[key]) for key in view_keys]
        return view_keys, view_widths, "|".join(view_keys)

    @staticmethod
    def _batched_obs_value(obs: dict[str, Any], key: str) -> np.ndarray | None:
        value = obs.get(key)
        if value is None:
            return None
        value_np = np.asarray(value)
        if value_np.ndim == 1:
            value_np = np.expand_dims(value_np, axis=0)
        return value_np

    def _policy_state_observation(
        self,
        obs: dict[str, Any],
        *,
        step_index: int | None = 0,
    ) -> PolicyObservation:
        rgb = self._policy_rgb_from_obs(obs)
        if rgb.ndim == 3:
            rgb = np.expand_dims(rgb, axis=0)
        view_keys, view_widths, concat_rgb_key = self._policy_view_metadata(obs)
        return PolicyObservation(
            rgb=rgb,
            q_robot=None,
            view_keys=view_keys,
            view_widths=view_widths,
            concat_rgb_key=concat_rgb_key,
            step_index=step_index,
            eef_pos=self._batched_obs_value(obs, "robot0_eef_pos"),
            eef_quat=self._batched_obs_value(obs, "robot0_eef_quat"),
            gripper_qpos=self._batched_obs_value(obs, "robot0_gripper_qpos"),
            dt=self.cfg.dt,
            action_mode=self.cfg.action_mode,
            pose_format=self.cfg.pose_format,
        )

    def _sync_policy_reset_state(
        self,
        policy: BasePolicy,
        obs: dict[str, Any],
        *,
        reset_alignment: bool = True,
    ) -> None:
        sync_method = getattr(policy, "sync_gripper_runtime_from_obs", None)
        if callable(sync_method):
            sync_method(
                self._policy_state_observation(obs, step_index=0),
                reset_alignment=reset_alignment,
            )

    def _policy_warmup_obs_if_supported(
        self,
        policy: BasePolicy,
        obs: dict[str, Any],
        *,
        step_index: int | None = None,
    ) -> None:
        warmup_method = getattr(policy, "warmup_obs", None)
        if callable(warmup_method):
            warmup_method(self._policy_state_observation(obs, step_index=step_index))

    def _settle_reset_action(self) -> np.ndarray:
        single_action_space = getattr(
            self.env, "single_action_space", self.env.action_space
        )
        action_shape = tuple(int(dim) for dim in single_action_space.shape)
        action = np.zeros(
            (max(1, int(getattr(self, "_num_envs", 1))), *action_shape),
            dtype=np.float32,
        )
        if self.cfg.settle_reset_gripper_action is not None and action_shape:
            action[..., -1] = float(self.cfg.settle_reset_gripper_action)
        return action

    def _run_settle_phase(
        self,
        env,
        policy: BasePolicy,
        obs: dict[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        settle_steps = max(int(self.cfg.settle_reset_steps), 0)
        if settle_steps <= 0:
            return obs, False

        settle_action = self._settle_reset_action()
        done = False
        for settle_idx in range(settle_steps):
            obs, reward, terminated, truncated, info = env.step(settle_action)
            del reward, info
            done = bool(np.any(terminated) or np.any(truncated))
            obs = _normalize_obs_for_policy(obs, self.image_keys)
            if self.cfg.settle_reset_feed_policy_context:
                self._policy_warmup_obs_if_supported(
                    policy,
                    obs,
                    step_index=-(settle_steps - settle_idx),
                )
            if done:
                break

        self._sync_policy_reset_state(policy, obs, reset_alignment=True)
        self._step_debug_write(
            f"[MimicgenRunner] Settled reset for {settle_steps} steps"
            + (
                f" with gripper={float(self.cfg.settle_reset_gripper_action):+.3f}"
                if self.cfg.settle_reset_gripper_action is not None
                else ""
            )
        )
        return obs, done

    @staticmethod
    def _storyboard_frame_to_uint8(frame: Any) -> np.ndarray:
        value = (
            frame.detach().cpu().numpy()
            if isinstance(frame, torch.Tensor)
            else np.asarray(frame)
        )
        if value.ndim == 4 and value.shape[0] == 1:
            value = value[0]
        if value.ndim == 3 and value.shape[0] in (1, 3):
            value = np.transpose(value, (1, 2, 0))
        if value.ndim != 3:
            raise ValueError(
                f"Expected image-like frame for storyboard, got shape {value.shape}"
            )
        if value.dtype != np.uint8:
            if np.issubdtype(value.dtype, np.floating):
                value = (value * 255.0).clip(0, 255).astype(np.uint8)
            else:
                value = value.clip(0, 255).astype(np.uint8)
        if value.shape[-1] == 1:
            value = np.repeat(value, 3, axis=-1)
        return value

    def _make_storyboard_frame(
        self,
        frame: Any,
        *,
        target_shape: tuple[int, int],
        title: str,
        subtitle: str | None = None,
    ) -> np.ndarray:
        image = self._storyboard_frame_to_uint8(frame)
        target_h, target_w = int(target_shape[0]), int(target_shape[1])
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        header_h = min(max(52, target_h // 8), 72)
        canvas[:header_h] = np.array([24, 24, 24], dtype=np.uint8)

        fit_h = max(1, target_h - header_h - 12)
        fit_w = max(1, target_w - 12)
        scale = min(
            fit_h / max(int(image.shape[0]), 1),
            fit_w / max(int(image.shape[1]), 1),
        )
        resized_h = max(1, int(round(image.shape[0] * scale)))
        resized_w = max(1, int(round(image.shape[1] * scale)))
        resized = media.resize_image(image, shape=(resized_h, resized_w))
        y0 = header_h + max((fit_h - resized_h) // 2, 0)
        x0 = max((target_w - resized_w) // 2, 0)
        canvas[y0 : y0 + resized_h, x0 : x0 + resized_w] = resized

        cv2.putText(
            canvas,
            title,
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if subtitle:
            cv2.putText(
                canvas,
                subtitle,
                (10, header_h - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (180, 220, 255),
                1,
                cv2.LINE_AA,
            )
        return canvas

    def _append_storyboard_sequence(
        self,
        videos_demo: dict[str, list[np.ndarray]],
        frames: list[Any],
        *,
        target_shape: tuple[int, int],
        title: str,
    ) -> None:
        total = len(frames)
        for idx, frame in enumerate(frames):
            rendered = self._make_storyboard_frame(
                frame,
                target_shape=target_shape,
                title=title,
                subtitle=f"frame {idx + 1}/{total}",
            )
            videos_demo["policy_story"].append(self._frame_to_batched_hwc(rendered))

    @staticmethod
    def _compact_storyboard_shape(shape: tuple[int, int]) -> tuple[int, int]:
        height, width = int(shape[0]), int(shape[1])
        compact_width = min(
            width,
            max(int(round(height * 3.0)), int(round(width * 0.72))),
        )
        compact_width = max(compact_width, min(width, 480))
        return height, compact_width

    def _save_video(self, frames, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        # Accept (1, H, W, C) from videos_demo; squeeze to (H, W, C) for writing
        if frames and frames[0].ndim == 4 and frames[0].shape[0] == 1:
            frames = [f[0] for f in frames]
        frames = [np.transpose(f, (1, 2, 0)) if f.shape[0] == 3 else f for f in frames]
        target_h = max(f.shape[0] for f in frames)
        target_w = max(f.shape[1] for f in frames)
        frames = self._pad_frames_to_target(
            frames, target_h, target_w, dtype=frames[0].dtype
        )
        media.write_video(path, frames, fps=self.cfg.video_fps)

    @staticmethod
    def _frame_to_batched_hwc(frame: np.ndarray) -> np.ndarray:
        """Convert a single frame to (1, H, W, C) for compatibility with RobomimicRunner video format.

        Always copies: recorded frames previously aliased live policy/env buffers, so a
        later in-place write (e.g. the desired-flow visualization rendered during the next
        replan) retroactively replaced already-recorded video frames with flow colorwheels.
        """
        f = np.array(frame, copy=True)
        if f.ndim == 3:
            if f.shape[0] == 3:  # CHW
                f = np.transpose(f, (1, 2, 0))
            return f[np.newaxis, ...]  # (1, H, W, C)
        return f

    @staticmethod
    def _reward_to_float(reward: Any) -> float:
        return (
            float(np.asarray(reward).flat[0])
            if np.isscalar(reward) is False
            else float(reward)
        )

    def _append_logged_step(
        self,
        *,
        traj: dict[str, list[np.ndarray]],
        videos_demo: dict[str, list[np.ndarray]],
        obs_rgb: np.ndarray | None,
        obs: dict[str, Any],
        action: np.ndarray,
        reward: float,
        timestep: int,
        policy_vis: Any | None = None,
        policy_out: Any | None = None,
    ) -> None:
        if obs_rgb is not None:
            traj["obs"].append(obs_rgb[0].copy())
        gripper_qpos_curr = obs.get("robot0_gripper_qpos")
        if gripper_qpos_curr is not None:
            gripper_qpos_curr = np.asarray(gripper_qpos_curr)
            if gripper_qpos_curr.ndim > 1:
                gripper_qpos_curr = gripper_qpos_curr[0]
            traj["robot0_gripper_qpos"].append(
                np.asarray(gripper_qpos_curr, dtype=np.float32)[None, ...]
            )
        gripper_qvel_curr = obs.get("robot0_gripper_qvel")
        if gripper_qvel_curr is not None:
            gripper_qvel_curr = np.asarray(gripper_qvel_curr)
            if gripper_qvel_curr.ndim > 1:
                gripper_qvel_curr = gripper_qvel_curr[0]
            traj["robot0_gripper_qvel"].append(
                np.asarray(gripper_qvel_curr, dtype=np.float32)[None, ...]
            )
        eef_pos_curr = obs.get("robot0_eef_pos")
        if eef_pos_curr is not None:
            eef_pos_curr = np.asarray(eef_pos_curr)
            if eef_pos_curr.ndim > 1:
                eef_pos_curr = eef_pos_curr[0]
            traj["robot0_eef_pos"].append(
                np.asarray(eef_pos_curr, dtype=np.float32)[None, ...]
            )
        eef_quat_curr = obs.get("robot0_eef_quat")
        if eef_quat_curr is not None:
            eef_quat_curr = np.asarray(eef_quat_curr)
            if eef_quat_curr.ndim > 1:
                eef_quat_curr = eef_quat_curr[0]
            traj["robot0_eef_quat"].append(
                np.asarray(eef_quat_curr, dtype=np.float32)[None, ...]
            )
        traj["action"].append(np.asarray(action, dtype=np.float32).copy())
        traj["reward"].append(np.array([float(reward)], dtype=np.float32))
        traj["timestep"].append(np.array([int(timestep)], dtype=np.int32))
        # Per-step raw artifacts (jacobian / flow / dream / tracks) when
        # MotionPolicyCfg.save_artifacts is on. trajectory.npz becomes a
        # complete replay artifact for paper figures.
        _raw = None
        if policy_out is not None and getattr(policy_out, "info", None):
            _raw = policy_out.info.get("raw_artifacts")
        if _raw is not None:
            for _k, _v in _raw.items():
                traj.setdefault(f"raw__{_k}", []).append(np.asarray(_v))
                traj.setdefault(f"raw__{_k}__step", []).append(np.int32(timestep))
        if obs_rgb is not None:
            videos_demo["obs"].append(self._frame_to_batched_hwc(obs_rgb[0]))
        vis_frame = None
        if policy_vis is not None:
            vis_frame = np.asarray(policy_vis)
            if vis_frame.ndim == 4:
                vis_frame = vis_frame[0]
        elif obs_rgb is not None:
            vis_frame = obs_rgb[0]
        if vis_frame is not None:
            if vis_frame.dtype != np.uint8:
                vis_frame = (vis_frame * 255.0).clip(0, 255).astype(np.uint8)
            videos_demo["policy"].append(self._frame_to_batched_hwc(vis_frame))
        for img_key in self.image_keys:
            if img_key in obs:
                v = obs[img_key]
                if np.isscalar(v) or (hasattr(v, "ndim") and v.ndim < 2):
                    continue
                v = np.asarray(v)
                if v.ndim == 4:
                    v = v[0]
                videos_demo[img_key].append(self._frame_to_batched_hwc(v))

    def _append_story_execution_frame(
        self,
        *,
        videos_demo: dict[str, list[np.ndarray]],
        exec_rgb: np.ndarray,
        active_story_shape: tuple[int, int] | None,
        active_story_chunk_index: int | None,
        subtitle: str | None,
    ) -> None:
        if active_story_shape is None:
            return
        exec_title = "execution"
        if active_story_chunk_index is not None:
            exec_title = f"execution: chunk {active_story_chunk_index}"
        exec_frame = self._make_storyboard_frame(
            exec_rgb,
            target_shape=active_story_shape,
            title=exec_title,
            subtitle=subtitle,
        )
        videos_demo["policy_story"].append(self._frame_to_batched_hwc(exec_frame))

    def _run_post_success_tail(
        self,
        *,
        env,
        obs: dict[str, Any],
        action: np.ndarray,
        traj: dict[str, list[np.ndarray]],
        videos_demo: dict[str, list[np.ndarray]],
        timestep_start: int,
        active_story_shape: tuple[int, int] | None,
        active_story_chunk_index: int | None,
    ) -> tuple[dict[str, Any], float, float, bool, bool]:
        tail_steps = max(int(self.cfg.post_success_steps), 0)
        if tail_steps <= 0:
            return obs, 0.0, -np.inf, False, False

        tail_reward_sum = 0.0
        tail_max_reward = -np.inf
        done = False
        tail_env_success = False
        settle_action = np.asarray(action, dtype=np.float32)
        for tail_idx in range(tail_steps):
            obs, reward, terminated, truncated, _info = env.step(settle_action)
            done = bool(np.any(terminated) or np.any(truncated))
            obs = _normalize_obs_for_policy(obs, self.image_keys)
            reward_f = self._reward_to_float(reward)
            tail_reward_sum += reward_f
            tail_max_reward = max(tail_max_reward, reward_f)
            success_threshold = self.cfg.success_reward_threshold
            if success_threshold is not None and reward_f >= float(success_threshold):
                tail_env_success = True

            feedback_rgb = self._policy_rgb_from_obs(obs)
            if feedback_rgb.ndim == 3:
                feedback_rgb = np.expand_dims(feedback_rgb, axis=0)
            self._append_logged_step(
                traj=traj,
                videos_demo=videos_demo,
                obs_rgb=feedback_rgb,
                obs=obs,
                action=settle_action,
                reward=reward_f,
                timestep=timestep_start + tail_idx,
                policy_vis=None,
            )
            self._append_story_execution_frame(
                videos_demo=videos_demo,
                exec_rgb=feedback_rgb[0],
                active_story_shape=active_story_shape,
                active_story_chunk_index=active_story_chunk_index,
                subtitle=f"post-success {tail_idx + 1}/{tail_steps}",
            )
            self._step_debug_write(
                f"[MimicgenRunner] Post-success tail step {tail_idx + 1}/{tail_steps} "
                f"reward={reward_f:.4f} done={done}"
            )
            if done:
                break
        return obs, tail_reward_sum, tail_max_reward, done, tail_env_success

    @staticmethod
    def _initial_state_template(
        init_state: Optional[Union[dict, np.ndarray]],
    ) -> Optional[dict[str, Any]]:
        if not isinstance(init_state, dict):
            return None
        template = {k: v for k, v in init_state.items() if k != "states"}
        return template or None

    @staticmethod
    def _compose_initial_state(
        state: Union[dict, np.ndarray],
        template: Optional[dict[str, Any]],
    ) -> Union[dict, np.ndarray]:
        if isinstance(state, dict):
            if template is None:
                return dict(state)
            merged = dict(template)
            merged.update(state)
            return merged
        if template is None:
            return state
        merged = dict(template)
        merged["states"] = state
        return merged

    def _custom_init_state_template(
        self,
        options: Optional[dict],
        *,
        h5file: h5py.File,
    ) -> Optional[dict[str, Any]]:
        if not isinstance(options, dict):
            return None

        init_state = options.get("init_state")
        template = self._initial_state_template(init_state)
        if template is not None:
            return template

        has_custom_warmup = (
            options.get("warmup_states") is not None and "warmup_states" in options
        )

        demo_key = options.get("demo_key")
        if demo_key is None and options.get("demo_idx") is not None:
            demo_key = f"demo_{int(options['demo_idx'])}"
        if has_custom_warmup and (
            not isinstance(demo_key, str) or demo_key not in h5file["data"]
        ):
            raise ValueError(
                "Custom warmup_states with a bare init_state require either "
                "a full init_state dict (including model / ep_meta) or a "
                "matching demo_key / demo_idx so MimicgenRunner can restore "
                "the correct reset template."
            )
        if not isinstance(demo_key, str) or demo_key not in h5file["data"]:
            return None

        demo_group = h5file["data"][demo_key]
        full_init_state = build_initial_state(
            demo_group,
            use_stored_model=self.cfg.use_stored_model,
            texture_overrides=self.cfg.texture_overrides,
            h5file=h5file,
            demo_key=demo_key,
        )
        return self._initial_state_template(full_init_state)

    @staticmethod
    def _iter_wrapped_env_chain(env: Any):
        current = env
        while current is not None:
            yield current
            current = getattr(current, "env", None)

    def _sync_runtime_episode_horizon(self, *, extra_steps: int = 0) -> None:
        max_episode_steps = int(self.cfg.max_episode_steps) + max(int(extra_steps), 0)
        envs = getattr(self.env, "envs", None)
        if not envs:
            return
        for env in envs:
            for wrapped in self._iter_wrapped_env_chain(env):
                if isinstance(wrapped, gym.wrappers.TimeLimit):
                    wrapped._max_episode_steps = max_episode_steps
                    spec = getattr(wrapped, "spec", None)
                    if spec is not None:
                        spec.max_episode_steps = max_episode_steps

    def _raw_robosuite_env_for_success_checks(self):
        envs = getattr(self.env, "envs", None)
        if not envs:
            return None
        wrapped_env = envs[0]
        for wrapped in self._iter_wrapped_env_chain(wrapped_env):
            if isinstance(wrapped, MimicGenImageWrapper):
                return wrapped._raw_robosuite_env()
        return None

    def _stack_three_relaxed_success(self) -> tuple[bool, dict[str, Any]]:
        if not bool(getattr(self.cfg, "relaxed_stack_three_success", True)):
            return False, {}
        raw_env = self._raw_robosuite_env_for_success_checks()
        required = (
            "cubeA",
            "cubeC",
            "_check_cubeA_stacked",
            "_check_cubeC_lifted",
            "check_contact",
        )
        if raw_env is None or not all(hasattr(raw_env, attr) for attr in required):
            return False, {}
        try:
            cubeA_stacked = bool(raw_env._check_cubeA_stacked())
            cubeC_lifted = bool(raw_env._check_cubeC_lifted())
            cubeC_touching_cubeA = bool(raw_env.check_contact(raw_env.cubeC, raw_env.cubeA))
            grasping_cubeC = False
            if hasattr(raw_env, "_check_grasp") and getattr(raw_env, "robots", None):
                grasping_cubeC = bool(
                    raw_env._check_grasp(
                        gripper=raw_env.robots[0].gripper,
                        object_geoms=raw_env.cubeC,
                    )
                )
            relaxed_success = cubeA_stacked and cubeC_lifted and cubeC_touching_cubeA
            return relaxed_success, {
                "cubeA_stacked": cubeA_stacked,
                "cubeC_lifted": cubeC_lifted,
                "cubeC_touching_cubeA": cubeC_touching_cubeA,
                "grasping_cubeC": grasping_cubeC,
            }
        except Exception:
            return False, {}

    def run(
        self,
        policy: BasePolicy,
        options: Optional[dict] = None,
        run_tag: Optional[str] = None,
    ) -> dict:
        """
        Run policy on one or more MimicGen demos. Each demo: reset_to(initial_state), then step with policy.

        options:
            - init_state: single state (array or dict) to run one episode from (like robomimic)
            - demo_indices: list of int (e.g. [0,1,2]) to run those demo indices
            - demo_keys: list of str (e.g. ["demo_0","demo_1"]) to run those keys
            - If None: run first num_demos_to_run demos.
        """
        self._sync_runtime_episode_horizon(
            extra_steps=int(self.cfg.settle_reset_steps)
            + max(int(self.cfg.post_success_steps), 0)
        )
        env = self.env
        # max_steps = number of policy calls (progress bar total); total env steps <= max_episode_steps
        max_steps = self.cfg.max_episode_steps // self.cfg.n_repeat

        with h5py.File(self.dataset_path, "r") as h5file:
            demo_keys = sorted(
                [k for k in h5file["data"].keys() if k.startswith("demo_")]
            )
            if not demo_keys:
                demo_keys = sorted(h5file["data"].keys())

            if options is not None and "init_state" in options:
                run_keys = ["_custom_"]
            elif options is not None and "demo_keys" in options:
                run_keys = [k for k in options["demo_keys"] if k in h5file["data"]]
            elif options is not None and "demo_indices" in options:
                run_keys = [
                    demo_keys[i]
                    for i in options["demo_indices"]
                    if 0 <= i < len(demo_keys)
                ]
            else:
                n = min(self.cfg.num_demos_to_run, len(demo_keys))
                run_keys = demo_keys[:n]

        if not run_keys:
            return {
                "demo_returns": np.array([], dtype=np.float32),
                "videos": {},
                "max_rewards": np.array([]),
                "max_reward_mean": 0.0,
                "save_dir": None,
            }

        n_demos = len(run_keys)
        episode_rewards = np.zeros(n_demos, dtype=np.float32)
        max_rewards = np.full(n_demos, -np.inf, dtype=np.float32)
        env_successes = np.zeros(n_demos, dtype=bool)
        relaxed_successes = np.zeros(n_demos, dtype=bool)
        all_videos = []  # list of dicts per demo: {"obs": [...], "policy": [...], ...}
        all_trajs = []

        preserve_policy_warm_start_state = bool(
            options is not None
            and isinstance(options, dict)
            and (
                options.get("preserve_policy_warm_start_state", False)
                or options.get("preserve_policy_state", False)
                or options.get("preserve_adaptive_controller", False)
            )
        )
        warm_start_state = (
            policy.get_warm_start_state() if preserve_policy_warm_start_state else None
        )
        policy.reset()
        if warm_start_state is not None:
            policy.set_warm_start_state(warm_start_state)

        custom_state_template: Optional[dict[str, Any]] = None
        if (
            options is not None
            and isinstance(options, dict)
            and "init_state" in options
            and run_keys == ["_custom_"]
        ):
            with h5py.File(self.dataset_path, "r") as h5file:
                custom_state_template = self._custom_init_state_template(
                    options,
                    h5file=h5file,
                )

        # Use outer progress bar only when multiple demos; inner Rollout bar is primary for display.
        demo_iterator = (
            tqdm.tqdm(run_keys, desc="[MimicgenRunner] Demos", leave=True)
            if n_demos > 1
            else run_keys
        )
        for demo_idx, demo_key in enumerate(demo_iterator):
            if (
                demo_key == "_custom_"
                and options is not None
                and "init_state" in options
            ):
                initial_state = self._compose_initial_state(
                    options["init_state"],
                    custom_state_template,
                )
            else:
                with h5py.File(self.dataset_path, "r") as h5file:
                    demo_group = h5file["data"][demo_key]
                    initial_state = build_initial_state(
                        demo_group,
                        use_stored_model=self.cfg.use_stored_model,
                        texture_overrides=self.cfg.texture_overrides,
                        h5file=h5file,
                        demo_key=demo_key,
                    )
            if not self.cfg.use_stored_model and isinstance(initial_state, dict):
                initial_state.pop("model", None)

            reset_options: dict[str, Any] = {"init_state": initial_state}
            if self.cfg.settle_reset_gripper_action is not None:
                reset_options["gripper_reset_action"] = float(
                    self.cfg.settle_reset_gripper_action
                )
            if self.cfg.settle_reset_zero_arm_joint_vel:
                reset_options["zero_arm_joint_vel"] = True
            obs, _ = env.reset(options=reset_options)
            obs = _normalize_obs_for_policy(obs, self.image_keys)
            self._sync_policy_reset_state(policy, obs)

            # Warmup: render consecutive demo states to fill the policy obs
            # queue with real temporal context (instead of N copies of frame 0).
            # Uses reset-to-state (not env.step) so TimeLimit is not consumed.
            #
            # For custom init states, pass options["warmup_states"] — a list/array
            # of MuJoCo states preceding init_state (e.g. frames leading up to
            # frame_idx). The last warmup state should be one step before init_state.
            n_warmup = self.cfg.demo_warmup_steps
            warmup_states = None
            if n_warmup > 0:
                if demo_key != "_custom_":
                    with h5py.File(self.dataset_path, "r") as h5file:
                        warmup_states = np.array(
                            h5file["data"][demo_key]["states"][: n_warmup + 1]
                        )
                elif options is not None and "warmup_states" in options:
                    warmup_states = np.asarray(options["warmup_states"])

            if warmup_states is not None and len(warmup_states) > 0:
                actual_warmup = min(n_warmup, len(warmup_states) - 1)
                for wi in range(actual_warmup):
                    warmup_init_state = self._compose_initial_state(
                        warmup_states[wi],
                        custom_state_template if demo_key == "_custom_" else None,
                    )
                    st_obs, _ = env.reset(options={"init_state": warmup_init_state})
                    st_obs = _normalize_obs_for_policy(st_obs, self.image_keys)
                    w_rgb = self._policy_rgb_from_obs(st_obs)
                    if w_rgb.ndim == 3:
                        w_rgb = np.expand_dims(w_rgb, axis=0)
                    view_keys, view_widths, concat_rgb_key = self._policy_view_metadata(
                        st_obs
                    )
                    w_obs = PolicyObservation(
                        rgb=w_rgb,
                        q_robot=None,
                        view_keys=view_keys,
                        view_widths=view_widths,
                        concat_rgb_key=concat_rgb_key,
                        step_index=-(actual_warmup - wi),
                    )
                    policy.warmup_obs(w_obs)
                final_warmup_init_state = self._compose_initial_state(
                    warmup_states[actual_warmup],
                    custom_state_template if demo_key == "_custom_" else None,
                )
                final_reset_options: dict[str, Any] = {
                    "init_state": final_warmup_init_state
                }
                if self.cfg.settle_reset_gripper_action is not None:
                    final_reset_options["gripper_reset_action"] = float(
                        self.cfg.settle_reset_gripper_action
                    )
                if self.cfg.settle_reset_zero_arm_joint_vel:
                    final_reset_options["zero_arm_joint_vel"] = True
                obs, _ = env.reset(options=final_reset_options)
                obs = _normalize_obs_for_policy(obs, self.image_keys)
                self._sync_policy_reset_state(policy, obs)
                self._step_debug_write(
                    f"[MimicgenRunner] Warmed up context with {actual_warmup} "
                    f"demo frames for {demo_key}"
                )

            obs, settle_done = self._run_settle_phase(env, policy, obs)

            ep_rew = 0.0
            ep_max_rew = -np.inf
            done = bool(settle_done)
            env_success_latched = False
            relaxed_success_latched = False
            videos_demo = {"obs": [], "policy": [], "policy_story": []}
            for k in self.image_keys:
                videos_demo[k] = []
            traj = {
                "timestep": [],
                "obs": [],
                "action": [],
                "reward": [],
                "robot0_gripper_qpos": [],
                "robot0_gripper_qvel": [],
                "robot0_eef_pos": [],
                "robot0_eef_quat": [],
            }
            active_story_shape: tuple[int, int] | None = None
            active_story_chunk_index: int | None = None
            active_story_exec_horizon = 0
            active_story_exec_step = 0

            pbar = tqdm.tqdm(
                range(max_steps),
                desc="[MimicgenRunner] Rollout",
                leave=(n_demos == 1),
                disable=False,
            )
            for step_cnt in pbar:
                if done:
                    break
                rgb = self._policy_rgb_from_obs(obs)
                if rgb.ndim == 3:
                    rgb = np.expand_dims(rgb, axis=0)
                eef_pos = obs.get("robot0_eef_pos")
                if eef_pos is not None and eef_pos.ndim == 1:
                    eef_pos = np.expand_dims(eef_pos, axis=0)
                eef_quat = obs.get("robot0_eef_quat")
                if eef_quat is not None and eef_quat.ndim == 1:
                    eef_quat = np.expand_dims(eef_quat, axis=0)
                gripper_qpos = obs.get("robot0_gripper_qpos")
                if gripper_qpos is not None and gripper_qpos.ndim == 1:
                    gripper_qpos = np.expand_dims(gripper_qpos, axis=0)
                view_keys, view_widths, concat_rgb_key = self._policy_view_metadata(obs)

                policy_obs = PolicyObservation(
                    rgb=rgb,
                    q_robot=None,
                    rgb_vis=rgb.copy() if rgb is not None else None,
                    view_keys=view_keys,
                    view_widths=view_widths,
                    concat_rgb_key=concat_rgb_key,
                    step_index=step_cnt,
                    eef_pos=eef_pos,
                    eef_quat=eef_quat,
                    gripper_qpos=gripper_qpos,
                    dt=self.cfg.dt,
                    action_mode=self.cfg.action_mode,
                    pose_format=self.cfg.pose_format,
                )
                policy_out: PolicyOutput = policy.predict_action(policy_obs)
                action = policy_out.action.copy()
                chunk_story_dream_vis = (
                    policy_out.info.get("chunk_story_dream_vis")
                    if policy_out.info
                    else None
                )
                chunk_story_context_rgb = (
                    policy_out.info.get("chunk_story_context_rgb")
                    if policy_out.info
                    else None
                )
                if chunk_story_dream_vis is not None:
                    chunk_story_dream_vis = np.asarray(chunk_story_dream_vis)
                    if (
                        chunk_story_dream_vis.ndim >= 4
                        and len(chunk_story_dream_vis) > 0
                    ):
                        active_story_shape = self._compact_storyboard_shape(
                            (
                                int(chunk_story_dream_vis[0].shape[0]),
                                int(chunk_story_dream_vis[0].shape[1]),
                            )
                        )
                        active_story_chunk_index = (
                            int(policy_out.info.get("chunk_story_index"))
                            if policy_out.info
                            and policy_out.info.get("chunk_story_index") is not None
                            else None
                        )
                        active_story_exec_horizon = (
                            int(policy_out.info.get("chunk_story_exec_horizon"))
                            if policy_out.info
                            and policy_out.info.get("chunk_story_exec_horizon")
                            is not None
                            else 0
                        )
                        active_story_exec_step = 0
                        if chunk_story_context_rgb is not None:
                            context_tensor = (
                                chunk_story_context_rgb.detach().cpu()
                                if isinstance(chunk_story_context_rgb, torch.Tensor)
                                else np.asarray(chunk_story_context_rgb)
                            )
                            if (
                                len(context_tensor.shape) >= 5
                                and context_tensor.shape[0] > 0
                            ):
                                context_frames = [
                                    context_tensor[0, t]
                                    for t in range(context_tensor.shape[1])
                                ]
                                context_title = "context"
                                if active_story_chunk_index is not None:
                                    context_title = (
                                        f"context: chunk {active_story_chunk_index}"
                                    )
                                self._append_storyboard_sequence(
                                    videos_demo,
                                    context_frames,
                                    target_shape=active_story_shape,
                                    title=context_title,
                                )
                        dream_title = "dream"
                        if active_story_chunk_index is not None:
                            dream_title = f"dream: chunk {active_story_chunk_index}"
                        self._append_storyboard_sequence(
                            videos_demo,
                            [
                                chunk_story_dream_vis[t]
                                for t in range(chunk_story_dream_vis.shape[0])
                            ],
                            target_shape=active_story_shape,
                            title=dream_title,
                        )
                if action.ndim == 1:
                    action = np.expand_dims(action, 0)
                action = np.asarray(action, dtype=np.float32)
                action *= self.cfg.action_scale

                # Debug print for action info (align with RobomimicRunner); use tqdm.write so bar is not overwritten
                action_flat = action.reshape(-1, action.shape[-1])
                self._step_debug_write(
                    f"[Step {step_cnt:3d}] Action - "
                    f"shape={action.shape}, "
                    f"mean={float(np.mean(action)):.4f}, "
                    f"min={float(np.min(action)):.4f}, "
                    f"max={float(np.max(action)):.4f}, "
                    f"std={float(np.std(action)):.4f}, "
                    f"sample[0]={action_flat[0]}"
                )
                if policy_out.info is not None:
                    gripper_debug = policy_out.info.get("gripper_debug")
                    if isinstance(gripper_debug, dict):
                        pre_clip = np.asarray(gripper_debug.get("gripper_pre_clip"))
                        raw = np.asarray(gripper_debug.get("gripper_raw"))
                        raw_filtered = np.asarray(
                            gripper_debug.get("gripper_raw_filtered")
                        )
                        final = np.asarray(gripper_debug.get("gripper_final"))
                        close_mask = np.asarray(gripper_debug.get("gripper_close_mask"))
                        open_mask = np.asarray(gripper_debug.get("gripper_open_mask"))
                        open_intent = np.asarray(
                            gripper_debug.get("gripper_open_intent")
                        )
                        open_blocked = np.asarray(
                            gripper_debug.get("gripper_open_blocked")
                        )
                        hold_remaining = np.asarray(
                            gripper_debug.get("gripper_hold_remaining")
                        )
                        mode = gripper_debug.get("gripper_mode")
                        close_thresh = gripper_debug.get("gripper_close_thresh")
                        open_thresh = gripper_debug.get("gripper_open_thresh")
                        if raw.size > 0 and final.size > 0:
                            pre_clip_msg = ""
                            if pre_clip.size > 0:
                                pre_clip_msg = (
                                    f" pre_clip={float(pre_clip.reshape(-1)[0]):.4f}"
                                )
                            filtered_msg = ""
                            if raw_filtered.size > 0:
                                filtered_msg = f" filtered={float(raw_filtered.reshape(-1)[0]):.4f}"
                            hold_msg = ""
                            if hold_remaining.size > 0:
                                hold_msg = f" hold={int(hold_remaining.reshape(-1)[0])}"
                            close_thresh_msg = (
                                f"{float(close_thresh):.4f}"
                                if close_thresh is not None
                                else "n/a"
                            )
                            open_thresh_msg = (
                                f"{float(open_thresh):.4f}"
                                if open_thresh is not None
                                else "n/a"
                            )
                            self._step_debug_write(
                                f"[Step {step_cnt:3d}] Gripper pre-gate - "
                                f"mode={mode}{pre_clip_msg} raw={float(raw.reshape(-1)[0]):.4f} "
                                f"{filtered_msg}"
                                f" close_thr={close_thresh_msg} "
                                f"open_thr={open_thresh_msg} "
                                f"close={bool(close_mask.reshape(-1)[0]) if close_mask.size > 0 else False} "
                                f"open={bool(open_mask.reshape(-1)[0]) if open_mask.size > 0 else False} "
                                f"open_intent={bool(open_intent.reshape(-1)[0]) if open_intent.size > 0 else False} "
                                f"blocked={bool(open_blocked.reshape(-1)[0]) if open_blocked.size > 0 else False} "
                                f"{hold_msg}"
                                f"final={float(final.reshape(-1)[0]):.4f}"
                            )
                    track_stats = policy_out.info.get("track_stats")
                    control_track_stats = policy_out.info.get("control_track_stats")
                    if isinstance(track_stats, dict):
                        self._step_debug_write(
                            f"[Step {step_cnt:3d}] AllTracker disp - "
                            f"dx=[{track_stats['disp_x_min']:.2f}, {track_stats['disp_x_max']:.2f}] "
                            f"dy=[{track_stats['disp_y_min']:.2f}, {track_stats['disp_y_max']:.2f}] "
                            f"norm_max={track_stats['disp_norm_max']:.2f} "
                            f"norm_p95={track_stats['disp_norm_p95']:.2f} "
                            f"selected={int(track_stats['count_selected'])}/{int(track_stats['count_total'])}"
                        )
                    if isinstance(control_track_stats, dict):
                        outlier_stats = policy_out.info.get("outlier_stats")
                        dropped_msg = ""
                        if isinstance(outlier_stats, dict):
                            dropped_msg = (
                                f" dropped={int(outlier_stats['count_dropped'])}"
                                f"/{int(outlier_stats['count_total'])}"
                            )
                        self._step_debug_write(
                            f"[Step {step_cnt:3d}] Control-track disp - "
                            f"dx=[{control_track_stats['disp_x_min']:.2f}, {control_track_stats['disp_x_max']:.2f}] "
                            f"dy=[{control_track_stats['disp_y_min']:.2f}, {control_track_stats['disp_y_max']:.2f}] "
                            f"norm_max={control_track_stats['disp_norm_max']:.2f} "
                            f"norm_p95={control_track_stats['disp_norm_p95']:.2f} "
                            f"selected={int(control_track_stats['count_selected'])}/{int(control_track_stats['count_total'])}"
                            f"{dropped_msg}"
                        )

                for _ in range(self.cfg.n_repeat):
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = bool(np.any(terminated) or np.any(truncated))
                    if done:
                        break
                obs = _normalize_obs_for_policy(obs, self.image_keys)
                feedback_rgb = self._policy_rgb_from_obs(obs)
                if feedback_rgb.ndim == 3:
                    feedback_rgb = np.expand_dims(feedback_rgb, axis=0)
                feedback_view_keys, feedback_view_widths, feedback_concat_rgb_key = (
                    self._policy_view_metadata(obs)
                )
                feedback_eef_pos = obs.get("robot0_eef_pos")
                if feedback_eef_pos is not None and feedback_eef_pos.ndim == 1:
                    feedback_eef_pos = np.expand_dims(feedback_eef_pos, axis=0)
                feedback_eef_quat = obs.get("robot0_eef_quat")
                if feedback_eef_quat is not None and feedback_eef_quat.ndim == 1:
                    feedback_eef_quat = np.expand_dims(feedback_eef_quat, axis=0)
                feedback_gripper_qpos = obs.get("robot0_gripper_qpos")
                if (
                    feedback_gripper_qpos is not None
                    and feedback_gripper_qpos.ndim == 1
                ):
                    feedback_gripper_qpos = np.expand_dims(
                        feedback_gripper_qpos, axis=0
                    )
                feedback_info = policy.observe_rollout_feedback(
                    PolicyObservation(
                        rgb=feedback_rgb,
                        q_robot=None,
                        view_keys=feedback_view_keys,
                        view_widths=feedback_view_widths,
                        concat_rgb_key=feedback_concat_rgb_key,
                        step_index=step_cnt + 1,
                        eef_pos=feedback_eef_pos,
                        eef_quat=feedback_eef_quat,
                        gripper_qpos=feedback_gripper_qpos,
                        dt=float(self.cfg.dt) * max(int(self.cfg.n_repeat), 1),
                        action_mode=self.cfg.action_mode,
                        pose_format=self.cfg.pose_format,
                    )
                )
                reward = self._reward_to_float(reward)
                ep_rew += reward
                ep_max_rew = max(ep_max_rew, reward)
                success_threshold = self.cfg.success_reward_threshold
                env_success_reached = success_threshold is not None and reward >= float(
                    success_threshold
                )
                relaxed_success_reached, relaxed_success_debug = (
                    self._stack_three_relaxed_success()
                )
                env_success_latched = bool(env_success_latched or env_success_reached)
                relaxed_success_latched = bool(
                    relaxed_success_latched
                    or env_success_latched
                    or relaxed_success_reached
                )
                success_terminated = bool(
                    env_success_reached or relaxed_success_reached
                )
                exec_subtitle = None
                if active_story_exec_horizon > 0:
                    exec_subtitle = (
                        f"step {active_story_exec_step + 1}/{active_story_exec_horizon}"
                    )
                self._append_story_execution_frame(
                    videos_demo=videos_demo,
                    exec_rgb=feedback_rgb[0],
                    active_story_shape=active_story_shape,
                    active_story_chunk_index=active_story_chunk_index,
                    subtitle=exec_subtitle,
                )
                self._append_logged_step(
                    traj=traj,
                    videos_demo=videos_demo,
                    obs_rgb=feedback_rgb,
                    obs=obs,
                    action=action,
                    reward=reward,
                    timestep=step_cnt + 1,
                    policy_vis=(
                        policy_out.info.get("policy_vis")
                        if policy_out.info is not None
                        else None
                    ),
                    policy_out=policy_out,
                )
                if success_terminated and not done:
                    success_reason = (
                        "env reward"
                        if env_success_reached
                        else "relaxed stack-three criterion"
                    )
                    self._step_debug_write(
                        f"[MimicgenRunner] Success reached via {success_reason}; recording "
                        f"{int(self.cfg.post_success_steps)} post-success steps: "
                        f"reward={reward:.4f}"
                    )
                    zero_action = np.zeros_like(action, dtype=np.float32)
                    obs, tail_reward_sum, tail_max_reward, tail_done, tail_env_success = (
                        self._run_post_success_tail(
                            env=env,
                            obs=obs,
                            action=zero_action,
                            traj=traj,
                            videos_demo=videos_demo,
                            timestep_start=step_cnt + 2,
                            active_story_shape=active_story_shape,
                            active_story_chunk_index=active_story_chunk_index,
                        )
                    )
                    ep_rew += tail_reward_sum
                    ep_max_rew = max(ep_max_rew, tail_max_reward)
                    env_success_latched = bool(env_success_latched or tail_env_success)
                    relaxed_success_latched = bool(
                        relaxed_success_latched or env_success_latched
                    )
                    done = True or tail_done
                elif success_terminated:
                    success_reason = (
                        "env reward"
                        if env_success_reached
                        else "relaxed stack-three criterion"
                    )
                    relaxed_suffix = ""
                    if relaxed_success_debug:
                        relaxed_suffix = (
                            f" cubeA_stacked={relaxed_success_debug.get('cubeA_stacked')}"
                            f" cubeC_lifted={relaxed_success_debug.get('cubeC_lifted')}"
                            f" cubeC_touching={relaxed_success_debug.get('cubeC_touching_cubeA')}"
                            f" grasping_cubeC={relaxed_success_debug.get('grasping_cubeC')}"
                        )
                    self._step_debug_write(
                        f"[MimicgenRunner] Success reached via {success_reason} on terminal env step: "
                        f"reward={reward:.4f}{relaxed_suffix}"
                    )
                if active_story_shape is not None:
                    active_story_exec_step += 1
                    if (
                        active_story_exec_horizon > 0
                        and active_story_exec_step >= active_story_exec_horizon
                    ):
                        active_story_shape = None
                        active_story_chunk_index = None
                        active_story_exec_horizon = 0
                        active_story_exec_step = 0

                if isinstance(feedback_info, dict) and "mismatch" in feedback_info:
                    self._step_debug_write(
                        f"[Step {step_cnt:3d}] Adaptive feedback - "
                        f"mismatch={float(feedback_info['mismatch']):.4f} "
                        f"track_err={float(feedback_info['normalized_track_error']):.4f} "
                        f"valid={float(feedback_info['valid_track_fraction']):.3f}"
                    )
                if isinstance(feedback_info, dict) and "mismatch_ema" in feedback_info:
                    self._step_debug_write(
                        f"[Step {step_cnt:3d}] Adaptive controller - "
                        f"mismatch_ema={float(feedback_info['mismatch_ema']):.4f} "
                        f"lam={float(feedback_info['lam_runtime']):.4f} "
                        f"action_scale={float(feedback_info['action_scale_runtime']):.4f}"
                    )

            episode_rewards[demo_idx] = ep_rew
            max_rewards[demo_idx] = ep_max_rew
            env_successes[demo_idx] = env_success_latched
            relaxed_successes[demo_idx] = relaxed_success_latched
            all_videos.append(videos_demo)
            all_trajs.append(traj)

        print("\n[MimicgenRunner] Rollout complete.")
        print(f"  Demo returns: {episode_rewards}  (mean={episode_rewards.mean():.3f})")

        save_dir = None
        if self.cfg.save_videos or self.cfg.save_trajectory or self.cfg.save_rrd:
            run_id = time.strftime("%Y%m%d_%H%M%S")
            if run_tag:
                safe_tag = self._sanitize_run_tag(run_tag)
                save_dir = Path(self.cfg.output_dir) / f"run_{run_id}_{safe_tag}"
            else:
                save_dir = Path(self.cfg.output_dir) / f"run_{run_id}"
            save_dir.mkdir(parents=True, exist_ok=True)
            metadata = {
                "runner_cfg": self._to_jsonable(self.cfg),
                "policy_class": policy.__class__.__name__,
                "policy_cfg": self._to_jsonable(getattr(policy, "cfg", None)),
                "max_reward_per_demo": self._to_jsonable(max_rewards),
                "max_reward_mean": float(np.mean(max_rewards)),
                "env_success_per_demo": self._to_jsonable(env_successes),
                "relaxed_success_per_demo": self._to_jsonable(relaxed_successes),
                "demo_keys": run_keys,
            }
            if run_tag:
                metadata["run_tag"] = run_tag
            with (save_dir / "config.json").open("w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

        if save_dir is not None and self.cfg.save_trajectory and all_trajs:
            def _concat_or_object(values):
                if not values:
                    return np.array([])
                try:
                    return np.concatenate(values, axis=0)
                except (ValueError, TypeError):
                    try:
                        return np.stack(values, axis=0)
                    except (ValueError, TypeError):
                        out = np.empty(len(values), dtype=object)
                        for i, v in enumerate(values):
                            out[i] = v
                        return out

            for di, traj in enumerate(all_trajs):
                if not traj["timestep"]:
                    continue
                traj_np = {k: _concat_or_object(v) for k, v in traj.items()}
                np.savez_compressed(save_dir / f"trajectory_demo{di}.npz", **traj_np)

        if save_dir is not None and self.cfg.save_videos:
            (save_dir / "videos").mkdir(parents=True, exist_ok=True)
            for di, videos_demo in enumerate(all_videos):
                for key in list(videos_demo.keys()):
                    frames = [f for f in videos_demo[key] if f is not None]
                    if not frames:
                        continue
                    self._save_video(
                        frames,
                        save_dir / "videos" / f"{key}_{run_keys[di]}.mp4",
                    )

        if save_dir is not None and self.cfg.save_rrd and rr is not None:
            if hasattr(rr, "save"):
                rr.save(str(save_dir / "recording.rrd"))
            else:
                print("[MimicgenRunner] rerun SDK has no save(); skipping RRD.")

        # Flatten videos for compatibility: first demo's videos as primary (list of (1, H, W, C))
        videos_flat = all_videos[0] if all_videos else {}
        return {
            "demo_returns": episode_rewards,
            "train_returns": episode_rewards,
            "eval_returns": np.array([], dtype=np.float32),
            "videos": videos_flat,
            "all_videos": all_videos,
            "max_rewards": max_rewards,
            "max_reward_mean": float(np.mean(max_rewards)),
            "env_successes": env_successes,
            "relaxed_successes": relaxed_successes,
            "save_dir": str(save_dir) if save_dir is not None else None,
            "demo_keys": run_keys,
        }


def format_run_results(results: dict):
    """Format MimicgenRunner results for display (metrics + stacked videos). Compatible with RunResults.from_raw."""
    from vera.env_runner.robomimic_runner import format_run_results as _format

    return _format(results)
