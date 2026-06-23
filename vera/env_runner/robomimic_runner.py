import collections
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import gymnasium as gym
import h5py
import mediapy as media
import cv2
import numpy as np
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import torch
import tqdm
from vera.env_runner.base_runner import BaseRunner, BaseRunnerCfg, rr
from vera.policy.base_policy import BasePolicy, PolicyObservation, PolicyOutput
from vera.utils.logging import cyan
from scipy.spatial.transform import Rotation


# ============================================================
# Pose Conversion Utilities
# ============================================================
def quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion (x,y,z,w) to axis-angle (rx,ry,rz).

    Args:
        quat: (4,) or (B,4) quaternion in (x,y,z,w) format

    Returns:
        (3,) or (B,3) axis-angle representation
    """
    if quat.ndim == 1:
        r = Rotation.from_quat(quat)
        return r.as_rotvec()
    else:
        # Batch processing
        r = Rotation.from_quat(quat)
        return r.as_rotvec()


def axis_angle_to_quat(axis_angle: np.ndarray) -> np.ndarray:
    """Convert axis-angle (rx,ry,rz) to quaternion (x,y,z,w).

    Args:
        axis_angle: (3,) or (B,3) axis-angle representation

    Returns:
        (4,) or (B,4) quaternion in (x,y,z,w) format
    """
    if axis_angle.ndim == 1:
        r = Rotation.from_rotvec(axis_angle)
        return r.as_quat()
    else:
        # Batch processing
        r = Rotation.from_rotvec(axis_angle)
        return r.as_quat()


def apply_vel_to_pose(
    pose_pos: np.ndarray,
    pose_rot: np.ndarray,
    vel_lin: np.ndarray,
    vel_ang: np.ndarray,
    dt: float,
    rot_format: Literal["quat", "axis_angle"] = "quat",
) -> tuple:
    """Integrate velocity to get next absolute pose.

    Args:
        pose_pos: (3,) or (B,3) current position
        pose_rot: (4,) or (B,4) current rotation (quat or axis_angle depending on rot_format)
        vel_lin: (3,) or (B,3) linear velocity
        vel_ang: (3,) or (B,3) angular velocity (axis-angle representation)
        dt: time step for integration
        rot_format: rotation format ('quat' or 'axis_angle')

    Returns:
        (next_pos, next_rot) both in same format as input
    """
    # Integrate linear velocity
    next_pos = pose_pos + vel_lin * dt

    # Convert rotation to quaternion if needed
    if rot_format == "axis_angle":
        pose_rot_quat = axis_angle_to_quat(pose_rot)
    else:
        pose_rot_quat = pose_rot

    # Integrate angular velocity (convert to quat increment)
    vel_ang_quat = axis_angle_to_quat(vel_ang * dt)

    # Compose rotations: next_rot = vel_delta * current_rot
    r_current = Rotation.from_quat(pose_rot_quat)
    r_vel = Rotation.from_quat(vel_ang_quat)
    r_next = r_vel * r_current
    next_rot_quat = r_next.as_quat()

    # Convert back to original format if needed
    if rot_format == "axis_angle":
        next_rot = quat_to_axis_angle(next_rot_quat)
    else:
        next_rot = next_rot_quat

    return next_pos, next_rot


@dataclass
class RobomimicRunnerCfg(BaseRunnerCfg):
    env_name: Literal["robomimic"]

    num_env_train: int = 10
    num_env_eval: int = 5
    max_episode_steps: int = 400

    dataset_path: str = ""  # path to HDF5 dataset
    render_size: int = 252
    n_repeat: int = 1  # action repeat
    action_scale: float = 1.0  # scale policy output to env action range
    output_dir: str = "outputs/robomimic_eval"
    save_videos: bool = True
    save_trajectory: bool = True
    save_rrd: bool = True
    video_fps: int = 10

    train_start_idx: int = 0  # starting demo index for training envs
    test_start_seed: int = 1000  # random seed for test envs
    render_obs_key: str = "agentview_image"  # which image key to render
    rendering_image_size: Optional[int] = (
        None  # override image resolution for rendering (e.g. 252, 504)
    )

    # Action mode configuration
    action_mode: Literal["velocity", "absolute"] = (
        "velocity"  # how to interpret policy output
    )
    pose_format: Literal["quat", "axis_angle"] = (
        "quat"  # pose rotation format: (x,y,z,w) or (rx,ry,rz)
    )
    dt: float = 0.1  # time step for velocity integration

    # Shape metadata (defaults for lift task)
    shape_meta: Optional[dict] = None

    def __post_init__(self):
        if self.shape_meta is None:
            self.shape_meta = {
                "obs": {
                    "agentview_image": {"shape": [3, 84, 84], "type": "rgb"},
                    "robot0_eye_in_hand_image": {"shape": [3, 84, 84], "type": "rgb"},
                    "object": {"shape": [10], "type": "low_dim"},
                    "robot0_joint_pos": {"shape": [7], "type": "low_dim"},
                    "robot0_joint_pos_cos": {"shape": [7], "type": "low_dim"},
                    "robot0_joint_pos_sin": {"shape": [7], "type": "low_dim"},
                    "robot0_joint_vel": {"shape": [7], "type": "low_dim"},
                    "robot0_eef_pos": {"shape": [3], "type": "low_dim"},
                    "robot0_eef_quat": {"shape": [4], "type": "low_dim"},
                    "robot0_eef_vel_lin": {"shape": [3], "type": "low_dim"},
                    "robot0_eef_vel_ang": {"shape": [3], "type": "low_dim"},
                    "robot0_gripper_qpos": {"shape": [2], "type": "low_dim"},
                    "robot0_gripper_qvel": {"shape": [2], "type": "low_dim"},
                },
                "action": {"shape": [7]},
            }


class RobomimicImageWrapper(gym.Env):
    """Wraps robomimic environment to provide image observations."""

    def __init__(
        self,
        env,
        shape_meta: dict,
        init_state: Optional[np.ndarray] = None,
        render_obs_key: str = "agentview_image",
    ):
        self.env = env
        self.render_obs_key = render_obs_key
        self.init_state = init_state
        self.seed_state_map = dict()
        self._seed = None
        self.shape_meta = shape_meta
        self.render_cache = None
        self.has_reset_before = False

        # setup spaces
        action_shape = shape_meta["action"]["shape"]
        action_space = gym.spaces.Box(
            low=-1, high=1, shape=action_shape, dtype=np.float32
        )
        self.action_space = action_space

        observation_space = gym.spaces.Dict()
        for key, value in shape_meta["obs"].items():
            shape = value["shape"]
            min_value, max_value = -1, 1
            if key.endswith("image"):
                min_value, max_value = 0, 1
            else:
                min_value, max_value = -1, 1

            this_space = gym.spaces.Box(
                low=min_value, high=max_value, shape=shape, dtype=np.float32
            )
            observation_space[key] = this_space
        self.observation_space = observation_space

    def get_observation(self, raw_obs=None):
        if raw_obs is None:
            raw_obs = self.env.get_observation()

        self.render_cache = raw_obs[self.render_obs_key].copy()

        filtered_obs = {}
        for key, spec in self.shape_meta["obs"].items():
            if key in raw_obs:
                val = raw_obs[key].copy()
                if spec.get("type") == "rgb":
                    val = val.astype(np.float32) / 255.0
                else:
                    val = val.astype(np.float32)
                filtered_obs[key] = val

        return filtered_obs

    def seed(self, seed=None):
        np.random.seed(seed=seed)
        self._seed = seed

    def reset(self, seed=None, options=None):
        """Reset the environment.

        Args:
            seed: Random seed to use for reset (gymnasium standard)
            options: Additional reset options (gymnasium standard).
                    Can contain 'init_state' key for per-env initial state.

        Returns:
            obs, info: Observation and info dict (gymnasium standard)
        """
        if seed is not None:
            self.seed(seed)

        # Check if init_state is provided via options
        init_state_from_options = None
        if options is not None and isinstance(options, dict):
            init_state_from_options = options.get("init_state", None)

        # Priority: options > instance init_state > seed-based reset
        if init_state_from_options is not None:
            if not self.has_reset_before:
                self.env.reset()
                self.has_reset_before = True
            raw_obs = self.env.reset_to({"states": init_state_from_options})
        elif self.init_state is not None:
            if not self.has_reset_before:
                self.env.reset()
                self.has_reset_before = True
            raw_obs = self.env.reset_to({"states": self.init_state})
        elif self._seed is not None:
            seed_val = self._seed
            if seed_val in self.seed_state_map:
                raw_obs = self.env.reset_to({"states": self.seed_state_map[seed_val]})
            else:
                np.random.seed(seed=seed_val)
                raw_obs = self.env.reset()
                state = self.env.get_state()["states"]
                self.seed_state_map[seed_val] = state
            self._seed = None
        else:
            raw_obs = self.env.reset()

        obs = self.get_observation(raw_obs)
        info = {}
        return obs, info

    def step(self, action):
        raw_obs, reward, done, info = self.env.step(action)
        obs = self.get_observation(raw_obs)
        # Convert old gym format (done) to gymnasium format (terminated, truncated)
        terminated = done
        truncated = False
        return obs, reward, terminated, truncated, info

    def render(self, mode="rgb_array"):
        if self.render_cache is None:
            raise RuntimeError("Must run reset or step before render.")
        return self.render_cache.copy()


def create_env(env_meta, shape_meta, enable_render=True):
    """Create a robomimic environment with specified metadata."""
    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta["obs"].items():
        modality_mapping[attr.get("type", "low_dim")].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=enable_render,
        use_image_obs=enable_render,
    )
    return env


class RobomimicRunner(BaseRunner):
    cfg: RobomimicRunnerCfg

    def __init__(
        self,
        cfg: RobomimicRunnerCfg,
        device: Optional[torch.device] = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda:0")
        super().__init__(cfg, device)

    @staticmethod
    def _sanitize_run_tag(tag: str) -> str:
        """Sanitize run tag for use in filenames."""
        safe = []
        for ch in tag.strip():
            if ch.isalnum() or ch in ("-", "_", "."):
                safe.append(ch)
            else:
                safe.append("_")
        return "".join(safe).strip("._-") or "run"

    def _prepare_reset_options(self, options, n_train, n_eval):
        """Prepare reset options and seeds for vectorized environment.

        Args:
            options: User-provided options (dict, list, or None)
            n_train: Number of training environments
            n_eval: Number of evaluation environments

        Returns:
            tuple: (reset_options, reset_seeds) ready for env.reset()
        """
        n_envs = n_train + n_eval
        starting_seed = 1

        if options is None:
            # Default behavior: load dataset init states for training envs
            init_states = []
            with h5py.File(self.dataset_path, "r") as f:
                demo_keys = sorted(list(f["data"].keys()))
                n_demos = len(demo_keys)

                for i in range(n_train):
                    demo_idx = (self.cfg.train_start_idx + i) % n_demos
                    demo_key = f"demo_{demo_idx}"
                    init_state = f["data"][demo_key]["states"][20]
                    init_states.append(init_state.copy())

            reset_options = [{"init_state": init_states[i]} for i in range(n_train)]
            reset_options += [None] * n_eval

            reset_seeds = [starting_seed + i for i in range(n_train)]
            reset_seeds += [self.cfg.test_start_seed + i for i in range(n_eval)]
        else:
            # User provided custom options
            if isinstance(options, list):
                reset_options = options
            else:
                reset_options = [options] * n_envs

            reset_seeds = [starting_seed + i for i in range(n_envs)]

        return reset_options, reset_seeds

    def _pad_frames_to_target(self, frames, target_h, target_w, dtype=np.uint8):
        """Pad frames with zeros to (target_h, target_w). Assumes HWC layout."""
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

    def _save_video(self, frames, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        frames = [np.transpose(f, (1, 2, 0)) if f.shape[0] == 3 else f for f in frames]
        target_h = max(f.shape[0] for f in frames)
        target_w = max(f.shape[1] for f in frames)
        frames = self._pad_frames_to_target(
            frames,
            target_h,
            target_w,
            dtype=frames[0].dtype,
        )
        media.write_video(path, frames, fps=self.cfg.video_fps)

    # -------------------------------------------------------- #
    # Setup environment
    # -------------------------------------------------------- #
    def setup_env(self) -> None:
        abs_action = False  # note: assume absolute actions for robomimic datasets

        """Setup vectorized robomimic environments."""
        dataset_path = Path(self.cfg.dataset_path).expanduser()

        # Load metadata and shape info from dataset
        env_meta = FileUtils.get_env_metadata_from_dataset(str(dataset_path))
        if abs_action:
            env_meta["env_kwargs"]["controller_configs"]["control_delta"] = False

        print("-" * 100)
        print("env_meta:")

        def print_env_meta_recursively(env_meta, prefix=""):
            for key, value in env_meta.items():
                if isinstance(value, dict):
                    print(f"{prefix}{key}:")
                    print_env_meta_recursively(value, prefix + "  ")
                else:
                    print(f"{prefix}{key}: {value}")

        print_env_meta_recursively(env_meta)

        # Override image size if specified
        if self.cfg.rendering_image_size is not None:
            print(
                f"[INFO] Overriding image size to {self.cfg.rendering_image_size}x{self.cfg.rendering_image_size}"
            )
            env_meta["env_kwargs"]["camera_heights"] = self.cfg.rendering_image_size
            env_meta["env_kwargs"]["camera_widths"] = self.cfg.rendering_image_size

        # Use shape_meta from config (pre-defined, not inferred from dataset)
        shape_meta = self.cfg.shape_meta

        # Override RGB observation shapes if image size is specified
        if self.cfg.rendering_image_size is not None:
            for _obs_key, spec in shape_meta["obs"].items():
                if spec.get("type", None) == "rgb":
                    spec["shape"] = [
                        3,
                        self.cfg.rendering_image_size,
                        self.cfg.rendering_image_size,
                    ]

        # Infer actual observation shapes from a dummy environment
        dummy_env = create_env(
            env_meta=env_meta, shape_meta=shape_meta, enable_render=True
        )
        dummy_env.env.hard_reset = False
        dummy_obs = dummy_env.get_observation()

        # Update shape_meta with actual shapes from environment
        for key in shape_meta["obs"].keys():
            if key in dummy_obs:
                actual_shape = list(dummy_obs[key].shape)
                shape_meta["obs"][key]["shape"] = actual_shape

        def env_fn():
            robomimic_env = create_env(
                env_meta=env_meta, shape_meta=shape_meta, enable_render=True
            )
            # Disable hard reset to reduce memory consumption
            robomimic_env.env.hard_reset = False
            wrapped_env = RobomimicImageWrapper(
                env=robomimic_env,
                shape_meta=shape_meta,
                init_state=None,
                render_obs_key=self.cfg.render_obs_key,
            )
            wrapped_env = gym.wrappers.PassiveEnvChecker(wrapped_env)
            wrapped_env = gym.wrappers.OrderEnforcing(wrapped_env)
            wrapped_env = gym.wrappers.TimeLimit(
                wrapped_env, max_episode_steps=self.cfg.max_episode_steps
            )
            return wrapped_env

        n_envs = self.cfg.num_env_train + self.cfg.num_env_eval
        env_fns = [lambda: env_fn() for _ in range(n_envs)]

        self.env = gym.vector.SyncVectorEnv(env_fns)
        self.shape_meta = shape_meta
        self.env_meta = env_meta
        self.dataset_path = str(dataset_path)

    # -------------------------------------------------------- #
    # Rollout
    # -------------------------------------------------------- #
    def run(self, policy: BasePolicy, options=None, run_tag: str | None = None):
        """Run one episode for all vectorized envs using the given policy.

        Args:
            policy (BasePolicy): The policy to use for action prediction.
            options (dict or list, optional): Initial state options for environment reset.
                    - If None: uses dataset init states for training, seed-based for eval
                    - If dict: broadcast to all envs
                    - If list: one option per env
            run_tag (str, optional): Extra tag appended to the output folder name.

        Returns:
            dict: Results containing train/eval returns, videos, and max rewards.
        """
        env = self.env

        n_train = self.cfg.num_env_train
        n_eval = self.cfg.num_env_eval
        n_envs = n_train + n_eval
        max_steps = self.cfg.max_episode_steps // self.cfg.n_repeat

        # Prepare reset options and seeds
        reset_options, reset_seeds = self._prepare_reset_options(
            options, n_train, n_eval
        )

        # Reset envs with deterministic seeds
        # Note: SyncVectorEnv.reset(options=...) expects a single dict, not a list
        # Options are broadcast to all sub-environments
        obs, info = env.reset(
            seed=reset_seeds, options=reset_options[0] if reset_options else None
        )
        policy.reset()

        # Episode tracking
        episode_rewards = np.zeros(n_envs, dtype=np.float32)
        max_rewards = np.full(n_envs, -np.inf, dtype=np.float32)
        done_flags = np.zeros(n_envs, dtype=bool)

        # Store videos for all available image keys
        videos = {"obs": [], "policy": []}
        image_keys = [k for k in obs.keys() if k.endswith("image")]
        for img_key in image_keys:
            videos[img_key] = []

        traj = {
            "timestep": [],
            "obs": [],
            "action": [],
            "reward": [],
        }

        for step_cnt in tqdm.trange(max_steps, desc="[RobomimicRunner] Rollout"):

            if np.all(done_flags):
                print(cyan("[RobomimicRunner] All envs done, stopping rollout."))
                break

            # -------------------------------------------------- #
            # Convert vectorized observations → PolicyObservation
            # -------------------------------------------------- #
            # Extract image from observations
            rgb_key = self.cfg.render_obs_key
            if rgb_key in obs:
                rgb = obs[rgb_key]
            else:
                # Fallback to first image key found
                image_keys = [k for k in obs.keys() if k.endswith("image")]
                rgb = obs[image_keys[0]] if image_keys else None

            rgb = rgb.copy()

            # Extract state information for pose tracking
            eef_pos = obs.get("robot0_eef_pos", None)
            if eef_pos is not None:
                eef_pos = eef_pos.copy()

            eef_quat = obs.get("robot0_eef_quat", None)
            if eef_quat is not None:
                eef_quat = eef_quat.copy()

            gripper_qpos = obs.get("robot0_gripper_qpos", None)
            if gripper_qpos is not None:
                gripper_qpos = gripper_qpos.copy()

            policy_obs = PolicyObservation(
                rgb=rgb,
                q_robot=None,
                rgb_vis=rgb.copy(),
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
            action *= self.cfg.action_scale

            # NOTE: Action scaling (translation, yaw), zeroing (roll/pitch), and gripper
            # (fixed or gated) are configured in MotionPolicyGripperCfg and applied
            # inside the policy (see vera.policy.motion_policy_gripper).

            # Debug print for action info
            print(
                f"[Step {step_cnt:3d}] Action - "
                f"shape={action.shape}, "
                f"mean={action.mean():.4f}, "
                f"min={action.min():.4f}, "
                f"max={action.max():.4f}, "
                f"std={action.std():.4f}, "
                f"sample[0]={action[0]}"
            )

            # -------------------------------------------------- #
            # Step the vectorized environment
            # -------------------------------------------------- #
            for _ in range(self.cfg.n_repeat):
                obs, reward, terminated, truncated, info = env.step(action)

            reward = np.asarray(reward, dtype=np.float32)
            terminated = np.asarray(terminated, dtype=bool)
            truncated = np.asarray(truncated, dtype=bool)
            done = np.logical_or(terminated, truncated)

            # accumulate reward only for unfinished envs
            episode_rewards[~done_flags] += reward[~done_flags]
            max_rewards = np.maximum(max_rewards, reward)
            done_flags = np.logical_or(done_flags, done)

            # Terminate rollout if any episode reward exceeds threshold
            if np.any(episode_rewards >= 30):
                print(cyan("[RobomimicRunner] Episode reward >= 30, stopping rollout."))
                break

            # Collect trajectory data
            traj["timestep"].append(np.full(n_envs, step_cnt, dtype=np.int32))
            if rgb is not None:
                traj["obs"].append(rgb.copy())
            traj["action"].append(action.copy())
            traj["reward"].append(reward.copy())

            # Collect video frames for main render key
            if rgb is not None:
                videos["obs"].append(rgb.copy())

            if policy_out.info is not None:
                policy_vis = policy_out.info.get("policy_vis")
            else:
                policy_vis = None

            # Use policy visualization if available; otherwise fall back to env RGB.
            vis_frame = policy_vis.copy() if policy_vis is not None else rgb.copy()

            # -------------------------------------------------- #
            # Overlay step index on policy visualization (action is drawn on left panel in motion_policy)
            # -------------------------------------------------- #
            # Expect vis_frame to be float32 in [0,1] with shape [B,H,W,C].
            try:
                vis_annot = vis_frame
                if vis_annot.dtype != np.uint8:
                    vis_annot = (vis_annot * 255.0).clip(0, 255).astype(np.uint8)

                # Work on a copy to avoid mutating stored frames elsewhere.
                vis_annot = vis_annot.copy()
                # Convert back to float32 [0,1] for consistency with rest of pipeline
                vis_annot = vis_annot.astype(np.float32) / 255.0

            except Exception:
                # In case of any unexpected shape/dtype, fall back silently.
                vis_annot = vis_frame

            videos["policy"].append(vis_annot)

            # Collect all image views for videos dict
            for img_key in image_keys:
                if img_key in obs:
                    videos[img_key].append(obs[img_key].copy())

            # Log to rerun - include all image views
            if rgb is not None:
                images = {
                    "env/obs": rgb[0],
                    "policy/vis": policy_vis[0] if policy_vis is not None else None,
                }

                # Log all available image observations
                for img_key in image_keys:
                    if img_key in obs:
                        images[f"env/{img_key}"] = obs[img_key][0]

                scalars = {
                    "action/norm": np.linalg.norm(action[0]),
                    "reward": reward[0],
                }
                self.log_rerun(step_cnt, images=images, scalars=scalars)

        # -------------------------------------------------------- #
        # Split train/eval and print results
        # -------------------------------------------------------- #
        train_rews = episode_rewards[:n_train]
        eval_rews = episode_rewards[n_train:]

        print("\n[RobomimicRunner] Rollout complete.")
        print(f"  Train returns: {train_rews}  (mean={train_rews.mean():.3f})")
        print(f"  Eval  returns: {eval_rews}  (mean={eval_rews.mean():.3f})")

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
                "max_reward_per_env": self._to_jsonable(max_rewards),
                "max_reward_mean": float(np.mean(max_rewards)),
            }

            if run_tag:
                metadata["run_tag"] = run_tag
            controller = getattr(policy, "controller", None)
            if controller is not None and hasattr(controller, "cfg"):
                metadata["controller_cfg"] = self._to_jsonable(controller.cfg)

            with (save_dir / "config.json").open("w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

        if save_dir is not None and self.cfg.save_trajectory:
            traj_np = {k: np.stack(v, axis=0) for k, v in traj.items()}
            np.savez_compressed(save_dir / "trajectory.npz", **traj_np)

        if save_dir is not None and self.cfg.save_videos:
            for key, frames in videos.items():
                for env_idx in range(n_envs):
                    env_frames = [f[env_idx] for f in frames]
                    self._save_video(
                        env_frames,
                        save_dir / "videos" / f"{key}_env{env_idx}.mp4",
                    )

        if save_dir is not None and self.cfg.save_rrd and rr is not None:
            if hasattr(rr, "save"):
                rr.save(str(save_dir / "recording.rrd"))
            else:
                print("[RobomimicRunner] rerun SDK has no save() method; skipping RRD.")

        # Normalize policy video frames to consistent shape (pad with zeros) for np.stack
        policy_frames = videos["policy"]
        if policy_frames:
            target_h = max(f.shape[1] for f in policy_frames)
            target_w = max(f.shape[2] for f in policy_frames)
            if any(
                f.shape[1] != target_h or f.shape[2] != target_w for f in policy_frames
            ):
                normalized = []
                for f in policy_frames:
                    # f: (n_envs, H, W, C)
                    h, w = f.shape[1], f.shape[2]
                    if h == target_h and w == target_w:
                        normalized.append(f)
                    else:
                        pad = np.zeros(
                            (f.shape[0], target_h, target_w, f.shape[3]),
                            dtype=f.dtype,
                        )
                        pad[:, :h, :w] = f
                        normalized.append(pad)
                videos["policy"] = normalized

        return {
            "train_returns": train_rews,
            "eval_returns": eval_rews,
            "videos": videos,
            "max_rewards": max_rewards,
            "max_reward_mean": float(np.mean(max_rewards)),
            "save_dir": str(save_dir) if save_dir is not None else None,
        }


# --------------------------------------------------
# Run results formatting and display
# --------------------------------------------------
def format_run_results(results: dict) -> "RunResults":
    """Convert raw run results into a display-ready format with stacked videos and metrics."""
    return RunResults.from_raw(results)


@dataclass
class RunResults:
    """Display-ready run results: stacked videos and metrics."""

    metrics: dict
    videos: dict[str, np.ndarray]  # key -> (n_envs, T, H, W, C)
    save_dir: str | None

    @classmethod
    def from_raw(cls, results: dict) -> "RunResults":
        raw_videos = results.get("videos", {})
        videos = {}
        for key, frames in raw_videos.items():
            if not frames:
                continue
            # Stack: list of (n_envs, H, W, C) -> (n_envs, T, H, W, C)
            stacked = _stack_video_frames(frames)
            if stacked is not None:
                videos[key] = stacked

        metrics = {
            "train_returns": results.get("train_returns"),
            "eval_returns": results.get("eval_returns"),
            "max_rewards": results.get("max_rewards"),
            "max_reward_mean": results.get("max_reward_mean"),
        }
        return cls(
            metrics=metrics,
            videos=videos,
            save_dir=results.get("save_dir"),
        )

    def show(
        self, fps: int = 5, height: int = 252, video_keys: list[str] | None = None
    ):
        """Print metrics and display all videos."""
        print("─── Metrics ───")
        for k, v in self.metrics.items():
            if v is not None:
                print(f"  {k}: {v}")
        if self.save_dir:
            print(f"  save_dir: {self.save_dir}")

        keys = video_keys or list(self.videos.keys())
        for key in keys:
            if key not in self.videos:
                continue
            arr = self.videos[key]
            print(f"\n─── {key} ───")
            media.show_videos(arr, fps=fps, height=height)


def _stack_video_frames(frames: list[np.ndarray]) -> np.ndarray | None:
    """Stack list of (n_envs, H, W, C) into (n_envs, T, H, W, C). Pads with zeros if shapes vary."""
    if not frames:
        return None
    target_h = max(f.shape[1] for f in frames)
    target_w = max(f.shape[2] for f in frames)
    n_envs = frames[0].shape[0]
    n_channels = frames[0].shape[3]
    dtype = frames[0].dtype

    out = np.zeros((n_envs, len(frames), target_h, target_w, n_channels), dtype=dtype)
    for t, f in enumerate(frames):
        h, w = f.shape[1], f.shape[2]
        if h == target_h and w == target_w:
            out[:, t] = f
        else:
            out[:, t, :h, :w] = f
    return out


# --------------------------------------------------
# 1. Hand-crafted policy: random actions
# --------------------------------------------------
class RandomPolicy(BasePolicy):
    def __init__(self, action_dim: int = 7):
        self.action_dim = action_dim
        self._action_scale = 0.1

    def reset(self):
        pass

    def predict_action(self, obs: PolicyObservation) -> PolicyOutput:
        # Infer batch size from RGB observation
        if obs.rgb is not None:
            batch_size = obs.rgb.shape[0]
        else:
            batch_size = 1

        # Generate batched actions: (batch_size, action_dim)
        action = np.random.uniform(
            -self._action_scale,
            self._action_scale,
            size=(batch_size, self.action_dim),
        ).astype(np.float32)
        return PolicyOutput(action=action, info=None)


# --------------------------------------------------
# 2. Hand-crafted policy: move straight down in z
# --------------------------------------------------
class MoveDownZPolicy(BasePolicy):
    def __init__(
        self,
        action_dim: int = 7,
        z_speed: float = 0.05,  # magnitude in [-1, 1] action units (velocity) or absolute target (absolute)
        z_index: int = 2,  # 3rd idx for position/velocity
        mode: Literal["velocity", "absolute"] = "velocity",
        rot_format: Literal["quat", "axis_angle"] = "quat",
    ):
        self.action_dim = action_dim
        self.z_speed = float(z_speed)
        self.z_index = int(z_index)
        self.mode = mode
        self.rot_format = rot_format
        self.target_eef_pos = None  # for absolute mode
        self.target_eef_rot = None  # for absolute mode

    def reset(self):
        self.target_eef_pos = None
        self.target_eef_rot = None

    def predict_action(self, obs: PolicyObservation) -> PolicyOutput:
        # Infer batch size from RGB observation
        batch_size = obs.rgb.shape[0] if obs.rgb is not None else 1

        if self.mode == "velocity":
            # Pure velocity mode: output velocity commands directly
            action = np.zeros((batch_size, self.action_dim), dtype=np.float32)
            action[:, self.z_index] = -abs(self.z_speed)
            eps = 1e-5
            action += eps * np.random.randn(*action.shape)
            action[..., -1] = -0.2

        elif self.mode == "absolute":
            # Absolute mode: integrate velocity to track a target position
            if obs.eef_pos is None:
                raise ValueError("eef_pos required for absolute mode")

            # Initialize target on first step
            if self.target_eef_pos is None:
                self.target_eef_pos = obs.eef_pos.copy()
                if obs.eef_quat is not None:
                    self.target_eef_rot = obs.eef_quat.copy()

            # Update target by integrating velocity
            vel_lin = np.zeros((batch_size, 3), dtype=np.float32)
            vel_lin[:, self.z_index] = -abs(self.z_speed)

            vel_ang = np.zeros((batch_size, 3), dtype=np.float32)

            # Apply velocity integration
            if obs.eef_quat is not None and self.target_eef_rot is not None:
                next_pos, next_rot = apply_vel_to_pose(
                    self.target_eef_pos,
                    self.target_eef_rot,
                    vel_lin,
                    vel_ang,
                    dt=obs.dt,
                    rot_format=obs.pose_format,
                )
                self.target_eef_pos = next_pos
                self.target_eef_rot = next_rot
            else:
                self.target_eef_pos = self.target_eef_pos + vel_lin * obs.dt

            # Compute error and convert to action (delta pose)
            pos_error = self.target_eef_pos - obs.eef_pos

            # For rotation, compute axis-angle error
            if obs.eef_quat is not None and self.target_eef_rot is not None:
                if obs.pose_format == "axis_angle":
                    target_rot_quat = axis_angle_to_quat(self.target_eef_rot)
                else:
                    target_rot_quat = self.target_eef_rot

                r_current = Rotation.from_quat(obs.eef_quat)
                r_target = Rotation.from_quat(target_rot_quat)
                r_error = r_target * r_current.inv()
                rot_error = r_error.as_rotvec()
            else:
                rot_error = np.zeros((batch_size, 3), dtype=np.float32)

            # Build action: [delta_pos (3), delta_rot (3), gripper (1)]
            action = np.zeros((batch_size, self.action_dim), dtype=np.float32)
            action[:, :3] = pos_error
            action[:, 3:6] = rot_error
            action[:, -1] = -0.2

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return PolicyOutput(action=action, info=None)


# --------------------------------------------------
# 3. Demo-based policy: replay demonstrated velocities
# --------------------------------------------------
class DemoVelocityPolicy(BasePolicy):
    """Replays velocity sequences from demonstrations with flexible modes.

    Supports:
    - velocity mode: outputs velocity commands directly
    - absolute mode: integrates velocities to track absolute target poses
    """

    def __init__(
        self,
        vel_seq: np.ndarray,  # (T,6) velocities [lin(3), ang(3)]
        grip_seq: np.ndarray,  # (T,1) gripper commands
        action_dim=7,
        vel_scale=1.0,
        grip_scale=1.0,
        max_steps=None,
        mode: Literal["velocity", "absolute"] = "velocity",
        rot_format: Literal["quat", "axis_angle"] = "quat",
    ):
        self.vel_seq = vel_seq
        self.grip_seq = grip_seq
        self.vel_scale = vel_scale
        self.grip_scale = grip_scale
        self.action_dim = action_dim
        self.max_steps = max_steps or len(vel_seq)
        self.t = 0
        self.mode = mode
        self.rot_format = rot_format
        self.target_eef_pos = None  # for absolute mode
        self.target_eef_rot = None  # for absolute mode

    def reset(self):
        self.t = 0
        self.target_eef_pos = None
        self.target_eef_rot = None

    def predict_action(self, obs: PolicyObservation) -> PolicyOutput:
        B = obs.rgb.shape[0] if obs.rgb is not None else 1

        idx = min(self.t, self.max_steps - 1)

        # velocity components from demo
        vel6 = self.vel_seq[idx] * self.vel_scale  # (6,)
        vel_lin = vel6[:3]  # (3,)
        vel_ang = vel6[3:]  # (3,)
        grip = self.grip_seq[idx] * self.grip_scale  # (1,)

        if self.mode == "velocity":
            # Pure velocity mode: output velocity directly
            action = np.zeros((B, self.action_dim), dtype=np.float32)
            action[:, :3] = vel_lin
            action[:, 3:6] = vel_ang
            action[:, -1] = grip.squeeze(-1) if grip.ndim > 0 else grip

        elif self.mode == "absolute":
            # Absolute mode: integrate velocity to track target
            if obs.eef_pos is None:
                raise ValueError("eef_pos required for absolute mode")

            # Initialize target on first step
            if self.target_eef_pos is None:
                self.target_eef_pos = obs.eef_pos.copy()
                if obs.eef_quat is not None:
                    self.target_eef_rot = obs.eef_quat.copy()

            # Update target by integrating velocity
            vel_lin_batch = np.tile(vel_lin, (B, 1))
            vel_ang_batch = np.tile(vel_ang, (B, 1))

            # Apply velocity integration
            if obs.eef_quat is not None and self.target_eef_rot is not None:
                next_pos, next_rot = apply_vel_to_pose(
                    self.target_eef_pos,
                    self.target_eef_rot,
                    vel_lin_batch,
                    vel_ang_batch,
                    dt=obs.dt,
                    rot_format=obs.pose_format,
                )
                self.target_eef_pos = next_pos
                self.target_eef_rot = next_rot
            else:
                self.target_eef_pos = self.target_eef_pos + vel_lin_batch * obs.dt

            # Compute error and convert to action (delta pose)
            pos_error = self.target_eef_pos - obs.eef_pos

            # For rotation, compute axis-angle error
            if obs.eef_quat is not None and self.target_eef_rot is not None:
                if obs.pose_format == "axis_angle":
                    target_rot_quat = axis_angle_to_quat(self.target_eef_rot)
                else:
                    target_rot_quat = self.target_eef_rot

                r_current = Rotation.from_quat(obs.eef_quat)
                r_target = Rotation.from_quat(target_rot_quat)
                r_error = r_target * r_current.inv()
                rot_error = r_error.as_rotvec()
            else:
                rot_error = np.zeros((B, 3), dtype=np.float32)

            # Build action: [delta_pos (3), delta_rot (3), gripper (1)]
            action = np.zeros((B, self.action_dim), dtype=np.float32)
            action[:, :3] = pos_error
            action[:, 3:6] = rot_error
            action[:, -1] = grip.squeeze(-1) if grip.ndim > 0 else grip

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        self.t += 1

        return PolicyOutput(action=action, info={"t": self.t})


if __name__ == "__main__":
    import numpy as np
    import torch

    # Initialize runner
    cfg = RobomimicRunnerCfg(
        env_name="robomimic",
        num_env_train=2,
        num_env_eval=1,
        max_episode_steps=100,
        dataset_path="/path/to/data/robomimic/datasets/lift/mh/image_abs.hdf5",
    )

    runner = RobomimicRunner(cfg, device=torch.device("cpu"))

    # Run parallel test rollout
    policy = RandomPolicy(action_dim=7)

    print("\n[TEST] Running parallel Robomimic vector rollout...")
    results = runner.run(policy)

    print("\n[TEST] Done.")
    print("Train returns:", results["train_returns"])
    print("Eval returns :", results["eval_returns"])
    print("Save dir:", results["save_dir"])
