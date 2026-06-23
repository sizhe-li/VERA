import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import gym_pusht  # noqa: F401
import gymnasium as gym
import imageio
import numpy as np
import torch
import tqdm
from vera.env_runner.base_runner import BaseRunner, BaseRunnerCfg, rr
from vera.env_runner.env_wrappers.pusht_image_env import PushTImageEnv
from vera.policy.base_policy import BasePolicy, PolicyObservation, PolicyOutput
from vera.utils.logging import cyan


@dataclass
class PushtRunnerCfg(BaseRunnerCfg):
    env_name: Literal["pusht"]

    num_env_train: int = 10
    num_env_eval: int = 5
    max_episode_steps: int = 200

    n_repeat: int = 1  # action repeat
    action_scale: float = 1.0  # scale policy output to env action range
    # Jacobian path: v_cmd from MotionPolicy needs *25 to reach env-pixel-velocity
    # units (hero tuning baked it in). IDM eval should override to 1.0 since
    # IDM outputs are already in env-action units.
    actions_vel_scale: float = 25.0
    output_dir: str = "outputs/pusht_eval"
    save_videos: bool = True
    save_trajectory: bool = True
    save_rrd: bool = True
    video_fps: int = 20


class PushTRunner(BaseRunner):
    cfg: PushtRunnerCfg

    def __init__(
        self,
        cfg: PushtRunnerCfg,
        device: torch.device = torch.device("cuda:0"),
    ) -> None:
        super().__init__(cfg, device)
        self.pos_ref = None  # <-- NEW: per-env reference target position

    @staticmethod
    def _sanitize_run_tag(tag: str) -> str:
        safe = []
        for ch in tag.strip():
            if ch.isalnum() or ch in ("-", "_", "."):
                safe.append(ch)
            else:
                safe.append("_")
        return "".join(safe).strip("._-") or "run"

    def _save_video(self, frames: list[np.ndarray], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        frames_u8 = [(np.clip(f, 0.0, 1.0) * 255).astype(np.uint8) for f in frames]
        imageio.mimwrite(path, frames_u8, fps=self.cfg.video_fps, macro_block_size=None)

    # -------------------------------------------------------- #
    # Setup environment
    # -------------------------------------------------------- #
    def setup_env(self) -> None:
        from gym_pusht.envs.pusht import PushTEnv

        def env_fn():
            # raw PushTEnv
            base = PushTEnv(
                obs_type="state",
                render_mode="rgb_array",
                visualization_width=252,
                visualization_height=252,
                observation_width=252,
                observation_height=252,
                # control_type="position",
                # render_action=False,
                control_type="velocity",
                render_action=False,
            )

            # 1️⃣ inject images closest to base env
            env = PushTImageEnv(base, render_size=252)

            # 2️⃣ normal Gym wrappers outside of it
            env = gym.wrappers.PassiveEnvChecker(env)
            env = gym.wrappers.OrderEnforcing(env)
            env = gym.wrappers.TimeLimit(
                env, max_episode_steps=self.cfg.max_episode_steps
            )

            return env

        n_envs = self.cfg.num_env_train + self.cfg.num_env_eval
        env_fns = [env_fn] * n_envs
        self.env = gym.vector.SyncVectorEnv(env_fns)

    # -------------------------------------------------------- #
    # Rollout
    # -------------------------------------------------------- #
    def run(self, policy: BasePolicy, options=None, run_tag: str | None = None):
        """Run one episode for all vectorized envs using the given policy.

        Args:
            policy (BasePolicy): The policy to use for action prediction.
            options (dict, optional): Additional options for environment reset.
            run_tag (str, optional): Extra tag appended to the output folder name.
        """

        env = self.env
        device = self.device

        n_train = self.cfg.num_env_train
        n_eval = self.cfg.num_env_eval
        n_envs = n_train + n_eval
        max_steps = self.cfg.max_episode_steps // self.cfg.n_repeat
        starting_seed = 1

        # Reset envs & policy
        seeds = np.arange(n_envs) + starting_seed
        seeds = [int(s) for s in seeds]  # force python ints

        obs, info = env.reset(seed=seeds, options=options)
        policy.reset()

        # agent positions from state: [agent_x, agent_y, block_x, block_y, angle]
        self.pos_ref = obs["state"][..., :2].copy()  # (n_envs, 2)

        # Episode tracking
        episode_rewards = np.zeros(n_envs, dtype=np.float32)
        max_rewards = np.full(n_envs, -np.inf, dtype=np.float32)
        done_flags = np.zeros(n_envs, dtype=bool)

        videos = {"clean": [], "vis": [], "policy": []}
        traj = {
            "timestep": [],
            "q_curr": [],
            "q_dot_curr": [],
            "u": [],
            "du_pred": [],
            "du_gt": [],
            "oflow_min": [],
            "oflow_max": [],
            "oflow_mean": [],
        }
        prev_q = None

        # envs_videos = [obs["image"]]
        # policy_videos = []

        for step_cnt in tqdm.trange(max_steps, desc="[PushTRunner] Rollout"):

            if np.all(done_flags):
                print(cyan("[PushTRunner] All envs done, stopping rollout."))
                break

            # -------------------------------------------------- #
            # Convert vectorized observations → PolicyObservation
            # -------------------------------------------------- #
            policy_obs = PolicyObservation(
                rgb=obs["image"],
                q_robot=obs["state"][..., :2],
                rgb_vis=obs["image_vis"],
                step_index=step_cnt,
            )

            policy_out: PolicyOutput = policy.predict_action(policy_obs)
            v_cmd = policy_out.action.copy()  # (n_envs, 2), velocity command
            v_cmd *= self.cfg.action_scale  # extra global scale if desired

            # -------------------------------------------------- #
            # Integrate velocity → position reference
            # -------------------------------------------------- #
            # You can think of dt_outer = 1 "step unit"; scaling is absorbed into vel_gain/action_scale.
            dt_outer = 1.0
            # self.pos_ref = self.pos_ref + v_cmd * dt_outer
            # current pusher position from env
            pos_env = obs["state"][..., :2]  # (n_envs, 2)
            # flip v_cmd [0, 1]
            # v_cmd[:, [0, 1]] = v_cmd[:, [1, 0]]
            pos_pred = pos_env + v_cmd * dt_outer

            # pos_pred[:] = np.array([252.0, 50.0])[
            #     None, :
            # ]  # TESTING: always go to center

            print(
                f"[Step {step_cnt}] pos_env={pos_env[0]} v_cmd={v_cmd[0]} pos_pred={pos_pred[0]}"
            )

            # proposed integrated reference
            # pos_pred = self.pos_ref + v_cmd * dt_outer
            self.pos_ref = pos_pred

            # Clamp to workspace bounds [0, 512] x [0, 512]
            self.pos_ref = np.clip(self.pos_ref, 0.0, 490.0)

            # This is what the env expects: a *position* target
            actions_pos = self.pos_ref.astype(np.float32)
            actions_vel = v_cmd.astype(np.float32) * self.cfg.actions_vel_scale

            # actions_vel = v_cmd.astype(np.float32) * 20

            # -------------------------------------------------- #
            # Step the vectorized environment
            # -------------------------------------------------- #
            for j in range(self.cfg.n_repeat):
                # obs, reward, terminated, truncated, info = env.step(actions_pos)
                obs, reward, terminated, truncated, info = env.step(actions_vel)

            reward = np.asarray(reward, dtype=np.float32)
            terminated = np.asarray(terminated, dtype=bool)
            truncated = np.asarray(truncated, dtype=bool)
            done = np.logical_or(terminated, truncated)

            # accumulate reward only for unfinished envs
            episode_rewards[~done_flags] += reward[~done_flags]
            max_rewards = np.maximum(max_rewards, reward)
            done_flags = np.logical_or(done_flags, done)

            q_curr = obs["state"].copy()
            if prev_q is None:
                q_dot_curr = np.zeros_like(q_curr)
            else:
                q_dot_curr = q_curr - prev_q
            prev_q = q_curr.copy()

            du_gt = None
            if isinstance(info, dict):
                for key in ("du_gt", "gt_action"):
                    if key in info:
                        du_gt = np.asarray(info[key])
                        break
            if du_gt is None:
                du_gt = np.full_like(v_cmd, np.nan)

            flow_stats = {}
            if policy_out.info is not None:
                flow_stats = policy_out.info.get("flow_stats", {})

            oflow_min = np.asarray(
                flow_stats.get("flow_mag_min", np.full(n_envs, np.nan))
            )
            oflow_max = np.asarray(
                flow_stats.get("flow_mag_max", np.full(n_envs, np.nan))
            )
            oflow_mean = np.asarray(
                flow_stats.get("flow_mag_mean", np.full(n_envs, np.nan))
            )

            traj["timestep"].append(np.full(n_envs, step_cnt, dtype=np.int32))
            traj["q_curr"].append(q_curr)
            traj["q_dot_curr"].append(q_dot_curr)
            traj["u"].append(actions_vel.copy())
            traj["du_pred"].append(v_cmd.copy())
            traj["du_gt"].append(du_gt)
            traj["oflow_min"].append(oflow_min)
            traj["oflow_max"].append(oflow_max)
            traj["oflow_mean"].append(oflow_mean)
            # Save raw artifacts (full jacobian / flow / dream / tracks) when
            # MotionPolicyCfg.save_artifacts is on. Stored per-step in traj
            # so trajectory.npz becomes a complete replay artifact.
            raw = (policy_out.info or {}).get("raw_artifacts") if policy_out.info else None
            if raw is not None:
                for k, v in raw.items():
                    traj.setdefault(f"raw__{k}", []).append(np.asarray(v))
                    traj.setdefault(f"raw__{k}__step", []).append(np.int32(step_cnt))

            videos["clean"].append(obs["image"])
            videos["vis"].append(obs["image_vis"])
            if policy_out.info is not None:
                policy_vis = policy_out.info.get("policy_vis")
            else:
                policy_vis = None
            videos["policy"].append(
                policy_vis if policy_vis is not None else obs["image"]
            )

            images = {
                "env/obs_image": obs["image"][0],
                "env/obs_vis": obs["image_vis"][0],
                "policy/vis": policy_vis[0] if policy_vis is not None else None,
            }
            scalars = {
                "action/du_pred": v_cmd[0],
                "action/pos_ref": actions_pos[0],
                "state/q": q_curr[0],
                "state/q_dot": q_dot_curr[0],
                "debug/flow_mag_min": oflow_min[0],
                "debug/flow_mag_max": oflow_max[0],
                "debug/flow_mag_mean": oflow_mean[0],
            }
            self.log_rerun(step_cnt, images=images, scalars=scalars)

        # -------------------------------------------------------- #
        # Split train/eval and print results
        # -------------------------------------------------------- #
        train_rews = episode_rewards[:n_train]
        eval_rews = episode_rewards[n_train:]

        print("\n[PushTRunner] Rollout complete.")
        train_mean = float(train_rews.mean()) if len(train_rews) > 0 else float("nan")
        eval_mean = float(eval_rews.mean()) if len(eval_rews) > 0 else float("nan")
        print(f"  Train returns: {train_rews}  (mean={train_mean:.3f})")
        print(f"  Eval  returns: {eval_rews}  (mean={eval_mean:.3f})")

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
            def _stack_or_object(values):
                if not values:
                    return np.array([], dtype=object)
                try:
                    return np.stack(values, axis=0)
                except (ValueError, TypeError):
                    out = np.empty(len(values), dtype=object)
                    for i, v in enumerate(values):
                        out[i] = v
                    return out

            traj_np = {k: _stack_or_object(v) for k, v in traj.items()}
            np.savez_compressed(save_dir / "trajectory.npz", **traj_np)

        if save_dir is not None and self.cfg.save_videos:
            for key, frames in videos.items():
                for env_idx in range(n_envs):
                    env_frames = [f[env_idx] for f in frames]
                    self._save_video(
                        env_frames,
                        save_dir / "videos" / f"{key}_env{env_idx}.mp4",
                    )

        if save_dir is not None and self.cfg.save_rrd and self._rerun_enabled():
            if hasattr(rr, "save"):
                rr.save(str(save_dir / "recording.rrd"))
            else:
                print("[PushTRunner] rerun SDK has no save() method; skipping RRD.")

        # # for each key in video, stack along axis 1
        # for key in videos.keys():
        #     videos[key] = np.stack(videos[key], axis=1)  # (N_envs, T, H, W, 3)

        return {
            "train_returns": train_rews,
            "eval_returns": eval_rews,
            "videos": videos,
            "max_rewards": max_rewards,
            "max_reward_mean": float(np.mean(max_rewards)),
            "save_dir": str(save_dir) if save_dir is not None else None,
        }


if __name__ == "__main__":
    import numpy as np
    import torch

    # --------------------------------------------------
    # 1. Hand-crafted policy: always move straight right
    # --------------------------------------------------
    class AlwaysRightPolicy(BasePolicy):
        def __init__(self, nu: int = 2):
            self.nu = nu
            self._action = torch.zeros(self.nu, dtype=torch.float32)
            # PushT: dx > 0 moves right
            # self._action[0] = +0.5

            self._action[0] += 0.5

        def reset(self):
            pass

        def predict_action(self, obs: PolicyObservation) -> PolicyOutput:
            return PolicyOutput(action=self._action.clone(), info=None)

    # --------------------------------------------------
    # 2. Initialize runner
    # --------------------------------------------------
    cfg = PushtRunnerCfg(
        env_name="pusht",
        num_env_train=3,
        num_env_eval=2,
        max_episode_steps=100,
    )

    runner = PushTRunner(cfg, device=torch.device("cpu"))

    # --------------------------------------------------
    # 3. Run parallel test rollout
    # --------------------------------------------------
    policy = AlwaysRightPolicy(nu=2)

    print("\n[TEST] Running parallel PushT vector rollout...")
    results = runner.run(policy)

    print("\n[TEST] Done.")
    print("Train returns:", results["train_returns"])
    print("Eval returns :", results["eval_returns"])
