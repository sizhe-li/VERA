from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Literal

import gymnasium as gym
import hydra
import mediapy as media
import numpy as np
import torch
import tqdm
from crm.common import load_hydra_cfg
from crm.pipeline.resolvers import register_resolvers
from gymnasium.vector.utils import concatenate, iterate
from vera.env_runner.base_runner import BaseRunner, BaseRunnerCfg, rr
from vera.policy.base_policy import BasePolicy, PolicyObservation, PolicyOutput
from vera.utils.logging import cyan
from pydrake.all import RollPitchYaw
from tasks_diffusion_policy.neural_jacobian.common.display_check import (
    if_docker_then_check_display_and_x_server,
)
from tasks_diffusion_policy.neural_jacobian.common.paths import (
    NEURAL_JACOBIAN_DIR,
)
from tasks_diffusion_policy.neural_jacobian.iiwa.env.converter import IiwaBimanual
from tasks_diffusion_policy.neural_jacobian.iiwa.env.env import (
    get_color_image_port_name_from_camera_name,
)
from tasks_diffusion_policy.neural_jacobian.iiwa.env.image_converter import (
    GymSyncComplianceWrapper,
    ImageEnvWrapper,
)

media.set_ffmpeg("/usr/bin/ffmpeg")


class SyncVectorEnv(gym.vector.SyncVectorEnv):
    def __init__(
        self,
        env_fns: Iterator[Callable[[], gym.Env]],
        copy: bool = True,
    ):
        """Override gymnasium's SyncVectorEnv to prevent automatic resets of terminated environments."""
        super().__init__(env_fns)

    def step(self, actions):
        """
        Step through each of the environments returning the batched results.

        Args:
        ----
            actions: batched actions

        Returns:
        -------
            The batched environment step results
        """
        actions = iterate(self.action_space, actions)

        observations, infos = [], {}
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            (
                observation,
                self._rewards[i],
                self._terminateds[i],
                self._truncateds[i],
                info,
            ) = env.step(action)

            if self._terminateds[i] or self._truncateds[i]:
                old_observation, old_info = observation, info
                info["final_observation"] = old_observation
                info["final_info"] = old_info

            observations.append(observation)
            infos = self._add_info(infos, info, i)
        self.observations = concatenate(
            self.single_observation_space, observations, self.observations
        )

        return (
            deepcopy(self.observations) if self.copy else self.observations,
            np.copy(self._rewards),
            np.copy(self._terminateds),
            np.copy(self._truncateds),
            infos,
        )


class AsyncVectorEnv(gym.vector.AsyncVectorEnv):
    def __init__(
        self,
        env_fns: Iterator[Callable[[], gym.Env]],
        copy: bool = True,
    ):
        """Override gymnasium's AsyncVectorEnv to prevent automatic resets of terminated environments."""
        super().__init__(env_fns)

    def step_wait(self):
        """
        Step through each of the environments returning the batched results.

        Returns:
        -------
            The batched environment step results
        """
        observations, infos = [], {}
        for i, env in enumerate(self.envs):
            (
                observation,
                self._rewards[i],
                self._terminateds[i],
                self._truncateds[i],
                info,
            ) = self.remotes[i].recv()

            if self._terminateds[i] or self._truncateds[i]:
                old_observation, old_info = observation, info
                info["final_observation"] = old_observation
                info["final_info"] = old_info

            observations.append(observation)
            infos = self._add_info(infos, info, i)
        self.observations = concatenate(
            self.single_observation_space, observations, self.observations
        )

        return (
            deepcopy(self.observations) if self.copy else self.observations,
            np.copy(self._rewards),
            np.copy(self._terminateds),
            np.copy(self._truncateds),
            infos,
        )


@dataclass
class IiwaRunnerCfg(BaseRunnerCfg):
    env_name: Literal["iiwa"]

    num_env_train: int = 2
    num_env_eval: int = 2
    max_episode_steps: int = 200

    n_repeat: int = 2
    horizon: int = 100
    action_scale: float = 1.0


class IiwaRunner(BaseRunner):
    cfg: IiwaRunnerCfg
    policy: BasePolicy

    seed: int
    q_cameras: dict[str, np.ndarray]

    dynamics_cfg: Any
    planner_cfg: Any

    def __init__(
        self,
        cfg: IiwaRunnerCfg,
        device: torch.device = torch.device("cuda:0"),
    ) -> None:
        super().__init__(cfg, device)

    def setup_env(self):
        if_docker_then_check_display_and_x_server()
        register_resolvers()

        iiwa_env_cfg = load_hydra_cfg(
            path=NEURAL_JACOBIAN_DIR
            / "iiwa"
            / "config"
            # / "wider_initial_condition.yaml"
            / "example.yaml"
        )
        # overriding the time limit for testing
        iiwa_env_cfg.env.time_limit = 10000

        def env_fn():
            cfg = iiwa_env_cfg
            env = IiwaBimanual(
                n_action_steps=cfg.env.n_action_steps,
                n_obs_steps=cfg.env.n_obs_steps,
                time_step_per_action=cfg.env.time_step_per_action,
                observation_update_period=cfg.env.observation_update_period,
                scenario_path=cfg.scenario_path,
                add_logger=cfg.env.add_logger,
                visualize=cfg.env.visualize,
                hardware=cfg.env.hardware,
                time_limit=cfg.env.time_limit,
                hardware_action_hold_time=cfg.env.hardware_action_hold_time,
            )

            return GymSyncComplianceWrapper(ImageEnvWrapper(env, cfg))

        # n_env
        n_envs = self.cfg.num_env_train + self.cfg.num_env_eval
        env_fns = [env_fn] * n_envs

        self.env = SyncVectorEnv(env_fns)
        # self.env = AsyncVectorEnv(env_fns)
        self.set_q_cameras(None)

    def set_q_cameras(self, q_cameras: dict[str, np.ndarray] | None):
        if q_cameras is None:
            camera_0_translation = np.array([0.8, 0.05, 1.5])
            camera_0_rotation = RollPitchYaw(
                np.deg2rad(-168), np.deg2rad(0.2328), np.deg2rad(90.0)
            )

            q_camera_0 = np.concatenate(
                [camera_0_rotation.ToQuaternion().wxyz(), camera_0_translation]
            )

            self.q_cameras = {
                "camera_0": q_camera_0,
            }
        else:
            self.q_cameras = q_cameras

    def get_q_cameras(self) -> dict[str, np.ndarray]:
        return self.q_cameras

    def run(self, policy: BasePolicy, options=None, run_tag: str | None = None):
        """Run one episode for all vectorized envs using the given policy.

        Args:
            policy (BasePolicy): The policy to use for action prediction.
            options (dict, optional): Additional options for environment reset.
        """

        # Reset envs & policy
        n_train = self.cfg.num_env_train
        n_eval = self.cfg.num_env_eval
        n_envs = n_train + n_eval

        starting_seed = 2345
        seeds = np.arange(n_envs) + starting_seed
        seeds = [int(s) for s in seeds]  # force python ints

        # max step info  | TODO: double check to make sure it aligns with iiwa env
        max_steps = self.cfg.max_episode_steps // self.cfg.n_repeat

        # get camera info
        q_cameras = self.get_q_cameras()
        assert len(list(q_cameras.keys())) == 1, "Only support one camera for now"
        camera_name = list(q_cameras.keys())[0]
        color_port_name = get_color_image_port_name_from_camera_name(camera_name)

        ###### reset and grab obs
        obs, info = self.env.reset(
            seed=seeds, options=dict(q_cameras=q_cameras, **(options or {}))
        )
        policy.reset()

        ### define function to extract last obs
        def extract_last_obs(obs):
            rgb_last = obs[color_port_name][:, -1, ..., :3]  # (B, H, W, 4) [0, 1]
            q_r_last = obs["q"][:, -1, 3:]  # (B, nq)
            time_last = obs["time"][:, -1]  # (B, 1)

            return rgb_last, q_r_last, time_last

        rgb_last, q_r_last, time_last = extract_last_obs(obs)

        ran = tqdm.trange(max_steps, desc="[IiwaRunner] Rollout")

        pos_ref = q_r_last.copy()
        update_pos_ref_every = 10

        # Episode tracking
        episode_rewards = np.zeros(n_envs, dtype=np.float32)
        done_flags = np.zeros(n_envs, dtype=bool)

        ret_videos = {
            "clean": [rgb_last],
            "policy": [rgb_last],
        }

        for step_cnt in ran:

            if np.all(done_flags):
                print(cyan("[PushTRunner] All envs done, stopping rollout."))
                break

            # -------------------------------------------------- #
            # Alpha blending of position reference
            # -------------------------------------------------- #

            if step_cnt % update_pos_ref_every == 0:
                alpha = 1.0
                pos_ref = alpha * q_r_last + (1 - alpha) * pos_ref

            # -------------------------------------------------- #
            # Compute the policy output
            # -------------------------------------------------- #
            policy_out: PolicyOutput = policy.predict_action(
                obs=PolicyObservation(
                    rgb=rgb_last,
                    q_robot=q_r_last,
                    rgb_vis=rgb_last,
                )
            )

            v_cmd = policy_out.action * self.cfg.action_scale  # (B, action_dim)
            # v_cmd[..., [2, 5]] *= 0.7

            print("v_cmd:", v_cmd[0])
            print("pos_ref:", pos_ref[0])

            # -------------------------------------------------- #
            # Integrate velocity → position reference
            # -------------------------------------------------- #
            pos_cmd = pos_ref + v_cmd  # NOTE: pos_cmd and pos_ref are different!

            # pos_ref_start = pos_ref
            # pos_ref_target = pos_ref + v_cmd

            # T_inner = self.env.action_space.shape[1]
            # delta_pos = (pos_ref_target - pos_ref_start) / T_inner
            # # time-indexed position reference trajectory
            # action = (
            #     pos_ref_start[:, None, :]
            #     + delta_pos[:, None, :]
            #     * np.arange(1, T_inner + 1, dtype=np.float32)[None, :, None]
            # )  # (B, T_inner, nq)

            for _ in range(self.cfg.n_repeat):
                # expand the dim to match time
                action = np.expand_dims(pos_cmd, axis=1)  # (B, 1, action_dim)
                action = np.repeat(
                    action, repeats=self.env.action_space.shape[1], axis=1
                )

                obs, reward, terminated, truncated, info = self.env.step(action)

            reward = np.asarray(reward, dtype=np.float32)
            terminated = np.asarray(terminated, dtype=bool)
            truncated = np.asarray(truncated, dtype=bool)
            done = np.logical_or(terminated, truncated)

            # accumulate reward only for unfinished envs
            episode_rewards[~done_flags] += reward[~done_flags]
            done_flags = np.logical_or(done_flags, done)

            rgb_last, q_r_last, time_last = extract_last_obs(obs)

            ret_videos["clean"].append(rgb_last)
            # ret_videos["policy"].append(
            #     policy_out.info["policy_vis"]
            #     if policy_out.info is not None
            #     else rgb_last
            # )

            ### rerun logging TODO: this is not going to cover the case for non motion policy
            if rr is not None and self.server_uri is not None:
                rr.set_time("frame_index", timestamp=step_cnt)

                # log clean video
                rr.log("vis/clean", rr.Image(rgb_last[0]))

                # log plan visualization
                if policy_out.info is not None and "policy_vis" in policy_out.info:
                    rr.log(
                        "vis/policy",
                        rr.Image(policy_out.info["policy_vis"]),
                    )

                # log action
                for dim_idx, val in enumerate(v_cmd[0]):
                    rr.log(f"v_cmd/{dim_idx}", rr.Scalars(val.item()))

        return ret_videos


if __name__ == "__main__":
    import hydra
    from omegaconf import OmegaConf

    # ----------------------------------------------------------------------
    # 1. Point Hydra to directory containing iiwa.yaml
    #    Example: project_root/configurations/runners/iiwa.yaml
    #
    #    The directory must contain:
    #      iiwa.yaml
    #        env_name: iiwa
    #        n_repeat: 2
    #        horizon: 100
    #
    #    and anything else your BaseRunner/IiwaRunner requires.
    # ----------------------------------------------------------------------
    config_dir = Path("../../configurations/runners")

    with hydra.initialize(version_base=None, config_path=str(config_dir)):
        cfg = hydra.compose(config_name="iiwa.yaml")

    print("Loaded cfg:\n", OmegaConf.to_yaml(cfg))

    # ----------------------------------------------------------------------
    # 2. Instantiate the runner
    # ----------------------------------------------------------------------
    runner_cfg = IiwaRunnerCfg(**cfg)  # Convert OmegaConf → dataclass
    runner = IiwaRunner(runner_cfg)

    # ----------------------------------------------------------------------
    # 3. Setup + run
    # ----------------------------------------------------------------------
    runner.setup_env()
    runner.run()
