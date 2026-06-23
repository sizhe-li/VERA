# okto/env_runner/pusht_image_env.py
import cv2
import gymnasium as gym
import numpy as np
from gym_pusht.envs.pusht import PushTEnv


class PushTImageEnv(gym.Wrapper):
    """
    Wrap PushTEnv to add an image observation keyed by "image",
    and manage cached rendering so vectorized envs return (N, H, W, 3).
    """

    env: PushTEnv

    def __init__(self, env: PushTEnv, render_size=252):
        super().__init__(env)
        self.render_size = render_size

        # Build new observation space
        self.observation_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.render_size, self.render_size, 3),
                    dtype=np.float32,
                ),
                "image_vis": gym.spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.render_size, self.render_size, 3),
                    dtype=np.float32,
                ),
                "state": self.env.observation_space,
            }
        )

        self._cached_img = None

    # --------------------------------------------------
    def _obs(self, obs_state):
        img_clean = self.env.render()
        img_clean = cv2.resize(img_clean, (self.render_size, self.render_size))

        img_vis = self.env._render(visualize=True, render_action=True)
        img_vis = cv2.resize(img_vis, (self.render_size, self.render_size))

        self._cached_img = img_clean

        return {
            "image": img_clean.astype(np.float32) / 255.0,  # what the model sees
            "image_vis": img_vis.astype(np.float32) / 255.0,  # for visualization
            "state": obs_state.astype(np.float32),
        }

    # --------------------------------------------------
    def reset(self, seed=None, options=None):
        obs_state, info = self.env.reset(seed=seed, options=options)
        return self._obs(obs_state), info

    # --------------------------------------------------
    def step(self, action):
        obs_state, reward, terminated, truncated, info = self.env.step(action)
        return self._obs(obs_state), reward, terminated, truncated, info

    # --------------------------------------------------
    def render(self):
        # Vector env will collect batched renders
        if self._cached_img is None:
            self._cached_img = self.env.render()
        return self._cached_img
