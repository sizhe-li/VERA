"""Closed-loop mimicgen (robosuite) backend for the controller.

Lets the SAME client-server protocol be tested closed-loop in sim: the robosuite env lives on the
client side (here), the policy lives on the server; the controller's loop sends obs -> infer ->
applies the returned action to the env, which responds. The sim analogue of RobotEnvBackend.

Implements the RobotBackend contract over the env_runner's gym vector env (num_envs=1): get_state()
returns the 2 mimicgen views (uint8) + proprio from the latest obs, apply_action() steps the env
with the policy's eef-delta action (D=7, Box[-1,1]), reset() starts a fresh episode. Heavy deps
(robosuite/mimicgen) load lazily; needs an EGL-capable GPU (MUJOCO_GL=egl, set here).
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("vera.controller.mimicgen_sim")

_PROPRIO_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]


class MimicgenSimBackend:
    def __init__(
        self,
        view_keys: List[str],
        *,
        dataset_path: str,
        render_size: int = 128,
        context_frames: int = 9,
        max_steps: int = 400,
        flip_images: bool = False,   # env obs come through the same pipeline as the training data
    ) -> None:
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        from vera.env_runner.mimicgen_runner import MimicgenRunner, MimicgenRunnerCfg

        self.view_keys = list(view_keys)
        self._flip = bool(flip_images)
        self._max_steps = int(max_steps)
        cfg = MimicgenRunnerCfg(
            env_name="mimicgen", dataset_path=dataset_path, render_size=render_size,
            render_obs_key=list(view_keys), num_demos_to_run=1, max_episode_steps=self._max_steps,
            n_repeat=1, action_scale=1.0, save_videos=False, save_trajectory=False, save_rrd=False,
            use_stored_model=False, demo_warmup_steps=max(int(context_frames) - 1, 0),
            log_step_debug=False,
        )
        self._runner = MimicgenRunner(cfg, device="cpu")
        self._runner.setup_env()
        self._venv = self._runner.env          # gym SyncVectorEnv, num_envs=1
        self._obs: Optional[dict] = None
        self._done = False
        self._steps = 0
        logger.info("MimicgenSimBackend: env up (dataset=%s, action_space=%s)",
                    dataset_path, self._venv.action_space)

    def _img(self, batched) -> np.ndarray:
        a = np.asarray(batched)[0]                     # unbatch (1,H,W,3) -> (H,W,3)
        if np.issubdtype(a.dtype, np.floating):        # env gives float [0,1]; the wire wants uint8
            a = (np.clip(a, 0.0, 1.0) * 255).astype(np.uint8)
        else:
            a = a.astype(np.uint8)
        if self._flip:
            a = a[::-1]
        return np.ascontiguousarray(a)

    def reset(self) -> None:
        out = self._venv.reset()
        self._obs = out[0] if isinstance(out, tuple) else out
        self._done = False
        self._steps = 0

    def get_state(self) -> Dict[str, Any]:
        assert self._obs is not None, "call reset() before get_state()"
        state: Dict[str, Any] = {}
        for k in self.view_keys:
            if k not in self._obs:
                raise KeyError(f"env obs missing view '{k}'; has {sorted(self._obs)[:12]}")
            state[k] = self._img(self._obs[k])
        for pk in _PROPRIO_KEYS:
            if pk in self._obs:
                state[pk] = np.asarray(self._obs[pk], dtype=np.float32)[0]
        state["_done"] = bool(self._done) or self._steps >= self._max_steps
        return state

    def apply_action(self, action: np.ndarray, action_space: str) -> None:
        a = np.asarray(action, dtype=np.float32).reshape(1, -1)   # (1, 7) for the vector env
        obs, _reward, terminated, truncated, _info = self._venv.step(a)
        self._obs = obs
        self._done = bool(np.any(terminated)) or bool(np.any(truncated))
        self._steps += 1
