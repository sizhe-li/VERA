"""Vendored PushT environment variant used by VERA's PushT runner and checkpoints.

Subclasses ``gym_pusht.envs.pusht.PushTEnv`` (upstream ``gym-pusht==0.1.5``,
Apache-2.0, huggingface/gym-pusht) and adds the modifications VERA was trained
and evaluated with. Upstream 0.1.5 does not include these, so the runner cannot
use the upstream class directly:

1. ``control_type="velocity"``: the action is a desired agent velocity (px/s,
   clipped to ``max_velocity``) instead of a PD position target. VERA's runner
   integrates the policy's du into velocity commands and requires this mode.
2. ``render_action`` flag: disables the action marker in rendered frames.
3. Background rendering: frames are drawn over ``light-gray-floor.png`` (shipped
   next to this file) instead of upstream's plain white fill. The released VERA
   PushT planner/IDM checkpoints were trained on renders with this background;
   plain-white frames are out-of-distribution for them.
4. ``_set_state`` applies rotation BEFORE translation. Upstream keeps a legacy
   order that corrupts the block's geometric pose for nonzero angles; VERA's
   zarr-seeded evaluation resets (``reset_to_state``) rely on the corrected order.
5. ``success_threshold`` is a constructor parameter, default 0.82 — the value
   VERA's internal training/evaluation env used. NOTE: upstream's default is
   0.95. The env reward is ``clip(coverage / success_threshold, 0, 1)`` and
   episode success is ``coverage > success_threshold``, so this value changes
   both the reward normalization and the env's own success criterion. Pass 0.95
   to recover upstream behavior.
"""

import os.path

import cv2
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
from gym_pusht.envs.pusht import PushTEnv
from gymnasium import spaces
from pymunk.vec2d import Vec2d

# Use gym-pusht's bundled draw options (the renderer upstream PushTEnv itself
# uses) rather than pymunk's stock ``pygame_util.DrawOptions``. The two shade
# shapes differently, and the released PushT planner/IDM checkpoints were
# trained on frames rendered by the bundled version.
from gym_pusht.envs.pymunk_override import DrawOptions

_BACKGROUND_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "light-gray-floor.png")


class VeraPushTEnv(PushTEnv):
    def __init__(
        self,
        obs_type="state",
        render_mode="rgb_array",
        block_cog=None,
        damping=None,
        observation_width=96,
        observation_height=96,
        visualization_width=680,
        visualization_height=680,
        render_action: bool = True,
        control_type: str = "position",
        max_velocity: float = 512.0,
        success_threshold: float = 0.82,
    ):
        super().__init__(
            obs_type=obs_type,
            render_mode=render_mode,
            block_cog=block_cog,
            damping=damping,
            observation_width=observation_width,
            observation_height=observation_height,
            visualization_width=visualization_width,
            visualization_height=visualization_height,
        )
        self.render_action = render_action

        if control_type not in ["position", "velocity"]:
            raise ValueError(f"Unknown control_type {control_type}")
        self.control_type = control_type
        self.max_velocity = max_velocity

        # velocity actions can be negative; replace the [0, 512] position box
        if self.control_type == "velocity":
            mv = float(self.max_velocity)
            self.action_space = spaces.Box(low=-mv, high=mv, shape=(2,), dtype=np.float32)

        self.success_threshold = float(success_threshold)

    def step(self, action):
        self.n_contact_points = 0
        n_steps = int(1 / (self.dt * self.control_hz))
        self._last_action = action

        for _ in range(n_steps):
            if self.control_type == "position":
                # PD position control (upstream behaviour)
                acceleration = self.k_p * (action - self.agent.position) + self.k_v * (
                    Vec2d(0, 0) - self.agent.velocity
                )
                self.agent.velocity += acceleration * self.dt
            else:
                # velocity control: action is desired velocity (px/s)
                if isinstance(action, Vec2d):
                    desired_vel = action
                else:
                    desired_vel = Vec2d(float(action[0]), float(action[1]))
                mv = self.max_velocity
                desired_vel = Vec2d(
                    max(min(desired_vel.x, mv), -mv), max(min(desired_vel.y, mv), -mv)
                )
                self.agent.velocity = desired_vel

            self.space.step(self.dt)

        coverage = self._get_coverage()
        reward = np.clip(coverage / self.success_threshold, 0.0, 1.0)
        terminated = is_success = coverage > self.success_threshold

        observation = self.get_obs()
        info = self._get_info()
        info["is_success"] = is_success
        info["coverage"] = coverage

        truncated = False
        return observation, reward, terminated, truncated, info

    def _draw(self):
        screen = pygame.Surface((512, 512))
        # textured floor background (training-render parity; upstream fills white)
        bkgd_img = pygame.image.load(_BACKGROUND_PNG)
        bkgd_img = pygame.transform.scale(bkgd_img, (512, 512))
        screen.blit(bkgd_img, (0, 0))

        draw_options = DrawOptions(screen)

        # Draw goal pose
        goal_body = self.get_goal_pose_body(self.goal_pose)
        for shape in self.block.shapes:
            goal_points = [goal_body.local_to_world(v) for v in shape.get_vertices()]
            goal_points = [
                pymunk.pygame_util.to_pygame(point, draw_options.surface)
                for point in goal_points
            ]
            goal_points += [goal_points[0]]
            pygame.draw.polygon(screen, pygame.Color("LightGreen"), goal_points)

        # Draw agent and block
        self.space.debug_draw(draw_options)
        return screen

    def _get_img(self, screen, width, height, render_action=False):
        img = np.transpose(np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2))
        img = cv2.resize(img, (width, height))
        render_size = min(width, height)
        if render_action and self._last_action is not None:
            action = np.array(self._last_action)
            # the marker is a position; convert velocity actions to a position
            if self.control_type == "velocity":
                action = np.array(self.agent.position) + action
            coord = (action / 512 * [height, width]).astype(np.int32)
            marker_size = int(8 / 96 * render_size)
            thickness = int(1 / 96 * render_size)
            cv2.drawMarker(
                img,
                coord,
                color=(255, 0, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=marker_size,
                thickness=thickness,
            )
        return img

    def render(self):
        return self._render(visualize=True, render_action=self.render_action)

    def _render(self, visualize=False, render_action=False):
        width, height = (
            (self.visualization_width, self.visualization_height)
            if visualize
            else (self.observation_width, self.observation_height)
        )
        screen = self._draw()

        if self.render_mode == "rgb_array":
            return self._get_img(screen, width=width, height=height, render_action=render_action)
        return super()._render(visualize=visualize)

    def _set_state(self, state):
        """Set the true geometric pose from [agent_x, agent_y, block_x, block_y, block_angle].

        Rotation must be applied BEFORE translation for non-symmetric bodies,
        otherwise the position is corrupted by the center-of-mass offset
        (upstream keeps the legacy opposite order for old-data compatibility).
        """
        self.agent.position = list(state[:2])
        self.agent.velocity = (0, 0)

        self.block.angle = float(state[4])  # rotate FIRST about COM
        self.block.position = list(state[2:4])  # then translate geometrically
        self.block.velocity = (0, 0)
        self.block.angular_velocity = 0.0

        self.space.step(self.dt)
