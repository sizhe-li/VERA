from __future__ import annotations
from typing import Any, Dict

import cv2
import mediapy as media
import numpy as np
import torch
from einops import rearrange
from torch import Tensor

from vera.utils import visualize_tracks
from vera.utils.jacobian_utils import visualize_jacobian

from .base_policy import PolicyObservation


class MotionPolicyVisualizationMixin:
    def _annotate_control_views(
        self,
        obs: PolicyObservation,
        image: np.ndarray,
    ) -> np.ndarray:
        if self._obs_view_count(obs) <= 1:
            return image

        annotated = image.copy()
        view_widths = self._view_widths_for_layout(int(image.shape[1]), obs)
        selected = set(self._resolve_control_view_indices(obs))
        start = 0
        for view_index, view_width in enumerate(view_widths):
            end = start + int(view_width)
            border_color = (80, 160, 80) if view_index in selected else (80, 80, 80)
            if view_index not in selected:
                annotated[:, start:end] = (0.55 * annotated[:, start:end]).astype(np.uint8)
            cv2.rectangle(
                annotated,
                (start, 0),
                (max(start, end - 1), max(0, annotated.shape[0] - 1)),
                border_color,
                1,
            )
            if view_index > 0:
                cv2.line(
                    annotated,
                    (start, 0),
                    (start, max(0, annotated.shape[0] - 1)),
                    border_color,
                    1,
                )
            start = end
        return annotated

    @staticmethod
    def _draw_frame_border(
        image: np.ndarray,
        *,
        color: tuple[int, int, int] = (255, 0, 0),
        thickness: int = 4,
    ) -> np.ndarray:
        bordered = image.copy()
        if bordered.ndim < 2:
            return bordered
        height, width = bordered.shape[:2]
        if height <= 0 or width <= 0:
            return bordered
        border_px = max(1, min(int(thickness), max(1, min(height, width) // 2)))
        cv2.rectangle(
            bordered,
            (0, 0),
            (max(0, width - 1), max(0, height - 1)),
            color,
            border_px,
        )
        return bordered

    def _make_context_strip(
        self,
        obs: PolicyObservation,
        context_rgb: Tensor | None,
        *,
        target_width: int,
        max_height: int,
    ) -> np.ndarray | None:
        del obs
        if context_rgb is None or context_rgb.ndim != 5 or context_rgb.shape[1] == 0:
            return None

        thumbnail_h = max(40, int(max_height))
        frames: list[np.ndarray] = []
        context_cpu = context_rgb.detach().float().cpu()
        for ctx_idx in range(int(context_cpu.shape[1])):
            frame = context_cpu[0, ctx_idx].permute(1, 2, 0).numpy().clip(0.0, 1.0)
            frame_uint8 = (frame * 255.0).astype(np.uint8)
            aspect_w = max(
                1,
                int(round(frame_uint8.shape[1] * thumbnail_h / frame_uint8.shape[0])),
            )
            thumb = media.resize_image(frame_uint8, shape=(thumbnail_h, aspect_w))
            frames.append(thumb)

        strip = np.concatenate(frames, axis=1)
        if strip.shape[1] != target_width:
            strip = media.resize_image(strip, shape=(thumbnail_h, target_width))
        return strip

    def _current_obs_for_policy_vis(
        self,
        obs: PolicyObservation,
        frame: Dict[str, Any],
        batch_idx: int,
        *,
        shape: tuple[int, int],
    ) -> np.ndarray:
        current_obs_rgb = frame.get("current_obs_rgb")
        if current_obs_rgb is not None:
            if isinstance(current_obs_rgb, torch.Tensor):
                arr = current_obs_rgb.detach().float().cpu().numpy()
            else:
                arr = np.asarray(current_obs_rgb)
            if arr.ndim == 4:
                arr = arr[batch_idx]
            if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
                arr = np.transpose(arr, (1, 2, 0))
            if arr.ndim == 3 and arr.shape[-1] == 3:
                if np.issubdtype(arr.dtype, np.floating):
                    arr = (arr.clip(0.0, 1.0) * 255.0).astype(np.uint8)
                else:
                    arr = arr.clip(0, 255).astype(np.uint8)
                return media.resize_image(arr, shape=shape)

        return media.resize_image(
            (obs.rgb_vis[batch_idx].copy() * 255).clip(0, 255).astype(np.uint8),
            shape=shape,
        )

    @staticmethod
    def _select_debug_batch_value(value: Any, batch_idx: int) -> Any:
        if value is None:
            return None
        arr = np.asarray(value)
        if arr.ndim == 0:
            return arr.item()
        if arr.shape[0] <= batch_idx:
            return None
        selected = arr[batch_idx]
        if np.isscalar(selected) or np.asarray(selected).ndim == 0:
            return np.asarray(selected).item()
        return np.asarray(selected)

    @classmethod
    def _select_debug_batch_frame_value(
        cls,
        value: Any,
        batch_idx: int,
        frame_idx: int | None,
    ) -> Any:
        selected = cls._select_debug_batch_value(value, batch_idx)
        if selected is None or frame_idx is None:
            return selected
        arr = np.asarray(selected)
        if arr.ndim < 2:
            return selected
        if arr.shape[0] <= frame_idx:
            return None
        selected_frame = arr[frame_idx]
        if np.isscalar(selected_frame) or np.asarray(selected_frame).ndim == 0:
            return np.asarray(selected_frame).item()
        return np.asarray(selected_frame)

    @staticmethod
    def _format_vector(vec: Any, count: int) -> str | None:
        if vec is None:
            return None
        arr = np.asarray(vec).reshape(-1)
        if arr.size < count:
            return None
        return "[" + ", ".join(f"{float(v):+.3f}" for v in arr[:count]) + "]"

    @staticmethod
    def _format_scalar(value: Any) -> str | None:
        if value is None:
            return None
        return f"{float(value):+.3f}"

    @staticmethod
    def _format_bool_flag(value: Any) -> str | None:
        if value is None:
            return None
        return "1" if bool(value) else "0"

    @staticmethod
    def _render_text_panel(
        width: int,
        lines: list[tuple[str, tuple[int, int, int]]],
        *,
        background: tuple[int, int, int] = (18, 18, 18),
    ) -> np.ndarray | None:
        if width <= 0 or not lines:
            return None
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.48
        thickness = 1
        pad_x = 8
        pad_y = 8
        line_gap = 7
        line_height = 0
        for text, _ in lines:
            (_, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
            line_height = max(line_height, text_h)
        panel_h = pad_y * 2 + len(lines) * line_height + max(0, len(lines) - 1) * line_gap
        panel = np.full((panel_h, width, 3), background, dtype=np.uint8)
        y = pad_y + line_height
        for text, color in lines:
            cv2.putText(
                panel,
                text,
                (pad_x, y),
                font,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA,
            )
            y += line_height + line_gap
        return panel

    def _control_view_legend_text(self, obs: PolicyObservation) -> str | None:
        view_count = self._obs_view_count(obs)
        if view_count <= 1:
            return None
        selected = set(self._resolve_control_view_indices(obs))
        entries = []
        for view_index in range(view_count):
            label_name = (
                obs.view_keys[view_index]
                if obs.view_keys is not None and view_index < len(obs.view_keys)
                else f"view_{view_index}"
            )
            label_state = "ctrl" if view_index in selected else "vis"
            entries.append(f"{label_name}[{label_state}]")
        return "views: " + " | ".join(entries)

    def _make_policy_vis_header(
        self,
        obs: PolicyObservation,
        *,
        panel_titles: list[str],
        step_index: int | None = None,
        dream_index: int | None = None,
        width: int,
    ) -> np.ndarray | None:
        lines: list[tuple[str, tuple[int, int, int]]] = []
        lines.append((" | ".join(panel_titles), (230, 230, 230)))
        meta_parts: list[str] = []
        if step_index is not None:
            meta_parts.append(f"step: {int(step_index)}")
        if dream_index is not None:
            meta_parts.append(f"dream: {int(dream_index)}")
        view_legend = self._control_view_legend_text(obs)
        if view_legend is not None:
            meta_parts.append(view_legend)
        if meta_parts:
            lines.append(("   ".join(meta_parts), (170, 220, 255)))
        return self._render_text_panel(width, lines, background=(24, 24, 24))

    def _threshold_jacobian_for_vis(self, jacobian: torch.Tensor) -> torch.Tensor:
        threshold = float(getattr(self.cfg, "jacobian_vis_abs_threshold", 0.0) or 0.0)
        if threshold <= 0.0:
            return jacobian
        return torch.where(
            jacobian.abs() < threshold,
            torch.zeros_like(jacobian),
            jacobian,
        )

    def _visualize_jacobian_per_view(
        self,
        jacobian_5d: torch.Tensor,
        obs: PolicyObservation,
    ) -> np.ndarray:
        """Render the Jacobian panel once per camera view, then concatenate.

        ``jacobian_5d`` has shape ``[1, cmd, s, h, w]`` where ``w`` is the
        full stitched multi-view width. ``visualize_jacobian``'s colormap
        pipeline does per-image min/max normalization on the full width,
        so one view's outlier pixels can crush every other view to black
        (the "pitch black jacobian" bug). Splitting along the width axis
        into per-view tiles and calling ``visualize_jacobian`` independently
        on each gives each view its own min/max range, matching training-
        time behavior in ``image_jacobian.py`` which loops over views.
        """
        jacobian_5d = self._threshold_jacobian_for_vis(jacobian_5d)
        view_count = max(1, int(self._obs_view_count(obs)))
        view_widths = list(obs.view_widths or [])
        total_w = int(jacobian_5d.shape[-1])
        if view_count <= 1 or total_w < view_count:
            return visualize_jacobian(
                jacobian=jacobian_5d,
                robot_name=self.robot_name,
                flow_scale=0.1,
            )
        # Work out per-view width in jacobian-space. If the caller gave us
        # pixel widths we rescale them; otherwise assume equal splits.
        if view_widths and len(view_widths) == view_count and sum(view_widths) > 0:
            scale = total_w / float(sum(view_widths))
            scaled = [max(1, int(round(w * scale))) for w in view_widths]
            scaled[-1] += total_w - sum(scaled)
            jac_view_widths = scaled
        else:
            per = total_w // view_count
            jac_view_widths = [per] * view_count
            jac_view_widths[-1] += total_w - per * view_count
        tiles: list[np.ndarray] = []
        start = 0
        for w in jac_view_widths:
            end = start + w
            tile = visualize_jacobian(
                jacobian=jacobian_5d[..., start:end],
                robot_name=self.robot_name,
                flow_scale=0.1,
            )
            # visualize_jacobian returns np.ndarray uint8, shape [3, H, W].
            tiles.append(np.asarray(tile))
            start = end
        return np.concatenate(tiles, axis=-1)

    def _make_action_hud_lines(
        self,
        *,
        batch_idx: int,
        frame_idx: int | None = None,
        action_debug: dict[str, Any] | None = None,
        gripper_debug: dict[str, Any] | None = None,
    ) -> list[tuple[str, tuple[int, int, int]]]:
        del gripper_debug
        action_final = self._select_debug_batch_frame_value(
            None if action_debug is None else action_debug.get("action_final"),
            batch_idx,
            frame_idx,
        )
        lines: list[tuple[str, tuple[int, int, int]]] = []
        if action_final is not None:
            action_arr = np.asarray(action_final).reshape(-1)
            preview = np.array2string(
                action_arr[: min(6, action_arr.shape[0])],
                precision=3,
                suppress_small=True,
            )
            suffix = " ..." if action_arr.shape[0] > 6 else ""
            lines.append((f"commanded u[:6]: {preview}{suffix}", (235, 235, 235)))
            lines.append(
                (f"|commanded u|: {float(np.linalg.norm(action_arr)):.3f}", (255, 210, 120))
            )
        return lines

    def _draw_action_on_frame(
        self,
        frame: np.ndarray,
        action: np.ndarray | None,
    ) -> np.ndarray:
        del action
        return frame

    def _make_policy_vis(
        self,
        obs: PolicyObservation,
        policy_outputs: Dict[str, Tensor],
        *,
        action_debug: dict[str, Any] | None = None,
        gripper_debug: dict[str, Any] | None = None,
    ) -> np.ndarray:
        assert obs.rgb_vis is not None, "rgb_vis required for policy visualization"
        vis_h, vis_w = self.image_size_motion_planner
        # Render at the valid (un-padded) multiview width so square views are not stretched across the
        # model's padded canvas (e.g. two 128x128 mimicgen views shown at 256 wide, not the 576 omni
        # canvas). The dream is already pad-cropped upstream; match it here for the obs/dream panels.
        if obs.view_widths is not None and len(obs.view_widths) >= 1:
            _valid_w = int(sum(int(w) for w in obs.view_widths))
            if 0 < _valid_w <= int(vis_w):
                vis_w = _valid_w
        jac_h, jac_w = self._concat_dynamics_size(self._obs_view_count(obs))
        jacobian = rearrange(
            policy_outputs["jacobian"],
            "b (h w s) c -> b c s h w",
            h=jac_h,
            w=jac_w,
        )
        jacobian = self.denormalize_dynamics_jacobian(jacobian)

        vis_batch = []
        for batch_idx in range(obs.rgb_vis.shape[0]):
            curr_track = policy_outputs["curr_track"][batch_idx]
            trgt_track = policy_outputs["trgt_track"][batch_idx]

            vis_policy_output = media.resize_image(
                (obs.rgb_vis[batch_idx].copy() * 255).clip(0, 255).astype(np.uint8),
                shape=(vis_h, vis_w),
            )
            vis_policy_output = self._annotate_control_views(obs, vis_policy_output)
            action_final = self._select_debug_batch_value(
                None if action_debug is None else action_debug.get("action_final"),
                batch_idx,
            )
            vis_policy_output = self._draw_action_on_frame(
                vis_policy_output,
                None if action_final is None else np.asarray(action_final),
            )

            vis_policy_output = visualize_tracks.draw_curr_trgt_tracks(
                obs=vis_policy_output,
                curr_track=curr_track,
                trgt_track=trgt_track,
                curr_visible=torch.ones_like(curr_track[..., 0]),
                point_radius=1,
                point_color=(0, 0, 0),
                arrow_color=(0, 255, 0),
                arrow_thickness=1,
                sparsity=self.cfg.vis_track_sparsity,
            )

            vis_jacobian = rearrange(
                self._visualize_jacobian_per_view(
                    jacobian[batch_idx : batch_idx + 1],
                    obs,
                ),
                "c h w -> h w c",
            )

            target_h = vis_policy_output.shape[0]
            if vis_jacobian.shape[0] != target_h:
                target_w = max(
                    1,
                    int(round(vis_jacobian.shape[1] * target_h / vis_jacobian.shape[0])),
                )
                vis_jacobian = media.resize_image(vis_jacobian, shape=(target_h, target_w))

            vis = np.concatenate([vis_policy_output, vis_jacobian], axis=1)
            context_strip = self._make_context_strip(
                obs,
                policy_outputs.get("context_rgb"),
                target_width=vis.shape[1],
                max_height=max(40, vis_h // 4),
            )
            header = self._make_policy_vis_header(
                obs,
                panel_titles=["current + tracks", "jacobian"],
                step_index=getattr(obs, "step_index", None),
                width=vis.shape[1],
            )
            footer = self._render_text_panel(
                vis.shape[1],
                self._make_action_hud_lines(
                    batch_idx=batch_idx,
                    action_debug=action_debug,
                    gripper_debug=gripper_debug,
                ),
                background=(14, 14, 14),
            )
            stacked_panels = [
                panel for panel in (header, vis, footer) if panel is not None  # filmstrip removed
            ]
            vis = np.concatenate(stacked_panels, axis=0)
            vis_batch.append(vis.astype(np.float32) / 255.0)

        return np.stack(vis_batch, axis=0)

    def _make_policy_vis_joint(
        self,
        obs: PolicyObservation,
        frames: list[Dict[str, Tensor]],
        dream_index: int | None = None,
        action: np.ndarray | None = None,
        *,
        action_debug: dict[str, Any] | None = None,
        gripper_debug: dict[str, Any] | None = None,
    ) -> np.ndarray:
        assert obs.rgb_vis is not None, "rgb_vis required for policy visualization"
        vis_h, vis_w = self._concat_dynamics_size(self._obs_view_count(obs))  # tracks live in dynamics-model pixel space (e.g. 252), NOT the planner canvas (vis-only fix)
        # Render at the valid (un-padded) multiview width so square views are not stretched across the
        # model's padded canvas (e.g. two 128x128 mimicgen views shown at 256 wide, not the 576 omni
        # canvas). The dream is already pad-cropped upstream; match it here for the obs/dream panels.
        if obs.view_widths is not None and len(obs.view_widths) >= 1:
            _valid_w = int(sum(int(w) for w in obs.view_widths))
            if 0 < _valid_w <= int(vis_w):
                vis_w = _valid_w
        jac_h, jac_w = self._concat_dynamics_size(self._obs_view_count(obs))
        vis_batch = []

        for frame_idx, frame in enumerate(frames):
            rgb = frame["rgb"]
            jacobian = rearrange(
                frame["jacobian"],
                "b (h w s) c -> b c s h w",
                h=jac_h,
                w=jac_w,
            )
            jacobian = self.denormalize_dynamics_jacobian(jacobian)

            for batch_idx in range(rgb.shape[0]):
                curr_track = frame["curr_track"][batch_idx]
                trgt_track = frame["trgt_track"][batch_idx]
                curr_visible = frame.get("curr_visible")
                if curr_visible is None:
                    curr_visible = torch.ones_like(curr_track[..., 0], dtype=torch.bool)
                else:
                    curr_visible = curr_visible[batch_idx]
                control_visible = frame.get("control_visible")
                if control_visible is None:
                    control_visible = curr_visible
                else:
                    control_visible = control_visible[batch_idx]

                current_obs = self._current_obs_for_policy_vis(
                    obs,
                    frame,
                    batch_idx,
                    shape=(vis_h, vis_w),
                )
                current_obs = self._annotate_control_views(obs, current_obs)
                act = None
                if action is not None:
                    act = action[batch_idx] if action.ndim > 1 else action
                current_obs = self._draw_action_on_frame(current_obs, act)

                rgb_np = (
                    rgb[batch_idx].permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255
                ).astype(np.uint8)
                vis_source = media.resize_image(rgb_np, shape=(vis_h, vis_w))
                vis_source = self._annotate_control_views(obs, vis_source)
                is_boundary_source_frame = bool(
                    frame.get(
                        "is_boundary_source_frame",
                        frame.get("is_first_dream_frame", action is None and frame_idx == 0),
                    )
                )
                if is_boundary_source_frame:
                    vis_source = self._draw_frame_border(vis_source)
                vis_source = visualize_tracks.draw_curr_trgt_tracks_dense(
                    obs=vis_source,
                    curr_track=curr_track,
                    trgt_track=trgt_track,
                    curr_visible=control_visible,
                    point_radius=1,
                    point_color=(0, 0, 0),
                    arrow_color=(0, 255, 0),
                    arrow_thickness=1,
                    sparsity=self.cfg.vis_track_sparsity_joint,
                    motion_thresh=1,
                    arrow_scale=0.3,
                )

                vis_jacobian = rearrange(
                    self._visualize_jacobian_per_view(
                        jacobian[batch_idx : batch_idx + 1],
                        obs,
                    ),
                    "c h w -> h w c",
                )

                target_h = current_obs.shape[0]
                if vis_jacobian.shape[0] != target_h:
                    target_w = max(
                        1,
                        int(round(vis_jacobian.shape[1] * target_h / vis_jacobian.shape[0])),
                    )
                    vis_jacobian = media.resize_image(vis_jacobian, shape=(target_h, target_w))

                vis = np.concatenate([current_obs, vis_source, vis_jacobian], axis=1)
                context_strip = self._make_context_strip(
                    obs,
                    frame.get("context_rgb"),
                    target_width=vis.shape[1],
                    max_height=max(40, vis_h // 4),
                )
                source_frame_role = str(frame.get("source_frame_role", "source"))
                target_frame_role = str(frame.get("target_frame_role", "target"))
                source_panel_title = f"{source_frame_role} + tracks -> {target_frame_role}"
                header = self._make_policy_vis_header(
                    obs,
                    panel_titles=["current", source_panel_title, "jacobian"],
                    step_index=getattr(obs, "step_index", None),
                    dream_index=dream_index,
                    width=vis.shape[1],
                )
                footer = self._render_text_panel(
                    vis.shape[1],
                    self._make_action_hud_lines(
                        batch_idx=batch_idx,
                        frame_idx=frame_idx,
                        action_debug=action_debug,
                        gripper_debug=gripper_debug,
                    ),
                    background=(14, 14, 14),
                )
                stacked_panels = [
                    panel for panel in (header, vis, footer) if panel is not None  # filmstrip removed
                ]
                vis = np.concatenate(stacked_panels, axis=0)
                vis_batch.append(vis.astype(np.float32) / 255.0)

        if not vis_batch:
            return np.zeros((0, vis_h, vis_w * 3, 3), dtype=np.float32)
        return np.stack(vis_batch, axis=0)
