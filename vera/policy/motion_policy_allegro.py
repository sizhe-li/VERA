"""
Allegro-specific motion policy wrapper that denormalizes joint-delta actions
before they are consumed by the Allegro runner / environment path.
"""

from typing import Any, Dict
from dataclasses import dataclass
import warnings

from einops import rearrange
import mediapy as media
import numpy as np
import torch
from torch import Tensor

from vera.datasets.normalization import (
    compute_jacobian_action_scales,
    get_flow_normalization_metadata,
)
from vera.utils import alltracker, visualize_tracks
from vera.utils.jacobian_utils import visualize_jacobian

from .base_policy import PolicyObservation, PolicyOutput
from .motion_policy import MotionPolicy, MotionPolicyCfg


@dataclass
class MotionPolicyAllegroCfg(MotionPolicyCfg):
    """Config for Allegro motion policy.

    Pass name='motion_policy_allegro' when constructing the config.
    """


class MotionPolicyAllegro(MotionPolicy):
    """MotionPolicy variant that returns physical Allegro joint deltas."""

    cfg: MotionPolicyAllegroCfg

    def __init__(self, cfg: MotionPolicyAllegroCfg, device: torch.device):
        super().__init__(cfg, device)
        self._warned_missing_action_denorm_meta = False
        self._warned_missing_ls_denorm_meta = False
        self._physical_action_passthrough_remaining = 0

    def reset(self):
        super().reset()
        self._physical_action_passthrough_remaining = 0

    def _debug_joint_planner_multiview_metadata(
        self,
        *,
        phase: str,
        obs: PolicyObservation,
        planner_view_widths: list[int],
        context_rgb_shape: tuple[int, ...],
        record: dict[str, Any] | None = None,
    ) -> None:
        fields = [
            "[MotionPolicyAllegro] planner multiview",
            f"phase={phase}",
            f"context_rgb_shape={context_rgb_shape}",
            f"obs_view_keys={obs.view_keys}",
            f"obs_view_widths={obs.view_widths}",
            f"planner_view_widths={planner_view_widths}",
        ]
        if record is not None:
            tracks = record.get("motion_tracks")
            fields.append(f"record_context_len={record.get('context_len')}")
            if isinstance(tracks, dict):
                meta = tracks.get("meta", {})
                fields.extend(
                    [
                        f"track_image_size={tracks.get('image_size')}",
                        f"track_disp_shape={tuple(tracks['disp'].shape)}",
                        f"track_meta_view_widths={meta.get('view_widths')}",
                        f"track_meta_view_keys={meta.get('view_keys')}",
                        f"per_view_track_counts={meta.get('per_view_track_counts')}",
                        f"track_temporal_stride={record.get('track_temporal_stride')}",
                        f"track_sample_indices={record.get('track_sample_indices')}",
                    ]
                )
        print(" ".join(fields), flush=True)

    def get_wire_metadata(self) -> Dict[str, Any]:
        meta = self.get_dynamics_normalization_metadata()
        action_abs_scale_raw = meta.get("action_abs_scale")
        action_abs_scale = (
            [float(x) for x in action_abs_scale_raw]
            if action_abs_scale_raw is not None
            else []
        )
        return {
            "action_mode": str(meta.get("action_mode", "absolute")),
            "action_abs_scale": action_abs_scale,
            "action_normalization_mode": str(
                meta.get("action_normalization_mode", "symmetric_percentile")
            ),
            "action_percentile": float(meta.get("action_percentile", 90.0)),
            "dim_u": int(len(action_abs_scale)) if action_abs_scale else 16,
            "robot_name": str(
                getattr(self, "robot_name", meta.get("robot_name", "allegro"))
            ),
            "actions_already_metric": True,
            "context_frames": int(self.context_frames),
            "action_chunk_horizon": int(self.cfg.action_chunk_horizon),
            "n_action_steps": int(self.cfg.n_action_steps),
        }

    def _has_action_denorm_metadata(self) -> bool:
        meta = self.get_dynamics_normalization_metadata()
        return any(
            meta.get(key) is not None
            for key in (
                "action_abs_scale",
                "action_mean",
                "action_std",
                "action_min",
                "action_max",
            )
        ) or float(meta.get("du_scale", 1.0)) != 1.0

    def get_final_action(self, du: Tensor) -> Tensor:
        if self._physical_action_passthrough_remaining > 0:
            self._physical_action_passthrough_remaining -= 1
            return du

        if self._has_action_denorm_metadata():
            return self.denormalize_dynamics_action(du)

        if not self._warned_missing_action_denorm_meta:
            warnings.warn(
                "MotionPolicyAllegro is missing action normalization metadata; "
                "using legacy scaled action units.",
                stacklevel=2,
            )
            self._warned_missing_action_denorm_meta = True
        return du

    def _mark_actions_already_physical(self, num_actions: int) -> None:
        self._physical_action_passthrough_remaining += max(int(num_actions), 0)

    @staticmethod
    def _match_vector_length(
        values: list[float],
        target_dim: int,
        *,
        default: float,
    ) -> list[float]:
        if target_dim <= 0:
            return []
        if not values:
            return [default] * target_dim
        if len(values) >= target_dim:
            return values[:target_dim]
        return values + [values[-1]] * (target_dim - len(values))

    def _allegro_action_scale_tensor(
        self,
        reference: Tensor,
        *,
        cmd_dim: int,
    ) -> Tensor:
        meta = self.get_dynamics_normalization_metadata()
        action_scales = compute_jacobian_action_scales(
            action_dim=cmd_dim,
            du_scale=float(meta.get("du_scale", meta.get("action_pre_scale", 1.0))),
            action_mean=meta.get("action_mean"),
            action_std=meta.get("action_std"),
            action_min=meta.get("action_min"),
            action_max=meta.get("action_max"),
            action_abs_scale=meta.get("action_abs_scale"),
        )
        action_scales = self._match_vector_length(action_scales, cmd_dim, default=1.0)
        return torch.as_tensor(
            action_scales,
            device=reference.device,
            dtype=reference.dtype,
        )

    def _allegro_flow_scale_tensor(
        self,
        reference: Tensor,
        *,
        state_dim: int,
    ) -> Tensor:
        meta = self.get_dynamics_normalization_metadata()
        flow_meta = get_flow_normalization_metadata(meta)
        flow_scales: list[float]
        if (
            flow_meta["flow_normalization_mode"] == "symmetric_percentile"
            and flow_meta["oflow_abs_scale"] is not None
        ):
            flow_scales = [float(v) for v in flow_meta["oflow_abs_scale"]]
        elif (
            flow_meta["flow_normalization_mode"] == "percentile_minmax"
            and flow_meta["oflow_percentile_min"] is not None
            and flow_meta["oflow_percentile_max"] is not None
        ):
            flow_scales = [
                0.5 * (float(mx) - float(mn))
                for mn, mx in zip(
                    flow_meta["oflow_percentile_min"],
                    flow_meta["oflow_percentile_max"],
                )
            ]
        else:
            flow_scales = [1.0 / float(flow_meta["oflow_scale"])] * state_dim
        flow_scales = self._match_vector_length(flow_scales, state_dim, default=1.0)
        return torch.as_tensor(
            flow_scales,
            device=reference.device,
            dtype=reference.dtype,
        )

    def _warn_missing_ls_denorm_meta(self) -> None:
        if self._warned_missing_ls_denorm_meta:
            return
        warnings.warn(
            "MotionPolicyAllegro is missing flow/action normalization metadata; "
            "using legacy model-space Jacobian units for LS solves.",
            stacklevel=2,
        )
        self._warned_missing_ls_denorm_meta = True

    def _can_denormalize_ls_jacobian(self) -> bool:
        meta = self.get_dynamics_normalization_metadata()
        has_flow = any(
            meta.get(key) is not None
            for key in (
                "oflow_abs_scale",
                "oflow_percentile_min",
                "oflow_percentile_max",
            )
        ) or float(meta.get("oflow_scale", 1.0)) != 1.0
        has_action = self._has_action_denorm_metadata()
        return has_flow or has_action

    def _denormalize_allegro_jacobian_pixel(self, jacobian_pixel: Tensor) -> Tensor:
        if not self._can_denormalize_ls_jacobian():
            self._warn_missing_ls_denorm_meta()
            return jacobian_pixel
        state_dim = int(jacobian_pixel.shape[-2])
        cmd_dim = int(jacobian_pixel.shape[-1])
        action_scale = self._allegro_action_scale_tensor(
            jacobian_pixel,
            cmd_dim=cmd_dim,
        ).view(1, 1, 1, cmd_dim)
        flow_scale = self._allegro_flow_scale_tensor(
            jacobian_pixel,
            state_dim=state_dim,
        ).view(1, 1, state_dim, 1)
        return jacobian_pixel * flow_scale * action_scale

    def _denormalize_allegro_jacobian_flat(
        self,
        jacobian_flat: Tensor,
        *,
        state_dim: int = 2,
    ) -> Tensor:
        if not self._can_denormalize_ls_jacobian():
            self._warn_missing_ls_denorm_meta()
            return jacobian_flat
        if jacobian_flat.shape[1] % state_dim != 0:
            raise ValueError(
                f"Expected flattened Jacobian rows divisible by state_dim={state_dim}, "
                f"got {jacobian_flat.shape}"
            )
        jacobian_pixel = rearrange(
            jacobian_flat,
            "b (n s) c -> b n s c",
            s=state_dim,
        )
        jacobian_pixel = self._denormalize_allegro_jacobian_pixel(jacobian_pixel)
        return rearrange(jacobian_pixel, "b n s c -> b (n s) c")

    def _flow_rgb_chunk_to_actions(
        self,
        obs: PolicyObservation,
        source_rgb: Tensor,
        target_rgb: Tensor,
        flow: Tensor,
        source_view_widths: list[int],
        lam_override: float | None = None,
    ) -> tuple[Tensor, list[Dict[str, Any]]]:
        B, T, C, H, W = target_rgb.shape
        if self.dynamics_model is None:
            return torch.zeros(B, T, C, device=target_rgb.device), []
        actions = []
        vis_frames: list[Dict[str, Any]] = []
        for t in range(T):
            source_rgb_t, J, _, target_size, target_view_widths = (
                self._compute_jacobian_from_rgb_tensor(
                    source_rgb[:, t].to(self.device),
                    source_view_widths,
                    view_keys=obs.view_keys,
                )
            )
            J_phys = self._denormalize_allegro_jacobian_flat(J)
            flow_rs = self._resize_multiview_flow(
                flow[:, t],
                source_view_widths,
                target_view_widths,
            )
            flow_flat = rearrange(flow_rs, "b c h w -> b (h w) c")
            curr = alltracker.gridcloud2d(
                B,
                target_size[0],
                target_size[1],
                norm=False,
                device=flow.device,
            )
            trgt = curr + flow_flat

            y = rearrange(flow_rs, "b c h w -> b (h w c)")
            control_mask = self._control_motion_mask(
                obs,
                target_size[0],
                target_view_widths,
                device=y.device,
                dtype=y.dtype,
            )
            du, _ = self.controller.solve(
                J_phys,
                y,
                weights=control_mask.unsqueeze(0).expand(y.shape[0], -1),
                lam_override=lam_override,
            )
            actions.append(du)
            vis_frames.append(
                {
                    "rgb": source_rgb_t,
                    "target_rgb": target_rgb[:, t].detach().cpu(),
                    "flow": flow_rs,
                    "jacobian": J_phys,
                    "curr_track": curr,
                    "trgt_track": trgt,
                    "target_view_widths": target_view_widths,
                }
            )
        action_tensor = torch.stack(actions, dim=1)
        self._mark_actions_already_physical(action_tensor.shape[1])
        return action_tensor, vis_frames

    def _track_rgb_chunk_to_actions(
        self,
        obs: PolicyObservation,
        source_rgb: Tensor,
        tracks: Dict[str, Any],
        source_view_widths: list[int],
        target_rgb: Tensor | None = None,
        lam_override: float | None = None,
    ) -> tuple[Tensor, list[Dict[str, Any]]]:
        B, _, C, _, _ = source_rgb.shape
        track_steps = int(tracks["disp"].shape[1])
        if self.dynamics_model is None:
            return torch.zeros(B, track_steps, C, device=source_rgb.device), []

        num_steps = min(source_rgb.shape[1], track_steps)
        actions = []
        vis_frames: list[Dict[str, Any]] = []
        for t in range(num_steps):
            rgb_t, jacobian_flat, jacobian_pixel, target_size, target_view_widths = (
                self._compute_jacobian_from_rgb_tensor(
                    source_rgb[:, t].to(self.device),
                    source_view_widths,
                    view_keys=obs.view_keys,
                )
            )
            jacobian_flat_phys = self._denormalize_allegro_jacobian_flat(jacobian_flat)
            jacobian_pixel_phys = self._denormalize_allegro_jacobian_pixel(jacobian_pixel)

            track_frame = self._resize_track_frame(
                tracks,
                t,
                target_size,
                source_view_widths=source_view_widths,
                target_view_widths=target_view_widths,
            )
            if t == 0:
                xy_src = track_frame["xy_src"]
                disp_dbg = track_frame["disp"]
                valid_dbg = track_frame["valid"] > 0.5
                if valid_dbg.any():
                    valid_xy = xy_src[valid_dbg]
                    valid_disp = disp_dbg[valid_dbg]
                    xy_min = valid_xy.amin(dim=0)
                    xy_max = valid_xy.amax(dim=0)
                    disp_mean = valid_disp.norm(dim=-1).mean()
                else:
                    xy_min = torch.zeros(2, device=xy_src.device)
                    xy_max = torch.zeros(2, device=xy_src.device)
                    disp_mean = torch.zeros((), device=xy_src.device)
                print(
                    "[MotionPolicyAllegro] track resize "
                    f"source_rgb_shape={tuple(source_rgb.shape)} "
                    f"track_image_size={tracks.get('image_size')} "
                    f"source_view_widths={source_view_widths} "
                    f"target_size={target_size} "
                    f"target_view_widths={target_view_widths} "
                    f"view_keys={obs.view_keys} "
                    f"valid_frac={float(valid_dbg.float().mean().item()):.4f} "
                    f"valid_xy_min=({float(xy_min[0].item()):.1f},{float(xy_min[1].item()):.1f}) "
                    f"valid_xy_max=({float(xy_max[0].item()):.1f},{float(xy_max[1].item()):.1f}) "
                    f"disp_norm_mean={float(disp_mean.item()):.3f}",
                    flush=True,
                )
            idx_src = (
                track_frame["idx_src"]
                .long()
                .clamp(0, target_size[0] * target_size[1] - 1)
            )
            gather_idx = (
                idx_src.unsqueeze(-1)
                .unsqueeze(-1)
                .expand(-1, -1, 2, jacobian_pixel_phys.shape[-1])
            )
            jacobian_sparse = torch.gather(jacobian_pixel_phys, dim=1, index=gather_idx)
            jacobian_sparse = rearrange(jacobian_sparse, "b n s c -> b (n s) c")

            raw_disp = track_frame["disp"]
            disp = raw_disp * self.cfg.motion_plan_scale
            motion_plan = rearrange(disp, "b n s -> b (n s)")
            control_track_mask = self._control_track_mask(
                obs,
                track_frame["xy_src"][..., 0],
                target_view_widths,
            )
            outlier_mask = self._track_outlier_mask(raw_disp, target_view_widths)
            weights = track_frame["valid"] * control_track_mask * (1.0 - outlier_mask)
            track_stats = self._summarize_displacements(raw_disp, track_frame["valid"])
            control_track_stats = self._summarize_displacements(disp, weights)
            outlier_stats = {
                "count_total": float(outlier_mask.numel()),
                "count_dropped": float((outlier_mask > 0.5).sum().item()),
            }
            if self.cfg.controller.weight_flow_thresh > 0:
                weights = (
                    weights
                    * (
                        disp.norm(dim=-1) > self.cfg.controller.weight_flow_thresh
                    ).float()
                )
            weights = weights.repeat_interleave(2, dim=1)

            du, _ = self.controller.solve(
                jacobian_sparse,
                motion_plan,
                weights,
                lam_override=lam_override,
            )
            actions.append(du)
            vis_frames.append(
                {
                    "rgb": rgb_t,
                    "target_rgb": (
                        target_rgb[:, t].detach().cpu()
                        if target_rgb is not None
                        else None
                    ),
                    "jacobian": jacobian_flat_phys,
                    "curr_track": track_frame["xy_src"],
                    "trgt_track": (track_frame["xy_src"] + disp).clamp(
                        min=torch.tensor([0.0, 0.0], device=self.device),
                        max=torch.tensor(
                            [target_size[1] - 1.0, target_size[0] - 1.0],
                            device=self.device,
                        ),
                    ),
                    "curr_visible": track_frame["valid"] > 0.5,
                    "control_visible": weights.reshape(weights.shape[0], -1, 2)[..., 0]
                    > 0.5,
                    "target_view_widths": target_view_widths,
                    "track_stats": track_stats,
                    "control_track_stats": control_track_stats,
                    "outlier_stats": outlier_stats,
                }
            )
        action_tensor = torch.stack(actions, dim=1)
        self._mark_actions_already_physical(action_tensor.shape[1])
        return action_tensor, vis_frames

    def _flow_to_action(self, obs: PolicyObservation, flow: Tensor) -> PolicyOutput:
        source_view_widths = self._source_view_widths_for_obs(obs, int(flow.shape[-1]))
        policy_outputs = self._post_process_flow(flow, source_view_widths)
        policy_outputs["context_rgb"] = torch.stack(
            list(self._queues["observation.images"]), dim=1
        )
        motion_plan = policy_outputs["motion_plan"].reshape(flow.shape[0], -1)
        J = self._denormalize_allegro_jacobian_flat(self.compute_jacobian(obs))
        flow_mag = policy_outputs["motion_plan"].norm(dim=-1)
        flow_stats = {
            "flow_mag_min": flow_mag.min(dim=1).values.detach().cpu().numpy(),
            "flow_mag_max": flow_mag.max(dim=1).values.detach().cpu().numpy(),
            "flow_mag_mean": flow_mag.mean(dim=1).detach().cpu().numpy(),
        }
        weights = (
            (flow_mag > self.cfg.controller.weight_flow_thresh).float()
            if self.cfg.controller.weight_flow_thresh > 0
            else None
        )
        control_mask = self._control_motion_mask(
            obs,
            self.image_size_dynamics_model[0],
            self._dynamics_view_widths(len(source_view_widths)),
            device=motion_plan.device,
            dtype=motion_plan.dtype,
        )
        control_mask = control_mask.unsqueeze(0).expand(motion_plan.shape[0], -1)
        weights = control_mask if weights is None else weights
        if weights is not None and weights.shape[1] != J.shape[1]:
            factor = J.shape[1] // weights.shape[1]
            if factor > 0 and weights.shape[1] * factor == J.shape[1]:
                weights = weights.repeat_interleave(factor, dim=1)
            else:
                raise ValueError(
                    f"weights/J mismatch: weights={weights.shape} J={J.shape}"
                )
        if weights is not None:
            weights = weights * control_mask
        runtime_controller = self._runtime_controller_params()
        du, ctrl_metrics = self.controller.solve(
            J,
            motion_plan,
            weights,
            lam_override=float(runtime_controller["lam_runtime"]),
        )
        du_pre_clip = self._apply_runtime_action_scaling(du, runtime_controller)
        du = du_pre_clip.clamp(
            -self.cfg.controller.clip_du, self.cfg.controller.clip_du
        )
        self._mark_actions_already_physical(1)
        du_final = self.get_final_action(du)
        action_debug = self._build_action_debug_info(du_pre_clip, du, du_final)
        policy_debug = self._collect_policy_debug_info(action_debug)
        self._build_action_feedback_payload(obs, action_debug)
        policy_vis = None
        if obs.rgb_vis is not None:
            policy_outputs["jacobian"] = J
            policy_vis = self._make_policy_vis(
                obs,
                policy_outputs,
                action_debug=policy_debug.get("action_debug"),
                gripper_debug=policy_debug.get("gripper_debug"),
            )
        info = {
            "controller_metrics": ctrl_metrics,
            "motion_plan": motion_plan,
            "flow_stats": flow_stats,
            "control_view_keys": self.cfg.control_view_keys,
            "policy_vis": policy_vis,
        }
        info.update(policy_debug)
        adaptive_debug = self._adaptive_debug_info()
        if adaptive_debug is not None:
            info["adaptive_controller"] = adaptive_debug
        return PolicyOutput(
            action=du_final.cpu().numpy(),
            info=info,
        )

    def _allegro_jacobian_grid_cols(self, cmd_dim: int) -> int:
        if cmd_dim % 4 == 0:
            return 4
        return max(2, int(np.ceil(np.sqrt(cmd_dim))))

    def _make_compact_allegro_jacobian_panel(
        self,
        jacobian: Tensor,
        *,
        target_h: int,
    ) -> np.ndarray:
        jacobian = self._threshold_jacobian_for_vis(jacobian)
        vis_jacobian = rearrange(
            visualize_jacobian(jacobian=jacobian, robot_name=self.robot_name),
            "c h w -> h w c",
        )

        cmd_dim = int(jacobian.shape[-4])
        if cmd_dim <= 0:
            return vis_jacobian

        source_rows = 2
        source_cols = (cmd_dim + source_rows - 1) // source_rows
        tile_h = max(1, vis_jacobian.shape[0] // source_rows)
        tile_w = max(1, vis_jacobian.shape[1] // source_cols)

        tiles = []
        for tile_idx in range(cmd_dim):
            row_idx = tile_idx // source_cols
            col_idx = tile_idx % source_cols
            y0 = row_idx * tile_h
            y1 = y0 + tile_h
            x0 = col_idx * tile_w
            x1 = x0 + tile_w
            tiles.append(vis_jacobian[y0:y1, x0:x1])

        compact_cols = self._allegro_jacobian_grid_cols(cmd_dim)
        compact_rows = int(np.ceil(cmd_dim / compact_cols))
        pad_tile = np.zeros_like(tiles[0])
        while len(tiles) < compact_rows * compact_cols:
            tiles.append(pad_tile.copy())

        row_strips = []
        for row_idx in range(compact_rows):
            start = row_idx * compact_cols
            row_tiles = tiles[start : start + compact_cols]
            row_strips.append(np.concatenate(row_tiles, axis=1))
        compact_panel = np.concatenate(row_strips, axis=0)

        if compact_panel.shape[0] != target_h:
            target_w = max(
                1,
                int(round(compact_panel.shape[1] * target_h / compact_panel.shape[0])),
            )
            compact_panel = media.resize_image(compact_panel, shape=(target_h, target_w))
        return compact_panel

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
        jac_h, jac_w = self._concat_dynamics_size(self._obs_view_count(obs))
        jacobian_raw = policy_outputs.get("jacobian")
        jacobian = (
            rearrange(
                jacobian_raw,
                "b (h w s) c -> b c s h w",
                h=jac_h,
                w=jac_w,
            )
            if jacobian_raw is not None
            else None
        )

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

            if jacobian is not None:
                vis_jacobian = self._make_compact_allegro_jacobian_panel(
                    jacobian[batch_idx : batch_idx + 1],
                    target_h=vis_policy_output.shape[0],
                )
                vis = np.concatenate([vis_policy_output, vis_jacobian], axis=1)
            else:
                vis = vis_policy_output
            context_strip = self._make_context_strip(
                obs,
                policy_outputs.get("context_rgb"),
                target_width=vis.shape[1],
                max_height=max(40, vis_h // 4),
            )
            panel_titles_top = ["current + tracks"]
            if jacobian is not None:
                panel_titles_top.append("jacobian")
            header = self._make_policy_vis_header(
                obs,
                panel_titles=panel_titles_top,
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
                panel for panel in (context_strip, header, vis, footer) if panel is not None
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
        vis_h, vis_w = self.image_size_motion_planner
        jac_h, jac_w = self._concat_dynamics_size(self._obs_view_count(obs))
        vis_batch = []

        for frame_idx, frame in enumerate(frames):
            rgb = frame["rgb"]
            jacobian_raw = frame.get("jacobian")
            jacobian = (
                rearrange(
                    jacobian_raw,
                    "b (h w s) c -> b c s h w",
                    h=jac_h,
                    w=jac_w,
                )
                if jacobian_raw is not None
                else None
            )

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
                draw_curr_track = curr_track
                draw_trgt_track = trgt_track
                src_h, src_w = rgb_np.shape[:2]
                if src_h != vis_source.shape[0] or src_w != vis_source.shape[1]:
                    scale = torch.as_tensor(
                        [
                            vis_source.shape[1] / max(float(src_w), 1.0),
                            vis_source.shape[0] / max(float(src_h), 1.0),
                        ],
                        device=curr_track.device,
                        dtype=curr_track.dtype,
                    )
                    draw_curr_track = curr_track * scale
                    draw_trgt_track = trgt_track * scale
                vis_source = visualize_tracks.draw_curr_trgt_tracks_dense(
                    obs=vis_source,
                    curr_track=draw_curr_track,
                    trgt_track=draw_trgt_track,
                    curr_visible=control_visible,
                    point_radius=1,
                    point_color=(0, 0, 0),
                    arrow_color=(0, 255, 0),
                    arrow_thickness=1,
                    sparsity=self.cfg.vis_track_sparsity_joint,
                    motion_thresh=1,
                    arrow_scale=1.5,
                )

                if jacobian is not None:
                    vis_jacobian = self._make_compact_allegro_jacobian_panel(
                        jacobian[batch_idx : batch_idx + 1],
                        target_h=current_obs.shape[0],
                    )
                    vis = np.concatenate(
                        [current_obs, vis_source, vis_jacobian], axis=1
                    )
                else:
                    vis = np.concatenate([current_obs, vis_source], axis=1)
                context_strip = self._make_context_strip(
                    obs,
                    frame.get("context_rgb"),
                    target_width=vis.shape[1],
                    max_height=max(40, vis_h // 4),
                )
                source_frame_role = str(frame.get("source_frame_role", "source"))
                target_frame_role = str(frame.get("target_frame_role", "target"))
                source_panel_title = f"{source_frame_role} + tracks -> {target_frame_role}"
                panel_titles = ["current", source_panel_title]
                if jacobian is not None:
                    panel_titles.append("jacobian")
                header = self._make_policy_vis_header(
                    obs,
                    panel_titles=panel_titles,
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
                    panel for panel in (context_strip, header, vis, footer) if panel is not None
                ]
                vis = np.concatenate(stacked_panels, axis=0)
                vis_batch.append(vis.astype(np.float32) / 255.0)

        if not vis_batch:
            return np.zeros((0, vis_h, vis_w * 3, 3), dtype=np.float32)
        return np.stack(vis_batch, axis=0)
