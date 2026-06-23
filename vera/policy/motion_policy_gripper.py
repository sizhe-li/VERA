"""
Motion policy variant for 7 DOF robots with a gripper: adds binary gripper gating
(last action dimension → open/close with hysteresis) on top of the base MotionPolicy.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, cast
import numpy as np
import torch
from torch import Tensor

from .base_policy import PolicyObservation, PolicyOutput
from .cartesian_policy_support import (
    AdaptiveChannelState,
    draw_cartesian_action_on_frame,
    gripper_width_from_qpos,
    quat_delta_to_rotvec,
)
from .motion_policy import MotionPolicy, MotionPolicyCfg, UNSET


@dataclass
class MotionPolicyGripperCfg(MotionPolicyCfg):
    """Config for motion policy with configurable gripper command behavior.

    The last action dimension is interpreted as gripper intent. It can be
    converted into either:

    - a binary open / close command with hysteresis (`gripper_command_mode="gated"`)
    - a scaled continuous command (`gripper_command_mode="continuous"`)

    Pass name='motion_policy_gripper' when constructing the config.
    """

    gripper_command_mode: Literal["gated", "continuous", "integrated_velocity"] = field(
        default="gated", kw_only=True
    )
    gripper_thresh: float = field(default=0.1, kw_only=True)
    gripper_close_thresh: float | None = field(default=0.05, kw_only=True)
    gripper_open_thresh: float | None = field(default=None, kw_only=True)
    # In gated mode, raw gripper intent with smaller magnitude than this value
    # is ignored and the previous desired open / close state is preserved.
    gripper_deadband_thresh: float = field(default=0.0, kw_only=True)
    gripper_signal_smoothing: float = field(default=0.5, kw_only=True)
    gripper_min_hold_steps: int = field(default=3, kw_only=True)
    # Continuous-mode scale applied to the last action dimension before clipping
    # to [-1, 1]. Ignored in gated mode.
    gripper_continuous_scale: float = field(default=1.0, kw_only=True)
    # integrated_velocity mode: integrates du[-1] into a discrete position
    # state g in [0, 1], then thresholds at 0.5 (DreamZero-style binarize but
    # applied to integrated velocity). One gain knob; no hysteresis / no
    # deadband / no hold steps. See _apply_integrated_velocity_gripper.
    gripper_integrate_alpha: float = field(default=5.0, kw_only=True)
    # Optional zero-knob Schmitt: once g crosses 0.5 in either direction,
    # snap to the edge (0.0 or 1.0) so subsequent small velocity noise can't
    # flicker the binary output. Same effect as a one-sided commit.
    gripper_integrate_snap_to_edge: bool = field(default=True, kw_only=True)
    # ── self-normalized (z-score) OPEN gate ──
    # 'absolute': open when filtered < -open_thresh (legacy; open_thresh=0.1 is
    #   ~0.01 sigma of the measured mimicgen intent distribution -> mid-carry drops).
    # 'zscore': open when (filtered - mu)/sigma < -z_k_open, with (mu, sigma)
    #   tracked online from the episode's own intent stream, initialized from a
    #   measured per-embodiment prior (Tier-1) worth z_prior_n0 pseudo-samples.
    #   Close stays absolute (validated). Units are dimensionless -> portable
    #   across embodiments (mimicgen/DROID/...) without retuning.
    gripper_open_gate: Literal["absolute", "zscore"] = field(
        default="absolute", kw_only=True
    )
    gripper_z_k_open: float = field(default=2.0, kw_only=True)
    gripper_z_prior_mu: float = field(default=0.0, kw_only=True)
    gripper_z_prior_sigma: float = field(default=1.0, kw_only=True)
    gripper_z_prior_n0: float = field(default=100.0, kw_only=True)
    # Treat per-env gripper widths at or below this value as "closed" when
    # reconciling policy state with the current simulator observation.
    gripper_state_width_threshold: float = field(default=1e-3, kw_only=True)
    # Re-align desired gripper state with the observed simulator state on the
    # first rollout step after a reset.
    gripper_realign_on_step0: bool = field(default=True, kw_only=True)
    gripper_jacobian_exclude_view_keys: list[str] = field(
        default_factory=lambda: ["agentview_image"], kw_only=True
    )
    # Action dimensions to suppress entirely for excluded views. Negative
    # indices are resolved relative to the action dimension count.
    gripper_jacobian_exclude_action_dims: tuple[int, ...] = field(
        default=(-1, -2, -3, -4), kw_only=True
    )
    # Action dimension to emphasize for gripper-relevant views.
    gripper_jacobian_focus_action_dim: int = field(default=-1, kw_only=True)
    # Optional explicit bottom crop in pixels for the focus action dimension.
    gripper_jacobian_focus_bottom_margin_px: int | None = field(
        default=None, kw_only=True
    )
    # Resolution-independent fallback when the explicit bottom crop is unset.
    gripper_jacobian_focus_bottom_margin_ratio: float = field(
        default=23.0 / 256.0, kw_only=True
    )
    # Gain applied to the focus action dimension on gripper-relevant views.
    gripper_jacobian_focus_gain: float = field(default=-10.0, kw_only=True)
    # Sparser track visualization (higher = fewer arrows); overrides base defaults.
    vis_track_sparsity: int = 200
    vis_track_sparsity_joint: int = 100
    # Command channels to zero in the Jacobian (e.g. [3, 4] = roll, pitch); controller won't use them.
    jacobian_zero_channels: tuple[int, ...] = (3, 4)

    # Action scaling/zeroing applied in get_final_action (after base controller scale/clip).
    action_scale_translation: float = field(default=0.1, kw_only=True)  # dims [0,1,2]
    action_scale_yaw: float = field(default=1.0 / 30.0, kw_only=True)  # dim [5]
    action_zero_channels: tuple[int, ...] = field(
        default=(3, 4), kw_only=True
    )  # e.g. roll, pitch
    # If set, override the gripper command with this constant; else use gating.
    # MUST default to None: the old -0.9 default silently pinned the gripper OPEN
    # (mimicgen convention: +1 close / -1 open) at every step, making grasps
    # impossible regardless of the IDM output (root cause of the 0% stack SR).
    gripper_fixed_value: float | None = field(default=None, kw_only=True)


@dataclass
class GripperJacobianContext:
    view_keys: list[str] | None
    view_count: int
    exclude_view_indices: list[int]
    allowed_view_indices: list[int]
    active_exclude_view_keys: list[str]
    active_allowed_view_keys: list[str] | None


@dataclass
class GripperRuntimeState:
    desired_closed: torch.Tensor
    observed_closed: torch.Tensor
    filtered_intent: torch.Tensor
    hold_remaining: torch.Tensor
    # Online intent statistics for the z-score open gate (per batch row):
    # running mean / M2 (Welford) seeded with the Tier-1 prior worth z_n
    # pseudo-samples. Updated once per gate call with the raw intent.
    z_n: torch.Tensor | None = None
    z_mean: torch.Tensor | None = None
    z_m2: torch.Tensor | None = None
    # Continuous position-equivalent state in [0, 1] for the integrated_velocity
    # mode. 0 = open, 1 = closed. Initialized from observed gripper state at
    # reset, then updated via du[-1] integration each step.
    integrated_position: torch.Tensor | None = None

    @property
    def batch_size(self) -> int:
        return int(self.desired_closed.shape[0])


class MotionPolicyGripper(MotionPolicy):
    """MotionPolicy variant with configurable gripper command behavior."""

    cfg: MotionPolicyGripperCfg

    def __init__(self, cfg: MotionPolicyGripperCfg, device: torch.device):
        super().__init__(cfg, device)
        self._gripper_runtime_state: GripperRuntimeState | None = None
        self._gripper_jacobian_context: GripperJacobianContext | None = None
        self._last_gripper_jacobian_mask_info: dict | None = None
        self._last_gripper_debug_info: dict | None = None

    def reset(self):
        super().reset()
        self._gripper_runtime_state = None
        self._clear_gripper_jacobian_context()
        self._last_gripper_jacobian_mask_info = None
        self._last_gripper_debug_info = None

    def get_wire_metadata(self) -> dict[str, Any]:
        """Handshake metadata for the deploy protocol (mirrors droid_native for the gripper policy).
        The gripper policy applies its own scaling/gating in get_final_action, so the emitted action
        is already env-ready -> actions_already_metric=True (client must not re-scale)."""
        meta = self.get_dynamics_normalization_metadata()
        abs_scale = [float(x) for x in meta.get("action_abs_scale", []) or []]
        return {
            "action_mode": str(meta.get("action_mode", "eef_delta")),
            "action_abs_scale": abs_scale,
            "dim_u": len(abs_scale) or int(getattr(self.cfg, "action_dim", 7) or 7),
            "gripper_dim_index": -1,
            "actions_already_metric": True,
            "context_frames": int(self.context_frames),
            "action_chunk_horizon": int(self.cfg.action_chunk_horizon),
            "n_action_steps": int(self.cfg.n_action_steps),
            "current_tracker_backend": str(self.cfg.motion_planner.tracker_backend),
        }

    def configure_runtime(
        self,
        *,
        gripper_fixed_value=UNSET,
        gripper_command_mode: Literal["gated", "continuous", "integrated_velocity"] | None = None,
        gripper_thresh: float | None = None,
        gripper_close_thresh: float | None = None,
        gripper_open_thresh: float | None = None,
        gripper_deadband_thresh: float | None = None,
        gripper_signal_smoothing: float | None = None,
        gripper_min_hold_steps: int | None = None,
        gripper_continuous_scale: float | None = None,
        gripper_integrate_alpha: float | None = None,
        gripper_integrate_snap_to_edge: bool | None = None,
        gripper_state_width_threshold: float | None = None,
        gripper_realign_on_step0: bool | None = None,
        gripper_open_gate: Literal["absolute", "zscore"] | None = None,
        gripper_z_k_open: float | None = None,
        gripper_z_prior_mu: float | None = None,
        gripper_z_prior_sigma: float | None = None,
        gripper_z_prior_n0: float | None = None,
        jacobian_zero_channels: tuple[int, ...] | list[int] | None = None,
        action_scale_translation: float | None = None,
        action_scale_yaw: float | None = None,
        action_zero_channels: tuple[int, ...] | list[int] | None = None,
        **base_kwargs: Any,
    ) -> dict:
        # Forward every non-gripper kwarg to the base policy verbatim. Previously the
        # base params were re-listed here, so any base kwarg NOT relisted (e.g.
        # lang_guidance / hist_guidance / sample_steps / control_view_keys /
        # debug_dump_task_name / debug_dump_new_run) raised TypeError — and because the
        # configure endpoint is atomic, the WHOLE request (incl. gripper params) was
        # dropped (M4). **base_kwargs makes the override transparent to the base API.
        runtime = super().configure_runtime(**base_kwargs)
        if gripper_fixed_value is not UNSET:
            self.cfg.gripper_fixed_value = (
                None
                if gripper_fixed_value is None
                else float(cast(float, gripper_fixed_value))
            )
        if gripper_command_mode is not None:
            self.cfg.gripper_command_mode = cast(
                Literal["gated", "continuous", "integrated_velocity"],
                gripper_command_mode,
            )
        if gripper_thresh is not None:
            self.cfg.gripper_thresh = float(gripper_thresh)
        if gripper_close_thresh is not None:
            self.cfg.gripper_close_thresh = float(gripper_close_thresh)
        if gripper_open_thresh is not None:
            self.cfg.gripper_open_thresh = float(gripper_open_thresh)
        if gripper_deadband_thresh is not None:
            self.cfg.gripper_deadband_thresh = float(gripper_deadband_thresh)
        if gripper_signal_smoothing is not None:
            self.cfg.gripper_signal_smoothing = float(gripper_signal_smoothing)
        if gripper_min_hold_steps is not None:
            self.cfg.gripper_min_hold_steps = int(gripper_min_hold_steps)
        if gripper_continuous_scale is not None:
            self.cfg.gripper_continuous_scale = float(gripper_continuous_scale)
        if gripper_integrate_alpha is not None:
            self.cfg.gripper_integrate_alpha = float(gripper_integrate_alpha)
        if gripper_integrate_snap_to_edge is not None:
            self.cfg.gripper_integrate_snap_to_edge = bool(gripper_integrate_snap_to_edge)
        if gripper_state_width_threshold is not None:
            self.cfg.gripper_state_width_threshold = float(
                gripper_state_width_threshold
            )
        if gripper_realign_on_step0 is not None:
            self.cfg.gripper_realign_on_step0 = bool(gripper_realign_on_step0)
        if gripper_open_gate is not None:
            if gripper_open_gate not in ("absolute", "zscore"):
                raise ValueError(
                    f"gripper_open_gate must be 'absolute' or 'zscore', got {gripper_open_gate!r}"
                )
            self.cfg.gripper_open_gate = cast(
                Literal["absolute", "zscore"], gripper_open_gate
            )
        if gripper_z_k_open is not None:
            self.cfg.gripper_z_k_open = float(gripper_z_k_open)
        z_prior_changed = False
        if gripper_z_prior_mu is not None:
            self.cfg.gripper_z_prior_mu = float(gripper_z_prior_mu)
            z_prior_changed = True
        if gripper_z_prior_sigma is not None:
            self.cfg.gripper_z_prior_sigma = float(gripper_z_prior_sigma)
            z_prior_changed = True
        if gripper_z_prior_n0 is not None:
            self.cfg.gripper_z_prior_n0 = float(gripper_z_prior_n0)
            z_prior_changed = True
        if z_prior_changed:
            # Re-seed the live Welford stats from the new prior. Without this the per-step
            # update reads _gripper_runtime_state (not cfg), so a mid-session prior change
            # via configure would silently no-op until the next reset() (M3).
            self._reseed_gripper_z_stats()
        if jacobian_zero_channels is not None:
            self.cfg.jacobian_zero_channels = tuple(
                int(ch) for ch in jacobian_zero_channels
            )
        if action_scale_translation is not None:
            self.cfg.action_scale_translation = float(action_scale_translation)
        if action_scale_yaw is not None:
            self.cfg.action_scale_yaw = float(action_scale_yaw)
        if action_zero_channels is not None:
            self.cfg.action_zero_channels = tuple(
                int(ch) for ch in action_zero_channels
            )
        runtime.update(
            {
                "gripper_fixed_value": self.cfg.gripper_fixed_value,
                "gripper_command_mode": str(self.cfg.gripper_command_mode),
                "gripper_thresh": float(self.cfg.gripper_thresh),
                "gripper_close_thresh": (
                    None
                    if self.cfg.gripper_close_thresh is None
                    else float(self.cfg.gripper_close_thresh)
                ),
                "gripper_open_thresh": (
                    None
                    if self.cfg.gripper_open_thresh is None
                    else float(self.cfg.gripper_open_thresh)
                ),
                "gripper_deadband_thresh": float(self.cfg.gripper_deadband_thresh),
                "gripper_signal_smoothing": float(self.cfg.gripper_signal_smoothing),
                "gripper_min_hold_steps": int(self.cfg.gripper_min_hold_steps),
                "gripper_continuous_scale": float(self.cfg.gripper_continuous_scale),
                "gripper_state_width_threshold": float(
                    self.cfg.gripper_state_width_threshold
                ),
                "gripper_realign_on_step0": bool(self.cfg.gripper_realign_on_step0),
                "gripper_open_gate": str(self.cfg.gripper_open_gate),
                "gripper_z_k_open": float(self.cfg.gripper_z_k_open),
                "gripper_z_prior_mu": float(self.cfg.gripper_z_prior_mu),
                "gripper_z_prior_sigma": float(self.cfg.gripper_z_prior_sigma),
                "gripper_z_prior_n0": float(self.cfg.gripper_z_prior_n0),
                "gripper_jacobian_exclude_action_dims": tuple(
                    int(ch) for ch in self.cfg.gripper_jacobian_exclude_action_dims
                ),
                "gripper_jacobian_focus_action_dim": int(
                    self.cfg.gripper_jacobian_focus_action_dim
                ),
                "gripper_jacobian_focus_bottom_margin_px": (
                    None
                    if self.cfg.gripper_jacobian_focus_bottom_margin_px is None
                    else int(self.cfg.gripper_jacobian_focus_bottom_margin_px)
                ),
                "gripper_jacobian_focus_bottom_margin_ratio": float(
                    self.cfg.gripper_jacobian_focus_bottom_margin_ratio
                ),
                "gripper_jacobian_focus_gain": float(
                    self.cfg.gripper_jacobian_focus_gain
                ),
                "jacobian_zero_channels": tuple(
                    int(ch) for ch in self.cfg.jacobian_zero_channels
                ),
                "action_scale_translation": float(self.cfg.action_scale_translation),
                "action_scale_yaw": float(self.cfg.action_scale_yaw),
                "action_zero_channels": tuple(
                    int(ch) for ch in self.cfg.action_zero_channels
                ),
            }
        )
        return runtime

    def _runtime_action_channel_scales(self, action_dim: int) -> np.ndarray:
        scales = np.ones(int(action_dim), dtype=np.float32)
        if action_dim <= 0:
            return scales
        channel_state = self._adaptive_channel_state
        scales[: min(3, action_dim)] = float(channel_state.translation)
        if action_dim > 3:
            scales[3 : min(6, action_dim)] = float(channel_state.rotation)
        if action_dim > 6:
            scales[6] = float(channel_state.gripper)
        return scales

    @staticmethod
    def _action_group_summary(action: np.ndarray | None) -> dict[str, Any] | None:
        if action is None:
            return None
        action_arr = np.asarray(action, dtype=np.float32)
        if action_arr.ndim == 1:
            action_arr = action_arr[None, :]
        translation = action_arr[..., :3]
        rotation = (
            action_arr[..., 3:6]
            if action_arr.shape[-1] > 3
            else np.zeros_like(translation)
        )
        gripper = (
            action_arr[..., 6:7]
            if action_arr.shape[-1] > 6
            else np.zeros((*action_arr.shape[:-1], 1), dtype=np.float32)
        )
        return {
            "translation": translation.copy(),
            "rotation": rotation.copy(),
            "gripper": gripper.copy(),
            "translation_norm": np.linalg.norm(translation, axis=-1),
            "rotation_norm": np.linalg.norm(rotation, axis=-1),
            "gripper_abs": np.abs(gripper[..., 0]),
        }

    @staticmethod
    def _extract_feedback_state(obs: PolicyObservation) -> dict[str, np.ndarray] | None:
        if obs.eef_pos is None and obs.eef_quat is None and obs.gripper_qpos is None:
            return None
        state: dict[str, np.ndarray] = {}
        if obs.eef_pos is not None:
            eef_pos = np.asarray(obs.eef_pos, dtype=np.float32)
            if eef_pos.ndim == 1:
                eef_pos = eef_pos[None, :]
            state["eef_pos"] = eef_pos.copy()
        if obs.eef_quat is not None:
            eef_quat = np.asarray(obs.eef_quat, dtype=np.float32)
            if eef_quat.ndim == 1:
                eef_quat = eef_quat[None, :]
            state["eef_quat"] = eef_quat.copy()
        if obs.gripper_qpos is not None:
            gripper_qpos = np.asarray(obs.gripper_qpos, dtype=np.float32)
            if gripper_qpos.ndim == 1:
                gripper_qpos = gripper_qpos[None, :]
            state["gripper_qpos"] = gripper_qpos.copy()
        return state

    def _feedback_payload_extras(self, obs: PolicyObservation) -> dict[str, Any]:
        del obs
        gripper_debug = getattr(self, "_last_gripper_debug_info", None)
        return {
            "gripper_mode": (
                None
                if not isinstance(gripper_debug, dict)
                else cast(dict[str, Any], gripper_debug).get("gripper_mode")
            )
        }

    def _estimate_realized_action(
        self,
        pre_state: dict[str, np.ndarray] | None,
        obs: PolicyObservation,
    ) -> np.ndarray | None:
        if pre_state is None:
            return None
        post_state = self._extract_feedback_state(obs)
        if post_state is None:
            return None
        batch_size = 1
        for key in ("eef_pos", "eef_quat", "gripper_qpos"):
            value = pre_state.get(key)
            if value is not None:
                batch_size = int(value.shape[0])
                break
        realized = np.zeros((batch_size, 7), dtype=np.float32)
        dt = max(float(obs.dt), float(self.cfg.adaptive_controller.grouped_action_dt_eps))
        if "eef_pos" in pre_state and "eef_pos" in post_state:
            delta_pos = post_state["eef_pos"] - pre_state["eef_pos"]
            realized[:, :3] = delta_pos / dt if obs.action_mode == "velocity" else delta_pos
        if "eef_quat" in pre_state and "eef_quat" in post_state:
            rotvec = quat_delta_to_rotvec(pre_state["eef_quat"], post_state["eef_quat"])
            realized[:, 3:6] = rotvec / dt if obs.action_mode == "velocity" else rotvec
        if "gripper_qpos" in pre_state and "gripper_qpos" in post_state:
            width_pre = gripper_width_from_qpos(pre_state["gripper_qpos"])
            width_post = gripper_width_from_qpos(post_state["gripper_qpos"])
            delta_width = (width_post - width_pre).reshape(batch_size, 1)
            realized[:, 6:7] = (
                delta_width / dt if obs.action_mode == "velocity" else delta_width
            )
        return realized

    def _update_grouped_adaptive_state(
        self,
        dream_action: np.ndarray | None,
        realized_action: np.ndarray | None,
        *,
        gripper_mode: str | None = None,
    ) -> dict[str, Any] | None:
        if dream_action is None or realized_action is None:
            return None
        dream = np.asarray(dream_action, dtype=np.float32)
        realized = np.asarray(realized_action, dtype=np.float32)
        if dream.ndim == 1:
            dream = dream[None, :]
        if realized.ndim == 1:
            realized = realized[None, :]
        cfg = self.cfg.adaptive_controller
        stationary_eps = max(float(cfg.grouped_action_eps), float(cfg.eps))
        alpha = float(np.clip(cfg.grouped_action_ema_alpha, 0.0, 1.0))
        step_up = max(float(cfg.grouped_action_step_up), 0.0)
        step_down = max(float(cfg.grouped_action_step_down), 0.0)
        group_specs = (
            (
                "translation",
                slice(0, 3),
                cfg.translation_gain_min_scale,
                cfg.translation_gain_max_scale,
            ),
            (
                "rotation",
                slice(3, 6),
                cfg.rotation_gain_min_scale,
                cfg.rotation_gain_max_scale,
            ),
            (
                "gripper",
                slice(6, 7),
                cfg.gripper_gain_min_scale,
                cfg.gripper_gain_max_scale,
            ),
        )
        if not bool(cfg.enable_gripper_channel_adaptation):
            group_specs = group_specs[:2]
        if gripper_mode == "gated":
            group_specs = tuple(
                spec for spec in group_specs if spec[0] != "gripper"
            )

        group_feedback: dict[str, Any] = {}
        updated_state = self._adaptive_channel_state.as_dict()
        for name, action_slice, gain_min, gain_max in group_specs:
            dream_group = dream[..., action_slice]
            realized_group = realized[..., action_slice]
            dream_mag = np.linalg.norm(dream_group, axis=-1)
            realized_mag = np.linalg.norm(realized_group, axis=-1)
            active_mask = dream_mag > stationary_eps
            if not np.any(active_mask):
                group_feedback[name] = {
                    "active": False,
                    "dream_mag": float(np.mean(dream_mag)),
                    "realized_mag": float(np.mean(realized_mag)),
                    "response_ratio": None,
                    "gain_scale": float(updated_state[name]),
                }
                continue
            response_ratio = realized_mag[active_mask] / (
                dream_mag[active_mask] + stationary_eps
            )
            mean_ratio = float(np.mean(response_ratio))
            current_gain = float(updated_state[name])
            if mean_ratio < 1.0:
                ideal_gain = current_gain / max(mean_ratio, stationary_eps)
                if step_up > 0.0:
                    max_up_gain = current_gain * (1.0 + step_up)
                    target_gain = min(ideal_gain, max_up_gain)
                else:
                    target_gain = ideal_gain
            else:
                target_gain = current_gain / max(mean_ratio, stationary_eps)
                if step_down > 0.0 and mean_ratio > 1.0:
                    overshoot = mean_ratio - 1.0
                    overshoot_extra = 1.0 + step_down * overshoot
                    target_gain /= overshoot_extra
            target_gain = float(np.clip(target_gain, gain_min, gain_max))
            new_gain = (1.0 - alpha) * current_gain + alpha * target_gain
            updated_state[name] = float(np.clip(new_gain, gain_min, gain_max))
            group_feedback[name] = {
                "active": True,
                "dream_mag": float(np.mean(dream_mag[active_mask])),
                "realized_mag": float(np.mean(realized_mag[active_mask])),
                "response_ratio": mean_ratio,
                "ideal_gain": float(current_gain / max(mean_ratio, stationary_eps)),
                "target_gain": float(target_gain),
                "gain_scale": float(updated_state[name]),
            }
        self._adaptive_channel_state = AdaptiveChannelState(
            translation=float(updated_state["translation"]),
            rotation=float(updated_state["rotation"]),
            gripper=float(updated_state["gripper"]),
        )
        return group_feedback

    def _make_action_hud_lines(
        self,
        *,
        batch_idx: int,
        frame_idx: int | None = None,
        action_debug: dict[str, Any] | None = None,
        gripper_debug: dict[str, Any] | None = None,
    ) -> list[tuple[str, tuple[int, int, int]]]:
        action_final = self._select_debug_batch_frame_value(
            None if action_debug is None else action_debug.get("action_final"),
            batch_idx,
            frame_idx,
        )
        lines: list[tuple[str, tuple[int, int, int]]] = []
        v_text = self._format_vector(action_final, 3)
        if v_text is not None:
            lines.append((f"v: {v_text}", (235, 235, 235)))
        w_text = self._format_vector(
            None if action_final is None else np.asarray(action_final)[3:6], 3
        )
        if w_text is not None:
            lines.append((f"w: {w_text}", (255, 210, 120)))

        action_pre_clip = self._select_debug_batch_frame_value(
            None if action_debug is None else action_debug.get("action_pre_clip"),
            batch_idx,
            frame_idx,
        )
        action_pre_gate = self._select_debug_batch_frame_value(
            None if action_debug is None else action_debug.get("action_pre_gate"),
            batch_idx,
            frame_idx,
        )
        final_g = (
            None if action_final is None else np.asarray(action_final).reshape(-1)[-1]
        )

        if gripper_debug is not None:
            mode = gripper_debug.get("gripper_mode")
            pre_gate = self._select_debug_batch_value(
                (
                    gripper_debug.get("gripper_pre_gate")
                    if "gripper_pre_gate" in gripper_debug
                    else gripper_debug.get("gripper_raw")
                ),
                batch_idx,
            )
            pre_clip = self._select_debug_batch_value(
                gripper_debug.get("gripper_pre_clip"), batch_idx
            )
            filtered = self._select_debug_batch_value(
                gripper_debug.get("gripper_raw_filtered"), batch_idx
            )
            close_thr = gripper_debug.get("gripper_close_thresh")
            open_thr = gripper_debug.get("gripper_open_thresh")
            hold = self._select_debug_batch_value(
                gripper_debug.get("gripper_hold_remaining"), batch_idx
            )
            close_mask = self._select_debug_batch_value(
                gripper_debug.get("gripper_close_mask"), batch_idx
            )
            open_mask = self._select_debug_batch_value(
                gripper_debug.get("gripper_open_mask"), batch_idx
            )
            open_intent = self._select_debug_batch_value(
                gripper_debug.get("gripper_open_intent"), batch_idx
            )
            open_blocked = self._select_debug_batch_value(
                gripper_debug.get("gripper_open_blocked"), batch_idx
            )

            # Chunk-safe scalar: these may be per-action arrays (length H) over a chunk, not scalars.
            def _scalar(v):
                return float(np.asarray(v).reshape(-1)[-1])

            g_parts = []
            prefix = "g"
            if mode is not None:
                prefix = f"g[{mode}]"
            if final_g is not None:
                g_parts.append(f"final={_scalar(final_g):+.3f}")
            if pre_gate is not None:
                g_parts.append(f"pre_gate={_scalar(pre_gate):+.3f}")
            if pre_clip is not None:
                g_parts.append(f"pre_clip={_scalar(pre_clip):+.3f}")
            elif action_pre_clip is not None:
                g_parts.append(f"pre_clip={_scalar(action_pre_clip):+.3f}")
            if filtered is not None:
                g_parts.append(f"filt={_scalar(filtered):+.3f}")
            lines.append((f"{prefix}: " + " ".join(g_parts), (200, 255, 200)))

            thr_parts = []
            if close_thr is not None:
                thr_parts.append(f"close_thr={float(close_thr):+.3f}")
            if open_thr is not None:
                thr_parts.append(f"open_thr={float(open_thr):+.3f}")
            if hold is not None:
                thr_parts.append(f"hold={int(_scalar(hold))}")
            close_flag = self._format_bool_flag(close_mask)
            if close_flag is not None:
                thr_parts.append(f"close={close_flag}")
            open_flag = self._format_bool_flag(open_mask)
            if open_flag is not None:
                thr_parts.append(f"open={open_flag}")
            intent_flag = self._format_bool_flag(open_intent)
            if intent_flag is not None:
                thr_parts.append(f"intent={intent_flag}")
            blocked_flag = self._format_bool_flag(open_blocked)
            if blocked_flag is not None:
                thr_parts.append(f"blocked={blocked_flag}")
            if thr_parts:
                lines.append(("thr: " + " ".join(thr_parts), (175, 220, 255)))
        elif final_g is not None:
            g_parts = [f"final={float(np.asarray(final_g).reshape(-1)[-1]):+.3f}"]
            if action_pre_gate is not None:
                g_parts.append(
                    f"pre_gate={float(np.asarray(action_pre_gate).reshape(-1)[-1]):+.3f}"
                )
            if action_pre_clip is not None:
                g_parts.append(
                    f"pre_clip={float(np.asarray(action_pre_clip).reshape(-1)[-1]):+.3f}"
                )
            lines.append(("g: " + " ".join(g_parts), (200, 255, 200)))
        return lines

    def _draw_action_on_frame(
        self,
        frame: np.ndarray,
        action: np.ndarray | None,
    ) -> np.ndarray:
        return draw_cartesian_action_on_frame(frame, action)

    def _set_gripper_jacobian_context(self, obs: PolicyObservation) -> None:
        view_keys = list(obs.view_keys) if obs.view_keys is not None else None
        view_count = self._obs_view_count(obs)
        exclude_keys = list(self.cfg.gripper_jacobian_exclude_view_keys or [])
        exclude_view_indices: list[int] = []
        active_exclude_view_keys: list[str] = []
        if view_keys is not None:
            for key in exclude_keys:
                if key in view_keys:
                    exclude_view_indices.append(view_keys.index(key))
                    active_exclude_view_keys.append(key)
        exclude_view_indices = sorted(set(exclude_view_indices))
        if view_count <= 1:
            allowed_view_indices = [0]
        else:
            allowed_view_indices = [
                view_idx
                for view_idx in range(int(view_count))
                if view_idx not in exclude_view_indices
            ]
        active_allowed_view_keys = (
            [view_keys[idx] for idx in allowed_view_indices]
            if view_keys is not None
            else None
        )
        self._gripper_jacobian_context = GripperJacobianContext(
            view_keys=view_keys,
            view_count=int(view_count),
            exclude_view_indices=exclude_view_indices,
            allowed_view_indices=allowed_view_indices,
            active_exclude_view_keys=active_exclude_view_keys,
            active_allowed_view_keys=active_allowed_view_keys,
        )
        self._last_gripper_jacobian_mask_info = {
            "gripper_jacobian_exclude_view_keys": exclude_keys,
            "gripper_jacobian_active_exclude_view_keys": active_exclude_view_keys,
            "gripper_jacobian_allowed_view_indices": list(allowed_view_indices),
            "gripper_jacobian_active_allowed_view_keys": active_allowed_view_keys,
            "gripper_jacobian_view_keys": view_keys,
            "gripper_jacobian_view_count": int(view_count),
        }

    def _clear_gripper_jacobian_context(self) -> None:
        self._gripper_jacobian_context = None

    @staticmethod
    def _flatten_view_rows(
        batch_rows: int,
        view_count: int,
        view_indices: list[int] | None,
    ) -> list[int]:
        if view_count <= 0 or batch_rows <= 0 or not view_indices:
            return []
        return [
            batch_idx * view_count + view_idx
            for batch_idx in range(batch_rows)
            for view_idx in view_indices
            if 0 <= view_idx < view_count
        ]

    @staticmethod
    def _resolve_action_dims(
        total_dims: int, dims: tuple[int, ...] | list[int]
    ) -> list[int]:
        resolved: list[int] = []
        for dim in dims:
            idx = int(dim)
            if idx < 0:
                idx = total_dims + idx
            if 0 <= idx < total_dims:
                resolved.append(idx)
        return sorted(set(resolved))

    def _focus_bottom_margin_px(self, height: int) -> int:
        if height <= 0:
            return 0
        if self.cfg.gripper_jacobian_focus_bottom_margin_px is not None:
            return max(
                0, min(int(self.cfg.gripper_jacobian_focus_bottom_margin_px), height)
            )
        ratio = max(float(self.cfg.gripper_jacobian_focus_bottom_margin_ratio), 0.0)
        return max(0, min(int(round(height * ratio)), height))

    def _mask_jacobian(
        self,
        jac: Tensor,
        *,
        view_keys: list[str] | None = None,
        num_views: int | None = None,
    ) -> Tensor:
        """Zero out configured action channels in raw jac (b c s h w); second dim is action."""
        jac = jac.clone()
        for ch in self.cfg.jacobian_zero_channels:
            if ch < jac.shape[1]:
                jac[:, ch, :, :, :] = 0.0

        context = self._gripper_jacobian_context
        exclude_view_indices = [] if context is None else context.exclude_view_indices
        allowed_view_indices = [] if context is None else context.allowed_view_indices
        view_count = 0 if context is None else int(context.view_count)
        exclude_rows: list[int] = []
        allowed_rows: list[int] = []
        if jac.shape[1] > 0:
            if view_count > 1 and jac.shape[0] % view_count == 0:
                batch_size = jac.shape[0] // view_count
                exclude_rows = self._flatten_view_rows(
                    batch_size, view_count, exclude_view_indices
                )
                allowed_rows = self._flatten_view_rows(
                    batch_size, view_count, allowed_view_indices
                )
            else:
                allowed_rows = list(range(int(jac.shape[0])))

            if exclude_rows:
                exclude_action_dims = self._resolve_action_dims(
                    int(jac.shape[1]), self.cfg.gripper_jacobian_exclude_action_dims
                )
                for action_dim in exclude_action_dims:
                    jac[exclude_rows, action_dim, :, :, :] = 0.0

            if allowed_rows:
                focus_dims = self._resolve_action_dims(
                    int(jac.shape[1]), [self.cfg.gripper_jacobian_focus_action_dim]
                )
                if focus_dims:
                    focus_dim = focus_dims[0]
                    bottom_margin = self._focus_bottom_margin_px(int(jac.shape[-2]))
                    if bottom_margin > 0:
                        jac[allowed_rows, focus_dim, :, -bottom_margin:, :] = 0.0
                    jac[allowed_rows, focus_dim, :, :, :] *= float(
                        self.cfg.gripper_jacobian_focus_gain
                    )

        if self._last_gripper_jacobian_mask_info is not None:
            self._last_gripper_jacobian_mask_info.update(
                {
                    "gripper_jacobian_excluded_row_count": len(exclude_rows),
                    "gripper_jacobian_allowed_row_count": len(allowed_rows),
                    "gripper_jacobian_exclude_action_dims": self._resolve_action_dims(
                        int(jac.shape[1]), self.cfg.gripper_jacobian_exclude_action_dims
                    ),
                    "gripper_jacobian_focus_action_dim": self._resolve_action_dims(
                        int(jac.shape[1]), [self.cfg.gripper_jacobian_focus_action_dim]
                    ),
                    "gripper_jacobian_focus_bottom_margin_px": self._focus_bottom_margin_px(
                        int(jac.shape[-2])
                    ),
                    "gripper_jacobian_focus_gain": float(
                        self.cfg.gripper_jacobian_focus_gain
                    ),
                }
            )

        return jac

    @staticmethod
    def _tensor_to_numpy(value: Tensor):
        return value.detach().cpu().numpy().copy()

    def _gripper_closed_from_obs(self, obs: PolicyObservation) -> torch.Tensor | None:
        gripper_qpos = obs.gripper_qpos
        if gripper_qpos is None or self.cfg.gripper_fixed_value is not None:
            return None

        gripper_qpos_np = np.asarray(gripper_qpos)
        if gripper_qpos_np.ndim == 1:
            gripper_qpos_np = np.expand_dims(gripper_qpos_np, axis=0)
        if gripper_qpos_np.shape[0] <= 0:
            return None

        if gripper_qpos_np.shape[-1] >= 2:
            # Use finger separation magnitude instead of signed width because
            # some envs expose mirrored finger joints with opposite sign
            # conventions (e.g. [+open, -open]).
            gripper_width = 0.5 * np.abs(
                gripper_qpos_np[..., 1] - gripper_qpos_np[..., 0]
            )
        else:
            gripper_width = np.abs(gripper_qpos_np[..., 0])

        closed_mask = gripper_width <= float(self.cfg.gripper_state_width_threshold)
        return torch.as_tensor(closed_mask, dtype=torch.bool, device=self.device)

    def _initialize_gripper_runtime_state(
        self,
        observed_closed: torch.Tensor,
    ) -> GripperRuntimeState:
        batch_size = int(observed_closed.shape[0])
        hold_remaining = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        initial_hold_steps = max(int(self.cfg.gripper_min_hold_steps), 0)
        if initial_hold_steps > 0:
            hold_remaining[observed_closed] = initial_hold_steps
        # Seed integrated position from observed: 1.0 if closed, 0.0 if open.
        # Used only by integrated_velocity mode.
        integrated_position = observed_closed.to(
            dtype=torch.float32, device=self.device
        )
        n0 = max(float(self.cfg.gripper_z_prior_n0), 1.0)
        prior_sigma = max(float(self.cfg.gripper_z_prior_sigma), 1e-6)
        return GripperRuntimeState(
            desired_closed=observed_closed.clone(),
            observed_closed=observed_closed.clone(),
            filtered_intent=torch.zeros(
                batch_size, dtype=torch.float32, device=self.device
            ),
            hold_remaining=hold_remaining,
            integrated_position=integrated_position,
            z_n=torch.full((batch_size,), n0, dtype=torch.float32, device=self.device),
            z_mean=torch.full(
                (batch_size,), float(self.cfg.gripper_z_prior_mu),
                dtype=torch.float32, device=self.device,
            ),
            z_m2=torch.full(
                (batch_size,), n0 * prior_sigma**2,
                dtype=torch.float32, device=self.device,
            ),
        )

    def _reseed_gripper_z_stats(self) -> None:
        """Re-seed the live Welford z-stats from the current cfg prior, preserving batch
        size. Used when configure_runtime changes the z-prior mid-session (M3)."""
        rs = self._gripper_runtime_state
        if rs is None or rs.z_n is None:
            return
        batch_size = int(rs.z_n.shape[0])
        n0 = max(float(self.cfg.gripper_z_prior_n0), 1.0)
        prior_sigma = max(float(self.cfg.gripper_z_prior_sigma), 1e-6)
        rs.z_n = torch.full(
            (batch_size,), n0, dtype=torch.float32, device=self.device
        )
        rs.z_mean = torch.full(
            (batch_size,), float(self.cfg.gripper_z_prior_mu),
            dtype=torch.float32, device=self.device,
        )
        rs.z_m2 = torch.full(
            (batch_size,), n0 * prior_sigma**2,
            dtype=torch.float32, device=self.device,
        )

    def sync_gripper_runtime_from_obs(
        self,
        obs: PolicyObservation,
        *,
        reset_alignment: bool = False,
    ) -> None:
        observed_closed = self._gripper_closed_from_obs(obs)
        if observed_closed is None:
            return

        needs_init = (
            self._gripper_runtime_state is None
            or self._gripper_runtime_state.batch_size != int(observed_closed.shape[0])
        )
        if needs_init:
            self._gripper_runtime_state = self._initialize_gripper_runtime_state(
                observed_closed
            )
            return

        runtime_state = self._gripper_runtime_state
        if runtime_state is None:
            return
        runtime_state.observed_closed = observed_closed.clone()
        if reset_alignment:
            realigned_state = self._initialize_gripper_runtime_state(observed_closed)
            runtime_state.desired_closed = realigned_state.desired_closed
            runtime_state.filtered_intent = realigned_state.filtered_intent
            runtime_state.hold_remaining = realigned_state.hold_remaining

    def _resolve_gripper_thresholds(self) -> tuple[float, float]:
        close_thresh = self.cfg.gripper_close_thresh
        open_thresh = self.cfg.gripper_open_thresh
        if close_thresh is None:
            close_thresh = self.cfg.gripper_thresh
        if open_thresh is None:
            open_thresh = self.cfg.gripper_thresh
        return float(close_thresh), float(open_thresh)

    def _apply_gripper_gate(self, du: Tensor) -> Tensor:
        """
        Convert the last action dimension into a binary open/close command
        with hysteresis, while keeping other dimensions unchanged.

        We interpret the raw last dimension as "intent":
          - if >  +thresh → close gripper
          - if <  -thresh → open gripper
          - otherwise     → keep previous state
        """
        if du.ndim == 1:
            du = du.unsqueeze(0)

        B = du.shape[0]
        device = du.device
        gripper_raw = du[..., -1]

        if (
            self._gripper_runtime_state is None
            or self._gripper_runtime_state.batch_size != B
        ):
            observed_closed = torch.zeros(B, dtype=torch.bool, device=device)
            self._gripper_runtime_state = self._initialize_gripper_runtime_state(
                observed_closed
            )
        runtime_state = self._gripper_runtime_state

        close_thresh, open_thresh = self._resolve_gripper_thresholds()
        smoothing = float(np.clip(self.cfg.gripper_signal_smoothing, 0.0, 0.999))
        hold_steps = max(int(self.cfg.gripper_min_hold_steps), 0)
        deadband_thresh = max(float(self.cfg.gripper_deadband_thresh), 0.0)
        deadband_active = gripper_raw.abs() < deadband_thresh
        gripper_filtered_candidate = (
            smoothing * runtime_state.filtered_intent + (1.0 - smoothing) * gripper_raw
        )
        gripper_filtered = torch.where(
            deadband_active,
            runtime_state.filtered_intent,
            gripper_filtered_candidate,
        )
        runtime_state.filtered_intent = gripper_filtered.detach()

        prev_closed = runtime_state.desired_closed.clone()
        hold_before = runtime_state.hold_remaining.clone()
        hold_after = torch.clamp(hold_before - 1, min=0)

        # Online intent stats (Welford with Tier-1 prior pseudo-count): track the SAME
        # signal the z-score tests — the FILTERED intent, not the raw intent. Previously
        # the Welford ran on gripper_raw while the z-score numerator used gripper_filtered
        # (EMA-smoothed), so the denominator's sigma overstated the numerator's spread and
        # |gripper_z| was deflated (~sqrt of the EMA variance ratio) — z_k_open fired well
        # below its nominal sigma multiple (M2). Tracking gripper_filtered makes z a true
        # standardized score of the gate's own signal, keeping the dimensionless/portable
        # property the docstring claims. Stats are still updated every call regardless of
        # gate mode, so a runtime switch to 'zscore' has warm stats.
        if runtime_state.z_n is None or runtime_state.z_n.shape[0] != B:
            seeded = self._initialize_gripper_runtime_state(
                torch.zeros(B, dtype=torch.bool, device=device)
            )
            runtime_state.z_n = seeded.z_n
            runtime_state.z_mean = seeded.z_mean
            runtime_state.z_m2 = seeded.z_m2
        runtime_state.z_n = runtime_state.z_n + 1.0
        delta = gripper_filtered - runtime_state.z_mean
        runtime_state.z_mean = runtime_state.z_mean + delta / runtime_state.z_n
        runtime_state.z_m2 = runtime_state.z_m2 + delta * (
            gripper_filtered - runtime_state.z_mean
        )
        z_sigma = torch.sqrt(runtime_state.z_m2 / runtime_state.z_n).clamp(min=1e-6)
        gripper_z = (gripper_filtered - runtime_state.z_mean) / z_sigma

        close_mask = (gripper_filtered > close_thresh) & (~deadband_active)
        if self.cfg.gripper_open_gate == "zscore":
            # Self-normalized open: only an excursion that is anomalously negative
            # relative to THIS episode's own intent distribution releases.
            open_intent = (gripper_z < -float(self.cfg.gripper_z_k_open)) & (
                ~deadband_active
            )
        else:
            open_intent = (gripper_filtered < -open_thresh) & (~deadband_active)
        open_blocked = hold_after > 0
        open_mask = open_intent & (~open_blocked)

        desired_closed = runtime_state.desired_closed.clone()
        desired_closed[close_mask] = True
        desired_closed[open_mask] = False

        just_closed = (~prev_closed) & desired_closed
        hold_after[just_closed] = hold_steps
        hold_after[~desired_closed] = 0
        runtime_state.desired_closed = desired_closed
        runtime_state.hold_remaining = hold_after

        du = du.clone()
        du[..., -1] = torch.where(
            runtime_state.desired_closed,
            torch.ones_like(gripper_raw),
            -torch.ones_like(gripper_raw),
        )
        self._last_gripper_debug_info = {
            "gripper_mode": "gated",
            "gripper_open_gate": str(self.cfg.gripper_open_gate),
            "gripper_z": self._tensor_to_numpy(gripper_z),
            "gripper_z_mean": self._tensor_to_numpy(runtime_state.z_mean),
            "gripper_z_sigma": self._tensor_to_numpy(z_sigma),
            "gripper_z_k_open": float(self.cfg.gripper_z_k_open),
            "gripper_raw": self._tensor_to_numpy(gripper_raw),
            "gripper_raw_filtered": self._tensor_to_numpy(gripper_filtered),
            "gripper_deadband_thresh": float(deadband_thresh),
            "gripper_deadband_active": self._tensor_to_numpy(deadband_active),
            "gripper_thresh": float(self.cfg.gripper_thresh),
            "gripper_close_thresh": float(close_thresh),
            "gripper_open_thresh": float(open_thresh),
            "gripper_signal_smoothing": float(smoothing),
            "gripper_close_mask": self._tensor_to_numpy(close_mask),
            "gripper_open_mask": self._tensor_to_numpy(open_mask),
            "gripper_open_intent": self._tensor_to_numpy(open_intent),
            "gripper_open_blocked": self._tensor_to_numpy(open_blocked),
            "gripper_observed_closed": self._tensor_to_numpy(
                runtime_state.observed_closed
            ),
            "gripper_desired_closed": self._tensor_to_numpy(
                runtime_state.desired_closed
            ),
            "gripper_closed_state": self._tensor_to_numpy(runtime_state.desired_closed),
            "gripper_hold_remaining": self._tensor_to_numpy(
                runtime_state.hold_remaining
            ),
            "gripper_final": self._tensor_to_numpy(du[..., -1]),
        }
        return du

    def _apply_continuous_gripper_command(self, du: Tensor) -> Tensor:
        if du.ndim == 1:
            du = du.unsqueeze(0)

        du = du.clone()
        gripper_raw = du[..., -1].clone()
        deadband_thresh = max(float(self.cfg.gripper_deadband_thresh), 0.0)
        deadband_active = torch.abs(gripper_raw) < deadband_thresh
        gripper_raw_after_deadband = torch.where(
            deadband_active,
            torch.zeros_like(gripper_raw),
            gripper_raw,
        )
        gripper_scaled = torch.clamp(
            gripper_raw_after_deadband * float(self.cfg.gripper_continuous_scale),
            min=-1.0,
            max=1.0,
        )
        du[..., -1] = gripper_scaled
        self._last_gripper_debug_info = {
            "gripper_mode": "continuous",
            "gripper_raw": self._tensor_to_numpy(gripper_raw),
            "gripper_raw_filtered": self._tensor_to_numpy(gripper_raw_after_deadband),
            "gripper_deadband_thresh": float(deadband_thresh),
            "gripper_deadband_active": self._tensor_to_numpy(deadband_active),
            "gripper_thresh": float(self.cfg.gripper_thresh),
            "gripper_close_thresh": None,
            "gripper_open_thresh": None,
            "gripper_signal_smoothing": float(self.cfg.gripper_signal_smoothing),
            "gripper_close_mask": self._tensor_to_numpy(
                torch.zeros_like(gripper_raw, dtype=torch.bool)
            ),
            "gripper_open_mask": self._tensor_to_numpy(
                torch.zeros_like(gripper_raw, dtype=torch.bool)
            ),
            "gripper_open_intent": self._tensor_to_numpy(
                torch.zeros_like(gripper_raw, dtype=torch.bool)
            ),
            "gripper_open_blocked": self._tensor_to_numpy(
                torch.zeros_like(gripper_raw, dtype=torch.bool)
            ),
            "gripper_observed_closed": (
                None
                if self._gripper_runtime_state is None
                else self._tensor_to_numpy(self._gripper_runtime_state.observed_closed)
            ),
            "gripper_desired_closed": None,
            "gripper_closed_state": None,
            "gripper_hold_remaining": (
                None
                if self._gripper_runtime_state is None
                else self._tensor_to_numpy(self._gripper_runtime_state.hold_remaining)
            ),
            "gripper_final": self._tensor_to_numpy(du[..., -1]),
            "gripper_continuous_scale": float(self.cfg.gripper_continuous_scale),
        }
        return du

    def _apply_integrated_velocity_gripper(self, du: Tensor) -> Tensor:
        """DreamZero-style binarization for velocity-mode gripper signal.

        Approach: integrate du[-1] (velocity) into a discrete position
        state g in [0, 1] (0 = open, 1 = closed). Threshold at 0.5 → binary
        ±1 command for the env. One gain knob (gripper_integrate_alpha);
        no hysteresis / no deadband / no hold steps.

        Optional snap-to-edge (gripper_integrate_snap_to_edge=True): once
        g crosses 0.5 in either direction, snap to 0.0 or 1.0 so subsequent
        small-velocity noise can't flicker the binary output.
        """
        if du.ndim == 1:
            du = du.unsqueeze(0)
        B = du.shape[0]
        device = du.device
        gripper_raw = du[..., -1].clone()

        if (
            self._gripper_runtime_state is None
            or self._gripper_runtime_state.batch_size != B
        ):
            observed_closed = torch.zeros(B, dtype=torch.bool, device=device)
            self._gripper_runtime_state = self._initialize_gripper_runtime_state(
                observed_closed
            )
        runtime_state = self._gripper_runtime_state
        if runtime_state.integrated_position is None:
            runtime_state.integrated_position = runtime_state.observed_closed.to(
                dtype=torch.float32, device=device
            )

        alpha = float(self.cfg.gripper_integrate_alpha)
        g = runtime_state.integrated_position
        g = torch.clamp(g + alpha * gripper_raw, min=0.0, max=1.0)

        if bool(self.cfg.gripper_integrate_snap_to_edge):
            # Once g crosses the midline, commit to the edge so noise can't
            # flicker the binary output. Equivalent to a zero-knob Schmitt.
            g = torch.where(g > 0.5, torch.ones_like(g), g)
            g = torch.where(g < 0.5, torch.zeros_like(g), g)

        runtime_state.integrated_position = g.detach()
        # Threshold at 0.5 → binary cmd. MimicGen convention: +1 close, -1 open.
        cmd = torch.where(
            g > 0.5,
            torch.ones_like(gripper_raw),
            -torch.ones_like(gripper_raw),
        )
        runtime_state.desired_closed = (g > 0.5).to(dtype=torch.bool)

        du = du.clone()
        du[..., -1] = cmd
        self._last_gripper_debug_info = {
            "gripper_mode": "integrated_velocity",
            "gripper_raw": self._tensor_to_numpy(gripper_raw),
            "gripper_raw_filtered": self._tensor_to_numpy(gripper_raw),
            "gripper_integrate_alpha": float(alpha),
            "gripper_integrate_snap_to_edge": bool(
                self.cfg.gripper_integrate_snap_to_edge
            ),
            "gripper_integrated_position": self._tensor_to_numpy(g),
            "gripper_deadband_thresh": 0.0,
            "gripper_deadband_active": self._tensor_to_numpy(
                torch.zeros_like(gripper_raw, dtype=torch.bool)
            ),
            "gripper_thresh": 0.5,
            "gripper_close_thresh": 0.5,
            "gripper_open_thresh": 0.5,
            "gripper_signal_smoothing": 0.0,
            "gripper_close_mask": self._tensor_to_numpy(g > 0.5),
            "gripper_open_mask": self._tensor_to_numpy(g <= 0.5),
            "gripper_open_intent": self._tensor_to_numpy(gripper_raw < 0.0),
            "gripper_open_blocked": self._tensor_to_numpy(
                torch.zeros_like(gripper_raw, dtype=torch.bool)
            ),
            "gripper_observed_closed": self._tensor_to_numpy(
                runtime_state.observed_closed
            ),
            "gripper_desired_closed": self._tensor_to_numpy(runtime_state.desired_closed),
            "gripper_closed_state": self._tensor_to_numpy(runtime_state.desired_closed),
            "gripper_hold_remaining": self._tensor_to_numpy(runtime_state.hold_remaining),
            "gripper_final": self._tensor_to_numpy(du[..., -1]),
        }
        return du

    @torch.no_grad()
    def predict_action(self, obs: PolicyObservation) -> PolicyOutput:
        self.sync_gripper_runtime_from_obs(
            obs,
            reset_alignment=bool(
                self.cfg.gripper_realign_on_step0
                and getattr(obs, "step_index", None) == 0
            ),
        )
        return super().predict_action(obs)

    def get_final_action(self, du: Tensor) -> Tensor:
        cfg = self.cfg
        du = du.clone()
        # Scale translation and yaw
        du[..., [0, 1, 2]] *= cfg.action_scale_translation
        du[..., [3, 4, 5]] *= cfg.action_scale_yaw

        # Zero roll/pitch (or other configured channels)
        for ch in cfg.action_zero_channels:
            if ch < du.shape[-1]:
                du[..., ch] = 0.0
        # Gripper: fixed value, continuous command, or binary gating
        if cfg.gripper_fixed_value is not None:
            gripper_raw = du[..., -1].clone()
            du[..., -1] = cfg.gripper_fixed_value
            close_thresh, open_thresh = self._resolve_gripper_thresholds()
            self._last_gripper_debug_info = {
                "gripper_mode": "fixed",
                "gripper_raw": self._tensor_to_numpy(gripper_raw),
                "gripper_raw_filtered": self._tensor_to_numpy(gripper_raw),
                "gripper_deadband_thresh": float(cfg.gripper_deadband_thresh),
                "gripper_deadband_active": self._tensor_to_numpy(
                    torch.zeros_like(gripper_raw, dtype=torch.bool)
                ),
                "gripper_thresh": float(cfg.gripper_thresh),
                "gripper_close_thresh": float(close_thresh),
                "gripper_open_thresh": float(open_thresh),
                "gripper_signal_smoothing": float(cfg.gripper_signal_smoothing),
                "gripper_close_mask": self._tensor_to_numpy(
                    gripper_raw > float(close_thresh)
                ),
                "gripper_open_mask": self._tensor_to_numpy(
                    gripper_raw < -float(open_thresh)
                ),
                "gripper_open_intent": self._tensor_to_numpy(
                    gripper_raw < -float(open_thresh)
                ),
                "gripper_open_blocked": self._tensor_to_numpy(
                    torch.zeros_like(gripper_raw, dtype=torch.bool)
                ),
                "gripper_closed_state": None,
                "gripper_hold_remaining": self._tensor_to_numpy(
                    torch.zeros_like(gripper_raw, dtype=torch.long)
                ),
                "gripper_final": self._tensor_to_numpy(du[..., -1]),
                "gripper_fixed_value": float(cfg.gripper_fixed_value),
            }
        elif cfg.gripper_command_mode == "continuous":
            du = self._apply_continuous_gripper_command(du)
        elif cfg.gripper_command_mode == "integrated_velocity":
            du = self._apply_integrated_velocity_gripper(du)
        else:
            du = self._apply_gripper_gate(du)
        return du

    def _collect_policy_debug_info(
        self,
        action_debug: dict[str, np.ndarray] | None,
    ) -> dict:
        info = super()._collect_policy_debug_info(action_debug)
        if self._last_gripper_debug_info is not None:
            gripper_debug = dict(self._last_gripper_debug_info)
            adaptive_channel_state = getattr(self, "_adaptive_channel_state", None)
            if adaptive_channel_state is not None:
                gripper_debug["adaptive_translation_scale"] = float(
                    adaptive_channel_state.translation
                )
                gripper_debug["adaptive_rotation_scale"] = float(
                    adaptive_channel_state.rotation
                )
                gripper_debug["adaptive_gripper_scale"] = float(
                    adaptive_channel_state.gripper
                )
            if action_debug is not None:
                pre_clip = action_debug.get("action_pre_clip")
                if pre_clip is not None:
                    gripper_debug["gripper_pre_clip"] = np.asarray(pre_clip)[
                        ..., -1
                    ].copy()
                pre_gate = action_debug.get("action_pre_gate")
                if pre_gate is not None:
                    gripper_debug["gripper_pre_gate"] = np.asarray(pre_gate)[
                        ..., -1
                    ].copy()
                final = action_debug.get("action_final")
                if final is not None:
                    gripper_debug["gripper_final"] = np.asarray(final)[..., -1].copy()
            info["gripper_debug"] = gripper_debug
        if self._last_gripper_jacobian_mask_info is not None:
            info["gripper_jacobian_mask"] = dict(self._last_gripper_jacobian_mask_info)
        return info

    def compute_jacobian(self, obs: PolicyObservation) -> Tensor:
        self._set_gripper_jacobian_context(obs)
        try:
            return super().compute_jacobian(obs)
        finally:
            self._clear_gripper_jacobian_context()

    def _flow_rgb_chunk_to_actions(
        self,
        obs: PolicyObservation,
        rgb: Tensor,
        flow: Tensor,
        source_view_widths: list[int],
        lam_override: float | None = None,
    ):
        self._set_gripper_jacobian_context(obs)
        try:
            return super()._flow_rgb_chunk_to_actions(
                obs,
                rgb,
                flow,
                source_view_widths,
                lam_override=lam_override,
            )
        finally:
            self._clear_gripper_jacobian_context()

    def _track_rgb_chunk_to_actions(
        self,
        obs: PolicyObservation,
        source_rgb: Tensor,
        tracks: dict,
        source_view_widths: list[int],
        target_rgb: Tensor | None = None,
        lam_override: float | None = None,
    ):
        self._set_gripper_jacobian_context(obs)
        try:
            return super()._track_rgb_chunk_to_actions(
                obs,
                source_rgb,
                tracks,
                source_view_widths,
                target_rgb=target_rgb,
                lam_override=lam_override,
            )
        finally:
            self._clear_gripper_jacobian_context()
