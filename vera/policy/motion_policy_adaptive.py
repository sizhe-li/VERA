import json
import warnings
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from torch import Tensor

from .base_policy import PolicyObservation
from .cartesian_policy_support import AdaptiveChannelState


class MotionPolicyAdaptiveMixin:
    def _planner_text_conditioning(self) -> str | list[str] | None:
        return self.cfg.text_conditioning

    def _adaptive_enabled(self) -> bool:
        return bool(
            getattr(getattr(self.cfg, "adaptive_controller", None), "enabled", False)
        )

    def _adaptive_mode(self) -> str:
        return str(getattr(self.cfg.adaptive_controller, "mode", "track"))

    def _adaptive_uses_grouped_state_delta(self) -> bool:
        return self._adaptive_enabled() and self._adaptive_mode() == "state_delta_grouped"

    def _adaptive_uses_track_mismatch_for_lam(self) -> bool:
        if not self._adaptive_enabled():
            return False
        if self._adaptive_mode() == "track":
            return True
        return bool(self.cfg.adaptive_controller.use_track_mismatch_for_lam)

    def _runtime_action_channel_scales(self, action_dim: int) -> np.ndarray:
        return np.ones(int(max(action_dim, 0)), dtype=np.float32)

    def _runtime_controller_params(self) -> dict[str, Any]:
        base_lam = float(self.cfg.controller.lam)
        base_action_scale = float(self.cfg.controller.action_scale)
        mismatch_ema = max(float(self._adaptive_mismatch_ema), 0.0)
        if not self._adaptive_enabled():
            return {
                "enabled": False,
                "mismatch_ema": mismatch_ema,
                "lam_runtime": base_lam,
                "action_scale_runtime": base_action_scale,
                "lam_scale": 1.0,
                "action_gain_scale": 1.0,
                "action_channel_scales_runtime": self._runtime_action_channel_scales(7),
                "channel_gain_state": self._adaptive_channel_state.as_dict(),
            }

        cfg = self.cfg.adaptive_controller
        lam_scale = 1.0
        if self._adaptive_uses_track_mismatch_for_lam():
            lam_scale = float(np.clip(1.0 + mismatch_ema, 1.0, cfg.lam_max_scale))

        gain_scale = 1.0
        if self._adaptive_mode() == "track":
            gain_scale = float(
                np.clip(
                    1.0 / (1.0 + mismatch_ema),
                    cfg.action_gain_min_scale,
                    cfg.action_gain_max_scale,
                )
            )
        action_channel_scales = self._runtime_action_channel_scales(7)
        return {
            "enabled": True,
            "mode": self._adaptive_mode(),
            "mismatch_ema": mismatch_ema,
            "lam_runtime": base_lam * lam_scale,
            "action_scale_runtime": base_action_scale * gain_scale,
            "lam_scale": lam_scale,
            "action_gain_scale": gain_scale,
            "action_channel_scales_runtime": action_channel_scales,
            "channel_gain_state": self._adaptive_channel_state.as_dict(),
        }

    def _adaptive_debug_info(self) -> dict[str, Any] | None:
        if not self._adaptive_enabled():
            return None
        info: dict[str, Any] = dict(self._runtime_controller_params())
        if self._adaptive_last_feedback is not None:
            info["last_feedback"] = dict(self._adaptive_last_feedback)
        info["pending_feedback_steps"] = int(len(self._feedback_queue))
        return info

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, dict):
            return {
                str(key): MotionPolicyAdaptiveMixin._json_safe(val)
                for key, val in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [MotionPolicyAdaptiveMixin._json_safe(item) for item in value]
        return value

    def _apply_runtime_action_scaling(
        self,
        du: Tensor,
        runtime_controller: dict[str, Any],
    ) -> Tensor:
        du_scaled = du * float(runtime_controller["action_scale_runtime"])
        channel_scales = runtime_controller.get("action_channel_scales_runtime")
        if channel_scales is None:
            return du_scaled
        scale_tensor = torch.as_tensor(
            np.asarray(channel_scales, dtype=np.float32),
            device=du.device,
            dtype=du.dtype,
        )
        if scale_tensor.shape[0] < du.shape[-1]:
            pad = torch.ones(
                du.shape[-1] - scale_tensor.shape[0],
                device=du.device,
                dtype=du.dtype,
            )
            scale_tensor = torch.cat([scale_tensor, pad], dim=0)
        return du_scaled * scale_tensor[: du.shape[-1]]

    @staticmethod
    def _action_group_summary(action: np.ndarray | None) -> dict[str, Any] | None:
        if action is None:
            return None
        action_arr = np.asarray(action, dtype=np.float32)
        if action_arr.ndim == 1:
            action_arr = action_arr[None, :]
        return {
            "action": action_arr.copy(),
            "action_norm": np.linalg.norm(action_arr, axis=-1),
            "action_abs_mean": np.mean(np.abs(action_arr), axis=-1),
        }

    @staticmethod
    def _extract_feedback_state(obs: PolicyObservation) -> dict[str, np.ndarray] | None:
        del obs
        return None

    def _feedback_payload_extras(self, obs: PolicyObservation) -> dict[str, Any]:
        del obs
        return {}

    def _build_action_feedback_payload(
        self,
        obs: PolicyObservation,
        action_debug: dict[str, np.ndarray] | None,
    ) -> None:
        if action_debug is None:
            self._last_action_feedback_payload = None
            return
        dream_action = action_debug.get("action_final")
        self._last_action_feedback_payload = {
            "step_index": None if obs.step_index is None else int(obs.step_index),
            "dream_index": int(self._current_dream_index),
            "action_debug": {
                key: np.asarray(value).copy() for key, value in action_debug.items()
            },
            "dream_action_groups": self._action_group_summary(dream_action),
            "pre_step_state": self._extract_feedback_state(obs),
            "action_mode": str(obs.action_mode),
            "dt": float(obs.dt),
        }
        self._last_action_feedback_payload.update(self._feedback_payload_extras(obs))

    def _estimate_realized_action(
        self,
        pre_state: dict[str, np.ndarray] | None,
        obs: PolicyObservation,
    ) -> np.ndarray | None:
        del pre_state, obs
        return None

    def _update_grouped_adaptive_state(
        self,
        dream_action: np.ndarray | None,
        realized_action: np.ndarray | None,
        *,
        gripper_mode: str | None = None,
    ) -> dict[str, Any] | None:
        del dream_action, realized_action, gripper_mode
        return None

    def get_warm_start_state(self) -> dict | None:
        if not self._adaptive_enabled():
            return None
        return {
            "adaptive_mismatch_ema": float(self._adaptive_mismatch_ema),
            "adaptive_channel_state": self._adaptive_channel_state.as_dict(),
        }

    def set_warm_start_state(self, state: dict | None) -> None:
        if not self._adaptive_enabled() or not isinstance(state, dict):
            return
        mismatch_ema = state.get("adaptive_mismatch_ema")
        if mismatch_ema is not None:
            self._adaptive_mismatch_ema = max(float(mismatch_ema), 0.0)
        channel_state = state.get("adaptive_channel_state")
        if isinstance(channel_state, dict):
            self._adaptive_channel_state = AdaptiveChannelState(
                translation=float(channel_state.get("translation", 1.0)),
                rotation=float(channel_state.get("rotation", 1.0)),
                gripper=float(channel_state.get("gripper", 1.0)),
            )

    def _adaptive_cfg_dict(self) -> dict[str, float | bool | str]:
        cfg = self.cfg.adaptive_controller
        return {
            "enabled": bool(cfg.enabled),
            "mode": str(cfg.mode),
            "ema_alpha": float(cfg.ema_alpha),
            "invalid_track_penalty": float(cfg.invalid_track_penalty),
            "lam_max_scale": float(cfg.lam_max_scale),
            "action_gain_min_scale": float(cfg.action_gain_min_scale),
            "action_gain_max_scale": float(cfg.action_gain_max_scale),
            "eps": float(cfg.eps),
            "use_track_mismatch_for_lam": bool(cfg.use_track_mismatch_for_lam),
            "grouped_action_ema_alpha": float(cfg.grouped_action_ema_alpha),
            "grouped_action_eps": float(cfg.grouped_action_eps),
            "grouped_action_dt_eps": float(cfg.grouped_action_dt_eps),
            "grouped_action_step_up": float(cfg.grouped_action_step_up),
            "grouped_action_step_down": float(cfg.grouped_action_step_down),
            "translation_gain_min_scale": float(cfg.translation_gain_min_scale),
            "translation_gain_max_scale": float(cfg.translation_gain_max_scale),
            "rotation_gain_min_scale": float(cfg.rotation_gain_min_scale),
            "rotation_gain_max_scale": float(cfg.rotation_gain_max_scale),
            "gripper_gain_min_scale": float(cfg.gripper_gain_min_scale),
            "gripper_gain_max_scale": float(cfg.gripper_gain_max_scale),
            "enable_gripper_channel_adaptation": bool(
                cfg.enable_gripper_channel_adaptation
            ),
        }

    def reset_adaptive_controller_state(self) -> None:
        self._feedback_queue.clear()
        # Honor synthetic warm-start: cfg.initial_mismatch_ema > 0 puts the
        # controller into a defensive regime at step 0 (lower gain_scale,
        # higher lam_scale), replacing the lost Mar-12 adaptive snapshot.
        initial_ema = float(
            getattr(self.cfg.adaptive_controller, "initial_mismatch_ema", 0.0)
        )
        self._adaptive_mismatch_ema = max(initial_ema, 0.0)
        self._adaptive_channel_state = AdaptiveChannelState()
        self._adaptive_last_feedback = None
        self._last_action_feedback_payload = None

    def configure_adaptive_controller(
        self,
        *,
        enabled: bool | None = None,
        reset_state: bool = False,
        mode: Literal["track", "state_delta_grouped"] | None = None,
        ema_alpha: float | None = None,
        invalid_track_penalty: float | None = None,
        lam_max_scale: float | None = None,
        action_gain_min_scale: float | None = None,
        action_gain_max_scale: float | None = None,
        eps: float | None = None,
        use_track_mismatch_for_lam: bool | None = None,
        grouped_action_ema_alpha: float | None = None,
        grouped_action_eps: float | None = None,
        grouped_action_dt_eps: float | None = None,
        grouped_action_step_up: float | None = None,
        grouped_action_step_down: float | None = None,
        translation_gain_min_scale: float | None = None,
        translation_gain_max_scale: float | None = None,
        rotation_gain_min_scale: float | None = None,
        rotation_gain_max_scale: float | None = None,
        gripper_gain_min_scale: float | None = None,
        gripper_gain_max_scale: float | None = None,
        enable_gripper_channel_adaptation: bool | None = None,
        initial_mismatch_ema: float | None = None,
    ) -> dict[str, Any]:
        cfg = self.cfg.adaptive_controller
        if enabled is not None:
            cfg.enabled = bool(enabled)
        if mode is not None:
            cfg.mode = cast(Literal["track", "state_delta_grouped"], mode)

        overrides = {
            "ema_alpha": ema_alpha,
            "invalid_track_penalty": invalid_track_penalty,
            "lam_max_scale": lam_max_scale,
            "action_gain_min_scale": action_gain_min_scale,
            "action_gain_max_scale": action_gain_max_scale,
            "eps": eps,
            "grouped_action_ema_alpha": grouped_action_ema_alpha,
            "grouped_action_eps": grouped_action_eps,
            "grouped_action_dt_eps": grouped_action_dt_eps,
            "grouped_action_step_up": grouped_action_step_up,
            "grouped_action_step_down": grouped_action_step_down,
            "translation_gain_min_scale": translation_gain_min_scale,
            "translation_gain_max_scale": translation_gain_max_scale,
            "rotation_gain_min_scale": rotation_gain_min_scale,
            "rotation_gain_max_scale": rotation_gain_max_scale,
            "gripper_gain_min_scale": gripper_gain_min_scale,
            "gripper_gain_max_scale": gripper_gain_max_scale,
            "initial_mismatch_ema": initial_mismatch_ema,
        }
        for key, value in overrides.items():
            if value is not None:
                setattr(cfg, key, value)
        if use_track_mismatch_for_lam is not None:
            cfg.use_track_mismatch_for_lam = bool(use_track_mismatch_for_lam)
        if enable_gripper_channel_adaptation is not None:
            cfg.enable_gripper_channel_adaptation = bool(
                enable_gripper_channel_adaptation
            )

        if reset_state:
            self.reset_adaptive_controller_state()
        return self.get_adaptive_controller_runtime()

    def get_adaptive_controller_runtime(self) -> dict[str, Any]:
        runtime: dict[str, Any] = dict(self._runtime_controller_params())
        runtime["pending_feedback_steps"] = int(len(self._feedback_queue))
        if self._adaptive_last_feedback is not None:
            runtime["last_feedback"] = dict(self._adaptive_last_feedback)
        return self._json_safe(
            {
                "adaptive_cfg": self._adaptive_cfg_dict(),
                "runtime": runtime,
                "warm_start_state": self.get_warm_start_state(),
            }
        )

    def get_adaptive_controller_snapshot(self) -> dict[str, Any]:
        runtime = self.get_adaptive_controller_runtime()
        return self._json_safe(
            {
                "adaptive_cfg": runtime["adaptive_cfg"],
                "warm_start_state": runtime["warm_start_state"],
                "final_runtime": runtime["runtime"],
            }
        )

    def load_adaptive_controller_snapshot(
        self,
        snapshot: dict[str, Any] | str | Path,
        *,
        apply_config: bool = True,
        reset_state: bool = True,
    ) -> dict[str, Any]:
        payload: dict[str, Any]
        if isinstance(snapshot, (str, Path)):
            snapshot_path = Path(snapshot)
            payload = json.loads(snapshot_path.read_text())
        elif isinstance(snapshot, dict):
            payload = snapshot
        else:
            raise TypeError("snapshot must be a dict, str path, or pathlib.Path")

        adaptive_cfg = payload.get("adaptive_cfg", {})
        if apply_config and isinstance(adaptive_cfg, dict):
            self.configure_adaptive_controller(
                enabled=adaptive_cfg.get("enabled"),
                reset_state=False,
                mode=adaptive_cfg.get("mode"),
                ema_alpha=adaptive_cfg.get("ema_alpha"),
                invalid_track_penalty=adaptive_cfg.get("invalid_track_penalty"),
                lam_max_scale=adaptive_cfg.get("lam_max_scale"),
                action_gain_min_scale=adaptive_cfg.get("action_gain_min_scale"),
                action_gain_max_scale=adaptive_cfg.get("action_gain_max_scale"),
                eps=adaptive_cfg.get("eps"),
                use_track_mismatch_for_lam=adaptive_cfg.get(
                    "use_track_mismatch_for_lam"
                ),
                grouped_action_ema_alpha=adaptive_cfg.get("grouped_action_ema_alpha"),
                grouped_action_eps=adaptive_cfg.get("grouped_action_eps"),
                grouped_action_dt_eps=adaptive_cfg.get("grouped_action_dt_eps"),
                grouped_action_step_up=adaptive_cfg.get("grouped_action_step_up"),
                grouped_action_step_down=adaptive_cfg.get("grouped_action_step_down"),
                translation_gain_min_scale=adaptive_cfg.get(
                    "translation_gain_min_scale"
                ),
                translation_gain_max_scale=adaptive_cfg.get(
                    "translation_gain_max_scale"
                ),
                rotation_gain_min_scale=adaptive_cfg.get("rotation_gain_min_scale"),
                rotation_gain_max_scale=adaptive_cfg.get("rotation_gain_max_scale"),
                gripper_gain_min_scale=adaptive_cfg.get("gripper_gain_min_scale"),
                gripper_gain_max_scale=adaptive_cfg.get("gripper_gain_max_scale"),
                enable_gripper_channel_adaptation=adaptive_cfg.get(
                    "enable_gripper_channel_adaptation"
                ),
            )

        if reset_state:
            self.reset_adaptive_controller_state()

        warm_start_state = payload.get("warm_start_state")
        if isinstance(warm_start_state, dict):
            self.set_warm_start_state(warm_start_state)
        else:
            final_runtime = payload.get("final_runtime", {})
            mismatch_ema = (
                final_runtime.get("mismatch_ema")
                if isinstance(final_runtime, dict)
                else None
            )
            if mismatch_ema is not None:
                self.set_warm_start_state(
                    {
                        "adaptive_mismatch_ema": mismatch_ema,
                        "adaptive_channel_state": (
                            final_runtime.get("channel_gain_state")
                            if isinstance(final_runtime, dict)
                            else None
                        ),
                    }
                )

        return self.get_adaptive_controller_runtime()

    def save_adaptive_controller_snapshot(
        self,
        path: str | Path,
    ) -> dict[str, Any]:
        snapshot = self.get_adaptive_controller_snapshot()
        Path(path).write_text(json.dumps(self._json_safe(snapshot), indent=2))
        return snapshot

    def _get_feedback_tracker(self):
        if self._feedback_tracker is not None:
            return self._feedback_tracker

        from vera.policy.world_models.tracker_backends import build_motion_tracker

        self._feedback_tracker = build_motion_tracker(
            self.cfg.motion_planner,
            device=self.device,
        )
        return self._feedback_tracker

    def _track_feedback_rgb_pair(
        self,
        rgb_pair: Tensor,
        *,
        view_keys: list[str] | None = None,
        view_widths: list[int] | None = None,
    ):
        from vera.policy.world_models.tracker_backends import infer_multiview_tracks

        rgb_pair = rgb_pair.to(device=self.device, dtype=torch.float32)
        tracker = self._get_feedback_tracker()
        return infer_multiview_tracks(
            tracker,
            rgb_pair,
            return_visualization=False,
            view_keys=view_keys,
            view_widths=view_widths,
        ).motion_tracks

    def _queue_adaptive_feedback_targets(
        self,
        source_rgb: Tensor,
        tracks: dict[str, Any],
        *,
        num_steps: int,
        view_keys: list[str] | None,
        view_widths: list[int],
        dream_index: int,
    ) -> None:
        if not self._adaptive_enabled() or num_steps <= 0:
            return
        required_keys = ("disp", "valid", "idx_src")
        if any(key not in tracks for key in required_keys):
            return
        for step_idx in range(int(num_steps)):
            self._feedback_queue.append(
                {
                    "source_rgb": source_rgb[:, step_idx].detach().cpu(),
                    "pred_disp": tracks["disp"][:, step_idx].detach().cpu(),
                    "pred_valid": tracks["valid"][:, step_idx].detach().cpu(),
                    "pred_idx_src": tracks["idx_src"][:, step_idx].detach().cpu(),
                    "view_keys": None if view_keys is None else list(view_keys),
                    "view_widths": [int(width) for width in view_widths],
                    "dream_index": int(dream_index),
                    "step_in_chunk": int(step_idx),
                }
            )

    def _compute_track_feedback_metrics(
        self,
        payload: dict[str, Any],
        obs: PolicyObservation,
    ) -> dict[str, Any]:
        next_rgb = self._resize(obs.rgb, self.image_size_motion_planner).detach().cpu()
        source_rgb = payload["source_rgb"]
        if next_rgb.shape != source_rgb.shape:
            raise ValueError(
                f"feedback rgb shape mismatch: next={tuple(next_rgb.shape)} source={tuple(source_rgb.shape)}"
            )

        actual_tracks = self._track_feedback_rgb_pair(
            torch.stack([source_rgb, next_rgb], dim=1),
            view_keys=payload.get("view_keys"),
            view_widths=payload.get("view_widths"),
        ).as_policy_dict()

        pred_disp = payload["pred_disp"].float()
        pred_valid = payload["pred_valid"].float()
        pred_idx_src = payload["pred_idx_src"].long()
        actual_disp = actual_tracks["disp"][:, 0].float()
        actual_valid = actual_tracks["valid"][:, 0].float()
        pred_idx_src = pred_idx_src.clamp(0, actual_disp.shape[1] - 1)
        gather_disp_idx = pred_idx_src.unsqueeze(-1).expand(-1, -1, actual_disp.shape[-1])
        actual_disp_at_pred = torch.gather(actual_disp, dim=1, index=gather_disp_idx)
        actual_valid_at_pred = torch.gather(actual_valid, dim=1, index=pred_idx_src)

        valid = pred_valid * actual_valid_at_pred
        valid_mask = valid > 0.5
        error_norm = (actual_disp_at_pred - pred_disp).norm(dim=-1)
        pred_norm = pred_disp.norm(dim=-1)
        if valid_mask.any():
            endpoint_rmse = float(torch.sqrt((error_norm[valid_mask] ** 2).mean()).item())
            normalized_track_error = float(
                (
                    error_norm[valid_mask]
                    / (pred_norm[valid_mask] + self.cfg.adaptive_controller.eps)
                )
                .mean()
                .item()
            )
        else:
            endpoint_rmse = float("nan")
            normalized_track_error = 1.0

        valid_track_fraction = float(valid.mean().item())
        mismatch = float(
            normalized_track_error
            + self.cfg.adaptive_controller.invalid_track_penalty
            * (1.0 - valid_track_fraction)
        )
        return {
            "mismatch": mismatch,
            "normalized_track_error": normalized_track_error,
            "endpoint_rmse": endpoint_rmse,
            "valid_track_fraction": valid_track_fraction,
            "dream_index": int(payload.get("dream_index", -1)),
            "step_in_chunk": int(payload.get("step_in_chunk", -1)),
        }

    @torch.no_grad()
    def observe_rollout_feedback(
        self,
        obs: PolicyObservation,
    ) -> dict | None:
        if not self._adaptive_enabled():
            return None
        if len(self._feedback_queue) == 0 and self._last_action_feedback_payload is None:
            return None

        payload = self._feedback_queue.popleft() if len(self._feedback_queue) > 0 else None
        try:
            feedback_summary: dict[str, Any] = {}
            if payload is not None:
                track_feedback = self._compute_track_feedback_metrics(payload, obs)
                feedback_summary.update(track_feedback)
                alpha = float(np.clip(self.cfg.adaptive_controller.ema_alpha, 0.0, 1.0))
                self._adaptive_mismatch_ema = (1.0 - alpha) * float(
                    self._adaptive_mismatch_ema
                ) + alpha * float(track_feedback["mismatch"])

            action_payload = self._last_action_feedback_payload
            if action_payload is not None and self._adaptive_uses_grouped_state_delta():
                action_debug = cast(dict[str, np.ndarray], action_payload["action_debug"])
                realized_action = self._estimate_realized_action(
                    cast(dict[str, np.ndarray] | None, action_payload.get("pre_step_state")),
                    obs,
                )
                grouped_feedback = self._update_grouped_adaptive_state(
                    action_debug.get("action_final"),
                    realized_action,
                    gripper_mode=(
                        None
                        if not isinstance(action_payload.get("gripper_mode"), str)
                        else cast(str, action_payload.get("gripper_mode"))
                    ),
                )
                feedback_summary["dream_action"] = self._action_group_summary(
                    action_debug.get("action_final")
                )
                feedback_summary["realized_action"] = self._action_group_summary(
                    realized_action
                )
                feedback_summary["grouped_action_feedback"] = grouped_feedback
                feedback_summary["feedback_step_index"] = action_payload.get("step_index")
                feedback_summary["dream_index"] = int(
                    action_payload.get(
                        "dream_index", feedback_summary.get("dream_index", -1)
                    )
                )

            runtime_controller = self._runtime_controller_params()
            feedback_summary.update(
                {
                    "mode": self._adaptive_mode(),
                    "channel_gain_state": runtime_controller["channel_gain_state"],
                    "action_channel_scales_runtime": np.asarray(
                        runtime_controller["action_channel_scales_runtime"],
                        dtype=np.float32,
                    ).copy(),
                }
            )
            self._adaptive_last_feedback = {
                **feedback_summary,
                "mismatch_ema": float(runtime_controller["mismatch_ema"]),
                "lam_runtime": float(runtime_controller["lam_runtime"]),
                "action_scale_runtime": float(
                    runtime_controller["action_scale_runtime"]
                ),
            }
            self._last_action_feedback_payload = None
            return dict(self._adaptive_last_feedback)
        except Exception as exc:
            warnings.warn(
                f"Adaptive rollout feedback failed; leaving controller state unchanged: {exc}",
                RuntimeWarning,
            )
            self._adaptive_last_feedback = {"error": str(exc)}
            self._last_action_feedback_payload = None
            return dict(self._adaptive_last_feedback)
