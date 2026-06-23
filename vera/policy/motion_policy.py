from collections import deque
from datetime import datetime, timezone
import atexit
import json
import os
import queue
import threading
from pathlib import Path
import re
import warnings
from typing import Any, Dict, Tuple, cast

import numpy as np
import torch
from einops import rearrange
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from vera.idm.jacobian.models.base import (
    InputObservation as JacobianInputObservation,
)
from vera.idm.registry import (
    resolve_algorithm_cfg,
    resolve_algorithm_instance,
)
from vera.datasets.normalization import (
    denormalize_action,
    denormalize_jacobian_tensor,
)
from vera.policy.motion_planner_registry import build_motion_planner
from vera.utils import alltracker
from vera.utils.transforms import resize_flow

from .base_policy import (
    BasePolicy,
    PolicyObservation,
    PolicyOutput,
    populate_queues,
)
from .cartesian_policy_support import (
    AdaptiveChannelState,
    AdaptiveControllerCfg,
)
from .motion_policy_loading import (
    _apply_wan_planner_overrides,
    _extract_normalization_metadata,
    _is_wan_motion_planner,
    _load_algorithm_config_from_path,
    _resolve_planner_checkpoint_and_config,
    _summarize_normalization_metadata,
    _write_resolved_algorithm_config,
    load_checkpoint,
)
from .motion_policy_adaptive import MotionPolicyAdaptiveMixin
from .motion_policy_types import (
    ControllerCfg,
    DynamicsCfg,
    JacobianController,
    ModelCheckpoint,
    MotionPolicyCfg,
    PlannerCfg,
    tikhonov_solve,
)
from .motion_policy_visualization import MotionPolicyVisualizationMixin


UNSET = object()

# Set to True only for debugging; keeps rollout logs free of context/shape dumps.
VERBOSE_POLICY_DEBUG = False


def _maybe_print(enabled: bool, *args, **kwargs) -> None:
    if enabled:
        print(*args, **kwargs)


# ================================================================
# Motion Policy
# ================================================================


class MotionPolicy(
    MotionPolicyAdaptiveMixin, MotionPolicyVisualizationMixin, BasePolicy
):
    """
    image → flow plan → dense target → Jacobian → calibrated controller → action
    """

    cfg: MotionPolicyCfg
    motion_planner: Any  # from registry (e.g. DFoTMotionPolicy)
    dynamics_model: Any  # from registry (e.g. JacobianVanilla)
    robot_name: str

    # Which side of the multiview canvas an omni model's black view-padding sits on. Used to crop
    # the pad off the dream so view separation is geometric (default "right" = omni view_pad_position).
    _view_pad_position: str = "right"

    def __init__(self, cfg: MotionPolicyCfg, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self._motion_planner_normalization_meta: dict[str, Any] = {}
        self._dynamics_normalization_meta: dict[str, Any] = {}

        self.load_motion_planner()

        # Non-zero FSDP ranks only participate in WAN forward collectives;
        # they never use the dynamics model or controller results.
        if self._is_fsdp_worker_rank():
            self.dynamics_model = None
            self.robot_name = "unknown"
            self._log(
                f"[rank {torch.distributed.get_rank()}] Skipping dynamics model (FSDP worker)"
            )
        else:
            self.load_dynamics_model()
            self._check_normalization_compatibility()

        self.controller = JacobianController(cfg.controller)

        self._queues: Dict[str, deque] = {}
        # Queues for joint (video) model: one action and one vis frame per step.
        self._action_queue: deque = deque()
        self._vis_queue: deque = deque()
        self._feedback_queue: deque = deque()
        self._feedback_tracker = None
        self._adaptive_mismatch_ema = 0.0
        self._adaptive_channel_state = AdaptiveChannelState()
        self._adaptive_last_feedback: dict[str, Any] | None = None
        self._last_action_feedback_payload: dict[str, Any] | None = None
        self._debug_dump_run_dir: Path | None = None
        self._debug_dump_chunk_idx = 0
        self._debug_dump_bytes_written = 0
        self._debug_dump_warned = False
        # Async chunk-save: a single daemon writer thread drains a bounded queue so the
        # zlib compress + npz write never blocks the inference critical path (the chunk
        # is returned to the client immediately; the save happens off-thread). The lock
        # guards the run-state counters / budget reservation shared with the writer.
        self._debug_dump_lock = threading.Lock()
        self._debug_dump_queue: "queue.Queue | None" = None
        self._debug_dump_thread: threading.Thread | None = None
        self._debug_dump_atexit_registered = False
        self.reset()

    def _verbose_enabled(self) -> bool:
        return bool(getattr(self.cfg, "verbose", True))

    def _log(self, *args, **kwargs) -> None:
        _maybe_print(self._verbose_enabled(), *args, **kwargs)

    # ------------------------- loading -------------------------

    def _load_model_from_checkpoint(
        self,
        ckpt_cfg: ModelCheckpoint,
        cfg_overrides: Dict[str, Any] | None = None,
    ):
        """Load a model from checkpoint using the algorithm registry."""
        run_path = f"{ckpt_cfg.entity}/{ckpt_cfg.project}/{ckpt_cfg.run_id}"

        ckpt_path, cfg_dict = load_checkpoint(ckpt_cfg, self.device)
        cfg = OmegaConf.create(cfg_dict)
        self._log(OmegaConf.to_yaml(cfg))

        algo_cfg_node = cfg.algorithm
        if cfg_overrides:
            flow_decoder_ckpt = cfg_overrides.pop("flow_decoder_ckpt", None)
            for key, value in cfg_overrides.items():
                OmegaConf.update(algo_cfg_node, key, value, merge=True)
            if flow_decoder_ckpt is not None:
                cfg_overrides["flow_decoder_ckpt"] = flow_decoder_ckpt

        # When loading WAN, resolve flow_decoder_ckpt to a local path if provided
        algo_name_from_cfg = algo_cfg_node.get("name", None) or getattr(
            algo_cfg_node, "name", None
        )
        if (
            algo_name_from_cfg == "wan_t2v"
            and cfg_overrides
            and cfg_overrides.get("flow_decoder_ckpt")
        ):
            flow_ckpt_cfg = cfg_overrides["flow_decoder_ckpt"]
            flow_ckpt_path, _ = load_checkpoint(flow_ckpt_cfg, self.device)
            OmegaConf.update(
                cfg, "algorithm.flow_decoder.ckpt_path", str(flow_ckpt_path), merge=True
            )
            OmegaConf.update(cfg, "algorithm.flow_decoder.enabled", True, merge=True)
            algo_cfg_node = cfg.algorithm

        # Inference-only world models (e.g. WAN) are built via motion planner factory;
        # trainable planners (e.g. DFOT) use the algorithm registry.
        algo_name_str = algo_name_from_cfg or "?"
        algo = build_motion_planner(algo_name_str, algo_cfg_node, device=self.device)
        if algo is not None:
            algo_name = algo_name_str
        else:
            algo_cfg = resolve_algorithm_cfg(algo_cfg_node)
            algo = resolve_algorithm_instance(algo_cfg)
            algo_name = (
                algo_cfg.get("name", "?")
                if isinstance(algo_cfg, dict)
                else getattr(algo_cfg, "name", "?")
            )

        # Load to CPU to avoid OOM for large checkpoints (e.g. WAN 14B). Then load_state_dict and move to device.
        state_dict = cast(
            Dict[str, Any],
            torch.load(
                ckpt_path,
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )["state_dict"],
        )
        adapt_state_dict = getattr(algo, "_adapt_state_dict_for_compile", None)
        if callable(adapt_state_dict):
            state_dict = adapt_state_dict(state_dict)

        msg = cast(Any, algo).load_state_dict(
            cast(Dict[str, Any], state_dict), strict=False
        )
        if not (getattr(algo, "_is_fsdp", False)):
            algo = algo.to(self.device)

        setattr(
            algo,
            "_loaded_normalization_metadata",
            _extract_normalization_metadata(cfg_dict),
        )

        return algo, msg, run_path, algo_name

    def load_motion_planner(self):
        planner_cfg = self.cfg.motion_planner
        resolved_ckpt_path, config_dict, source = (
            _resolve_planner_checkpoint_and_config(
                planner_cfg,
                self.device,
                verbose=self._verbose_enabled(),
            )
        )

        cfg = OmegaConf.create(config_dict)
        algo_name_from_cfg = cfg.algorithm.get("name", None) or getattr(
            cfg.algorithm, "name", None
        )
        algo_cfg_node = cfg.algorithm
        planner_name = algo_name_from_cfg or "wan_t2v"
        is_wan = _is_wan_motion_planner(planner_name)

        if is_wan:
            _apply_wan_planner_overrides(
                cfg,
                planner_cfg,
                sample_steps=planner_cfg.diffusion_sampling_timesteps,
            )
            algo_cfg_node = cfg.algorithm

            wan_14b = planner_name == "wan_i2v" or (
                getattr(getattr(algo_cfg_node, "model", None), "dim", 0) == 5120
            )
            # if wan_14b and not (
            #     torch.distributed.is_available()
            #     and torch.distributed.is_initialized()
            #     and torch.distributed.get_world_size() > 1
            # ):
            #     raise RuntimeError(
            #         "14B WAN (wan_i2v) does not fit on a single GPU. FSDP was not used because this process "
            #         "is single-GPU (e.g. Jupyter notebook). Launch with torchrun to use FSDP: "
            #         "torchrun --nproc_per_node=4 python your_script.py  (or use a script that imports and runs "
            #         "the policy). For interactive notebook, use 1.3B (wan_t2v) or run the notebook from a "
            #         "torchrun worker (non-standard)."
            #     )

            resolved_config_path = _write_resolved_algorithm_config(algo_cfg_node)
            try:
                algo = cast(Any, build_motion_planner)(
                    planner_name,
                    algo_cfg_node,
                    device=self.device,
                    config_path=str(resolved_config_path),
                    ckpt_path=(
                        str(resolved_ckpt_path.resolve())
                        if resolved_ckpt_path is not None
                        else None
                    ),
                )
            finally:
                resolved_config_path.unlink(missing_ok=True)

            if algo is None:
                raise RuntimeError(
                    f"Failed to build WAN motion planner for algorithm '{planner_name}'."
                )
            algo_name = planner_name
            msg = "WanAllTrackerPipeline.from_config"
        else:
            OmegaConf.update(
                algo_cfg_node,
                "diffusion.sampling_timesteps",
                planner_cfg.diffusion_sampling_timesteps,
                merge=True,
            )
            algo = build_motion_planner(planner_name, algo_cfg_node, device=self.device)
            if algo is not None:
                algo_name = planner_name
            else:
                algo_cfg = resolve_algorithm_cfg(algo_cfg_node)
                algo = resolve_algorithm_instance(algo_cfg)
                algo_name = (
                    algo_cfg.get("name", "?")
                    if isinstance(algo_cfg, dict)
                    else getattr(algo_cfg, "name", "?")
                )

            if resolved_ckpt_path is not None:
                # Load to CPU to avoid OOM. Then load_state_dict and move model to device.
                state_dict = cast(
                    Dict[str, Any],
                    torch.load(
                        resolved_ckpt_path,
                        map_location="cpu",
                        weights_only=False,
                        mmap=True,
                    )["state_dict"],
                )
                adapt_state_dict = getattr(algo, "_adapt_state_dict_for_compile", None)
                if callable(adapt_state_dict):
                    state_dict = adapt_state_dict(state_dict)
                msg = cast(Any, algo).load_state_dict(
                    cast(Dict[str, Any], state_dict), strict=False
                )
            else:
                msg = "raw init (no checkpoint)"

            if not (getattr(algo, "_is_fsdp", False)):
                algo = algo.to(self.device)
        self.motion_planner = algo
        self._motion_planner_normalization_meta = _extract_normalization_metadata(
            config_dict
        )
        setattr(
            algo,
            "_loaded_normalization_metadata",
            self._motion_planner_normalization_meta,
        )

        if source == "wandb" and planner_cfg.ckpt is not None:
            run_path = f"{planner_cfg.ckpt.entity}/{planner_cfg.ckpt.project}/{planner_cfg.ckpt.run_id}"
            opt = getattr(planner_cfg.ckpt, "option", "latest")
            self._log(
                f"✅ Loaded motion planner from {run_path} (option={opt}, algo={algo_name}) – {msg}"
            )
        elif source == "local" and resolved_ckpt_path is not None:
            self._log(
                f"✅ Loaded motion planner from {resolved_ckpt_path} (algo={algo_name}) – {msg}"
            )
        else:
            self._log(f"✅ Motion planner raw init (algo={algo_name}) – {msg}")

    def load_dynamics_model(self):
        dyn_cfg = self.cfg.dynamics_model
        local_path = getattr(dyn_cfg, "ckpt_path", None)
        if local_path is not None:
            from pathlib import Path as _Path
            ckpt_path = _Path(local_path)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Dynamics checkpoint not found: {ckpt_path}")
            for _name in ("config.yaml", "config.yml", "run_config.yaml"):
                sidecar = ckpt_path.parent / _name
                if sidecar.exists():
                    break
            else:
                raise FileNotFoundError(
                    f"No config.yaml / run_config.yaml found next to {ckpt_path}. "
                    "Place a run_config.yaml (wandb run.config dump) alongside the ckpt."
                )
            cfg_dict = _load_algorithm_config_from_path(sidecar)
            algo_cfg = resolve_algorithm_cfg(OmegaConf.create(cfg_dict).algorithm)
            algo = resolve_algorithm_instance(algo_cfg)
            state_dict = cast(Dict[str, Any], torch.load(
                ckpt_path, map_location="cpu", weights_only=False, mmap=True
            )["state_dict"])
            # Adapt torch.compile-prefixed keys (model.* <-> model._orig_mod.*) exactly as
            # the wandb checkpoint path does — a no-op when algo.cfg.compile is False. Without
            # this, a ckpt trained with compile=true silently fails to load (all keys unmatched
            # under strict=False) and the IDM runs at random init. Assert a clean load so this
            # can never fail silently again.
            if hasattr(algo, "_adapt_state_dict_for_compile"):
                state_dict = algo._adapt_state_dict_for_compile(state_dict)
            missing, unexpected = algo.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    f"local dynamics load mismatch for {ckpt_path.name}: "
                    f"missing={len(missing)} unexpected={len(unexpected)} "
                    f"(first missing: {list(missing)[:3]}, first unexpected: {list(unexpected)[:3]})"
                )
            algo = algo.to(self.device)
            setattr(algo, "_loaded_normalization_metadata",
                    _extract_normalization_metadata(cfg_dict))
            self.dynamics_model = algo
            msg = f"local:{ckpt_path.name}"
            run_path = str(ckpt_path)
            algo_name = getattr(algo_cfg, "name", "?") if not isinstance(algo_cfg, dict) else algo_cfg.get("name", "?")
        else:
            self.dynamics_model, msg, run_path, algo_name = (
                self._load_model_from_checkpoint(dyn_cfg.ckpt)
            )
        self._dynamics_normalization_meta = getattr(
            self.dynamics_model,
            "_loaded_normalization_metadata",
            {},
        )
        self.robot_name = getattr(self.dynamics_model.cfg, "robot_name", "unknown")
        option = getattr(getattr(dyn_cfg, "ckpt", None), "option", "local")
        self._log(
            f"✅ Loaded dynamics model from {run_path} "
            f"(option={option}, algo={algo_name}) – {msg}"
        )

    def _check_normalization_compatibility(self) -> None:
        mode = getattr(self.cfg, "normalization_compatibility_mode", "warn")
        if mode == "off":
            return

        planner_meta = self._motion_planner_normalization_meta or {}
        dynamics_meta = self._dynamics_normalization_meta or {}
        if not planner_meta or not dynamics_meta:
            return

        compare_keys = [
            "oflow_scale",
            "oflow_std",
            "oflow_abs_scale",
            "flow_normalization_space",
            "flow_normalization_mode",
            "action_normalization_mode",
            "action_pre_scale",
            "action_abs_scale",
        ]
        mismatches: list[str] = []
        for key in compare_keys:
            left = planner_meta.get(key)
            right = dynamics_meta.get(key)
            if left is None or right is None:
                continue
            if left != right:
                mismatches.append(f"{key}: planner={left} dynamics={right}")

        if not mismatches:
            return

        message = (
            "Motion planner and dynamics model advertise different normalization metadata. "
            "Runtime behavior is unchanged and still uses legacy scaled units by default, "
            "but mixed-unit models can silently distort action magnitudes.\n"
            f"planner: {_summarize_normalization_metadata(planner_meta)}\n"
            f"dynamics: {_summarize_normalization_metadata(dynamics_meta)}\n"
            f"mismatches: {', '.join(mismatches)}"
        )
        if mode == "error":
            raise RuntimeError(message)
        warnings.warn(message, stacklevel=2)

    def get_dynamics_normalization_metadata(self) -> dict[str, Any]:
        return dict(self._dynamics_normalization_meta)

    def get_wire_metadata(self) -> dict[str, Any]:
        """Handshake metadata for the deploy protocol (gripperless base policy, e.g. PushT/DROID).

        Mirrors ``MotionPolicyGripper.get_wire_metadata`` minus the gripper gating. The base
        policy emits the controller's solved action chunk as-is (no env-metric re-scaling /
        gating), so ``actions_already_metric=False``: any embodiment-specific unit conversion
        (e.g. the PushT runner's ``actions_vel_scale``) is the client/runner's job, not the
        policy's. ``gripper_dim_index=-1`` (no gripper channel). Used by the adapter factory's
        ``_build_server_config`` for the on-the-wire handshake.
        """
        meta = self.get_dynamics_normalization_metadata()
        abs_scale = [float(x) for x in meta.get("action_abs_scale", []) or []]
        return {
            "action_mode": str(meta.get("action_mode", "velocity")),
            "action_abs_scale": abs_scale,
            "dim_u": len(abs_scale) or int(getattr(self.cfg, "action_dim", 0) or 2),
            "gripper_dim_index": -1,
            "actions_already_metric": False,
            "context_frames": int(self.context_frames),
            "action_chunk_horizon": int(self.cfg.action_chunk_horizon),
            "n_action_steps": int(self.cfg.n_action_steps),
            "current_tracker_backend": str(
                getattr(self.cfg.motion_planner, "tracker_backend", "none")
            ),
        }

    def denormalize_dynamics_action(self, du_model: Tensor) -> Tensor:
        return denormalize_action(
            du_model,
            du_scale=float(self._dynamics_normalization_meta.get("du_scale", 1.0)),
            action_mean=self._dynamics_normalization_meta.get("action_mean"),
            action_std=self._dynamics_normalization_meta.get("action_std"),
            action_min=self._dynamics_normalization_meta.get("action_min"),
            action_max=self._dynamics_normalization_meta.get("action_max"),
            action_abs_scale=self._dynamics_normalization_meta.get("action_abs_scale"),
        )

    def denormalize_dynamics_jacobian(self, jacobian_scaled: Tensor) -> Tensor:
        cmd_dim = int(jacobian_scaled.shape[-4]) if jacobian_scaled.ndim >= 5 else 0
        return denormalize_jacobian_tensor(
            jacobian_scaled,
            self._dynamics_normalization_meta,
            cmd_dim=cmd_dim,
        )

    # ------------------------- helpers -------------------------

    @staticmethod
    def _is_fsdp_worker_rank() -> bool:
        """True when running as a non-zero rank in an FSDP process group."""
        if (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
        ):
            return False
        return torch.distributed.get_rank() > 0

    @property
    def image_size_motion_planner(self) -> Tuple[int, int]:
        x_shape = getattr(self.motion_planner.cfg, "x_shape", None)
        if x_shape is not None and len(x_shape) >= 3:
            return (int(x_shape[-2]), int(x_shape[-1]))
        C = self.motion_planner.cfg.x_shape[-1]
        return (C, C)

    @property
    def image_size_dynamics_model(self) -> Tuple[int, int]:
        if self.dynamics_model is None:
            return (128, 128)
        sz = self.dynamics_model.cfg.image_size
        if isinstance(sz, (list, tuple)):
            return (int(sz[0]), int(sz[1]))
        return (int(sz), int(sz))

    @staticmethod
    def _resolve_view_widths(
        total_width: int,
        view_widths: list[int] | None = None,
        view_count: int | None = None,
    ) -> list[int]:
        if view_widths is not None and len(view_widths) > 1:
            base = np.asarray(view_widths, dtype=np.float64)
        elif view_count is not None and view_count > 1:
            base = np.ones(view_count, dtype=np.float64)
        else:
            return [int(total_width)]

        base = np.maximum(base, 1.0)
        scaled = np.ones(len(base), dtype=np.int64)
        remaining = int(total_width) - int(scaled.sum())
        if remaining > 0:
            weights = base / base.sum()
            raw_extra = weights * remaining
            extra = np.floor(raw_extra).astype(np.int64)
            scaled += extra
            leftover = remaining - int(extra.sum())
            if leftover > 0:
                order = np.argsort(-(raw_extra - extra))
                for idx in order[:leftover]:
                    scaled[idx] += 1
        scaled[-1] += int(total_width) - int(scaled.sum())
        if np.any(scaled <= 0):
            raise ValueError(
                f"Invalid multiview width allocation: total_width={total_width}, scaled={scaled.tolist()}"
            )
        return [int(v) for v in scaled.tolist()]

    @staticmethod
    def _split_along_width(x: Tensor, widths: list[int]) -> list[Tensor]:
        if sum(widths) != int(x.shape[-1]):
            raise ValueError(
                f"Width split mismatch: sum(widths)={sum(widths)} tensor_width={int(x.shape[-1])}"
            )
        splits = []
        start = 0
        for width in widths:
            end = start + int(width)
            splits.append(x[..., start:end])
            start = end
        return splits

    @staticmethod
    def _obs_view_count(obs: PolicyObservation) -> int:
        if obs.view_widths is not None and len(obs.view_widths) > 0:
            return len(obs.view_widths)
        if obs.view_keys is not None and len(obs.view_keys) > 0:
            return len(obs.view_keys)
        return 1

    def _resolve_control_view_indices(self, obs: PolicyObservation) -> list[int]:
        num_views = self._obs_view_count(obs)
        if self.cfg.control_view_keys is None or num_views <= 1:
            return list(range(num_views))
        if obs.view_keys is None or len(obs.view_keys) == 0:
            raise ValueError(
                "control_view_keys requires PolicyObservation.view_keys for multiview control"
            )

        resolved: list[int] = []
        missing: list[str] = []
        for key in self.cfg.control_view_keys:
            if key in obs.view_keys:
                resolved.append(obs.view_keys.index(key))
            else:
                missing.append(key)
        if missing:
            raise ValueError(
                f"Unknown control_view_keys {missing}; available views: {obs.view_keys}"
            )
        if not resolved:
            raise ValueError("control_view_keys resolved to an empty control view set")
        return sorted(set(resolved))

    @staticmethod
    def _view_widths_for_layout(
        total_width: int,
        obs: PolicyObservation,
    ) -> list[int]:
        return MotionPolicy._resolve_view_widths(
            total_width=total_width,
            view_widths=obs.view_widths,
            view_count=MotionPolicy._obs_view_count(obs),
        )

    @staticmethod
    def _pixel_mask_for_view_indices(
        height: int,
        view_widths: list[int],
        selected_view_indices: list[int],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        width_mask = torch.zeros(sum(view_widths), device=device, dtype=dtype)
        start = 0
        selected = set(int(idx) for idx in selected_view_indices)
        for view_index, view_width in enumerate(view_widths):
            end = start + int(view_width)
            if view_index in selected:
                width_mask[start:end] = 1.0
            start = end
        return width_mask.unsqueeze(0).expand(int(height), -1).reshape(-1)

    def _control_motion_mask(
        self,
        obs: PolicyObservation,
        height: int,
        view_widths: list[int],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        pixel_mask = self._pixel_mask_for_view_indices(
            height,
            view_widths,
            self._resolve_control_view_indices(obs),
            device=device,
            dtype=dtype,
        )
        return pixel_mask.repeat_interleave(2)

    def _control_track_mask(
        self,
        obs: PolicyObservation,
        x_coords: Tensor,
        view_widths: list[int],
    ) -> Tensor:
        if len(view_widths) <= 1:
            return torch.ones_like(x_coords, dtype=torch.float32)
        mask = torch.zeros_like(x_coords, dtype=torch.float32)
        start = 0
        selected = set(self._resolve_control_view_indices(obs))
        for view_index, view_width in enumerate(view_widths):
            end = start + int(view_width)
            if view_index == len(view_widths) - 1:
                view_mask = (x_coords >= float(start)) & (x_coords <= float(end))
            else:
                view_mask = (x_coords >= float(start)) & (x_coords < float(end))
            if view_index in selected:
                mask = torch.where(view_mask, torch.ones_like(mask), mask)
            start = end
        return mask

    def _source_view_widths_for_obs(
        self,
        obs: PolicyObservation,
        total_width: int,
    ) -> list[int]:
        return self._resolve_view_widths(
            total_width=total_width,
            view_widths=obs.view_widths,
            view_count=self._obs_view_count(obs),
        )

    def _planner_view_widths(self, obs: PolicyObservation, canvas_width: int) -> list[int]:
        """True per-view widths handed to the planner for tracking the dream output.

        When the client declares ``view_widths`` we pass them LITERALLY — their sum may be smaller
        than the model's padded canvas (an omni model concatenates the real views then black-pads up
        to its canvas width). The planner crops that pad before tracking, so the tracker and the
        Jacobian separate views by the real geometry instead of stretching the widths across the pad
        (the [288,288]-instead-of-[128,128] bug). Falls back to the resolved split when view_widths
        is unavailable (single-view / unknown layout)."""
        if obs.view_widths is not None and len(obs.view_widths) > 1:
            return [int(w) for w in obs.view_widths]
        return self._source_view_widths_for_obs(obs, canvas_width)

    def _dynamics_view_widths(self, num_views: int) -> list[int]:
        return [int(self.image_size_dynamics_model[1])] * max(int(num_views), 1)

    def _concat_dynamics_size(self, num_views: int) -> Tuple[int, int]:
        dyn_h, dyn_w = self.image_size_dynamics_model
        return (int(dyn_h), int(dyn_w) * max(int(num_views), 1))

    @staticmethod
    def _map_x_between_view_layouts(
        x: Tensor,
        source_view_widths: list[int],
        target_view_widths: list[int],
    ) -> Tensor:
        if len(source_view_widths) != len(target_view_widths):
            raise ValueError(
                "source_view_widths and target_view_widths must have the same length"
            )

        mapped = torch.zeros_like(x)
        source_offsets = np.cumsum([0, *source_view_widths[:-1]])
        target_offsets = np.cumsum([0, *target_view_widths[:-1]])
        for idx, (src_off, src_w, tgt_off, tgt_w) in enumerate(
            zip(source_offsets, source_view_widths, target_offsets, target_view_widths)
        ):
            upper = src_off + src_w
            if idx == len(source_view_widths) - 1:
                mask = (x >= float(src_off)) & (x <= float(upper))
            else:
                mask = (x >= float(src_off)) & (x < float(upper))
            local_x = (x - float(src_off)).clamp(0.0, max(float(src_w) - 1.0, 0.0))
            scale_x = float(tgt_w) / max(float(src_w), 1.0)
            mapped_view = float(tgt_off) + local_x * scale_x
            mapped = torch.where(mask, mapped_view, mapped)
        return mapped

    def _resize_multiview_rgb(
        self,
        rgb: Tensor,
        source_view_widths: list[int],
    ) -> Tuple[Tensor, list[int]]:
        target_h = int(self.image_size_dynamics_model[0])
        target_view_widths = self._dynamics_view_widths(len(source_view_widths))
        rgb_views = self._split_along_width(rgb, source_view_widths)
        resized_views = [
            torch.nn.functional.interpolate(
                view.to(self.device),
                size=(target_h, target_view_widths[idx]),
                mode="bilinear",
            )
            for idx, view in enumerate(rgb_views)
        ]
        return torch.cat(resized_views, dim=-1), target_view_widths

    def _resize_multiview_flow(
        self,
        flow: Tensor,
        source_view_widths: list[int],
        target_view_widths: list[int],
    ) -> Tensor:
        target_h = int(self.image_size_dynamics_model[0])
        flow_views = self._split_along_width(flow, source_view_widths)
        resized_views = [
            resize_flow(flow_view, target_h, target_view_widths[idx])
            for idx, flow_view in enumerate(flow_views)
        ]
        return torch.cat(resized_views, dim=-1)

    def _resolve_view_ids(
        self, view_keys: list[str] | None, num_views: int
    ) -> Tensor | None:
        """Map client-side `view_keys` to training-time view embedding ids.

        Default: None (model-side default of all-zeros applies, i.e. view 0).
        Subclasses with view-embedding-aware dynamics (e.g. DROID) must
        override this so each camera gets the embedding it was trained under.
        """
        return None

    def _compute_jacobian_from_rgb_tensor(
        self,
        rgb: Tensor,
        source_view_widths: list[int],
        view_keys: list[str] | None = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tuple[int, int], list[int]]:
        if self.dynamics_model is None:
            raise RuntimeError(
                "dynamics_model is None (FSDP worker rank); compute_jacobian not available"
            )

        resized_rgb, target_view_widths = self._resize_multiview_rgb(
            rgb, source_view_widths
        )
        num_views = len(target_view_widths)
        rgb_views = self._split_along_width(resized_rgb, target_view_widths)
        rgb_batch = rearrange(torch.stack(rgb_views, dim=1), "b v c h w -> (b v) c h w")

        # Per-view embedding ids must be built in the same [b0v0, b0v1, ..., b1v0, ...]
        # order the (b v) -> (b v) rearrange above produces.
        per_view_ids = self._resolve_view_ids(view_keys, num_views)
        if per_view_ids is not None:
            view_ids = per_view_ids.to(rgb_batch.device).repeat(rgb.shape[0])
        else:
            view_ids = None
        jacobian = self.dynamics_model.model.compute_jacobian(
            JacobianInputObservation(rgb=rgb_batch, view_ids=view_ids)
        )
        jacobian = jacobian[:, :, :2]
        jacobian = self._mask_jacobian(
            jacobian,
            view_keys=view_keys,
            num_views=num_views,
        )
        jacobian = rearrange(
            jacobian,
            "(b v) c s h w -> b v c s h w",
            b=rgb.shape[0],
            v=num_views,
        )
        jacobian = rearrange(jacobian, "b v c s h w -> b c s h (v w)")
        jacobian_flat = rearrange(jacobian, "b c s h w -> b (h w s) c")
        jacobian_pixel = rearrange(jacobian, "b c s h w -> b (h w) s c")
        target_size = (int(jacobian.shape[-2]), int(jacobian.shape[-1]))
        return (
            resized_rgb,
            jacobian_flat,
            jacobian_pixel,
            target_size,
            target_view_widths,
        )

    @property
    def context_frames(self) -> int:
        if self.cfg.context_frames is not None:
            return max(int(self.cfg.context_frames), 1)
        n = getattr(self.motion_planner, "n_context_frames", None) or getattr(
            self.motion_planner.cfg, "context_frames", 1
        )
        return max(int(n), 1)

    @property
    def is_joint(self) -> bool:
        cfg = self.motion_planner.cfg
        name = getattr(cfg, "name", None) or (
            cfg.get("name", "") if hasattr(cfg, "get") else ""
        )
        return name in (
            "dfot_motion_policy_joint",
            "wan_t2v",
            "wan_i2v",
            "wan_ar_df",
            "wan_ar_tf",
        )

    def reset(self):
        n_ctx = self.context_frames
        self._queues = {"observation.images": deque(maxlen=n_ctx)}
        self.controller.reset()
        self._action_queue.clear()
        self._vis_queue.clear()
        self._next_dream_index = 0
        self._current_dream_index = 0
        self._worker_step_counter = 0
        self.reset_adaptive_controller_state()
        # Start a fresh dump run; do NOT clear the writer queue — in-flight items
        # already captured their (old-run) out_path and should finish flushing.
        self._reset_debug_dump_run_state_locked()

    def configure_runtime(
        self,
        *,
        motion_plan_scale: float | None = None,
        action_chunk_horizon: int | None = None,
        n_action_steps: int | None = None,
        verbose: bool | None = None,
        jacobian_vis_abs_threshold: float | None = None,
        debug_dump_enabled: bool | None = None,
        debug_dump_dir: str | None = None,
        debug_dump_model_name: str | None = None,
        debug_dump_task_name: str | None = None,
        debug_dump_new_run: bool | None = None,
        debug_dump_max_chunks: int | None = None,
        debug_dump_max_bytes: int | None = None,
        debug_dump_min_free_bytes: int | None = None,
        lang_guidance: float | None = None,
        hist_guidance: float | None = None,
        sample_steps: int | None = None,
        context_frames: int | None = None,
        control_view_keys: Any = UNSET,
        text_conditioning: Any = UNSET,
    ) -> dict[str, Any]:
        if control_view_keys is not UNSET:
            # restrict the jacobian/IDM solve to a subset of views (pixel mask) while the
            # planner keeps dreaming over all of them — e.g. ["hand"] for wrist-only control.
            # None restores all-view control. Read per-infer in _resolve_control_view_indices,
            # which validates names against obs.view_keys at solve time.
            if control_view_keys is not None:
                if isinstance(control_view_keys, str):
                    control_view_keys = [control_view_keys]
                control_view_keys = [str(k) for k in control_view_keys]
                if not control_view_keys:
                    raise ValueError("control_view_keys must be None or a non-empty list")
            self.cfg.control_view_keys = control_view_keys
        if motion_plan_scale is not None:
            self.cfg.motion_plan_scale = float(motion_plan_scale)
        if action_chunk_horizon is not None:
            self.cfg.action_chunk_horizon = int(action_chunk_horizon)
        if n_action_steps is not None:
            self.cfg.n_action_steps = int(n_action_steps)
        if context_frames is not None:
            # Live context-window cap. context_frames is a pure runtime SLICE bound:
            # the property (context_frames, above) feeds `cond[:, -n_ctx:]` in the WAN
            # branch, which trims DOWN to the largest valid k*stride+1 and lets the
            # planner handle variable context length -- so no model reload is needed.
            # The client must still supply >= this many real frames before the first
            # plan or the slice just returns fewer; clamp >=1.
            self.cfg.context_frames = max(1, int(context_frames))
        if verbose is not None:
            self.cfg.verbose = bool(verbose)
        if jacobian_vis_abs_threshold is not None:
            self.cfg.jacobian_vis_abs_threshold = max(
                0.0,
                float(jacobian_vis_abs_threshold),
            )
        if debug_dump_dir is not None:
            self.cfg.debug_dump_dir = str(debug_dump_dir)
            self._reset_debug_dump_run_state_locked()
        if debug_dump_model_name is not None:
            new_model_name = str(debug_dump_model_name)
            if new_model_name != getattr(self.cfg, "debug_dump_model_name", None):
                self.cfg.debug_dump_model_name = new_model_name
                self._reset_debug_dump_run_state_locked()
        if debug_dump_task_name is not None:
            new_task_name = str(debug_dump_task_name)
            if new_task_name != getattr(self.cfg, "debug_dump_task_name", None):
                self.cfg.debug_dump_task_name = new_task_name
                self._reset_debug_dump_run_state_locked()
        if debug_dump_max_chunks is not None:
            self.cfg.debug_dump_max_chunks = max(0, int(debug_dump_max_chunks))
        if debug_dump_max_bytes is not None:
            self.cfg.debug_dump_max_bytes = max(0, int(debug_dump_max_bytes))
        if debug_dump_min_free_bytes is not None:
            self.cfg.debug_dump_min_free_bytes = max(
                0,
                int(debug_dump_min_free_bytes),
            )
        if debug_dump_enabled is not None:
            self.cfg.debug_dump_enabled = bool(debug_dump_enabled)
            if not self.cfg.debug_dump_enabled:
                self._reset_debug_dump_run_state_locked()
        if debug_dump_new_run:
            self._reset_debug_dump_run_state_locked()
        # WAN CFG-style guidance scales live as plain attributes on the inner
        # WAN algo and are read every sampling pass, so we can mutate them
        # in-place. Wan{All,AR}TrackerPipeline wraps the algo at `_model`;
        # plain WanTextToVideo (no pipeline) holds it directly.
        guidance_target = getattr(
            self.motion_planner, "_model", self.motion_planner
        )
        if lang_guidance is not None and hasattr(guidance_target, "lang_guidance"):
            guidance_target.lang_guidance = max(0.0, float(lang_guidance))
        if hist_guidance is not None and hasattr(guidance_target, "hist_guidance"):
            guidance_target.hist_guidance = max(0.0, float(hist_guidance))
        if sample_steps is not None and hasattr(guidance_target, "sample_steps"):
            # denoising steps are read per generate call -> live latency/quality knob
            guidance_target.sample_steps = max(1, int(sample_steps))
        if text_conditioning is not UNSET:
            if text_conditioning is not None and not isinstance(
                text_conditioning, (str, list)
            ):
                raise TypeError(
                    "text_conditioning must be None, a prompt string, or a list of prompts"
                )
            self.cfg.text_conditioning = text_conditioning
            if debug_dump_task_name is None:
                inferred_task = self._debug_dump_label_from_value(
                    text_conditioning,
                    fallback="task_unset",
                )
                if inferred_task != getattr(self.cfg, "debug_dump_task_name", None):
                    self.cfg.debug_dump_task_name = inferred_task
                    self._reset_debug_dump_run_state_locked()
        self.cfg.n_action_steps = min(
            int(self.cfg.n_action_steps),
            int(self.cfg.action_chunk_horizon),
        )
        return {
            "motion_plan_scale": float(self.cfg.motion_plan_scale),
            "control_view_keys": (
                list(self.cfg.control_view_keys)
                if self.cfg.control_view_keys is not None
                else None
            ),
            "action_chunk_horizon": int(self.cfg.action_chunk_horizon),
            "n_action_steps": int(self.cfg.n_action_steps),
            "verbose": bool(self.cfg.verbose),
            "jacobian_vis_abs_threshold": float(
                self.cfg.jacobian_vis_abs_threshold
            ),
            "debug_dump_enabled": bool(self.cfg.debug_dump_enabled),
            "debug_dump_dir": str(self.cfg.debug_dump_dir),
            "debug_dump_model_name": getattr(
                self.cfg,
                "debug_dump_model_name",
                None,
            ),
            "debug_dump_task_name": getattr(
                self.cfg,
                "debug_dump_task_name",
                None,
            ),
            "debug_dump_run_dir": (
                None if self._debug_dump_run_dir is None else str(self._debug_dump_run_dir)
            ),
            "debug_dump_max_chunks": int(self.cfg.debug_dump_max_chunks),
            "debug_dump_max_bytes": int(self.cfg.debug_dump_max_bytes),
            "debug_dump_min_free_bytes": int(self.cfg.debug_dump_min_free_bytes),
            "sample_steps": int(getattr(guidance_target, "sample_steps", 0) or 0),
            "context_frames": int(self.context_frames),
            "lang_guidance": float(
                getattr(guidance_target, "lang_guidance", 0.0) or 0.0
            ),
            "hist_guidance": float(
                getattr(guidance_target, "hist_guidance", 0.0) or 0.0
            ),
            "text_conditioning": self.cfg.text_conditioning,
        }

    def check_alive(self) -> None:
        """Notebook compatibility hook; local policies are always alive."""
        return None

    def warmup_obs(self, obs: PolicyObservation) -> None:
        """Feed one observation into the context queue without running the planner.

        Called by the server when a client sends ``run_policy=False``, so the
        queue gets the observation but no plan/action is computed.
        """
        im = self._resize(obs.rgb, self.image_size_motion_planner)
        self._queues["observation.images"].append(im)

    @torch.no_grad()
    def debug_plan(self, obs: PolicyObservation) -> dict[str, Any]:
        """Return a small summary of the next joint plan without mutating queues."""
        im = self._resize(obs.rgb, self.image_size_motion_planner)
        debug_queues = {
            key: deque(value, maxlen=value.maxlen)
            for key, value in self._queues.items()
        }
        debug_queues = populate_queues(debug_queues, {"observation.images": im})
        cond = torch.stack(list(debug_queues["observation.images"]), dim=1)

        plan_output = (
            self._compute_plan_joint(obs, cond)
            if self.is_joint
            else self._compute_plan_flow(cond)
        )
        if not self.is_joint:
            flow = cast(Tensor, plan_output)
            return {
                "mode": "flow",
                "flow_shape": tuple(flow.shape),
            }

        xs, record = self._split_joint_plan_output(plan_output)
        n_ctx = self.context_frames
        rgb_future = xs[:, n_ctx:, :3]
        flow_future = xs[:, n_ctx:, 3:5]
        motion_tracks = None if record is None else record.get("motion_tracks")
        track_source_rgb = None if record is None else record.get("track_source_rgb")
        summary: dict[str, Any] = {
            "mode": "joint",
            "context_frames": int(n_ctx),
            "xs_shape": tuple(xs.shape),
            "rgb_future_shape": tuple(rgb_future.shape),
            "flow_future_shape": tuple(flow_future.shape),
            "future_pixel_frames": int(
                getattr(self.motion_planner, "future_pixel_frames", rgb_future.shape[1])
            ),
            "required_pixel_frames": int(
                getattr(self.motion_planner, "required_pixel_frames", n_ctx)
            ),
            "track_steps": (
                int(motion_tracks["disp"].shape[1]) if motion_tracks is not None else 0
            ),
            "track_source_steps": (
                int(track_source_rgb.shape[1])
                if isinstance(track_source_rgb, torch.Tensor)
                else 0
            ),
            "has_alltracker_vis": bool(
                record is not None and "alltracker_vis" in record
            ),
            "view_keys": list(obs.view_keys) if obs.view_keys is not None else None,
            "view_widths": (
                list(obs.view_widths) if obs.view_widths is not None else None
            ),
            "control_view_keys": (
                list(self.cfg.control_view_keys)
                if self.cfg.control_view_keys is not None
                else None
            ),
        }
        if isinstance(record, dict):
            if "planner_sampling_path" in record:
                summary["planner_sampling_path"] = record["planner_sampling_path"]
            if "text_conditioning" in record:
                summary["text_conditioning"] = record["text_conditioning"]
            if "text_conditioning_enabled" in record:
                summary["text_conditioning_enabled"] = bool(
                    record["text_conditioning_enabled"]
                )
        if motion_tracks is not None:
            valid = motion_tracks["valid"].float()
            summary["track_valid_mean"] = float(valid.mean().item())
            summary["track_valid_fraction"] = float((valid > 0.5).float().mean().item())
            summary["track_disp_stats"] = self._summarize_displacements(
                motion_tracks["disp"],
                valid,
            )
            if isinstance(motion_tracks.get("meta"), dict):
                summary["track_meta"] = dict(motion_tracks["meta"])
        if isinstance(track_source_rgb, torch.Tensor) and track_source_rgb.shape[1] > 0:
            context_last = cond[:, -1:].detach().cpu()
            boundary_mae = (
                (track_source_rgb[:, :1].detach().cpu() - context_last).abs().mean()
            )
            summary["boundary_source_mae"] = float(boundary_mae.item())
        return summary

    # ------------------------- core primitives -------------------------

    def _debug_joint_planner_multiview_metadata(
        self,
        *,
        phase: str,
        obs: PolicyObservation,
        planner_view_widths: list[int],
        context_rgb_shape: tuple[int, ...],
        record: dict[str, Any] | None = None,
    ) -> None:
        """Optional subclass hook for debugging multiview planner metadata."""
        return None

    def compute_plan(self, obs: PolicyObservation) -> Any:
        cond = self._enqueue_observation_images(obs)
        if self.is_joint:
            return self._compute_plan_joint(obs, cond)
        return self._compute_plan_flow(cond)

    def _compute_plan_flow(self, cond: Tensor) -> Tensor:
        if hasattr(self.motion_planner, "_pad_to_max_tokens"):
            cond = self.motion_planner._pad_to_max_tokens(cond)
        xs, _ = self.motion_planner._sample_sequence(
            batch_size=cond.shape[0],
            conditions=cond,
            x_shape=(2, *self.image_size_motion_planner),
        )
        xs = self.motion_planner._unnormalize_x(xs)
        return xs[:, -1] * self.cfg.motion_plan_scale

    def _compute_plan_joint(
        self,
        obs: PolicyObservation,
        cond: Tensor,
    ) -> Tuple[Tensor, dict[str, Any] | None]:
        n_ctx = self.context_frames
        if hasattr(self.motion_planner, "generate_policy_chunk"):
            # Pass the real queue through. When shorter than n_ctx the slice
            # just returns all available frames; the planner handles variable
            # context length.
            context_rgb = cond[:, -n_ctx:]
            # WAN's VAE has temporal stride S (typically 4), so only pixel
            # counts of the form k*S+1 encode cleanly into integer latent
            # frames. Trim DOWN to the largest valid k*S+1 by dropping the
            # OLDEST frames -- NEVER front-pad by duplicating the first
            # frame, which tells the model that the early timesteps were
            # static and silences motion at session start (especially bad
            # for short-context recipes like E4 with M=1 trained on a
            # single context latent).
            vae_stride_attr = getattr(
                getattr(self.motion_planner, "_model", None),
                "vae_stride",
                None,
            )
            wan_temporal_stride = (
                int(vae_stride_attr[0]) if vae_stride_attr is not None else 4
            )
            T_ctx = context_rgb.shape[1]
            k_max = max(0, (T_ctx - 1) // wan_temporal_stride)
            valid_T = k_max * wan_temporal_stride + 1
            if valid_T < T_ctx:
                context_rgb = context_rgb[:, -valid_T:]
            horizon = self.cfg.action_chunk_horizon
            planner_view_widths = self._planner_view_widths(
                obs,
                int(context_rgb.shape[-1]),
            )
            text_cond = self._planner_text_conditioning()
            self._debug_joint_planner_multiview_metadata(
                phase="before_generate",
                obs=obs,
                planner_view_widths=planner_view_widths,
                context_rgb_shape=tuple(context_rgb.shape),
            )

            xs, record = self.motion_planner.generate_policy_chunk(
                context_rgb=context_rgb,
                horizon=horizon,
                include_boundary_step=True,
                view_keys=obs.view_keys,
                view_widths=planner_view_widths,
                text=text_cond,
            )
            self._debug_joint_planner_multiview_metadata(
                phase="after_generate",
                obs=obs,
                planner_view_widths=planner_view_widths,
                context_rgb_shape=tuple(context_rgb.shape),
                record=record,
            )

            return xs.to(cond.device), record

        # n_ctx_tokens = getattr(self.motion_planner, "n_context_tokens", n_ctx)
        warnings.warn("Using context frames from motion policy", UserWarning)

        H, W = self.image_size_motion_planner
        # Pass the real queue through. Short queues return all available frames.
        context_rgb = cond[:, -n_ctx:]
        # Use the actual (possibly-short) context length, not the target n_ctx.
        n_ctx_tokens = int(context_rgb.shape[1])

        context_flow = (
            torch.zeros(
                cond.shape[0],
                context_rgb.shape[1],
                2,
                H,
                W,
                device=cond.device,
                dtype=cond.dtype,
            )
            * 0.00
        )

        context = torch.cat([context_rgb, context_flow], dim=2)

        # max_tokens = getattr(self.motion_planner, "max_tokens", 64)
        # horizon = min(self.cfg.action_chunk_horizon, max(1, max_tokens - n_ctx_tokens))

        horizon = self.cfg.action_chunk_horizon

        # print(
        #     "self.cfg.action_chunk_horizon:",
        #     self.cfg.action_chunk_horizon,
        #     "n_ctx_tokens:",
        #     n_ctx_tokens,
        # )

        length = n_ctx_tokens + horizon

        context_mask = torch.zeros(
            cond.shape[0], length, dtype=torch.long, device=cond.device
        )
        context_mask[:, :n_ctx_tokens] = 1
        pad_len = length - n_ctx_tokens
        pad = torch.randn(
            cond.shape[0], pad_len, 5, H, W, device=cond.device, dtype=cond.dtype
        )
        full_context = torch.cat(
            [self.motion_planner._normalize_x(context), pad], dim=1
        )

        if VERBOSE_POLICY_DEBUG:
            print(
                f"              n_ctx_tokens: {n_ctx_tokens},\n"
                f"              horizon: {horizon},\n"
                f"              length: {length},\n"
                f"              context.shape: {context.shape},\n"
                f"              full_context.shape: {full_context.shape},\n"
                f"              pad.shape: {pad.shape},\n"
            )

        # Sliding-window path: when the desired length exceeds the model's
        # max_tokens (positional-embedding limit), generate the future in
        # chunks, sliding the context forward each time. Needed e.g. for
        # action_chunk_horizon > max_tokens - n_ctx_tokens.
        max_tokens = int(getattr(self.motion_planner, "max_tokens", length))
        if length > max_tokens:
            normalized_ctx = self.motion_planner._normalize_x(context)  # (B, n_ctx, 5, H, W)
            chunk_horizon = max(1, max_tokens - n_ctx_tokens)
            futures: list[Tensor] = []
            curr_ctx = normalized_ctx  # (B, n_ctx, 5, H, W)
            B = cond.shape[0]
            generated = 0
            while generated < horizon:
                cur_n = int(curr_ctx.shape[1])
                cur_pad_len = max_tokens - cur_n
                cur_pad = torch.randn(
                    B, cur_pad_len, 5, H, W,
                    device=cond.device, dtype=cond.dtype,
                )
                cur_full = torch.cat([curr_ctx, cur_pad], dim=1)
                cur_mask = torch.zeros(
                    B, max_tokens, dtype=torch.long, device=cond.device
                )
                cur_mask[:, :cur_n] = 1
                chunk_xs, _ = self.motion_planner._sample_sequence(
                    batch_size=B,
                    length=max_tokens,
                    context=cur_full,
                    context_mask=cur_mask,
                )
                # First cur_n tokens are the context we passed in; rest are new.
                new_futures = chunk_xs[:, cur_n:]  # (B, max_tokens-cur_n, 5, H, W)
                futures.append(new_futures)
                generated += int(new_futures.shape[1])
                # Slide context: keep the most recent n_ctx_tokens tokens from
                # the chunk to act as context for the next pass.
                curr_ctx = chunk_xs[:, -n_ctx_tokens:]
            all_futures = torch.cat(futures, dim=1)[:, :horizon]
            xs_norm = torch.cat([normalized_ctx, all_futures], dim=1)
            xs = self.motion_planner._unnormalize_x(xs_norm)
            return xs, None

        xs, record = self.motion_planner._sample_sequence(
            batch_size=cond.shape[0],
            length=length,
            context=full_context,
            context_mask=context_mask,
        )
        xs = self.motion_planner._unnormalize_x(xs)
        return xs, record

    def _mask_jacobian(
        self,
        jac: Tensor,
        *,
        view_keys: list[str] | None = None,
        num_views: int | None = None,
    ) -> Tensor:
        """Override in subclasses to zero out action channels. jac shape: [B, C, S, H, W] (b c s h w), C = action dim."""
        return jac

    def _preprocess_solve_inputs(
        self,
        J: Tensor,
        y: Tensor,
        weights: Tensor | None,
        *,
        path_kind: str,
        target_view_widths: list[int] | None = None,
        track_xs: Tensor | None = None,
    ) -> Tuple[Tensor, Tensor, Tensor | None]:
        """Hook: transform (J, y, weights) just before the least-squares solve.
        path_kind is "flow" or "track". Subclasses can override to implement
        e.g. per-view flow normalization. Default: pass-through."""
        return J, y, weights

    def compute_jacobian(self, obs: PolicyObservation) -> Tensor:
        source_view_widths = self._source_view_widths_for_obs(
            obs, int(obs.rgb.shape[-2])
        )
        rgb = torch.from_numpy(obs.rgb).permute(0, 3, 1, 2).to(self.device)
        _, jacobian_flat, _, _, _ = self._compute_jacobian_from_rgb_tensor(
            rgb,
            source_view_widths,
            view_keys=obs.view_keys,
        )
        return jacobian_flat

    def _resize(self, rgb: np.ndarray, size: Tuple[int, int]) -> Tensor:
        if rgb.ndim == 3:
            rgb = rgb[np.newaxis]
        x = torch.from_numpy(rgb).permute(0, 3, 1, 2).to(self.device)
        # F.interpolate(bilinear, align_corners=None) is NOT a true no-op even
        # when target size == input size — it resamples at pixel-center
        # coordinates and subtly smooths the input. Short-circuit that case so
        # the server path matches the in-process path (which loads frames at
        # model resolution and passes them through without any resize).
        if int(x.shape[-2]) == int(size[0]) and int(x.shape[-1]) == int(size[1]):
            return x
        # Match TRAINING (and the server path's _resize_context_rgb:1881-1889): a
        # multiview canvas at the right height but NARROWER than the planner width is
        # BLACK-PADDED on the RIGHT (view_pad_position=right), NOT stretched. The omni
        # model saw allegro's three 128px views (=384) padded to 576 with the right
        # slot black; bilinear-stretching to 576 distorts each view ~1.5x and (after
        # the dream's sum(view_widths) crop) scrambles the view split fed to the
        # tracker/Jacobian -> inconsistent per-step du -> oscillatory, non-accumulating
        # motion. Only the in-process joint path reaches here with W<target (the server
        # path already pads via _resize_context_rgb); short-circuit above keeps the
        # matched-size case (incl. the legacy 384-wide directional model) untouched.
        if int(x.shape[-2]) == int(size[0]) and int(x.shape[-1]) < int(size[1]):
            pad = torch.zeros(
                (x.shape[0], x.shape[1], int(size[0]), int(size[1])),
                dtype=x.dtype, device=x.device,
            )
            pad[..., : int(x.shape[-1])] = x
            return pad
        return torch.nn.functional.interpolate(x, size=size, mode="bilinear")

    def _enqueue_observation_images(self, obs: PolicyObservation) -> Tensor:
        """Append the latest observation once and return the full context queue.

        Queue behavior for chunked joint control:

            call k:   obs_t     -> enqueue -> [obs_t-12 ... obs_t]
                      plan/pop  -> action_t -> env.step -> obs_t+1

            call k+1: obs_t+1   -> enqueue -> [obs_t-11 ... obs_t+1]
                      pop only  -> action_t+1 -> env.step -> obs_t+2

        So even while reusing a dreamed chunk, the context still advances
        serially with each real observation returned by the runner.
        """
        im = self._resize(obs.rgb, self.image_size_motion_planner)
        self._queues = populate_queues(self._queues, {"observation.images": im})
        return torch.stack(list(self._queues["observation.images"]), dim=1)

    # ------------------------- policy -------------------------

    @torch.no_grad()
    def predict_action(self, obs: PolicyObservation) -> PolicyOutput:
        if self.is_joint:
            return self._predict_action_joint(obs)
        return self._predict_action_flow(obs)

    def _predict_action_flow(self, obs: PolicyObservation) -> PolicyOutput:
        cond = self._enqueue_observation_images(obs)
        flow = self._compute_plan_flow(cond)
        return self._flow_to_action(obs, flow)

    def get_final_action(self, du: Tensor) -> Tensor:
        """Return the action to execute and visualize. Subclasses (e.g. gripper gating) may override to transform du."""
        return du

    @staticmethod
    def _tensor_to_numpy_copy(value: Tensor | np.ndarray | None) -> np.ndarray | None:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy().copy()
        return np.asarray(value).copy()

    def _build_action_debug_info(
        self,
        du_pre_clip: Tensor | np.ndarray,
        du_pre_gate: Tensor | np.ndarray,
        du_final: Tensor | np.ndarray,
    ) -> dict[str, np.ndarray]:
        return {
            "action_pre_clip": cast(
                np.ndarray, self._tensor_to_numpy_copy(du_pre_clip)
            ),
            "action_pre_gate": cast(
                np.ndarray, self._tensor_to_numpy_copy(du_pre_gate)
            ),
            "action_final": cast(np.ndarray, self._tensor_to_numpy_copy(du_final)),
        }

    def _collect_policy_debug_info(
        self,
        action_debug: dict[str, np.ndarray] | None,
    ) -> dict[str, Any]:
        info: dict[str, Any] = {}
        if action_debug is not None:
            info["action_debug"] = action_debug
        return info

    @staticmethod
    def _debug_dump_label_from_value(
        value: Any,
        *,
        fallback: str = "unset",
    ) -> str:
        if value is None:
            return fallback
        if isinstance(value, (list, tuple)):
            parts = [
                MotionPolicy._debug_dump_label_from_value(v, fallback="")
                for v in value[:3]
            ]
            joined = "_".join(p for p in parts if p)
            if len(value) > 3:
                joined = f"{joined}_plus{len(value) - 3}"
            return joined or fallback
        text = str(value).strip()
        return text or fallback

    @staticmethod
    def _debug_dump_slug(value: Any, *, fallback: str, max_len: int = 72) -> str:
        text = MotionPolicy._debug_dump_label_from_value(value, fallback=fallback)
        text = text.lower()
        text = re.sub(r"[^a-z0-9]+", "_", text)
        text = re.sub(r"_+", "_", text).strip("_")
        if not text:
            text = fallback
        return text[:max_len].strip("_") or fallback

    def _reset_debug_dump_run_state(self) -> None:
        self._debug_dump_run_dir = None
        self._debug_dump_chunk_idx = 0
        self._debug_dump_bytes_written = 0
        self._debug_dump_warned = False

    def _reset_debug_dump_run_state_locked(self) -> None:
        """Lock-guarded reset for callers NOT already holding _debug_dump_lock (reset()
        and every configure_runtime path), so the run-state counters can't be zeroed
        out from under the async writer's read-modify-write. The lock is non-reentrant —
        only call this from an unlocked context."""
        with self._debug_dump_lock:
            self._reset_debug_dump_run_state()

    def _debug_dump_model_label(self) -> str:
        explicit = getattr(self.cfg, "debug_dump_model_name", None)
        if explicit:
            return str(explicit)
        planner_cfg = getattr(self.cfg, "motion_planner", None)
        algo_path = getattr(planner_cfg, "algorithm_config_path", None)
        if algo_path:
            return Path(str(algo_path)).stem
        return "model_unknown"

    def _debug_dump_task_label(self) -> str:
        explicit = getattr(self.cfg, "debug_dump_task_name", None)
        if explicit:
            return str(explicit)
        return self._debug_dump_label_from_value(
            getattr(self.cfg, "text_conditioning", None),
            fallback="task_unset",
        )

    def _write_debug_dump_run_metadata(self, run_dir: Path) -> None:
        planner_cfg = getattr(self.cfg, "motion_planner", None)
        payload = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
            "model_name": self._debug_dump_model_label(),
            "task_name": self._debug_dump_task_label(),
            "text_conditioning": getattr(self.cfg, "text_conditioning", None),
            "action_chunk_horizon": int(getattr(self.cfg, "action_chunk_horizon", 0) or 0),
            "n_action_steps": int(getattr(self.cfg, "n_action_steps", 0) or 0),
            "context_frames": getattr(self.cfg, "context_frames", None),
            "control_view_keys": getattr(self.cfg, "control_view_keys", None),
            "motion_plan_scale": float(getattr(self.cfg, "motion_plan_scale", 0.0) or 0.0),
            "algorithm_config_path": (
                None
                if planner_cfg is None
                else getattr(planner_cfg, "algorithm_config_path", None)
            ),
            "debug_dump_dir": str(getattr(self.cfg, "debug_dump_dir", "")),
        }
        try:
            with (run_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)
                f.write("\n")
        except OSError as exc:
            self._log(f"[debug-dump] WARN failed to write run metadata: {exc}")

    def _ensure_debug_dump_run_dir(self) -> Path | None:
        if not bool(getattr(self.cfg, "debug_dump_enabled", False)):
            return None
        if self._debug_dump_run_dir is not None:
            return self._debug_dump_run_dir
        root = Path(str(getattr(self.cfg, "debug_dump_dir", ""))).expanduser()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_slug = self._debug_dump_slug(
            self._debug_dump_model_label(),
            fallback="model_unknown",
        )
        task_slug = self._debug_dump_slug(
            self._debug_dump_task_label(),
            fallback="task_unset",
        )
        run_name = f"run_{stamp}__model-{model_slug}__task-{task_slug}__pid{os.getpid()}"
        run_dir = root / run_name
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            run_dir = root / f"{run_name}_{self._debug_dump_chunk_idx}"
            run_dir.mkdir(parents=True, exist_ok=True)
        self._debug_dump_run_dir = run_dir
        self._debug_dump_chunk_idx = 0
        self._debug_dump_bytes_written = 0
        self._debug_dump_warned = False
        self._write_debug_dump_run_metadata(run_dir)
        return run_dir

    def _debug_dump_budget_allows(self, run_dir: Path) -> bool:
        max_chunks = int(getattr(self.cfg, "debug_dump_max_chunks", 0) or 0)
        max_bytes = int(getattr(self.cfg, "debug_dump_max_bytes", 0) or 0)
        min_free = int(getattr(self.cfg, "debug_dump_min_free_bytes", 0) or 0)
        if max_chunks > 0 and self._debug_dump_chunk_idx >= max_chunks:
            if not self._debug_dump_warned:
                self._log(
                    f"[debug-dump] disabled after {self._debug_dump_chunk_idx} chunks "
                    f"(max={max_chunks})"
                )
                self._debug_dump_warned = True
            return False
        if max_bytes > 0 and self._debug_dump_bytes_written >= max_bytes:
            if not self._debug_dump_warned:
                self._log(
                    f"[debug-dump] disabled after {self._debug_dump_bytes_written} bytes "
                    f"(max={max_bytes})"
                )
                self._debug_dump_warned = True
            return False
        if min_free > 0:
            try:
                usage = os.statvfs(run_dir)
                free_bytes = int(usage.f_bavail * usage.f_frsize)
            except OSError as exc:
                # statvfs can throw on some FUSE/overlay/network mounts. Don't let a stat
                # FAILURE masquerade as a full disk and silently disable dumps for the whole
                # run (min_free defaults to 2 GiB, so this branch is always active). Proceed,
                # but warn once so the operator knows the free-space guard is inactive.
                if not self._debug_dump_warned:
                    self._log(
                        f"[debug-dump] WARN statvfs({run_dir}) failed: {exc}; "
                        f"free-space guard inactive (proceeding)"
                    )
                    self._debug_dump_warned = True
                free_bytes = min_free
            if free_bytes < min_free:
                if not self._debug_dump_warned:
                    self._log(
                        f"[debug-dump] disabled; free space {free_bytes} < {min_free}"
                    )
                    self._debug_dump_warned = True
                return False
        return True

    @staticmethod
    def _dump_tensor(value: Any) -> np.ndarray | None:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _stack_vis_frame_values(
        self,
        vis_frames: list[Dict[str, Any]],
        key: str,
    ) -> np.ndarray | None:
        values = []
        for frame in vis_frames:
            value = self._dump_tensor(frame.get(key))
            if value is not None:
                values.append(value)
        if not values:
            return None
        try:
            return np.stack(values, axis=1)
        except ValueError:
            return np.asarray(values, dtype=object)

    def _dump_policy_chunk_trajectory(
        self,
        *,
        kind: str,
        obs: PolicyObservation,
        actions_raw: Tensor | None,
        actions_final: Tensor | np.ndarray | None,
        action_debug: dict[str, Any] | None,
        vis_frames: list[Dict[str, Any]],
        context_rgb: Tensor | None,
        info: dict[str, Any] | None = None,
    ) -> str | None:
        # Reserve the run dir + chunk index under the lock (shared with the async
        # writer that updates _debug_dump_bytes_written). The budget gate must read a
        # consistent (chunk_idx, bytes_written) snapshot, so it lives inside the lock.
        with self._debug_dump_lock:
            run_dir = self._ensure_debug_dump_run_dir()
            if run_dir is None or not self._debug_dump_budget_allows(run_dir):
                return None
            idx = self._debug_dump_chunk_idx
            self._debug_dump_chunk_idx += 1
        now = datetime.now(timezone.utc)
        stamp = now.strftime("%Y%m%dT%H%M%S_%fZ")
        out_path = run_dir / f"{stamp}_chunk_{idx:06d}_{kind}.npz"

        payload: dict[str, Any] = {
            "created_at_utc": np.asarray(now.isoformat()),
            "created_at_unix": np.asarray(now.timestamp(), dtype=np.float64),
            "dump_index": np.asarray(idx, dtype=np.int64),
            "kind": np.asarray(kind),
            "step_index": np.asarray(
                -1 if obs.step_index is None else int(obs.step_index),
                dtype=np.int64,
            ),
            "dt": np.asarray(float(obs.dt), dtype=np.float32),
            "action_mode": np.asarray(str(obs.action_mode)),
            "view_keys_json": np.asarray(json.dumps(list(obs.view_keys or []))),
            "view_widths": np.asarray(obs.view_widths or [], dtype=np.int64),
            "metadata_json": np.asarray(json.dumps(info or {}, default=str)),
        }

        optional_values = {
            "q_robot": obs.q_robot,
            "q_robot_trajectory": getattr(obs, "q_robot_trajectory", None),
            "state_trajectory": getattr(obs, "state_trajectory", None),
            "context_rgb": context_rgb,
            "actions_raw": actions_raw,
            "actions_final": actions_final,
        }
        if action_debug is not None:
            for key, value in action_debug.items():
                optional_values[f"action_debug_{key}"] = value
        for key in (
            "rgb",
            "target_rgb",
            "flow",
            "jacobian",
            "curr_track",
            "trgt_track",
            "curr_visible",
            "control_visible",
            "target_view_widths",
        ):
            optional_values[key] = self._stack_vis_frame_values(vis_frames, key)

        for key, value in optional_values.items():
            arr = self._dump_tensor(value)
            if arr is not None:
                payload[key] = arr

        # Copy-on-handoff (load-bearing for async correctness): _dump_tensor returns
        # `.cpu().numpy()`, which for an already-CPU tensor (e.g. client-owned q_robot /
        # state_trajectory) ALIASES the tensor's storage. The next chunk may mutate that
        # tensor while the writer thread is still compressing this payload -> torn reads.
        # A copy here detaches every array from live storage; it's cheap relative to the
        # zlib compress and dominated only by the genuinely-aliased small arrays (the big
        # GPU-sourced arrays are already fresh from `.cpu()` / `np.stack`).
        payload = {
            k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v)
            for k, v in payload.items()
        }

        # Hand the fully-detached payload to the writer thread and return immediately.
        item = (out_path, payload, now, idx, kind, obs.step_index)
        self._ensure_debug_dump_worker()
        q = self._debug_dump_queue
        if q is not None:
            try:
                q.put_nowait(item)
            except queue.Full:
                # Never block inference on the writer. Drops are ~impossible in practice
                # (a write is ~10x faster than a chunk) and only happen on a disk stall.
                self._log(
                    f"[debug-dump] WARN writer behind; dropping chunk {idx:06d} "
                    f"(queue full)"
                )
        return str(out_path)

    def _ensure_debug_dump_worker(self) -> None:
        """Lazily start the single daemon writer thread (no cost on non-dump runs)."""
        t = self._debug_dump_thread
        if t is not None and t.is_alive():
            return
        with self._debug_dump_lock:
            t = self._debug_dump_thread
            if t is not None and t.is_alive():
                return
            if self._debug_dump_queue is None:
                self._debug_dump_queue = queue.Queue(maxsize=8)
            self._debug_dump_thread = threading.Thread(
                target=self._debug_dump_worker_loop,
                name="vera-debug-dump-writer",
                daemon=True,
            )
            self._debug_dump_thread.start()
            if not self._debug_dump_atexit_registered:
                atexit.register(self.flush)
                self._debug_dump_atexit_registered = True

    def _debug_dump_worker_loop(self) -> None:
        q = self._debug_dump_queue
        if q is None:
            return
        while True:
            item = q.get()
            try:
                if item is None:  # shutdown sentinel
                    return
                self._debug_dump_write_one(*item)
            except Exception as exc:  # noqa: BLE001 — one bad chunk must not kill the thread
                self._log(f"[debug-dump] WARN writer error: {exc}")
            finally:
                q.task_done()

    def _debug_dump_write_one(
        self, out_path: Path, payload: dict[str, Any], now, idx: int, kind: str,
        step_index: int | None,
    ) -> None:
        # Atomic write: compress to a temp file, then os.replace into place so a reader
        # never sees a partial .npz (and a crash leaves an orphan .tmp, not a torn .npz).
        tmp_path = Path(str(out_path) + ".tmp")
        try:
            with open(tmp_path, "wb") as fh:
                np.savez_compressed(fh, **payload)
            os.replace(tmp_path, out_path)
            written = int(out_path.stat().st_size)
        except OSError as exc:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            with self._debug_dump_lock:
                if not self._debug_dump_warned:
                    self._log(f"[debug-dump] WARN failed to save {out_path}: {exc}")
                    self._debug_dump_warned = True
            return
        with self._debug_dump_lock:
            # Only charge bytes to the run that produced this chunk. reset()/configure can
            # start a NEW run while old-run items are still draining (we intentionally don't
            # clear the queue); charging their bytes to the new run's counter would corrupt
            # its max_bytes budget (cross-run contamination). out_path.parent is the run dir.
            if (
                self._debug_dump_run_dir is not None
                and out_path.parent == self._debug_dump_run_dir
            ):
                self._debug_dump_bytes_written += written
        manifest = {
            "path": str(out_path),
            "bytes": written,
            "created_at_utc": now.isoformat(),
            "dump_index": idx,
            "kind": kind,
            "step_index": step_index,
            "keys": sorted(payload.keys()),
        }
        try:
            with (out_path.parent / "manifest.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(manifest) + "\n")
        except OSError as exc:
            self._log(f"[debug-dump] WARN manifest append failed: {exc}")
        self._log(f"[debug-dump] wrote {out_path} ({written / 1024**2:.1f} MiB)")

    def flush(self) -> None:
        """Drain any queued chunk dumps and stop the writer (called on shutdown).

        Idempotent and safe to call when dumping was never used. The websocket
        transport's SIGTERM/SIGINT/atexit handler forwards here via the adapter.
        """
        q = self._debug_dump_queue
        t = self._debug_dump_thread
        if q is None or t is None or not t.is_alive():
            return
        try:
            q.put(None, timeout=5)
            t.join(timeout=60)
        except Exception as exc:  # noqa: BLE001
            self._log(f"[debug-dump] WARN flush failed: {exc}")

    def _jacobian_abs_stats_for_vis_frames(
        self,
        vis_frames: list[Dict[str, Any]] | None,
    ) -> dict[str, float] | None:
        if not vis_frames:
            return None
        max_samples = 1_000_000
        per_frame_samples = max(1, int(np.ceil(max_samples / len(vis_frames))))
        samples: list[Tensor] = []
        total_values = 0
        for frame in vis_frames:
            jacobian = frame.get("jacobian")
            if not isinstance(jacobian, torch.Tensor):
                continue
            values = jacobian.detach().float().abs().reshape(-1)
            total_values += int(values.numel())
            if values.numel() > per_frame_samples:
                step = max(1, int(np.ceil(values.numel() / per_frame_samples)))
                values = values[::step][:per_frame_samples]
            values = values[torch.isfinite(values)]
            if values.numel() > 0:
                samples.append(values.cpu())
        if not samples:
            return None
        vals = torch.cat(samples)
        quantiles = torch.tensor(
            [0.0, 0.5, 0.9, 0.95, 0.99, 0.999, 1.0],
            dtype=vals.dtype,
        )
        qs = torch.quantile(vals, quantiles)
        return {
            "min": float(qs[0].item()),
            "p50": float(qs[1].item()),
            "p90": float(qs[2].item()),
            "p95": float(qs[3].item()),
            "p99": float(qs[4].item()),
            "p999": float(qs[5].item()),
            "max": float(qs[6].item()),
            "mean": float(vals.mean().item()),
            "sample_count": float(vals.numel()),
            "total_count": float(total_values),
        }

    def _split_joint_plan_output(
        self,
        plan_output: Any,
    ) -> Tuple[Tensor, dict[str, Any] | None]:
        if isinstance(plan_output, tuple):
            return plan_output
        return plan_output, None

    def _resize_context_rgb(self, context_rgb: np.ndarray) -> Tensor:
        """Convert client-owned RGB context to planner-sized [B, T, C, H, W]."""
        rgb = np.asarray(context_rgb)
        if rgb.ndim == 4:
            rgb = rgb[np.newaxis]
        if rgb.ndim != 5 or rgb.shape[-1] != 3:
            raise ValueError(
                "context_rgb must have shape [T,H,W,3] or [B,T,H,W,3], "
                f"got {tuple(rgb.shape)}"
            )
        if rgb.dtype == np.uint8:
            rgb = rgb.astype(np.float32) / 255.0
        elif not np.issubdtype(rgb.dtype, np.floating) or rgb.dtype != np.float32:
            rgb = rgb.astype(np.float32)

        x = torch.from_numpy(rgb).permute(0, 1, 4, 2, 3).to(self.device)
        B, T, C, H, W = x.shape
        if int(H) == int(self.image_size_motion_planner[0]) and int(W) == int(
            self.image_size_motion_planner[1]
        ):
            return x
        x_flat = rearrange(x, "b t c h w -> (b t) c h w")
        target_h, target_w = int(self.image_size_motion_planner[0]), int(self.image_size_motion_planner[1])
        _, _, H_, W_ = x_flat.shape
        if int(H_) == target_h and int(W_) < target_w:
            # Match TRAINING: a multiview canvas narrower than the planner width is BLACK-PADDED on
            # the RIGHT (view_pad_position=right), NOT stretched — e.g. the omni model sees mimicgen's
            # 2-view 256-wide context padded to 576, with the 3rd view slot black. Stretching would
            # horizontally distort the image away from the training distribution.
            pad = torch.zeros((x_flat.shape[0], x_flat.shape[1], target_h, target_w),
                              dtype=x_flat.dtype, device=x_flat.device)
            pad[..., :W_] = x_flat
            x_flat = pad
        else:
            x_flat = torch.nn.functional.interpolate(
                x_flat, size=self.image_size_motion_planner, mode="bilinear",
            )
        return rearrange(x_flat, "(b t) c h w -> b t c h w", b=B, t=T)

    def _append_context_rgb_to_queue(self, context_rgb: np.ndarray) -> Tensor:
        """Append uploaded frames to the legacy image queue and return the queue."""
        context = self._resize_context_rgb(context_rgb)
        for frame_idx in range(context.shape[1]):
            self._queues["observation.images"].append(context[:, frame_idx])
        return torch.stack(list(self._queues["observation.images"]), dim=1)

    def _plan_joint_action_chunk_raw(
        self,
        obs: PolicyObservation,
        cond: Tensor,
        *,
        execute_horizon: int | None = None,
        queue_adaptive_feedback: bool = False,
    ) -> tuple[Tensor, list[Dict[str, Any]], dict[str, Any]]:
        """Plan and solve a joint-policy action chunk without touching action_queue."""
        plan_output = self._compute_plan_joint(obs, cond)
        xs, record = self._split_joint_plan_output(plan_output)
        planner_sampling_path = None
        text_conditioning = None
        text_conditioning_enabled = None
        if isinstance(record, dict):
            planner_sampling_path = record.get("planner_sampling_path")
            text_conditioning = record.get("text_conditioning")
            if "text_conditioning_enabled" in record:
                text_conditioning_enabled = bool(record["text_conditioning_enabled"])

        runtime_controller = self._runtime_controller_params()
        n_ctx_tokens = int(
            record["context_len"]
            if isinstance(record, dict) and "context_len" in record
            else self.context_frames
        )

        rgb_future = xs[:, n_ctx_tokens:, :3]
        flow_future = xs[:, n_ctx_tokens:, 3:5] * self.cfg.motion_plan_scale
        # The planner crops view-padding off the dream (rgb_future is now the valid width); align the
        # padded context `cond` to it so context+dream concatenations downstream match. No-op when
        # there's no pad (cond and dream already share a width).
        if int(cond.shape[-1]) != int(rgb_future.shape[-1]):
            _vw = int(rgb_future.shape[-1])
            cond = (cond[..., :_vw] if self._view_pad_position == "right"
                    else cond[..., int(cond.shape[-1]) - _vw:])
        motion_tracks = None if record is None else record.get("motion_tracks")
        track_source_rgb = None if record is None else record.get("track_source_rgb")
        track_target_rgb = None if record is None else record.get("track_target_rgb")
        max_exec = (
            int(execute_horizon)
            if execute_horizon is not None
            else int(self.cfg.n_action_steps)
        )
        max_exec = max(max_exec, 0)

        if motion_tracks is not None and motion_tracks["disp"].shape[1] > 0:
            available_track_steps = int(motion_tracks["disp"].shape[1])
            available_source_steps = (
                int(track_source_rgb.shape[1])
                if isinstance(track_source_rgb, torch.Tensor)
                else int(rgb_future.shape[1])
            )
            available_target_steps = (
                int(track_target_rgb.shape[1])
                if isinstance(track_target_rgb, torch.Tensor)
                else int(rgb_future.shape[1])
            )
            n_exec = min(
                available_source_steps,
                available_target_steps,
                available_track_steps,
                max_exec,
            )
        else:
            n_exec = min(int(flow_future.shape[1]), max_exec)

        if VERBOSE_POLICY_DEBUG:
            print(
                f"n_exec: {n_exec} "
                f"rgb_future.shape: {rgb_future.shape} "
                f"flow_future.shape: {flow_future.shape}"
            )

        if n_exec <= 0:
            action_dim = int(
                len(
                    getattr(
                        self,
                        "get_dynamics_normalization_metadata",
                        lambda: {},
                    )().get("action_abs_scale", [])
                )
                or 0
            )
            if action_dim <= 0:
                action_dim = int(getattr(self, "action_dim", 0) or 0)
            return (
                torch.zeros(cond.shape[0], 0, action_dim, device=self.device),
                [],
                {
                    "action_chunk_horizon": int(self.cfg.action_chunk_horizon),
                    "action_chunk_length": 0,
                    "context_frames_used": int(cond.shape[1]),
                    "planner_context_len": int(n_ctx_tokens),
                    "planner_sampling_path": planner_sampling_path,
                    "text_conditioning": text_conditioning,
                    "text_conditioning_enabled": text_conditioning_enabled,
                },
            )

        flow_source_rgb = torch.cat(
            [
                cond[:, -1:],
                rgb_future[:, : max(n_exec - 1, 0)],
            ],
            dim=1,
        )
        source_tensor = (
            track_source_rgb[:, :n_exec]
            if isinstance(track_source_rgb, torch.Tensor)
            else flow_source_rgb[:, :n_exec]
        )
        joint_source_view_widths = self._source_view_widths_for_obs(
            obs,
            int(source_tensor.shape[-1]),
        )
        planned_dream_index = self._next_dream_index
        if motion_tracks is not None and n_exec > 0:
            target_tensor = (
                track_target_rgb[:, :n_exec]
                if isinstance(track_target_rgb, torch.Tensor)
                else rgb_future[:, :n_exec]
            )
            if queue_adaptive_feedback:
                self._queue_adaptive_feedback_targets(
                    source_tensor,
                    motion_tracks,
                    num_steps=n_exec,
                    view_keys=obs.view_keys,
                    view_widths=joint_source_view_widths,
                    dream_index=planned_dream_index,
                )
            actions, vis_frames = self._track_rgb_chunk_to_actions(
                obs,
                source_tensor,
                motion_tracks,
                joint_source_view_widths,
                target_rgb=target_tensor,
                lam_override=float(runtime_controller["lam_runtime"]),
            )
        else:
            actions, vis_frames = self._flow_rgb_chunk_to_actions(
                obs,
                flow_source_rgb[:, :n_exec],
                rgb_future[:, :n_exec],
                flow_future[:, :n_exec],
                joint_source_view_widths,
                lam_override=float(runtime_controller["lam_runtime"]),
            )

        for t, frame in enumerate(vis_frames):
            frame["context_rgb"] = cond
            frame["is_boundary_source_frame"] = bool(t == 0)
            frame["source_frame_role"] = "boundary" if t == 0 else "dream"
            frame["target_frame_role"] = "dream"

        info = {
            "action_chunk_horizon": int(self.cfg.action_chunk_horizon),
            "action_chunk_length": int(actions.shape[1]),
            "context_frames_used": int(cond.shape[1]),
            "planner_context_len": int(n_ctx_tokens),
            "control_view_keys": self.cfg.control_view_keys,
            "planner_sampling_path": planner_sampling_path,
            "text_conditioning": text_conditioning,
            "text_conditioning_enabled": text_conditioning_enabled,
            "chunk_story_index": int(planned_dream_index),
            "dream_rollout": self._build_dream_rollout_payload(
                dream_index=planned_dream_index,
                context_rgb=cond[:, -n_ctx_tokens:],
                future_rgb=rgb_future,
                exec_horizon=n_exec,
                planner_context_len=n_ctx_tokens,
                planner_sampling_path=planner_sampling_path,
                text_conditioning=text_conditioning,
                text_conditioning_enabled=text_conditioning_enabled,
            ),
        }
        return actions, vis_frames, info

    @staticmethod
    def _cpu_float_numpy(value: Tensor | np.ndarray | None) -> np.ndarray | None:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().float().numpy()
        return np.asarray(value, dtype=np.float32)

    @staticmethod
    def _adjacent_frame_mad_summary(value: Tensor | np.ndarray | None) -> dict[str, Any] | None:
        arr = MotionPolicy._cpu_float_numpy(value)
        if arr is None:
            return None
        if arr.ndim == 5:
            arr = arr[0]
        if arr.ndim != 4:
            return None
        if arr.shape[1] >= 3 and arr.shape[-1] != 3:
            arr = np.transpose(arr[:, :3], (0, 2, 3, 1))
        elif arr.shape[-1] >= 3:
            arr = arr[..., :3]
        else:
            return None
        if arr.shape[0] < 2:
            return {
                "n_frames": int(arr.shape[0]),
                "n_pairs": 0,
                "values_u8": [],
            }
        diffs = np.mean(
            np.abs(arr[1:].astype(np.float32) - arr[:-1].astype(np.float32)),
            axis=(1, 2, 3),
        )
        finite = diffs[np.isfinite(diffs)]
        if finite.size == 0:
            return None
        scale = 255.0 if float(np.nanmax(np.abs(arr))) <= 2.0 else 1.0
        values = finite * scale
        return {
            "n_frames": int(arr.shape[0]),
            "n_pairs": int(values.size),
            "min_u8": float(np.min(values)),
            "p25_u8": float(np.percentile(values, 25)),
            "median_u8": float(np.median(values)),
            "p75_u8": float(np.percentile(values, 75)),
            "max_u8": float(np.max(values)),
            "lt1_u8": int(np.sum(values < 1.0)),
            "lt3_u8": int(np.sum(values < 3.0)),
            "values_u8": [float(x) for x in values.tolist()],
        }

    def _build_dream_rollout_payload(
        self,
        *,
        dream_index: int,
        context_rgb: Tensor | np.ndarray,
        future_rgb: Tensor | np.ndarray,
        exec_horizon: int,
        planner_context_len: int,
        planner_sampling_path: Any = None,
        text_conditioning: Any = None,
        text_conditioning_enabled: Any = None,
    ) -> dict[str, Any]:
        """Small server-local payload for the dashboard's per-dream rollout strip."""
        context_np = self._cpu_float_numpy(context_rgb)
        future_np = self._cpu_float_numpy(future_rgb)
        future_len = int(future_np.shape[1]) if future_np is not None and future_np.ndim >= 5 else 0
        exec_len = max(0, min(int(exec_horizon), future_len))
        return {
            "dream_index": int(dream_index),
            "context_rgb": context_np,
            "future_rgb": future_np,
            "exec_horizon": exec_len,
            "planner_context_len": int(planner_context_len),
            "action_chunk_horizon": int(self.cfg.action_chunk_horizon),
            "n_action_steps": int(self.cfg.n_action_steps),
            "planner_sampling_path": planner_sampling_path,
            "text_conditioning": text_conditioning,
            "text_conditioning_enabled": text_conditioning_enabled,
            "context_adj_mad": self._adjacent_frame_mad_summary(context_np),
            "future_adj_mad": self._adjacent_frame_mad_summary(future_np),
        }

    def _finalize_action_chunk(
        self,
        obs: PolicyObservation,
        actions: Tensor,
    ) -> tuple[Tensor, dict[str, Any]]:
        runtime_controller = self._runtime_controller_params()
        du_pre_clip = self._apply_runtime_action_scaling(actions, runtime_controller)
        du = du_pre_clip.clamp(
            -self.cfg.controller.clip_du, self.cfg.controller.clip_du
        )
        final_steps = [self.get_final_action(du[:, t]) for t in range(du.shape[1])]
        if final_steps:
            du_final = torch.stack(final_steps, dim=1)
        else:
            du_final = du
        action_debug = self._build_action_debug_info(du_pre_clip, du, du_final)
        policy_debug = self._collect_policy_debug_info(action_debug)
        self._build_action_feedback_payload(obs, action_debug)
        return du_final, policy_debug

    @torch.no_grad()
    def predict_action_chunk(
        self,
        obs: PolicyObservation,
        *,
        context_rgb: np.ndarray | None = None,
        execute_horizon: int | None = None,
        context_update_mode: str = "replace",
    ) -> PolicyOutput:
        """Return a whole executable action chunk for client-owned context mode."""
        if not self.is_joint:
            raise RuntimeError("predict_action_chunk is only supported for joint policies")
        context_update_mode = str(context_update_mode or "replace").lower()
        if context_update_mode not in {"replace", "append"}:
            raise ValueError(
                "context_update_mode must be 'replace' or 'append', "
                f"got {context_update_mode!r}"
            )
        if context_update_mode == "append":
            cond = (
                self._append_context_rgb_to_queue(context_rgb)
                if context_rgb is not None
                else self._enqueue_observation_images(obs)
            )
        else:
            cond = (
                self._resize_context_rgb(context_rgb)
                if context_rgb is not None
                else self._enqueue_observation_images(obs)
            )
        if self.dynamics_model is None:
            self._compute_plan_joint(obs, cond)
            return PolicyOutput(
                action=np.zeros((cond.shape[0], 0, 0), dtype=np.float32),
                info={
                    "action_chunk_length": 0,
                    "context_update_mode": context_update_mode,
                    "queue_len": len(self._queues.get("observation.images", [])),
                    "queue_max": int(
                        self._queues.get("observation.images", deque()).maxlen or 0
                    ),
                },
            )

        actions, vis_frames, info = self._plan_joint_action_chunk_raw(
            obs,
            cond,
            execute_horizon=execute_horizon,
            queue_adaptive_feedback=False,
        )
        actions_final, policy_debug = self._finalize_action_chunk(obs, actions)
        self._current_dream_index = int(info.get("chunk_story_index", 0))
        self._next_dream_index += 1

        policy_vis = None
        desired_flow_numpy = None
        desired_source_rgb_numpy = None
        track_stats = None
        control_track_stats = None
        outlier_stats = None
        if vis_frames:
            first_frame = vis_frames[0]
            track_stats = first_frame.get("track_stats")
            control_track_stats = first_frame.get("control_track_stats")
            outlier_stats = first_frame.get("outlier_stats")
            flow_t = first_frame.get("flow")
            if flow_t is not None:
                try:
                    f = flow_t.detach()
                    if f.ndim == 4:
                        f = f[0]
                    desired_flow_numpy = f.cpu().float().numpy()
                except Exception:
                    desired_flow_numpy = None
            src_rgb = first_frame.get("rgb")
            if src_rgb is not None:
                try:
                    s = src_rgb.detach()
                    if s.ndim == 4:
                        s = s[0]
                    desired_source_rgb_numpy = s.cpu().float().numpy()
                except Exception:
                    desired_source_rgb_numpy = None
            if obs.rgb_vis is not None:
                policy_vis = self._make_policy_vis_joint(
                    obs,
                    vis_frames,
                    dream_index=self._current_dream_index,
                    action=None,
                    action_debug=cast(
                        dict[str, Any] | None,
                        policy_debug.get("action_debug"),
                    ),
                    gripper_debug=cast(
                        dict[str, Any] | None,
                        policy_debug.get("gripper_debug"),
                    ),
        )
        jacobian_abs_stats = self._jacobian_abs_stats_for_vis_frames(vis_frames)
        debug_dump_path = self._dump_policy_chunk_trajectory(
            kind=f"plan_chunk_{context_update_mode}",
            obs=obs,
            actions_raw=actions,
            actions_final=actions_final,
            action_debug=cast(
                dict[str, Any] | None,
                policy_debug.get("action_debug"),
            ),
            vis_frames=vis_frames,
            context_rgb=cond,
            info={
                **info,
                "context_update_mode": context_update_mode,
                "jacobian_abs_stats": jacobian_abs_stats,
            },
        )

        info.update(
            {
                "context_update_mode": context_update_mode,
                "queue_len": len(self._queues.get("observation.images", [])),
                "queue_max": int(
                    self._queues.get("observation.images", deque()).maxlen or 0
                ),
                "track_stats": track_stats,
                "control_track_stats": control_track_stats,
                "outlier_stats": outlier_stats,
                "jacobian_abs_stats": jacobian_abs_stats,
                "policy_vis": policy_vis,
                "desired_flow": desired_flow_numpy,
                "desired_source_rgb": desired_source_rgb_numpy,
                "chunk_story_vis_frames": vis_frames,
                "chunk_story_context_rgb": cond.detach().cpu(),
                "chunk_story_dream_vis": policy_vis,
                "chunk_story_exec_horizon": int(actions_final.shape[1]),
                "debug_dump_path": debug_dump_path,
            }
        )
        info.update(policy_debug)
        adaptive_debug = self._adaptive_debug_info()
        if adaptive_debug is not None:
            info["adaptive_controller"] = adaptive_debug
        return PolicyOutput(
            action=actions_final.cpu().numpy(),
            info=info,
        )

    def _predict_action_joint(self, obs: PolicyObservation) -> PolicyOutput:
        cond = self._enqueue_observation_images(obs)
        # Worker ranks only participate in FSDP collectives (compute_plan);
        # they don't need the dynamics model, Jacobian, or action output.
        # They must call compute_plan at the same cadence as rank 0 (every
        # n_action_steps calls) to keep FSDP collectives in sync.
        if self.dynamics_model is None:
            if not hasattr(self, "_worker_step_counter"):
                self._worker_step_counter = 0
            if self._worker_step_counter == 0:
                self._compute_plan_joint(obs, cond)
            self._worker_step_counter = (self._worker_step_counter + 1) % max(
                self.cfg.n_action_steps, 1
            )
            return PolicyOutput(
                action=np.zeros((1, 7)),
                info={"action_chunk_remaining": 0},
            )

        chunk_story_context_rgb = None
        chunk_story_dream_vis = None
        chunk_story_exec_horizon = None
        chunk_story_index = None
        dream_rollout = None
        planner_sampling_path = None
        text_conditioning = None
        text_conditioning_enabled = None
        debug_dump_path = None
        if len(self._action_queue) == 0:
            plan_output = self._compute_plan_joint(obs, cond)
            xs, record = self._split_joint_plan_output(plan_output)
            if isinstance(record, dict):
                planner_sampling_path = record.get("planner_sampling_path")
                text_conditioning = record.get("text_conditioning")
                if "text_conditioning_enabled" in record:
                    text_conditioning_enabled = bool(
                        record["text_conditioning_enabled"]
                    )
            context_rgb_for_vis = cond
            planned_dream_index = self._next_dream_index
            runtime_controller = self._runtime_controller_params()

            # Use the planner-reported context length (= number of pixel
            # frames actually fed to the WAN VAE). Falls back to the
            # configured budget for planners that don't advertise it.
            n_ctx_tokens = int(
                record["context_len"]
                if isinstance(record, dict) and "context_len" in record
                else self.context_frames
            )

            rgb_future = xs[:, n_ctx_tokens:, :3]
            flow_future = xs[:, n_ctx_tokens:, 3:5] * self.cfg.motion_plan_scale
            motion_tracks = None if record is None else record.get("motion_tracks")
            track_source_rgb = (
                None if record is None else record.get("track_source_rgb")
            )
            track_target_rgb = (
                None if record is None else record.get("track_target_rgb")
            )
            if motion_tracks is not None and motion_tracks["disp"].shape[1] > 0:
                available_track_steps = motion_tracks["disp"].shape[1]
                available_source_steps = (
                    track_source_rgb.shape[1]
                    if isinstance(track_source_rgb, torch.Tensor)
                    else rgb_future.shape[1]
                )
                available_target_steps = (
                    track_target_rgb.shape[1]
                    if isinstance(track_target_rgb, torch.Tensor)
                    else rgb_future.shape[1]
                )
                n_exec = min(
                    available_source_steps,
                    available_target_steps,
                    available_track_steps,
                    self.cfg.n_action_steps,
                )
            else:
                n_exec = min(
                    flow_future.shape[1],
                    self.cfg.n_action_steps,
                )

            if VERBOSE_POLICY_DEBUG:
                print(
                    f"n_exec: {n_exec} "
                    f"rgb_future.shape: {rgb_future.shape} "
                    f"flow_future.shape: {flow_future.shape}"
                )

            # Omni planners pad the context canvas (e.g. 576) wider than the real views;
            # the generated dream (rgb_future) is cropped back to the valid view width.
            # Crop the context to match before concatenating (no-op when widths already match).
            _valid_w = int(rgb_future.shape[-1])
            flow_source_rgb = torch.cat(
                [
                    context_rgb_for_vis[:, -1:, ..., :_valid_w],
                    rgb_future[:, : max(n_exec - 1, 0)],
                ],
                dim=1,
            )
            source_tensor = (
                track_source_rgb[:, :n_exec]
                if isinstance(track_source_rgb, torch.Tensor)
                else flow_source_rgb[:, :n_exec]
            )
            joint_source_view_widths = self._source_view_widths_for_obs(
                obs,
                int(source_tensor.shape[-1]),
            )
            if motion_tracks is not None and n_exec > 0:
                target_tensor = (
                    track_target_rgb[:, :n_exec]
                    if isinstance(track_target_rgb, torch.Tensor)
                    else rgb_future[:, :n_exec]
                )
                self._queue_adaptive_feedback_targets(
                    source_tensor,
                    motion_tracks,
                    num_steps=n_exec,
                    view_keys=obs.view_keys,
                    view_widths=joint_source_view_widths,
                    dream_index=planned_dream_index,
                )
                actions, vis_frames = self._track_rgb_chunk_to_actions(
                    obs,
                    source_tensor,
                    motion_tracks,
                    joint_source_view_widths,
                    target_rgb=target_tensor,
                    lam_override=float(runtime_controller["lam_runtime"]),
                )
            else:
                actions, vis_frames = self._flow_rgb_chunk_to_actions(
                    obs,
                    flow_source_rgb[:, :n_exec],
                    rgb_future[:, :n_exec],
                    flow_future[:, :n_exec],
                    joint_source_view_widths,
                    lam_override=float(runtime_controller["lam_runtime"]),
                )
            if obs.rgb_vis is not None and vis_frames:
                chunk_story_context_rgb = context_rgb_for_vis.detach().cpu()
                chunk_story_dream_vis = self._make_policy_vis_joint(
                    obs,
                    vis_frames,
                    dream_index=planned_dream_index,
                    action=None,
                )
                chunk_story_exec_horizon = int(actions.shape[1])
                chunk_story_index = int(planned_dream_index)
            dream_rollout = self._build_dream_rollout_payload(
                dream_index=planned_dream_index,
                context_rgb=context_rgb_for_vis[:, -n_ctx_tokens:],
                future_rgb=rgb_future,
                exec_horizon=n_exec,
                planner_context_len=n_ctx_tokens,
                planner_sampling_path=planner_sampling_path,
                text_conditioning=text_conditioning,
                text_conditioning_enabled=text_conditioning_enabled,
            )
            # Mechanism B trajectory dump (the rich rgb + q_robot + per-frame
            # jacobian + tracks recorder). The chunked API path
            # (predict_action_chunk) already calls _dump_policy_chunk_trajectory
            # at its replan boundary; mirror that here so single-step clients
            # using predict_action also produce per-replan npz dumps. No-ops if
            # cfg.debug_dump_enabled is False.
            debug_dump_path = self._dump_policy_chunk_trajectory(
                kind="plan_step",
                obs=obs,
                actions_raw=actions,
                actions_final=None,
                action_debug=None,
                vis_frames=vis_frames,
                context_rgb=context_rgb_for_vis,
                info={
                    "context_update_mode": "step",
                    "planner_sampling_path": planner_sampling_path,
                    "text_conditioning": text_conditioning,
                    "text_conditioning_enabled": text_conditioning_enabled,
                    "chunk_story_index": int(planned_dream_index)
                    if planned_dream_index is not None
                    else 0,
                    "n_exec": int(n_exec),
                },
            )
            self._current_dream_index = planned_dream_index
            self._next_dream_index += 1
            for t in range(actions.shape[1]):
                self._action_queue.append(actions[:, t])
            for t, frame in enumerate(vis_frames):
                frame["context_rgb"] = context_rgb_for_vis
                frame["is_boundary_source_frame"] = bool(t == 0)
                frame["source_frame_role"] = "boundary" if t == 0 else "dream"
                frame["target_frame_role"] = "dream"
                self._vis_queue.append(frame)

        du = self._action_queue.popleft()
        runtime_controller = self._runtime_controller_params()
        du_pre_clip = self._apply_runtime_action_scaling(du, runtime_controller)
        du = du_pre_clip.clamp(
            -self.cfg.controller.clip_du, self.cfg.controller.clip_du
        )
        # Subclasses (e.g. gripper gating) can transform the action; use final action for vis and return.
        du_final = self.get_final_action(du)
        action_debug = self._build_action_debug_info(du_pre_clip, du, du_final)
        policy_debug = self._collect_policy_debug_info(action_debug)
        self._build_action_feedback_payload(obs, action_debug)

        policy_vis = None
        track_stats = None
        control_track_stats = None
        outlier_stats = None
        desired_flow_numpy = None
        desired_source_rgb_numpy = None
        frame = None
        if len(self._vis_queue) > 0 and obs.rgb_vis is not None:
            frame = self._vis_queue.popleft()
            track_stats = frame.get("track_stats")
            control_track_stats = frame.get("control_track_stats")
            outlier_stats = frame.get("outlier_stats")
            # Surface the predicted flow + paired source frame so the viewer
            # can render it next to an achieved-flow estimate. Shape: [2,H,W]
            # or [H,W,2] depending on path (flow vs track). Numpy + CPU.
            flow_t = frame.get("flow")
            if flow_t is not None:
                try:
                    f = flow_t.detach()
                    if f.ndim == 4:  # [B, 2, H, W] or [B, H, W, 2]
                        f = f[0]
                    desired_flow_numpy = f.cpu().float().numpy()
                except Exception:
                    desired_flow_numpy = None
            src_rgb = frame.get("rgb")
            if src_rgb is not None:
                try:
                    s = src_rgb.detach()
                    if s.ndim == 4:
                        s = s[0]
                    desired_source_rgb_numpy = s.cpu().float().numpy()
                except Exception:
                    desired_source_rgb_numpy = None
            policy_vis = self._make_policy_vis_joint(
                obs,
                [frame],
                dream_index=self._current_dream_index,
                action=du_final.cpu().numpy(),
                action_debug=cast(
                    dict[str, Any] | None, policy_debug.get("action_debug")
                ),
                gripper_debug=cast(
                    dict[str, Any] | None, policy_debug.get("gripper_debug")
                ),
            )

        info = {
            "action_chunk_remaining": len(self._action_queue),
            "control_view_keys": self.cfg.control_view_keys,
            "planner_sampling_path": planner_sampling_path,
            "text_conditioning": text_conditioning,
            "text_conditioning_enabled": text_conditioning_enabled,
            "track_stats": track_stats,
            "control_track_stats": control_track_stats,
            "outlier_stats": outlier_stats,
            "jacobian_abs_stats": self._jacobian_abs_stats_for_vis_frames(
                [frame] if frame is not None else None
            ),
            "policy_vis": policy_vis,
            "desired_flow": desired_flow_numpy,
            "desired_source_rgb": desired_source_rgb_numpy,
            "chunk_story_context_rgb": chunk_story_context_rgb,
            "chunk_story_dream_vis": chunk_story_dream_vis,
            "chunk_story_exec_horizon": chunk_story_exec_horizon,
            "chunk_story_index": chunk_story_index,
            "dream_rollout": dream_rollout,
            "debug_dump_path": debug_dump_path,
        }
        # When --save-artifacts is on, expose raw tensors for paper-figure
        # replay. Joint path version: includes dream RGB + tracks + flow.
        if getattr(self.cfg, "save_artifacts", False):
            ra = {
                "du_pre_clip": du_pre_clip.detach().cpu().numpy(),
                "du_final": du_final.detach().cpu().numpy(),
            }
            # Per-step "frame" dict from vis_queue holds tracks/jacobian/flow
            try:
                if 'frame' in dir() and isinstance(frame, dict):
                    for k in ("rgb", "flow", "jacobian", "curr_track", "trgt_track",
                              "tracks_curr", "tracks_target",
                              "control_track", "control_visible",
                              "context_rgb"):
                        if k in frame:
                            v = frame[k]
                            if isinstance(v, torch.Tensor):
                                v = v.detach().cpu().numpy()
                            ra[k] = v
            except Exception:
                pass
            if desired_flow_numpy is not None:
                ra["desired_flow"] = desired_flow_numpy
            if desired_source_rgb_numpy is not None:
                ra["desired_source_rgb"] = desired_source_rgb_numpy
            if chunk_story_dream_vis is not None:
                ra["chunk_dream_vis"] = (
                    chunk_story_dream_vis.detach().cpu().numpy()
                    if isinstance(chunk_story_dream_vis, torch.Tensor)
                    else np.asarray(chunk_story_dream_vis)
                )
            if chunk_story_context_rgb is not None:
                ra["chunk_context_rgb"] = (
                    chunk_story_context_rgb.detach().cpu().numpy()
                    if isinstance(chunk_story_context_rgb, torch.Tensor)
                    else np.asarray(chunk_story_context_rgb)
                )
            info["raw_artifacts"] = ra
        info.update(policy_debug)
        adaptive_debug = self._adaptive_debug_info()
        if adaptive_debug is not None:
            info["adaptive_controller"] = adaptive_debug
        return PolicyOutput(
            action=du_final.cpu().numpy(),
            info=info,
        )

    def _flow_rgb_chunk_to_actions(
        self,
        obs: PolicyObservation,
        source_rgb: Tensor,
        target_rgb: Tensor,
        flow: Tensor,
        source_view_widths: list[int],
        lam_override: float | None = None,
    ) -> Tuple[Tensor, list[Dict[str, Any]]]:
        """Compute actions from RGB+flow chunk. Returns (actions, vis_frames).
        vis_frames: list of dicts per timestep with rgb, flow, jacobian, curr_track, trgt_track.
        """
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
            weights_solve = control_mask.unsqueeze(0).expand(y.shape[0], -1)
            J_s, y_s, weights_solve = self._preprocess_solve_inputs(
                J, y, weights_solve,
                path_kind="flow",
                target_view_widths=target_view_widths,
            )
            du, _ = self.controller.solve(
                J_s,
                y_s,
                weights=weights_solve,
                lam_override=lam_override,
            )
            actions.append(du)
            vis_frames.append(
                {
                    "rgb": source_rgb_t,
                    "target_rgb": target_rgb[:, t].detach().cpu(),
                    "flow": flow_rs,
                    "jacobian": J,
                    "curr_track": curr,
                    "trgt_track": trgt,
                    "target_view_widths": target_view_widths,
                }
            )
        return torch.stack(actions, dim=1), vis_frames

    @staticmethod
    def _xy_to_linear_index(xy: Tensor, height: int, width: int) -> Tensor:
        xy_round = torch.round(xy).long()
        x = xy_round[..., 0].clamp(0, width - 1)
        y = xy_round[..., 1].clamp(0, height - 1)
        return y * width + x

    @staticmethod
    def _summarize_displacements(
        disp: Tensor,
        valid: Tensor | None = None,
    ) -> dict[str, float]:
        disp_flat = disp.detach().reshape(-1, disp.shape[-1]).float()
        total_count = int(disp_flat.shape[0])
        if valid is not None:
            valid_mask = valid.detach().reshape(-1) > 0.5
            valid_count = int(valid_mask.sum().item())
            disp_sel = disp_flat[valid_mask] if valid_count > 0 else None
        else:
            valid_count = total_count
            disp_sel = disp_flat

        if disp_sel is None or disp_sel.numel() == 0:
            return {
                "count_total": float(total_count),
                "count_selected": float(valid_count),
                "disp_x_min": float("nan"),
                "disp_x_max": float("nan"),
                "disp_y_min": float("nan"),
                "disp_y_max": float("nan"),
                "disp_norm_min": float("nan"),
                "disp_norm_max": float("nan"),
                "disp_norm_mean": float("nan"),
                "disp_norm_p95": float("nan"),
            }

        disp_x = disp_sel[:, 0]
        disp_y = disp_sel[:, 1]
        disp_norm = disp_sel.norm(dim=-1)
        return {
            "count_total": float(total_count),
            "count_selected": float(valid_count),
            "disp_x_min": float(disp_x.min().item()),
            "disp_x_max": float(disp_x.max().item()),
            "disp_y_min": float(disp_y.min().item()),
            "disp_y_max": float(disp_y.max().item()),
            "disp_norm_min": float(disp_norm.min().item()),
            "disp_norm_max": float(disp_norm.max().item()),
            "disp_norm_mean": float(disp_norm.mean().item()),
            "disp_norm_p95": float(torch.quantile(disp_norm, 0.95).item()),
        }

    def _track_outlier_mask(
        self,
        disp: Tensor,
        target_view_widths: list[int],
    ) -> Tensor:
        max_view_width = (
            float(max(target_view_widths))
            if target_view_widths
            else float(disp.shape[-2] if disp.ndim >= 2 else 1)
        )
        dx_limit = (
            float(self.cfg.max_track_abs_dx)
            if self.cfg.max_track_abs_dx is not None
            else max(max_view_width - 1.0, 1.0)
        )
        norm_limit = (
            float(self.cfg.max_track_disp_norm)
            if self.cfg.max_track_disp_norm is not None
            else max(max_view_width - 1.0, 1.0)
        )
        disp_norm = disp.norm(dim=-1)
        return ((disp[..., 0].abs() >= dx_limit) | (disp_norm >= norm_limit)).float()

    def _resize_track_frame(
        self,
        tracks: Dict[str, Any],
        timestep: int,
        target_size: Tuple[int, int],
        source_view_widths: list[int] | None = None,
        target_view_widths: list[int] | None = None,
    ) -> Dict[str, Tensor]:
        height, width = target_size
        source_h, source_w = tracks.get("image_size", (height, width))
        scale_y = float(height) / float(source_h)

        xy_src = tracks["xy_src"][:, timestep].to(self.device).float()
        disp = tracks["disp"][:, timestep].to(self.device).float()
        valid = tracks["valid"][:, timestep].to(self.device).float()
        finite_tracks = (
            torch.isfinite(xy_src).all(dim=-1)
            & torch.isfinite(disp).all(dim=-1)
            & torch.isfinite(valid)
        )
        xy_src = torch.nan_to_num(
            xy_src, nan=0.0, posinf=1.0e6, neginf=-1.0e6
        ).clamp(-1.0e6, 1.0e6)
        disp = torch.nan_to_num(
            disp, nan=0.0, posinf=1.0e6, neginf=-1.0e6
        ).clamp(-1.0e6, 1.0e6)
        valid = torch.nan_to_num(valid, nan=0.0, posinf=0.0, neginf=0.0)
        valid = valid * finite_tracks.float()
        xy_tgt = xy_src + disp

        if (
            source_view_widths is not None
            and target_view_widths is not None
            and len(source_view_widths) > 1
        ):
            xy_src_resized = xy_src.clone()
            xy_src_resized[..., 0] = self._map_x_between_view_layouts(
                xy_src[..., 0],
                source_view_widths,
                target_view_widths,
            )
            xy_src_resized[..., 1] *= scale_y

            xy_tgt_resized = xy_tgt.clone()
            xy_tgt_resized[..., 0] = self._map_x_between_view_layouts(
                xy_tgt[..., 0],
                source_view_widths,
                target_view_widths,
            )
            xy_tgt_resized[..., 1] *= scale_y
            disp_resized = xy_tgt_resized - xy_src_resized
        else:
            scale_x = float(width) / float(source_w)
            xy_src_resized = xy_src.clone()
            xy_src_resized[..., 0] *= scale_x
            xy_src_resized[..., 1] *= scale_y

            disp_resized = disp.clone()
            disp_resized[..., 0] *= scale_x
            disp_resized[..., 1] *= scale_y
            xy_tgt_resized = xy_src_resized + disp_resized

        in_bounds_src = (
            (xy_src_resized[..., 0] >= 0)
            & (xy_src_resized[..., 0] <= (width - 1))
            & (xy_src_resized[..., 1] >= 0)
            & (xy_src_resized[..., 1] <= (height - 1))
        )
        in_bounds_tgt = (
            (xy_tgt_resized[..., 0] >= 0)
            & (xy_tgt_resized[..., 0] <= (width - 1))
            & (xy_tgt_resized[..., 1] >= 0)
            & (xy_tgt_resized[..., 1] <= (height - 1))
        )
        valid = valid * (in_bounds_src & in_bounds_tgt).float()

        xy_min = torch.tensor([0.0, 0.0], device=self.device)
        xy_max = torch.tensor([width - 1.0, height - 1.0], device=self.device)
        xy_src_resized = xy_src_resized.clamp(min=xy_min, max=xy_max)
        xy_tgt_resized = xy_tgt_resized.clamp(min=xy_min, max=xy_max)

        return {
            "xy_src": xy_src_resized,
            "xy_tgt": xy_tgt_resized,
            "disp": disp_resized,
            "valid": valid,
            "idx_src": self._xy_to_linear_index(xy_src_resized, height, width),
        }

    def _track_rgb_chunk_to_actions(
        self,
        obs: PolicyObservation,
        source_rgb: Tensor,
        tracks: Dict[str, Any],
        source_view_widths: list[int],
        target_rgb: Tensor | None = None,
        lam_override: float | None = None,
    ) -> Tuple[Tensor, list[Dict[str, Any]]]:
        """Compute actions from RGB plus sparse AllTracker displacements."""
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

            track_frame = self._resize_track_frame(
                tracks,
                t,
                target_size,
                source_view_widths=source_view_widths,
                target_view_widths=target_view_widths,
            )
            idx_src = (
                track_frame["idx_src"]
                .long()
                .clamp(0, target_size[0] * target_size[1] - 1)
            )
            gather_idx = (
                idx_src.unsqueeze(-1)
                .unsqueeze(-1)
                .expand(-1, -1, 2, jacobian_pixel.shape[-1])
            )
            jacobian_sparse = torch.gather(jacobian_pixel, dim=1, index=gather_idx)
            jacobian_sparse = rearrange(jacobian_sparse, "b n s c -> b (n s) c")

            raw_disp = track_frame["disp"]
            disp = raw_disp * self.cfg.motion_plan_scale
            motion_plan = rearrange(disp, "b n s -> b (n s)")
            control_track_mask = self._control_track_mask(
                obs,
                track_frame["xy_src"][..., 0],
                target_view_widths,
            )
            # Reject implausible tracker outputs before applying any user-tuned
            # motion scaling so scaling does not change the outlier heuristic.
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

            J_s, y_s, weights_s = self._preprocess_solve_inputs(
                jacobian_sparse, motion_plan, weights,
                path_kind="track",
                target_view_widths=target_view_widths,
                track_xs=track_frame["xy_src"][..., 0],
            )
            du, _ = self.controller.solve(
                J_s,
                y_s,
                weights_s,
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
                    "jacobian": jacobian_flat,
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
        return torch.stack(actions, dim=1), vis_frames

    def _flow_to_action(self, obs: PolicyObservation, flow: Tensor) -> PolicyOutput:
        source_view_widths = self._source_view_widths_for_obs(obs, int(flow.shape[-1]))
        policy_outputs = self._post_process_flow(flow, source_view_widths)
        policy_outputs["context_rgb"] = torch.stack(
            list(self._queues["observation.images"]), dim=1
        )
        motion_plan = policy_outputs["motion_plan"].reshape(flow.shape[0], -1)
        J = self.compute_jacobian(obs)
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
                action_debug=cast(
                    dict[str, Any] | None, policy_debug.get("action_debug")
                ),
                gripper_debug=cast(
                    dict[str, Any] | None, policy_debug.get("gripper_debug")
                ),
            )
        info = {
            "controller_metrics": ctrl_metrics,
            "motion_plan": motion_plan,
            "flow_stats": flow_stats,
            "control_view_keys": self.cfg.control_view_keys,
            "policy_vis": policy_vis,
        }
        # When --save-artifacts is on, expose raw tensors for paper-figure
        # replay. Each is detached + cpu'd for safe pickling.
        if getattr(self.cfg, "save_artifacts", False):
            info["raw_artifacts"] = {
                "jacobian": J.detach().cpu().numpy(),
                "motion_plan_flow": motion_plan.detach().cpu().numpy(),
                "flow_pred": flow.detach().cpu().numpy(),
                "context_rgb": policy_outputs["context_rgb"].detach().cpu().numpy(),
                "du_pre_clip": du_pre_clip.detach().cpu().numpy(),
                "du_final": du_final.detach().cpu().numpy(),
            }
            # Pull whatever motion-track / dream entries the planner produced.
            # _post_process_flow puts curr_track/trgt_track; joint path uses
            # tracks_curr/tracks_target/dream_rgb. Save all that exist.
            for k in (
                "curr_track", "trgt_track",
                "tracks_curr", "tracks_target",
                "motion_tracks", "control_track", "control_visible",
                "dream_rgb", "dream_frames", "rgb",
            ):
                if k in policy_outputs:
                    v = policy_outputs[k]
                    if isinstance(v, torch.Tensor):
                        v = v.detach().cpu().numpy()
                    info["raw_artifacts"][k] = v
        info.update(policy_debug)
        adaptive_debug = self._adaptive_debug_info()
        if adaptive_debug is not None:
            info["adaptive_controller"] = adaptive_debug
        return PolicyOutput(
            action=du_final.cpu().numpy(),
            info=info,
        )

    # ------------------------- flow utils -------------------------

    def _post_process_flow(
        self,
        flow: Tensor,
        source_view_widths: list[int] | None = None,
    ) -> Dict[str, Tensor]:
        if source_view_widths is None:
            source_view_widths = [int(flow.shape[-1])]
        target_view_widths = self._dynamics_view_widths(len(source_view_widths))
        res = self._concat_dynamics_size(len(source_view_widths))
        flow_rs = self._resize_multiview_flow(
            flow, source_view_widths, target_view_widths
        )
        flow_flat = rearrange(flow_rs, "b c h w -> b (h w) c")
        curr = alltracker.gridcloud2d(
            flow.shape[0], res[0], res[1], norm=False, device=flow.device
        )
        return {
            "motion_plan": flow_flat,
            "curr_track": curr,
            "trgt_track": curr + flow_flat,
        }
