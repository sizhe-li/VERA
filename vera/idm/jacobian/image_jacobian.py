import math
import json
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, cast

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from einops import rearrange
from jaxtyping import Float, Int
from lightning.pytorch.utilities.types import STEP_OUTPUT
from vera.idm.common.base_pytorch_algo import BasePytorchAlgo, BasePytorchAlgoCfg
from vera.idm.jacobian.models.base import (
    InputCommand,
    InputObservation,
    JacobianFieldInterface,
    JacobianFieldOutput,
)
from omegaconf import DictConfig, OmegaConf

from vera.idm.jacobian.models.registry import (
    resolve_model_cfg,
    resolve_model_instance,
)
from vera.idm.registry import register_algorithm
from vera.datasets.normalization import (
    denormalize_flow_tensor,
    denormalize_jacobian_tensor,
    get_action_normalization_metadata,
    get_flow_normalization_metadata,
    get_jacobian_action_scale_tensor,
)
from vera.utils import jacobian_utils
from vera.utils.logging_utils import get_sanity_metrics, safe_asdict
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torchvision.utils import flow_to_image


#  utility
def flatten_time_dim(x: Tensor) -> Tensor:
    """[B, T, ...] -> [B*T, ...]"""
    return rearrange(x, "b t ... -> (b t) ...")


@torch.no_grad()
def flow_to_image_with_denom(
    flow: Tensor,
    denom: float,
) -> Tensor:
    """
    HSV color encoding of an optical flow tensor with an EXPLICIT magnitude
    denominator. `flow` shape: (2, H, W) or (N, 2, H, W) float. Returns uint8
    image, shape (3, H, W) or (N, 3, H, W) — matches `torchvision.utils.flow_to_image`'s
    output layout.

    Why: torchvision's `flow_to_image` divides by the per-tensor max-norm,
    which when called per-frame amplifies sub-pixel noise in low-motion
    frames into vivid color artifacts. Passing a constant episode-wide denom
    (e.g., the max-norm over the full episode) keeps low-motion frames dark
    and only "hot" frames look bright.
    """
    if flow.dtype != torch.float:
        flow = flow.float()
    orig_shape = flow.shape
    if flow.ndim == 3:
        flow = flow[None]
    if flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError(f"Expected (2,H,W) or (N,2,H,W), got {tuple(orig_shape)}")
    eps = torch.finfo(flow.dtype).eps
    mag = torch.sqrt((flow ** 2).sum(dim=1)).clamp_min(0)  # (N, H, W)
    ang = torch.atan2(flow[:, 1], flow[:, 0])              # (N, H, W) in (-pi, pi]
    H_chan = ((ang + np.pi) / (2 * np.pi) * 179).clamp(0, 179).to(torch.uint8)
    S_chan = torch.full_like(H_chan, 255)
    V = torch.clamp(mag / max(float(denom), eps), 0.0, 1.0) * 255.0
    V = V.to(torch.uint8)
    hsv_np = torch.stack([H_chan, S_chan, V], dim=-1).cpu().numpy()  # (N, H, W, 3)
    out = np.empty_like(hsv_np)
    import cv2 as _cv2
    for i in range(out.shape[0]):
        out[i] = _cv2.cvtColor(hsv_np[i], _cv2.COLOR_HSV2RGB)
    img = torch.from_numpy(out).permute(0, 3, 1, 2).contiguous()  # (N, 3, H, W) uint8
    if len(orig_shape) == 3:
        img = img[0]
    return img


def flatten_time_view_dim(x: Tensor) -> Tensor:
    """[B, T, V, ...] -> [B*T*V, ...]"""
    return rearrange(x, "b t v ... -> (b t v) ...")


def _agent_debug_log(hypothesis_id: str, location: str, message: str, data: dict[str, Any]) -> None:
    # region agent log
    try:
        payload = {
            "sessionId": "cc3db4",
            "runId": "pre-fix",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open(
            "/path/to/repo/.cursor/debug-cc3db4.log",
            "a",
            encoding="utf-8",
        ) as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass
    # endregion


@dataclass
class CheckpointCfg:
    reset_optimizer: bool = False
    strict: bool = True


@dataclass
class LRSchedulerCfg:
    name: Literal["constant_with_warmup"]
    num_warmup_steps: int = 1000


@dataclass
class OptimizerCfg:
    name: Literal["adamw", "adam", "sgd"] = "adamw"

    lr: float = 1e-4
    weight_decay: float = 1e-3

    beta: Tuple[float, float] = (0.9, 0.99)
    lr_scheduler: LRSchedulerCfg = field(
        default_factory=lambda: LRSchedulerCfg(name="constant_with_warmup")
    )


@dataclass
class LoggingCfg:
    loss_freq: int = 100
    grad_norm_freq: Optional[int] = None
    deterministic: Optional[int] = None
    max_num_videos: int = 4
    max_validation_batches: int = 1
    max_validation_samples: int = 1
    max_validation_frames: Optional[int] = 4
    max_validation_views: Optional[int] = 1
    max_grad_stride_mismatch_names: int = 8


@dataclass
class ImageJacobianCfg(BasePytorchAlgoCfg):
    name: Literal["image_jacobian"]
    robot_name: str = ""

    compile: bool = False

    model: Any = field(default=None)
    optimizer: OptimizerCfg = field(default_factory=OptimizerCfg)
    logging: LoggingCfg = field(default_factory=LoggingCfg)
    checkpoint: CheckpointCfg = field(default_factory=CheckpointCfg)
    image_size: List[int] = field(default_factory=lambda: [252, 252])
    supervision: Literal["flow", "tracks", "flow+tracks"] = "tracks"
    flow_loss: Literal["mse", "charbonnier"] = "charbonnier"
    flow_charbonnier_eps: float = 1e-3
    motion_aware_flow_weighting: bool = False
    motion_aware_flow_threshold: float = 0.02
    motion_aware_moving_weight: float = 4.0
    view_flow_balance_mode: Literal["none", "active_pixels"] = "none"
    view_balance_min_active_ratio: float = 0.01
    view_balance_max_view_weight: float = 5.0
    view_balanced_flow_loss: bool = False
    log_flow_per_view: bool = False
    inverse_action_weight: float = 1e-3
    inverse_action_damping: float = 1e-2
    # pmap-style regularization (adapted from VGGT point-map loss). All OFF by default.
    flow_gradient_weight: float = 0.0
    jacobian_tv_weight: float = 0.0
    jacobian_tv_edge_aware: bool = True
    jacobian_tv_edge_beta: float = 10.0
    predict_uncertainty: bool = False
    pmap_uncertainty_weight: float = 0.0
    # Upper cap applied to the predicted confidence inside the aleatoric loss
    # (0 = disabled, preserving the original mimicgen-pmap behavior). Bounds the
    # `conf * residual` gradient multiplier on the flow head: on DROID (run
    # 7sv5x0ai) the residual stagnates ~0.5, yet -w*log(conf) kept inflating
    # confidence (mean 0.8 -> 53 by step 4k) until the loop went unstable and
    # NaN'd at step ~4.4k. mimicgen (h8wdb112) only stayed healthy because its
    # residual genuinely fell to ~0.02.
    pmap_confidence_clamp: float = 0.0


@register_algorithm("image_jacobian", cfg_cls=ImageJacobianCfg)
class ImageJacobian(BasePytorchAlgo):
    """
    An algorithm for training image jacobian models
    """

    cfg: ImageJacobianCfg
    model: JacobianFieldInterface

    def __init__(self, cfg: ImageJacobianCfg):
        super().__init__(cfg)
        self._train_start_time_s: float | None = None
        self._param_partition_logged = False
        self._current_batch_ready_time_s: float | None = None
        self._current_batch_idx: int | None = None
        self._current_batch_should_log = False
        self._last_batch_end_time_s: float | None = None
        self._last_batch_wait_s: float | None = None
        self._last_training_step_s: float | None = None
        self._last_training_logging_s: float | None = None
        self._last_batch_total_s: float | None = None
        self._before_backward_start_s: float | None = None
        self._last_backward_s: float | None = None
        self._last_optimizer_step_s: float | None = None
        self._last_post_training_step_s: float | None = None
        self._last_unattributed_step_s: float | None = None
        self._last_compute_timing: Dict[str, float] = {}
        self._last_grad_logging_s: float | None = None
        self._last_grad_stride_metrics: Dict[str, float] = {}
        self._reported_grad_stride_mismatch = False
        self._validation_start_time_s: float | None = None
        self._validation_forward_s = 0.0
        self._validation_visualization_s = 0.0
        self._validation_media_log_s = 0.0
        self._validation_batches_seen = 0
        self._validation_video_groups_logged = 0

    def _validation_namespace(self, dataloader_idx: int = 0) -> str:
        datamodule = getattr(getattr(self, "trainer", None), "datamodule", None)
        names = getattr(datamodule, "validation_dataloader_names", None)
        if names and 0 <= int(dataloader_idx) < len(names):
            return f"validation/{names[int(dataloader_idx)]}"
        return "validation"

    # ---------------------------------------------------------------------
    # Prepare Model, Optimizer, and Metrics
    # ---------------------------------------------------------------------

    def _build_model(self):
        # Allow TF32-backed matmuls on Ampere+ GPUs; this is a safe speed win for training.
        torch.set_float32_matmul_precision("high")
        # Backward compat: old checkpoints had image_size at algorithm level only.
        model_cfg_input = self.cfg.model
        if not hasattr(model_cfg_input, "__dataclass_fields__"):
            if isinstance(model_cfg_input, DictConfig):
                cfg_dict = OmegaConf.to_container(model_cfg_input, resolve=True)
            else:
                cfg_dict = dict(model_cfg_input)
            if "image_size" not in cfg_dict and getattr(self.cfg, "image_size", None) is not None:
                algo_sz = self.cfg.image_size
                cfg_dict["image_size"] = (
                    int(algo_sz[0]) if isinstance(algo_sz, (list, tuple)) else int(algo_sz)
                )
            model_cfg_input = cfg_dict
        model_cfg = resolve_model_cfg(model_cfg_input)
        model = resolve_model_instance(model_cfg)
        self.model = cast(
            JacobianFieldInterface,
            torch.compile(model, disable=not self.cfg.compile),
        )

    def configure_optimizers(self):
        if hasattr(self.model, "get_optimizer_param_groups"):
            transition_params = getattr(self.model, "get_optimizer_param_groups")(
                base_lr=self.cfg.lr
            )
        else:
            transition_params = list(self.model.parameters())

        optimizer_dynamics = torch.optim.AdamW(
            transition_params,
            lr=self.cfg.lr,
            weight_decay=self.cfg.optimizer.weight_decay,
            betas=self.cfg.optimizer.beta,
        )

        # lr_scheduler_config = {
        #     "scheduler": get_scheduler(
        #         optimizer=optimizer_dynamics,
        #         name=self.cfg.optimizer.lr_scheduler.name,
        #         num_warmup_steps=self.cfg.optimizer.lr_scheduler.num_warmup_steps,
        #     ),
        #     "interval": "step",
        #     "frequency": 1,
        # }

        return {
            "optimizer": optimizer_dynamics,
            # "lr_scheduler": lr_scheduler_config,
        }

    # ---------------------------------------------------------------------
    # Data Preprocessing
    # ---------------------------------------------------------------------

    def _get_action_scale_tensor(self, device: torch.device, cmd_dim: int) -> Tensor:
        return get_jacobian_action_scale_tensor(
            self.dataset_metadata,
            device=device,
            dtype=torch.float32,
            cmd_dim=cmd_dim,
        )

    def _get_zero_action_model_tensor(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        cmd_dim: int,
    ) -> Tensor:
        action_meta = get_action_normalization_metadata(self.dataset_metadata)
        zero = torch.zeros(cmd_dim, device=device, dtype=dtype)
        mode = str(action_meta.get("action_normalization_mode", "none"))
        if mode in {"none", "symmetric_percentile"}:
            return zero
        if (
            mode == "minmax"
            and action_meta.get("action_min") is not None
            and action_meta.get("action_max") is not None
        ):
            amin = torch.as_tensor(
                action_meta["action_min"], device=device, dtype=dtype
            )
            amax = torch.as_tensor(
                action_meta["action_max"], device=device, dtype=dtype
            )
            return ((zero - amin) / (amax - amin + 1e-8)) * 2.0 - 1.0
        return zero

    def _get_zero_flow_model_tensor(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        spatial_dim: int,
    ) -> Tensor:
        flow_meta = get_flow_normalization_metadata(self.dataset_metadata)
        zero = torch.zeros(spatial_dim, device=device, dtype=dtype)
        mode = str(flow_meta.get("flow_normalization_mode", "scale"))
        if mode in {"scale", "symmetric_percentile"}:
            return zero
        if (
            mode == "percentile_minmax"
            and flow_meta.get("oflow_percentile_min") is not None
            and flow_meta.get("oflow_percentile_max") is not None
        ):
            fmin = torch.as_tensor(
                flow_meta["oflow_percentile_min"], device=device, dtype=dtype
            )
            fmax = torch.as_tensor(
                flow_meta["oflow_percentile_max"], device=device, dtype=dtype
            )
            return ((zero - fmin) / (fmax - fmin + 1e-8)) * 2.0 - 1.0
        return zero

    # def on_after_batch_transfer(
    #     self, batch: Dict, dataloader_idx: int
    # ) -> Tuple[Tensor, Tensor, Optional[Tensor], Tensor, Optional[Tensor]]:
    #    """Flatten time dimension and move to device."""
    #   pass

    # ---------------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------------

    def on_train_start(self) -> None:
        self._train_start_time_s = time.perf_counter()
        self._param_partition_logged = False

    @staticmethod
    def _sanitize_metric_key(name: str) -> str:
        return name.replace(".", "/")

    def _log_scalar_group(
        self,
        prefix: str,
        metrics: Dict[str, float | int | Tensor],
        *,
        on_step: bool,
        on_epoch: bool,
    ) -> None:
        for key, value in metrics.items():
            if isinstance(value, Tensor):
                if value.numel() != 1:
                    continue
                value = value.detach()
                if not torch.isfinite(value):
                    continue
            elif isinstance(value, float) and not math.isfinite(value):
                continue
            self.log(
                f"{prefix}{self._sanitize_metric_key(key)}",
                value,
                on_step=on_step,
                on_epoch=on_epoch,
                sync_dist=True,
            )

    def _log_sanity_metric_dict(self, prefix: str, metrics: Dict[str, float | int]) -> None:
        self._log_scalar_group(
            prefix,
            metrics,
            on_step=True,
            on_epoch=False,
        )

    def _log_training_speed(self) -> None:
        if self._train_start_time_s is None:
            self._train_start_time_s = time.perf_counter()

        elapsed_hours = max((time.perf_counter() - self._train_start_time_s) / 3600.0, 1e-9)
        world_size = max(int(getattr(self.trainer, "world_size", 1)), 1)
        gpu_hours = elapsed_hours * float(world_size)
        completed_steps = float(self.global_step + 1)

        perf_metrics = {
            "wall_hours": elapsed_hours,
            "gpu_hours": gpu_hours,
            "steps_per_wall_hour": completed_steps / elapsed_hours,
            "steps_per_gpu_hour": completed_steps / gpu_hours,
        }
        self._log_scalar_group(
            "perf/training/",
            perf_metrics,
            on_step=True,
            on_epoch=False,
        )

    def _log_timing_metrics(self, namespace: str) -> None:
        timing_metrics: Dict[str, float] = {}
        if self._last_batch_wait_s is not None:
            timing_metrics["batch_wait_s"] = self._last_batch_wait_s
        if self._last_training_step_s is not None:
            timing_metrics["training_step_s"] = self._last_training_step_s
        if self._last_training_logging_s is not None:
            timing_metrics["training_logging_s"] = self._last_training_logging_s
        if self._last_batch_total_s is not None:
            timing_metrics["batch_total_s"] = self._last_batch_total_s
        if self._last_backward_s is not None:
            timing_metrics["backward_s"] = self._last_backward_s
        if self._last_optimizer_step_s is not None:
            timing_metrics["optimizer_step_s"] = self._last_optimizer_step_s
        if self._last_post_training_step_s is not None:
            timing_metrics["post_training_step_s"] = self._last_post_training_step_s
        if self._last_unattributed_step_s is not None:
            timing_metrics["unattributed_step_s"] = self._last_unattributed_step_s
        if self._last_grad_logging_s is not None:
            timing_metrics["grad_logging_s"] = self._last_grad_logging_s
        timing_metrics.update(self._last_compute_timing)
        timing_metrics.update(self._last_grad_stride_metrics)
        for key, value in timing_metrics.items():
            self.log(
                f"perf/{namespace}/{self._sanitize_metric_key(key)}",
                value,
                on_step=True,
                on_epoch=False,
                sync_dist=False,
            )

    def _log_parameter_partition_summary(self) -> None:
        if self._param_partition_logged:
            return

        active_tensors = 0
        frozen_tensors = 0
        active_elements = 0
        frozen_elements = 0
        for _, param in self.model.named_parameters():
            n = int(param.numel())
            if param.requires_grad:
                active_tensors += 1
                active_elements += n
            else:
                frozen_tensors += 1
                frozen_elements += n

        self._log_scalar_group(
            "params/",
            {
                "active_tensors": active_tensors,
                "frozen_tensors": frozen_tensors,
                "active_elements": active_elements,
                "frozen_elements": frozen_elements,
            },
            on_step=True,
            on_epoch=False,
        )
        self._param_partition_logged = True

    def _log_gradient_groups(self) -> None:
        grad_metrics: Dict[str, float] = {}
        total_sq = 0.0
        active_with_grad = 0
        active_missing_grad = 0
        frozen_with_grad = 0

        for name, param in self.model.named_parameters():
            prefix = "active" if param.requires_grad else "frozen"
            if param.grad is None:
                if param.requires_grad:
                    active_missing_grad += 1
                continue
            grad = param.grad.detach()
            if not torch.isfinite(grad).all():
                continue
            norm_val = float(torch.linalg.vector_norm(grad).item())
            grad_metrics[f"{prefix}/{name}"] = norm_val
            total_sq += norm_val * norm_val
            if param.requires_grad:
                active_with_grad += 1
            else:
                frozen_with_grad += 1

        grad_metrics["summary/active_with_grad_tensors"] = float(active_with_grad)
        grad_metrics["summary/active_missing_grad_tensors"] = float(active_missing_grad)
        grad_metrics["summary/frozen_with_grad_tensors"] = float(frozen_with_grad)
        grad_metrics["summary/total"] = math.sqrt(total_sq)

        self._log_scalar_group(
            "grad_norm/",
            grad_metrics,
            on_step=True,
            on_epoch=False,
        )

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        now = time.perf_counter()
        self._current_batch_ready_time_s = now
        self._current_batch_idx = int(batch_idx)
        self._current_batch_should_log = batch_idx % self.cfg.logging.loss_freq == 0
        self._last_backward_s = None
        self._last_optimizer_step_s = None
        self._last_post_training_step_s = None
        self._last_unattributed_step_s = None
        self._last_grad_stride_metrics = {}
        if self._last_batch_end_time_s is not None:
            self._last_batch_wait_s = now - self._last_batch_end_time_s

    def on_train_batch_end(
        self,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        end_t = time.perf_counter()
        if self._current_batch_ready_time_s is not None:
            self._last_batch_total_s = end_t - self._current_batch_ready_time_s
        if self._last_batch_total_s is not None and self._last_training_step_s is not None:
            post_training_step_s = self._last_batch_total_s - self._last_training_step_s
            self._last_post_training_step_s = post_training_step_s
            unattributed = post_training_step_s
            for value in (
                self._last_training_logging_s,
                self._last_backward_s,
                self._last_optimizer_step_s,
                self._last_grad_logging_s,
            ):
                if value is not None:
                    unattributed -= value
            self._last_unattributed_step_s = unattributed
        self._last_batch_end_time_s = end_t
        if self._current_batch_should_log:
            self._log_timing_metrics("training")
        super().on_train_batch_end(outputs, batch, batch_idx)

    def _collect_grad_stride_metrics(self) -> Dict[str, float]:
        mismatched_tensors = 0
        mismatched_elements = 0
        channels_last_grad_tensors = 0
        mismatched_names: list[str] = []

        for name, param in self.model.named_parameters():
            grad = param.grad
            if grad is None:
                continue
            if tuple(grad.stride()) == tuple(param.stride()):
                continue

            mismatched_tensors += 1
            mismatched_elements += int(param.numel())
            if grad.ndim == 4 and grad.is_contiguous(memory_format=torch.channels_last):
                channels_last_grad_tensors += 1
            if len(mismatched_names) < int(self.cfg.logging.max_grad_stride_mismatch_names):
                mismatched_names.append(
                    f"{name}: param_stride={tuple(param.stride())}, grad_stride={tuple(grad.stride())}"
                )

        if mismatched_names and not self._reported_grad_stride_mismatch:
            print(
                "[ImageJacobian] Grad stride mismatches detected:\n  "
                + "\n  ".join(mismatched_names)
            )
            self._reported_grad_stride_mismatch = True

        return {
            "grad_stride_mismatch_tensors": float(mismatched_tensors),
            "grad_stride_mismatch_elements": float(mismatched_elements),
            "grad_stride_mismatch_channels_last_tensors": float(
                channels_last_grad_tensors
            ),
        }

    def _model_supports_joint_multiview(self) -> bool:
        return bool(getattr(self.model, "supports_joint_multiview", False))

    @staticmethod
    def _prepare_model_inputs(
        *,
        rgb: Tensor,
        du: Tensor,
        flow_gt: Optional[Tensor],
        joint_multiview: bool,
    ) -> tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor], Tensor, Optional[Tensor]]:
        flow_gt_btv: Optional[Tensor] = None

        if rgb.ndim == 6:
            bsz, tlen, num_views, _, _, _ = rgb.shape
            if joint_multiview:
                model_rgb = rearrange(rgb, "b t v c h w -> (b t) v c h w")
                model_du = flatten_time_dim(du)
                loss_du = du[:, :, None, :].expand(-1, -1, num_views, -1)
                loss_du = flatten_time_view_dim(loss_du)
                view_ids = torch.arange(num_views, device=rgb.device, dtype=torch.long)
                view_ids = view_ids.view(1, 1, num_views).expand(bsz, tlen, num_views)
                view_ids = rearrange(view_ids, "b t v -> (b t) v")
                if flow_gt is not None:
                    flow_gt_btv = flow_gt
                    flow_gt = flatten_time_view_dim(flow_gt)
                return model_rgb, model_du, flow_gt, flow_gt_btv, loss_du, view_ids

            model_du = du[:, :, None, :].expand(-1, -1, num_views, -1)
            model_du = flatten_time_view_dim(model_du)
            view_ids = torch.arange(num_views, device=rgb.device, dtype=torch.long)
            view_ids = view_ids.view(1, 1, num_views).expand(bsz, tlen, num_views)
            view_ids = flatten_time_view_dim(view_ids)
            model_rgb = flatten_time_view_dim(rgb)
            if flow_gt is not None:
                flow_gt_btv = flow_gt
                flow_gt = flatten_time_view_dim(flow_gt)
            return model_rgb, model_du, flow_gt, flow_gt_btv, model_du, view_ids

        model_du, model_rgb, flow_gt = [
            flatten_time_dim(x) if x is not None else None for x in (du, rgb, flow_gt)
        ]
        if model_rgb is None or model_du is None:
            raise ValueError("rgb and du must be present when preparing model inputs.")
        view_ids = torch.zeros(model_rgb.shape[0], device=model_rgb.device, dtype=torch.long)
        return model_rgb, model_du, flow_gt, None, model_du, view_ids

    @staticmethod
    def _flatten_joint_model_output(model_output: JacobianFieldOutput) -> JacobianFieldOutput:
        jacobian = model_output.jacobian
        optical_flow = model_output.optical_flow
        scene_flow = model_output.scene_flow

        if jacobian.ndim == 6:
            jacobian = rearrange(jacobian, "b v c s h w -> (b v) c s h w")
        if optical_flow.ndim == 5:
            optical_flow = rearrange(optical_flow, "b v s h w -> (b v) s h w")
        if scene_flow is not None and scene_flow.ndim == 5:
            scene_flow = rearrange(scene_flow, "b v s h w -> (b v) s h w")

        flow_confidence = model_output.flow_confidence
        if flow_confidence is not None and flow_confidence.ndim == 5:
            flow_confidence = rearrange(flow_confidence, "b v s h w -> (b v) s h w")

        return JacobianFieldOutput(
            jacobian=jacobian,
            optical_flow=optical_flow,
            scene_flow=scene_flow,
            flow_confidence=flow_confidence,
        )

    @staticmethod
    def _select_joint_view_output(
        model_output: JacobianFieldOutput,
        view_idx: int,
    ) -> JacobianFieldOutput:
        if model_output.jacobian.ndim != 6 or model_output.optical_flow.ndim != 5:
            raise ValueError("Expected joint multi-view model output.")
        scene_flow = None
        if model_output.scene_flow is not None:
            if model_output.scene_flow.ndim != 5:
                raise ValueError("Expected scene_flow to match joint multi-view layout.")
            scene_flow = model_output.scene_flow[:, view_idx]
        flow_confidence = None
        if model_output.flow_confidence is not None:
            flow_confidence = model_output.flow_confidence[:, view_idx]
        return JacobianFieldOutput(
            jacobian=model_output.jacobian[:, view_idx],
            optical_flow=model_output.optical_flow[:, view_idx],
            scene_flow=scene_flow,
            flow_confidence=flow_confidence,
        )

    def training_step(self, batch, batch_idx, namespace="training") -> STEP_OUTPUT:
        should_log = batch_idx % self.cfg.logging.loss_freq == 0
        step_t0 = time.perf_counter()
        (
            losses,
            loss_metrics,
            total_loss,
            model_output,
            diagnostics,
            compute_timing,
        ) = self._compute_loss(batch, namespace=namespace, collect_diagnostics=should_log)
        self._last_training_step_s = time.perf_counter() - step_t0
        self._last_compute_timing = compute_timing

        if should_log:
            log_t0 = time.perf_counter()
            self._log_losses(losses, loss_metrics, total_loss, diagnostics, namespace)
            self._log_training_speed()

            # sanity checks (unchanged)
            self._log_sanity_metric_dict("sanity/input/", get_sanity_metrics(batch))
            self._log_sanity_metric_dict(
                "sanity/output/", get_sanity_metrics(safe_asdict(model_output))
            )
            self._last_training_logging_s = time.perf_counter() - log_t0
        else:
            self._last_training_logging_s = None

        return {
            "loss": total_loss,
            "losses": losses,
        }

    def on_before_backward(self, loss: Tensor) -> None:
        self._before_backward_start_s = time.perf_counter()

    def on_after_backward(self) -> None:
        if self._before_backward_start_s is not None:
            self._last_backward_s = time.perf_counter() - self._before_backward_start_s
        else:
            self._last_backward_s = None
        self._last_grad_stride_metrics = self._collect_grad_stride_metrics()

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        if (
            self.cfg.logging.grad_norm_freq
            and self.global_step % self.cfg.logging.grad_norm_freq == 0
        ):
            t0 = time.perf_counter()
            self._log_parameter_partition_summary()
            self._log_gradient_groups()
            self._last_grad_logging_s = time.perf_counter() - t0
        else:
            self._last_grad_logging_s = None

    def optimizer_step(self, *args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        out = super().optimizer_step(*args, **kwargs)
        self._last_optimizer_step_s = time.perf_counter() - t0
        return out

    # ---------------------------------------------------------------------
    # Validation & Test
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def on_validation_start(self) -> None:
        self._validation_start_time_s = time.perf_counter()
        self._validation_forward_s = 0.0
        self._validation_visualization_s = 0.0
        self._validation_media_log_s = 0.0
        self._validation_batches_seen = 0
        self._validation_video_groups_logged = 0

    @torch.no_grad()
    def validation_step(
        self,
        batch,
        batch_idx,
        dataloader_idx: int = 0,
        namespace: str | None = None,
    ) -> STEP_OUTPUT:
        # if batch_idx > 0:
        #     return  # only support first batch for simplicity

        if namespace is None:
            namespace = self._validation_namespace(dataloader_idx)
            
        _agent_debug_log(
            "H1",
            "image_jacobian.py:validation_step:entry",
            "validation_step entered",
            {
                "batch_idx": int(batch_idx),
                "is_global_zero": bool(getattr(self.trainer, "is_global_zero", False)),
                "rank": int(dist.get_rank()) if dist.is_initialized() else 0,
                "max_validation_batches": int(self.cfg.logging.max_validation_batches),
                "max_num_videos": int(self.cfg.logging.max_num_videos),
            },
        )
        if batch_idx >= int(self.cfg.logging.max_validation_batches):
            # All ranks return together at the same batch_idx — consistent.
            return

        (
            losses,
            loss_metrics,
            total_loss,
            _,
            diagnostics,
            _,
        ) = self._compute_loss(
            batch,
            namespace=namespace,
            collect_diagnostics=True,
        )
        self._log_losses(
            losses,
            loss_metrics,
            total_loss,
            diagnostics,
            namespace=namespace,
        )

        # Past this point, only rank 0 emits viz/media (no sync_dist calls).
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        if not self.trainer.is_global_zero:
            return

        step_t0 = time.perf_counter()
        step_forward_s = 0.0
        step_visualization_s = 0.0
        step_media_log_s = 0.0
        du = batch["du"]  # B x T x nq
        rgb = batch["rgb"]  # B x T x 3 x H x W or B x T x V x 3 x H x W
        gt_flow = batch.get("flow")  # B x T x 2 x H x W or B x T x V x 2 x H x W
        tracks = batch.get("tracks")  # dict or None
        self._validation_batches_seen += 1

        if rgb.ndim == 6:
            B, T, V, _, H, W = rgb.shape
        else:
            B, T, _, H, W = rgb.shape
            V = 1

        meta = self.dataset_metadata or {}
        view_names = meta.get("views") or meta.get("camera_views")

        K = min(B, int(self.cfg.logging.max_validation_samples))
        max_frames = self.cfg.logging.max_validation_frames
        vis_frames = T if max_frames is None else min(T, int(max_frames))
        max_views = self.cfg.logging.max_validation_views
        view_count = V if max_views is None else min(V, int(max_views))
        max_video_groups = max(0, int(self.cfg.logging.max_num_videos))
        video_groups_logged = int(self._validation_video_groups_logged)
        joint_multiview = bool(V > 1 and self._model_supports_joint_multiview())

        for b in range(K):
            joint_output_bt: JacobianFieldOutput | None = None
            gt_flow_joint_bt = gt_flow[b] if (gt_flow is not None and joint_multiview) else None
            if joint_multiview:
                view_ids_bt = torch.arange(V, device=rgb.device, dtype=torch.long)
                view_ids_bt = view_ids_bt.view(1, V).expand(T, V)
                model_t0 = time.perf_counter()
                joint_output_bt = self.model(
                    InputObservation(rgb=rgb[b], view_ids=view_ids_bt),
                    InputCommand(du=du[b]),
                )
                model_elapsed = time.perf_counter() - model_t0
                self._validation_forward_s += model_elapsed
                step_forward_s += model_elapsed
            for v in range(view_count):
                if self._validation_video_groups_logged >= max_video_groups:
                    _agent_debug_log(
                        "H2",
                        "image_jacobian.py:validation_step:video_guard",
                        "validation_step early return due max videos",
                        {
                            "video_groups_logged": int(self._validation_video_groups_logged),
                            "max_video_groups": int(max_video_groups),
                            "batch_idx": int(batch_idx),
                        },
                    )
                    return
                view_name = (
                    view_names[v]
                    if isinstance(view_names, list) and v < len(view_names)
                    else f"view{v}"
                )
                vis_dict = {
                    "video/grid": [],
                }

                # Scaling: dataset feeds flow_scaled = oflow_scale * flow_physical; model predicts flow_scaled.
                # For visualization we convert to physical: flow_physical = flow_scaled / oflow_scale.
                # Jacobian: J_model = d(flow_scaled)/d(du_norm); after action_scale we have d(flow_scaled)/d(du_phys);
                # for physical pixel motion per du we divide by oflow_scale.
                robot_name = (
                    (self.dataset_metadata or {}).get("robot_name")
                    or self.cfg.robot_name
                    or "pusher"
                )

                # Fixed set of track indices from first frame for temporal continuity
                track_keep_indices: Optional[Tensor] = None
                if tracks is not None:
                    xy_src_0, _, _, _ = self._get_tracks_at_frame(tracks, b, v, 0, V)
                    H_vis, W_vis = rgb.shape[-2], rgb.shape[-1]
                    track_keep_indices = self._compute_track_keep_indices(
                        xy_src_0, H_vis, W_vis, sparsity=300
                    )

                if joint_multiview:
                    rgb_bt = rgb[b, :, v]
                    gt_flow_bt = gt_flow_joint_bt[:, v] if gt_flow_joint_bt is not None else None
                    if joint_output_bt is None:
                        raise RuntimeError("Joint VGGT validation output was not computed.")
                    out_bt = self._select_joint_view_output(joint_output_bt, v)
                else:
                    rgb_bt = rgb[b, :, v] if rgb.ndim == 6 else rgb[b]  # (T, 3, H, W)
                    gt_flow_bt = None
                    if gt_flow is not None:
                        gt_flow_bt = (
                            gt_flow[b, :, v] if gt_flow.ndim == 6 else gt_flow[b]
                        )
                    view_ids_bt = torch.full(
                        (rgb_bt.shape[0],),
                        fill_value=int(v if V > 1 else 0),
                        device=rgb_bt.device,
                        dtype=torch.long,
                    )
                    model_t0 = time.perf_counter()
                    out_bt = self.model(
                        InputObservation(rgb=rgb_bt, view_ids=view_ids_bt),
                        InputCommand(du=du[b]),
                    )
                    model_elapsed = time.perf_counter() - model_t0
                    self._validation_forward_s += model_elapsed
                    step_forward_s += model_elapsed

                # Precompute an episode-shared magnitude denominator for the
                # HSV flow viz so that low-motion frames render dark instead
                # of having sub-pixel noise amplified by a per-frame max.
                # We use the max-norm across pred AND gt (both denormalized
                # to physical pixel units) so the two viz panels are directly
                # visually comparable at the same color scale.
                pred_full_phys = denormalize_flow_tensor(
                    out_bt.optical_flow.float(), self.dataset_metadata
                )  # (T, 2, H, W)
                ep_denom = float(
                    torch.sqrt((pred_full_phys ** 2).sum(dim=1)).max().item()
                )
                if gt_flow_bt is not None:
                    gt_full_phys = denormalize_flow_tensor(
                        gt_flow_bt.float(), self.dataset_metadata
                    )
                    ep_denom = max(
                        ep_denom,
                        float(torch.sqrt((gt_full_phys ** 2).sum(dim=1)).max().item()),
                    )
                else:
                    gt_full_phys = None
                # Avoid divide-by-zero on dead-still episodes.
                ep_denom = max(ep_denom, 1e-3)

                vis_t0 = time.perf_counter()
                for t in range(vis_frames):
                    rgb_btv = rgb_bt[t]  # (3, H, W)
                    gt_flow_btv = gt_flow_bt[t] if gt_flow_bt is not None else None

                    # ----------------------------
                    # Dense flow (convert to physical pixel flow for viz)
                    # ----------------------------
                    pred_flow_phys = pred_full_phys[t]
                    vis_pred_flow = flow_to_image_with_denom(
                        pred_flow_phys, denom=ep_denom
                    ).cpu().numpy()
                    if gt_flow_btv is not None:
                        gt_flow_phys = gt_full_phys[t]
                        vis_gt_flow = flow_to_image_with_denom(
                            gt_flow_phys, denom=ep_denom
                        ).cpu().numpy()
                    else:
                        vis_gt_flow = np.zeros_like(vis_pred_flow)

                    # ----------------------------
                    # Jacobian (d(flow_physical)/d(du_phys) for arrow viz)
                    # Use a near-square sub-tile grid + resize so high-dim
                    # Jacobians (e.g. 16-DOF allegro hand) fit a single
                    # H×W slot in the composite layout below.
                    # ----------------------------
                    denorm_jacobian_phys = denormalize_jacobian_tensor(
                        out_bt.jacobian[t : t + 1],
                        self.dataset_metadata,
                        cmd_dim=out_bt.jacobian.shape[-4],
                    )
                    num_cmd_jac = denorm_jacobian_phys.shape[-4]
                    jac_grid_shape = self._near_square_grid(num_cmd_jac)
                    vis_jacobian = jacobian_utils.visualize_jacobian(
                        denorm_jacobian_phys,
                        robot_name=robot_name,
                        flow_scale=0.1,
                        grid_shape=jac_grid_shape,
                        target_hw=(H, W),
                    )

                    # ----------------------------
                    # Track visualization (fixed track set from t=0 for continuity)
                    # ----------------------------
                    if tracks is not None:
                        xy_src, pixel_selector, gt_disp, visibility = (
                            self._get_tracks_at_frame(tracks, b, v, t, V)
                        )
                        flow_pred = rearrange(
                            out_bt.optical_flow[t : t + 1], "b c h w -> b h w c"
                        )  # (1,H,W,2)
                        flow_flat = flow_pred.view(1, -1, 2)  # (1,HW,2)
                        flow_hw = flow_flat.shape[1]  # H*W
                        # Clamp indices so gather never goes out of bounds (avoids CUDA assert
                        # when track data was produced at different resolution or has bad indices).
                        pixel_selector_safe = pixel_selector.clamp(0, flow_hw - 1)
                        valid_mask = (
                            (pixel_selector >= 0) & (pixel_selector < flow_hw)
                        ).to(device=visibility.device, dtype=visibility.dtype)
                        visibility_safe = visibility * valid_mask
                        pred_disp = torch.gather(
                            flow_flat,
                            dim=1,
                            index=pixel_selector_safe[None, :, None]
                            .long()
                            .expand(1, -1, 2),
                        )[
                            0
                        ]  # (N,2)
                        track_vis = self.visualize_pixel_motion(
                            xy_src=xy_src,
                            pixel_selector=pixel_selector_safe,
                            gt_pixel_motion=gt_disp,
                            pred_pixel_motion=pred_disp,
                            gt_pixel_visibility=visibility_safe,
                            rgb=rgb_btv,
                            sparsity=300,
                            keep_indices=track_keep_indices,
                        )
                    else:
                        track_vis = np.zeros((3, H, W), dtype=np.uint8)

                    # ----------------------------
                    # 2x3 composite grid:
                    #   row 1: [RGB | Jacobian | Tracks]
                    #   row 2: [GT flow | Pred flow | |pred-gt|]
                    # All tiles share H×W; jacobian is pre-resized to (H, W)
                    # so the layout stays balanced for any cmd_dim.
                    # ----------------------------
                    rgb_np = (rgb_btv.cpu().numpy() * 255).astype(np.uint8)
                    flow_err = np.abs(
                        vis_pred_flow.astype(np.int16) - vis_gt_flow.astype(np.int16)
                    ).clip(0, 255).astype(np.uint8)

                    top_row = np.concatenate(
                        [
                            self._add_panel_label(rgb_np, "RGB"),
                            self._add_panel_label(vis_jacobian, f"Jacobian ({num_cmd_jac}d)"),
                            self._add_panel_label(track_vis, "Tracks (red:gt blue:pred)"),
                        ],
                        axis=2,
                    )
                    bottom_row = np.concatenate(
                        [
                            self._add_panel_label(vis_gt_flow, "Flow gt"),
                            self._add_panel_label(vis_pred_flow, "Flow pred"),
                            self._add_panel_label(flow_err, "Flow |pred-gt|"),
                        ],
                        axis=2,
                    )
                    vis_grid = np.concatenate([top_row, bottom_row], axis=1)
                    vis_dict["video/grid"].append(vis_grid)
                vis_elapsed = time.perf_counter() - vis_t0
                self._validation_visualization_s += vis_elapsed
                step_visualization_s += vis_elapsed

                # ----------------------------
                # Log one video per batch element / view
                # ----------------------------
                media_t0 = time.perf_counter()
                log_dict = {
                    f"{namespace}/{k}/b{b}/{view_name}": wandb.Video(
                        np.stack(vframes, axis=0),
                        fps=12,
                        format="mp4",
                        caption=(
                            "row1: RGB | Jacobian | Tracks  "
                            "row2: GT flow | Pred flow | |pred-gt|  "
                            "(tracks: red=gt, blue=pred)"
                        ),
                    )
                    for k, vframes in vis_dict.items()
                }

                self.logger.experiment.log(
                    {**log_dict, "trainer/global_step": self.global_step}
                )
                media_elapsed = time.perf_counter() - media_t0
                self._validation_media_log_s += media_elapsed
                step_media_log_s += media_elapsed
                video_groups_logged += 1
                self._validation_video_groups_logged += 1

        step_elapsed = time.perf_counter() - step_t0
        residual = (
            step_elapsed - step_forward_s - step_visualization_s - step_media_log_s
        )
        if residual > 0:
            self._validation_visualization_s += residual

    @torch.no_grad()
    def on_validation_end(self) -> None:
        if not getattr(self.trainer, "is_global_zero", False):
            return
        if self._validation_start_time_s is None or self.logger is None:
            return

        total_s = time.perf_counter() - self._validation_start_time_s
        self.logger.experiment.log(
            {
                "perf/validation/wall_s": total_s,
                "perf/validation/model_forward_s": self._validation_forward_s,
                "perf/validation/visualization_s": self._validation_visualization_s,
                "perf/validation/media_log_s": self._validation_media_log_s,
                "perf/validation/batches_seen": float(self._validation_batches_seen),
                "perf/validation/video_groups_logged": float(
                    self._validation_video_groups_logged
                ),
                "trainer/global_step": self.global_step,
            }
        )
        _agent_debug_log(
            "H3",
            "image_jacobian.py:on_validation_end",
            "validation_end perf logged",
            {
                "batches_seen": int(self._validation_batches_seen),
                "video_groups_logged": int(self._validation_video_groups_logged),
                "global_step": int(getattr(self, "global_step", -1)),
            },
        )

    @staticmethod
    def _add_panel_label(image_chw: np.ndarray, label: str) -> np.ndarray:
        # Add a 22-px black header strip ABOVE the panel and write the label
        # there. The previous implementation overwrote the top 24 px of the
        # panel itself, which hid the top sub-cells of high-DOF Jacobian grids
        # and the top portion of track-motion panels.
        c, h, w = image_chw.shape
        label_h = 22
        header = np.zeros((label_h, w, 3), dtype=np.uint8)  # HWC, black
        cv2.putText(
            header,
            label,
            (8, label_h - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        header_chw = header.transpose(2, 0, 1)
        return np.concatenate([header_chw, image_chw], axis=1)

    @staticmethod
    def _concat_side_by_side(left_chw: np.ndarray, right_chw: np.ndarray) -> np.ndarray:
        if left_chw.shape[1:] != right_chw.shape[1:]:
            raise ValueError(
                f"Expected matching spatial size, got {left_chw.shape} and {right_chw.shape}"
            )
        return np.concatenate([left_chw, right_chw], axis=2)

    @staticmethod
    def _stack_grid(top_chw: np.ndarray, bottom_chw: np.ndarray) -> np.ndarray:
        if top_chw.shape[2] != bottom_chw.shape[2]:
            raise ValueError(
                f"Expected matching widths, got {top_chw.shape} and {bottom_chw.shape}"
            )
        return np.concatenate([top_chw, bottom_chw], axis=1)

    @staticmethod
    def _near_square_grid(n: int) -> Tuple[int, int]:
        # Pick (grid_h, grid_w) close to square so per-channel jacobian tiles
        # don't degenerate into a 2×N strip for high-DOF robots (16-DOF allegro
        # → 4×4, 23-DOF drake → 5×5 with padding).
        if n <= 0:
            raise ValueError(f"_near_square_grid expected n>0, got {n}")
        grid_h = max(int(round(n ** 0.5)), 1)
        grid_w = (n + grid_h - 1) // grid_h
        return grid_h, grid_w

    @staticmethod
    def _get_tracks_at_frame(
        tracks: Any,
        b: int,
        v: int,
        t: int,
        V: int,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Return (xy_src, pixel_selector, gt_disp, visibility) for batch b, view v, time t.
        Handles list[dict] vs dict and variable ndim of track tensors.
        """
        if isinstance(tracks, list):
            tr = tracks[b]
            idx_src = tr["idx_src"]
            if idx_src.ndim == 3:  # (T,V,N)
                xy_src = tr["xy_src"][t, v]
                pixel_selector = tr["idx_src"][t, v]
                gt_disp = tr["disp"][t, v]
                visibility = tr["valid"][t, v]
            elif idx_src.ndim == 2:  # (T,N)
                xy_src = tr["xy_src"][t]
                pixel_selector = tr["idx_src"][t]
                gt_disp = tr["disp"][t]
                visibility = tr["valid"][t]
            else:
                raise ValueError(f"Unexpected tracks idx_src shape: {idx_src.shape}")
        else:
            tr = tracks
            idx_src = tr["idx_src"]
            if idx_src.ndim == 4:  # (B,T,V,N)
                xy_src = tr["xy_src"][b, t, v]
                pixel_selector = tr["idx_src"][b, t, v]
                gt_disp = tr["disp"][b, t, v]
                visibility = tr["valid"][b, t, v]
            elif idx_src.ndim == 3:  # (B,T,N)
                xy_src = tr["xy_src"][b, t]
                pixel_selector = tr["idx_src"][b, t]
                gt_disp = tr["disp"][b, t]
                visibility = tr["valid"][b, t]
            elif idx_src.ndim == 2:  # (T,N)
                xy_src = tr["xy_src"][t]
                pixel_selector = tr["idx_src"][t]
                gt_disp = tr["disp"][t]
                visibility = tr["valid"][t]
            else:
                raise ValueError(f"Unexpected tracks idx_src shape: {idx_src.shape}")
        return xy_src, pixel_selector, gt_disp, visibility

    @staticmethod
    def _compute_track_keep_indices(
        xy_src: Float[Tensor, "num_points 2"],
        H: int,
        W: int,
        sparsity: int = 100,
    ) -> Int[Tensor, "num_keep"]:
        """
        Choose a fixed subset of track indices using spatial grid (one per cell).
        Use the same logic as visualize_pixel_motion sparsification so that
        when we pass these indices to all frames, the same tracks are shown
        over time for visual continuity.
        """
        N = xy_src.shape[0]
        base_area = 8 * 8
        approx_arrows = max(int((H * W) / base_area / max(sparsity / 100.0, 1e-3)), 1)
        approx_arrows = min(approx_arrows, N)
        grid_size = max(int(np.sqrt(approx_arrows)), 1)
        gx = max(W // grid_size, 1)
        gy = max(H // grid_size, 1)
        x = xy_src[:, 0]
        y = xy_src[:, 1]
        cell_x = torch.div(x, gx, rounding_mode="floor").long()
        cell_y = torch.div(y, gy, rounding_mode="floor").long()
        cell_x = cell_x.clamp(0, W // gx)
        cell_y = cell_y.clamp(0, H // gy)
        num_x = (W // gx) + 1
        cell_id = cell_y * num_x + cell_x  # (N,)
        max_cell = cell_id.max().item() + 1
        first_idx = torch.full((max_cell,), N, device=cell_id.device, dtype=torch.long)
        idx = torch.arange(N, device=cell_id.device)
        first_idx.scatter_reduce_(0, cell_id, idx, reduce="amin", include_self=True)
        unique_indices = first_idx[first_idx < N]
        keep = torch.zeros(N, dtype=torch.bool, device=cell_id.device)
        keep[unique_indices] = True
        return torch.where(keep)[0]

    def visualize_pixel_motion(
        self,
        xy_src: Float[Tensor, "num_points 2"],  # tracked positions (x,y)
        pixel_selector: Int[Tensor, "num_points"],  # for gathering pred flow
        gt_pixel_motion: Float[Tensor, "num_points 2"],
        pred_pixel_motion: Float[Tensor, "num_points 2"],
        gt_pixel_visibility: Float[Tensor, "num_points"],
        rgb: Float[Tensor, "3 H W"],
        sparsity: int = 100,  # higher -> fewer arrows
        motion_thresh: float = 0.05,  # de-normalized pixels; very small by default
        keep_indices: Optional[Int[Tensor, "num_keep"]] = None,
    ):
        """
        Lagrangian track visualization.
        Red  = GT motion
        Blue = predicted motion

        If keep_indices is provided, only those track indices are drawn (e.g. from
        first frame for temporal consistency). Otherwise sparsification is done
        per-call via spatial grid.
        """
        # --------------------------------------------------
        # Scale (no in-place ops): undo oflow normalization
        # Datasets set oflow_scale in metadata (from config or derived from oflow_std).
        # --------------------------------------------------
        gt_motion = denormalize_flow_tensor(gt_pixel_motion, self.dataset_metadata)
        pred_motion = denormalize_flow_tensor(
            pred_pixel_motion,
            self.dataset_metadata,
        )

        H, W = rgb.shape[1], rgb.shape[2]

        if keep_indices is not None:
            # Use a fixed set of tracks (e.g. from first frame) for visual continuity
            N_cur = xy_src.shape[0]
            valid = keep_indices < N_cur
            keep_indices = keep_indices[valid]
            xy_src = xy_src[keep_indices]
            gt_motion = gt_motion[keep_indices]
            pred_motion = pred_motion[keep_indices]
            visibility = gt_pixel_visibility[keep_indices]
        else:
            # Per-call sparsification (spatial grid)
            base_area = 8 * 8
            approx_arrows = max(
                int((H * W) / base_area / max(sparsity / 100.0, 1e-3)), 1
            )
            approx_arrows = min(approx_arrows, int(xy_src.shape[0]))
            grid_size = max(int(np.sqrt(approx_arrows)), 1)
            gx = max(W // grid_size, 1)
            gy = max(H // grid_size, 1)
            x = xy_src[:, 0]
            y = xy_src[:, 1]
            cell_x = torch.div(x, gx, rounding_mode="floor").long()
            cell_y = torch.div(y, gy, rounding_mode="floor").long()
            cell_x = cell_x.clamp(0, W // gx)
            cell_y = cell_y.clamp(0, H // gy)
            num_x = (W // gx) + 1
            cell_id = cell_y * num_x + cell_x  # (N,)
            N = cell_id.shape[0]
            max_cell = cell_id.max().item() + 1
            first_idx = torch.full(
                (max_cell,), N, device=cell_id.device, dtype=torch.long
            )
            idx = torch.arange(N, device=cell_id.device)
            first_idx.scatter_reduce_(0, cell_id, idx, reduce="amin", include_self=True)
            unique_indices = first_idx[first_idx < N]
            keep = torch.zeros(N, dtype=torch.bool, device=cell_id.device)
            keep[unique_indices] = True
            xy_src = xy_src[keep]
            gt_motion = gt_motion[keep]
            pred_motion = pred_motion[keep]
            visibility = gt_pixel_visibility[keep]

        # --------------------------------------------------
        # Image prep (with supersampling for low-res)
        # --------------------------------------------------
        rgb_np = np.ascontiguousarray(
            (rgb.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        )

        short_side = float(min(H, W))
        scale_canvas = 3 if short_side <= 256 else 1
        if scale_canvas > 1:
            canvas = cv2.resize(
                rgb_np,
                (W * scale_canvas, H * scale_canvas),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            canvas = rgb_np

        # --------------------------------------------------
        # Arrow geometry (lengths scale with image size)
        # --------------------------------------------------
        # viz_scale = 0.50 * short_side
        viz_scale = 1.0

        def to_canvas(pt):
            return (int(pt[0] * scale_canvas), int(pt[1] * scale_canvas))

        # --------------------------------------------------
        # Draw arrows
        # --------------------------------------------------
        for i in range(xy_src.shape[0]):
            if visibility[i] <= 0.5:
                continue

            x0, y0 = xy_src[i]
            gt_dx, gt_dy = gt_motion[i]
            pr_dx, pr_dy = pred_motion[i]

            # skip only truly tiny motion (use vector norm in de-normalized pixels)
            gt_norm = torch.sqrt(gt_dx**2 + gt_dy**2)
            pr_norm = torch.sqrt(pr_dx**2 + pr_dy**2)
            if gt_norm <= motion_thresh and pr_norm <= motion_thresh:
                continue

            # Arrows are scaled proportionally with image size (no min/max clipping)
            start = to_canvas((x0, y0))
            gt_end = to_canvas((x0 + viz_scale * gt_dx, y0 + viz_scale * gt_dy))
            pr_end = to_canvas((x0 + viz_scale * pr_dx, y0 + viz_scale * pr_dy))

            # Colors: GT = red, Pred = blue (RGB format)
            gt_color = (255, 0, 0)  # Red in RGB
            pr_color = (0, 0, 255)  # Blue in RGB
            outline_color = (0, 0, 0)

            def draw_arrow(end_pt, color):
                cv2.arrowedLine(canvas, start, end_pt, outline_color, 4, tipLength=0.35)
                cv2.arrowedLine(canvas, start, end_pt, color, 2, tipLength=0.25)

            draw_arrow(gt_end, gt_color)
            draw_arrow(pr_end, pr_color)

        # Downsample back to original resolution if we supersampled
        if scale_canvas > 1:
            canvas = cv2.resize(canvas, (W, H), interpolation=cv2.INTER_AREA)

        return rearrange(canvas, "h w c -> c h w")

    def on_validation_epoch_start(self) -> None:
        if self.cfg.logging.deterministic is not None:
            self.generator = torch.Generator(device=self.device).manual_seed(
                self.global_rank
                + self.trainer.world_size * self.cfg.logging.deterministic
            )

    def on_validation_epoch_end(self) -> None:
        self.generator = None

    def test_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        return self.validation_step(*args, **kwargs, namespace="test")

    def on_test_epoch_start(self) -> None:
        self.on_validation_epoch_start()

    def on_test_epoch_end(self) -> None:
        self.on_validation_epoch_end()

    # ---------------------------------------------------------------------
    # Loss Computation
    # ---------------------------------------------------------------------
    @staticmethod
    def _flatten_tracks(tracks: Dict[str, Tensor]) -> Dict[str, Tensor]:
        xy = tracks["xy_src"]
        disp = tracks["disp"]
        valid = tracks["valid"]
        idx = tracks["idx_src"]

        if idx.ndim == 4:  # (B,T,V,N)
            xy, disp, valid, idx = map(flatten_time_view_dim, (xy, disp, valid, idx))
        elif idx.ndim == 3:  # (B,T,N)
            xy, disp, valid, idx = map(flatten_time_dim, (xy, disp, valid, idx))
        elif idx.ndim == 2:  # (T,N) or already flattened
            pass
        else:
            raise ValueError(f"Unexpected tracks idx_src shape: {idx.shape}")

        return {
            "xy_src": xy,
            "disp": disp,
            "valid": valid,
            "idx_src": idx,
        }

    def _gather_track_motion(
        self,
        model_output: JacobianFieldOutput,
        tracks: Dict[str, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        tracks_flat = self._flatten_tracks(tracks)
        idx = tracks_flat["idx_src"]
        disp = tracks_flat["disp"]
        valid = tracks_flat["valid"] > 0

        flow_pred = rearrange(
            model_output.optical_flow,
            "b c h w -> b h w c",
        )
        if idx.shape[0] != flow_pred.shape[0]:
            raise ValueError(
                "Tracks batch size does not match model output: "
                f"tracks={idx.shape[0]} vs pred={flow_pred.shape[0]}"
            )

        flow_flat = flow_pred.view(flow_pred.shape[0], -1, flow_pred.shape[-1])
        flow_hw = flow_flat.shape[1]
        idx_safe = idx.clamp(0, flow_hw - 1).long()
        idx_valid = (idx >= 0) & (idx < flow_hw)
        valid_mask = valid & idx_valid
        pred_disp = torch.gather(
            flow_flat,
            dim=1,
            index=idx_safe[..., None].expand(-1, -1, flow_flat.shape[-1]),
        )

        return pred_disp, disp, valid_mask

    def _solve_action_least_squares(
        self,
        jacobian_matrix: Float[Tensor, "b num_rows c_dim"],
        motion: Float[Tensor, "b num_rows"],
        row_weights: Optional[Float[Tensor, "b num_rows"]] = None,
    ) -> Float[Tensor, "b c_dim"]:
        autocast_ctx = (
            torch.amp.autocast("cuda", enabled=False)
            if jacobian_matrix.is_cuda
            else nullcontext()
        )
        with autocast_ctx:
            cmd_dim = jacobian_matrix.shape[-1]
            J = jacobian_matrix.to(dtype=torch.float32)
            y = motion.to(dtype=torch.float32)

            if row_weights is None:
                weights = torch.ones_like(y, dtype=torch.float32)
            else:
                weights = row_weights.to(dtype=torch.float32).clamp_min(0.0)

            sqrt_weights = torch.sqrt(weights)
            J_weighted = J * sqrt_weights.unsqueeze(-1)
            y_weighted = y * sqrt_weights

            jtj = torch.bmm(J_weighted.transpose(1, 2), J_weighted).to(torch.float32)
            jty = torch.bmm(
                J_weighted.transpose(1, 2), y_weighted.unsqueeze(-1)
            ).to(torch.float32)
            eye = torch.eye(cmd_dim, device=jtj.device, dtype=torch.float32).unsqueeze(0)
            diag_mean = jtj.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True)
            ridge = (
                float(self.cfg.inverse_action_damping) * diag_mean.clamp_min(1.0)
            ).unsqueeze(-1)
            jtj = jtj + ridge * eye
            du_hat = torch.linalg.solve(jtj, jty).squeeze(-1)

        return du_hat

    def _recover_action_from_dense_flow(
        self,
        jacobian: Float[Tensor, "b c_dim s_dim h w"],
        flow_gt: Float[Tensor, "b s_dim h w"],
    ) -> Float[Tensor, "b c_dim"]:
        jacobian_matrix = rearrange(jacobian, "b c s h w -> b (s h w) c")
        motion = rearrange(flow_gt, "b s h w -> b (s h w)")
        return self._solve_action_least_squares(jacobian_matrix, motion)

    def _recover_action_from_tracks(
        self,
        jacobian: Float[Tensor, "b c_dim s_dim h w"],
        tracks: Dict[str, Tensor],
    ) -> Float[Tensor, "b c_dim"]:
        tracks_flat = self._flatten_tracks(tracks)
        idx = tracks_flat["idx_src"]
        disp = tracks_flat["disp"]
        valid = tracks_flat["valid"] > 0

        jacobian_flat = rearrange(jacobian, "b c s h w -> b (h w) s c")
        num_rows = jacobian_flat.shape[1]
        idx_safe = idx.clamp(0, num_rows - 1).long()
        idx_valid = (idx >= 0) & (idx < num_rows)
        valid_mask = valid & idx_valid
        jacobian_sparse = torch.gather(
            jacobian_flat,
            dim=1,
            index=idx_safe[..., None, None].expand(
                -1, -1, jacobian_flat.shape[-2], jacobian_flat.shape[-1]
            ),
        )

        jacobian_matrix = rearrange(jacobian_sparse, "b n s c -> b (n s) c")
        motion = rearrange(disp, "b n s -> b (n s)")
        row_weights = rearrange(
            valid_mask[..., None].expand(-1, -1, disp.shape[-1]),
            "b n s -> b (n s)",
        ).to(dtype=jacobian.dtype)
        return self._solve_action_least_squares(jacobian_matrix, motion, row_weights)

    def _compute_inverse_action_loss(
        self,
        model_output: JacobianFieldOutput,
        du: Float[Tensor, "b c_dim"],
        flow_gt: Optional[Float[Tensor, "b s_dim h w"]],
        tracks: Optional[Dict[str, Tensor]],
    ) -> tuple[Optional[Tensor], Dict[str, Tensor]]:
        diagnostics: Dict[str, Tensor] = {}

        du_hat = None
        source = None
        if flow_gt is not None:
            du_hat = self._recover_action_from_dense_flow(model_output.jacobian, flow_gt)
            source = "flow"
        elif tracks is not None:
            du_hat = self._recover_action_from_tracks(model_output.jacobian, tracks)
            source = "tracks"

        if du_hat is None or source is None:
            return None, diagnostics

        du_hat_safe = torch.nan_to_num(
            du_hat.to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0
        ).clamp_(-10.0, 10.0)
        du_target = du.to(torch.float32)
        residual = du_hat_safe - du_target
        diagnostics[f"du_hat_{source}_std"] = du_hat_safe.detach().std()
        diagnostics[f"du_hat_{source}_norm_mean"] = (
            du_hat_safe.detach().norm(dim=-1).mean()
        )
        diagnostics[f"du_hat_{source}_mse"] = residual.detach().pow(2).mean()
        diagnostics[f"du_hat_{source}_smooth_l1"] = F.smooth_l1_loss(
            du_hat_safe,
            du_target,
            beta=1.0,
        ).detach()
        diagnostics[f"du_hat_{source}_finite_ratio"] = torch.isfinite(du_hat).float().mean()

        if self.cfg.inverse_action_weight <= 0:
            return None, diagnostics

        loss = float(self.cfg.inverse_action_weight) * F.smooth_l1_loss(
            du_hat_safe,
            du_target,
            beta=1.0,
        )
        return loss, diagnostics

    def _collect_diagnostics(
        self,
        model_output: JacobianFieldOutput,
        du: Float[Tensor, "b c_dim"],
        flow_gt: Optional[Float[Tensor, "b s_dim h w"]],
        tracks: Optional[Dict[str, Tensor]],
    ) -> Dict[str, Tensor]:
        jacobian = model_output.jacobian.detach().to(torch.float32)
        pred_flow = model_output.optical_flow.detach().to(torch.float32)
        du_detached = du.detach().to(torch.float32)
        diagnostics: Dict[str, Tensor] = {
            "jacobian_std": jacobian.std(),
            "pred_flow_std": pred_flow.std(),
            "du_norm_mean": du_detached.norm(dim=-1).mean(),
            "jacobian_abs_mean": jacobian.abs().mean(),
        }

        du_mean = du_detached.mean(dim=0)
        du_std = du_detached.std(dim=0, unbiased=False)
        zero_action_model = self._get_zero_action_model_tensor(
            device=du_detached.device,
            dtype=du_detached.dtype,
            cmd_dim=du_detached.shape[-1],
        )
        for dim_idx in range(du_detached.shape[-1]):
            diagnostics[f"du_dim{dim_idx}_mean"] = du_mean[dim_idx]
            diagnostics[f"du_dim{dim_idx}_std"] = du_std[dim_idx]
            diagnostics[f"zero_action_model_dim{dim_idx}"] = zero_action_model[dim_idx]

        jacobian_cmd_rms = torch.sqrt(jacobian.pow(2).mean(dim=(0, 2, 3, 4)))
        for dim_idx in range(jacobian.shape[1]):
            diagnostics[f"jacobian_cmd{dim_idx}_rms"] = jacobian_cmd_rms[dim_idx]

        jacobian_pixel_rms = torch.sqrt(jacobian.pow(2).mean(dim=(1, 2)))
        diagnostics["jacobian_near_zero_frac_1e-4"] = (
            jacobian_pixel_rms < 1e-4
        ).float().mean()
        zero_flow_model = self._get_zero_flow_model_tensor(
            device=pred_flow.device,
            dtype=pred_flow.dtype,
            spatial_dim=pred_flow.shape[1],
        )
        pred_flow_mean = pred_flow.mean(dim=(0, 2, 3))
        pred_flow_std = pred_flow.std(dim=(0, 2, 3), unbiased=False)
        for channel_idx in range(pred_flow.shape[1]):
            diagnostics[f"pred_flow_ch{channel_idx}_mean"] = pred_flow_mean[channel_idx]
            diagnostics[f"pred_flow_ch{channel_idx}_std"] = pred_flow_std[channel_idx]
            diagnostics[f"zero_flow_model_ch{channel_idx}"] = zero_flow_model[channel_idx]

        if flow_gt is not None:
            flow_gt = flow_gt.detach().to(torch.float32)
            gt_flow_std = flow_gt.std()
            diagnostics["gt_flow_std"] = gt_flow_std
            diagnostics["gt_flow_norm_mean"] = flow_gt.norm(dim=1).mean()
            diagnostics["pred_gt_flow_std_ratio"] = (
                pred_flow.std() / gt_flow_std.clamp_min(1e-6)
            )
            gt_flow_mean = flow_gt.mean(dim=(0, 2, 3))
            gt_flow_std_per_channel = flow_gt.std(dim=(0, 2, 3), unbiased=False)
            for channel_idx in range(flow_gt.shape[1]):
                diagnostics[f"gt_flow_ch{channel_idx}_mean"] = gt_flow_mean[channel_idx]
                diagnostics[f"gt_flow_ch{channel_idx}_std"] = (
                    gt_flow_std_per_channel[channel_idx]
                )

        if tracks is not None:
            pred_disp, gt_disp, valid_mask = self._gather_track_motion(model_output, tracks)
            if valid_mask.any():
                gt_disp_valid = gt_disp[valid_mask]
                pred_disp_valid = pred_disp[valid_mask]
                diagnostics["gt_track_motion_std"] = gt_disp_valid.detach().std()
                diagnostics["pred_track_motion_std"] = pred_disp_valid.detach().std()
                diagnostics["gt_track_motion_norm_mean"] = (
                    gt_disp_valid.detach().norm(dim=-1).mean()
                )
                diagnostics["pred_track_motion_norm_mean"] = (
                    pred_disp_valid.detach().norm(dim=-1).mean()
                )

        return diagnostics

    def _track_loss(
        self,
        model_output: JacobianFieldOutput,
        tracks: Dict[str, Tensor],
    ) -> Tensor:
        """
        Sparse supervision using motion tracks.
        """

        pred_disp, gt_disp, valid_mask = self._gather_track_motion(model_output, tracks)
        if not valid_mask.any():
            return model_output.optical_flow.sum() * 0.0

        return self._dense_flow_loss(
            pred_disp[valid_mask],
            gt_disp[valid_mask],
        )

    def _dense_flow_loss(
        self,
        pred_flow: Tensor,
        flow_gt: Tensor,
    ) -> Tensor:
        if pred_flow.shape != flow_gt.shape:
            raise RuntimeError(
                "Flow shape mismatch before loss computation: "
                f"pred_flow.shape={tuple(pred_flow.shape)}, "
                f"flow_gt.shape={tuple(flow_gt.shape)}. "
                "This usually means the dataset flow geometry does not match the "
                "model output geometry (for example, stale metadata cache paths "
                "still pointing to raw full-resolution flow instead of the derived "
                "train-resolution cache)."
            )
        if self.cfg.flow_loss == "mse":
            return F.mse_loss(pred_flow, flow_gt)

        if self.cfg.flow_loss == "charbonnier":
            diff = pred_flow - flow_gt
            eps = self.cfg.flow_charbonnier_eps
            return torch.sqrt(diff.pow(2) + eps**2).mean()

        raise ValueError(f"Unsupported flow loss: {self.cfg.flow_loss}")

    def _flow_pixel_loss_btv(self, pred_flow_btv: Tensor, flow_gt_btv: Tensor) -> Tensor:
        if pred_flow_btv.shape != flow_gt_btv.shape:
            raise RuntimeError(
                "Flow shape mismatch before pixelwise loss computation: "
                f"pred_flow_btv.shape={tuple(pred_flow_btv.shape)}, "
                f"flow_gt_btv.shape={tuple(flow_gt_btv.shape)}"
            )
        if self.cfg.flow_loss == "mse":
            return (pred_flow_btv - flow_gt_btv).pow(2).mean(dim=3)
        if self.cfg.flow_loss == "charbonnier":
            eps = self.cfg.flow_charbonnier_eps
            return torch.sqrt((pred_flow_btv - flow_gt_btv).pow(2) + eps**2).mean(dim=3)
        raise ValueError(f"Unsupported flow loss: {self.cfg.flow_loss}")

    def _get_view_names(self, num_views: int) -> list[str]:
        meta = self.dataset_metadata or {}
        raw_names = meta.get("views") or meta.get("camera_views")
        names: list[str] = []
        for view_idx in range(num_views):
            if isinstance(raw_names, list) and view_idx < len(raw_names):
                names.append(str(raw_names[view_idx]))
            else:
                names.append(f"view{view_idx}")
        return names

    def _active_pixel_balanced_flow_loss_per_view(
        self,
        *,
        pred_flow_btv: Tensor,
        flow_gt_btv: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        pixel_loss = self._flow_pixel_loss_btv(pred_flow_btv, flow_gt_btv)
        gt_mag = torch.linalg.vector_norm(flow_gt_btv, dim=3)
        moving_mask = gt_mag > float(self.cfg.motion_aware_flow_threshold)
        moving = moving_mask.to(pixel_loss.dtype)

        active_counts = moving.sum(dim=(0, 1, 3, 4))
        total_per_view = float(moving.shape[0] * moving.shape[1] * moving.shape[3] * moving.shape[4])
        active_ratio = active_counts / max(total_per_view, 1.0)

        active_loss = (pixel_loss * moving).sum(dim=(0, 1, 3, 4)) / active_counts.clamp_min(1.0)
        fallback_loss = pixel_loss.mean(dim=(0, 1, 3, 4))
        view_losses = torch.where(active_counts > 0, active_loss, fallback_loss)

        min_ratio = max(float(self.cfg.view_balance_min_active_ratio), 1e-6)
        max_weight = max(float(self.cfg.view_balance_max_view_weight), 1.0)
        view_weights = (1.0 / active_ratio.clamp_min(min_ratio)).clamp_max(max_weight)
        view_weights = view_weights / view_weights.mean().clamp_min(1e-6)
        return view_losses, active_ratio, view_weights

    def _dense_flow_loss_per_view(
        self,
        *,
        pred_flow_btv: Tensor,
        flow_gt_btv: Tensor,
    ) -> Tensor:
        if pred_flow_btv.shape != flow_gt_btv.shape:
            raise RuntimeError(
                "Flow shape mismatch before per-view loss computation: "
                f"pred_flow_btv.shape={tuple(pred_flow_btv.shape)}, "
                f"flow_gt_btv.shape={tuple(flow_gt_btv.shape)}"
            )
        view_losses = [
            self._dense_flow_loss(
                pred_flow_btv[:, :, view_idx], flow_gt_btv[:, :, view_idx]
            )
            for view_idx in range(pred_flow_btv.shape[2])
        ]
        return torch.stack(view_losses, dim=0)

    def _motion_aware_dense_flow_loss(
        self,
        *,
        pred_flow_btv: Tensor,
        flow_gt_btv: Tensor,
    ) -> Tensor:
        return self._motion_aware_dense_flow_loss_per_view(
            pred_flow_btv=pred_flow_btv, flow_gt_btv=flow_gt_btv
        ).mean()

    def _motion_aware_dense_flow_loss_per_view(
        self,
        *,
        pred_flow_btv: Tensor,
        flow_gt_btv: Tensor,
    ) -> Tensor:
        # Use gt motion magnitude as weighting signal to avoid background-dominated minima.
        tau = float(self.cfg.motion_aware_flow_threshold)
        moving_weight = float(self.cfg.motion_aware_moving_weight)
        gt_mag = torch.linalg.vector_norm(flow_gt_btv, dim=3)
        moving_mask = gt_mag > tau
        weights = torch.where(
            moving_mask,
            torch.full_like(gt_mag, moving_weight),
            torch.ones_like(gt_mag),
        )
        weights = weights / weights.mean().clamp_min(1e-6)
        channel_weights = weights.unsqueeze(3)

        if self.cfg.flow_loss == "mse":
            diff = (pred_flow_btv - flow_gt_btv).pow(2)
            weighted = diff * channel_weights
            return weighted.mean(dim=(0, 1, 3, 4, 5))
        if self.cfg.flow_loss == "charbonnier":
            eps = self.cfg.flow_charbonnier_eps
            charbonnier = torch.sqrt((pred_flow_btv - flow_gt_btv).pow(2) + eps**2)
            weighted = charbonnier * channel_weights
            return weighted.mean(dim=(0, 1, 3, 4, 5))
        raise ValueError(f"Unsupported flow loss: {self.cfg.flow_loss}")

    def _flow_gradient_loss(self, pred_flow: Tensor, flow_gt: Tensor) -> Tensor:
        """Match the spatial gradient of achieved flow to GT flow (VGGT's grad term).

        pred_flow / flow_gt: [N, 2, H, W].
        """

        def _spatial_grad(x: Tensor) -> tuple[Tensor, Tensor]:
            gx = x[..., :, 1:] - x[..., :, :-1]
            gy = x[..., 1:, :] - x[..., :-1, :]
            return gx, gy

        pgx, pgy = _spatial_grad(pred_flow)
        ggx, ggy = _spatial_grad(flow_gt)
        eps = self.cfg.flow_charbonnier_eps

        def charb(d: Tensor) -> Tensor:
            return torch.sqrt(d.pow(2) + eps * eps)

        return charb(pgx - ggx).mean() + charb(pgy - ggy).mean()

    def _jacobian_tv_loss(self, jacobian: Tensor, rgb: Tensor) -> Tensor:
        """Edge-aware total-variation on the jacobian field (floater killer).

        jacobian: ``[..., H, W]`` where everything before the trailing ``H, W`` is a
        mix of spatial-batch and channel dims. ``rgb`` is the model input, also
        ``[..., 3, H, W]``. Both share the same spatial-batch size N (= b*t*v); we
        fold each to ``[N, K, H, W]`` so a single per-pixel edge weight ``[N, 1, H, W]``
        broadcasts over the jacobian channels K (= c_dim*s_dim).

        The jacobian carries the view axis inside N (shape ``[b*t*v, c, s, H, W]``),
        whereas ``rgb`` keeps the view axis as a separate dim (``[b*t, v, 3, H, W]``).
        We therefore fold *all* leading dims of rgb (except the color+spatial tail)
        into N as well, which yields the same N and keeps the per-view correspondence.
        """
        h, w = jacobian.shape[-2], jacobian.shape[-1]
        # Fold all jacobian dims except (H, W) into a single map axis: every jacobian
        # channel (c_dim*s_dim) at every spatial-batch entry becomes its own [1,H,W]
        # map. TV is per-channel spatial smoothness, so this is exactly what we want.
        j = jacobian.reshape(-1, h, w).unsqueeze(1)  # [M, 1, H, W], M = b*t*v*c_dim*s_dim

        gx = (j[..., :, 1:] - j[..., :, :-1]).abs()  # [M, 1, H, W-1]
        gy = (j[..., 1:, :] - j[..., :-1, :]).abs()  # [M, 1, H-1, W]

        if self.cfg.jacobian_tv_edge_aware:
            beta = float(self.cfg.jacobian_tv_edge_beta)
            # Fold rgb to [N, 3, H, W] (N = b*t*v) then build a per-pixel edge weight.
            rgb_n = rgb.reshape(-1, *rgb.shape[-3:])  # [N, 3, H, W]
            igx = rgb_n[..., :, 1:] - rgb_n[..., :, :-1]
            igy = rgb_n[..., 1:, :] - rgb_n[..., :-1, :]
            # Mean over color channels -> [N, 1, H, W-1] / [N, 1, H-1, W].
            wx = torch.exp(-beta * igx.abs().mean(dim=-3, keepdim=True))
            wy = torch.exp(-beta * igy.abs().mean(dim=-3, keepdim=True))
            # j was folded as [M, 1, H, W] with M = N * jac_channels. Expand the weight
            # over the jac-channel multiplicity so it broadcasts cleanly.
            n = wx.shape[0]
            rep = j.shape[0] // n
            wx = wx.repeat_interleave(rep, dim=0)
            wy = wy.repeat_interleave(rep, dim=0)
            gx = gx * wx
            gy = gy * wy

        return gx.mean() + gy.mean()

    def _compute_loss(
        self,
        batch,
        namespace="training",
        *,
        collect_diagnostics: bool = True,
    ) -> tuple[
        dict[str, Tensor],
        dict[str, Tensor],
        Tensor,
        JacobianFieldOutput,
        dict[str, Tensor],
        dict[str, float],
    ]:
        timing: dict[str, float] = {}
        t0 = time.perf_counter()
        du = batch["du"]
        rgb = batch["rgb"]
        flow_gt = batch.get("flow")
        tracks = batch.get("tracks")
        flow_gt_btv: Optional[Tensor] = None
        view_ids: Optional[Tensor] = None

        if isinstance(tracks, list) and len(tracks) > 0:
            tracks = {k: torch.stack([t[k] for t in tracks], dim=0) for k in tracks[0]}

        joint_multiview = bool(rgb.ndim == 6 and self._model_supports_joint_multiview())
        rgb, model_du, flow_gt, flow_gt_btv, du, view_ids = self._prepare_model_inputs(
            rgb=rgb,
            du=du,
            flow_gt=flow_gt,
            joint_multiview=joint_multiview,
        )
        timing["flatten_inputs_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        model_output: JacobianFieldOutput = self.model(
            InputObservation(rgb=rgb, view_ids=view_ids),
            InputCommand(du=model_du),
        )
        if joint_multiview:
            model_output = self._flatten_joint_model_output(model_output)
        timing["model_forward_s"] = time.perf_counter() - t0

        losses: dict[str, Tensor] = {}
        loss_metrics: dict[str, Tensor] = {}
        flow_balance_diagnostics: dict[str, Tensor] = {}

        t0 = time.perf_counter()
        if "flow" in self.cfg.supervision:
            assert flow_gt is not None
            balance_mode = str(self.cfg.view_flow_balance_mode).lower()
            use_legacy_view_mean = bool(self.cfg.view_balanced_flow_loss)
            if flow_gt_btv is not None and (balance_mode == "active_pixels" or use_legacy_view_mean):
                pred_flow_btv = rearrange(
                    model_output.optical_flow,
                    "(b t v) c h w -> b t v c h w",
                    b=flow_gt_btv.shape[0],
                    t=flow_gt_btv.shape[1],
                    v=flow_gt_btv.shape[2],
                )
                if balance_mode == "active_pixels":
                    flow_view_losses, active_ratio, view_weights = (
                        self._active_pixel_balanced_flow_loss_per_view(
                            pred_flow_btv=pred_flow_btv,
                            flow_gt_btv=flow_gt_btv,
                        )
                    )
                    losses["flow"] = (flow_view_losses * view_weights).sum()
                    view_names = self._get_view_names(flow_view_losses.shape[0])
                    for view_idx, view_name in enumerate(view_names):
                        safe_view_name = view_name.replace("/", "_")
                        flow_balance_diagnostics[f"flow_active_ratio/view_{safe_view_name}"] = (
                            active_ratio[view_idx].detach()
                        )
                        flow_balance_diagnostics[f"flow_view_weight/view_{safe_view_name}"] = (
                            view_weights[view_idx].detach()
                        )
                elif self.cfg.motion_aware_flow_weighting:
                    flow_view_losses = self._motion_aware_dense_flow_loss_per_view(
                        pred_flow_btv=pred_flow_btv,
                        flow_gt_btv=flow_gt_btv,
                    )
                    losses["flow"] = flow_view_losses.mean()
                elif use_legacy_view_mean:
                    flow_view_losses = self._dense_flow_loss_per_view(
                        pred_flow_btv=pred_flow_btv,
                        flow_gt_btv=flow_gt_btv,
                    )
                    losses["flow"] = flow_view_losses.mean()
                else:
                    flow_view_losses = self._dense_flow_loss_per_view(
                        pred_flow_btv=pred_flow_btv,
                        flow_gt_btv=flow_gt_btv,
                    )
                    losses["flow"] = flow_view_losses.mean()
                if self.cfg.log_flow_per_view:
                    view_names = self._get_view_names(flow_view_losses.shape[0])
                    for view_idx, view_name in enumerate(view_names):
                        safe_view_name = view_name.replace("/", "_")
                        loss_metrics[f"flow/view_{safe_view_name}"] = flow_view_losses[
                            view_idx
                        ]
            else:
                if self.cfg.motion_aware_flow_weighting and flow_gt_btv is not None:
                    pred_flow_btv = rearrange(
                        model_output.optical_flow,
                        "(b t v) c h w -> b t v c h w",
                        b=flow_gt_btv.shape[0],
                        t=flow_gt_btv.shape[1],
                        v=flow_gt_btv.shape[2],
                    )
                    losses["flow"] = self._motion_aware_dense_flow_loss(
                        pred_flow_btv=pred_flow_btv,
                        flow_gt_btv=flow_gt_btv,
                    )
                else:
                    losses["flow"] = self._dense_flow_loss(
                        model_output.optical_flow,
                        flow_gt,
                    )

        if "tracks" in self.cfg.supervision:
            assert tracks is not None
            losses["tracks"] = self._track_loss(model_output, tracks)
        timing["supervision_loss_s"] = time.perf_counter() - t0

        # pmap-style regularization terms (adapted from VGGT point-map loss).
        # All gated behind their weights/flags; default-off -> losses dict unchanged.
        t0 = time.perf_counter()
        pred_flow = model_output.optical_flow
        jacobian = model_output.jacobian
        if self.cfg.flow_gradient_weight > 0 and flow_gt is not None:
            losses["flow_gradient"] = self.cfg.flow_gradient_weight * self._flow_gradient_loss(
                pred_flow, flow_gt
            )
        if self.cfg.jacobian_tv_weight > 0:
            losses["jacobian_tv"] = self.cfg.jacobian_tv_weight * self._jacobian_tv_loss(
                jacobian, rgb
            )
        if (
            self.cfg.predict_uncertainty
            and model_output.flow_confidence is not None
            and flow_gt is not None
        ):
            confidence = model_output.flow_confidence  # positive map, [N, 1, H, W]
            if self.cfg.pmap_confidence_clamp > 0:
                # Cap the confidence used by BOTH aleatoric terms so easy/static
                # pixels cannot inflate conf without bound (see cfg field docs).
                # clamp() zeroes the -log(conf) reward gradient past the cap,
                # which is exactly the intended behavior.
                confidence = confidence.clamp(max=self.cfg.pmap_confidence_clamp)
            eps = self.cfg.flow_charbonnier_eps
            # Per-pixel charbonnier residual averaged over the 2 flow channels -> [N, 1, H, W].
            r = torch.sqrt((pred_flow - flow_gt).pow(2) + eps * eps).mean(dim=-3, keepdim=True)
            aleatoric = (confidence * r).mean() - self.cfg.pmap_uncertainty_weight * torch.log(
                confidence
            ).mean()
            losses["flow_aleatoric"] = aleatoric
        timing["pmap_regularization_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        inverse_action_loss, inverse_action_diagnostics = self._compute_inverse_action_loss(
            model_output,
            du,
            flow_gt,
            tracks,
        )
        if inverse_action_loss is not None:
            losses["inverse_action"] = inverse_action_loss
        timing["inverse_action_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        diagnostics: dict[str, Tensor] = {}
        if collect_diagnostics:
            diagnostics = self._collect_diagnostics(model_output, du, flow_gt, tracks)
            diagnostics.update(inverse_action_diagnostics)
            diagnostics.update(flow_balance_diagnostics)
        timing["diagnostics_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        total_loss = torch.stack(list(losses.values())).sum()
        timing["loss_aggregation_s"] = time.perf_counter() - t0

        return losses, loss_metrics, total_loss, model_output, diagnostics, timing

    def _log_losses(
        self,
        losses: dict[str, Tensor],
        loss_metrics: dict[str, Tensor],
        total_loss: Tensor,
        diagnostics: dict[str, Tensor],
        namespace: str,
    ):
        if namespace.startswith("validation") and getattr(self.trainer, "is_global_zero", False):
            _agent_debug_log(
                "H4",
                "image_jacobian.py:_log_losses",
                "validation loss key emitted",
                {
                    "key": f"loss/{namespace}/total",
                    "num_losses": int(len(losses)),
                    "num_diagnostics": int(len(diagnostics)),
                },
            )
        log_kwargs = dict(
            on_step=(namespace == "training"),
            on_epoch=(namespace != "training"),
            sync_dist=True,
            add_dataloader_idx=False,
        )

        self.log(f"loss/{namespace}/total", total_loss, **log_kwargs)

        for name, value in losses.items():
            self.log(
                f"loss/{namespace}/{self._sanitize_metric_key(name)}",
                value,
                **log_kwargs,
            )
        for name, value in loss_metrics.items():
            self.log(
                f"loss/{namespace}/{self._sanitize_metric_key(name)}",
                value,
                **log_kwargs,
            )

        inverse_action_mse = diagnostics.get("du_hat_flow_mse")
        if inverse_action_mse is None:
            inverse_action_mse = diagnostics.get("du_hat_tracks_mse")
        if inverse_action_mse is not None:
            self.log(
                f"metrics/{namespace}/action_mse",
                inverse_action_mse,
                **log_kwargs,
            )

        for name, value in diagnostics.items():
            self.log(
                f"diagnostics/{namespace}/{self._sanitize_metric_key(name)}",
                value,
                **log_kwargs,
            )

    # ---------------------------------------------------------------------
    # Checkpoint Utils
    # ---------------------------------------------------------------------
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # If using torch.compile, un-wrap keys before saving
        if self.cfg.compile:
            checkpoint["state_dict"] = {
                k.replace("_orig_mod.", ""): v
                for k, v in checkpoint["state_dict"].items()
            }

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        print("Loading checkpoint...")

        if self.cfg.compile:
            checkpoint["state_dict"] = {
                (
                    k.replace("model.", "model._orig_mod.", 1)
                    if k.startswith("model.") and "_orig_mod." not in k
                    else k
                ): v
                for k, v in checkpoint["state_dict"].items()
            }
            print("Adjusted checkpoint keys for torch.compile.")

        super().on_load_checkpoint(checkpoint)

        if self.cfg.checkpoint.reset_optimizer:
            checkpoint["optimizer_states"] = []

    def _adapt_state_dict_for_compile(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        assert isinstance(state_dict, dict), type(state_dict)

        if not self.cfg.compile:
            return state_dict

        adapted = {}
        for k, v in state_dict.items():
            if k.startswith("model.") and "_orig_mod." not in k:
                k = k.replace("model.", "model._orig_mod.", 1)
            adapted[k] = v

        print("Adapted state_dict keys for torch.compile. type:", type(adapted))

        return adapted
