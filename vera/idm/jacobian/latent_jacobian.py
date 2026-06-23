"""
Latent Jacobian algorithm: train encoder, decoder, and J(z) with five configurable losses.
Uses consecutive frame pairs (curr, next, du) from batch; supports forward_uses_gt_action flag.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, cast

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from einops import rearrange
from torchvision.utils import flow_to_image
from lightning.pytorch.utilities import grad_norm
from lightning.pytorch.utilities.types import STEP_OUTPUT
from omegaconf import DictConfig, OmegaConf

from vera.idm.common.base_pytorch_algo import BasePytorchAlgo, BasePytorchAlgoCfg
from vera.idm.jacobian.models.latent_jacobian_field import LatentJacobianField
from vera.idm.jacobian.models.registry import (
    resolve_model_cfg,
    resolve_model_instance,
)
from vera.idm.registry import register_algorithm
from vera.datasets.normalization import denormalize_flow_tensor
from vera.utils.jacobian_utils import visualize_latent_jacobian
from vera.utils.logging_utils import get_sanity_metrics, safe_asdict
from torch import Tensor
from torch.optim.optimizer import Optimizer


def flatten_time_dim(x: Tensor) -> Tensor:
    return rearrange(x, "b t ... -> (b t) ...")


def flatten_time_view_dim(x: Tensor) -> Tensor:
    return rearrange(x, "b t v ... -> (b t v) ...")


@dataclass
class CheckpointCfg:
    reset_optimizer: bool = False
    strict: bool = True


@dataclass
class OptimizerCfg:
    name: Literal["adamw", "adam", "sgd"] = "adamw"
    lr: float = 1e-4
    weight_decay: float = 1e-3
    beta: List[float] = field(default_factory=lambda: [0.9, 0.99])


@dataclass
class LoggingCfg:
    loss_freq: int = 100
    grad_norm_freq: Optional[int] = None
    max_num_videos: int = 4
    max_frames_per_video: int = 32
    verbose_vis: bool = True  # Print vis shapes for debugging; set False once vis works
    vis_on_sanity_check: bool = True  # If True, run visualization during initial sanity check (on start)


@dataclass
class LatentJacobianCfg(BasePytorchAlgoCfg):
    name: Literal["latent_jacobian"]
    robot_name: str = ""
    compile: bool = False
    model: Any = field(default=None)
    optimizer: OptimizerCfg = field(default_factory=OptimizerCfg)
    logging: LoggingCfg = field(default_factory=LoggingCfg)
    checkpoint: CheckpointCfg = field(default_factory=CheckpointCfg)
    image_size: List[int] = field(default_factory=lambda: [252, 252])

    forward_uses_gt_action: bool = True
    loss_backward_action: bool = True
    loss_abs_pred: bool = True
    loss_rel_pred: bool = True
    loss_z_dot_decode: bool = True
    loss_recon: bool = True

    weight_backward_action: float = 1.0
    weight_recon: float = 1.0
    # Jacobian is penalized via flow-like signals (rel_pred, z_dot_decode). Keep these large so J-related losses are not dwarfed by recon.
    weight_abs_pred: float = 50.0    # Dec(z_pred_next) vs Dec(z_next)
    weight_rel_pred: float = 2000.0    # Dec(J@du) vs Dec(delta_z) — key for flow/Jacobian (like image_jacobian flow)
    weight_z_dot_decode: float = 30.0  # Dec(delta_z) vs [I_dot, V] — direct flow signal


@register_algorithm("latent_jacobian", cfg_cls=LatentJacobianCfg)
class LatentJacobian(BasePytorchAlgo):
    cfg: LatentJacobianCfg
    model: LatentJacobianField

    def __init__(self, cfg: LatentJacobianCfg):
        super().__init__(cfg)

    def _build_model(self):
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
            LatentJacobianField,
            torch.compile(model, disable=not self.cfg.compile),
        )

    def configure_optimizers(self):
        params = list(self.model.parameters())
        optimizer = torch.optim.AdamW(
            params,
            lr=self.cfg.lr,
            weight_decay=self.cfg.optimizer.weight_decay,
            betas=tuple(self.cfg.optimizer.beta),
        )
        return {"optimizer": optimizer}

    def _get_pairs_from_batch(self, batch: Dict[str, Any]) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor], Tensor]:
        rgb = batch["rgb"]
        flow = batch.get("flow")
        du = batch["du"]

        if rgb.ndim == 6:
            B, T, V, C, H, W = rgb.shape
            rgb_curr = rgb[:, :-1]
            rgb_next = rgb[:, 1:]
            rgb_curr = flatten_time_view_dim(rgb_curr)
            rgb_next = flatten_time_view_dim(rgb_next)
            du = du[:, :-1]
            if du.ndim == 4:
                du = flatten_time_view_dim(du)
            else:
                du = flatten_time_dim(du)
            if flow is not None:
                flow_curr = flatten_time_view_dim(flow[:, :-1])
                flow_next = flatten_time_view_dim(flow[:, 1:])
            else:
                flow_curr = flow_next = None
        else:
            rgb_curr = rgb[:, :-1]
            rgb_next = rgb[:, 1:]
            rgb_curr = flatten_time_dim(rgb_curr)
            rgb_next = flatten_time_dim(rgb_next)
            du = flatten_time_dim(du[:, :-1])
            if flow is not None:
                flow_curr = flatten_time_dim(flow[:, :-1])
                flow_next = flatten_time_dim(flow[:, 1:])
            else:
                flow_curr = flow_next = None

        return rgb_curr, rgb_next, flow_curr, flow_next, du

    def _compute_loss(
        self, batch: Dict[str, Any], namespace: str = "training"
    ) -> Tuple[Dict[str, Tensor], Tensor, Dict[str, Any]]:
        rgb_curr, rgb_next, flow_curr, flow_next, du_gt = self._get_pairs_from_batch(batch)
        B = rgb_curr.shape[0]
        _, _, H, W = rgb_curr.shape

        model = self.model
        use_gt = self.cfg.forward_uses_gt_action

        z_t = model.encode(rgb_curr, flow_curr)
        z_next = model.encode(rgb_next, flow_next)
        z_t_flat = model.flatten_z(z_t)
        z_next_flat = model.flatten_z(z_next)
        delta_z_gt = z_next_flat - z_t_flat

        J_t = model.compute_jacobian_from_z(z_t)
        _, C, h, w = z_t.shape

        if use_gt:
            du = du_gt
            du_pred = model.solve_latent_action(z_t, delta_z_gt, J_t) if self.cfg.loss_backward_action else None
        else:
            du_pred = model.solve_latent_action(z_t, delta_z_gt, J_t)
            du = du_pred

        z_dot_pred = torch.einsum("bdu,bu->bd", J_t, du)
        z_pred_next_flat = z_t_flat + z_dot_pred
        z_pred_next = z_pred_next_flat.reshape(B, C, h, w)
        z_dot_gt = delta_z_gt
        z_dot_gt_reshaped = z_dot_gt.reshape(B, C, h, w)

        # Decode once for recon and z_dot targets; reuse in losses and sanity metrics.
        dec_z_t = model.decode(z_t, (H, W))
        dec_z_dot_gt = model.decode(z_dot_gt_reshaped, (H, W))

        losses: Dict[str, Tensor] = {}

        if self.cfg.loss_backward_action and du_pred is not None:
            losses["backward_action"] = self.cfg.weight_backward_action * F.mse_loss(du_pred, du_gt)

        if self.cfg.loss_abs_pred:
            dec_pred_next = model.decode(z_pred_next, (H, W))
            dec_z_next = model.decode(z_next, (H, W))
            losses["abs_pred"] = self.cfg.weight_abs_pred * F.l1_loss(dec_pred_next, dec_z_next)

        if self.cfg.loss_rel_pred:
            dec_z_dot_pred = model.decode(z_dot_pred.reshape(B, C, h, w), (H, W))
            losses["rel_pred"] = self.cfg.weight_rel_pred * F.l1_loss(dec_z_dot_pred, dec_z_dot_gt)

        if self.cfg.loss_z_dot_decode:
            I_dot = rgb_next - rgb_curr
            if flow_curr is not None:
                target = torch.cat([I_dot, flow_curr], dim=1)
            else:
                target = F.pad(I_dot, (0, 0, 0, 0, 0, 2), value=0.0)
            if dec_z_dot_gt.shape[1] != target.shape[1]:
                target = target[:, : dec_z_dot_gt.shape[1]]
            losses["z_dot_decode"] = self.cfg.weight_z_dot_decode * F.l1_loss(dec_z_dot_gt, target)

        if self.cfg.loss_recon:
            recon_target = rgb_curr
            if dec_z_t.shape[1] >= 3:
                losses["recon"] = self.cfg.weight_recon * F.l1_loss(
                    dec_z_t[:, :3], recon_target
                )

        total = sum(losses.values()) if losses else torch.tensor(0.0, device=z_t.device)

        model_output = {
            "z_t": z_t.detach(),
            "z_next": z_next.detach(),
            "J_t": J_t.detach(),
            "du_gt": du_gt.detach(),
            "dec_z": dec_z_t.detach(),
            "dec_z_dot": dec_z_dot_gt.detach(),
        }
        return losses, total, model_output

    def training_step(self, batch, batch_idx, namespace="training") -> STEP_OUTPUT:
        losses, total_loss, model_output = self._compute_loss(batch, namespace=namespace)

        if batch_idx % self.cfg.logging.loss_freq == 0:
            self.log(f"{namespace}/loss", total_loss, on_step=True, on_epoch=False, sync_dist=True)
            for name, value in losses.items():
                self.log(
                    f"{namespace}/loss_{name}",
                    value,
                    on_step=True,
                    on_epoch=False,
                    sync_dist=True,
                )
            for k, v in get_sanity_metrics(batch).items():
                self.log(f"sanity/input_{k}", v, on_step=True, on_epoch=False, sync_dist=True)
            for k, v in get_sanity_metrics(safe_asdict(model_output)).items():
                self.log(f"sanity/output_{k}", v, on_step=True, on_epoch=False, sync_dist=True)

        return {"loss": total_loss, "losses": losses, "model_output": model_output}

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        if (
            self.cfg.logging.grad_norm_freq
            and self.global_step % self.cfg.logging.grad_norm_freq == 0
        ):
            norms = grad_norm(self.model, norm_type=2)
            self.log_dict(norms)

    def _to_uint8_rgb(self, x: Tensor) -> np.ndarray:
        """(B, C, H, W), C>=3 -> (3, H, W) uint8 for wandb.Video (T, C, H, W). Values in [0, 1]."""
        if x.ndim == 3:
            x = x[None]
        arr = x[:, :3].float().cpu().numpy()
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255).astype(np.uint8)
        return np.ascontiguousarray(arr[0])  # (3, H, W)

    def _to_uint8_diff(self, x: Tensor) -> np.ndarray:
        """(B, C, H, W) difference (e.g. I_dot in [-1,1]) -> (3, H, W) uint8. 0 = gray (128)."""
        if x.ndim == 3:
            x = x[None]
        arr = x[:, :3].float().cpu().numpy()
        arr = (arr + 1.0) * 0.5
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255).astype(np.uint8)
        return np.ascontiguousarray(arr[0])  # (3, H, W)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        # All ranks must run _compute_loss and return the same dict so DDP collectives don't hang.
        losses, total_loss, model_output = self._compute_loss(batch, namespace="validation")
        self.log("validation/loss", total_loss, on_step=False, on_epoch=True, sync_dist=True)
        for name, value in losses.items():
            self.log(
                f"validation/loss_{name}",
                value,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        # Video visualization: only on rank 0, and only when not sanity/DDP/first-batch skip.
        # Optionally skip heavy vis during sanity check (set vis_on_sanity_check=False for faster startup).
        if getattr(self.trainer, "sanity_checking", False) and not getattr(
            self.cfg.logging, "vis_on_sanity_check", True
        ):
            return {"loss": total_loss, "losses": losses}
        # Video vis only on rank 0 (match image_jacobian: no sanity skip when vis_on_sanity_check, no batch_idx filter).
        if not self.trainer.is_global_zero:
            return {"loss": total_loss, "losses": losses}
        logger = getattr(self, "logger", None) or getattr(self.trainer, "logger", None)
        experiment = getattr(logger, "experiment", None) if logger else None
        if experiment is None:
            return {"loss": total_loss, "losses": losses}
        rgb = batch["rgb"]
        du = batch["du"]
        flow = batch.get("flow")
        if rgb.ndim == 6:
            B, T, V, C, H, W = rgb.shape
            rgb = rgb[:, :, 0]
            if flow is not None:
                flow = flow[:, :, 0]
        else:
            B, T, C, H, W = rgb.shape
        if T < 2:
            return {"loss": total_loss, "losses": losses}
        model = self.model
        use_gt_action = self.cfg.forward_uses_gt_action
        max_b = min(B, self.cfg.logging.max_num_videos)
        max_t = min(T - 1, self.cfg.logging.max_frames_per_video)
        video_captions = {
            "video/context": "Input frame (rgb_curr)",
            "video/decoded_I_dot": "Dec(z_dot_gt) → [I_dot; V] (decoder on latent delta)",
            "video/gt_I_dot": "Ground truth I_dot = rgb_next - rgb_curr (0 = gray)",
            "video/decoded_next": "Dec(z_next) decoded next latent",
            "video/decoded_pred_next": "Dec(z_t + J_t @ du) forward-predicted next",
            "video/pred_flow": "Predicted flow from Dec(z_dot)_ch3:5 (flow_to_image)",
            "video/gt_flow": "GT flow (flow_to_image)",
            "video/latent_jacobian": "J(z_t) latent Jacobian grid (resized for display)",
        }
        log_dict: Dict[str, Any] = {"trainer/global_step": self.global_step}
        # Target size for latent J grid so it's visible on wandb (match context H,W or min 128)
        jac_vis_target_hw = (max(H, 128), max(W, 128))
        for b in range(max_b):
            vis_dict: Dict[str, List[np.ndarray]] = {
                "video/context": [],
                "video/decoded_I_dot": [],
                "video/gt_I_dot": [],
                "video/decoded_next": [],
                "video/decoded_pred_next": [],
                "video/pred_flow": [],
                "video/gt_flow": [],
                "video/latent_jacobian": [],
            }
            for t in range(max_t):
                rgb_curr = rgb[b : b + 1, t]
                rgb_next = rgb[b : b + 1, t + 1]
                flow_curr = flow[b : b + 1, t] if flow is not None else None
                flow_next = flow[b : b + 1, t + 1] if flow is not None else None
                du_t = du[b : b + 1, t]
                z_t = model.encode(rgb_curr, flow_curr)
                z_next = model.encode(rgb_next, flow_next)
                _, C_enc, h, w = z_t.shape
                z_t_flat = model.flatten_z(z_t)
                z_next_flat = model.flatten_z(z_next)
                delta_z_gt = z_next_flat - z_t_flat
                J_t = model.compute_jacobian_from_z(z_t)
                z_dot_gt_reshaped = delta_z_gt.reshape(1, C_enc, h, w)
                decoded_I_dot = model.decode(z_dot_gt_reshaped, (H, W))
                decoded_next = model.decode(z_next, (H, W))
                I_dot_gt = rgb_next - rgb_curr
                # Forward prediction: z_pred_next = z_t + J_t @ du
                du_forward = du_t if use_gt_action else model.solve_latent_action(z_t, delta_z_gt, J_t)
                z_dot_pred = torch.einsum("bdu,bu->bd", J_t, du_forward)
                z_pred_next_flat = z_t_flat + z_dot_pred
                z_pred_next = z_pred_next_flat.reshape(1, C_enc, h, w)
                decoded_pred_next = model.decode(z_pred_next, (H, W))
                vis_dict["video/context"].append(self._to_uint8_rgb(rgb_curr))
                vis_dict["video/decoded_I_dot"].append(self._to_uint8_rgb(decoded_I_dot))
                vis_dict["video/gt_I_dot"].append(self._to_uint8_diff(I_dot_gt))
                vis_dict["video/decoded_next"].append(self._to_uint8_rgb(decoded_next))
                vis_dict["video/decoded_pred_next"].append(self._to_uint8_rgb(decoded_pred_next))
                # Predicted flow from decoder [I_dot; V]: channels 3:5 (if present)
                if decoded_I_dot.shape[1] >= 5:
                    pred_flow = denormalize_flow_tensor(
                        decoded_I_dot[:, 3:5].float(),
                        self.dataset_metadata,
                    )
                    vis_pred_flow = flow_to_image(pred_flow[0]).cpu().numpy()
                else:
                    vis_pred_flow = np.zeros((3, H, W), dtype=np.uint8)
                if flow_curr is not None:
                    gt_flow_phys = denormalize_flow_tensor(
                        flow_curr[0].float(),
                        self.dataset_metadata,
                    )
                    vis_gt_flow = flow_to_image(gt_flow_phys).cpu().numpy()
                else:
                    vis_gt_flow = np.zeros((3, H, W), dtype=np.uint8)
                vis_dict["video/pred_flow"].append(np.ascontiguousarray(vis_pred_flow))
                vis_dict["video/gt_flow"].append(np.ascontiguousarray(vis_gt_flow))
                # Latent Jacobian grid: resize to jac_vis_target_hw so it's visible on wandb
                dim_z_full, dim_u = J_t.shape[1], J_t.shape[2]
                max_viz_z = min(64, dim_z_full)
                J_slice = J_t[0, :max_viz_z, :].float().cpu().numpy()
                J_vis_input = J_slice[:, :, None, None]
                jac_vis = visualize_latent_jacobian(
                    J_vis_input,
                    dim_z=max_viz_z,
                    dim_u=dim_u,
                    cell_size=(8, 8),
                )
                if (jac_vis.shape[1], jac_vis.shape[2]) != jac_vis_target_hw:
                    jac_vis = rearrange(jac_vis, "c h w -> h w c")
                    jac_vis = cv2.resize(
                        jac_vis,
                        (jac_vis_target_hw[1], jac_vis_target_hw[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    jac_vis = rearrange(jac_vis, "h w c -> c h w")
                vis_dict["video/latent_jacobian"].append(np.ascontiguousarray(jac_vis))
            for k, vframes in vis_dict.items():
                stacked = np.stack(vframes, axis=0)
                # wandb.Video expects (T, C, H, W) = (num_frames, channels, height, width)
                if stacked.ndim == 4 and stacked.shape[-1] == 3:
                    stacked = np.ascontiguousarray(rearrange(stacked, "t h w c -> t c h w"))
                if getattr(self.cfg.logging, "verbose_vis", True):
                    t, c, h, w = stacked.shape[0], stacked.shape[1], stacked.shape[2], stacked.shape[3]
                    print(f"[LatentJacobian vis] {k}/b{b}: per-frame {vframes[0].shape if vframes else None} -> stacked (T={t}, C={c}, H={h}, W={w})")
                caption = video_captions.get(k, "")
                log_dict[f"{k}/b{b}"] = wandb.Video(
                    stacked,
                    fps=12,
                    format="mp4",
                    caption=caption,
                )
        # Match image_jacobian: log with trainer/global_step so wandb shows at current step (incl. 0 at sanity check).
        experiment.log({**log_dict, "trainer/global_step": self.global_step})
        return {"loss": total_loss, "losses": losses}

    def test_step(self, *args, **kwargs) -> STEP_OUTPUT:
        return self.validation_step(*args, **kwargs)
