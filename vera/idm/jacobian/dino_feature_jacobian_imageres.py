"""DINO-feature Jacobian (image-resolution variant).

Same idea as `dino_feature_jacobian.py` but uses the existing
`pure_transformer_jacobian_dino` model (which has a `LightweightConvDecoder`
that upsamples to image resolution) with `spatial_dim` repurposed to mean
the DINO feature dim.

Target: bilinearly-upsampled DINO patch features at image resolution.
Loss: per-pixel cosine direction (weighted by target magnitude) + relative
magnitude MSE + (optional) backward action loss with stop-gradient.

This was the v6 architecture that overfit-tested at cos=0.80 in 800 steps,
much better than the v3-v5 patch-resolution MLP-head architecture.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal

import torch
import torch.nn.functional as F
from einops import rearrange
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor

from vera.idm.common.base_pytorch_algo import BasePytorchAlgo, BasePytorchAlgoCfg
from vera.idm.jacobian.models.base import InputCommand, InputObservation
from vera.idm.jacobian.models.registry import (
    resolve_model_cfg,
    resolve_model_instance,
)
from vera.idm.registry import register_algorithm


@dataclass
class OptimizerCfg:
    name: Literal["adamw", "adam", "sgd"] = "adamw"
    lr: float = 3e-4
    weight_decay: float = 1e-4
    beta: List[float] = field(default_factory=lambda: [0.9, 0.99])


@dataclass
class LoggingCfg:
    loss_freq: int = 50


@dataclass
class DinoFeatureJacobianImageResCfg(BasePytorchAlgoCfg):
    name: Literal["dino_feature_jacobian_imageres"]
    robot_name: str = ""
    image_size: List[int] = field(default_factory=lambda: [252, 252])
    model: Any = field(default=None)

    optimizer: OptimizerCfg = field(default_factory=OptimizerCfg)
    logging: LoggingCfg = field(default_factory=LoggingCfg)

    forward_cos_weight: float = 1.0
    forward_mag_weight: float = 0.1
    backward_action_weight: float = 0.0  # off by default; backward at image res is expensive
    backward_action_damping: float = 1e-2
    head_init_std: float = 0.01


@register_algorithm("dino_feature_jacobian_imageres",
                     cfg_cls=DinoFeatureJacobianImageResCfg)
class DinoFeatureJacobianImageRes(BasePytorchAlgo):
    """DINO-feature Jacobian at image resolution, using existing conv-decoder model."""

    cfg: DinoFeatureJacobianImageResCfg

    def _build_model(self) -> None:
        model_cfg = resolve_model_cfg(self.cfg.model)
        self.model = resolve_model_instance(model_cfg)
        # Zero-init the head's last conv so initial J ≈ 0
        if hasattr(self.model, "head") and self.cfg.head_init_std > 0:
            with torch.no_grad():
                for m in self.model.head.modules():
                    if isinstance(m, torch.nn.Conv2d) and m.out_channels == self.model.command_dim * self.model.spatial_dim:
                        m.weight.data.normal_(0.0, float(self.cfg.head_init_std))
                        if m.bias is not None: m.bias.data.zero_()

    @torch.no_grad()
    def _frozen_dino_features_imageres(self, rgb: Tensor) -> Tensor:
        """Return (B, D, H, W) features by bilinear-upsampling DINO patches."""
        h = self.model.backbone.dino.forward_features(rgb)
        if isinstance(h, dict):
            patch = h["x_norm_patchtokens"]
        else:
            patch = h[:, 1:]
        gh, gw = self.model.backbone.grid_size
        feat = rearrange(patch, "b (gh gw) d -> b d gh gw", gh=gh, gw=gw)
        return F.interpolate(feat, size=(rgb.shape[-2], rgb.shape[-1]),
                             mode="bilinear", align_corners=False)

    def _compute_loss(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        rgb = batch["rgb"]
        du = batch["du"]
        if rgb.ndim != 5 or rgb.shape[1] < 2:
            raise ValueError(f"expected (B,T>=2,C,H,W), got {tuple(rgb.shape)}")
        rgb_curr, rgb_next = rgb[:, 0], rgb[:, 1]
        action = du[:, 0]

        out = self.model(InputObservation(rgb=rgb_curr), InputCommand(du=action))
        # out.optical_flow: (B, spatial_dim=D, H, W)
        pred = out.optical_flow

        target = (self._frozen_dino_features_imageres(rgb_next)
                  - self._frozen_dino_features_imageres(rgb_curr)).detach()
        # Match dtype to pred so cosine loss is consistent
        target = target.to(pred.dtype)

        losses: Dict[str, Tensor] = {}

        if self.cfg.forward_cos_weight > 0:
            pp = rearrange(pred, "b d h w -> (b h w) d")
            gt = rearrange(target, "b d h w -> (b h w) d")
            cos = F.cosine_similarity(pp, gt, dim=-1, eps=1e-6)
            w = gt.norm(dim=-1)
            w = w / (w.sum() + 1e-6)
            cos_loss = ((1 - cos) * w * cos.shape[0]).mean()
            losses["forward_cos"] = self.cfg.forward_cos_weight * cos_loss
            losses["_forward_cos_raw"] = cos.mean().detach()

            # cos at moving pixels diagnostic
            with torch.no_grad():
                tn_flat = gt.norm(dim=-1)
                topk = max(1, int(0.1 * tn_flat.numel()))
                _, idx = tn_flat.topk(topk)
                losses["_cos_at_moving"] = cos[idx].mean().detach()

        if self.cfg.forward_mag_weight > 0:
            pn = pred.float().norm(dim=1)
            tn = target.float().norm(dim=1)
            mag_loss = ((pn - tn) ** 2).mean() / (tn.pow(2).mean().clamp_min(1e-6))
            losses["forward_mag"] = self.cfg.forward_mag_weight * mag_loss

        if self.cfg.backward_action_weight > 0:
            # Backward at image resolution is expensive (gh*gw*D = 252*252*384 = 24M).
            # Subsample to gh=18, gw=18 patch resolution before solving.
            with torch.no_grad():
                gh_target = 18
                pred_lr = F.interpolate(pred, size=(gh_target, gh_target), mode="bilinear", align_corners=False)
                tgt_lr = F.interpolate(target, size=(gh_target, gh_target), mode="bilinear", align_corners=False)
            j = out.jacobian
            j_lr = F.interpolate(rearrange(j, "b c s h w -> b (c s) h w"),
                                 size=(gh_target, gh_target), mode="bilinear", align_corners=False)
            j_lr = rearrange(j_lr, "b (c s) gh gw -> b c s gh gw",
                             c=self.model.command_dim, s=self.model.spatial_dim)
            j_flat = rearrange(j_lr, "b c s gh gw -> b (s gh gw) c")
            t_flat = rearrange(tgt_lr.detach(), "b s gh gw -> b (s gh gw)").float()
            with torch.autocast("cuda", enabled=False):
                j32 = j_flat.float()
                jt_j = torch.einsum("bma,bmc->bac", j32, j32)
                jt_f = torch.einsum("bma,bm->ba", j32, t_flat)
                eye = torch.eye(self.model.command_dim, device=j32.device)
                diag_mean = torch.diagonal(jt_j, dim1=-2, dim2=-1).mean(-1).clamp_min(1.0)
                reg = float(self.cfg.backward_action_damping) * diag_mean[:, None, None] * eye[None]
                du_hat = torch.linalg.solve(jt_j + reg, jt_f.unsqueeze(-1)).squeeze(-1).to(action.dtype)
            losses["backward_action"] = self.cfg.backward_action_weight * F.smooth_l1_loss(du_hat, action)
            losses["_action_recon_mse"] = F.mse_loss(du_hat, action).detach()

        return losses

    def training_step(self, batch, batch_idx, namespace="training") -> STEP_OUTPUT:
        losses = self._compute_loss(batch)
        total = sum(v for k, v in losses.items() if not k.startswith("_"))
        if batch_idx % self.cfg.logging.loss_freq == 0:
            for k, v in losses.items():
                if torch.is_tensor(v):
                    self.log(f"{namespace}/{k}", v, on_step=True, on_epoch=False)
            self.log(f"{namespace}/total_loss", total, prog_bar=True, on_step=True, on_epoch=False)
        return {"loss": total, "losses": losses}

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0, namespace="validation"):
        losses = self._compute_loss(batch)
        total = sum(v for k, v in losses.items() if not k.startswith("_"))
        for k, v in losses.items():
            if torch.is_tensor(v):
                self.log(f"{namespace}/{k}", v, on_step=False, on_epoch=True)
        self.log(f"{namespace}/total_loss", total, prog_bar=True, on_step=False, on_epoch=True)
        return {"loss": total}

    def configure_optimizers(self):
        learnable = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.AdamW(learnable,
                                  lr=float(self.cfg.optimizer.lr),
                                  betas=tuple(self.cfg.optimizer.beta),
                                  weight_decay=float(self.cfg.optimizer.weight_decay))
