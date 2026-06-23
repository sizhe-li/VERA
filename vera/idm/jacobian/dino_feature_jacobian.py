"""DINO-feature Jacobian algorithm.

A minimal training algorithm for the DINO-feature Jacobian field. The loss
is MSE between the predicted feature delta J(o_t) @ du and the actual
DINO feature delta dino(o_{t+1}) - dino(o_t).

DINO is frozen throughout. Gradients flow only into the small transformer
head + MLP that sit between DINO and the per-patch Jacobian output.

Optionally adds a backward action recovery loss with stop-gradient on the
target (so backward action does not contaminate the encoder).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

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
    lr: float = 1e-4
    weight_decay: float = 1e-4
    beta: List[float] = field(default_factory=lambda: [0.9, 0.99])


@dataclass
class LoggingCfg:
    loss_freq: int = 50


@dataclass
class DinoFeatureJacobianCfg(BasePytorchAlgoCfg):
    name: Literal["dino_feature_jacobian"]
    robot_name: str = ""
    image_size: List[int] = field(default_factory=lambda: [252, 252])
    model: Any = field(default=None)

    optimizer: OptimizerCfg = field(default_factory=OptimizerCfg)
    logging: LoggingCfg = field(default_factory=LoggingCfg)

    # Loss weights. After two failed attempts (collapse + divergence under
    # EMA target normalization), default to scale-invariant cosine loss
    # plus a magnitude-matching loss + backward action loss.
    forward_cos_weight: float = 1.0       # 1 - cos(pred, target) per-pixel mean
    forward_mag_weight: float = 0.1       # MSE on per-pixel norms (scale-invariant via target_norm)
    backward_action_weight: float = 1.0   # smooth_L1(J^+ @ stop_grad(target), du)
    backward_action_damping: float = 1e-2

    # Old-style L2 in normalized feature space, kept for back-compat. Off by default now.
    forward_feature_weight: float = 0.0
    target_normalize_per_dim: bool = False
    target_norm_momentum: float = 0.99
    # Initialize the head's final linear so initial J ≈ 0 (avoids huge random predictions).
    head_init_std: float = 0.01


@register_algorithm("dino_feature_jacobian", cfg_cls=DinoFeatureJacobianCfg)
class DinoFeatureJacobian(BasePytorchAlgo):
    """Train a per-patch Jacobian J ∈ R^(D × A) on PushT (frozen DINO MVP)."""

    cfg: DinoFeatureJacobianCfg

    def _build_model(self) -> None:
        model_cfg = resolve_model_cfg(self.cfg.model)
        self.model = resolve_model_instance(model_cfg)
        D = int(model_cfg.spatial_dim)
        self.register_buffer("target_dim_std", torch.ones(D))

        # Re-init the head's last linear with small std so initial J ≈ 0.
        if hasattr(self.model, "head") and self.cfg.head_init_std > 0:
            head = self.model.head
            last_linear = None
            if isinstance(head, torch.nn.Sequential):
                for m in head:
                    if isinstance(m, torch.nn.Linear):
                        last_linear = m
            elif isinstance(head, torch.nn.Linear):
                last_linear = head
            if last_linear is not None:
                with torch.no_grad():
                    last_linear.weight.data.normal_(0.0, float(self.cfg.head_init_std))
                    if last_linear.bias is not None:
                        last_linear.bias.data.zero_()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _frozen_dino_features(self, rgb: Tensor) -> Tensor:
        """Compute (B, gh, gw, D) DINO patch features for `rgb` with no grad."""
        with torch.no_grad():
            hidden = self.model.backbone.dino.forward_features(rgb)
            # dinov2 returns dict with "x_norm_patchtokens" or similar; fall back
            if isinstance(hidden, dict):
                if "x_norm_patchtokens" in hidden:
                    patch = hidden["x_norm_patchtokens"]
                elif "x_prenorm" in hidden:
                    patch = hidden["x_prenorm"][:, 1:]  # drop CLS
                else:
                    raise KeyError(
                        f"Unexpected DINO output keys: {list(hidden.keys())}"
                    )
            else:
                # Tensor: assume includes CLS at position 0.
                patch = hidden[:, 1:]
        gh, gw = self.model.backbone.grid_size
        return rearrange(patch, "b (gh gw) d -> b gh gw d", gh=gh, gw=gw)

    @staticmethod
    def _solve_action_least_squares(
        jacobian_flat: Tensor,
        feature_delta_flat: Tensor,
        damping: float,
    ) -> Tensor:
        """Solve (J^T J + λI) du = J^T δf for du.

        jacobian_flat: (B, M, A) where M = (gh*gw) * D (flattened patch×feat).
        feature_delta_flat: (B, M).
        Returns: (B, A).
        """
        # torch.linalg.solve doesn't support fp16; force fp32 + disable autocast.
        orig_dtype = jacobian_flat.dtype
        with torch.autocast("cuda", enabled=False):
            jf = jacobian_flat.float()
            ff = feature_delta_flat.float()
            bsz, m, a_dim = jf.shape
            jt_j = torch.einsum("bma,bmc->bac", jf, jf)
            jt_f = torch.einsum("bma,bm->ba", jf, ff)
            eye = torch.eye(a_dim, device=jf.device, dtype=jf.dtype)
            diag_mean = torch.diagonal(jt_j, dim1=-2, dim2=-1).mean(-1).clamp_min(1.0)
            regularizer = float(damping) * diag_mean[:, None, None] * eye[None]
            du_hat = torch.linalg.solve(jt_j + regularizer, jt_f.unsqueeze(-1)).squeeze(-1)
        return du_hat.to(orig_dtype)

    # ------------------------------------------------------------------
    # Forward / loss
    # ------------------------------------------------------------------
    def _compute_loss(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        # batch["rgb"]: (B, T, C, H, W) where T=2 (current, next)
        # batch["du"]:  (B, T, A)
        rgb = batch["rgb"]
        du = batch["du"]

        if rgb.ndim != 5 or rgb.shape[1] < 2:
            raise ValueError(
                f"DinoFeatureJacobian expects rgb of shape (B, T>=2, C, H, W); got {tuple(rgb.shape)}"
            )

        rgb_curr = rgb[:, 0]
        rgb_next = rgb[:, 1]
        action = du[:, 0]  # action that takes curr -> next

        # Predict per-patch Jacobian and feature delta.
        obs = InputObservation(rgb=rgb_curr)
        cmd = InputCommand(du=action)
        out = self.model.forward(obs, cmd)
        # out.jacobian:    (B, D, A, gh, gw)
        # out.optical_flow: (B, D, gh, gw)  — we repurpose this slot for predicted δfeature
        feature_delta_pred = out.optical_flow

        # Target: dino(rgb_next) - dino(rgb_curr) at patch resolution
        feat_curr = self._frozen_dino_features(rgb_curr)  # (B, gh, gw, D)
        feat_next = self._frozen_dino_features(rgb_next)
        feature_delta_gt = (feat_next - feat_curr).detach()  # (B, gh, gw, D)
        # Reshape to match predicted layout (B, D, gh, gw)
        feature_delta_gt = rearrange(feature_delta_gt, "b gh gw d -> b d gh gw")

        losses: Dict[str, Tensor] = {}

        # ---- (A) cosine forward loss (scale-invariant, primary objective) ----
        # Per-pixel cosine sim, weighted by per-pixel target magnitude so the
        # loss focuses on actually-moving patches.
        if self.cfg.forward_cos_weight > 0:
            pred_pp = rearrange(feature_delta_pred, "b d gh gw -> (b gh gw) d")
            gt_pp = rearrange(feature_delta_gt, "b d gh gw -> (b gh gw) d")
            cos = F.cosine_similarity(pred_pp, gt_pp, dim=-1, eps=1e-6)
            # weight by target magnitude so static patches don't dominate
            w = gt_pp.norm(dim=-1)
            w = w / (w.sum() + 1e-6)
            cos_loss = (1.0 - cos) * w * cos.shape[0]  # rescale so mean weight ~1
            cos_loss = cos_loss.mean()
            losses["forward_cos"] = self.cfg.forward_cos_weight * cos_loss
            losses["_forward_cos_raw"] = cos.mean().detach()  # unweighted mean

        # ---- (B) magnitude matching (per-pixel norm) ----
        if self.cfg.forward_mag_weight > 0:
            pred_norm = feature_delta_pred.float().norm(dim=1)  # (B, gh, gw)
            gt_norm = feature_delta_gt.float().norm(dim=1)
            # scale-invariant: relative MSE
            mag_loss = ((pred_norm - gt_norm) ** 2).mean() / (gt_norm.pow(2).mean().clamp_min(1e-6))
            losses["forward_mag"] = self.cfg.forward_mag_weight * mag_loss
            losses["_pred_norm_mean"] = pred_norm.mean().detach()
            losses["_gt_norm_mean"] = gt_norm.mean().detach()

        # ---- (C) optional legacy MSE in (optionally normalized) space ----
        if self.cfg.forward_feature_weight > 0:
            if self.cfg.target_normalize_per_dim:
                with torch.no_grad():
                    cur_std = feature_delta_gt.float().std(dim=(0, 2, 3)).clamp_min(1e-3)
                    m = float(self.cfg.target_norm_momentum)
                    self.target_dim_std.mul_(m).add_(cur_std, alpha=1 - m)
                std_b = self.target_dim_std.view(1, -1, 1, 1).to(feature_delta_gt.dtype)
                fpd_n = feature_delta_pred / std_b
                fgt_n = feature_delta_gt / std_b
            else:
                fpd_n, fgt_n = feature_delta_pred, feature_delta_gt
            loss_forward = F.mse_loss(fpd_n, fgt_n)
            losses["forward_feature"] = self.cfg.forward_feature_weight * loss_forward
            losses["_raw_forward_feature_mse"] = loss_forward.detach()

        # Backward action loss: J^+ @ δfeature should recover the true action.
        # Per user note: stop-gradient on target so this path does NOT leak
        # gradients back through the (frozen) DINO encoder; only the Jacobian
        # head is updated by it.
        if self.cfg.backward_action_weight > 0:
            jacobian = out.jacobian  # (B, D, A, gh, gw)
            jacobian_flat = rearrange(
                jacobian, "b d a gh gw -> b (d gh gw) a"
            )
            target_flat = rearrange(feature_delta_gt.detach(), "b d gh gw -> b (d gh gw)")
            du_hat = self._solve_action_least_squares(
                jacobian_flat,
                target_flat,
                damping=float(self.cfg.backward_action_damping),
            )
            loss_backward = F.smooth_l1_loss(du_hat, action)
            losses["backward_action"] = (
                self.cfg.backward_action_weight * loss_backward
            )
            losses["_raw_backward_action_mse"] = F.mse_loss(du_hat, action).detach()

        # Diagnostics
        with torch.no_grad():
            # cosine similarity on patch-flattened deltas
            pred_flat = feature_delta_pred.flatten(1)
            tgt_flat = feature_delta_gt.flatten(1)
            cos = F.cosine_similarity(pred_flat, tgt_flat, dim=-1).mean()
            losses["_cos_sim_pred_vs_gt"] = cos.detach()
            losses["_target_delta_norm"] = tgt_flat.norm(dim=-1).mean().detach()

        return losses

    def training_step(self, batch, batch_idx, namespace="training") -> STEP_OUTPUT:
        losses = self._compute_loss(batch)
        total = sum(v for k, v in losses.items() if not k.startswith("_"))
        if batch_idx % self.cfg.logging.loss_freq == 0:
            for k, v in losses.items():
                if torch.is_tensor(v):
                    self.log(f"{namespace}/{k}", v, prog_bar=False, on_step=True, on_epoch=False)
            self.log(f"{namespace}/total_loss", total, prog_bar=True, on_step=True, on_epoch=False)
        return {"loss": total, "losses": losses}

    @torch.no_grad()
    def validation_step(
        self,
        batch,
        batch_idx,
        dataloader_idx: int = 0,
        namespace: str = "validation",
    ) -> STEP_OUTPUT:
        losses = self._compute_loss(batch)
        total = sum(v for k, v in losses.items() if not k.startswith("_"))
        for k, v in losses.items():
            if torch.is_tensor(v):
                self.log(f"{namespace}/{k}", v, prog_bar=False, on_step=False, on_epoch=True)
        self.log(f"{namespace}/total_loss", total, prog_bar=True, on_step=False, on_epoch=True)
        # also log action-recon MSE if backward isn't already computed
        if self.cfg.backward_action_weight == 0.0:
            jacobian = self.model.compute_jacobian(InputObservation(rgb=batch["rgb"][:, 0]))
            feat_curr = self._frozen_dino_features(batch["rgb"][:, 0])
            feat_next = self._frozen_dino_features(batch["rgb"][:, 1])
            target = (feat_next - feat_curr)
            jacobian_flat = rearrange(jacobian, "b d a gh gw -> b (d gh gw) a")
            target_flat = rearrange(
                rearrange(target, "b gh gw d -> b d gh gw"),
                "b d gh gw -> b (d gh gw)",
            )
            du_hat = self._solve_action_least_squares(
                jacobian_flat, target_flat, damping=float(self.cfg.backward_action_damping)
            )
            mse = F.mse_loss(du_hat, batch["du"][:, 0])
            self.log(f"{namespace}/action_recon_mse", mse, prog_bar=True, on_step=False, on_epoch=True)
        return {"loss": total}

    # ------------------------------------------------------------------
    # Optimizer (only learnable params: head + transformer above frozen DINO)
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        learnable = [p for p in self.model.parameters() if p.requires_grad]
        if self.cfg.optimizer.name == "adamw":
            opt = torch.optim.AdamW(
                learnable,
                lr=float(self.cfg.optimizer.lr),
                betas=tuple(self.cfg.optimizer.beta),
                weight_decay=float(self.cfg.optimizer.weight_decay),
            )
        elif self.cfg.optimizer.name == "adam":
            opt = torch.optim.Adam(
                learnable,
                lr=float(self.cfg.optimizer.lr),
                betas=tuple(self.cfg.optimizer.beta),
                weight_decay=float(self.cfg.optimizer.weight_decay),
            )
        else:
            opt = torch.optim.SGD(learnable, lr=float(self.cfg.optimizer.lr), momentum=0.9)
        return opt
