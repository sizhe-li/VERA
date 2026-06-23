"""Chunked DINO-feature Jacobian algorithm.

Trains a chunked per-patch Jacobian J ∈ R^(T × D × A) such that for an
anchor observation o_0 and a chunk of actions du_0..du_{T-1}:

    δfeature_pred[t, p] = J[t, p] @ du_t   ≈   dino(o_{t+1})[p] − dino(o_t)[p]

(Per-step delta interpretation. We could alternatively supervise
cumulative deltas; per-step keeps the semantics local and adds up to the
same total along a chunk.)

DINO frozen. Loss is the per-timestep MSE in feature space, summed.
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
    lr: float = 1e-4
    weight_decay: float = 1e-4
    beta: List[float] = field(default_factory=lambda: [0.9, 0.99])


@dataclass
class LoggingCfg:
    loss_freq: int = 50


@dataclass
class DinoChunkJacobianCfg(BasePytorchAlgoCfg):
    name: Literal["dino_chunk_jacobian"]
    robot_name: str = ""
    image_size: List[int] = field(default_factory=lambda: [252, 252])
    model: Any = field(default=None)

    optimizer: OptimizerCfg = field(default_factory=OptimizerCfg)
    logging: LoggingCfg = field(default_factory=LoggingCfg)

    forward_feature_weight: float = 1.0
    backward_action_weight: float = 0.0
    backward_action_damping: float = 1e-2
    target_normalize_per_dim: bool = False


@register_algorithm("dino_chunk_jacobian", cfg_cls=DinoChunkJacobianCfg)
class DinoChunkJacobian(BasePytorchAlgo):
    """Train chunked per-patch Jacobian J ∈ R^(T × D × A) on PushT/MimicGen."""

    cfg: DinoChunkJacobianCfg

    def _build_model(self) -> None:
        model_cfg = resolve_model_cfg(self.cfg.model)
        self.model = resolve_model_instance(model_cfg)

    def _frozen_dino_features(self, rgb: Tensor) -> Tensor:
        """rgb: (B*T, 3, H, W) -> (B*T, gh, gw, D)."""
        with torch.no_grad():
            hidden = self.model.backbone.dino.forward_features(rgb)
            if isinstance(hidden, dict):
                if "x_norm_patchtokens" in hidden:
                    patch = hidden["x_norm_patchtokens"]
                else:
                    patch = hidden["x_prenorm"][:, 1:]
            else:
                patch = hidden[:, 1:]
        gh, gw = self.model.backbone.grid_size
        return rearrange(patch, "b (gh gw) d -> b gh gw d", gh=gh, gw=gw)

    def _compute_loss(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        rgb = batch["rgb"]   # (B, T, 3, H, W) — anchor at t=0; chunk frames 1..T-1
        du = batch["du"]      # (B, T, A) — actions du[t] takes rgb[t] -> rgb[t+1]
        T_full = rgb.shape[1]
        T_chunk = self.model.chunk_length

        if T_full < T_chunk + 1:
            raise ValueError(
                f"chunk_length={T_chunk} requires at least T+1={T_chunk+1} frames in batch; "
                f"got T={T_full}. Set dataset.window_length accordingly."
            )

        rgb_anchor = rgb[:, 0]  # (B, 3, H, W)
        # action chunk used to roll the anchor forward
        du_chunk = du[:, :T_chunk]  # (B, T, A)

        # Predict per-step feature deltas across the chunk from the anchor.
        out = self.model.forward(InputObservation(rgb=rgb_anchor),
                                 InputCommand(du=du_chunk))
        # jacobian:        (B, T, D, A, gh, gw)
        # optical_flow:    (B, T*D, gh, gw)  (predicted per-step δfeature, flat)
        b = rgb.shape[0]
        D = self.model.feature_dim
        gh, gw = self.model.backbone.grid_size
        feature_delta_pred = out.optical_flow.view(b, T_chunk, D, gh, gw)

        # Compute target per-step feature deltas using frozen DINO.
        # Stack frames 0..T to one big batch so we run DINO once.
        rgb_seq = rgb[:, : T_chunk + 1]  # (B, T+1, 3, H, W)
        rgb_flat = rearrange(rgb_seq, "b t c h w -> (b t) c h w")
        feat_flat = self._frozen_dino_features(rgb_flat)  # ((B*(T+1)), gh, gw, D)
        feat_seq = feat_flat.view(b, T_chunk + 1, gh, gw, D)
        # per-step delta: feat[t+1] - feat[t]
        feature_delta_gt = feat_seq[:, 1:] - feat_seq[:, :-1]  # (B, T, gh, gw, D)
        feature_delta_gt = rearrange(feature_delta_gt, "b t gh gw d -> b t d gh gw").detach()

        if self.cfg.target_normalize_per_dim:
            std = feature_delta_gt.flatten(0, 1).std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-3)
            feature_delta_gt = feature_delta_gt / std
            feature_delta_pred = feature_delta_pred / std

        loss_forward = F.mse_loss(feature_delta_pred, feature_delta_gt)

        losses: Dict[str, Tensor] = {
            "forward_feature": self.cfg.forward_feature_weight * loss_forward,
            "_raw_forward_feature_mse": loss_forward.detach(),
        }

        # Per-timestep MSE diagnostic
        with torch.no_grad():
            for t in range(T_chunk):
                losses[f"_step{t}_mse"] = F.mse_loss(
                    feature_delta_pred[:, t], feature_delta_gt[:, t]
                ).detach()
            pred_flat = feature_delta_pred.flatten(2)
            tgt_flat = feature_delta_gt.flatten(2)
            cos = F.cosine_similarity(pred_flat, tgt_flat, dim=-1).mean()
            losses["_cos_sim_pred_vs_gt"] = cos.detach()

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
        opt = torch.optim.AdamW(
            learnable,
            lr=float(self.cfg.optimizer.lr),
            betas=tuple(self.cfg.optimizer.beta),
            weight_decay=float(self.cfg.optimizer.weight_decay),
        )
        return opt
