from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from vera.idm.common.base_pytorch_algo import BasePytorchAlgo, BasePytorchAlgoCfg
from vera.idm.inverse_dynamics.models.idm_transformer_model import (
    IDMTransformerModel,
)
from vera.idm.registry import register_algorithm


@dataclass
class IDMTransformerModelCfg:
    flow_channels: int = 2
    action_dim: int = 7
    backbone_preset: str = "base"
    neck_preset: str = "M"
    freeze_backbone: bool = True
    fusion_channels: int = 128
    prediction_hidden_channels: int = 64
    use_uv_coords: bool = False
    uv_fourier_frequencies: int = 10
    decoder_init_std: float | None = None
    decoder_feature_dim: int = 128
    mlp_hidden_dim: int = 256


@dataclass
class IDMTransformerOptimizerCfg:
    weight_decay: float = 1e-4
    beta: list[float] = field(default_factory=lambda: [0.9, 0.999])


@dataclass
class IDMTransformerCfg(BasePytorchAlgoCfg):
    name: Literal["idm_transformer"]
    image_size: list[int] = field(default_factory=lambda: [252, 252])
    model: IDMTransformerModelCfg = field(default_factory=IDMTransformerModelCfg)
    optimizer: IDMTransformerOptimizerCfg = field(
        default_factory=IDMTransformerOptimizerCfg
    )


@register_algorithm("idm_transformer", cfg_cls=IDMTransformerCfg)
class IDMTransformer(BasePytorchAlgo):
    cfg: IDMTransformerCfg
    model: IDMTransformerModel

    def _validate_config(self) -> None:
        if self.cfg.model.flow_channels < 1:
            raise ValueError("model.flow_channels must be >= 1.")
        if self.cfg.model.action_dim < 1:
            raise ValueError("model.action_dim must be >= 1.")

    def _build_model(self) -> None:
        self._validate_config()
        image_size = self.cfg.image_size
        if isinstance(image_size, list):
            if len(image_size) < 1:
                raise ValueError("image_size must contain at least one value.")
            image_size = image_size[0]
        self.model = IDMTransformerModel(
            image_size=int(image_size),
            flow_channels=self.cfg.model.flow_channels,
            action_dim=self.cfg.model.action_dim,
            backbone_preset=self.cfg.model.backbone_preset,
            neck_preset=self.cfg.model.neck_preset,
            freeze_backbone=self.cfg.model.freeze_backbone,
            fusion_channels=self.cfg.model.fusion_channels,
            prediction_hidden_channels=self.cfg.model.prediction_hidden_channels,
            use_uv_coords=self.cfg.model.use_uv_coords,
            uv_fourier_frequencies=self.cfg.model.uv_fourier_frequencies,
            decoder_init_std=self.cfg.model.decoder_init_std,
            decoder_feature_dim=self.cfg.model.decoder_feature_dim,
            mlp_hidden_dim=self.cfg.model.mlp_hidden_dim,
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.optimizer.weight_decay,
            betas=tuple(self.cfg.optimizer.beta),
        )

    @staticmethod
    def _select_flow_input(batch: dict[str, Tensor]) -> Tensor:
        for key in ("optical_flow", "flow"):
            if key not in batch:
                continue
            flow = batch[key]
            if flow.ndim == 4:
                return flow
            if flow.ndim == 5:
                if flow.shape[1] != 1:
                    raise ValueError(
                        f"Expected {key} to contain one flow map in [B,1,C,H,W], got {tuple(flow.shape)}"
                    )
                return flow
            raise ValueError(
                f"Expected {key} to be [B,C,H,W] or [B,1,C,H,W], got {tuple(flow.shape)}"
            )
        raise KeyError("Batch must contain either optical_flow or flow.")

    @staticmethod
    def _select_target_action(batch: dict[str, Tensor], action_dim: int) -> Tensor:
        if "action" in batch:
            action = batch["action"]
        elif "du_chunk" in batch:
            action = batch["du_chunk"]
            if action.ndim == 3:
                action = action[:, 0]
        elif "du" in batch:
            action = batch["du"]
            if action.ndim == 3:
                action = action[:, 0]
        else:
            raise KeyError("Batch must contain either action, du_chunk, or du.")

        if action.ndim != 2:
            raise ValueError(f"Expected target action shape [B,A], got {tuple(action.shape)}")
        if action.shape[1] != action_dim:
            raise ValueError(
                f"Expected target action dim={action_dim}, got {action.shape[1]}"
            )
        return action

    def _compute_batch(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        flow = self._select_flow_input(batch)
        action = self._select_target_action(batch, action_dim=self.cfg.model.action_dim).to(
            self.device
        )
        view_ids = batch.get("view_ids")
        if isinstance(view_ids, Tensor):
            view_ids = view_ids.to(self.device)
        pred = self.model(flow.to(self.device), view_ids=view_ids)
        loss = F.mse_loss(pred, action)
        metrics = {
            "action_mse": loss.detach(),
        }
        return loss, metrics

    def training_step(self, batch, batch_idx):
        loss, metrics = self._compute_batch(batch)
        self.log("loss/training/total", loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log(
            "metrics/training/action_mse",
            metrics["action_mse"],
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        return {"loss": loss}

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss, metrics = self._compute_batch(batch)
        self.log("loss/validation/total", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log(
            "metrics/validation/action_mse",
            metrics["action_mse"],
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return {"loss": loss}

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        loss, _ = self._compute_batch(batch)
        self.log("loss/test/total", loss, on_step=False, on_epoch=True, sync_dist=True)
        return {"loss": loss}

    @torch.no_grad()
    def sample_action(
        self,
        flow: Tensor,
        view_ids: Tensor | None = None,
    ) -> Tensor:
        return self.model(flow, view_ids=view_ids)
