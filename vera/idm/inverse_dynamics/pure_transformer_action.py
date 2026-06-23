from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from vera.idm.common.base_pytorch_algo import BasePytorchAlgo, BasePytorchAlgoCfg
from vera.idm.inverse_dynamics.models.pure_transformer_action_model import (
    PureTransformerActionModel,
)
from vera.idm.registry import register_algorithm
from vera.utils.pusht_action_visualization import build_pusht_action_overlay_video


@dataclass
class PureTransformerActionModelCfg:
    input_mode: Literal["rgb_pair", "flow", "rgb_flow"] = "rgb_pair"
    action_dim: int = 7
    patch_size: int = 14
    tokenizer_in_channels: int = 6
    embed_dim: int = 384
    depth: int = 6
    num_heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    drop_path_rate: float = 0.0
    head_hidden_dim: int = 256


@dataclass
class PureTransformerActionOptimizerCfg:
    weight_decay: float = 1e-4
    beta: list[float] = field(default_factory=lambda: [0.9, 0.999])


@dataclass
class PureTransformerActionLoggingCfg:
    loss_freq: int = 100
    max_validation_videos: int = 2
    validation_video_fps: int = 6


@dataclass
class PureTransformerActionCfg(BasePytorchAlgoCfg):
    name: Literal["pure_transformer_action"]
    image_size: list[int] = field(default_factory=lambda: [252, 252])
    model: PureTransformerActionModelCfg = field(
        default_factory=PureTransformerActionModelCfg
    )
    optimizer: PureTransformerActionOptimizerCfg = field(
        default_factory=PureTransformerActionOptimizerCfg
    )
    logging: PureTransformerActionLoggingCfg = field(
        default_factory=PureTransformerActionLoggingCfg
    )


@register_algorithm("pure_transformer_action", cfg_cls=PureTransformerActionCfg)
class PureTransformerAction(BasePytorchAlgo):
    cfg: PureTransformerActionCfg
    model: PureTransformerActionModel

    def _build_model(self) -> None:
        image_size = self.cfg.image_size
        if len(image_size) != 2:
            raise ValueError(f"Expected image_size [H,W], got {image_size}")
        self.model = PureTransformerActionModel(
            input_mode=self.cfg.model.input_mode,
            image_size=(int(image_size[0]), int(image_size[1])),
            action_dim=int(self.cfg.model.action_dim),
            patch_size=int(self.cfg.model.patch_size),
            tokenizer_in_channels=int(self.cfg.model.tokenizer_in_channels),
            embed_dim=int(self.cfg.model.embed_dim),
            depth=int(self.cfg.model.depth),
            num_heads=int(self.cfg.model.num_heads),
            mlp_ratio=float(self.cfg.model.mlp_ratio),
            dropout=float(self.cfg.model.dropout),
            drop_path_rate=float(self.cfg.model.drop_path_rate),
            head_hidden_dim=int(self.cfg.model.head_hidden_dim),
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.optimizer.weight_decay,
            betas=tuple(self.cfg.optimizer.beta),
        )

    @staticmethod
    def _select_first_rgb_frame(x: Tensor) -> Tensor:
        if x.ndim == 4:
            return x
        if x.ndim == 5:
            return x[:, 0]
        if x.ndim == 6:
            return x[:, 0, 0]
        raise ValueError(f"Unsupported rgb tensor shape: {tuple(x.shape)}")

    @staticmethod
    def _select_first_flow(x: Tensor) -> Tensor:
        if x.ndim == 4:
            return x
        if x.ndim == 5:
            return x[:, 0]
        if x.ndim == 6:
            return x[:, 0, 0]
        raise ValueError(f"Unsupported flow tensor shape: {tuple(x.shape)}")

    @classmethod
    def _select_rgb_pair(cls, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        if "rgb_src" in batch and "rgb_tgt" in batch:
            return (
                cls._select_first_rgb_frame(batch["rgb_src"]),
                cls._select_first_rgb_frame(batch["rgb_tgt"]),
            )
        if "rgb" not in batch:
            raise KeyError("Batch must contain either (rgb_src, rgb_tgt) or rgb.")
        rgb = batch["rgb"]
        if rgb.ndim == 6:
            rgb = rgb[:, :, 0]
        if rgb.ndim != 5 or rgb.shape[1] < 2:
            raise ValueError(
                f"Expected rgb with shape [B,T,C,H,W] or [B,T,V,C,H,W], got {tuple(batch['rgb'].shape)}"
            )
        return rgb[:, 0], rgb[:, 1]

    @classmethod
    def _select_flow_input(cls, batch: dict[str, Tensor]) -> Tensor:
        if "flow" in batch:
            return cls._select_first_flow(batch["flow"])
        if "optical_flow" in batch:
            return cls._select_first_flow(batch["optical_flow"])
        raise KeyError("Batch must contain flow or optical_flow for flow mode.")

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
            raise KeyError("Batch must contain action, du_chunk, or du.")

        if action.ndim != 2:
            raise ValueError(f"Expected target action [B,A], got {tuple(action.shape)}")
        if int(action.shape[1]) != int(action_dim):
            raise ValueError(
                f"Expected action_dim={action_dim}, got {int(action.shape[1])}"
            )
        return action

    def _compute_batch(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        action = self._select_target_action(
            batch,
            action_dim=int(self.cfg.model.action_dim),
        ).to(self.device)

        if self.cfg.model.input_mode == "rgb_pair":
            rgb_src, rgb_tgt = self._select_rgb_pair(batch)
            pred = self.model(rgb_src.to(self.device), rgb_tgt.to(self.device))
        elif self.cfg.model.input_mode == "rgb_flow":
            rgb_src, _ = self._select_rgb_pair(batch)
            flow = self._select_flow_input(batch)
            pred = self.model(rgb_src.to(self.device), flow.to(self.device))
        elif self.cfg.model.input_mode == "flow":
            flow = self._select_flow_input(batch)
            pred = self.model(flow.to(self.device))
        else:
            raise ValueError(f"Unsupported input_mode: {self.cfg.model.input_mode}")

        loss = F.mse_loss(pred, action)
        metrics = {
            "action_mse": loss.detach(),
            "pred_action_norm": pred.detach().norm(dim=-1).mean(),
            "target_action_norm": action.detach().norm(dim=-1).mean(),
        }
        return loss, metrics

    @staticmethod
    def _select_rgb_sequence(batch: dict[str, Tensor]) -> Tensor | None:
        if "rgb" not in batch:
            return None
        rgb = batch["rgb"]
        if rgb.ndim == 5:
            return rgb
        if rgb.ndim == 6:
            return rgb[:, :, 0]
        return None

    def _forward_prediction(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        action = self._select_target_action(
            batch,
            action_dim=int(self.cfg.model.action_dim),
        ).to(self.device)
        if self.cfg.model.input_mode == "rgb_pair":
            rgb_src, rgb_tgt = self._select_rgb_pair(batch)
            pred = self.model(rgb_src.to(self.device), rgb_tgt.to(self.device))
        elif self.cfg.model.input_mode == "rgb_flow":
            rgb_src, _ = self._select_rgb_pair(batch)
            flow = self._select_flow_input(batch)
            pred = self.model(rgb_src.to(self.device), flow.to(self.device))
        elif self.cfg.model.input_mode == "flow":
            flow = self._select_flow_input(batch)
            pred = self.model(flow.to(self.device))
        else:
            raise ValueError(f"Unsupported input_mode: {self.cfg.model.input_mode}")
        return pred, action

    def _maybe_log_validation_videos(
        self,
        batch: dict[str, Tensor],
        pred: Tensor,
        action: Tensor,
        *,
        namespace: str,
        batch_idx: int,
    ) -> None:
        if batch_idx != 0 or self.logger is None or not hasattr(self.logger, "experiment"):
            return
        trainer = getattr(self, "trainer", None)
        if trainer is not None and not trainer.is_global_zero:
            return
        if int(self.cfg.model.action_dim) != 2:
            return
        rgb = self._select_rgb_sequence(batch)
        if rgb is None:
            return
        max_videos = max(0, int(self.cfg.logging.max_validation_videos))
        if max_videos == 0:
            return

        num_videos = min(int(rgb.shape[0]), int(pred.shape[0]), int(action.shape[0]), max_videos)
        fps = int(self.cfg.logging.validation_video_fps)
        for sample_idx in range(num_videos):
            video = build_pusht_action_overlay_video(
                rgb_sequence=rgb[sample_idx],
                gt_action=action[sample_idx],
                pred_action=pred[sample_idx],
            )
            self.log_video(
                f"{namespace}/pusht_action_overlay_b{sample_idx}",
                video,
                fps=fps,
            )

    def _log_metrics(
        self,
        prefix: str,
        metrics: dict[str, Tensor],
        *,
        on_step: bool,
        on_epoch: bool,
    ) -> None:
        for key, value in metrics.items():
            self.log(
                f"{prefix}{key}",
                value,
                on_step=on_step,
                on_epoch=on_epoch,
                sync_dist=True,
                add_dataloader_idx=False,
            )

    def _validation_namespace(self, dataloader_idx: int = 0) -> str:
        datamodule = getattr(getattr(self, "trainer", None), "datamodule", None)
        names = getattr(datamodule, "validation_dataloader_names", None)
        if names and 0 <= int(dataloader_idx) < len(names):
            return f"validation/{names[int(dataloader_idx)]}"
        return "validation"

    def training_step(self, batch, batch_idx):
        loss, metrics = self._compute_batch(batch)
        self.log(
            "loss/training/total",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        if batch_idx % int(self.cfg.logging.loss_freq) == 0:
            self._log_metrics(
                "metrics/training/",
                metrics,
                on_step=True,
                on_epoch=False,
            )
        return {"loss": loss}

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        namespace = self._validation_namespace(dataloader_idx)
        pred, action = self._forward_prediction(batch)
        loss = F.mse_loss(pred, action)
        metrics = {
            "action_mse": loss.detach(),
            "pred_action_norm": pred.detach().norm(dim=-1).mean(),
            "target_action_norm": action.detach().norm(dim=-1).mean(),
        }
        self.log(
            f"loss/{namespace}/total",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            add_dataloader_idx=False,
        )
        self._log_metrics(
            f"metrics/{namespace}/",
            metrics,
            on_step=False,
            on_epoch=True,
        )
        self._maybe_log_validation_videos(
            batch,
            pred.detach(),
            action.detach(),
            namespace=namespace,
            batch_idx=batch_idx,
        )
        return {"loss": loss}

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        pred, action = self._forward_prediction(batch)
        loss = F.mse_loss(pred, action)
        metrics = {
            "action_mse": loss.detach(),
            "pred_action_norm": pred.detach().norm(dim=-1).mean(),
            "target_action_norm": action.detach().norm(dim=-1).mean(),
        }
        self.log(
            "loss/test/total",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self._log_metrics(
            "metrics/test/",
            metrics,
            on_step=False,
            on_epoch=True,
        )
        self._maybe_log_validation_videos(
            batch,
            pred.detach(),
            action.detach(),
            namespace="test",
            batch_idx=batch_idx,
        )
        return {"loss": loss}

    @torch.no_grad()
    def sample_action(
        self,
        primary: Tensor,
        secondary: Tensor | None = None,
    ) -> Tensor:
        return self.model(primary.to(self.device), None if secondary is None else secondary.to(self.device))
