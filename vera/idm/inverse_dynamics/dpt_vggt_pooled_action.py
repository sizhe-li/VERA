from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from vera.idm.common.base_pytorch_algo import BasePytorchAlgoCfg
from vera.idm.inverse_dynamics.models.dpt_vggt_pooled_action_model import (
    DptVggtPooledActionModel,
)
from vera.idm.inverse_dynamics.pure_transformer_action import (
    PureTransformerAction,
    PureTransformerActionLoggingCfg,
    PureTransformerActionOptimizerCfg,
)
from vera.idm.registry import register_algorithm


@dataclass
class DptVggtPooledActionModelCfg:
    input_mode: Literal["rgb_pair", "flow", "rgb_flow"] = "rgb_pair"
    action_dim: int = 7
    backbone_preset: str = "small"
    out_indices: Optional[Sequence[int]] = None
    neck_preset: Optional[str] = "M"
    neck_hidden_sizes: Optional[Sequence[int]] = None
    freeze_backbone: bool = True
    fusion_channels: int = 128
    head_hidden_dim: int = 256
    flow_adapter_hidden_dim: int = 32
    use_max_pool: bool = True


@dataclass
class DptVggtPooledActionCfg(BasePytorchAlgoCfg):
    name: Literal["dpt_vggt_pooled_action"]
    image_size: list[int] = field(default_factory=lambda: [252, 252])
    model: DptVggtPooledActionModelCfg = field(
        default_factory=DptVggtPooledActionModelCfg
    )
    optimizer: PureTransformerActionOptimizerCfg = field(
        default_factory=PureTransformerActionOptimizerCfg
    )
    logging: PureTransformerActionLoggingCfg = field(
        default_factory=PureTransformerActionLoggingCfg
    )


@register_algorithm("dpt_vggt_pooled_action", cfg_cls=DptVggtPooledActionCfg)
class DptVggtPooledAction(PureTransformerAction):
    cfg: DptVggtPooledActionCfg
    model: DptVggtPooledActionModel

    def _build_model(self) -> None:
        image_size = self.cfg.image_size
        if len(image_size) != 2:
            raise ValueError(f"Expected image_size [H,W], got {image_size}")
        self.model = DptVggtPooledActionModel(
            input_mode=self.cfg.model.input_mode,
            image_size=(int(image_size[0]), int(image_size[1])),
            action_dim=int(self.cfg.model.action_dim),
            backbone_preset=str(self.cfg.model.backbone_preset),
            out_indices=self.cfg.model.out_indices,
            neck_preset=self.cfg.model.neck_preset,
            neck_hidden_sizes=self.cfg.model.neck_hidden_sizes,
            freeze_backbone=bool(self.cfg.model.freeze_backbone),
            fusion_channels=int(self.cfg.model.fusion_channels),
            head_hidden_dim=int(self.cfg.model.head_hidden_dim),
            flow_adapter_hidden_dim=int(self.cfg.model.flow_adapter_hidden_dim),
            use_max_pool=bool(self.cfg.model.use_max_pool),
        )

    @staticmethod
    def _tensor_finite_summary(name: str, value: Tensor) -> str:
        detached = value.detach()
        finite_mask = torch.isfinite(detached)
        finite = detached[finite_mask]
        bad = int((~finite_mask).sum().item())
        total = int(detached.numel())
        if finite.numel() == 0:
            return (
                f"{name}: shape={tuple(detached.shape)} dtype={detached.dtype} "
                f"bad={bad}/{total} finite=none"
            )
        finite_f32 = finite.float()
        return (
            f"{name}: shape={tuple(detached.shape)} dtype={detached.dtype} "
            f"bad={bad}/{total} min={finite_f32.min().item():.6g} "
            f"max={finite_f32.max().item():.6g} mean={finite_f32.mean().item():.6g} "
            f"std={finite_f32.std(unbiased=False).item():.6g}"
        )

    @classmethod
    def _require_finite(cls, name: str, value: Tensor) -> None:
        if torch.isfinite(value).all():
            return
        raise FloatingPointError(cls._tensor_finite_summary(name, value))

    @staticmethod
    def _flatten_batch_time(value: Tensor) -> Tensor:
        if value.ndim < 3:
            raise ValueError(f"Expected batch/time tensor, got {tuple(value.shape)}")
        leading = int(value.shape[0]) * int(value.shape[1])
        return value.contiguous().reshape(leading, *value.shape[2:])

    @staticmethod
    def _select_rgb_sequence(batch: dict[str, Tensor]) -> Tensor:
        if "rgb" not in batch:
            raise KeyError("Batch must contain rgb for sequence supervision.")
        rgb = batch["rgb"]
        if rgb.ndim == 6:
            rgb = rgb[:, :, 0]
        if rgb.ndim == 4:
            rgb = rgb[:, None]
        if rgb.ndim != 5:
            raise ValueError(
                f"Expected rgb [B,T,C,H,W] or [B,T,V,C,H,W], got {tuple(batch['rgb'].shape)}"
            )
        return rgb

    @staticmethod
    def _select_flow_sequence(batch: dict[str, Tensor]) -> Tensor:
        if "flow" in batch:
            flow = batch["flow"]
        elif "optical_flow" in batch:
            flow = batch["optical_flow"]
        else:
            raise KeyError("Batch must contain flow or optical_flow for flow mode.")

        if flow.ndim == 6:
            flow = flow[:, :, 0]
        if flow.ndim == 4:
            flow = flow[:, None]
        if flow.ndim != 5:
            raise ValueError(
                f"Expected flow [B,T,2,H,W] or [B,T,V,2,H,W], got {tuple(flow.shape)}"
            )
        return flow

    def _select_target_action_sequence(
        self, batch: dict[str, Tensor], num_steps: int
    ) -> Tensor:
        if "action" in batch:
            action = batch["action"]
        elif "du_chunk" in batch:
            action = batch["du_chunk"]
        elif "du" in batch:
            action = batch["du"]
        else:
            raise KeyError("Batch must contain action, du_chunk, or du.")

        if action.ndim == 3:
            if int(action.shape[1]) < int(num_steps):
                raise ValueError(
                    f"Expected at least {num_steps} action steps, got {tuple(action.shape)}"
                )
            action = action[:, :num_steps]
            action = self._flatten_batch_time(action)
        elif action.ndim == 2:
            if int(num_steps) != 1:
                raise ValueError(
                    f"Cannot use unsequenced action {tuple(action.shape)} for {num_steps} steps"
                )
        else:
            raise ValueError(f"Expected target action [B,A] or [B,T,A], got {tuple(action.shape)}")

        if int(action.shape[1]) != int(self.cfg.model.action_dim):
            raise ValueError(
                f"Expected action_dim={int(self.cfg.model.action_dim)}, got {int(action.shape[1])}"
            )
        return action.to(self.device)

    def _compute_batch(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        pred, action = self._forward_prediction(batch)
        loss = F.mse_loss(pred, action)
        self._require_finite("loss/action_mse", loss)
        metrics = {
            "action_mse": loss.detach(),
            "pred_action_norm": pred.detach().norm(dim=-1).mean(),
            "target_action_norm": action.detach().norm(dim=-1).mean(),
        }
        return loss, metrics

    def _forward_prediction(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        if self.cfg.model.input_mode == "rgb_pair":
            if "rgb_src" in batch and "rgb_tgt" in batch:
                rgb_src, rgb_tgt = self._select_rgb_pair(batch)
                num_steps = 1
            else:
                rgb = self._select_rgb_sequence(batch)
                if int(rgb.shape[1]) < 2:
                    raise ValueError(
                        f"rgb_pair sequence supervision needs at least 2 frames, got {tuple(rgb.shape)}"
                    )
                num_steps = int(rgb.shape[1]) - 1
                rgb_src = self._flatten_batch_time(rgb[:, :num_steps])
                rgb_tgt = self._flatten_batch_time(rgb[:, 1 : num_steps + 1])
            action = self._select_target_action_sequence(batch, num_steps)
            rgb_src = rgb_src.to(self.device)
            rgb_tgt = rgb_tgt.to(self.device)
            self._require_finite("rgb_src", rgb_src)
            self._require_finite("rgb_tgt", rgb_tgt)
            pred = self.model(rgb_src, rgb_tgt)
        elif self.cfg.model.input_mode == "rgb_flow":
            rgb = self._select_rgb_sequence(batch)
            flow = self._select_flow_sequence(batch)
            if int(rgb.shape[1]) < 2:
                raise ValueError(
                    f"rgb_flow sequence supervision needs at least 2 RGB frames, got {tuple(rgb.shape)}"
                )
            num_steps = min(int(rgb.shape[1]) - 1, int(flow.shape[1]))
            rgb_src = self._flatten_batch_time(rgb[:, :num_steps])
            flow = self._flatten_batch_time(flow[:, :num_steps])
            action = self._select_target_action_sequence(batch, num_steps)
            rgb_src = rgb_src.to(self.device)
            flow = flow.to(self.device)
            self._require_finite("rgb_src", rgb_src)
            self._require_finite("flow", flow)
            pred = self.model(rgb_src, flow)
        elif self.cfg.model.input_mode == "flow":
            flow = self._select_flow_sequence(batch)
            if flow.ndim != 5:
                raise ValueError(f"Expected flow sequence, got {tuple(flow.shape)}")
            num_steps = int(flow.shape[1])
            flow = self._flatten_batch_time(flow)
            action = self._select_target_action_sequence(batch, num_steps)
            flow = flow.to(self.device)
            self._require_finite("flow", flow)
            pred = self.model(flow)
        else:
            raise ValueError(f"Unsupported input_mode: {self.cfg.model.input_mode}")

        self._require_finite("pred_action", pred)
        return pred, action
