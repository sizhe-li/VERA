from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from vera.idm.common.base_pytorch_algo import BasePytorchAlgoCfg
from vera.idm.inverse_dynamics.models.pure_transformer_dino_action_model import (
    PureTransformerDinoActionModel,
)
from vera.idm.inverse_dynamics.pure_transformer_action import (
    PureTransformerAction,
)
from vera.idm.registry import register_algorithm


@dataclass
class PureTransformerDinoActionModelCfg:
    input_mode: Literal["rgb_pair", "flow", "rgb_flow"] = "rgb_pair"
    action_dim: int = 7
    dino_model_name: str = "dinov2_vits14"
    freeze_dino: bool = True
    embed_dim: int = 384
    depth: int = 6
    num_heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    drop_path_rate: float = 0.0
    head_hidden_dim: int = 256
    flow_adapter_hidden_dim: int = 32
    rgb_pair_include_difference_tokens: bool = False


@dataclass
class PureTransformerDinoActionOptimizerCfg:
    weight_decay: float = 1e-4
    beta: list[float] = field(default_factory=lambda: [0.9, 0.999])


@dataclass
class PureTransformerDinoActionLoggingCfg:
    loss_freq: int = 100
    max_validation_videos: int = 2
    validation_video_fps: int = 6


@dataclass
class PureTransformerDinoActionCfg(BasePytorchAlgoCfg):
    name: Literal["pure_transformer_dino_action"]
    image_size: list[int] = field(default_factory=lambda: [252, 252])
    model: PureTransformerDinoActionModelCfg = field(
        default_factory=PureTransformerDinoActionModelCfg
    )
    optimizer: PureTransformerDinoActionOptimizerCfg = field(
        default_factory=PureTransformerDinoActionOptimizerCfg
    )
    logging: PureTransformerDinoActionLoggingCfg = field(
        default_factory=PureTransformerDinoActionLoggingCfg
    )


@register_algorithm(
    "pure_transformer_dino_action",
    cfg_cls=PureTransformerDinoActionCfg,
)
class PureTransformerDinoAction(PureTransformerAction):
    cfg: PureTransformerDinoActionCfg
    model: PureTransformerDinoActionModel

    def _build_model(self) -> None:
        image_size = self.cfg.image_size
        if len(image_size) != 2:
            raise ValueError(f"Expected image_size [H,W], got {image_size}")
        self.model = PureTransformerDinoActionModel(
            input_mode=self.cfg.model.input_mode,
            image_size=(int(image_size[0]), int(image_size[1])),
            action_dim=int(self.cfg.model.action_dim),
            dino_model_name=str(self.cfg.model.dino_model_name),
            freeze_dino=bool(self.cfg.model.freeze_dino),
            embed_dim=int(self.cfg.model.embed_dim),
            depth=int(self.cfg.model.depth),
            num_heads=int(self.cfg.model.num_heads),
            mlp_ratio=float(self.cfg.model.mlp_ratio),
            dropout=float(self.cfg.model.dropout),
            drop_path_rate=float(self.cfg.model.drop_path_rate),
            head_hidden_dim=int(self.cfg.model.head_hidden_dim),
            flow_adapter_hidden_dim=int(self.cfg.model.flow_adapter_hidden_dim),
            rgb_pair_include_difference_tokens=bool(
                self.cfg.model.rgb_pair_include_difference_tokens
            ),
        )
