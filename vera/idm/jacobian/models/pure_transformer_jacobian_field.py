from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch.nn.functional as F
from einops import einsum, rearrange
from jaxtyping import Float
from torch import Tensor, nn

from vera.idm.common.pure_transformer import PureTransformerBackbone

from .base import (
    BaseModelCfg,
    InputCommand,
    InputObservation,
    JacobianFieldInterface,
    JacobianFieldOutput,
)
from .registry import register_model


@dataclass
class PureTransformerJacobianFieldCfg(BaseModelCfg):
    name: Literal["pure_transformer_jacobian"]
    image_size: int
    patch_size: int = 14
    tokenizer_in_channels: int = 6
    embed_dim: int = 384
    depth: int = 6
    num_heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    drop_path_rate: float = 0.0
    head_hidden_dim: int = 256


@register_model("pure_transformer_jacobian", cfg_cls=PureTransformerJacobianFieldCfg)
class PureTransformerJacobianField(JacobianFieldInterface):
    cfg: PureTransformerJacobianFieldCfg

    def __init__(self, model_cfg: PureTransformerJacobianFieldCfg):
        super().__init__(cfg=model_cfg)
        self.command_dim = int(model_cfg.command_dim)
        self.spatial_dim = int(model_cfg.spatial_dim)
        self.backbone = PureTransformerBackbone(
            image_size=int(model_cfg.image_size),
            patch_size=int(model_cfg.patch_size),
            tokenizer_in_channels=int(model_cfg.tokenizer_in_channels),
            embed_dim=int(model_cfg.embed_dim),
            depth=int(model_cfg.depth),
            num_heads=int(model_cfg.num_heads),
            mlp_ratio=float(model_cfg.mlp_ratio),
            dropout=float(model_cfg.dropout),
            drop_path_rate=float(model_cfg.drop_path_rate),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(int(model_cfg.embed_dim)),
            nn.Linear(int(model_cfg.embed_dim), int(model_cfg.head_hidden_dim)),
            nn.GELU(),
            nn.Linear(
                int(model_cfg.head_hidden_dim),
                self.command_dim * self.spatial_dim,
            ),
        )

    def shared_parameter_counts(self):
        return self.backbone.shared_parameter_counts()

    def head_parameter_count(self) -> int:
        return int(sum(param.numel() for param in self.head.parameters()))

    def compute_jacobian(
        self, input_obs: InputObservation
    ) -> Float[Tensor, "batch c_dim s_dim height width"]:
        rgb = input_obs.rgb
        height, width = rgb.shape[-2:]
        hidden = self.backbone.forward_features(rgb)
        patch_tokens = hidden[:, 1:]
        patch_outputs = self.head(patch_tokens)
        grid_height, grid_width = self.backbone.grid_size
        jacobian_patch = rearrange(
            patch_outputs,
            "b (gh gw) (c_dim s_dim) -> b (c_dim s_dim) gh gw",
            gh=grid_height,
            gw=grid_width,
            c_dim=self.command_dim,
            s_dim=self.spatial_dim,
        )
        jacobian_dense = F.interpolate(
            jacobian_patch,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        return rearrange(
            jacobian_dense,
            "b (c_dim s_dim) h w -> b c_dim s_dim h w",
            c_dim=self.command_dim,
            s_dim=self.spatial_dim,
        )

    def forward(
        self,
        input_obs: InputObservation,
        input_cmd: InputCommand,
    ) -> JacobianFieldOutput:
        jacobian = self.compute_jacobian(input_obs)
        optical_flow = einsum(
            jacobian,
            input_cmd.du,
            "b c_dim s_dim h w, b c_dim -> b s_dim h w",
        )
        return JacobianFieldOutput(jacobian=jacobian, optical_flow=optical_flow)
