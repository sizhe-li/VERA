from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch.nn.functional as F
from einops import einsum, rearrange
from jaxtyping import Float
from torch import Tensor, nn

from vera.idm.common.dino_tokenizer_transformer import (
    DinoTokenizerTransformerBackbone,
)

from .base import (
    BaseModelCfg,
    InputCommand,
    InputObservation,
    JacobianFieldInterface,
    JacobianFieldOutput,
)
from .registry import register_model


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.activation(self.conv1(x))
        x = self.conv2(x)
        return self.activation(x + residual)


class LightweightConvDecoder(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        hidden_dim: int,
        out_channels: int,
    ):
        super().__init__()
        self.token_norm = nn.LayerNorm(int(embed_dim))
        self.input_proj = nn.Conv2d(int(embed_dim), int(hidden_dim), kernel_size=1)
        self.pre_upsample = ResidualConvBlock(int(hidden_dim))
        self.post_upsample = ResidualConvBlock(int(hidden_dim))
        self.output_proj = nn.Conv2d(int(hidden_dim), int(out_channels), kernel_size=1)

    def forward(
        self,
        patch_tokens: Tensor,
        *,
        grid_size: tuple[int, int],
        output_size: tuple[int, int],
    ) -> Tensor:
        grid_height, grid_width = grid_size
        x = self.token_norm(patch_tokens)
        x = rearrange(
            x,
            "b (gh gw) c -> b c gh gw",
            gh=grid_height,
            gw=grid_width,
        )
        x = self.input_proj(x)
        x = self.pre_upsample(x)
        x = self.post_upsample(x)
        x = self.output_proj(x)
        return F.interpolate(
            x,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )


@dataclass
class PureTransformerJacobianDinoFieldCfg(BaseModelCfg):
    name: Literal["pure_transformer_jacobian_dino"]
    image_size: int
    dino_model_name: str = "dinov2_vits14"
    freeze_dino: bool = True
    embed_dim: int = 384
    depth: int = 6
    num_heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    drop_path_rate: float = 0.0
    head_hidden_dim: int = 256


@register_model(
    "pure_transformer_jacobian_dino",
    cfg_cls=PureTransformerJacobianDinoFieldCfg,
)
class PureTransformerJacobianDinoField(JacobianFieldInterface):
    cfg: PureTransformerJacobianDinoFieldCfg

    def __init__(self, model_cfg: PureTransformerJacobianDinoFieldCfg):
        super().__init__(cfg=model_cfg)
        self.command_dim = int(model_cfg.command_dim)
        self.spatial_dim = int(model_cfg.spatial_dim)
        self.backbone = DinoTokenizerTransformerBackbone(
            image_size=int(model_cfg.image_size),
            dino_model_name=str(model_cfg.dino_model_name),
            freeze_dino=bool(model_cfg.freeze_dino),
            embed_dim=int(model_cfg.embed_dim),
            depth=int(model_cfg.depth),
            num_heads=int(model_cfg.num_heads),
            mlp_ratio=float(model_cfg.mlp_ratio),
            dropout=float(model_cfg.dropout),
            drop_path_rate=float(model_cfg.drop_path_rate),
        )
        self.head = LightweightConvDecoder(
            embed_dim=int(model_cfg.embed_dim),
            hidden_dim=int(model_cfg.head_hidden_dim),
            out_channels=self.command_dim * self.spatial_dim,
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
        grid_height, grid_width = self.backbone.grid_size
        jacobian_dense = self.head(
            patch_tokens,
            grid_size=(grid_height, grid_width),
            output_size=(height, width),
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
