"""
Standalone DPT v2 Jacobian field: configurable backbone/neck and multi-scale decoder.
All DPT v2–specific logic lives here; no dependency on dpt_jacobian_field beyond base/registry.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange
from jaxtyping import Float
from torch import Tensor
from transformers import Dinov2Config, DPTConfig

from ..backbones.dpt import DptWrapper
from .base import (
    BaseModelCfg,
    InputCommand,
    InputObservation,
    JacobianFieldInterface,
    JacobianFieldOutput,
)
from .registry import register_model


# ----------------------------------------
# Multi-scale decoder (V2)
# ----------------------------------------


class DptJacobianDecoderV2(nn.Module):
    """
    Lightweight multi-scale decoder inspired by VGGT's `DPTHead`:
    fuse a small pyramid of DPT features with simple conv blocks.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        fusion_channels: int = 128,
        out_mean=0.0,
        out_std=0.1,
    ):
        super().__init__()

        self.fusion_channels = fusion_channels

        # Project each DPT neck feature to a common fusion dim.
        # We allocate a few slots and only use as many as are provided at runtime.
        self.proj_layers = nn.ModuleList(
            [nn.Conv2d(in_channels, fusion_channels, kernel_size=1) for _ in range(4)]
        )

        # Small refinement block reused at each fusion step.
        self.fuse_block = nn.Sequential(
            nn.Conv2d(fusion_channels, fusion_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_channels, fusion_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(fusion_channels, fusion_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_channels // 2, out_channels, kernel_size=1),
        )

        self._init_weights(out_mean=out_mean, out_std=out_std)

    def _init_weights(self, out_mean: float, out_std: float):
        # Initialize all convs in the decoder with a moderate std.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        # Explicitly control the last conv (index -1 in the Sequential) to set the
        # output distribution scale and mean.
        last_layer = self.out_conv[-1]
        if isinstance(last_layer, nn.Conv2d):
            torch.nn.init.normal_(last_layer.weight, mean=0.0, std=out_std)
            if last_layer.bias is not None:
                torch.nn.init.constant_(last_layer.bias, out_mean)

    def forward(
        self,
        features: Sequence[Tensor] | Tensor,
        target_hw: tuple[int, int],
    ) -> Tensor:
        """
        Args:
            features: list/tuple of feature maps from the DPT neck (low→high resolution),
                      or a single feature map for fallback.
            target_hw: (H, W) of the target image resolution.
        """
        # Fallback: single feature tensor, behave like the original head.
        if isinstance(features, Tensor):
            x = F.interpolate(
                features, size=target_hw, mode="bilinear", align_corners=False
            )
            return self.out_conv(x)

        if len(features) == 0:
            raise ValueError("DptJacobianDecoderV2 expects at least one feature map.")

        # Use up to the number of projection layers we have.
        feats: List[Tensor] = list(features)[-len(self.proj_layers) :]
        projected: List[Tensor] = []
        for idx, feat in enumerate(feats):
            proj = self.proj_layers[min(idx, len(self.proj_layers) - 1)]
            projected.append(proj(feat))

        # Start from the coarsest feature and progressively fuse higher-res ones.
        x = projected[-1]
        for f in reversed(projected[:-1]):
            x = F.interpolate(
                x, size=f.shape[-2:], mode="bilinear", align_corners=False
            )
            x = x + f
            x = self.fuse_block(x)

        # Final upsample to full image resolution and prediction conv.
        x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)
        x = self.out_conv(x)
        return x


# ----------------------------------------
# DPT config builder (presets / shortcuts)
# ----------------------------------------


def _build_dpt_config_from_cfg(
    image_size: int,
    backbone_preset: str = "small",
    out_indices: Optional[Sequence[int]] = None,
    neck_preset: Optional[str] = None,
    neck_hidden_sizes: Optional[Sequence[int]] = None,
) -> DPTConfig:
    """
    Build a small DPT config from simple presets, mirroring the DPT shortcuts.

    - backbone_preset: small | base | large | giant
    - neck_preset: S | M | L | XL  (or provide neck_hidden_sizes directly)
    - out_indices: which backbone stages to use (e.g. [1, 4] for @DPT/S).
    """
    backbone_preset = str(backbone_preset).lower()
    backbone_model_map = {
        "small": "facebook/dinov2-small",
        "base": "facebook/dinov2-base",
        "large": "facebook/dinov2-large",
        "giant": "facebook/dinov2-giant",
    }
    backbone_model = backbone_model_map.get(
        backbone_preset, backbone_model_map["small"]
    )

    if out_indices is None:
        out_indices = [1, 2, 3, 4]

    backbone_config = Dinov2Config.from_pretrained(
        backbone_model,
        out_features=["stage1", "stage2", "stage3", "stage4"],
        reshape_hidden_states=False,
        out_indices=list(out_indices),
    )

    # Resolve neck hidden sizes: explicit > preset > small default.
    if neck_hidden_sizes is None:
        if neck_preset is not None:
            preset_map = {
                "S": [128, 128],
                "M": [96, 96, 128, 128],
                "L": [128, 192, 192, 256],
                "XL": [256, 256, 384, 384],
            }
            neck_hidden_sizes = preset_map.get(str(neck_preset).upper())

        if neck_hidden_sizes is None:
            default = [96, 96, 128, 128]
            neck_hidden_sizes = default[: len(out_indices)]

    dpt_config = DPTConfig(
        backbone_config=backbone_config,
        image_size=image_size,
        neck_hidden_sizes=list(neck_hidden_sizes),
    )
    return dpt_config


# ----------------------------------------
# Config
# ----------------------------------------


@dataclass
class DptV2JacobianFieldCfg(BaseModelCfg):
    """
    Cleaner, configurable DPT Jacobian model that understands the @DPT/* shortcuts.
    """

    name: Literal["dpt_v2"]
    image_size: int  # required for dpt patching to behave correctly
    freeze_backbone: bool = True

    # Match fields used by shortcut/DPT/*.yaml
    backbone_preset: str = "small"  # small | base | large | giant
    out_indices: Optional[Sequence[int]] = None
    neck_preset: Optional[str] = None  # S | M | L | XL
    neck_hidden_sizes: Optional[Sequence[int]] = None


# ----------------------------------------
# Model
# ----------------------------------------


@register_model("dpt_v2", cfg_cls=DptV2JacobianFieldCfg)
class DptV2JacobianField(JacobianFieldInterface):
    """
    DPT-based Jacobian field with a small, VGGT-inspired multi-scale decoder and
    config driven entirely by the DPT shortcuts (backbone_preset, neck_preset, ...).
    """

    cfg: DptV2JacobianFieldCfg

    def __init__(self, model_cfg: DptV2JacobianFieldCfg):
        super().__init__(cfg=model_cfg)

        self.command_dim = model_cfg.command_dim
        self.spatial_dim = model_cfg.spatial_dim

        # --------------------
        # Build DPT encoder from presets
        # --------------------
        dpt_config = _build_dpt_config_from_cfg(
            image_size=model_cfg.image_size,
            backbone_preset=model_cfg.backbone_preset,
            out_indices=model_cfg.out_indices,
            neck_preset=model_cfg.neck_preset,
            neck_hidden_sizes=model_cfg.neck_hidden_sizes,
        )
        self.encoder = DptWrapper(dpt_config)

        # Freeze backbone if requested
        if model_cfg.freeze_backbone and self.encoder.backbone is not None:
            for param in self.encoder.backbone.parameters():
                param.requires_grad = False

        # --------------------
        # Lightweight multi-scale decoder
        # --------------------
        dpt_out_channels = dpt_config.fusion_hidden_size
        self.decoder = DptJacobianDecoderV2(
            in_channels=dpt_out_channels,
            out_channels=self.command_dim * self.spatial_dim,
            fusion_channels=64,
            out_mean=0.0,
            out_std=1e-5,
        )

    def compute_jacobian(
        self, input_obs: InputObservation
    ) -> Float[Tensor, "b c_dim s_dim h w"]:
        rgb = input_obs.rgb
        B, _, H, W = rgb.shape

        # DPT encoder: list of neck features (low→high resolution)
        features = self.encoder(rgb)

        jacobian_flat = self.decoder(features, target_hw=(H, W))

        jacobian = rearrange(
            jacobian_flat,
            "b (c_dim s_dim) h w -> b c_dim s_dim h w",
            c_dim=self.command_dim,
            s_dim=self.spatial_dim,
        )

        return jacobian

    def forward(self, input_obs: InputObservation, input_cmd: InputCommand):
        jacobian = self.compute_jacobian(input_obs)

        dx = einsum(
            jacobian,
            input_cmd.du,
            "b c_dim s_dim h w, b c_dim -> b s_dim h w",
        )

        return JacobianFieldOutput(
            jacobian=jacobian,
            optical_flow=dx,
        )
