from dataclasses import dataclass
from typing import Literal, Optional, Sequence

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


# initialize with a configurable value for all learnable (non-backbone) modules
def init_weights(m, std: float = 1e-4):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        if m.weight is not None:
            torch.nn.init.normal_(m.weight, mean=0.0, std=std)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


# ----------------------------------------
# Decoder Head for DPT → Jacobian
# ----------------------------------------


class DptJacobianDecoder(nn.Module):
    """
    Takes a low-res DPT feature map and decodes to full-res Jacobian field.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, out_channels, kernel_size=1),
        )

    def forward(self, x, target_hw):
        # x: (B, C, h, w)
        x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)
        x = self.net(x)

        return x


def _build_dpt_config(
    image_size: int,
    backbone_preset: str = "small",
    out_indices: Optional[Sequence[int]] = None,
    neck_preset: Optional[str] = None,
    neck_hidden_sizes: Optional[Sequence[int]] = None,
) -> DPTConfig:
    backbone_model_map = {
        "small": "facebook/dinov2-small",
        "base": "facebook/dinov2-base",
        "large": "facebook/dinov2-large",
        "giant": "facebook/dinov2-giant",
    }
    backbone_model = backbone_model_map.get(
        str(backbone_preset).lower(), backbone_model_map["small"]
    )
    if out_indices is None:
        out_indices = [1, 2, 3, 4]

    backbone_config = Dinov2Config.from_pretrained(
        backbone_model,
        out_features=["stage1", "stage2", "stage3", "stage4"],
        reshape_hidden_states=False,
        out_indices=list(out_indices),
    )

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
            default_sizes = [96, 96, 128, 128]
            neck_hidden_sizes = default_sizes[: len(out_indices)]

    return DPTConfig(
        backbone_config=backbone_config,
        image_size=image_size,
        neck_hidden_sizes=list(neck_hidden_sizes),
    )


# ----------------------------------------
# Config
# ----------------------------------------


@dataclass
class DptJacobianFieldCfg(BaseModelCfg):
    name: Literal["dpt"]
    image_size: int
    freeze_backbone: bool = True
    backbone_preset: str = "small"
    out_indices: Optional[Sequence[int]] = None
    neck_preset: Optional[str] = None
    neck_hidden_sizes: Optional[Sequence[int]] = None
    decoder_init_std: Optional[float] = None
    output_scale: float = 1.0


# ----------------------------------------
# Model
# ----------------------------------------


@register_model("dpt", cfg_cls=DptJacobianFieldCfg)
class DptJacobianField(JacobianFieldInterface):
    cfg: DptJacobianFieldCfg

    def __init__(self, model_cfg: DptJacobianFieldCfg):
        super().__init__(cfg=model_cfg)

        self.command_dim = model_cfg.command_dim
        self.spatial_dim = model_cfg.spatial_dim

        # --------------------
        # Build DPT Encoder
        # --------------------
        dpt_config = _build_dpt_config(
            image_size=model_cfg.image_size,
            backbone_preset=model_cfg.backbone_preset,
            out_indices=model_cfg.out_indices,
            neck_preset=model_cfg.neck_preset,
            neck_hidden_sizes=model_cfg.neck_hidden_sizes,
        )

        self.encoder = DptWrapper(dpt_config)

        # Freeze backbone if requested
        if self.cfg.freeze_backbone and self.encoder.backbone is not None:
            for param in self.encoder.backbone.parameters():
                param.requires_grad = False

        # --------------------
        # Decoder head
        # --------------------
        dpt_out_channels = dpt_config.fusion_hidden_size
        self.decoder = DptJacobianDecoder(
            in_channels=dpt_out_channels,
            out_channels=self.command_dim * self.spatial_dim,
        )
        if model_cfg.decoder_init_std is not None:
            self.decoder.apply(
                lambda module: init_weights(module, std=model_cfg.decoder_init_std)
            )
        self.output_scale = float(model_cfg.output_scale)

    # ----------------------------------------
    # Core Jacobian computation
    # ----------------------------------------

    def compute_jacobian(self, input_obs: InputObservation) -> Tensor:

        rgb = input_obs.rgb
        B, _, H, W = rgb.shape

        # DPT encoder: list of feature maps (pyramid)
        features = self.encoder(rgb)

        # Use the most semantic (last) feature map
        feat = features[-1]  # (B, C, h, w)
        # print(f"feat shape: {feat.shape}")

        # Decode to full resolution
        jacobian_flat = self.decoder(feat, target_hw=(H, W))
        # (B, c_dim * s_dim, H, W)

        jacobian = rearrange(
            jacobian_flat,
            "b (c_dim s_dim) h w -> b c_dim s_dim h w",
            c_dim=self.command_dim,
            s_dim=self.spatial_dim,
        )

        return jacobian * self.output_scale

    # ----------------------------------------
    # Forward: identical semantics to TransformerJacobianField
    # ----------------------------------------

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
