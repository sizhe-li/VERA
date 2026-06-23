"""
Latent-space encoder-decoder Jacobian field with DPT-like architecture.
Encoder([rgb; optional flow]) -> z, Decoder(z) -> [image; optional flow], J(z) -> (dim_z, dim_u).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

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


def init_weights(m, std: float = 1e-2):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        if m.weight is not None:
            nn.init.normal_(m.weight, mean=0.0, std=std)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def init_encoder_neck_last_layer_small(encoder: DptWrapper, std: float = 1e-2) -> None:
    """Re-initialize the last Conv2d/Linear in the encoder neck with small weights.
    Used so the latent encoder starts with small-magnitude outputs; does not affect
    DptJacobianField which uses the same backbone/dpt.py without this init."""
    last_layer = None
    for _name, m in encoder.neck.named_modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            last_layer = m
    if last_layer is not None:
        if last_layer.weight is not None:
            nn.init.normal_(last_layer.weight, mean=0.0, std=std)
        if getattr(last_layer, "bias", None) is not None:
            nn.init.zeros_(last_layer.bias)


# ---------------------------------------------------------------------------
# Decoder head: latent (B, C, h, w) -> (B, out_channels, H, W)
# ---------------------------------------------------------------------------


class LatentDecoderHead(nn.Module):
    """DPT-style conv head: bilinear upsample + convs to image resolution."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, out_channels, kernel_size=1),
        )

    def forward(self, x: Tensor, target_hw: Tuple[int, int]) -> Tensor:
        x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)
        return self.net(x)


# ---------------------------------------------------------------------------
# Jacobian head: z (B, C, h, w) -> J (B, latent_image_token_dim, latent_action_dim)
# Uses a shared per-token linear to avoid O(dim_z^2) parameters.
# ---------------------------------------------------------------------------


class JacobianHead(nn.Module):
    """Per-token linear: at each spatial position (C channels), predict a (C, dim_u) block of J.
    One shared Linear(C, C * latent_action_dim) -> ~33K params instead of billions."""

    def __init__(
        self,
        channel_dim: int,
        latent_action_dim: int,
        init_std: float = 1e-6,
    ):
        super().__init__()
        self.channel_dim = channel_dim
        self.latent_action_dim = latent_action_dim
        self.linear = nn.Linear(channel_dim, channel_dim * latent_action_dim)
        nn.init.normal_(self.linear.weight, mean=0.0, std=init_std)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(
        self, z_flat: Float[Tensor, "b n_tokens d"]
    ) -> Float[Tensor, "b dim_z dim_u"]:
        B, N, C = z_flat.shape
        # (B, N, C) -> (B, N, C * dim_u)
        out = self.linear(z_flat)
        # (B, N, C, dim_u) -> (B, N*C, dim_u)
        out = out.reshape(B, N, C, self.latent_action_dim)
        return rearrange(out, "b n c u -> b (n c) u")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class LatentJacobianFieldCfg(BaseModelCfg):
    name: Literal["latent_dpt"]
    image_size: int = 252
    freeze_backbone: bool = True
    use_optical_flow: bool = False
    latent_image_token_dim: int = 0
    latent_action_dim: int = 0
    command_dim: int = 0
    spatial_dim: int = 1
    # VGGT-style small init for task heads / latent output
    jacobian_head_init_std: float = 1e-6
    encoder_neck_init_std: float = 1e-2
    latent_norm: Literal["none", "layernorm", "groupnorm"] = "layernorm"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@register_model("latent_dpt", cfg_cls=LatentJacobianFieldCfg)
class LatentJacobianField(JacobianFieldInterface):
    cfg: LatentJacobianFieldCfg

    def __init__(self, model_cfg: LatentJacobianFieldCfg):
        super().__init__(cfg=model_cfg)

        self.use_optical_flow = model_cfg.use_optical_flow
        in_channels = 5 if self.use_optical_flow else 3
        if in_channels != 3:
            self.input_proj = nn.Conv2d(in_channels, 3, kernel_size=1)
        else:
            self.input_proj = nn.Identity()

        backbone_config = Dinov2Config.from_pretrained(
            "facebook/dinov2-small",
            out_features=["stage1", "stage2", "stage3", "stage4"],
            reshape_hidden_states=False,
            out_indices=[1, 2, 3, 4],
        )
        dpt_config = DPTConfig(
            backbone_config=backbone_config,
            image_size=model_cfg.image_size,
            neck_hidden_sizes=[96, 96, 128, 128],
        )
        self.encoder = DptWrapper(dpt_config)
        init_encoder_neck_last_layer_small(
            self.encoder, std=getattr(model_cfg, "encoder_neck_init_std", 1e-2)
        )

        if model_cfg.freeze_backbone and self.encoder.backbone is not None:
            for p in self.encoder.backbone.parameters():
                p.requires_grad = False

        dpt_out_channels = dpt_config.fusion_hidden_size
        self._dpt_out_channels = dpt_out_channels
        self._h = self._w = None

        out_channels = 5 if self.use_optical_flow else 3
        self.decoder = LatentDecoderHead(
            in_channels=dpt_out_channels,
            out_channels=out_channels,
        )
        self.decoder.apply(lambda m: init_weights(m, std=1e-2))

        if model_cfg.latent_action_dim <= 0:
            raise ValueError("latent_action_dim must be set in config")
        self.latent_image_token_dim = model_cfg.latent_image_token_dim
        self.latent_action_dim = model_cfg.latent_action_dim
        self.jacobian_head = JacobianHead(
            channel_dim=dpt_out_channels,
            latent_action_dim=model_cfg.latent_action_dim,
            init_std=getattr(model_cfg, "jacobian_head_init_std", 1e-6),
        )

        latent_norm_type = getattr(model_cfg, "latent_norm", "layernorm")
        if latent_norm_type == "layernorm":
            self.latent_norm = nn.LayerNorm(dpt_out_channels)
        elif latent_norm_type == "groupnorm":
            self.latent_norm = nn.GroupNorm(num_groups=1, num_channels=dpt_out_channels)
        else:
            self.latent_norm = nn.Identity()

        self.command_dim = model_cfg.latent_action_dim or model_cfg.command_dim
        self.spatial_dim = model_cfg.spatial_dim

    def _get_encoder_input(self, rgb: Tensor, flow: Optional[Tensor] = None) -> Tensor:
        if self.use_optical_flow and flow is not None:
            x = torch.cat([rgb, flow], dim=1)
        else:
            x = rgb
        return self.input_proj(x)

    def encode(
        self,
        rgb: Float[Tensor, "b 3 H W"],
        flow: Optional[Float[Tensor, "b 2 H W"]] = None,
    ) -> Float[Tensor, "b C h w"]:
        x = self._get_encoder_input(rgb, flow)
        features = self.encoder(x)
        z = features[-1]
        if isinstance(self.latent_norm, nn.LayerNorm):
            z = z.permute(0, 2, 3, 1)
            z = self.latent_norm(z)
            z = z.permute(0, 3, 1, 2)
        else:
            z = self.latent_norm(z)
        B, C, h, w = z.shape
        self._h, self._w = h, w
        return z

    def flatten_z(self, z: Float[Tensor, "b C h w"]) -> Float[Tensor, "b latent_dim"]:
        return rearrange(z, "b c h w -> b (c h w)")

    def decode(
        self,
        z: Float[Tensor, "b C h w"],
        target_hw: Tuple[int, int],
    ) -> Float[Tensor, "b out_c H W"]:
        return self.decoder(z, target_hw)

    def compute_jacobian_from_z(
        self, z: Float[Tensor, "b C h w"]
    ) -> Float[Tensor, "b dim_z dim_u"]:
        B, C, h, w = z.shape
        z_flat = rearrange(z, "b c h w -> b (h w) c")
        return self.jacobian_head(z_flat)

    def compute_jacobian(
        self, input_obs: InputObservation
    ) -> Float[Tensor, "b c_dim s_dim h w"]:
        z = self.encode(input_obs.rgb, getattr(input_obs, "flow", None))
        J = self.compute_jacobian_from_z(z)
        return J.unsqueeze(-1).unsqueeze(-1)

    def solve_latent_action(
        self,
        z_t: Float[Tensor, "b C h w"],
        delta_z: Float[Tensor, "b latent_dim"],
        J_t: Float[Tensor, "b latent_dim dim_u"],
    ) -> Float[Tensor, "b dim_u"]:
        # Normal equations: du = (J^T J)^{-1} J^T delta_z. O(dim_z * dim_u) instead of
        # lstsq's O(dim_z^2), so we avoid OOM when latent_dim is huge (e.g. 5M).
        # Always use float32: torch.linalg.solve/lstsq do not support float16.
        dtype_orig = J_t.dtype
        J_f = J_t.to(torch.float32)
        delta_f = delta_z.to(torch.float32).unsqueeze(-1)  # (B, dim_z, 1)
        # JTJ (B, dim_u, dim_u), JTd (B, dim_u, 1)
        JTJ = torch.bmm(J_f.transpose(-2, -1), J_f)
        JTd = torch.bmm(J_f.transpose(-2, -1), delta_f)
        # Small regularizer for numerical stability
        JTJ = JTJ + 1e-6 * torch.eye(
            J_t.shape[-1], device=JTJ.device, dtype=torch.float32
        ).unsqueeze(0)
        JTJ = JTJ.to(torch.float32)
        JTd = JTd.to(torch.float32)
        if JTJ.is_cuda:
            with torch.cuda.amp.autocast(enabled=False):
                du_pred = torch.linalg.solve(JTJ, JTd).squeeze(-1)
        else:
            du_pred = torch.linalg.solve(JTJ, JTd).squeeze(-1)
        return du_pred.to(dtype_orig)

    def predict_next_state(
        self,
        z_t: Float[Tensor, "b C h w"],
        J_t: Float[Tensor, "b latent_dim dim_u"],
        du: Float[Tensor, "b dim_u"],
    ) -> Tuple[Float[Tensor, "b C h w"], Float[Tensor, "b latent_dim"]]:
        z_flat = self.flatten_z(z_t)
        z_dot = einsum(J_t, du, "b d u,b u->b d")
        z_next_flat = z_flat + z_dot
        z_next = z_next_flat.reshape(z_t.shape)
        return z_next, z_dot

    def forward(
        self, input_obs: InputObservation, input_cmd: InputCommand
    ) -> JacobianFieldOutput:
        z = self.encode(input_obs.rgb, getattr(input_obs, "flow", None))
        J = self.compute_jacobian_from_z(z)
        B, _, H, W = input_obs.rgb.shape
        z_dot = einsum(J, input_cmd.du, "b d u,b u->b d")
        z_flat = self.flatten_z(z)
        z_next_flat = z_flat + z_dot
        z_next = z_next_flat.reshape(z.shape)
        decoded_flow = self.decode(z_next, (H, W))
        if decoded_flow.shape[1] >= 5:
            optical_flow = decoded_flow[:, 3:5]
        else:
            optical_flow = torch.zeros(B, 2, H, W, device=z.device, dtype=z.dtype)
        return JacobianFieldOutput(
            jacobian=J.unsqueeze(-1).unsqueeze(-1),
            optical_flow=optical_flow,
        )
