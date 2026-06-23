from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange
from jaxtyping import Float
from torch import Tensor
from transformers import DPTConfig

from vera.idm.common.dpt_config import build_dpt_config
from ..backbones.dpt import DptWrapper
from .base import (
    BaseModelCfg,
    InputCommand,
    InputObservation,
    JacobianFieldInterface,
    JacobianFieldOutput,
)
from .registry import register_model


def _build_dpt_config(
    image_size: int,
    backbone_preset: str = "small",
    out_indices: Optional[Sequence[int]] = None,
    neck_preset: Optional[str] = None,
    neck_hidden_sizes: Optional[Sequence[int]] = None,
) -> DPTConfig:
    return build_dpt_config(
        image_size=image_size,
        backbone_preset=backbone_preset,
        out_indices=out_indices,
        neck_preset=neck_preset,
        neck_hidden_sizes=neck_hidden_sizes,
    )


def custom_interpolate(
    x: Tensor,
    size: tuple[int, int],
    *,
    mode: str = "bilinear",
    align_corners: bool = False,
) -> Tensor:
    """Chunk interpolation when a full call would exceed PyTorch's element limit."""
    int_max = 1_610_612_736
    input_elements = size[0] * size[1] * x.shape[0] * x.shape[1]
    if input_elements <= int_max:
        return F.interpolate(x, size=size, mode=mode, align_corners=align_corners)

    num_chunks = (input_elements // int_max) + 1
    chunks = torch.chunk(x, chunks=num_chunks, dim=0)
    interpolated = [
        F.interpolate(chunk, size=size, mode=mode, align_corners=align_corners)
        for chunk in chunks
    ]
    return torch.cat(interpolated, dim=0).contiguous()


class ResidualConvUnit(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = x.contiguous()
        residual = x
        x = self.activation(x)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        return x + residual


class FeatureFusionBlock(nn.Module):
    """
    VGGT/DPT-style refinement block:
    fuse a skip feature at the current scale, refine locally, then resize upward.
    """

    def __init__(self, channels: int, *, has_residual: bool):
        super().__init__()
        self.has_residual = has_residual
        if has_residual:
            self.skip_unit = ResidualConvUnit(channels)
        self.output_unit = ResidualConvUnit(channels)
        self.out_conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(
        self,
        x: Tensor,
        skip: Tensor | None = None,
        *,
        size: tuple[int, int],
    ) -> Tensor:
        if self.has_residual:
            if skip is None:
                raise ValueError("FeatureFusionBlock expected a skip tensor.")
            x = x.contiguous() + self.skip_unit(skip.contiguous())
        x = self.output_unit(x.contiguous())
        x = custom_interpolate(
            x, size=size, mode="bilinear", align_corners=False
        ).contiguous()
        return self.out_conv(x)


class VggtStyleJacobianDecoder(nn.Module):
    """
    A task-specific multiscale head that keeps the DPT pyramid alive all the way to
    prediction, instead of collapsing to the last feature map before decoding.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        fusion_channels: int = 128,
        prediction_hidden_channels: int = 64,
        max_scales: int = 4,
        use_uv_coords: bool = False,
        uv_fourier_frequencies: int = 10,
        final_init_std: float | None = None,
        use_view_embedding: bool = False,
        num_view_embeddings: int = 3,
        view_embedding_dim: int = 32,
    ):
        super().__init__()
        self.max_scales = max_scales
        self.use_uv_coords = use_uv_coords
        self.uv_fourier_frequencies = uv_fourier_frequencies
        self.use_view_embedding = bool(use_view_embedding)
        self.num_view_embeddings = int(num_view_embeddings)
        self.view_embedding = (
            nn.Embedding(self.num_view_embeddings, int(view_embedding_dim))
            if self.use_view_embedding
            else None
        )
        self.view_embedding_proj = (
            nn.Linear(int(view_embedding_dim), int(fusion_channels))
            if self.use_view_embedding
            else None
        )
        self.proj_layers = nn.ModuleList(
            [
                nn.Conv2d(in_channels, fusion_channels, kernel_size=1)
                for _ in range(max_scales)
            ]
        )
        self.fusion_blocks = nn.ModuleList(
            [
                FeatureFusionBlock(
                    fusion_channels,
                    has_residual=(idx != 0),
                )
                for idx in range(max_scales)
            ]
        )
        self.output_conv1 = nn.Conv2d(
            fusion_channels, fusion_channels, kernel_size=3, padding=1
        )
        coord_channels = self._coord_feature_dim()
        self.output_conv2 = nn.Sequential(
            nn.Conv2d(
                fusion_channels + coord_channels,
                prediction_hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(prediction_hidden_channels, out_channels, kernel_size=1),
        )

        if final_init_std is not None:
            last_layer = self.output_conv2[-1]
            if isinstance(last_layer, nn.Conv2d):
                nn.init.normal_(last_layer.weight, mean=0.0, std=final_init_std)
                if last_layer.bias is not None:
                    nn.init.zeros_(last_layer.bias)

    def _view_embedding_bias(
        self,
        *,
        batch_size: int,
        view_ids: Tensor | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor | None:
        if not self.use_view_embedding:
            return None
        if self.view_embedding is None or self.view_embedding_proj is None:
            raise RuntimeError("View embedding modules are not initialized.")

        if view_ids is None:
            view_ids = torch.zeros(batch_size, device=device, dtype=torch.long)
        else:
            if view_ids.ndim != 1 or int(view_ids.shape[0]) != int(batch_size):
                raise ValueError(
                    "Expected view_ids shape [batch], got "
                    f"{tuple(view_ids.shape)} for batch={batch_size}"
                )
            view_ids = view_ids.to(device=device, dtype=torch.long)

        if torch.any(view_ids < 0) or torch.any(view_ids >= self.num_view_embeddings):
            max_id = int(view_ids.max().item()) if view_ids.numel() > 0 else -1
            raise ValueError(
                "view_ids out of range for configured embedding table: "
                f"max_id={max_id}, num_view_embeddings={self.num_view_embeddings}"
            )

        emb = self.view_embedding(view_ids)
        emb = self.view_embedding_proj(emb).to(dtype=dtype)
        return emb[:, :, None, None]

    def _coord_feature_dim(self) -> int:
        if not self.use_uv_coords:
            return 0
        # Legacy-style positional encoding: include raw coordinates plus sin/cos pairs.
        return 2 + 4 * max(int(self.uv_fourier_frequencies), 0)

    def _build_coord_features(
        self,
        batch_size: int,
        target_hw: tuple[int, int],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor | None:
        if not self.use_uv_coords:
            return None

        height, width = target_hw
        ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        coord_feats = [grid_x, grid_y]
        for freq_idx in range(self.uv_fourier_frequencies):
            # Match the old positional encoding style: sin/cos(pi * 2^k * coord).
            freq = torch.pi * float(2**freq_idx)
            coord_feats.extend(
                [
                    torch.sin(freq * grid_x),
                    torch.cos(freq * grid_x),
                    torch.sin(freq * grid_y),
                    torch.cos(freq * grid_y),
                ]
            )
        coords = torch.stack(coord_feats, dim=0).unsqueeze(0)
        return coords.expand(batch_size, -1, -1, -1)

    def forward(
        self,
        features: Sequence[Tensor],
        *,
        target_hw: tuple[int, int],
        view_ids: Tensor | None = None,
    ) -> Tensor:
        if len(features) == 0:
            raise ValueError(
                "VggtStyleJacobianDecoder expects at least one feature map."
            )

        feats = list(features)[-self.max_scales :]
        projected = [
            proj(feat.contiguous())
            for proj, feat in zip(self.proj_layers[: len(feats)], feats)
        ]
        view_bias = self._view_embedding_bias(
            batch_size=projected[0].shape[0],
            view_ids=view_ids,
            device=projected[0].device,
            dtype=projected[0].dtype,
        )
        if view_bias is not None:
            projected = [(feat + view_bias).contiguous() for feat in projected]

        if len(projected) == 1:
            x = projected[0]
        else:
            x = self.fusion_blocks[0](
                projected[0],
                size=projected[1].shape[-2:],
            )
            for idx in range(1, len(projected)):
                next_size = (
                    projected[idx + 1].shape[-2:]
                    if idx + 1 < len(projected)
                    else projected[idx].shape[-2:]
                )
                x = self.fusion_blocks[idx](
                    x,
                    projected[idx],
                    size=next_size,
                )

        x = self.output_conv1(x.contiguous())
        x = custom_interpolate(
            x, size=target_hw, mode="bilinear", align_corners=False
        ).contiguous()
        coord_features = self._build_coord_features(
            x.shape[0],
            target_hw,
            device=x.device,
            dtype=x.dtype,
        )
        if coord_features is not None:
            x = torch.cat([x, coord_features], dim=1).contiguous()
        return self.output_conv2(x)


@dataclass
class DptVggtFusionJacobianFieldCfg(BaseModelCfg):
    name: Literal["dpt_vggt_fusion"]
    image_size: int
    freeze_backbone: bool = True
    backbone_preset: str = "small"
    out_indices: Optional[Sequence[int]] = None
    neck_preset: Optional[str] = None
    neck_hidden_sizes: Optional[Sequence[int]] = None
    fusion_channels: int = 128
    prediction_hidden_channels: int = 64
    use_uv_coords: bool = False
    uv_fourier_frequencies: int = 10
    decoder_init_std: float | None = None
    output_scale: float = 1.0
    use_view_embedding: bool = False
    num_view_embeddings: int = 3
    view_embedding_dim: int = 32


@register_model("dpt_vggt_fusion", cfg_cls=DptVggtFusionJacobianFieldCfg)
class DptVggtFusionJacobianField(JacobianFieldInterface):
    cfg: DptVggtFusionJacobianFieldCfg

    def __init__(self, model_cfg: DptVggtFusionJacobianFieldCfg):
        super().__init__(cfg=model_cfg)

        self.command_dim = model_cfg.command_dim
        self.spatial_dim = model_cfg.spatial_dim

        dpt_config = _build_dpt_config(
            image_size=model_cfg.image_size,
            backbone_preset=model_cfg.backbone_preset,
            out_indices=model_cfg.out_indices,
            neck_preset=model_cfg.neck_preset,
            neck_hidden_sizes=model_cfg.neck_hidden_sizes,
        )
        self.encoder = DptWrapper(dpt_config)

        if model_cfg.freeze_backbone and self.encoder.backbone is not None:
            for param in self.encoder.backbone.parameters():
                param.requires_grad = False

        dpt_out_channels = dpt_config.fusion_hidden_size
        self.decoder = VggtStyleJacobianDecoder(
            in_channels=dpt_out_channels,
            out_channels=self.command_dim * self.spatial_dim,
            fusion_channels=model_cfg.fusion_channels,
            prediction_hidden_channels=model_cfg.prediction_hidden_channels,
            use_uv_coords=model_cfg.use_uv_coords,
            uv_fourier_frequencies=model_cfg.uv_fourier_frequencies,
            final_init_std=model_cfg.decoder_init_std,
            use_view_embedding=model_cfg.use_view_embedding,
            num_view_embeddings=model_cfg.num_view_embeddings,
            view_embedding_dim=model_cfg.view_embedding_dim,
        )
        self.output_scale = float(model_cfg.output_scale)

    def compute_jacobian(
        self, input_obs: InputObservation
    ) -> Float[Tensor, "batch c_dim s_dim height width"]:
        rgb = input_obs.rgb
        _, _, height, width = rgb.shape
        features = [feat.contiguous() for feat in self.encoder(rgb)]
        jacobian_flat = self.decoder(
            features,
            target_hw=(height, width),
            view_ids=input_obs.view_ids,
        )
        jacobian = rearrange(
            jacobian_flat,
            "b (c_dim s_dim) h w -> b c_dim s_dim h w",
            c_dim=self.command_dim,
            s_dim=self.spatial_dim,
        )
        return jacobian * self.output_scale

    def forward(
        self,
        input_obs: InputObservation,
        input_cmd: InputCommand,
    ) -> JacobianFieldOutput:
        jacobian = self.compute_jacobian(input_obs)
        flow = einsum(
            jacobian,
            input_cmd.du,
            "b c_dim s_dim h w, b c_dim -> b s_dim h w",
        )

        return JacobianFieldOutput(jacobian=jacobian, optical_flow=flow)
