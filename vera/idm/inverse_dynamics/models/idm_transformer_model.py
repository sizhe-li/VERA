from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from vera.idm.jacobian.backbones.dpt import DptWrapper
from vera.idm.jacobian.models.dpt_vggt_fusion_jacobian_field import (
    VggtStyleJacobianDecoder,
    _build_dpt_config,
)


class IDMTransformerModel(nn.Module):
    """
    Flow-to-action model that reuses the Jacobian DINOv2+DPT+VGGT decoder path.
    Only the final readout changes: pooled decoder features -> MLP -> action vector.
    """

    def __init__(
        self,
        *,
        image_size: int,
        flow_channels: int,
        action_dim: int,
        backbone_preset: str = "base",
        neck_preset: str = "M",
        freeze_backbone: bool = True,
        fusion_channels: int = 128,
        prediction_hidden_channels: int = 64,
        use_uv_coords: bool = False,
        uv_fourier_frequencies: int = 10,
        decoder_init_std: float | None = None,
        decoder_feature_dim: int = 128,
        mlp_hidden_dim: int = 256,
    ):
        super().__init__()
        self.flow_channels = int(flow_channels)
        self.action_dim = int(action_dim)
        dpt_config = _build_dpt_config(
            image_size=int(image_size),
            backbone_preset=backbone_preset,
            neck_preset=neck_preset,
        )
        self.flow_adapter = nn.Conv2d(self.flow_channels, 3, kernel_size=1)
        self.encoder = DptWrapper(dpt_config)
        if freeze_backbone and self.encoder.backbone is not None:
            for param in self.encoder.backbone.parameters():
                param.requires_grad = False

        dpt_out_channels = dpt_config.fusion_hidden_size
        self.decoder = VggtStyleJacobianDecoder(
            in_channels=dpt_out_channels,
            out_channels=int(decoder_feature_dim),
            fusion_channels=int(fusion_channels),
            prediction_hidden_channels=int(prediction_hidden_channels),
            use_uv_coords=bool(use_uv_coords),
            uv_fourier_frequencies=int(uv_fourier_frequencies),
            final_init_std=decoder_init_std,
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.LayerNorm(int(decoder_feature_dim)),
            nn.Linear(int(decoder_feature_dim), int(mlp_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(mlp_hidden_dim), self.action_dim),
        )

    @staticmethod
    def _as_flow_map(flow: Tensor) -> Tensor:
        if flow.ndim == 4:
            return flow
        if flow.ndim == 5:
            if flow.shape[1] != 1:
                raise ValueError(
                    f"Expected a single flow map in [B,1,C,H,W], got {tuple(flow.shape)}"
                )
            return flow[:, 0]
        raise ValueError(f"Expected flow [B,C,H,W] or [B,1,C,H,W], got {tuple(flow.shape)}")

    def forward(self, flow: Tensor, view_ids: Optional[Tensor] = None) -> Tensor:
        flow_map = self._as_flow_map(flow)
        encoder_input = self.flow_adapter(flow_map)
        features = self.encoder(encoder_input)
        decoder_features = self.decoder(
            features,
            target_hw=flow_map.shape[-2:],
            view_ids=view_ids,
        )
        pooled = self.pool(decoder_features).flatten(1)
        return self.head(pooled)
