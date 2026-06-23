from __future__ import annotations

from collections import OrderedDict
from typing import Literal, Optional, Sequence

import torch
from torch import Tensor, nn

from vera.idm.common.dpt_config import build_dpt_config
from vera.idm.jacobian.backbones.dpt import DptWrapper


class DptVggtPooledActionModel(nn.Module):
    def __init__(
        self,
        *,
        input_mode: Literal["rgb_pair", "flow", "rgb_flow"],
        image_size: int | tuple[int, int],
        action_dim: int,
        backbone_preset: str,
        out_indices: Optional[Sequence[int]],
        neck_preset: Optional[str],
        neck_hidden_sizes: Optional[Sequence[int]],
        freeze_backbone: bool,
        fusion_channels: int,
        head_hidden_dim: int,
        flow_adapter_hidden_dim: int,
        use_max_pool: bool = True,
    ):
        super().__init__()
        self.input_mode = str(input_mode)
        self.action_dim = int(action_dim)
        self.fusion_channels = int(fusion_channels)
        self.use_max_pool = bool(use_max_pool)

        if isinstance(image_size, int):
            resolved_image_size = int(image_size)
        else:
            if len(image_size) != 2:
                raise ValueError(f"Expected image_size [H,W], got {image_size}")
            if int(image_size[0]) != int(image_size[1]):
                raise ValueError(
                    "DPT config builder currently expects square image_size, got "
                    f"{tuple(image_size)}"
                )
            resolved_image_size = int(image_size[0])

        self.dpt_config = build_dpt_config(
            image_size=resolved_image_size,
            backbone_preset=str(backbone_preset),
            out_indices=out_indices,
            neck_preset=neck_preset,
            neck_hidden_sizes=neck_hidden_sizes,
        )
        self.encoder = DptWrapper(self.dpt_config)

        if freeze_backbone and self.encoder.backbone is not None:
            for param in self.encoder.backbone.parameters():
                param.requires_grad = False

        dpt_channels = int(self.dpt_config.fusion_hidden_size)
        num_scales = len(self.dpt_config.neck_hidden_sizes)
        self.scale_projections = nn.ModuleList(
            [
                nn.Conv2d(dpt_channels, self.fusion_channels, kernel_size=1)
                for _ in range(num_scales)
            ]
        )

        if self.input_mode in {"flow", "rgb_flow"}:
            self.flow_adapter = nn.Sequential(
                nn.Conv2d(2, int(flow_adapter_hidden_dim), kernel_size=1),
                nn.GELU(),
                nn.Conv2d(int(flow_adapter_hidden_dim), 3, kernel_size=1),
            )
        else:
            self.flow_adapter = None

        stream_count = {"rgb_pair": 2, "flow": 1, "rgb_flow": 2}.get(self.input_mode)
        if stream_count is None:
            raise ValueError(f"Unsupported input_mode: {self.input_mode}")

        pool_multiplier = 2 if self.use_max_pool else 1
        pooled_dim = stream_count * num_scales * self.fusion_channels * pool_multiplier
        self.head = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, int(head_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(head_hidden_dim), self.action_dim),
        )

    def shared_parameter_counts(self):
        counts = OrderedDict()
        counts["backbone"] = int(
            sum(param.numel() for param in self.encoder.backbone.parameters())
            if self.encoder.backbone is not None
            else 0
        )
        counts["dpt_neck"] = int(
            sum(param.numel() for param in self.encoder.neck.parameters())
        )
        counts["scale_projections"] = int(
            sum(param.numel() for param in self.scale_projections.parameters())
        )
        counts["flow_adapter"] = (
            0
            if self.flow_adapter is None
            else int(sum(param.numel() for param in self.flow_adapter.parameters()))
        )
        return counts

    def head_parameter_count(self) -> int:
        return int(sum(param.numel() for param in self.head.parameters()))

    def _encode(self, x: Tensor) -> list[Tensor]:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(
                f"Expected image-like input [B,3,H,W], got {tuple(x.shape)}"
            )
        return [feature.contiguous() for feature in self.encoder(x)]

    def _pool_features(self, features: Sequence[Tensor]) -> Tensor:
        if len(features) != len(self.scale_projections):
            raise ValueError(
                "Expected feature/projection count to match, got "
                f"{len(features)} and {len(self.scale_projections)}"
            )

        pooled: list[Tensor] = []
        for feature, projection in zip(features, self.scale_projections):
            x = projection(feature.contiguous())
            pooled.append(x.mean(dim=(2, 3)))
            if self.use_max_pool:
                pooled.append(x.amax(dim=(2, 3)))
        return torch.cat(pooled, dim=-1)

    def _encode_and_pool_rgb(self, x: Tensor) -> Tensor:
        return self._pool_features(self._encode(x))

    def _encode_and_pool_flow(self, flow: Tensor) -> Tensor:
        if self.flow_adapter is None:
            raise RuntimeError("Flow adapter is not initialized for this input mode.")
        if flow.ndim != 4 or flow.shape[1] != 2:
            raise ValueError(f"Expected flow input [B,2,H,W], got {tuple(flow.shape)}")
        pseudo_rgb = self.flow_adapter(flow)
        return self._pool_features(self._encode(pseudo_rgb))

    def forward(self, primary: Tensor, secondary: Tensor | None = None) -> Tensor:
        if self.input_mode == "rgb_pair":
            if secondary is None:
                raise ValueError("rgb_pair mode expects a secondary RGB frame.")
            pooled = torch.cat(
                [
                    self._encode_and_pool_rgb(primary),
                    self._encode_and_pool_rgb(secondary),
                ],
                dim=-1,
            )
        elif self.input_mode == "flow":
            pooled = self._encode_and_pool_flow(primary)
        elif self.input_mode == "rgb_flow":
            if secondary is None:
                raise ValueError("rgb_flow mode expects a secondary flow tensor.")
            pooled = torch.cat(
                [
                    self._encode_and_pool_rgb(primary),
                    self._encode_and_pool_flow(secondary),
                ],
                dim=-1,
            )
        else:
            raise ValueError(f"Unsupported input_mode: {self.input_mode}")

        return self.head(pooled)
