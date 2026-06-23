from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

from vera.idm.common.pure_transformer import PureTransformerBackbone


class PureTransformerActionModel(nn.Module):
    def __init__(
        self,
        *,
        input_mode: Literal["rgb_pair", "flow", "rgb_flow"],
        image_size: int | tuple[int, int],
        action_dim: int,
        patch_size: int,
        tokenizer_in_channels: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        drop_path_rate: float,
        head_hidden_dim: int,
    ):
        super().__init__()
        self.input_mode = str(input_mode)
        self.action_dim = int(action_dim)
        self.backbone = PureTransformerBackbone(
            image_size=image_size,
            patch_size=int(patch_size),
            tokenizer_in_channels=int(tokenizer_in_channels),
            embed_dim=int(embed_dim),
            depth=int(depth),
            num_heads=int(num_heads),
            mlp_ratio=float(mlp_ratio),
            dropout=float(dropout),
            drop_path_rate=float(drop_path_rate),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(int(embed_dim)),
            nn.Linear(int(embed_dim), int(head_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(head_hidden_dim), self.action_dim),
        )

    def shared_parameter_counts(self):
        return self.backbone.shared_parameter_counts()

    def head_parameter_count(self) -> int:
        return int(sum(param.numel() for param in self.head.parameters()))

    def _pack_input(self, primary: Tensor, secondary: Tensor | None) -> Tensor:
        if self.input_mode == "rgb_pair":
            if primary.ndim != 4 or secondary is None or secondary.ndim != 4:
                raise ValueError(
                    "rgb_pair mode expects two image tensors with shape [B,3,H,W]."
                )
            return torch.cat([primary, secondary], dim=1)
        if self.input_mode == "rgb_flow":
            if primary.ndim != 4 or primary.shape[1] != 3:
                raise ValueError(
                    "rgb_flow mode expects the primary tensor to have shape [B,3,H,W]."
                )
            if secondary is None or secondary.ndim != 4 or secondary.shape[1] != 2:
                raise ValueError(
                    "rgb_flow mode expects the secondary tensor to have shape [B,2,H,W]."
                )
            return torch.cat([primary, secondary], dim=1)
        if self.input_mode == "flow":
            if primary.ndim != 4:
                raise ValueError("flow mode expects a tensor with shape [B,C,H,W].")
            return primary
        raise ValueError(f"Unsupported input_mode: {self.input_mode}")

    def forward(self, primary: Tensor, secondary: Tensor | None = None) -> Tensor:
        packed = self._pack_input(primary, secondary)
        hidden = self.backbone.forward_features(packed)
        action_token = hidden[:, 0]
        return self.head(action_token)
