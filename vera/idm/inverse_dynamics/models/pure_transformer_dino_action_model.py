from __future__ import annotations

from collections import OrderedDict
from typing import Literal

import torch
from torch import Tensor, nn

from vera.idm.common.dino_tokenizer_transformer import (
    DinoTokenizerTransformerBackbone,
)


class PureTransformerDinoActionModel(nn.Module):
    def __init__(
        self,
        *,
        input_mode: Literal["rgb_pair", "flow", "rgb_flow"],
        image_size: int | tuple[int, int],
        action_dim: int,
        dino_model_name: str,
        freeze_dino: bool,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        drop_path_rate: float,
        head_hidden_dim: int,
        flow_adapter_hidden_dim: int,
        rgb_pair_include_difference_tokens: bool = False,
    ):
        super().__init__()
        self.input_mode = str(input_mode)
        self.action_dim = int(action_dim)
        self.rgb_pair_include_difference_tokens = bool(rgb_pair_include_difference_tokens)
        max_visual_tokens = None
        if self.input_mode in {"rgb_pair", "rgb_flow"}:
            if isinstance(image_size, int):
                image_height, image_width = int(image_size), int(image_size)
            else:
                image_height, image_width = int(image_size[0]), int(image_size[1])
            patch_size = DinoTokenizerTransformerBackbone.infer_patch_size_from_name(
                dino_model_name
            )
            num_patches = (image_height // patch_size) * (image_width // patch_size)
            if self.input_mode == "rgb_pair":
                sequence_multiplier = 3 if self.rgb_pair_include_difference_tokens else 2
            else:
                sequence_multiplier = 2
            max_visual_tokens = sequence_multiplier * num_patches

        self.backbone = DinoTokenizerTransformerBackbone(
            image_size=image_size,
            dino_model_name=str(dino_model_name),
            freeze_dino=bool(freeze_dino),
            embed_dim=int(embed_dim),
            depth=int(depth),
            num_heads=int(num_heads),
            mlp_ratio=float(mlp_ratio),
            dropout=float(dropout),
            drop_path_rate=float(drop_path_rate),
            max_visual_tokens=max_visual_tokens,
        )

        if self.input_mode in {"flow", "rgb_flow"}:
            self.flow_adapter = nn.Sequential(
                nn.Conv2d(2, int(flow_adapter_hidden_dim), kernel_size=1),
                nn.GELU(),
                nn.Conv2d(int(flow_adapter_hidden_dim), 3, kernel_size=1),
            )
        else:
            self.flow_adapter = None

        self.head = nn.Sequential(
            nn.LayerNorm(int(embed_dim)),
            nn.Linear(int(embed_dim), int(head_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(head_hidden_dim), self.action_dim),
        )

    def shared_parameter_counts(self):
        counts = OrderedDict(self.backbone.shared_parameter_counts())
        counts["flow_adapter"] = (
            0
            if self.flow_adapter is None
            else int(sum(param.numel() for param in self.flow_adapter.parameters()))
        )
        return counts

    def head_parameter_count(self) -> int:
        return int(sum(param.numel() for param in self.head.parameters()))

    def _pack_tokens(self, primary: Tensor, secondary: Tensor | None) -> Tensor:
        if self.input_mode == "rgb_pair":
            if primary.ndim != 4 or secondary is None or secondary.ndim != 4:
                raise ValueError(
                    "rgb_pair mode expects two image tensors with shape [B,3,H,W]."
                )
            first_tokens = self.backbone.tokenize_image(primary)
            second_tokens = self.backbone.tokenize_image(secondary)
            token_groups = [first_tokens, second_tokens]
            if self.rgb_pair_include_difference_tokens:
                token_groups.append(second_tokens - first_tokens)
            return torch.cat(token_groups, dim=1)
        if self.input_mode == "rgb_flow":
            if primary.ndim != 4 or primary.shape[1] != 3:
                raise ValueError(
                    "rgb_flow mode expects the primary tensor to have shape [B,3,H,W]."
                )
            if secondary is None or secondary.ndim != 4 or secondary.shape[1] != 2:
                raise ValueError(
                    "rgb_flow mode expects the secondary tensor to have shape [B,2,H,W]."
                )
            rgb_tokens = self.backbone.tokenize_image(primary)
            pseudo_rgb = self.flow_adapter(secondary)
            flow_tokens = self.backbone.tokenize_image(pseudo_rgb)
            return torch.cat([rgb_tokens, flow_tokens], dim=1)
        if self.input_mode == "flow":
            if primary.ndim != 4 or primary.shape[1] != 2:
                raise ValueError("flow mode expects a tensor with shape [B,2,H,W].")
            pseudo_rgb = self.flow_adapter(primary)
            return self.backbone.tokenize_image(pseudo_rgb)
        raise ValueError(f"Unsupported input_mode: {self.input_mode}")

    def visual_token_count(self) -> int:
        if self.input_mode == "rgb_pair":
            sequence_multiplier = 3 if self.rgb_pair_include_difference_tokens else 2
            return sequence_multiplier * self.backbone.num_patches
        if self.input_mode == "rgb_flow":
            return 2 * self.backbone.num_patches
        return self.backbone.num_patches

    def forward(self, primary: Tensor, secondary: Tensor | None = None) -> Tensor:
        visual_tokens = self._pack_tokens(primary, secondary)
        hidden = self.backbone.forward_token_sequence(visual_tokens)
        action_token = hidden[:, 0]
        return self.head(action_token)
