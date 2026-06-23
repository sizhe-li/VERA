from __future__ import annotations

from collections import OrderedDict
from typing import Sequence

import torch
from einops import rearrange
from torch import Tensor, nn


def normalize_image_size(image_size: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(image_size, int):
        return int(image_size), int(image_size)
    if len(image_size) != 2:
        raise ValueError(f"Expected image_size to have length 2, got {image_size}")
    return int(image_size[0]), int(image_size[1])


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x * random_tensor / keep_prob


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        drop_path: float,
    ):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop_path1 = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim=dim, hidden_dim=hidden_dim, dropout=dropout)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x: Tensor) -> Tensor:
        attn_input = self.norm1(x)
        attn_out, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + self.drop_path1(attn_out)
        x = x + self.drop_path2(self.ffn(self.norm2(x)))
        return x


class PureTransformerBackbone(nn.Module):
    def __init__(
        self,
        *,
        image_size: int | Sequence[int],
        patch_size: int,
        tokenizer_in_channels: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        drop_path_rate: float,
    ):
        super().__init__()
        self.image_size = normalize_image_size(image_size)
        self.patch_size = int(patch_size)
        self.tokenizer_in_channels = int(tokenizer_in_channels)
        self.embed_dim = int(embed_dim)

        image_height, image_width = self.image_size
        if image_height % self.patch_size != 0 or image_width % self.patch_size != 0:
            raise ValueError(
                "image_size must be divisible by patch_size, got "
                f"{self.image_size} and {self.patch_size}"
            )

        self.grid_size = (
            image_height // self.patch_size,
            image_width // self.patch_size,
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.patch_embed = nn.Conv2d(
            self.tokenizer_in_channels,
            self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.special_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, self.embed_dim)
        )
        self.pos_drop = nn.Dropout(float(dropout))

        drop_path_values = torch.linspace(0.0, float(drop_path_rate), int(depth)).tolist()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=self.embed_dim,
                    num_heads=int(num_heads),
                    mlp_ratio=float(mlp_ratio),
                    dropout=float(dropout),
                    drop_path=float(drop_path_values[idx]),
                )
                for idx in range(int(depth))
            ]
        )
        self.norm = nn.LayerNorm(self.embed_dim)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.special_token, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.patch_embed.weight)
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _pad_channels(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input [B,C,H,W], got {tuple(x.shape)}")
        if x.shape[1] > self.tokenizer_in_channels:
            raise ValueError(
                f"Expected at most {self.tokenizer_in_channels} channels, got {x.shape[1]}"
            )
        if x.shape[-2:] != self.image_size:
            raise ValueError(
                f"Expected image size {self.image_size}, got {tuple(x.shape[-2:])}"
            )
        if x.shape[1] == self.tokenizer_in_channels:
            return x
        padded = x.new_zeros(
            x.shape[0],
            self.tokenizer_in_channels,
            x.shape[2],
            x.shape[3],
        )
        padded[:, : x.shape[1]] = x
        return padded

    def forward_features(self, x: Tensor) -> Tensor:
        x = self._pad_channels(x)
        x = self.patch_embed(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        special_token = self.special_token.expand(x.shape[0], -1, -1)
        x = torch.cat([special_token, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def shared_parameter_counts(self) -> OrderedDict[str, int]:
        tokenizer_params = (
            sum(param.numel() for param in self.patch_embed.parameters())
            + int(self.special_token.numel())
            + int(self.pos_embed.numel())
        )
        trunk_params = (
            sum(param.numel() for block in self.blocks for param in block.parameters())
            + sum(param.numel() for param in self.norm.parameters())
        )
        return OrderedDict(
            tokenizer=int(tokenizer_params),
            trunk=int(trunk_params),
        )

    def shared_parameter_total(self) -> int:
        counts = self.shared_parameter_counts()
        return int(counts["tokenizer"] + counts["trunk"])
