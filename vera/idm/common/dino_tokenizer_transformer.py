from __future__ import annotations

from collections import OrderedDict
from typing import Sequence

import torch
from torch import Tensor, nn

from .pure_transformer import TransformerBlock, normalize_image_size


class DinoTokenizerTransformerBackbone(nn.Module):
    def __init__(
        self,
        *,
        image_size: int | Sequence[int],
        dino_model_name: str,
        freeze_dino: bool,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        drop_path_rate: float,
        max_visual_tokens: int | None = None,
    ):
        super().__init__()
        self.image_size = normalize_image_size(image_size)
        self.embed_dim = int(embed_dim)
        self.dino_model_name = str(dino_model_name)
        self.freeze_dino = bool(freeze_dino)

        self.dino = self._load_dino_model(self.dino_model_name)
        self.patch_size = int(getattr(self.dino, "patch_size"))

        image_height, image_width = self.image_size
        if image_height % self.patch_size != 0 or image_width % self.patch_size != 0:
            raise ValueError(
                "image_size must be divisible by DINO patch_size, got "
                f"{self.image_size} and {self.patch_size}"
            )

        if self.freeze_dino:
            for param in self.dino.parameters():
                param.requires_grad = False
            self.dino.eval()

        self.grid_size = (
            image_height // self.patch_size,
            image_width // self.patch_size,
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.max_visual_tokens = int(
            self.num_patches if max_visual_tokens is None else max_visual_tokens
        )
        if self.max_visual_tokens < self.num_patches:
            raise ValueError(
                "max_visual_tokens must be at least the single-image patch count, got "
                f"{self.max_visual_tokens} < {self.num_patches}"
            )

        dino_embed_dim = self._infer_dino_embed_dim()
        if dino_embed_dim == self.embed_dim:
            self.token_projection = nn.Identity()
        else:
            self.token_projection = nn.Linear(dino_embed_dim, self.embed_dim)

        self.special_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.max_visual_tokens + 1, self.embed_dim)
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

    @staticmethod
    def _load_dino_model(model_name: str) -> nn.Module:
        return torch.hub.load("facebookresearch/dinov2", model_name)

    @staticmethod
    def infer_patch_size_from_name(model_name: str) -> int:
        name = str(model_name)
        trailing_digits = []
        for char in reversed(name):
            if char.isdigit():
                trailing_digits.append(char)
            elif trailing_digits:
                break
        if not trailing_digits:
            raise ValueError(f"Unable to infer DINO patch size from model name {model_name}")
        return int("".join(reversed(trailing_digits)))

    def _infer_dino_embed_dim(self) -> int:
        for attr_name in ("embed_dim", "num_features"):
            value = getattr(self.dino, attr_name, None)
            if value is not None:
                return int(value)
        raise AttributeError(
            f"Unable to infer DINO embedding dimension for model {type(self.dino).__name__}"
        )

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.special_token, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
        if isinstance(self.token_projection, nn.Linear):
            nn.init.xavier_uniform_(self.token_projection.weight)
            if self.token_projection.bias is not None:
                nn.init.zeros_(self.token_projection.bias)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _validate_rgb(self, x: Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(f"Expected input [B,C,H,W], got {tuple(x.shape)}")
        if x.shape[1] != 3:
            raise ValueError(f"Expected RGB input with 3 channels, got {x.shape[1]}")
        if x.shape[-2:] != self.image_size:
            raise ValueError(
                f"Expected image size {self.image_size}, got {tuple(x.shape[-2:])}"
            )

    def _extract_patch_tokens(self, x: Tensor) -> Tensor:
        tokens = self.dino.get_intermediate_layers(x, n=1)
        if isinstance(tokens, (list, tuple)):
            tokens = tokens[0]
        if isinstance(tokens, dict):
            if "x_norm_patchtokens" not in tokens:
                raise KeyError(
                    "DINO tokenizer output dict must contain 'x_norm_patchtokens'."
                )
            tokens = tokens["x_norm_patchtokens"]
        if not isinstance(tokens, Tensor):
            raise TypeError(f"Expected tensor patch tokens, got {type(tokens)}")
        return tokens

    def tokenize_image(self, x: Tensor) -> Tensor:
        self._validate_rgb(x)
        patch_tokens = self._extract_patch_tokens(x)
        return self.token_projection(patch_tokens)

    def forward_token_sequence(self, visual_tokens: Tensor) -> Tensor:
        return self.forward_token_sequence_intermediates(visual_tokens)[-1]

    def forward_token_sequence_intermediates(
        self,
        visual_tokens: Tensor,
        *,
        layer_indices: Sequence[int] | None = None,
    ) -> list[Tensor]:
        if visual_tokens.ndim != 3:
            raise ValueError(
                f"Expected visual_tokens [B,N,C], got {tuple(visual_tokens.shape)}"
            )
        if visual_tokens.shape[-1] != self.embed_dim:
            raise ValueError(
                f"Expected token dim {self.embed_dim}, got {visual_tokens.shape[-1]}"
            )
        if visual_tokens.shape[1] > self.max_visual_tokens:
            raise ValueError(
                f"Expected at most {self.max_visual_tokens} visual tokens, got {visual_tokens.shape[1]}"
            )
        special_token = self.special_token.expand(visual_tokens.shape[0], -1, -1)
        hidden = torch.cat([special_token, visual_tokens], dim=1)
        hidden = self.pos_drop(hidden + self.pos_embed[:, : hidden.shape[1]])

        if layer_indices is None:
            selected_indices = {len(self.blocks) - 1}
        else:
            selected_indices = {
                int(idx) if int(idx) >= 0 else len(self.blocks) + int(idx)
                for idx in layer_indices
            }
        if any(idx < 0 or idx >= len(self.blocks) for idx in selected_indices):
            raise ValueError(
                "layer_indices must refer to valid transformer block indices, got "
                f"{sorted(selected_indices)} for depth={len(self.blocks)}"
            )

        outputs: list[Tensor] = []
        for block_idx, block in enumerate(self.blocks):
            hidden = block(hidden)
            if block_idx in selected_indices:
                outputs.append(self.norm(hidden))
        if len(outputs) == 0:
            raise RuntimeError("No transformer intermediate features were selected.")
        return outputs

    def forward_features(self, x: Tensor) -> Tensor:
        return self.forward_token_sequence(self.tokenize_image(x))

    def forward_feature_intermediates(
        self,
        x: Tensor,
        *,
        layer_indices: Sequence[int] | None = None,
    ) -> list[Tensor]:
        return self.forward_token_sequence_intermediates(
            self.tokenize_image(x),
            layer_indices=layer_indices,
        )

    def shared_parameter_counts(self) -> OrderedDict[str, int]:
        frozen_tokenizer = sum(param.numel() for param in self.dino.parameters())
        trainable_tokenizer_adapter = (
            sum(param.numel() for param in self.token_projection.parameters())
            + int(self.special_token.numel())
            + int(self.pos_embed.numel())
        )
        trainable_trunk = (
            sum(param.numel() for block in self.blocks for param in block.parameters())
            + sum(param.numel() for param in self.norm.parameters())
        )
        return OrderedDict(
            frozen_tokenizer=int(frozen_tokenizer),
            trainable_tokenizer_adapter=int(trainable_tokenizer_adapter),
            trainable_trunk=int(trainable_trunk),
        )
