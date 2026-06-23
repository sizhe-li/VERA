"""DINO-feature Jacobian field.

Predicts a per-patch Jacobian J ∈ R^(D × A) where D is the DINO feature
dimension and A is the action dimension. Multiplying J by an action
perturbation δa yields a predicted change in DINO features at each patch:

    δfeature_pred[p] = J[p] @ δa

This is the feature-space analogue of the optical-flow Jacobian: we learn
the local linear map between commands and learned representations rather
than between commands and optical flow.

DINO is loaded frozen and stays frozen (MVP). The feature target is
computed by running the same frozen DINO on the next frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from einops import einsum, rearrange
from jaxtyping import Float
from torch import Tensor, nn

from vera.idm.common.dino_tokenizer_transformer import (
    DinoTokenizerTransformerBackbone,
)

from .base import (
    BaseModelCfg,
    InputCommand,
    InputObservation,
    JacobianFieldInterface,
    JacobianFieldOutput,
)
from .registry import register_model


@dataclass
class DinoFeatureJacobianFieldCfg(BaseModelCfg):
    # NOTE: BaseModelCfg.spatial_dim is repurposed to mean DINO feature dim.
    # name + image_size kept positional (no default) to match dataclass
    # inheritance ordering pattern used elsewhere in this codebase.
    name: Literal["dino_feature_jacobian"]
    image_size: int
    dino_model_name: str = "dinov2_vits14"
    freeze_dino: bool = True
    embed_dim: int = 384
    depth: int = 4
    num_heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    drop_path_rate: float = 0.0
    head_hidden_dim: int = 512

    @property
    def feature_dim(self) -> int:
        return int(self.spatial_dim)


@register_model(
    "dino_feature_jacobian",
    cfg_cls=DinoFeatureJacobianFieldCfg,
)
class DinoFeatureJacobianField(JacobianFieldInterface):
    """Per-patch Jacobian J ∈ R^(D × A) over a frozen DINOv2 feature space."""

    cfg: DinoFeatureJacobianFieldCfg

    def __init__(self, model_cfg: DinoFeatureJacobianFieldCfg):
        super().__init__(cfg=model_cfg)
        self.command_dim = int(model_cfg.command_dim)
        # spatial_dim is repurposed as DINO feature dim
        self.feature_dim = int(model_cfg.spatial_dim)

        # Backbone: frozen DINO + small learnable transformer on top.
        self.backbone = DinoTokenizerTransformerBackbone(
            image_size=int(model_cfg.image_size),
            dino_model_name=str(model_cfg.dino_model_name),
            freeze_dino=bool(model_cfg.freeze_dino),
            embed_dim=int(model_cfg.embed_dim),
            depth=int(model_cfg.depth),
            num_heads=int(model_cfg.num_heads),
            mlp_ratio=float(model_cfg.mlp_ratio),
            dropout=float(model_cfg.dropout),
            drop_path_rate=float(model_cfg.drop_path_rate),
        )

        # Per-patch MLP head producing the (feature_dim × command_dim) Jacobian flat.
        out_per_token = self.feature_dim * self.command_dim
        hidden = int(model_cfg.head_hidden_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(int(model_cfg.embed_dim)),
            nn.Linear(int(model_cfg.embed_dim), hidden),
            nn.GELU(),
            nn.Linear(hidden, out_per_token),
        )

    def shared_parameter_counts(self):
        return self.backbone.shared_parameter_counts()

    def head_parameter_count(self) -> int:
        return int(sum(param.numel() for param in self.head.parameters()))

    @property
    def grid_size(self) -> tuple[int, int]:
        return self.backbone.grid_size

    def compute_jacobian(
        self, input_obs: InputObservation
    ) -> Float[Tensor, "batch feature_dim command_dim grid_h grid_w"]:
        rgb = input_obs.rgb
        hidden = self.backbone.forward_features(rgb)
        patch_tokens = hidden[:, 1:]  # drop the special token
        gh, gw = self.backbone.grid_size
        # (B, N, embed_dim) -> (B, N, feature_dim * command_dim)
        flat = self.head(patch_tokens)
        # -> (B, feature_dim, command_dim, gh, gw)
        return rearrange(
            flat,
            "b (gh gw) (f c) -> b f c gh gw",
            gh=gh,
            gw=gw,
            f=self.feature_dim,
            c=self.command_dim,
        )

    def forward(
        self,
        input_obs: InputObservation,
        input_cmd: InputCommand,
    ) -> JacobianFieldOutput:
        jacobian = self.compute_jacobian(input_obs)
        # Predict δfeature at each patch given the action.
        feature_delta = einsum(
            jacobian,
            input_cmd.du,
            "b f c gh gw, b c -> b f gh gw",
        )
        # Reuse the JacobianFieldOutput slot meant for optical_flow to carry
        # the predicted feature delta. The owning algorithm interprets it.
        return JacobianFieldOutput(jacobian=jacobian, optical_flow=feature_delta)
