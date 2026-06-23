"""Chunked DINO-feature Jacobian field.

Predicts a per-patch Jacobian J of shape (T, D, A) for each spatial patch,
where T is the action-chunk length, D is the DINO feature dim, A is the
action dim. Given a chunk of actions (T, A) the field predicts the
per-step DINO feature delta at each patch:

    δfeature_pred[t, p] = J[t, p] @ du[t]

This allows training on a *chunk* of consecutive frames (rgb_0 ... rgb_T)
from a single anchor observation rgb_0.

Memory note. Without factorization the head outputs T·D·A channels per
patch. Pusht (T=5, D=384, A=2): 3,840 ch — fine. Allegro (T=5, D=384,
A=16): 30,720 ch — bigger but tractable on H200. We optionally support
low-rank factorization J[t] = U[t] @ V[t]^T with rank R << D, cutting the
output to T·(D+A)·R channels per patch.
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
class DinoChunkJacobianFieldCfg(BaseModelCfg):
    name: Literal["dino_chunk_jacobian"]
    image_size: int
    chunk_length: int = 5
    # Factorization: 0 = full J[t]∈R^{D×A}; >0 = low-rank U[t]∈R^{D×R}, V[t]∈R^{R×A}.
    rank: int = 0
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
    "dino_chunk_jacobian",
    cfg_cls=DinoChunkJacobianFieldCfg,
)
class DinoChunkJacobianField(JacobianFieldInterface):
    """Per-patch chunked Jacobian J ∈ R^(T × D × A) over frozen DINOv2."""

    cfg: DinoChunkJacobianFieldCfg

    def __init__(self, model_cfg: DinoChunkJacobianFieldCfg):
        super().__init__(cfg=model_cfg)
        self.command_dim = int(model_cfg.command_dim)
        self.feature_dim = int(model_cfg.spatial_dim)
        self.chunk_length = int(model_cfg.chunk_length)
        self.rank = int(model_cfg.rank)

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

        hidden = int(model_cfg.head_hidden_dim)
        embed = int(model_cfg.embed_dim)

        if self.rank > 0:
            # Low-rank factorization output:  T * (D + A) * R channels per patch.
            out_per_token = self.chunk_length * (self.feature_dim + self.command_dim) * self.rank
        else:
            # Full Jacobian:  T * D * A channels per patch.
            out_per_token = self.chunk_length * self.feature_dim * self.command_dim

        self.head = nn.Sequential(
            nn.LayerNorm(embed),
            nn.Linear(embed, hidden),
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
    ) -> Float[Tensor, "batch chunk feature_dim command_dim grid_h grid_w"]:
        """Returns (B, T, D, A, gh, gw) — full materialized chunked Jacobian.

        For low-rank mode we materialize J[t] = U[t] @ V[t]^T here too so
        downstream code can treat both modes uniformly. The memory savings
        of the low-rank factorization come from the head output, not from
        the materialized Jacobian.
        """
        rgb = input_obs.rgb
        hidden = self.backbone.forward_features(rgb)
        patch_tokens = hidden[:, 1:]
        gh, gw = self.backbone.grid_size
        flat = self.head(patch_tokens)

        if self.rank > 0:
            # split flat into U and V chunks
            t_dr = self.chunk_length * self.feature_dim * self.rank
            u, v = flat[..., :t_dr], flat[..., t_dr:]
            u = rearrange(u, "b (gh gw) (t d r) -> b t d r gh gw",
                          gh=gh, gw=gw, t=self.chunk_length, d=self.feature_dim, r=self.rank)
            v = rearrange(v, "b (gh gw) (t a r) -> b t a r gh gw",
                          gh=gh, gw=gw, t=self.chunk_length, a=self.command_dim, r=self.rank)
            # J[t] = U[t] @ V[t]^T  -> (B, T, D, A, gh, gw)
            jacobian = einsum(u, v, "b t d r gh gw, b t a r gh gw -> b t d a gh gw")
        else:
            jacobian = rearrange(
                flat,
                "b (gh gw) (t d a) -> b t d a gh gw",
                gh=gh, gw=gw,
                t=self.chunk_length, d=self.feature_dim, a=self.command_dim,
            )
        return jacobian

    def forward(
        self,
        input_obs: InputObservation,
        input_cmd: InputCommand,
    ) -> JacobianFieldOutput:
        """input_cmd.du expected shape (B, T, A) — full chunk of actions."""
        jacobian = self.compute_jacobian(input_obs)  # (B, T, D, A, gh, gw)
        du = input_cmd.du  # (B, T, A)
        # Predict per-timestep δfeature for each patch.
        feature_delta = einsum(
            jacobian,
            du,
            "b t d a gh gw, b t a -> b t d gh gw",
        )
        # Reshape into the (B, S, gh, gw) slot of JacobianFieldOutput by
        # collapsing T into the channel dim. Caller must know to split.
        b, t, d, gh, gw = feature_delta.shape
        feature_delta_flat = rearrange(feature_delta, "b t d gh gw -> b (t d) gh gw")
        return JacobianFieldOutput(
            jacobian=jacobian, optical_flow=feature_delta_flat,
        )
