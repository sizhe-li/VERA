from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

from einops import einsum, rearrange
from jaxtyping import Float
from torch import Tensor, nn

from vera.idm.common.dino_tokenizer_transformer import (
    DinoTokenizerTransformerBackbone,
)
from vera.idm.jacobian.models.dpt_vggt_fusion_jacobian_field import (
    VggtStyleJacobianDecoder,
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
class PureTransformerJacobianDinoVggtFieldCfg(BaseModelCfg):
    name: Literal["pure_transformer_jacobian_dino_vggt"]
    image_size: int
    dino_model_name: str = "dinov2_vits14"
    freeze_dino: bool = True
    embed_dim: int = 384
    depth: int = 6
    num_heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    drop_path_rate: float = 0.0
    readout_mode: Literal["final", "intermediate"] = "final"
    intermediate_layer_indices: Sequence[int] = field(
        default_factory=lambda: [1, 2, 3, 5]
    )
    fusion_channels: int = 128
    prediction_hidden_channels: int = 64
    use_uv_coords: bool = False
    uv_fourier_frequencies: int = 0
    output_scale: float = 1.0


@register_model(
    "pure_transformer_jacobian_dino_vggt",
    cfg_cls=PureTransformerJacobianDinoVggtFieldCfg,
)
class PureTransformerJacobianDinoVggtField(JacobianFieldInterface):
    cfg: PureTransformerJacobianDinoVggtFieldCfg

    def __init__(self, model_cfg: PureTransformerJacobianDinoVggtFieldCfg):
        super().__init__(cfg=model_cfg)
        self.command_dim = int(model_cfg.command_dim)
        self.spatial_dim = int(model_cfg.spatial_dim)
        self.readout_mode = str(model_cfg.readout_mode)
        self.intermediate_layer_indices = [
            int(idx) for idx in model_cfg.intermediate_layer_indices
        ]
        self.output_scale = float(model_cfg.output_scale)

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
        decoder_scales = (
            1
            if self.readout_mode == "final"
            else min(len(self.intermediate_layer_indices), int(model_cfg.depth))
        )
        self.decoder = VggtStyleJacobianDecoder(
            in_channels=int(model_cfg.embed_dim),
            out_channels=self.command_dim * self.spatial_dim,
            fusion_channels=int(model_cfg.fusion_channels),
            prediction_hidden_channels=int(model_cfg.prediction_hidden_channels),
            max_scales=decoder_scales,
            use_uv_coords=bool(model_cfg.use_uv_coords),
            uv_fourier_frequencies=int(model_cfg.uv_fourier_frequencies),
        )

    def shared_parameter_counts(self):
        return self.backbone.shared_parameter_counts()

    def head_parameter_count(self) -> int:
        return int(sum(param.numel() for param in self.decoder.parameters()))

    def _tokens_to_feature_map(self, hidden: Tensor) -> Tensor:
        patch_tokens = hidden[:, 1:]
        grid_height, grid_width = self.backbone.grid_size
        return rearrange(
            patch_tokens,
            "b (gh gw) c -> b c gh gw",
            gh=grid_height,
            gw=grid_width,
        ).contiguous()

    def _extract_feature_maps(self, rgb: Tensor) -> list[Tensor]:
        if self.readout_mode == "final":
            return [self._tokens_to_feature_map(self.backbone.forward_features(rgb))]
        if self.readout_mode == "intermediate":
            hidden_states = self.backbone.forward_feature_intermediates(
                rgb,
                layer_indices=self.intermediate_layer_indices,
            )
            return [self._tokens_to_feature_map(hidden) for hidden in hidden_states]
        raise ValueError(f"Unsupported readout_mode: {self.readout_mode}")

    def compute_jacobian(
        self, input_obs: InputObservation
    ) -> Float[Tensor, "batch c_dim s_dim height width"]:
        rgb = input_obs.rgb
        height, width = rgb.shape[-2:]
        jacobian_flat = self.decoder(
            self._extract_feature_maps(rgb),
            target_hw=(height, width),
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
        optical_flow = einsum(
            jacobian,
            input_cmd.du,
            "b c_dim s_dim h w, b c_dim -> b s_dim h w",
        )
        return JacobianFieldOutput(jacobian=jacobian, optical_flow=optical_flow)
