from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange
from jaxtyping import Float
from torch import Tensor

from .base import (
    BaseModelCfg,
    InputCommand,
    InputObservation,
    JacobianFieldInterface,
    JacobianFieldOutput,
)
from .registry import register_model

try:
    from vggt.heads.dpt_head import _make_fusion_block, _make_scratch, custom_interpolate
    from vggt.heads.utils import create_uv_grid, position_grid_to_embed
    from vggt.models.vggt import VGGT
except ImportError:  # pragma: no cover - exercised only when VGGT is unavailable
    VGGT = None
    _make_fusion_block = None
    _make_scratch = None
    custom_interpolate = None
    create_uv_grid = None
    position_grid_to_embed = None


def _default_intermediate_layer_idx() -> list[int]:
    return [4, 11, 17, 23]


class JacobianDptHead(nn.Module):
    def __init__(
        self,
        *,
        dim_in: int,
        patch_size: int,
        output_dim: int,
        features: int,
        out_channels: Sequence[int],
        intermediate_layer_idx: Sequence[int],
        pos_embed: bool = True,
        final_init_std: float | None = None,
        predict_uncertainty: bool = False,
    ) -> None:
        super().__init__()
        if _make_scratch is None or _make_fusion_block is None or custom_interpolate is None:
            raise ImportError("VGGT DPT utilities are unavailable. Ensure the `vggt` package is installed.")

        self.patch_size = int(patch_size)
        self.pos_embed = bool(pos_embed)
        self.intermediate_layer_idx = list(intermediate_layer_idx)

        self.norm = nn.LayerNorm(dim_in)
        self.projects = nn.ModuleList(
            [
                nn.Conv2d(dim_in, int(out_channel), kernel_size=1, stride=1, padding=0)
                for out_channel in out_channels
            ]
        )
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=int(out_channels[0]),
                    out_channels=int(out_channels[0]),
                    kernel_size=4,
                    stride=4,
                    padding=0,
                ),
                nn.ConvTranspose2d(
                    in_channels=int(out_channels[1]),
                    out_channels=int(out_channels[1]),
                    kernel_size=2,
                    stride=2,
                    padding=0,
                ),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=int(out_channels[3]),
                    out_channels=int(out_channels[3]),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )

        self.scratch = _make_scratch(list(out_channels), int(features), expand=False)
        self.scratch.refinenet1 = _make_fusion_block(int(features))
        self.scratch.refinenet2 = _make_fusion_block(int(features))
        self.scratch.refinenet3 = _make_fusion_block(int(features))
        self.scratch.refinenet4 = _make_fusion_block(int(features), has_residual=False)
        self.scratch.output_conv1 = nn.Conv2d(
            int(features), int(features // 2), kernel_size=3, stride=1, padding=1
        )
        self.output_conv2 = nn.Sequential(
            nn.Conv2d(int(features // 2), int(features // 8), kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(features // 8), int(output_dim), kernel_size=1, stride=1, padding=0),
        )

        self.predict_uncertainty = bool(predict_uncertainty)
        if self.predict_uncertainty:
            self.uncertainty_conv = nn.Sequential(
                nn.Conv2d(int(features // 2), int(features // 8), kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(int(features // 8), 1, kernel_size=1, stride=1, padding=0),
            )

        if final_init_std is not None:
            last_layer = self.output_conv2[-1]
            if isinstance(last_layer, nn.Conv2d):
                nn.init.normal_(last_layer.weight, mean=0.0, std=float(final_init_std))
                if last_layer.bias is not None:
                    nn.init.zeros_(last_layer.bias)

    def _apply_pos_embed(self, x: Tensor, width: int, height: int, ratio: float = 0.1) -> Tensor:
        if create_uv_grid is None or position_grid_to_embed is None:
            raise ImportError("VGGT positional embedding utilities are unavailable.")
        patch_w = x.shape[-1]
        patch_h = x.shape[-2]
        pos_embed = create_uv_grid(
            patch_w,
            patch_h,
            aspect_ratio=width / max(height, 1),
            dtype=x.dtype,
            device=x.device,
        )
        pos_embed = position_grid_to_embed(pos_embed, x.shape[1])
        pos_embed = pos_embed * ratio
        pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pos_embed

    def _scratch_forward(self, features: Sequence[Tensor]) -> Tensor:
        layer_1, layer_2, layer_3, layer_4 = features
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        out = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        del layer_4_rn, layer_4
        out = self.scratch.refinenet3(out, layer_3_rn, size=layer_2_rn.shape[2:])
        del layer_3_rn, layer_3
        out = self.scratch.refinenet2(out, layer_2_rn, size=layer_1_rn.shape[2:])
        del layer_2_rn, layer_2
        out = self.scratch.refinenet1(out, layer_1_rn)
        del layer_1_rn, layer_1
        return self.scratch.output_conv1(out)

    def forward(
        self,
        aggregated_tokens_list: Sequence[Tensor],
        *,
        images: Tensor,
        patch_start_idx: int,
    ) -> Tensor:
        batch_size, num_views, _, height, width = images.shape
        patch_h = height // self.patch_size
        patch_w = width // self.patch_size

        projected_features: list[Tensor] = []
        for project_idx, layer_idx in enumerate(self.intermediate_layer_idx):
            tokens = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]
            tokens = rearrange(tokens, "b s p c -> (b s) p c")
            tokens = self.norm(tokens)
            tokens = rearrange(tokens, "n (h w) c -> n c h w", h=patch_h, w=patch_w)
            tokens = self.projects[project_idx](tokens)
            if self.pos_embed:
                tokens = self._apply_pos_embed(tokens, width, height)
            tokens = self.resize_layers[project_idx](tokens)
            projected_features.append(tokens)

        fused = self._scratch_forward(projected_features)
        fused = custom_interpolate(
            fused,
            size=(patch_h * self.patch_size, patch_w * self.patch_size),
            mode="bilinear",
            align_corners=True,
        )
        if self.pos_embed:
            fused = self._apply_pos_embed(fused, width, height)
        output = self.output_conv2(fused)
        output = rearrange(output, "(b s) c h w -> b s c h w", b=batch_size, s=num_views)
        if self.predict_uncertainty:
            unc = self.uncertainty_conv(fused)
            unc = rearrange(unc, "(b s) c h w -> b s c h w", b=batch_size, s=num_views)
            return output, unc
        return output


@dataclass
class VggtJacobianFieldCfg(BaseModelCfg):
    name: Literal["vggt_jacobian"]
    image_size: int
    checkpoint_path: Optional[str] = None
    pretrained_model_id: Optional[str] = None
    strict_checkpoint_load: bool = True
    freeze_aggregator: bool = False
    aggregator_lr_multiplier: float = 1.0
    patch_size: int = 14
    embed_dim: int = 1024
    decoder_features: int = 256
    decoder_out_channels: Sequence[int] = field(
        default_factory=lambda: [256, 512, 1024, 1024]
    )
    decoder_intermediate_layer_idx: Sequence[int] = field(
        default_factory=_default_intermediate_layer_idx
    )
    decoder_pos_embed: bool = True
    decoder_init_std: float | None = None
    output_scale: float = 1.0
    predict_uncertainty: bool = False


@register_model("vggt_jacobian", cfg_cls=VggtJacobianFieldCfg)
class VggtJacobianField(JacobianFieldInterface):
    cfg: VggtJacobianFieldCfg
    supports_joint_multiview = True

    def __init__(self, model_cfg: VggtJacobianFieldCfg):
        super().__init__(cfg=model_cfg)
        if VGGT is None:
            raise ImportError("VGGT is unavailable. Ensure the `vggt` package is installed.")

        self.command_dim = int(model_cfg.command_dim)
        self.spatial_dim = int(model_cfg.spatial_dim)
        self.patch_size = int(model_cfg.patch_size)
        self.embed_dim = int(model_cfg.embed_dim)

        self.vggt = self._init_vggt_model(model_cfg)
        self._strip_unused_heads()
        self._load_pretrained_checkpoint(model_cfg)
        self._strip_unused_heads()

        if model_cfg.freeze_aggregator:
            for param in self.vggt.aggregator.parameters():
                param.requires_grad = False

        self.decoder = JacobianDptHead(
            dim_in=2 * self.embed_dim,
            patch_size=self.patch_size,
            output_dim=self.command_dim * self.spatial_dim,
            features=int(model_cfg.decoder_features),
            out_channels=list(model_cfg.decoder_out_channels),
            intermediate_layer_idx=list(model_cfg.decoder_intermediate_layer_idx),
            pos_embed=bool(model_cfg.decoder_pos_embed),
            final_init_std=model_cfg.decoder_init_std,
            predict_uncertainty=bool(model_cfg.predict_uncertainty),
        )
        self.output_scale = float(model_cfg.output_scale)
        self.predict_uncertainty = bool(model_cfg.predict_uncertainty)

    def get_optimizer_param_groups(
        self, *, base_lr: float
    ) -> list[nn.Parameter] | list[dict[str, object]]:
        if float(self.cfg.aggregator_lr_multiplier) == 1.0:
            return list(self.parameters())

        aggregator_params = [
            param
            for param in self.vggt.aggregator.parameters()
            if param.requires_grad
        ]
        decoder_params = [
            param
            for name, param in self.named_parameters()
            if param.requires_grad and not name.startswith("vggt.aggregator.")
        ]

        param_groups: list[dict[str, object]] = []
        if decoder_params:
            param_groups.append(
                {
                    "params": decoder_params,
                    "lr": float(base_lr),
                }
            )
        if aggregator_params:
            param_groups.append(
                {
                    "params": aggregator_params,
                    "lr": float(base_lr) * float(self.cfg.aggregator_lr_multiplier),
                }
            )
        return param_groups

    def _build_vggt_model(self, model_cfg: VggtJacobianFieldCfg) -> nn.Module:
        return VGGT(
            img_size=int(model_cfg.image_size),
            patch_size=int(model_cfg.patch_size),
            embed_dim=int(model_cfg.embed_dim),
            enable_camera=False,
            enable_point=False,
            enable_depth=False,
            enable_track=False,
        )

    def _load_from_pretrained_model_id(
        self, model_cfg: VggtJacobianFieldCfg
    ) -> nn.Module:
        if not hasattr(VGGT, "from_pretrained"):
            raise AttributeError("VGGT does not expose from_pretrained().")
        return VGGT.from_pretrained(str(model_cfg.pretrained_model_id))

    def _init_vggt_model(self, model_cfg: VggtJacobianFieldCfg) -> nn.Module:
        if model_cfg.pretrained_model_id:
            return self._load_from_pretrained_model_id(model_cfg)
        return self._build_vggt_model(model_cfg)

    def _strip_unused_heads(self) -> None:
        self.vggt.camera_head = None
        self.vggt.depth_head = None
        self.vggt.point_head = None
        self.vggt.track_head = None

    @staticmethod
    def _extract_state_dict(checkpoint: dict[str, Tensor] | Tensor) -> dict[str, Tensor]:
        if isinstance(checkpoint, dict):
            if "model" in checkpoint and isinstance(checkpoint["model"], dict):
                return checkpoint["model"]
            if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
                return checkpoint["state_dict"]
            return checkpoint
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")

    @staticmethod
    def _sanitize_state_dict(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        sanitized: dict[str, Tensor] = {}
        drop_prefixes = ("camera_head.", "depth_head.", "point_head.", "track_head.")
        for key, value in state_dict.items():
            normalized_key = key
            if normalized_key.startswith("module."):
                normalized_key = normalized_key[len("module.") :]
            if normalized_key.startswith("model."):
                normalized_key = normalized_key[len("model.") :]
            if normalized_key.startswith(drop_prefixes):
                continue
            sanitized[normalized_key] = value
        return sanitized

    def _load_pretrained_checkpoint(self, model_cfg: VggtJacobianFieldCfg) -> None:
        if not model_cfg.checkpoint_path:
            return

        checkpoint_path = Path(model_cfg.checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = self._sanitize_state_dict(self._extract_state_dict(checkpoint))
        missing, unexpected = self.vggt.load_state_dict(
            state_dict,
            strict=bool(model_cfg.strict_checkpoint_load),
        )
        if unexpected:
            raise RuntimeError(f"Unexpected VGGT checkpoint keys after sanitization: {unexpected}")
        if missing and bool(model_cfg.strict_checkpoint_load):
            raise RuntimeError(f"Missing VGGT checkpoint keys: {missing}")

    def _pad_to_patch_multiple(self, rgb: Tensor) -> tuple[Tensor, int, int]:
        height = int(rgb.shape[-2])
        width = int(rgb.shape[-1])
        pad_h = (-height) % self.patch_size
        pad_w = (-width) % self.patch_size
        if pad_h == 0 and pad_w == 0:
            return rgb, height, width
        batch_size, num_views = rgb.shape[:2]
        rgb = rearrange(rgb, "b v c h w -> (b v) c h w")
        rgb = F.pad(rgb, (0, pad_w, 0, pad_h), mode="replicate")
        rgb = rearrange(rgb, "(b v) c h w -> b v c h w", b=batch_size, v=num_views)
        return rgb, height, width

    def compute_jacobian(
        self, input_obs: InputObservation
    ) -> Float[Tensor, "batch c_dim s_dim height width"]:
        jacobian, _confidence = self._compute_jacobian_and_confidence(input_obs)
        return jacobian

    def _compute_jacobian_and_confidence(
        self, input_obs: InputObservation
    ) -> tuple[Tensor, Optional[Tensor]]:
        rgb = input_obs.rgb
        squeeze_view = rgb.ndim == 4
        if squeeze_view:
            rgb = rgb.unsqueeze(1)
        if rgb.ndim != 5:
            raise ValueError(
                "VGGT jacobian expects rgb with shape [B, 3, H, W] or [B, V, 3, H, W], "
                f"got {tuple(rgb.shape)}"
            )

        rgb, orig_height, orig_width = self._pad_to_patch_multiple(rgb)
        aggregated_tokens_list, patch_start_idx = self.vggt.aggregator(rgb)
        decoder_out = self.decoder(
            aggregated_tokens_list,
            images=rgb,
            patch_start_idx=patch_start_idx,
        )
        if self.predict_uncertainty:
            jacobian_flat, unc = decoder_out
        else:
            jacobian_flat, unc = decoder_out, None
        jacobian = rearrange(
            jacobian_flat,
            "b v (c_dim s_dim) h w -> b v c_dim s_dim h w",
            c_dim=self.command_dim,
            s_dim=self.spatial_dim,
        )
        jacobian = jacobian[..., :orig_height, :orig_width]
        jacobian = jacobian * self.output_scale

        confidence: Optional[Tensor] = None
        if unc is not None:
            unc = unc[..., :orig_height, :orig_width]
            # Convert raw output to a positive confidence map before returning.
            confidence = F.softplus(unc) + 1e-4

        if squeeze_view:
            jacobian = jacobian[:, 0]
            if confidence is not None:
                confidence = confidence[:, 0]
        return jacobian, confidence

    def forward(
        self,
        input_obs: InputObservation,
        input_cmd: InputCommand,
    ) -> JacobianFieldOutput:
        jacobian, confidence = self._compute_jacobian_and_confidence(input_obs)
        if jacobian.ndim == 6:
            flow = einsum(
                jacobian,
                input_cmd.du,
                "b v c_dim s_dim h w, b c_dim -> b v s_dim h w",
            )
        else:
            flow = einsum(
                jacobian,
                input_cmd.du,
                "b c_dim s_dim h w, b c_dim -> b s_dim h w",
            )

        return JacobianFieldOutput(
            jacobian=jacobian, optical_flow=flow, flow_confidence=confidence
        )
