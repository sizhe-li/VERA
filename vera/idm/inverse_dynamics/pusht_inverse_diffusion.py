from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from vera.idm.common.base_pytorch_algo import BasePytorchAlgo, BasePytorchAlgoCfg
from vera.idm.dfot.diffusion.noise_schedule import make_beta_schedule
from vera.idm.registry import register_algorithm
from vera.datasets.normalization import (
    denormalize_action,
    get_action_normalization_metadata,
)
from vera.utils.logging_utils import get_sanity_metrics, log_video_tensor

"""
The input/output behavior is now meant to look like this during training:


full sampled window from dataset
(B, T=5, C, H, W) and (B, T=5, action_dim)
time --->
frames:   [ f0 ][ f1 ][ f2 ][ f3 ][ f4 ]
actions:      a0   a1   a2   a3   a4
              |    |    |
              |    |    +-- future action chunk step 3
              |    +------- future action chunk step 2
              +------------ future action chunk step 1
split into strict IDM pieces:
context/source:
  rgb_src = [ f0 , f1 ]                  shape: (B, 2, C, H, W)
future/target:
  rgb_tgt = [ f2 , f3 , f4 ]             shape: (B, 3, C, H, W)
supervision:
  du_chunk = [ a1 , a2 , a3 ]            shape: (B, 3, action_dim)
model behavior:
  (rgb_src, rgb_tgt, noisy_du_chunk, diffusion_t)
      |
      v
  predict clean future action chunk
      |
      v
  du_chunk_hat = [ â1 , â2 , â3 ]
And at inference time, the standalone policy currently behaves like:

observed context only:
  rgb_src = [ recent 2 frames ]
target future:
  rgb_tgt = placeholder future of length 3
            (right now: repeat last observed frame)
model output:
  predicted 3-step action chunk
runner uses:
  first action immediately
  optional chunk info stays available in policy info
"""


def extract(a: Tensor, t: Tensor, x_shape: torch.Size) -> Tensor:
    out = a[t]
    return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))


def timestep_embedding(timesteps: Tensor, dim: int, max_period: int = 10000) -> Tensor:
    half = dim // 2
    device = timesteps.device
    freqs = torch.exp(
        -torch.log(torch.tensor(float(max_period), device=device))
        * torch.arange(start=0, end=half, device=device, dtype=torch.float32)
        / max(half, 1)
    )
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class FrameEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(128, hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.net(x)
        x = x.flatten(1)
        return self.proj(x)


class PushTInverseDiffusionModel(nn.Module):
    def __init__(
        self,
        *,
        action_dim: int,
        action_horizon: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float,
        max_source_frames: int,
        max_target_frames: int,
        in_channels: int,
        timestep_dim: int,
        use_modality_timestep_adapters: bool = False,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.max_source_frames = max_source_frames
        self.max_target_frames = max_target_frames
        self.action_token_dim = action_dim * action_horizon
        self.use_modality_timestep_adapters = bool(use_modality_timestep_adapters)

        self.frame_encoder = FrameEncoder(
            in_channels=in_channels, hidden_dim=hidden_dim
        )
        self.action_embed = nn.Linear(self.action_token_dim, hidden_dim)
        self.timestep_mlp = nn.Sequential(
            nn.Linear(timestep_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        if self.use_modality_timestep_adapters:
            self.visual_timestep_adapter = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.action_timestep_adapter = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.visual_timestep_adapter = None
            self.action_timestep_adapter = None
        self.src_pos_embed = nn.Parameter(
            torch.randn(1, max_source_frames, hidden_dim) * 0.02
        )
        self.tgt_pos_embed = nn.Parameter(
            torch.randn(1, max_target_frames, hidden_dim) * 0.02
        )
        self.action_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.action_type_embed = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.src_type_embed = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.tgt_type_embed = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, self.action_token_dim)

    def forward(
        self,
        rgb_src: Tensor,
        rgb_tgt: Tensor,
        noisy_action: Tensor,
        timesteps: Tensor,
    ) -> Tensor:
        if rgb_src.ndim != 5:
            raise ValueError(
                f"Expected rgb_src to have shape [B,T,C,H,W], got {tuple(rgb_src.shape)}"
            )
        if rgb_tgt.ndim != 5:
            raise ValueError(
                f"Expected rgb_tgt to have shape [B,T,C,H,W], got {tuple(rgb_tgt.shape)}"
            )
        batch_size, num_src_frames = rgb_src.shape[:2]
        _, num_tgt_frames = rgb_tgt.shape[:2]
        if num_src_frames > self.max_source_frames:
            raise ValueError(
                f"num_src_frames={num_src_frames} exceeds max_source_frames={self.max_source_frames}"
            )
        if num_tgt_frames > self.max_target_frames:
            raise ValueError(
                f"num_tgt_frames={num_tgt_frames} exceeds max_target_frames={self.max_target_frames}"
            )

        src_tokens = self.frame_encoder(rgb_src.flatten(0, 1)).view(
            batch_size, num_src_frames, -1
        )
        src_tokens = (
            src_tokens + self.src_pos_embed[:, :num_src_frames] + self.src_type_embed
        )
        tgt_tokens = self.frame_encoder(rgb_tgt.flatten(0, 1)).view(
            batch_size, num_tgt_frames, -1
        )
        tgt_tokens = (
            tgt_tokens + self.tgt_pos_embed[:, :num_tgt_frames] + self.tgt_type_embed
        )

        time_embed = self.timestep_mlp(
            timestep_embedding(timesteps, self.timestep_mlp[0].in_features)
        )
        action_time_embed = time_embed
        if (
            self.use_modality_timestep_adapters
            and self.visual_timestep_adapter is not None
            and self.action_timestep_adapter is not None
        ):
            visual_time_embed = self.visual_timestep_adapter(time_embed)
            action_time_embed = self.action_timestep_adapter(time_embed)
            src_tokens = src_tokens + visual_time_embed[:, None]
            tgt_tokens = tgt_tokens + visual_time_embed[:, None]
        action_token = (
            self.action_token
            + self.action_type_embed
            + self.action_embed(noisy_action)[:, None]
            + action_time_embed[:, None]
        )

        tokens = torch.cat([action_token, src_tokens, tgt_tokens], dim=1)
        hidden = self.transformer(tokens)
        action_hidden = self.norm(hidden[:, 0])
        return self.head(action_hidden)


@dataclass
class InverseDiffusionModelCfg:
    hidden_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    mlp_dim: int = 512
    dropout: float = 0.1
    context_frames: int = 2
    future_frames: int = 3
    action_horizon: int = 3
    in_channels: int = 3
    action_dim: int = 2
    timestep_dim: int = 256
    use_modality_timestep_adapters: bool = False


@dataclass
class InverseDiffusionScheduleCfg:
    timesteps: int = 100
    sampling_timesteps: int = 20
    beta_schedule: Literal[
        "cosine", "sigmoid", "sd", "linear", "alphas_cumprod_linear"
    ] = "cosine"
    objective: Literal["pred_noise", "pred_x0", "pred_v"] = "pred_noise"
    ddim_sampling_eta: float = 0.0
    clip_noise: float = 5.0
    min_alpha_cumprod: float = 1e-12
    schedule_fn_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class InverseDiffusionOptimizerCfg:
    weight_decay: float = 1e-4
    beta: list[float] = field(default_factory=lambda: [0.9, 0.999])


@dataclass
class InverseDiffusionLoggingCfg:
    loss_freq: int = 100
    max_validation_videos: int = 1
    validation_video_fps: int = 8


@dataclass
class PushTInverseDiffusionCfg(BasePytorchAlgoCfg):
    name: Literal["pusht_inverse_diffusion"]
    image_size: list[int] = field(default_factory=lambda: [252, 252])
    model: InverseDiffusionModelCfg = field(default_factory=InverseDiffusionModelCfg)
    diffusion: InverseDiffusionScheduleCfg = field(
        default_factory=InverseDiffusionScheduleCfg
    )
    action_norm_method: Literal["std", "quantile"] | None = None
    used_action_channel_ids: list[int] | None = None
    inverse_used_action_channel_ids: list[int] | None = None
    optimizer: InverseDiffusionOptimizerCfg = field(
        default_factory=InverseDiffusionOptimizerCfg
    )
    logging: InverseDiffusionLoggingCfg = field(
        default_factory=InverseDiffusionLoggingCfg
    )


@register_algorithm("pusht_inverse_diffusion", cfg_cls=PushTInverseDiffusionCfg)
class PushTInverseDiffusion(BasePytorchAlgo):
    cfg: PushTInverseDiffusionCfg
    model: PushTInverseDiffusionModel

    def _validate_config(self) -> None:
        if self.cfg.model.context_frames < 1:
            raise ValueError("model.context_frames must be >= 1.")
        if self.cfg.model.future_frames < 1:
            raise ValueError("model.future_frames must be >= 1.")
        if self.cfg.model.action_horizon < 1:
            raise ValueError("model.action_horizon must be >= 1.")
        if self.cfg.model.action_horizon > self.cfg.model.future_frames:
            raise ValueError(
                "model.action_horizon must be <= model.future_frames "
                f"({self.cfg.model.action_horizon} > {self.cfg.model.future_frames})."
            )

    def _build_model(self):
        torch.set_float32_matmul_precision("high")
        self._validate_config()
        self.model = PushTInverseDiffusionModel(
            action_dim=self.cfg.model.action_dim,
            action_horizon=self.cfg.model.action_horizon,
            hidden_dim=self.cfg.model.hidden_dim,
            num_layers=self.cfg.model.num_layers,
            num_heads=self.cfg.model.num_heads,
            mlp_dim=self.cfg.model.mlp_dim,
            dropout=self.cfg.model.dropout,
            max_source_frames=self.cfg.model.context_frames,
            max_target_frames=self.cfg.model.future_frames,
            in_channels=self.cfg.model.in_channels,
            timestep_dim=self.cfg.model.timestep_dim,
            use_modality_timestep_adapters=self.cfg.model.use_modality_timestep_adapters,
        )
        self._build_diffusion_buffers()

    def _build_diffusion_buffers(self) -> None:
        diffusion_cfg = self.cfg.diffusion
        betas = make_beta_schedule(
            schedule=diffusion_cfg.beta_schedule,
            timesteps=diffusion_cfg.timesteps,
            zero_terminal_snr=diffusion_cfg.objective != "pred_noise",
            **diffusion_cfg.schedule_fn_kwargs,
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        min_alpha_cumprod = float(diffusion_cfg.min_alpha_cumprod)
        alphas_cumprod = alphas_cumprod.clamp(min=min_alpha_cumprod, max=1.0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0).clamp(
            min=min_alpha_cumprod,
            max=1.0,
        )
        one_minus_alphas_cumprod = (1.0 - alphas_cumprod).clamp(min=1e-12, max=1.0)

        register_buffer = lambda name, value: self.register_buffer(  # noqa: E731
            name, value.to(torch.float32), persistent=False
        )
        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(one_minus_alphas_cumprod),
        )
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0)
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / one_minus_alphas_cumprod
        )
        register_buffer("posterior_variance", posterior_variance)
        register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / one_minus_alphas_cumprod,
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / one_minus_alphas_cumprod,
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.optimizer.weight_decay,
            betas=tuple(self.cfg.optimizer.beta),
        )

    @staticmethod
    def _select_rgb_context(rgb: Tensor) -> Tensor:
        if rgb.ndim == 6:
            if rgb.shape[2] != 1:
                raise ValueError(
                    f"PushT inverse diffusion expects a single view, got rgb shape {tuple(rgb.shape)}"
                )
            rgb = rgb[:, :, 0]
        if rgb.ndim != 5:
            raise ValueError(
                f"Expected rgb to have shape [B,T,C,H,W], got {tuple(rgb.shape)}"
            )
        return rgb

    def _split_rgb_window(self, rgb: Tensor) -> tuple[Tensor, Tensor]:
        rgb = self._select_rgb_context(rgb)
        context_frames = int(self.cfg.model.context_frames)
        future_frames = int(self.cfg.model.future_frames)
        expected = context_frames + future_frames
        if rgb.shape[1] < expected:
            raise ValueError(
                f"Expected rgb window with at least {expected} frames, got {rgb.shape[1]}"
            )
        return (
            rgb[:, :context_frames],
            rgb[:, context_frames : context_frames + future_frames],
        )

    def _prepare_conditioning(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        if "rgb_src" in batch and "rgb_tgt" in batch:
            rgb_src = self._select_rgb_context(batch["rgb_src"])
            rgb_tgt = self._select_rgb_context(batch["rgb_tgt"])
            return rgb_src.to(self.device), rgb_tgt.to(self.device)
        if "rgb" in batch:
            rgb_src, rgb_tgt = self._split_rgb_window(batch["rgb"])
            return rgb_src.to(self.device), rgb_tgt.to(self.device)
        raise KeyError("Batch must contain either (rgb_src, rgb_tgt) or rgb.")

    def _select_target_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        action_dim = int(self.cfg.model.action_dim)
        action_horizon = int(self.cfg.model.action_horizon)
        if "du_chunk" in batch:
            du_chunk = batch["du_chunk"]
        elif "du" in batch:
            du = batch["du"]
            if du.ndim != 3:
                raise ValueError(
                    f"Expected du to have shape [B,T,C] when du_chunk is missing, got {tuple(du.shape)}"
                )
            start = int(self.cfg.model.context_frames) - 1
            end = start + action_horizon
            du_chunk = du[:, start:end]
        else:
            raise KeyError("Batch must contain either du_chunk or du.")

        if du_chunk.ndim == 2:
            du_chunk = du_chunk[:, None]
        if du_chunk.ndim != 3:
            raise ValueError(
                f"Expected du_chunk to have shape [B,H,A], got {tuple(du_chunk.shape)}"
            )
        if du_chunk.shape[1] != action_horizon:
            raise ValueError(
                f"Expected action horizon {action_horizon}, got {du_chunk.shape[1]}"
            )
        used_channels = self.cfg.used_action_channel_ids
        if used_channels is None:
            if du_chunk.shape[2] != action_dim:
                raise ValueError(
                    f"Expected action_dim {action_dim}, got {du_chunk.shape[2]}"
                )
            return du_chunk.to(self.device)

        if len(used_channels) != action_dim:
            raise ValueError(
                "When used_action_channel_ids is set, its length must match "
                f"model.action_dim ({len(used_channels)} != {action_dim})."
            )
        if len(used_channels) == 0:
            raise ValueError("used_action_channel_ids must not be empty.")
        if len(set(used_channels)) != len(used_channels):
            raise ValueError("used_action_channel_ids must contain unique indices.")
        if min(used_channels) < 0:
            raise ValueError("used_action_channel_ids must be non-negative.")
        if max(used_channels) >= du_chunk.shape[2]:
            raise ValueError(
                f"used_action_channel_ids has index {max(used_channels)} but du_chunk has only "
                f"{du_chunk.shape[2]} channels."
            )
        index = torch.tensor(used_channels, device=du_chunk.device, dtype=torch.long)
        return du_chunk.index_select(2, index).to(self.device)

    def _select_action_metadata_channels(
        self,
        values: Any,
        *,
        expected_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor | None:
        if values is None:
            return None
        value = torch.as_tensor(values, device=device, dtype=dtype).flatten()
        used_channels = self.cfg.used_action_channel_ids
        if used_channels is not None:
            if len(used_channels) == 0:
                return None
            if max(used_channels) >= value.numel():
                return None
            index = torch.tensor(used_channels, device=device, dtype=torch.long)
            value = value.index_select(0, index)
        if value.numel() != expected_dim:
            return None
        return value

    def _normalize_action_chunk(self, du_chunk: Tensor) -> Tensor:
        method = self.cfg.action_norm_method
        if method is None:
            return du_chunk
        action_meta = get_action_normalization_metadata(self.dataset_metadata)
        action_dim = du_chunk.shape[-1]
        if method == "std":
            mean = self._select_action_metadata_channels(
                action_meta.get("action_mean"),
                expected_dim=action_dim,
                device=du_chunk.device,
                dtype=du_chunk.dtype,
            )
            std = self._select_action_metadata_channels(
                action_meta.get("action_std"),
                expected_dim=action_dim,
                device=du_chunk.device,
                dtype=du_chunk.dtype,
            )
            if mean is None or std is None:
                return du_chunk
            view_shape = (1,) * (du_chunk.ndim - 1) + (action_dim,)
            mean = mean.view(view_shape)
            std = std.clamp(min=1e-6).view(view_shape)
            return (du_chunk - mean) / std
        if method == "quantile":
            low = self._select_action_metadata_channels(
                action_meta.get("action_min"),
                expected_dim=action_dim,
                device=du_chunk.device,
                dtype=du_chunk.dtype,
            )
            high = self._select_action_metadata_channels(
                action_meta.get("action_max"),
                expected_dim=action_dim,
                device=du_chunk.device,
                dtype=du_chunk.dtype,
            )
            if low is None or high is None:
                return du_chunk
            center = (high + low) * 0.5
            scale = (high - low).abs().clamp(min=1e-6) * 0.5
            view_shape = (1,) * (du_chunk.ndim - 1) + (action_dim,)
            center = center.view(view_shape)
            scale = scale.view(view_shape)
            return (du_chunk - center) / scale
        raise ValueError(
            f"Unsupported action_norm_method={method}. Use one of: null, std, quantile."
        )

    def _flatten_action_chunk(self, du_chunk: Tensor) -> Tensor:
        return du_chunk.reshape(du_chunk.shape[0], -1)

    def _unflatten_action_chunk(self, flat_action: Tensor) -> Tensor:
        return flat_action.reshape(
            flat_action.shape[0],
            int(self.cfg.model.action_horizon),
            int(self.cfg.model.action_dim),
        )

    @staticmethod
    def _is_validation_sequence_batch(batch: dict[str, Tensor]) -> bool:
        return "rgb_sequence" in batch and "du_sequence" in batch

    def _build_sliding_window_batch(
        self,
        *,
        rgb_sequence: Tensor,
        du_sequence: Tensor,
        window_starts: Tensor,
    ) -> dict[str, Tensor]:
        context_frames = int(self.cfg.model.context_frames)
        future_frames = int(self.cfg.model.future_frames)
        action_horizon = int(self.cfg.model.action_horizon)
        window_length = context_frames + future_frames
        start_values = [int(v) for v in window_starts.detach().cpu().tolist()]
        if not start_values:
            raise ValueError("window_starts must contain at least one sliding window.")

        rgb_src = torch.stack(
            [rgb_sequence[start : start + context_frames] for start in start_values],
            dim=0,
        )
        rgb_tgt = torch.stack(
            [
                rgb_sequence[
                    start + context_frames : start + context_frames + future_frames
                ]
                for start in start_values
            ],
            dim=0,
        )
        action_start = context_frames - 1
        du_chunk = torch.stack(
            [
                du_sequence[
                    start + action_start : start + action_start + action_horizon
                ]
                for start in start_values
            ],
            dim=0,
        )
        if rgb_src.shape[1] + rgb_tgt.shape[1] != window_length:
            raise ValueError(
                f"Expected window length {window_length}, got {rgb_src.shape[1] + rgb_tgt.shape[1]}"
            )
        return {
            "rgb_src": rgb_src,
            "rgb_tgt": rgb_tgt,
            "du_chunk": du_chunk,
        }

    def _aggregate_action_chunks(
        self,
        *,
        action_chunks: Tensor,
        window_starts: Tensor,
        sequence_length: int,
    ) -> tuple[Tensor, Tensor]:
        action_horizon = int(self.cfg.model.action_horizon)
        action_offset = int(self.cfg.model.context_frames) - 1
        action_dim = int(self.cfg.model.action_dim)
        summed = torch.zeros(
            sequence_length,
            action_dim,
            device=action_chunks.device,
            dtype=action_chunks.dtype,
        )
        counts = torch.zeros(
            sequence_length,
            device=action_chunks.device,
            dtype=action_chunks.dtype,
        )
        horizon_offsets = torch.arange(action_horizon, device=action_chunks.device)

        for chunk_idx, start in enumerate(window_starts.to(action_chunks.device)):
            timeline_indices = start + action_offset + horizon_offsets
            summed.index_add_(0, timeline_indices, action_chunks[chunk_idx])
            counts.index_add_(
                0,
                timeline_indices,
                torch.ones(action_horizon, device=counts.device, dtype=counts.dtype),
            )

        averaged = torch.full_like(summed, float("nan"))
        valid = counts > 0
        averaged[valid] = summed[valid] / counts[valid].unsqueeze(-1)
        return averaged, counts

    @staticmethod
    def _rgb_sequence_to_uint8(rgb_sequence: Tensor) -> np.ndarray:
        rgb_cpu = rgb_sequence.detach().cpu()
        if rgb_cpu.ndim != 4:
            raise ValueError(
                f"Expected rgb_sequence to have shape [T,C,H,W], got {tuple(rgb_cpu.shape)}"
            )
        if rgb_cpu.dtype != torch.uint8:
            rgb_cpu = (torch.clamp(rgb_cpu, 0.0, 1.0) * 255.0).to(torch.uint8)
        if rgb_cpu.shape[1] == 1:
            rgb_cpu = rgb_cpu.repeat(1, 3, 1, 1)
        if rgb_cpu.shape[1] != 3:
            raise ValueError(
                f"Expected rgb_sequence to have 3 channels, got {rgb_cpu.shape[1]}"
            )
        return rgb_cpu.permute(0, 2, 3, 1).numpy()

    @staticmethod
    def _add_panel_label(image_chw: np.ndarray, label: str) -> np.ndarray:
        canvas = np.ascontiguousarray(image_chw.transpose(1, 2, 0).copy())
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 24), (0, 0, 0), thickness=-1)
        cv2.putText(
            canvas,
            label,
            (8, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return canvas.transpose(2, 0, 1)

    @staticmethod
    def _concat_side_by_side(left_chw: np.ndarray, right_chw: np.ndarray) -> np.ndarray:
        if left_chw.shape[1] != right_chw.shape[1]:
            raise ValueError(
                f"Expected matching heights, got {left_chw.shape} and {right_chw.shape}"
            )
        return np.concatenate([left_chw, right_chw], axis=2)

    @staticmethod
    def _render_action_curve_panel(
        gt_actions: np.ndarray,
        pred_actions: np.ndarray,
        frame_idx: int,
        *,
        height: int,
        width: int,
    ) -> np.ndarray:
        canvas = np.full((height, width, 3), 18, dtype=np.uint8)
        if gt_actions.shape != pred_actions.shape:
            raise ValueError(
                f"Expected matching action shapes, got {gt_actions.shape} and {pred_actions.shape}"
            )
        if gt_actions.ndim != 2 or gt_actions.shape[1] != 2:
            raise ValueError(
                f"Expected [T,2] actions for PushT visualization, got {gt_actions.shape}"
            )

        total_steps = gt_actions.shape[0]
        margin_left = 52
        margin_right = 18
        margin_top = 20
        margin_bottom = 28
        row_gap = 18
        plot_width = max(width - margin_left - margin_right, 32)
        plot_height = max((height - margin_top - margin_bottom - row_gap) // 2, 32)
        x_coords = np.linspace(
            margin_left,
            margin_left + plot_width,
            total_steps,
        ).astype(np.int32)

        finite_pred = pred_actions[np.isfinite(pred_actions)]
        finite_values = [gt_actions.reshape(-1)]
        if finite_pred.size:
            finite_values.append(finite_pred.reshape(-1))
        max_abs = max(float(np.max(np.abs(np.concatenate(finite_values)))), 1e-4)
        max_abs *= 1.15

        gt_color = (90, 220, 90)
        pred_color = (255, 191, 0)
        cursor_color = (220, 220, 220)
        zero_color = (80, 80, 80)

        def project_y(value: float, top: int) -> int:
            normalized = 0.5 - (float(value) / (2.0 * max_abs))
            normalized = float(np.clip(normalized, 0.0, 1.0))
            return int(top + normalized * plot_height)

        def draw_series(values: np.ndarray, top: int, color: tuple[int, int, int]) -> None:
            prev_point: tuple[int, int] | None = None
            for x, value in zip(x_coords, values):
                if not np.isfinite(value):
                    prev_point = None
                    continue
                point = (int(x), project_y(float(value), top))
                if prev_point is not None:
                    cv2.line(canvas, prev_point, point, color, 2, cv2.LINE_AA)
                prev_point = point

        for dim, label in enumerate(("du_x", "du_y")):
            top = margin_top + dim * (plot_height + row_gap)
            bottom = top + plot_height
            mid_y = project_y(0.0, top)
            cv2.rectangle(
                canvas,
                (margin_left, top),
                (margin_left + plot_width, bottom),
                (58, 58, 58),
                1,
            )
            cv2.line(
                canvas,
                (margin_left, mid_y),
                (margin_left + plot_width, mid_y),
                zero_color,
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                label,
                (8, top + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )
            draw_series(gt_actions[:, dim], top, gt_color)
            draw_series(pred_actions[:, dim], top, pred_color)

        cursor_idx = int(np.clip(frame_idx, 0, total_steps - 1))
        cursor_x = int(x_coords[cursor_idx])
        cv2.line(
            canvas,
            (cursor_x, margin_top),
            (cursor_x, height - margin_bottom),
            cursor_color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "GT",
            (margin_left, height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            gt_color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "Recon",
            (margin_left + 42, height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            pred_color,
            1,
            cv2.LINE_AA,
        )
        return canvas

    def _build_validation_video(
        self,
        *,
        rgb_sequence: Tensor,
        gt_actions: Tensor,
        pred_actions: Tensor,
    ) -> np.ndarray:
        rgb_frames = self._rgb_sequence_to_uint8(rgb_sequence)
        gt_np = gt_actions.detach().cpu().numpy()
        pred_np = pred_actions.detach().cpu().numpy()
        panel_width = max(rgb_frames.shape[2], 360)
        video_frames: list[np.ndarray] = []
        for frame_idx, rgb_frame in enumerate(rgb_frames):
            left = self._add_panel_label(
                rgb_frame.transpose(2, 0, 1),
                "PushT RGB",
            )
            right = self._render_action_curve_panel(
                gt_np,
                pred_np,
                frame_idx,
                height=rgb_frame.shape[0],
                width=panel_width,
            )
            right = self._add_panel_label(
                right.transpose(2, 0, 1),
                "GT vs reconstructed du",
            )
            frame_chw = self._concat_side_by_side(left, right)
            video_frames.append(frame_chw)
        return np.stack(video_frames, axis=0)

    def _summarize_validation_sequence_sample(
        self,
        *,
        rgb_sequence: Tensor,
        du_sequence: Tensor,
        window_starts: Tensor,
    ) -> dict[str, Any]:
        window_batch = self._build_sliding_window_batch(
            rgb_sequence=rgb_sequence,
            du_sequence=du_sequence,
            window_starts=window_starts,
        )
        loss, metrics, _ = self.compute_batch(window_batch)
        pred_chunks = self.sample_action(
            window_batch["rgb_src"],
            window_batch["rgb_tgt"],
            return_normalized=False,
        )
        pred_timeline, pred_counts = self._aggregate_action_chunks(
            action_chunks=pred_chunks,
            window_starts=window_starts.to(pred_chunks.device),
            sequence_length=int(du_sequence.shape[0]),
        )
        gt_timeline = self._denormalize_action(du_sequence.to(self.device)).detach()
        valid = pred_counts > 0
        if valid.any():
            metrics["timeline_action_mse"] = F.mse_loss(
                pred_timeline[valid],
                gt_timeline[valid],
            )
            metrics["timeline_pred_action_norm"] = pred_timeline[valid].norm(dim=-1).mean()
            metrics["timeline_target_action_norm"] = gt_timeline[valid].norm(dim=-1).mean()
        metrics["timeline_prediction_coverage"] = valid.float().mean()
        metrics["timeline_num_windows"] = torch.tensor(
            float(window_starts.numel()),
            device=self.device,
        )
        return {
            "loss": loss,
            "metrics": metrics,
            "gt_timeline": gt_timeline,
            "pred_timeline": pred_timeline.detach(),
        }

    def _log_validation_sequence_videos(
        self,
        *,
        batch: dict[str, Tensor],
        summaries: list[dict[str, Any]],
        batch_idx: int,
    ) -> None:
        if batch_idx != 0 or self.logger is None:
            return
        if self.trainer is not None and not self.trainer.is_global_zero:
            return
        if not hasattr(self.logger, "experiment"):
            return

        max_videos = max(0, int(self.cfg.logging.max_validation_videos))
        if max_videos == 0:
            return

        for sample_idx, summary in enumerate(summaries[:max_videos]):
            video = self._build_validation_video(
                rgb_sequence=batch["rgb_sequence"][sample_idx],
                gt_actions=summary["gt_timeline"],
                pred_actions=summary["pred_timeline"],
            )
            episode_idx = None
            if "episode_idx" in batch:
                episode_idx = int(batch["episode_idx"][sample_idx].item())
            caption = (
                f"episode={episode_idx}"
                if episode_idx is not None
                else "PushT validation reconstruction"
            )
            log_video_tensor(
                name=f"validation/pusht_reconstruction_{sample_idx}",
                video=video,
                step=self.global_step,
                fps=int(self.cfg.logging.validation_video_fps),
                caption=caption,
                logger=self.logger.experiment,
            )

    def _validation_sequence_step(self, batch: dict[str, Tensor], batch_idx: int):
        batch_size = int(batch["rgb_sequence"].shape[0])
        summaries: list[dict[str, Any]] = []
        loss_values: list[Tensor] = []
        metric_values: dict[str, list[Tensor | float | int]] = {}

        for sample_idx in range(batch_size):
            summary = self._summarize_validation_sequence_sample(
                rgb_sequence=batch["rgb_sequence"][sample_idx],
                du_sequence=batch["du_sequence"][sample_idx],
                window_starts=batch["window_starts"][sample_idx],
            )
            summaries.append(summary)
            loss_values.append(summary["loss"])
            for key, value in summary["metrics"].items():
                metric_values.setdefault(key, []).append(value)

        loss = torch.stack(loss_values).mean()
        reduced_metrics: dict[str, Tensor | float | int] = {}
        for key, values in metric_values.items():
            tensor_values = [
                value if isinstance(value, Tensor) else torch.tensor(value, device=self.device)
                for value in values
            ]
            reduced_metrics[key] = torch.stack(
                [value.to(torch.float32) for value in tensor_values]
            ).mean()

        self.log(
            "loss/validation/total", loss, on_step=False, on_epoch=True, sync_dist=True
        )
        self._log_metric_dict(
            "diagnostics/validation/",
            reduced_metrics,
            on_step=False,
            on_epoch=True,
        )
        self._log_validation_sequence_videos(
            batch=batch,
            summaries=summaries,
            batch_idx=batch_idx,
        )
        return {"loss": loss}

    def q_sample(self, x_start: Tensor, timesteps: Tensor, noise: Tensor) -> Tensor:
        return (
            extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape)
            * noise
        )

    def predict_start_from_noise(
        self, x_t: Tensor, timesteps: Tensor, noise: Tensor
    ) -> Tensor:
        return (
            extract(self.sqrt_recip_alphas_cumprod, timesteps, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, timesteps, x_t.shape) * noise
        )

    def predict_noise_from_start(
        self, x_t: Tensor, timesteps: Tensor, x0: Tensor
    ) -> Tensor:
        return (x_t - extract(self.sqrt_alphas_cumprod, timesteps, x_t.shape) * x0) / (
            extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_t.shape) + 1e-8
        )

    def predict_v(self, x_start: Tensor, timesteps: Tensor, noise: Tensor) -> Tensor:
        return (
            extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape) * noise
            - extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape)
            * x_start
        )

    def predict_start_from_v(self, x_t: Tensor, timesteps: Tensor, v: Tensor) -> Tensor:
        return (
            extract(self.sqrt_alphas_cumprod, timesteps, x_t.shape) * x_t
            - extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_t.shape) * v
        )

    def _model_predictions(
        self,
        rgb_src: Tensor,
        rgb_tgt: Tensor,
        noisy_action: Tensor,
        timesteps: Tensor,
    ):
        model_out = self.model(
            rgb_src=rgb_src,
            rgb_tgt=rgb_tgt,
            noisy_action=noisy_action,
            timesteps=timesteps,
        )
        objective = self.cfg.diffusion.objective
        if objective == "pred_noise":
            pred_noise = torch.clamp(
                model_out, -self.cfg.diffusion.clip_noise, self.cfg.diffusion.clip_noise
            )
            pred_x0 = self.predict_start_from_noise(noisy_action, timesteps, pred_noise)
        elif objective == "pred_x0":
            pred_x0 = model_out
            pred_noise = self.predict_noise_from_start(noisy_action, timesteps, pred_x0)
        elif objective == "pred_v":
            pred_x0 = self.predict_start_from_v(noisy_action, timesteps, model_out)
            pred_noise = self.predict_noise_from_start(noisy_action, timesteps, pred_x0)
        else:
            raise ValueError(f"Unsupported diffusion objective: {objective}")
        return model_out, pred_noise, pred_x0

    def _training_targets(
        self,
        x_start: Tensor,
        noisy_action: Tensor,
        timesteps: Tensor,
        noise: Tensor,
    ) -> Tensor:
        objective = self.cfg.diffusion.objective
        if objective == "pred_noise":
            return noise
        if objective == "pred_x0":
            return x_start
        if objective == "pred_v":
            return self.predict_v(x_start, timesteps, noise)
        raise ValueError(f"Unsupported diffusion objective: {objective}")

    def _denormalize_action(self, du_model: Tensor) -> Tensor:
        method = self.cfg.action_norm_method
        if method is not None:
            action_meta = get_action_normalization_metadata(self.dataset_metadata)
            action_dim = du_model.shape[-1]
            if method == "std":
                mean = self._select_action_metadata_channels(
                    action_meta.get("action_mean"),
                    expected_dim=action_dim,
                    device=du_model.device,
                    dtype=du_model.dtype,
                )
                std = self._select_action_metadata_channels(
                    action_meta.get("action_std"),
                    expected_dim=action_dim,
                    device=du_model.device,
                    dtype=du_model.dtype,
                )
                if mean is None or std is None:
                    return du_model
                view_shape = (1,) * (du_model.ndim - 1) + (action_dim,)
                mean = mean.view(view_shape)
                std = std.clamp(min=1e-6).view(view_shape)
                return du_model * std + mean
            if method == "quantile":
                low = self._select_action_metadata_channels(
                    action_meta.get("action_min"),
                    expected_dim=action_dim,
                    device=du_model.device,
                    dtype=du_model.dtype,
                )
                high = self._select_action_metadata_channels(
                    action_meta.get("action_max"),
                    expected_dim=action_dim,
                    device=du_model.device,
                    dtype=du_model.dtype,
                )
                if low is None or high is None:
                    return du_model
                center = (high + low) * 0.5
                scale = (high - low).abs().clamp(min=1e-6) * 0.5
                view_shape = (1,) * (du_model.ndim - 1) + (action_dim,)
                center = center.view(view_shape)
                scale = scale.view(view_shape)
                return du_model * scale + center
            raise ValueError(
                f"Unsupported action_norm_method={method}. Use one of: null, std, quantile."
            )

        action_meta = get_action_normalization_metadata(self.dataset_metadata)
        return denormalize_action(
            du_model,
            du_scale=float(action_meta.get("du_scale", 1.0)),
            action_mean=action_meta.get("action_mean"),
            action_std=action_meta.get("action_std"),
            action_min=action_meta.get("action_min"),
            action_max=action_meta.get("action_max"),
            action_abs_scale=action_meta.get("action_abs_scale"),
        )

    def _denormalize_action_chunk(self, du_model: Tensor) -> Tensor:
        return self._denormalize_action(du_model)

    def reconstruct_action_chunk(
        self, action_chunk: Tensor, *, fill_value: float = 0.0
    ) -> Tensor:
        inverse_channels = self.cfg.inverse_used_action_channel_ids
        if inverse_channels is None:
            return action_chunk
        if len(inverse_channels) != action_chunk.shape[-1]:
            raise ValueError(
                "inverse_used_action_channel_ids length must match action chunk channels "
                f"({len(inverse_channels)} != {action_chunk.shape[-1]})."
            )
        if len(inverse_channels) == 0:
            raise ValueError("inverse_used_action_channel_ids must not be empty.")
        if len(set(inverse_channels)) != len(inverse_channels):
            raise ValueError(
                "inverse_used_action_channel_ids must contain unique indices."
            )
        if min(inverse_channels) < 0:
            raise ValueError("inverse_used_action_channel_ids must be non-negative.")

        output_dim = int(max(inverse_channels)) + 1
        full_chunk = torch.full(
            (*action_chunk.shape[:-1], output_dim),
            fill_value=fill_value,
            device=action_chunk.device,
            dtype=action_chunk.dtype,
        )
        index = torch.tensor(
            inverse_channels,
            device=action_chunk.device,
            dtype=torch.long,
        )
        full_chunk.index_copy_(2, index, action_chunk)
        return full_chunk

    def _log_metric_dict(
        self,
        prefix: str,
        metrics: dict[str, Tensor | float | int],
        *,
        on_step: bool,
        on_epoch: bool,
    ) -> None:
        for key, value in metrics.items():
            if isinstance(value, Tensor):
                if value.numel() != 1:
                    continue
                value = value.detach()
                if not torch.isfinite(value):
                    continue
            elif isinstance(value, float) and not math.isfinite(value):
                continue
            self.log(
                f"{prefix}{key}",
                value,
                on_step=on_step,
                on_epoch=on_epoch,
                sync_dist=True,
            )

    def _log_sanity_metric_dict(self, prefix: str, metrics: dict[str, float | int]) -> None:
        self._log_metric_dict(prefix, metrics, on_step=True, on_epoch=False)

    def _build_action_diagnostics(
        self,
        *,
        rgb_src: Tensor,
        rgb_tgt: Tensor,
        target_action: Tensor,
        timesteps: Tensor,
        pred_x0: Tensor,
        model_out: Tensor,
        target: Tensor,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        with torch.no_grad():
            true_chunk = self._unflatten_action_chunk(target_action.detach())
            true_chunk_denorm = self._denormalize_action_chunk(true_chunk)

            random_pred_chunk = self._unflatten_action_chunk(pred_x0.detach())
            random_pred_chunk_safe = torch.nan_to_num(
                random_pred_chunk.to(torch.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).to(random_pred_chunk.dtype)

            eval_timesteps = torch.zeros_like(timesteps)
            eval_noisy_action = self.q_sample(
                target_action.detach(),
                eval_timesteps,
                torch.zeros_like(target_action),
            )
            _, _, eval_pred_x0 = self._model_predictions(
                rgb_src,
                rgb_tgt,
                eval_noisy_action,
                eval_timesteps,
            )
            eval_pred_chunk = self._unflatten_action_chunk(eval_pred_x0.detach())
            eval_pred_chunk_safe = torch.nan_to_num(
                eval_pred_chunk.to(torch.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).to(eval_pred_chunk.dtype)
            eval_pred_chunk_denorm = self._denormalize_action_chunk(eval_pred_chunk_safe)

            diagnostics = {
                "loss": F.mse_loss(model_out.detach(), target.detach()),
                "action_mse": F.mse_loss(eval_pred_chunk_denorm, true_chunk_denorm),
                "pred_action_norm": eval_pred_chunk_denorm.norm(dim=-1).mean(),
                "target_action_norm": true_chunk_denorm.norm(dim=-1).mean(),
                "pred_action_norm_normalized": eval_pred_chunk_safe.norm(dim=-1).mean(),
                "target_action_norm_normalized": true_chunk.norm(dim=-1).mean(),
                "pred_action_finite_ratio": torch.isfinite(eval_pred_chunk).float().mean(),
                "pred_action_finite_ratio_random_t": torch.isfinite(random_pred_chunk)
                .float()
                .mean(),
                "sampled_t_mean": timesteps.float().mean(),
                "sampled_t_max": timesteps.float().max(),
            }
            sanity_tensors = {
                "pred_action_t0_normalized": eval_pred_chunk,
                "pred_action_t0_denormalized": eval_pred_chunk_denorm,
                "pred_action_random_t_normalized": random_pred_chunk,
                "target_action_normalized": true_chunk,
                "target_action_denormalized": true_chunk_denorm,
            }
        return diagnostics, sanity_tensors

    def compute_batch(
        self, batch: dict[str, Tensor]
    ) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor]]:
        rgb_src, rgb_tgt = self._prepare_conditioning(batch)
        target_action_chunk = self._select_target_action_chunk(batch)
        target_action_chunk = self._normalize_action_chunk(target_action_chunk)
        target_action = self._flatten_action_chunk(target_action_chunk)

        batch_size = target_action.shape[0]
        timesteps = torch.randint(
            low=0,
            high=self.cfg.diffusion.timesteps,
            size=(batch_size,),
            device=self.device,
        )
        noise = torch.randn_like(target_action)
        noise = torch.clamp(
            noise, -self.cfg.diffusion.clip_noise, self.cfg.diffusion.clip_noise
        )
        noisy_action = self.q_sample(target_action, timesteps, noise)
        model_out, _, pred_x0 = self._model_predictions(
            rgb_src, rgb_tgt, noisy_action, timesteps
        )
        target = self._training_targets(target_action, noisy_action, timesteps, noise)
        loss = F.mse_loss(model_out, target)

        metrics, sanity_tensors = self._build_action_diagnostics(
            rgb_src=rgb_src,
            rgb_tgt=rgb_tgt,
            target_action=target_action,
            timesteps=timesteps,
            pred_x0=pred_x0,
            model_out=model_out,
            target=target,
        )
        return loss, metrics, sanity_tensors

    def training_step(self, batch, batch_idx):
        loss, metrics, sanity_tensors = self.compute_batch(batch)
        self.log(
            "loss/training/total", loss, on_step=True, on_epoch=True, sync_dist=True
        )
        should_log = batch_idx % self.cfg.logging.loss_freq == 0
        if should_log:
            self._log_metric_dict(
                "diagnostics/training/",
                metrics,
                on_step=True,
                on_epoch=False,
            )
            self._log_sanity_metric_dict(
                "sanity/training/",
                get_sanity_metrics(sanity_tensors),
            )
        return {"loss": loss}

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        if self._is_validation_sequence_batch(batch):
            return self._validation_sequence_step(batch, batch_idx)
        loss, metrics, _ = self.compute_batch(batch)
        self.log(
            "loss/validation/total", loss, on_step=False, on_epoch=True, sync_dist=True
        )
        self._log_metric_dict(
            "diagnostics/validation/",
            metrics,
            on_step=False,
            on_epoch=True,
        )
        return {"loss": loss}

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        loss, metrics, _ = self.compute_batch(batch)
        self.log("loss/test/total", loss, on_step=False, on_epoch=True, sync_dist=True)
        self._log_metric_dict(
            "diagnostics/test/",
            metrics,
            on_step=False,
            on_epoch=True,
        )
        return {"loss": loss}

    @torch.no_grad()
    def sample_action(
        self,
        rgb_src: Tensor,
        rgb_tgt: Tensor | None = None,
        *,
        sampling_timesteps: int | None = None,
        eta: float | None = None,
        return_normalized: bool = False,
        expand_full_action: bool = False,
    ) -> Tensor:
        if rgb_tgt is None:
            rgb_src, rgb_tgt = self._split_rgb_window(rgb_src)
        else:
            rgb_src = self._select_rgb_context(rgb_src)
            rgb_tgt = self._select_rgb_context(rgb_tgt)

        rgb_src = rgb_src.to(self.device)
        rgb_tgt = rgb_tgt.to(self.device)
        batch_size = rgb_src.shape[0]
        action = torch.randn(
            batch_size,
            self.cfg.model.action_dim * self.cfg.model.action_horizon,
            device=self.device,
            dtype=rgb_src.dtype,
        )
        diffusion_cfg = self.cfg.diffusion
        sample_steps = int(
            sampling_timesteps
            if sampling_timesteps is not None
            else diffusion_cfg.sampling_timesteps
        )
        sample_steps = max(1, min(sample_steps, diffusion_cfg.timesteps))
        ddim_eta = diffusion_cfg.ddim_sampling_eta if eta is None else float(eta)

        step_values = torch.linspace(
            diffusion_cfg.timesteps - 1,
            -1,
            steps=sample_steps + 1,
            device=self.device,
        ).long()
        for idx in range(sample_steps):
            t = torch.full(
                (batch_size,),
                int(step_values[idx].item()),
                device=self.device,
                dtype=torch.long,
            )
            t_next = int(step_values[idx + 1].item())
            _, pred_noise, pred_x0 = self._model_predictions(
                rgb_src, rgb_tgt, action, t
            )
            if t_next < 0:
                action = pred_x0
                continue

            alpha = self.alphas_cumprod[t].view(batch_size, 1)
            alpha_next = self.alphas_cumprod[torch.full_like(t, t_next)].view(
                batch_size, 1
            )
            sigma = ddim_eta * torch.sqrt(
                ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).clamp(
                    min=0.0
                )
            )
            c = torch.sqrt((1 - alpha_next - sigma**2).clamp(min=0.0))
            noise = torch.randn_like(action)
            noise = torch.clamp(
                noise, -diffusion_cfg.clip_noise, diffusion_cfg.clip_noise
            )
            action = pred_x0 * torch.sqrt(alpha_next) + pred_noise * c + sigma * noise

        action = self._unflatten_action_chunk(action)
        if return_normalized:
            if expand_full_action:
                return self.reconstruct_action_chunk(action)
            return action
        denorm_action = self._denormalize_action_chunk(action)
        if expand_full_action:
            return self.reconstruct_action_chunk(denorm_action)
        return denorm_action
