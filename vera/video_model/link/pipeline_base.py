"""
link/pipeline_base.py

Abstract base class and I/O contracts for video generation pipelines.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class VideoCondition:
    """
    Conditioning inputs for video generation.

    context_frames: [B, T_ctx, C, H, W] float tensor in [-1, 1].
        For T2V models pass None or a short seed clip used as history.
        For I2V models the first frame is used as image conditioning.

    text: prompt string shared across the batch, or a list of per-sample strings.
        Pass None to use a zero/empty embedding (model-dependent).
    """

    context_frames: Optional[torch.Tensor] = None
    text: Optional[str | list[str]] = None


@dataclass
class GenerationConfig:
    """
    Controls what is returned and optional overrides for generation.

    decode_outputs: any subset of {"rgb", "flow", "flow_rgb", "latents"}.
        "rgb"      → [B, T, C, H, W] in [-1, 1]
        "flow"     → [B, T, 2, H, W] raw optical flow (model units)
        "flow_rgb" → [B, T, 3, H, W] flow visualized as RGB in [-1, 1]
        "latents"  → [B, C, T_lat, H_lat, W_lat] raw VAE latents
    """

    decode_outputs: list[str] = field(default_factory=lambda: ["rgb"])


class BaseVideoPipeline(ABC):
    """
    Abstract inference interface for video generation models.

    Subclasses must implement:
        generate(condition, config) -> dict[str, Tensor]

    The returned dict always has keys matching config.decode_outputs.
    Values are float tensors on CPU unless the subclass documents otherwise.
    """

    @abstractmethod
    def generate(
        self,
        condition: VideoCondition,
        config: GenerationConfig | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Generate video frames given conditioning inputs.

        Args:
            condition: context frames and/or text prompt.
            config: what to decode and optional generation overrides.
                    Defaults to GenerationConfig() (rgb only) when None.

        Returns: future generated frames, without the context frames.
            dict mapping each requested output type to a tensor:
                "rgb"      [B, T, C, H, W]  in [-1, 1]
                "flow"     [B, T, 2, H, W]
                "flow_rgb" [B, T, 3, H, W]  in [-1, 1]
                "latents"  [B, C, T_lat, H_lat, W_lat]
        """

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Device the model parameters live on."""

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype:
        """Compute dtype (e.g. torch.bfloat16)."""
