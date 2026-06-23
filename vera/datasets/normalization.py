"""
Action and flow normalization utilities.

Config schema (dataset YAML):
  action_mean: [float, ...]   # per-dim mean for z-score
  action_std: [float, ...]    # per-dim std for z-score
  action_min: [float, ...]    # per-dim min for min-max normalization
  action_max: [float, ...]    # per-dim max for min-max normalization
  action_abs_scale: [float, ...]  # per-dim symmetric zero-preserving scale
  du_scale: float             # optional pre-scale before action normalization
  oflow_std: [float, ...]     # per-channel std (scale only, no shift)
  oflow_scale: float          # scalar: flow_normalized = flow_raw * oflow_scale
                              # when oflow_std present: oflow_scale = 1/mean(oflow_std)
                              # can override explicitly (e.g. 0.05 for RAFT)
  flow_normalization_mode: "scale" | "percentile_minmax" | "symmetric_percentile"
  oflow_percentile_min/max: [float, ...]  # per-channel percentile bounds for [-1, 1]
  oflow_abs_scale: [float, ...]  # per-channel symmetric zero-preserving scale
"""

from __future__ import annotations

from typing import Any, List, Mapping, Sequence

import torch
from torch import Tensor


def resolve_oflow_scale(
    oflow_scale: float | None,
    oflow_std: List[float] | None,
) -> float:
    """
    Resolve the scalar flow scale from config.

    Priority:
      1. Explicit oflow_scale (e.g. 0.05 for RAFT outputs)
      2. Derived from oflow_std: 1 / mean(oflow_std) for scale-only normalization
      3. Default 1.0
    """
    if oflow_scale is not None:
        return float(oflow_scale)
    if oflow_std is not None and len(oflow_std) > 0:
        avg_std = sum(oflow_std) / len(oflow_std)
        return 1.0 / (avg_std + 1e-8) if avg_std > 0 else 1.0
    return 1.0


def resolve_effective_oflow_scale(
    oflow_scale: float | None,
    oflow_std: List[float] | None,
    flow_scale_factor: float = 1.0,
) -> float:
    """Resolve the total flow scale applied by the dataset."""
    return resolve_oflow_scale(oflow_scale, oflow_std) * float(flow_scale_factor)


def resolve_action_normalization_mode(
    action_mean: Sequence[float] | None,
    action_std: Sequence[float] | None,
    action_min: Sequence[float] | None,
    action_max: Sequence[float] | None,
    action_abs_scale: Sequence[float] | None = None,
) -> str:
    """Infer action normalization mode from config or metadata."""
    if action_abs_scale is not None:
        return "symmetric_percentile"
    if action_mean is not None and action_std is not None:
        return "zscore"
    if action_min is not None and action_max is not None:
        return "minmax"
    return "none"


def compute_jacobian_action_scales(
    *,
    action_dim: int,
    du_scale: float = 1.0,
    action_mean: Sequence[float] | None = None,
    action_std: Sequence[float] | None = None,
    action_min: Sequence[float] | None = None,
    action_max: Sequence[float] | None = None,
    action_abs_scale: Sequence[float] | None = None,
    eps: float = 1e-8,
) -> list[float]:
    """
    Per-dimension scale converting J_model = d(flow_scaled)/d(du_model) into
    d(flow_scaled)/d(du_physical).

    `du_scale` is treated as a pre-scale applied before any learned normalization,
    so it contributes multiplicatively in every mode.
    """
    base_scale = float(du_scale)
    if action_abs_scale is not None and len(action_abs_scale) == action_dim:
        return [base_scale * 1.0 / (s + eps) for s in action_abs_scale]
    if (
        action_min is not None
        and action_max is not None
        and len(action_min) == action_dim
        and len(action_max) == action_dim
    ):
        return [
            base_scale * 2.0 / (mx - mn + eps)
            for mn, mx in zip(action_min, action_max)
        ]
    if action_std is not None and len(action_std) == action_dim:
        return [base_scale * 1.0 / (s + eps) for s in action_std]
    return [base_scale] * action_dim


def denormalize_action(
    du_model: Tensor,
    *,
    du_scale: float = 1.0,
    action_mean: Sequence[float] | None = None,
    action_std: Sequence[float] | None = None,
    action_min: Sequence[float] | None = None,
    action_max: Sequence[float] | None = None,
    action_abs_scale: Sequence[float] | None = None,
    eps: float = 1e-8,
) -> Tensor:
    """Convert model-space action back to physical action units."""
    du = du_model
    if action_abs_scale is not None:
        scale = torch.as_tensor(action_abs_scale, device=du.device, dtype=du.dtype)
        du = du * (scale + eps)
    elif action_mean is not None and action_std is not None:
        mean = torch.as_tensor(action_mean, device=du.device, dtype=du.dtype)
        std = torch.as_tensor(action_std, device=du.device, dtype=du.dtype)
        du = du * (std + eps) + mean
    elif action_min is not None and action_max is not None:
        amin = torch.as_tensor(action_min, device=du.device, dtype=du.dtype)
        amax = torch.as_tensor(action_max, device=du.device, dtype=du.dtype)
        du = (du + 1.0) * 0.5
        du = du * (amax - amin + eps) + amin
    return du / float(du_scale)


def get_oflow_scale_from_metadata(metadata: Mapping[str, Any] | None) -> float:
    if not isinstance(metadata, Mapping):
        return 1.0
    return float(metadata.get("oflow_scale", 1.0))


def get_flow_normalization_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        return {
            "flow_normalization_mode": "scale",
            "oflow_scale": 1.0,
            "oflow_percentile_min": None,
            "oflow_percentile_max": None,
            "oflow_abs_scale": None,
        }
    flow_mode = metadata.get("flow_normalization_mode")
    if (
        flow_mode is None
        and metadata.get("oflow_abs_scale") is not None
    ):
        flow_mode = "symmetric_percentile"
    if (
        flow_mode is None
        and metadata.get("oflow_percentile_min") is not None
        and metadata.get("oflow_percentile_max") is not None
    ):
        flow_mode = "percentile_minmax"
    if flow_mode is None:
        flow_mode = "scale"
    return {
        "flow_normalization_mode": flow_mode,
        "oflow_scale": get_oflow_scale_from_metadata(metadata),
        "oflow_percentile_min": metadata.get("oflow_percentile_min"),
        "oflow_percentile_max": metadata.get("oflow_percentile_max"),
        "oflow_abs_scale": metadata.get("oflow_abs_scale"),
    }


def _flow_channel_stat_tensor(values: Sequence[float], reference: Tensor) -> Tensor:
    stats = torch.as_tensor(values, device=reference.device, dtype=reference.dtype)
    shape = [1] * reference.ndim
    shape[-3] = stats.numel()
    return stats.view(*shape)


def get_action_normalization_metadata(
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        return {
            "du_scale": 1.0,
            "action_mean": None,
            "action_std": None,
            "action_min": None,
            "action_max": None,
            "action_abs_scale": None,
            "action_normalization_mode": "none",
        }

    action_mean = metadata.get("action_mean")
    action_std = metadata.get("action_std")
    action_min = metadata.get("action_min")
    action_max = metadata.get("action_max")
    action_abs_scale = metadata.get("action_abs_scale")
    return {
        "du_scale": float(metadata.get("du_scale", metadata.get("action_pre_scale", 1.0))),
        "action_mean": action_mean,
        "action_std": action_std,
        "action_min": action_min,
        "action_max": action_max,
        "action_abs_scale": action_abs_scale,
        "action_normalization_mode": metadata.get(
            "action_normalization_mode",
            resolve_action_normalization_mode(
                action_mean,
                action_std,
                action_min,
                action_max,
                action_abs_scale,
            ),
        ),
    }


def get_jacobian_action_scale_tensor(
    metadata: Mapping[str, Any] | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
    cmd_dim: int,
) -> Tensor:
    action_scales = None
    if isinstance(metadata, Mapping):
        action_scales = metadata.get("jacobian_action_scales")

    action_scale_tensor = torch.as_tensor(
        action_scales if action_scales is not None else [1.0] * cmd_dim,
        device=device,
        dtype=dtype,
    )
    if action_scale_tensor.ndim == 0:
        action_scale_tensor = action_scale_tensor.expand(cmd_dim)
    if action_scale_tensor.numel() < cmd_dim:
        pad_val = (
            action_scale_tensor[-1].item()
            if action_scale_tensor.numel() > 0
            else 1.0
        )
        action_scale_tensor = torch.nn.functional.pad(
            action_scale_tensor,
            (0, cmd_dim - action_scale_tensor.numel()),
            value=pad_val,
        )
    elif action_scale_tensor.numel() > cmd_dim:
        action_scale_tensor = action_scale_tensor[:cmd_dim]
    return action_scale_tensor.view(1, cmd_dim, 1, 1, 1)


def denormalize_flow_tensor(
    flow_scaled: Tensor,
    metadata: Mapping[str, Any] | None = None,
) -> Tensor:
    """Convert scaled flow to physical flow units."""
    flow_meta = get_flow_normalization_metadata(metadata)
    if (
        flow_meta["flow_normalization_mode"] == "symmetric_percentile"
        and flow_meta["oflow_abs_scale"] is not None
    ):
        flow_scale = _flow_channel_stat_tensor(
            flow_meta["oflow_abs_scale"], flow_scaled
        )
        return flow_scaled * (flow_scale + 1e-8)
    if (
        flow_meta["flow_normalization_mode"] == "percentile_minmax"
        and flow_meta["oflow_percentile_min"] is not None
        and flow_meta["oflow_percentile_max"] is not None
    ):
        flow_min = _flow_channel_stat_tensor(
            flow_meta["oflow_percentile_min"], flow_scaled
        )
        flow_max = _flow_channel_stat_tensor(
            flow_meta["oflow_percentile_max"], flow_scaled
        )
        flow_phys = (flow_scaled + 1.0) * 0.5
        return flow_phys * (flow_max - flow_min + 1e-8) + flow_min
    return flow_scaled / flow_meta["oflow_scale"]


def denormalize_jacobian_tensor(
    jacobian_scaled: Tensor,
    metadata: Mapping[str, Any] | None,
    *,
    cmd_dim: int,
) -> Tensor:
    """Convert scaled Jacobian to physical flow per physical action units."""
    action_scale = get_jacobian_action_scale_tensor(
        metadata,
        device=jacobian_scaled.device,
        dtype=jacobian_scaled.dtype,
        cmd_dim=cmd_dim,
    )
    flow_meta = get_flow_normalization_metadata(metadata)
    if (
        flow_meta["flow_normalization_mode"] == "symmetric_percentile"
        and flow_meta["oflow_abs_scale"] is not None
    ):
        flow_scale = _flow_channel_stat_tensor(
            flow_meta["oflow_abs_scale"], jacobian_scaled
        )
        return jacobian_scaled * action_scale * (flow_scale + 1e-8)
    if (
        flow_meta["flow_normalization_mode"] == "percentile_minmax"
        and flow_meta["oflow_percentile_min"] is not None
        and flow_meta["oflow_percentile_max"] is not None
    ):
        flow_min = _flow_channel_stat_tensor(
            flow_meta["oflow_percentile_min"], jacobian_scaled
        )
        flow_max = _flow_channel_stat_tensor(
            flow_meta["oflow_percentile_max"], jacobian_scaled
        )
        flow_denorm = (flow_max - flow_min + 1e-8) * 0.5
        return jacobian_scaled * action_scale * flow_denorm
    return jacobian_scaled * action_scale / flow_meta["oflow_scale"]
