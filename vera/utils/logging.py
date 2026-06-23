from dataclasses import is_dataclass

import torch
from colorama import Fore


def color_text(text: str, color: str) -> str:
    return getattr(Fore, color.upper()) + text + Fore.RESET


def cyan(text):
    return color_text(text, "cyan")


def red(text):
    return color_text(text, "red")


def green(text):
    return color_text(text, "green")


def get_sanity_metrics(x: dict, prefix: str = "") -> dict:
    """Recursively collect tensor stats for sanity logging."""
    metrics = {}
    if isinstance(x, (list, tuple)):
        for i, item in enumerate(x):
            metrics.update(get_sanity_metrics(item, f"{prefix}{i}_"))
        return metrics

    if not isinstance(x, dict):
        return metrics

    for k, v in x.items():
        key = f"{prefix}{k}"
        # Recursive dict
        if isinstance(v, dict):
            metrics.update(get_sanity_metrics(v, f"{key}_"))
        # Tensor stats
        elif isinstance(v, torch.Tensor) and v.is_floating_point():
            v_det = v.detach()
            metrics[f"{key}_mean"] = float(v_det.mean().cpu())
            metrics[f"{key}_std"] = float(v_det.std().cpu())
            metrics[f"{key}_min"] = float(v_det.min().cpu())
            metrics[f"{key}_max"] = float(v_det.max().cpu())
            metrics[f"{key}_nan"] = int(torch.isnan(v_det).any())
            metrics[f"{key}_inf"] = int(torch.isinf(v_det).any())
    return metrics


def safe_asdict(obj):
    """A safe version of asdict that handles PyTorch tensors."""
    if is_dataclass(obj):
        result = {}
        for key, value in obj.__dict__.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.detach().cpu()
            else:
                result[key] = safe_asdict(value)
        return result
    elif isinstance(obj, dict):
        return {key: safe_asdict(value) for key, value in obj.items()}
    else:
        return obj
