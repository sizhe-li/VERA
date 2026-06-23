# robotflow/core/geom/math_utils.py
from __future__ import annotations

import numpy as np
import torch


def normalize_range(values, old_min, old_max, new_min=0.0, new_max=1.0):
    """Rescale values linearly from [old_min, old_max] → [new_min, new_max].

    Supports numpy arrays, torch tensors, or floats.
    """
    old_range = max(old_max - old_min, 1e-8)
    new_range = new_max - new_min

    if isinstance(values, torch.Tensor):
        return ((values - old_min) / old_range) * new_range + new_min
    elif isinstance(values, np.ndarray):
        return ((values - old_min) / old_range) * new_range + new_min
    else:
        return (values - old_min) / old_range * new_range + new_min
