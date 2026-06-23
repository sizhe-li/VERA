"""No-op cache shims (DROID h5 cache).

okto's action_loader maintained a DROID hdf5 read cache; the unified
vera.datasets.core loads via its own packed codec (core/packed.py) and does not
use that cache, so these are inert. Kept so experiments' data_modules, which call
them to configure/clear caches, import and run unchanged.
"""
from __future__ import annotations


def configure_droid_h5_cache(*, maxsize: int | None = None) -> None:
    """No-op: the unified core does not maintain a DROID h5 cache."""
    return None


def clear_droid_h5_cache() -> None:
    """No-op."""
    return None


__all__ = ["configure_droid_h5_cache", "clear_droid_h5_cache"]
