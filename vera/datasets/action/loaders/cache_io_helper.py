"""No-op IO-cache shims.

okto's cache_io_helper sized video/zarr/npz/trajectory LRU caches; the unified
core does its own IO, so configuring/clearing them is inert. Signatures match
okto so data_module calls are valid.
"""
from __future__ import annotations


def configure_io_cache_settings(
    *,
    video_cache_size: int | None = None,
    zarr_cache_size: int | None = None,
    npz_cache_size: int | None = None,
    trajectory_cache_size: int | None = None,
) -> None:
    """No-op."""
    return None


def clear_all_caches() -> None:
    """No-op."""
    return None


__all__ = ["configure_io_cache_settings", "clear_all_caches"]
