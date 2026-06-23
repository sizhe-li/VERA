"""Jacobian-IDM dataset — separate-view output (replaces okto datasets/action/*).

Thin wrapper: forces ``layout="separate"`` so ``rgb`` is ``[T, V, C, H, W]`` (or ``[T, C, H, W]``
when single-view), which is what the Jacobian transformer ingests (views as their own dimension).
"""

from __future__ import annotations

from dataclasses import replace

from vera.datasets.base import DatasetConfig, UnifiedDataset
from vera.datasets.core.layout import SEPARATE
from vera.datasets.core.sources import Source
from vera.datasets.core.view_loader import ViewLoader


class IDMDataset(UnifiedDataset):
    def __init__(self, source: Source, view_loader: ViewLoader, cfg: DatasetConfig, seed: int = 0):
        super().__init__(source, view_loader, replace(cfg, layout=SEPARATE), seed=seed)
