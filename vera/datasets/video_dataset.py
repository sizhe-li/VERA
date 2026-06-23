"""Video-model dataset — width-tiled multiview output (replaces flow-planner droid_flow/mixture).

Thin wrapper: forces ``layout="tiled"`` so ``rgb`` is ``[T, C, H, W*V]`` (the WAN canvas).
"""

from __future__ import annotations

from dataclasses import replace

from vera.datasets.base import DatasetConfig, UnifiedDataset
from vera.datasets.core.layout import TILED
from vera.datasets.core.sources import Source
from vera.datasets.core.view_loader import ViewLoader


class VideoModelDataset(UnifiedDataset):
    def __init__(self, source: Source, view_loader: ViewLoader, cfg: DatasetConfig, seed: int = 0):
        super().__init__(source, view_loader, replace(cfg, layout=TILED), seed=seed)
