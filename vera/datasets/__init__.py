"""vera.datasets — unified dataset package feeding BOTH stages.

One loader, two layouts:
  * ``IDMDataset``        -> ``[T, V, C, H, W]`` (Jacobian IDM)
  * ``VideoModelDataset`` -> ``[T, C, H, W*V]`` (WAN video model)

Both compose the same ``vera.datasets.core`` primitives (fps-aware ``frame_sampler`` + view-layout
``layout``), so their outputs are provably consistent (``tiled == cat(separate)``; see the parity test).
"""

from vera.datasets.base import DatasetConfig, UnifiedDataset
from vera.datasets.core import (
    SEPARATE,
    TILED,
    FrameSamplerConfig,
    apply_layout,
    sample_frame_indices,
)
from vera.datasets.idm_dataset import IDMDataset
from vera.datasets.video_dataset import VideoModelDataset

__all__ = [
    "DatasetConfig",
    "UnifiedDataset",
    "IDMDataset",
    "VideoModelDataset",
    "FrameSamplerConfig",
    "sample_frame_indices",
    "apply_layout",
    "SEPARATE",
    "TILED",
]
