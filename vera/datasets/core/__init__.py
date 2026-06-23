"""vera.datasets.core — the shared, consumer-agnostic dataset primitives.

Load once, sample once, then adapt layout:

    sources.py       -> resolve episode paths per source (DROID/allegro_sim/mimicgen/pusht)
    packed.py        -> packed-NPZ codec (qint8/zstd) [reused from okto loaders]
    frame_sampler.py -> ONE fps-aware temporal sampler                       [implemented]
    view_loader.py   -> load every view separately -> [T, V, C, H, W]
    layout.py        -> view-layout adapter: "separate" (IDM) | "tiled" (WM) [implemented]

``frame_sampler`` and ``layout`` are fully implemented + unit-tested (pure logic, no data). The
data-backed pieces (``view_loader``, ``sources``, ``packed``) carry the contract + DROID skeleton in
Phase 0; the decord/packed decoding is ported in Phase 3.
"""

from vera.datasets.core.frame_sampler import (
    FrameSamplerConfig,
    sample_frame_indices,
)
from vera.datasets.core.layout import (
    SEPARATE,
    TILED,
    apply_layout,
    pad_views,
    to_separate,
    to_tiled,
)

__all__ = [
    "FrameSamplerConfig",
    "sample_frame_indices",
    "SEPARATE",
    "TILED",
    "apply_layout",
    "pad_views",
    "to_separate",
    "to_tiled",
]
