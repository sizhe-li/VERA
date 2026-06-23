"""VERA — video-planner + Jacobian inverse-dynamics manipulation system.

Two-stage, decoupled:
  - ``vera.video_model``  : WAN video planner (embodiment-agnostic)        [Phase 1]
  - ``vera.idm`` / ``vera.policy`` : Jacobian inverse-dynamics + policy    [Phase 2]
  - ``vera.datasets``     : unified loader feeding BOTH stages             [Phase 0+]

This package is being assembled by copy-and-adapt from ``project/okto`` (IDM) and
``third_party/flow-planner`` (video planner); both originals remain runnable during the
migration. See ``ARCHITECTURE.md``.

The top-level ``__init__`` deliberately avoids importing the heavy stage subpackages so the
Phase-0 dataset-core scaffold imports without the full training stack installed. Import the
stage you need explicitly, e.g. ``from vera.datasets import build_dataset``.
"""

__version__ = "0.0.1"


def enable_legacy_ckpt_loading():
    """Make pre-rename ``okto.*`` checkpoints loadable in vera.

    Lightning ``.ckpt`` files (and any ``torch.save`` of objects) may pickle class
    references like ``okto.algorithms.…``; after the okto→vera rename those import
    paths are gone, so ``torch.load`` would raise ``ModuleNotFoundError: okto``.
    This registers ``sys.modules`` aliases so the pickle resolves to the vera modules.

    Opt-in (call before ``torch.load`` of a legacy checkpoint) rather than automatic,
    so it never shadows a genuinely-installed ``okto`` package in the same process.
    State_dict *tensors* don't need this (their keys are attribute paths, rename-safe);
    it's only for pickled class refs.
    """
    import sys
    import importlib

    mapping = {
        "okto": "vera",
        "okto.algorithms": "vera.idm",   # the one rename
        "okto.policy": "vera.policy",
        "okto.utils": "vera.utils",
        "okto.datasets": "vera.datasets",
        "okto.experiments": "vera.experiments",
        "okto.server": "vera.server",
        "okto.env_runner": "vera.env_runner",
    }
    for old, new in mapping.items():
        if old not in sys.modules:
            try:
                sys.modules[old] = importlib.import_module(new)
            except Exception:
                pass  # subpackage may have unmet optional deps; skip


__all__ = ["__version__", "enable_legacy_ckpt_loading"]
