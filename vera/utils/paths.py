"""Public-release path resolution.

vera's configs reference data/checkpoint roots via Hydra ``${oc.env:VERA_DATA_PREFIX,...}``
interpolation so a deployment overrides them with env vars instead of editing yaml. This
module exposes the same roots for *code* paths. Defaults preserve the original lab paths so
nothing breaks in-place; set the env vars for a fresh environment.
"""
from __future__ import annotations
import os

DATA_PREFIX_ENV = "VERA_DATA_PREFIX"
SCRATCH_PREFIX_ENV = "VERA_SCRATCH_PREFIX"


def data_prefix() -> str:
    """Root for shared datasets/checkpoints (override with $VERA_DATA_PREFIX)."""
    return os.environ.get(DATA_PREFIX_ENV, "/path/to/data")


def scratch_prefix() -> str:
    """Root for scratch data (override with $VERA_SCRATCH_PREFIX)."""
    return os.environ.get(SCRATCH_PREFIX_ENV, "/path/to")


def resolve(path: str) -> str:
    """Rewrite a hardcoded lab path onto the configured prefixes."""
    return path.replace("/path/to/data", data_prefix()).replace(
        "/path/to", scratch_prefix()
    )
