# okto/models/registry.py
from __future__ import annotations

from vera.utils.registry import Registry

from .base import BaseModelCfg, JacobianFieldInterface

# -----------------------------------------------------------------------------
# Global registry
# -----------------------------------------------------------------------------
MODEL_REGISTRY = Registry[JacobianFieldInterface, BaseModelCfg]("model")

# -----------------------------------------------------------------------------
# Public API (stable surface)
# -----------------------------------------------------------------------------
register_model = MODEL_REGISTRY.register
resolve_model_cfg = MODEL_REGISTRY.resolve_cfg
resolve_model_instance = MODEL_REGISTRY.build
list_models = MODEL_REGISTRY.list
