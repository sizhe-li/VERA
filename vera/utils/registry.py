# okto/utils/registry.py
from __future__ import annotations

from typing import Any, Dict, Generic, Type, TypeVar

from omegaconf import DictConfig

from vera.utils.config.tools import get_typed_config

TImpl = TypeVar("TImpl")
TCfg = TypeVar("TCfg")


class Registry(Generic[TImpl, TCfg]):
    def __init__(self, name: str, *, name_aliases: Dict[str, str] | None = None):
        self.name = name
        self._impl: Dict[str, Type[TImpl]] = {}
        self._cfg: Dict[str, Type[TCfg] | None] = {}
        self._name_aliases = name_aliases or {}

    def _resolve_name(self, name: str) -> str:
        return self._name_aliases.get(name, name)

    # --------------------------------------------------
    # Registration
    # --------------------------------------------------
    def register(
        self,
        name: str,
        *,
        cfg_cls: Type[TCfg] | None = None,
    ):
        def decorator(cls: Type[TImpl]):
            if name in self._impl:
                raise KeyError(f"{self.name} '{name}' already registered")
            self._impl[name] = cls
            self._cfg[name] = cfg_cls
            return cls

        return decorator

    # --------------------------------------------------
    # Lookup
    # --------------------------------------------------
    def get_impl(self, name: str) -> Type[TImpl]:
        name = self._resolve_name(name)
        if name not in self._impl:
            raise KeyError(
                f"Unknown {self.name} '{name}'. "
                f"Available: {sorted(self._impl.keys())}"
            )
        return self._impl[name]

    def get_cfg(self, name: str) -> Type[TCfg] | None:
        name = self._resolve_name(name)
        if name not in self._cfg:
            raise KeyError(
                f"Unknown {self.name} config '{name}'. "
                f"Available: {sorted(self._cfg.keys())}"
            )
        return self._cfg[name]

    # --------------------------------------------------
    # Resolution (Hydra-friendly)
    # --------------------------------------------------
    def resolve_cfg(self, cfg: Any):
        if cfg is None:
            return None

        # already typed
        if not isinstance(cfg, (dict, DictConfig)):
            return cfg

        name = cfg.get("name", None)
        if name is None:
            raise ValueError(f"{self.name} config must contain `name`")

        canonical_name = self._resolve_name(name)
        cfg_cls = self.get_cfg(canonical_name)

        # allow raw DictConfig passthrough
        if cfg_cls is None:
            return cfg

        # Resolve with canonical name so typed config has correct name
        if isinstance(cfg, DictConfig):
            from omegaconf import OmegaConf

            cfg = OmegaConf.to_container(cfg, resolve=True)
        cfg = dict(cfg)
        cfg["name"] = canonical_name
        return get_typed_config(cfg_cls, cfg)

    def build(self, cfg: Any):
        cfg = self.resolve_cfg(cfg)
        name = cfg["name"] if isinstance(cfg, dict) else cfg.name
        impl_cls = self.get_impl(name)
        return impl_cls(cfg)

    # --------------------------------------------------
    # Introspection
    # --------------------------------------------------
    def list(self) -> list[str]:
        return sorted(self._impl.keys())
