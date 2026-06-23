# okto/algorithms/registry.py
from vera.idm.common.base_pytorch_algo import (
    BasePytorchAlgo,
    BasePytorchAlgoCfg,
)
from vera.utils.registry import Registry

# Backward compatibility: old checkpoint configs may use "jacobian_vanilla".
ALGO_REGISTRY = Registry[BasePytorchAlgo, BasePytorchAlgoCfg](
    "algorithm",
    name_aliases={
        "jacobian_vanilla": "image_jacobian"
    },  # TODO: temporary alias for backward compatibility
)

register_algorithm = ALGO_REGISTRY.register
resolve_algorithm_cfg = ALGO_REGISTRY.resolve_cfg
resolve_algorithm_instance = ALGO_REGISTRY.build
list_algorithms = ALGO_REGISTRY.list

# from __future__ import annotations
# from typing import Any, Dict, List, Type, Union
# from omegaconf import DictConfig

# from vera.idm.common.base_pytorch_algo import (
#     BasePytorchAlgo,
#     BasePytorchAlgoCfg,
# )
# from vera.config.tools import get_typed_config

# # -----------------------------------------------------------------------------
# # Global registries
# # -----------------------------------------------------------------------------
# _ALGO_REGISTRY: Dict[str, Type[BasePytorchAlgo]] = {}
# _ALGO_CFG_REGISTRY: Dict[str, Type[BasePytorchAlgoCfg]] = {}


# # -----------------------------------------------------------------------------
# # Registration decorator (single source of truth)
# # -----------------------------------------------------------------------------
# def register_algorithm(
#     name: str,
#     *,
#     cfg_cls: Type[BasePytorchAlgoCfg] | None,
# ):
#     """
#     Register an algorithm and its config under a shared name.

#     Example:
#         @register_algorithm("dfot", cfg_cls=DFoTCfg)
#         class DFoTAlgo(BasePytorchAlgo):
#             ...
#     """

#     def decorator(cls: Type[BasePytorchAlgo]):
#         if name in _ALGO_REGISTRY:
#             raise KeyError(f"Algorithm '{name}' already registered")

#         _ALGO_REGISTRY[name] = cls
#         _ALGO_CFG_REGISTRY[name] = cfg_cls
#         return cls

#     return decorator


# # -----------------------------------------------------------------------------
# # Lookup helpers
# # -----------------------------------------------------------------------------
# def get_algorithm_class(name: str) -> Type[BasePytorchAlgo]:
#     if name not in _ALGO_REGISTRY:
#         raise KeyError(
#             f"Unknown algorithm '{name}'. "
#             f"Available: {sorted(_ALGO_REGISTRY.keys())}"
#         )
#     return _ALGO_REGISTRY[name]


# def get_algorithm_cfg_class(name: str) -> Type[BasePytorchAlgoCfg]:
#     if name not in _ALGO_CFG_REGISTRY:
#         raise KeyError(
#             f"Unknown algorithm config '{name}'. "
#             f"Available: {sorted(_ALGO_CFG_REGISTRY.keys())}"
#         )
#     return _ALGO_CFG_REGISTRY[name]


# # -----------------------------------------------------------------------------
# # Resolution helpers (Hydra-friendly)
# # -----------------------------------------------------------------------------


# def resolve_algorithm_cfg(
#     cfg: Union[Dict[str, Any], DictConfig, BasePytorchAlgoCfg],
# ):
#     if isinstance(cfg, BasePytorchAlgoCfg):
#         return cfg

#     name = cfg.get("name", None)
#     if name is None:
#         raise ValueError("Algorithm config must contain `name`")

#     cfg_cls = get_algorithm_cfg_class(name)

#     # ✅ IMPORTANT: raw DictConfig passthrough
#     if cfg_cls is None:
#         return cfg

#     return get_typed_config(cfg_cls, cfg)


# def resolve_algorithm_instance(cfg):
#     algo_cls = get_algorithm_class(cfg["name"] if isinstance(cfg, dict) else cfg.name)
#     return algo_cls(cfg)


# # def resolve_algorithm_cfg(
# #     cfg: Union[Dict[str, Any], DictConfig, BasePytorchAlgoCfg],
# # ) -> BasePytorchAlgoCfg:
# #     """
# #     Resolve a raw config (dict / DictConfig / dataclass) into a typed algo cfg.
# #     """
# #     if isinstance(cfg, BasePytorchAlgoCfg):
# #         return cfg

# #     if isinstance(cfg, DictConfig):
# #         cfg = dict(cfg)

# #     name = cfg.get("name", None)
# #     if name is None:
# #         raise ValueError("Algorithm config must contain `name`")

# #     cfg_cls = get_algorithm_cfg_class(name)
# #     return get_typed_config(cfg_cls, cfg)


# # def resolve_algorithm_instance(cfg: BasePytorchAlgoCfg) -> BasePytorchAlgo:
# #     """
# #     Instantiate an algorithm from a resolved config.
# #     """
# #     algo_cls = get_algorithm_class(cfg.name)
# #     return algo_cls(cfg)


# # -----------------------------------------------------------------------------
# # Introspection
# # -----------------------------------------------------------------------------
# def list_algorithms(verbose: bool = True) -> List[str]:
#     names = sorted(set(_ALGO_REGISTRY.keys()) | set(_ALGO_CFG_REGISTRY.keys()))

#     if verbose:
#         print("🤖 Registered algorithms:")
#         for name in names:
#             print(
#                 f"  - {name:<18} "
#                 f"(Algo: {'✅' if name in _ALGO_REGISTRY else '❌'}, "
#                 f"Cfg: {'✅' if name in _ALGO_CFG_REGISTRY else '❌'})"
#             )
#         print()

#     return names
