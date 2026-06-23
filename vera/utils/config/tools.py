from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Type, TypeVar, Union

from dacite import Config, from_dict
from omegaconf import DictConfig, OmegaConf

TYPE_HOOKS = {
    Path: Path,
}


T = TypeVar("T")


def get_typed_config(
    data_class: Type[T],
    cfg: Union[DictConfig, dict[str, Any]],
    extra_type_hooks: dict = {},
) -> T:
    """
    Convert an OmegaConf DictConfig or plain dict into a strongly typed dataclass.

    Supports auto-conversion of common field types like Path and tuple.
    """
    # Convert dict -> DictConfig if necessary
    if isinstance(cfg, dict):
        cfg = OmegaConf.create(cfg)

    # Convert to plain dict
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Base type hooks
    base_hooks = {
        Path: Path,  # convert str -> Path
        tuple: lambda v: tuple(v) if isinstance(v, list) else v,
    }

    # Merge hooks (user-specified overrides take precedence)
    hooks = {**base_hooks, **extra_type_hooks}

    return from_dict(
        data_class=data_class, data=cfg_dict, config=Config(type_hooks=hooks)
    )


def separate_multiple_defaults(data_class_union):
    """Return a function that will pull individual configurations out of a merged dict.
    For example, the merged dict might look like this:

    {
        a: ...
        b: ...
    }

    The returned function will generate this:

    [{ name: a, ... }, { name: b, ... }]

    In other words, this function makes the types for default lists with single and
    multiple items be typed identically.
    """

    def separate_fn(joined: dict) -> list:
        # The dummy allows the union to be converted.
        @dataclass
        class Dummy:
            dummy: data_class_union

        return [
            get_typed_config(Dummy, DictConfig({"dummy": {"name": name, **cfg}})).dummy
            for name, cfg in joined.items()
        ]

    return separate_fn
