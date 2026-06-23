"""Weighted multi-source mixture over vera ``VideoModelDataset`` subsets.

Reduced, self-contained port of flow-planner ``datasets/mixture.py``: an
``IterableDataset`` that routes each draw to one subset by weight (absolute =
constant per-subset probability, relative = ×len(subset)), with conditional
outer-field propagation honoring ``propagate_skip`` (combined_4env skips ``width``
and ``pad_views_to`` so each subset keeps its native value). Each subset is a
``VideoModelDataset`` built by ``registry.build_dataset`` from its own cfg node, so
every sample already carries the WAN contract (``videos``/``prompts``/
``src_n_frames``/``has_bbox``/``bbox_render``/``task_class``). NO ``import okto`` /
``import flow_planner``.

Cf. flow-planner mixture.py lines 65-179. The OpenX/dummy/etc. subset classes are
intentionally dropped — vera's mixture only mixes the 4 robot embodiments built
through the unified core.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from omegaconf import OmegaConf

try:
    from torch.utils.data import Dataset, IterableDataset
except Exception:  # pragma: no cover
    Dataset = object  # type: ignore
    IterableDataset = object  # type: ignore


# Fields forwarded from the outer (mixture) cfg to each subset cfg, mirroring
# flow-planner mixture.py's ``propagated_fields`` (minus latent/embed-only ones we
# do not load). A field is forwarded only when (a) it is set (not None) on the outer
# cfg and (b) it is not in ``propagate_skip`` — so combined_4env keeps per-subset
# native ``width``/``pad_views_to`` while sharing ``n_frames``/``pad_to_width``.
_PROPAGATED_FIELDS = [
    "height", "n_frames", "fps", "image_to_video",
    "pad_views_to", "load_optical_flow", "pad_to_width", "width",
    "max_text_tokens", "id_token",
]


def _get(cfg: Any, key: str, default: Any = None) -> Any:
    try:
        val = cfg[key]
    except Exception:
        val = getattr(cfg, key, None)
    return default if val is None else val


class MixtureDataset(IterableDataset):
    """Fault-tolerant weighted mixture of ``VideoModelDataset`` subsets.

    The cfg layout matches flow-planner: subset cfgs live under ``subset/<name>``
    keys, ``training``/``validation`` carry ``weight_type`` + per-subset ``weight``,
    and ``propagate_skip`` lists outer fields NOT to forward.
    """

    def __init__(self, cfg: Any, split: str = "training"):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.debug = bool(_get(cfg, "debug", False))
        if split == "all":
            raise ValueError("split cannot be 'all' for MixtureDataset")

        # Lazy import to keep this module import-light + self-contained.
        from vera.datasets import registry

        # Collect subset cfg nodes keyed by name (strip the "subset/" prefix).
        items = cfg.items() if hasattr(cfg, "items") else dict(cfg).items()
        subset_cfg: Dict[str, Any] = {
            k.split("/", 1)[1]: v for k, v in items if str(k).startswith("subset/")
        }

        split_node = _get(cfg, split)
        weight = dict(_get(split_node, "weight"))
        for key in weight:
            if key not in subset_cfg:
                raise ValueError(
                    f"Dataset '{key}' in weights but not in configuration (subset/{key})"
                )
        subset_cfg = {k: v for k, v in subset_cfg.items() if k in weight}
        weight_type = str(_get(split_node, "weight_type", "absolute"))

        propagate_skip = set(_get(cfg, "propagate_skip", []) or [])
        self.subset_names: List[str] = list(subset_cfg.keys())
        self.subsets: List[Dataset] = []
        for name, scfg in subset_cfg.items():
            # Hydra composes cfgs in struct mode; propagation below ADDS outer fields
            # to each subset, so unlock the subset node first (else ConfigAttributeError
            # "Key ... not in struct"). Makes build_dataset(mixture_cfg) work directly,
            # not only via the WAN data-module's external set_struct workaround.
            if OmegaConf.is_config(scfg):
                OmegaConf.set_struct(scfg, False)
            # Conditional outer-field propagation (flow-planner mixture.py 103-108).
            for field in _PROPAGATED_FIELDS:
                if field in propagate_skip:
                    continue
                outer = _get(cfg, field)
                if outer is not None:
                    try:
                        scfg[field] = outer
                    except Exception:
                        setattr(scfg, field, outer)
            ds = registry.build_dataset(scfg, stage=split)
            self.subsets.append(ds)
            if weight_type == "relative":
                weight[name] = weight[name] * len(ds)

        total = sum(weight.values())
        self.normalized_weights = {k: v / total for k, v in weight.items()}

        # cumsum routing table (flow-planner mixture.py 139-143).
        self.cumsum_weights: Dict[str, float] = {}
        acc = 0.0
        for k in self.subset_names:
            acc += self.normalized_weights[k]
            self.cumsum_weights[k] = acc

    def __iter__(self):
        while True:
            rand = np.random.random()
            selected = self.subset_names[-1]
            for name in self.subset_names:
                if rand <= self.cumsum_weights[name]:
                    selected = name
                    break
            ds = self.subsets[self.subset_names.index(selected)]
            try:
                idx = np.random.randint(len(ds))
                yield ds[idx]
            except Exception as e:  # fault tolerant (flow-planner mixture.py 173-178)
                if self.debug:
                    raise
                print(f"[MixtureDataset] error sampling from {selected}: {e}")
                continue
