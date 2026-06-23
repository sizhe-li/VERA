"""Video-generation (WAN / OMNI) training experiment for vera.main.

This wires the already-ported WAN training algorithm
(``vera.video_model.algorithms.wan.{wan_t2v,wan_i2v}``) into the standard
``BaseLightningExperiment`` fit loop, so the OMNI/WAN video model can be TRAINED
from ``vera.main`` (not just served for inference).

Composition (hydra):
  experiment: video_generation
  algorithm:  wan_t2v | wan_i2v   (raw-DictConfig passthrough via the IDM registry)
  dataset:    combined_4env (MixtureDataset) or a single tiled subset
              (mimicgen / allegro_sim / allegro_real / droid_flow)

The WAN algorithm is a ``vera.video_model...BasePytorchAlgo`` (a LightningModule);
``trainer.fit(self.algo, datamodule=...)`` drives forward -> flow-matching MSE loss
-> backward -> optimizer step exactly like the flow-planner ``exp_video.py`` path.

ONLY this file + the video-generation configs are added for this wiring; the
datasets/core and the WAN algorithm itself are untouched.
"""

from __future__ import annotations

from typing import Set

import torch
from omegaconf import DictConfig

from .base_exp import BaseLightningExperiment
from .data_modules import BaseDataModule


# ---------------------------------------------------------------------------
# Register the WAN algorithms into the IDM algorithm registry that
# ``BaseExperiment._build_algo`` resolves through. We register with
# ``cfg_cls=None`` so the raw Hydra ``cfg.algorithm`` DictConfig is passed
# straight to ``Wan{TextToVideo,ImageToVideo}.__init__(cfg)`` (which reads it via
# attribute access). Guarded so re-import does not raise "already registered".
# ---------------------------------------------------------------------------
def _register_wan_algorithms() -> None:
    from vera.idm.registry import ALGO_REGISTRY, register_algorithm
    from vera.video_model.algorithms.wan.wan_i2v import WanImageToVideo
    from vera.video_model.algorithms.wan.wan_t2v import WanTextToVideo

    if "wan_t2v" not in ALGO_REGISTRY.list():
        register_algorithm("wan_t2v", cfg_cls=None)(WanTextToVideo)
    if "wan_i2v" not in ALGO_REGISTRY.list():
        register_algorithm("wan_i2v", cfg_cls=None)(WanImageToVideo)


def _patch_tuned_state_dict_prefix() -> None:
    """Make ``_load_tuned_state_dict`` robust to the OMNI checkpoint prefix.

    The exported OMNI ``video_model.ckpt`` stores WanModel weights under the
    Lightning double prefix ``model.model.`` (the algo's ``self.model`` adds one
    ``model.`` and Lightning wraps the LightningModule with another). The default
    ``WanTextToVideo._load_tuned_state_dict`` strips only ``model.``, leaving
    ``model.*`` keys that match nothing -> "all keys missing".

    The inference servers monkeypatch this same method to try ``model.model.``
    first; we apply the identical prefix-robust loader here so TRAINING from the
    exported OMNI checkpoint via vera.main loads the weights correctly. This is a
    thin wiring shim — the WAN algorithm source is not edited.
    """
    import gc
    import logging

    from vera.video_model.algorithms.wan.wan_t2v import (
        WanTextToVideo,
        _load_checkpoint_weights_only,
    )

    if getattr(WanTextToVideo, "_vera_prefix_robust_loader", False):
        return

    def _load_tuned_state_dict(self, prefix=None):
        ckpt = _load_checkpoint_weights_only(
            self.cfg.model.tuned_ckpt_path, mmap=True, map_location="cpu"
        )
        sd = ckpt["state_dict"]
        # Try the most-nested prefix first. ``model.model.`` is the Lightning
        # double prefix; ``model.`` is the single-prefix (older) layout.
        for try_prefix in ("model._orig_mod.", "model.model.", "model."):
            filtered = {
                k[len(try_prefix):]: v
                for k, v in sd.items()
                if k.startswith(try_prefix)
                # Avoid grabbing sibling submodules (vae./clip./text_encoder.)
                # when stripping the bare ``model.`` prefix.
                and not k[len(try_prefix):].split(".", 1)[0]
                in ("vae", "clip", "text_encoder", "vae_mean", "vae_inv_std")
            }
            if filtered:
                logging.info(
                    "[wan] loaded %d tuned weights via prefix '%s'",
                    len(filtered),
                    try_prefix,
                )
                del ckpt
                gc.collect()
                return filtered
        del ckpt
        gc.collect()
        raise RuntimeError(
            f"No usable model prefix in {self.cfg.model.tuned_ckpt_path}"
        )

    WanTextToVideo._load_tuned_state_dict = _load_tuned_state_dict
    WanTextToVideo._vera_prefix_robust_loader = True


_register_wan_algorithms()
_patch_tuned_state_dict_prefix()


class VideoMixtureDataModule(BaseDataModule):
    """Data module that knows how to build the cross-embodiment mixture.

    ``vera.datasets.registry.build_dataset`` routes single tiled subsets
    (mimicgen / allegro_* / droid_flow) to ``VideoModelDataset`` directly, but it
    has no route for the ``combined_4env`` mixture *name* (the mixture is its own
    ``IterableDataset`` composed of subset cfgs). We detect a mixture cfg here (it
    carries ``subset/<name>`` keys) and construct ``MixtureDataset`` directly,
    leaving every non-mixture cfg on the unchanged registry path.

    This lives in the experiment/data-module layer on purpose so the
    self-contained ``vera/datasets`` package is not modified.
    """

    @staticmethod
    def _is_mixture_cfg(cfg) -> bool:
        try:
            keys = list(cfg.keys())
        except Exception:
            return False
        return any(str(k).startswith("subset/") for k in keys)

    def _build_dataset_from_cfg_node(self, *, split, cache_key, dataset_cfg_node):
        if not self._is_mixture_cfg(dataset_cfg_node):
            return super()._build_dataset_from_cfg_node(
                split=split,
                cache_key=cache_key,
                dataset_cfg_node=dataset_cfg_node,
            )

        # MixtureDataset expects "training"/"validation" splits (not "test"/"all").
        mixture_split = "validation" if split in ("validation", "test") else "training"
        if cache_key not in self._datasets:
            from omegaconf import OmegaConf
            from vera.datasets.mixture import MixtureDataset

            # MixtureDataset propagates outer fields (image_to_video, pad_to_width,
            # ...) into each subset cfg via __setitem__. Hydra composes configs in
            # struct mode (no new keys), so deep-copy to a non-struct container
            # before constructing the mixture. This is a config-layer adaptation
            # (the datasets package is untouched).
            mixture_cfg = OmegaConf.create(
                OmegaConf.to_container(dataset_cfg_node, resolve=True)
            )
            OmegaConf.set_struct(mixture_cfg, False)
            self._datasets[cache_key] = MixtureDataset(
                mixture_cfg, split=mixture_split
            )
        return self._datasets[cache_key]


class VideoGenerationExperiment(BaseLightningExperiment):
    """Train the WAN / OMNI video model end-to-end via vera.main."""

    compatible_algorithms: Set[str] = {"wan_t2v", "wan_i2v"}
    compatible_datasets: Set[str] = {
        "combined_4env",
        "mimicgen",
        "allegro_sim",
        "allegro_real",
        "droid_flow",
        "droid",
    }

    data_module_cls = VideoMixtureDataModule

    def __init__(self, root_cfg: DictConfig, logger=None, ckpt_path=None) -> None:
        super().__init__(root_cfg, logger, ckpt_path)

    def _build_common_callbacks(self):
        # The WAN algorithm carries non-DDP/FSDP-friendly frozen sub-modules (VAE,
        # T5, CLIP) and manages its own EMA-free optimizer; the base EMA callback
        # (which deep-copies the full parameter set) is unnecessary and memory-heavy
        # for a 14B model. Honor the experiment's ema.enable flag (default off here).
        ema_cfg = getattr(self.cfg, "ema", None)
        if ema_cfg is not None and bool(getattr(ema_cfg, "enable", False)):
            return super()._build_common_callbacks()
        return []

    def _build_strategy(self):
        # Mirror flow-planner exp_video: support ddp / fsdp / single-device. The WAN
        # algo exposes ``classes_to_shard()`` for FSDP module wrapping.
        strategy = getattr(self.cfg, "strategy", None)

        if torch.cuda.device_count() <= 1:
            return "auto"

        if strategy in (None, "auto", "ddp"):
            return super()._build_strategy()

        if strategy == "fsdp":
            from lightning.pytorch.strategies.fsdp import FSDPStrategy
            from torch.distributed.fsdp import BackwardPrefetch, MixedPrecision
            from torch.distributed.fsdp.wrap import ModuleWrapPolicy

            gpus_per_node = torch.cuda.device_count()
            total_gpus = self.cfg.num_nodes * gpus_per_node
            device_mesh = (
                (total_gpus // 32, 32) if total_gpus >= 32 else (1, total_gpus)
            )
            assert self.algo is not None
            return FSDPStrategy(
                mixed_precision=MixedPrecision(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.bfloat16,
                    buffer_dtype=torch.bfloat16,
                ),
                auto_wrap_policy=ModuleWrapPolicy(self.algo.classes_to_shard()),
                state_dict_type="sharded",
                sharding_strategy="HYBRID_SHARD",
                device_mesh=device_mesh,
                backward_prefetch=BackwardPrefetch.BACKWARD_POST,
            )

        return strategy

    def _inject_dataset_metadata(self):
        # The WAN algorithm has no ``set_dataset_metadata`` hook (it is not an IDM
        # model); skip the metadata injection but still trigger datamodule.setup so
        # the dataloaders are constructed identically to the IDM path.
        if hasattr(self.data_module, "setup"):
            self.data_module.setup(stage=None)
