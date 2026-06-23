import os
from collections import Counter
from typing import Any, Set

import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from omegaconf import DictConfig
from torch.utils.data._utils.collate import default_collate

# 🔑 NEW: registry imports
from vera.datasets.registry import (
    build_dataset,
    resolve_dataset_cfg,
)


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, root_cfg: DictConfig, compatible_datasets: Set[str]) -> None:
        super().__init__()
        self.root_cfg = root_cfg
        self.exp_cfg = root_cfg.experiment
        self.compatible_datasets = compatible_datasets

        # Cache resolved configs per split so validation can override training.
        self._dataset_cfgs: dict[str, object] = {}
        self._datasets: dict[str, torch.utils.data.Dataset] = {}
        self._logged_loader_settings: set[str] = set()
        self.validation_dataloader_names: list[str] = []

        # 🔑 NEW
        self.dataset_metadata: dict[str, float] | None = None

    # ---------------------------------------------------------------------
    # Dataset construction (single source of truth)
    # ---------------------------------------------------------------------

    def get_dataset_metadata(self) -> dict[str, float]:
        return self.dataset_metadata or {}

    def setup(self, stage: str | None = None) -> None:
        """
        Called by Lightning before fit/validate/test.
        Safe place to extract dataset-level metadata.
        """
        # Build a representative dataset (train split is canonical)
        self._configure_data_caches(self.exp_cfg.training.data)
        dataset = self._build_dataset("training")

        if hasattr(dataset, "get_metadata"):
            self.dataset_metadata = dataset.get_metadata()
        else:
            self.dataset_metadata = {}

        # Optional: log once
        if self.trainer is not None and self.trainer.is_global_zero:
            print(f"[DataModule] Dataset metadata: {self.dataset_metadata}")

    def _select_dataset_cfg_node(self, split: str):
        if split == "validation":
            validation_cfg = self.root_cfg.get("validation_dataset")
            if validation_cfg is not None:
                return validation_cfg
        return self.root_cfg.dataset

    def _validation_dataset_cfg_nodes(self) -> list[tuple[str, Any]] | None:
        validation_datasets = self.root_cfg.get("validation_datasets")
        if validation_datasets is None:
            return None
        if not isinstance(validation_datasets, (dict, DictConfig)):
            raise TypeError(
                "validation_datasets must be a mapping of name -> dataset config"
            )
        nodes = [(str(name), cfg) for name, cfg in validation_datasets.items()]
        if not nodes:
            raise ValueError("validation_datasets is present but empty")
        return nodes

    @staticmethod
    def _sanitize_loader_name(name: str) -> str:
        sanitized = str(name).strip().replace("/", "_").replace(" ", "_")
        return sanitized or "unnamed"

    def _build_dataset(self, split: str) -> torch.utils.data.Dataset:
        return self._build_dataset_from_cfg_node(
            split=split,
            cache_key=split,
            dataset_cfg_node=self._select_dataset_cfg_node(split),
        )

    def _build_dataset_from_cfg_node(
        self,
        *,
        split: str,
        cache_key: str,
        dataset_cfg_node: Any,
    ) -> torch.utils.data.Dataset:
        if split not in {"training", "validation", "test"}:
            raise NotImplementedError(f"split '{split}' is not implemented")

        # --------------------------------------------------------------
        # 1. Compatibility check
        # --------------------------------------------------------------
        dataset_name = dataset_cfg_node.name
        if dataset_name not in self.compatible_datasets:
            raise ValueError(
                f"Dataset '{dataset_name}' not allowed. "
                f"Allowed: {sorted(self.compatible_datasets)}"
            )

        # --------------------------------------------------------------
        # 2. Resolve typed dataset config (once)
        # --------------------------------------------------------------
        if cache_key not in self._dataset_cfgs:
            self._dataset_cfgs[cache_key] = resolve_dataset_cfg(dataset_cfg_node)

        if cache_key not in self._datasets:
            # --------------------------------------------------------------
            # 3. Instantiate dataset via registry
            # --------------------------------------------------------------
            self._datasets[cache_key] = build_dataset(
                self._dataset_cfgs[cache_key], stage=split
            )
        return self._datasets[cache_key]

    # ---------------------------------------------------------------------
    # Dataloader helpers (unchanged)
    # ---------------------------------------------------------------------
    @staticmethod
    def _get_shuffle(
        split: str, dataset: torch.utils.data.Dataset, default: bool
    ) -> bool:
        if isinstance(dataset, torch.utils.data.IterableDataset):
            return False
        if split == "training":
            return bool(default)
        # Validation/test remain non-shuffled unless explicitly enabled in split cfg.
        return bool(default)

    def _get_loader_generator(self, split: str, shuffle: bool):
        if not shuffle:
            return None
        split_cfg = self.exp_cfg[split]
        seed = getattr(split_cfg.data, "shuffle_seed", None)
        if seed is None:
            seed = self.root_cfg.get("seed", 0)
        gen = torch.Generator()
        gen.manual_seed(int(seed))
        return gen

    @staticmethod
    def _get_num_workers(num_workers: int) -> int:
        return min(os.cpu_count(), num_workers)

    @staticmethod
    def _read_cache_setting(cache_cfg: Any, key: str) -> int | None:
        if cache_cfg is None:
            return None
        value = getattr(cache_cfg, key, None)
        if value is None:
            return None
        return int(value)

    @classmethod
    def _configure_data_caches(cls, data_cfg: Any) -> None:
        cache_cfg = getattr(data_cfg, "cache", None)
        if cache_cfg is None:
            return

        from vera.datasets.action.loaders.action_loader import configure_droid_h5_cache
        from vera.datasets.action.loaders.cache_io_helper import configure_io_cache_settings

        configure_io_cache_settings(
            video_cache_size=cls._read_cache_setting(cache_cfg, "video"),
            zarr_cache_size=cls._read_cache_setting(cache_cfg, "zarr"),
            trajectory_cache_size=cls._read_cache_setting(cache_cfg, "trajectory"),
            npz_cache_size=cls._read_cache_setting(cache_cfg, "npz"),
        )
        configure_droid_h5_cache(
            maxsize=cls._read_cache_setting(cache_cfg, "droid_h5")
        )

    @classmethod
    def _worker_init(cls, dataset: torch.utils.data.Dataset, data_cfg: Any, worker_id: int):
        # Ensure each worker starts with clean process-local caches after fork.
        from vera.datasets.action.loaders.action_loader import clear_droid_h5_cache
        from vera.datasets.action.loaders.cache_io_helper import clear_all_caches

        clear_all_caches()
        clear_droid_h5_cache()
        cls._configure_data_caches(data_cfg)
        if hasattr(dataset, "worker_init_fn"):
            dataset.worker_init_fn(worker_id)

    @staticmethod
    def _clone_tensor_leaves(value: Any) -> Any:
        """Recursively clone tensor leaves to avoid non-resizable storage issues."""
        if isinstance(value, torch.Tensor):
            # Clone to fresh, contiguous storage that DataLoader can safely collate.
            return value.contiguous().clone()
        if isinstance(value, dict):
            return {k: BaseDataModule._clone_tensor_leaves(v) for k, v in value.items()}
        if isinstance(value, list):
            return [BaseDataModule._clone_tensor_leaves(v) for v in value]
        if isinstance(value, tuple):
            return tuple(BaseDataModule._clone_tensor_leaves(v) for v in value)
        return value

    @classmethod
    def _manual_tensor_safe_collate(cls, batch: list[Any]) -> Any:
        """
        Fallback collate implementation that avoids resize_ on shared storage.
        """
        elem = batch[0]
        if isinstance(elem, torch.Tensor):
            return torch.stack([b.contiguous() for b in batch], dim=0)
        if isinstance(elem, dict):
            return {k: cls._manual_tensor_safe_collate([d[k] for d in batch]) for k in elem}
        if isinstance(elem, tuple):
            return tuple(cls._manual_tensor_safe_collate(list(items)) for items in zip(*batch))
        if isinstance(elem, list):
            return [cls._manual_tensor_safe_collate(list(items)) for items in zip(*batch)]
        # Scalars/strings/other small metadata can use default_collate safely.
        return default_collate(batch)

    @classmethod
    def _collect_tensor_leaf_specs(
        cls, value: Any, path: str, out: list[tuple[str, tuple[int, ...], str]]
    ) -> None:
        if isinstance(value, torch.Tensor):
            out.append((path, tuple(int(x) for x in value.shape), str(value.dtype)))
            return
        if isinstance(value, dict):
            for key in sorted(value.keys(), key=lambda x: str(x)):
                cls._collect_tensor_leaf_specs(value[key], f"{path}.{key}", out)
            return
        if isinstance(value, (list, tuple)):
            for idx, item in enumerate(value):
                cls._collect_tensor_leaf_specs(item, f"{path}[{idx}]", out)
            return

    @classmethod
    def _sample_tensor_signature(cls, sample: Any) -> tuple[tuple[str, tuple[int, ...], str], ...]:
        specs: list[tuple[str, tuple[int, ...], str]] = []
        cls._collect_tensor_leaf_specs(sample, path="root", out=specs)
        if not specs:
            return (("root", tuple(), "NO_TENSOR"),)
        return tuple(specs)

    @classmethod
    def _majority_signature(
        cls, batch: list[Any]
    ) -> tuple[tuple[str, tuple[int, ...], str], ...] | None:
        if not batch:
            return None
        signatures = [cls._sample_tensor_signature(sample) for sample in batch]
        counts = Counter(signatures)
        return counts.most_common(1)[0][0]

    @classmethod
    def _filter_batch_by_majority_signature(
        cls, batch: list[Any]
    ) -> tuple[list[Any], list[dict[str, Any]], str]:
        if not batch:
            return [], [], "[]"
        kept_signature = cls._majority_signature(batch)
        if kept_signature is None:
            return [], [], "[]"

        kept: list[Any] = []
        dropped: list[dict[str, Any]] = []
        kept_signature_text = str(kept_signature)
        for i, sample in enumerate(batch):
            signature = cls._sample_tensor_signature(sample)
            if signature == kept_signature:
                kept.append(sample)
                continue

            if isinstance(sample, dict):
                dropped.append(
                    {
                        "batch_pos": i,
                        "sample_idx": sample.get("_sample_idx"),
                        "episode_idx": sample.get("_episode_idx"),
                        "timesteps": sample.get("_timesteps"),
                        "signature": str(signature),
                    }
                )
            else:
                dropped.append(
                    {
                        "batch_pos": i,
                        "signature": str(signature),
                    }
                )
        return kept, dropped, kept_signature_text

    @classmethod
    def _log_collate_failure_debug(cls, batch: list[Any], exc: Exception) -> None:
        # Lightweight diagnostics to help isolate faulty samples.
        preview = []
        for i, sample in enumerate(batch[:8]):
            if isinstance(sample, dict):
                item = {
                    "batch_pos": i,
                    "sample_idx": sample.get("_sample_idx"),
                    "episode_idx": sample.get("_episode_idx"),
                    "timesteps": sample.get("_timesteps"),
                }
                preview.append(item)
        print(
            "[DataModule] default_collate failed. "
            f"Using tensor-safe fallback. Preview: {preview}. "
            f"Error: {exc.__class__.__name__}: {exc}"
        )

    # Recoverable default_collate failures: non-resizable shared storage, and
    # mixed tensor shapes within a batch (e.g. a rare DROID episode with 2
    # camera views in a 3-view dataset; crashed jacobian-learning/7sv5x0ai).
    # Both are handled by the fallback chain ending in majority-signature drop.
    _RECOVERABLE_COLLATE_ERRORS = (
        "not resizable",
        "stack expects each tensor to be equal size",
    )

    @classmethod
    def _safe_default_collate(cls, batch: list[Any]) -> Any:
        """
        Keep default_collate fast path, but recover from non-resizable storage
        tensors and mixed-shape batches.
        """
        try:
            return default_collate(batch)
        except RuntimeError as exc:
            if not any(msg in str(exc) for msg in cls._RECOVERABLE_COLLATE_ERRORS):
                raise
            cls._log_collate_failure_debug(batch, exc)
            if os.environ.get("OKTO_FAIL_ON_COLLATE_ERROR", "0") == "1":
                raise
            sanitized_batch = [cls._clone_tensor_leaves(sample) for sample in batch]
            try:
                return default_collate(sanitized_batch)
            except RuntimeError:
                # Final fallback: bypass default tensor collate storage resizing path.
                try:
                    return cls._manual_tensor_safe_collate(sanitized_batch)
                except RuntimeError as stack_exc:
                    # Mixed-shape batches (e.g. [T,V,C,H,W] + [T,C,H,W]) cannot be stacked.
                    filtered_batch, dropped_samples, kept_signature = (
                        cls._filter_batch_by_majority_signature(sanitized_batch)
                    )
                    if dropped_samples:
                        print(
                            "[DataModule] Dropping incompatible samples during collate. "
                            f"kept={len(filtered_batch)}/{len(sanitized_batch)} "
                            f"kept_signature={kept_signature} "
                            f"dropped={dropped_samples}"
                        )
                    if not filtered_batch:
                        raise RuntimeError(
                            "Collate fallback dropped all samples due to incompatible tensor "
                            f"signatures. Original error: {stack_exc}"
                        ) from stack_exc
                    return cls._manual_tensor_safe_collate(filtered_batch)

    def _dataloader_for_dataset(
        self,
        split: str,
        dataset: torch.utils.data.Dataset,
        *,
        loader_label: str,
    ) -> TRAIN_DATALOADERS | EVAL_DATALOADERS:
        split_cfg = self.exp_cfg[split]
        self._configure_data_caches(split_cfg.data)

        num_workers = self._get_num_workers(split_cfg.data.num_workers)
        # Optional knobs (not all experiment yamls define them)
        pin_memory = bool(getattr(split_cfg.data, "pin_memory", False))
        prefetch_factor = getattr(split_cfg.data, "prefetch_factor", None)
        persistent_workers_cfg = bool(getattr(split_cfg.data, "persistent_workers", True))
        persistent_workers = persistent_workers_cfg and (num_workers > 0)
        shuffle = self._get_shuffle(split, dataset, split_cfg.data.shuffle)
        generator = self._get_loader_generator(split, shuffle)

        if loader_label not in self._logged_loader_settings:
            print(
                "[DataModule] "
                f"{loader_label} loader: batch_size={split_cfg.batch_size}, "
                f"num_workers={num_workers}, shuffle={shuffle}, "
                f"shuffle_seed={getattr(split_cfg.data, 'shuffle_seed', None) if shuffle else None}, "
                f"persistent_workers={persistent_workers}, "
                f"prefetch_factor={prefetch_factor if num_workers > 0 else None}, "
                f"pin_memory={pin_memory}"
            )
            self._logged_loader_settings.add(loader_label)

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=split_cfg.batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=pin_memory,
            generator=generator,
            prefetch_factor=prefetch_factor if (num_workers > 0) else None,
            persistent_workers=persistent_workers,
            collate_fn=self._safe_default_collate,
            worker_init_fn=lambda worker_id: self._worker_init(
                dataset, split_cfg.data, worker_id
            ),
        )

    def _dataloader(self, split: str) -> TRAIN_DATALOADERS | EVAL_DATALOADERS:
        dataset = self._build_dataset(split)
        return self._dataloader_for_dataset(split, dataset, loader_label=split)

    # ---------------------------------------------------------------------
    # Lightning hooks (unchanged)
    # ---------------------------------------------------------------------
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self._dataloader("training")

    def val_dataloader(self) -> EVAL_DATALOADERS:
        named_validation_nodes = self._validation_dataset_cfg_nodes()
        if named_validation_nodes is None:
            self.validation_dataloader_names = ["validation"]
            return self._dataloader("validation")

        loaders: list[torch.utils.data.DataLoader] = []
        datasets: list[torch.utils.data.Dataset] = []
        names: list[str] = []

        for raw_name, dataset_cfg_node in named_validation_nodes:
            name = self._sanitize_loader_name(raw_name)
            cache_key = f"validation:{name}"
            dataset = self._build_dataset_from_cfg_node(
                split="validation",
                cache_key=cache_key,
                dataset_cfg_node=dataset_cfg_node,
            )
            datasets.append(dataset)
            names.append(name)
            loaders.append(
                self._dataloader_for_dataset(
                    "validation",
                    dataset,
                    loader_label=cache_key,
                )
            )

        include_all = bool(self.root_cfg.get("validation_include_all", True))
        if include_all and len(datasets) > 1:
            names.append("all")
            loaders.append(
                self._dataloader_for_dataset(
                    "validation",
                    torch.utils.data.ConcatDataset(datasets),
                    loader_label="validation:all",
                )
            )

        self.validation_dataloader_names = names
        return loaders

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self._dataloader("test")
