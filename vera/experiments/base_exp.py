"""
This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research
template [repo](https://github.com/buoyancy99/research-template).
By its MIT license, you must keep the above sentence in `README.md`
and the `LICENSE` file to credit the author.
"""

import pathlib
import json
import time
from abc import ABC
from typing import Optional, Union, cast

import hydra
import lightning.pytorch as pl
import torch
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.strategies.ddp import DDPStrategy
from omegaconf import DictConfig

from vera.idm.common.base_pytorch_algo import BasePytorchAlgo
from vera.idm.registry import (
    resolve_algorithm_cfg,
    resolve_algorithm_instance,
)

from ..utils.distributed_utils import rank_zero_print
from ..utils.lightning_utils import EMA
from ..utils.print_utils import cyan
from .data_modules import BaseDataModule

torch.set_float32_matmul_precision("high")


def _agent_debug_log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    # region agent log
    try:
        payload = {
            "sessionId": "cc3db4",
            "runId": "pre-fix",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open(
            "/path/to/repo/.cursor/debug-cc3db4.log",
            "a",
            encoding="utf-8",
        ) as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass
    # endregion


class BaseExperiment(ABC):
    """
    Abstract class for an experiment. This generalizes the pytorch lightning Trainer & lightning Module to more
    flexible experiments that doesn't fit in the typical ml loop, e.g. multi-stage reinforcement learning benchmarks.
    """

    # each key has to be a yaml file under '[project_root]/configurations/algorithm' without .yaml suffix
    compatible_algorithms: set[str] = set()

    def __init__(
        self,
        root_cfg: DictConfig,
        logger: Optional[WandbLogger] = None,
        ckpt_path: Optional[Union[str, pathlib.Path]] = None,
    ) -> None:
        """
        Constructor

        Args:
            cfg: configuration file that contains everything about the experiment
            logger: a pytorch-lightning WandbLogger instance
            ckpt_path: an optional path to saved checkpoint
        """
        super().__init__()
        self.root_cfg = root_cfg
        self.cfg = root_cfg.experiment
        self.debug = root_cfg.debug
        self.logger = logger if logger else False
        self.ckpt_path = ckpt_path
        self.algo: Optional[BasePytorchAlgo] = None

    def _build_algo(self) -> BasePytorchAlgo:
        algo_cfg_node = self.root_cfg.algorithm
        algo_name = algo_cfg_node.name

        if algo_name not in self.compatible_algorithms:
            raise ValueError(
                f"Algorithm '{algo_name}' not supported by {self.__class__.__name__}. "
                f"Supported: {sorted(self.compatible_algorithms)}"
            )

        # 1. Resolve typed config
        algo_cfg = resolve_algorithm_cfg(algo_cfg_node)

        # 2. Instantiate algorithm via registry
        algo = cast(BasePytorchAlgo, resolve_algorithm_instance(algo_cfg))

        return algo

    def exec_task(self, task: str) -> None:
        """
        Executing a certain task specified by string. Each task should be a stage of experiment.
        In most computer vision / nlp applications, tasks should be just train and test.
        In reinforcement learning, you might have more stages such as collecting dataset etc

        Args:
            task: a string specifying a task implemented for this experiment
        """

        if hasattr(self, task) and callable(getattr(self, task)):
            rank_zero_print(cyan("Executing task:"), f"{task} out of {self.cfg.tasks}")
            getattr(self, task)()
        else:
            raise ValueError(
                f"Specified task '{task}' not defined for class {self.__class__.__name__} or is not callable."
            )


class BaseLightningExperiment(BaseExperiment):
    """
    Abstract class for pytorch lightning experiments. Useful for computer vision & nlp where main components are
    simply models, datasets and train loop.
    """

    # each key has to be a yaml file under '[project_root]/configurations/algorithm' without .yaml suffix
    compatible_algorithms: set[str] = set()

    # each key has to be a yaml file under '[project_root]/configurations/dataset' without .yaml suffix
    compatible_datasets: set[str] = set()
    data_module_cls = BaseDataModule

    def __init__(
        self,
        root_cfg: DictConfig,
        logger: Optional[WandbLogger] = None,
        ckpt_path: Optional[Union[str, pathlib.Path]] = None,
    ) -> None:
        super().__init__(root_cfg, logger, ckpt_path)
        self.data_module = self.data_module_cls(root_cfg, self.compatible_datasets)

    def _build_common_callbacks(self):
        return [EMA(**self.cfg.ema)]

    def _get_output_dir(self) -> pathlib.Path:
        return pathlib.Path(HydraConfig.get().runtime.output_dir)

    def _build_strategy(self):
        return (
            DDPStrategy(find_unused_parameters=self.cfg.find_unused_parameters)
            if torch.cuda.device_count() > 1
            else "auto"
        )

    def _build_trainer(self, stage: str, callbacks: list) -> pl.Trainer:
        training_cfg = self.cfg.training
        enable_progress_bar = bool(getattr(training_cfg, "enable_progress_bar", True))
        progress_bar_refresh_rate = int(
            getattr(training_cfg, "progress_bar_refresh_rate", 1)
        )
        if enable_progress_bar:
            callbacks = list(callbacks) + [
                TQDMProgressBar(refresh_rate=max(progress_bar_refresh_rate, 1))
            ]

        trainer_kwargs = dict(
            accelerator="auto",
            logger=self.logger,
            devices="auto",
            num_nodes=self.cfg.num_nodes,
            strategy=self._build_strategy(),
            callbacks=callbacks,
            enable_progress_bar=enable_progress_bar,
            log_every_n_steps=int(getattr(training_cfg, "log_every_n_steps", 50)),
            detect_anomaly=False,  # self.cfg.debug,
        )

        if stage == "training":
            trainer_kwargs.update(
                gradient_clip_val=self.cfg.training.optim.gradient_clip_val,
                val_check_interval=self.cfg.validation.val_every_n_step,
                limit_val_batches=self.cfg.validation.limit_batch,
                check_val_every_n_epoch=self.cfg.validation.val_every_n_epoch,
                accumulate_grad_batches=self.cfg.training.optim.accumulate_grad_batches,
                precision=self.cfg.training.precision,
                num_sanity_val_steps=(
                    int(self.cfg.debug)
                    if self.cfg.validation.num_sanity_val_steps is None
                    else self.cfg.validation.num_sanity_val_steps
                ),
                max_epochs=self.cfg.training.max_epochs,
                max_steps=self.cfg.training.max_steps,
                max_time=self.cfg.training.max_time,
                reload_dataloaders_every_n_epochs=self.cfg.reload_dataloaders_every_n_epochs,
            )
        elif stage == "validation":
            trainer_kwargs.update(
                limit_val_batches=self.cfg.validation.limit_batch,
                precision=self.cfg.validation.precision,
                inference_mode=self.cfg.validation.inference_mode,
            )
        elif stage == "test":
            trainer_kwargs.update(
                limit_test_batches=self.cfg.test.limit_batch,
                precision=self.cfg.test.precision,
                inference_mode=self.cfg.test.inference_mode,
            )
        else:
            raise ValueError(f"Unsupported trainer stage '{stage}'.")

        return pl.Trainer(**trainer_kwargs)

    def _log_training_workload_advisory(self) -> None:
        dataset_cfg = getattr(self.root_cfg, "dataset", None)
        if dataset_cfg is None:
            return

        dataset_name = getattr(dataset_cfg, "name", None)
        batch_size = int(getattr(self.cfg.training, "batch_size", 1))
        num_frames = int(
            getattr(getattr(dataset_cfg, "sampling", None), "num_frames", 1)
        )
        views = list(getattr(getattr(dataset_cfg, "camera", None), "views", []) or [])
        num_views = len(views)
        world_size = max(torch.cuda.device_count(), 1)
        num_workers = int(getattr(getattr(self.cfg.training, "data", None), "num_workers", 0))
        prefetch_factor = getattr(getattr(self.cfg.training, "data", None), "prefetch_factor", None)

        if (
            dataset_name == "droid"
            and world_size >= 4
            and batch_size <= 1
            and num_frames >= 16
            and num_views >= 3
        ):
            rank_zero_print(
                cyan("[Perf advisory]"),
                (
                    "This run is using a loader-heavy DROID configuration "
                    f"(world_size={world_size}, batch_size={batch_size}, "
                    f"num_frames={num_frames}, views={num_views}, "
                    f"num_workers={num_workers}, prefetch_factor={prefetch_factor}). "
                    "If GPU utilization stays low, compare 1-2 GPUs against 8 GPUs before "
                    "investing in more DDP tuning."
                ),
            )

    def training(self) -> None:
        """
        All training happens here
        """
        if not self.algo:
            self.algo = self._build_algo()
        assert self.algo is not None

        # inject dataset metadata
        self._inject_dataset_metadata()

        if self.cfg.training.compile:
            self.algo = cast(BasePytorchAlgo, torch.compile(self.algo))

        callbacks = []
        if self.logger:
            callbacks.append(LearningRateMonitor("step", True))
        if "checkpointing" in self.cfg.training:
            _agent_debug_log(
                "H5",
                "base_exp.py:training:model_checkpoint",
                "checkpoint callback configured",
                {
                    "monitor": self.cfg.training.checkpointing.get("monitor", None),
                    "mode": self.cfg.training.checkpointing.get("mode", None),
                    "save_top_k": self.cfg.training.checkpointing.get("save_top_k", None),
                    "save_last": self.cfg.training.checkpointing.get("save_last", None),
                    "every_n_train_steps": self.cfg.training.checkpointing.get("every_n_train_steps", None),
                },
            )
            callbacks.append(
                ModelCheckpoint(
                    self._get_output_dir() / "checkpoints",
                    **self.cfg.training.checkpointing,
                )
            )
        callbacks += self._build_common_callbacks()
        self._log_training_workload_advisory()

        trainer = self._build_trainer("training", callbacks)

        # if self.debug:
        #     self.logger.watch(self.algo, log="all")

        # When override_optim_state is True, load only model weights from the checkpoint
        # and start training from step 0 with config lr (no optimizer/scheduler/step from ckpt).
        # Works with load=, resume=, or +requeue=.
        ckpt_path_for_fit = self.ckpt_path
        if self.ckpt_path and self.cfg.training.get("override_optim_state", False):
            rank_zero_print(
                cyan("Override optim state: loading model weights only, ignoring checkpoint optim/step.")
            )
            ckpt = torch.load(
                str(self.ckpt_path),
                map_location="cpu",
                weights_only=False,
            )
            state_dict = ckpt.get("state_dict", ckpt)
            # Default strict=False so checkpoint can omit buffers (e.g. data_mean, data_std) set from dataset metadata.
            strict = self.cfg.training.get("override_strict_load", False)
            self.algo.load_state_dict(state_dict, strict=strict)
            ckpt_path_for_fit = None

        trainer.fit(
            self.algo,
            datamodule=self.data_module,
            ckpt_path=ckpt_path_for_fit,
        )

    def validation(self) -> None:
        """
        All validation happens here
        """
        if not self.algo:
            self.algo = self._build_algo()
        assert self.algo is not None

        # inject dataset metadata
        self._inject_dataset_metadata()

        if self.cfg.validation.compile:
            self.algo = cast(BasePytorchAlgo, torch.compile(self.algo))

        callbacks = [] + self._build_common_callbacks()

        trainer = self._build_trainer("validation", callbacks)

        # if self.debug:
        #     self.logger.watch(self.algo, log="all")

        trainer.validate(
            self.algo,
            datamodule=self.data_module,
            ckpt_path=self.ckpt_path,
        )

    def test(self) -> None:
        """
        All testing happens here
        """
        if not self.algo:
            self.algo = self._build_algo()
        assert self.algo is not None

        # inject dataset metadata
        self._inject_dataset_metadata()

        if self.cfg.test.compile:
            self.algo = cast(BasePytorchAlgo, torch.compile(self.algo))

        callbacks = [] + self._build_common_callbacks()

        trainer = self._build_trainer("test", callbacks)

        # Only load the checkpoint if only testing. Otherwise, it will have been loaded
        # and further trained during train.
        trainer.test(
            self.algo,
            datamodule=self.data_module,
            ckpt_path=self.ckpt_path,
        )

    def _inject_dataset_metadata(self):
        """
        Inject dataset-level semantic metadata into the algorithm (once).
        Safe to call multiple times.
        """
        # Ensure datamodule is initialized
        if hasattr(self.data_module, "setup"):
            self.data_module.setup(stage=None)

        assert self.algo is not None
        if hasattr(self.algo, "set_dataset_metadata"):
            metadata = self.data_module.get_dataset_metadata()
            self.algo.set_dataset_metadata(metadata)
