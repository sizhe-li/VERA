import pathlib
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import einops
import lightning.pytorch as pl
import numpy as np
import torch
import wandb
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning_utilities.core.apply_func import apply_to_collection
from PIL import Image

from vera.utils.distributed_utils import rank_zero_print
from vera.utils.print_utils import cyan


@dataclass
class ProfilingCfg:
    enable: bool = False
    step_unit: Literal["optimizer", "batch"] = "optimizer"
    start_after_k_steps: Optional[int] = None
    every_k_steps: Optional[int] = None
    max_windows: int = 1
    warmup: int = 10
    active: int = 4
    wait: Optional[int] = None
    skip_first: int = 0
    with_stack: bool = True
    record_shapes: bool = False
    profile_memory: bool = False
    export_to_chrome: bool = True
    trace_dir: str = "profiles"
    filename_prefix: str = "fit"


@dataclass
class BasePytorchAlgoCfg:
    """
    A base configuration dataclass for Pytorch algorithms.
    """

    name: Literal[""]
    debug: bool = False
    lr: float = 1e-3
    profiling: ProfilingCfg = field(default_factory=ProfilingCfg)


class BasePytorchAlgo(pl.LightningModule, ABC):
    def __init__(self, cfg: BasePytorchAlgoCfg):
        super().__init__()
        self.cfg = cfg
        self.debug = self.cfg.debug
        self.should_validate_ema_weights = False

        self.dataset_metadata: dict[str, Any] = {}
        self._profiler: Optional[torch.profiler.profile] = None
        self._profiling_step_count = 0
        self._profiling_windows_started = 0
        self._profiling_window_active = False
        self._profiling_window_steps = 0
        self._profiling_current_window_index = 0
        self._profiling_state: Literal["idle", "window_open", "done"] = "idle"

        self._build_model()

    def set_dataset_metadata(self, metadata: dict[str, Any]) -> None:
        if getattr(self, "_dataset_metadata_set", False):
            return

        self.dataset_metadata = metadata or {}
        self._dataset_metadata_set = True

        rank_zero_print(
            cyan("[Algo] Received dataset metadata:"),
            self.dataset_metadata,
        )

    @abstractmethod
    def _build_model(self):
        """
        Create all pytorch nn.Modules here.
        """
        raise NotImplementedError

    @abstractmethod
    def training_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        r"""Here you compute and return the training loss and some additional metrics for e.g. the progress bar or
        logger.

        Args:
            batch: The output of your data iterable, normally a :class:`~torch.utils.data.DataLoader`.
            batch_idx: The index of this batch.
            dataloader_idx: (only if multiple dataloaders used) The index of the dataloader that produced this batch.

        Return:
            Any of these options:
            - :class:`~torch.Tensor` - The loss tensor
            - ``dict`` - A dictionary. Can include any keys, but must include the key ``'loss'``.
            - ``None`` - Skip to the next batch. This is only supported for automatic optimization.
                This is not supported for multi-GPU, TPU, IPU, or DeepSpeed.

        In this step you'd normally do the forward pass and calculate the loss for a batch.
        You can also do fancier things like multiple forward passes or something model specific.

        Example::

            def training_step(self, batch, batch_idx):
                x, y, z = batch
                out = self.encoder(x)
                loss = self.loss(out, x)
                return loss

        To use multiple optimizers, you can switch to 'manual optimization' and control their stepping:

        .. code-block:: python

            def __init__(self):
                super().__init__()
                self.automatic_optimization = False


            # Multiple optimizers (e.g.: GANs)
            def training_step(self, batch, batch_idx):
                opt1, opt2 = self.optimizers()

                # do training_step with encoder
                ...
                opt1.step()
                # do training_step with decoder
                ...
                opt2.step()

        Note:
            When ``accumulate_grad_batches`` > 1, the loss returned here will be automatically
            normalized by ``accumulate_grad_batches`` internally.

        """
        return super().training_step(*args, **kwargs)

    def configure_optimizers(self):
        """
        Return an optimizer. If you need to use more than one optimizer, refer to pytorch lightning documentation:
        https://lightning.ai/docs/pytorch/stable/common/optimization.html
        """
        parameters = self.parameters()
        return torch.optim.Adam(parameters, lr=self.cfg.lr)

    def _get_profiling_start_step(self) -> Optional[int]:
        profiling_cfg = self.cfg.profiling
        if profiling_cfg.start_after_k_steps is not None:
            return int(profiling_cfg.start_after_k_steps)
        if profiling_cfg.every_k_steps is not None:
            return int(profiling_cfg.every_k_steps)
        if profiling_cfg.wait is not None:
            return int(profiling_cfg.wait) + int(profiling_cfg.skip_first)
        return None

    def _get_trace_root(self) -> pathlib.Path:
        log_dir = getattr(self.trainer, "log_dir", None) or getattr(
            self.trainer, "default_root_dir", "."
        )
        trace_root = pathlib.Path(str(log_dir)) / str(self.cfg.profiling.trace_dir)
        trace_root.mkdir(parents=True, exist_ok=True)
        return trace_root

    def _get_trace_name(self, window_index: int) -> str:
        rank = int(getattr(self, "global_rank", 0))
        world_size = int(getattr(self.trainer, "world_size", 1))
        prefix = str(self.cfg.profiling.filename_prefix)
        return f"{prefix}-rank{rank:02d}-of{world_size:02d}-window{window_index:02d}"

    def on_trace_ready(self, prof: torch.profiler.profile) -> None:
        if not self.cfg.profiling.export_to_chrome:
            return
        handler = torch.profiler.tensorboard_trace_handler(
            str(self._get_trace_root()),
            worker_name=self._get_trace_name(self._profiling_current_window_index),
            use_gzip=False,
        )
        handler(prof)

    def _build_profiler(self) -> torch.profiler.profile:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        return torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=0,
                warmup=int(self.cfg.profiling.warmup),
                active=int(self.cfg.profiling.active),
                repeat=1,
            ),
            on_trace_ready=self.on_trace_ready,
            profile_memory=bool(self.cfg.profiling.profile_memory),
            with_stack=bool(self.cfg.profiling.with_stack),
            record_shapes=bool(self.cfg.profiling.record_shapes),
        )

    def _reset_profiling_state(self) -> None:
        self._profiler = None
        self._profiling_step_count = 0
        self._profiling_windows_started = 0
        self._profiling_window_active = False
        self._profiling_window_steps = 0
        self._profiling_current_window_index = 0
        self._profiling_state = "idle"

    def _profiling_is_enabled(self) -> bool:
        profiling_cfg = self.cfg.profiling
        if not profiling_cfg.enable:
            return False
        return int(profiling_cfg.active) > 0 and int(profiling_cfg.max_windows) > 0

    def _should_open_profile_window(self) -> bool:
        if not self._profiling_is_enabled():
            return False
        if self._profiling_window_active or self._profiler is not None:
            return False
        if self._profiling_state == "done":
            return False

        max_windows = int(self.cfg.profiling.max_windows)
        if max_windows > 0 and self._profiling_windows_started >= max_windows:
            self._profiling_state = "done"
            return False

        start_step = self._get_profiling_start_step()
        current_step = self._profiling_step_count
        if start_step is None:
            start_step = 0
        if current_step < start_step:
            return False

        every_k_steps = self.cfg.profiling.every_k_steps
        if every_k_steps is None:
            return self._profiling_windows_started == 0 and current_step == start_step

        period = int(every_k_steps)
        if period <= 0:
            return False
        return (current_step - start_step) % period == 0

    def _open_profile_window(self) -> None:
        if self._profiling_window_active:
            return
        self._profiling_windows_started += 1
        self._profiling_current_window_index = self._profiling_windows_started
        self._profiler = self._build_profiler()
        self._profiler.__enter__()
        self._profiling_window_active = True
        self._profiling_window_steps = 0
        self._profiling_state = "window_open"

    def _close_profile_window(self) -> None:
        if self._profiler is not None:
            self._profiler.__exit__(None, None, None)
            self._profiler = None
        self._profiling_window_active = False
        self._profiling_window_steps = 0
        max_windows = int(self.cfg.profiling.max_windows)
        if max_windows > 0 and self._profiling_windows_started >= max_windows:
            self._profiling_state = "done"
        else:
            self._profiling_state = "idle"

    def _run_manual_profiler_step(self, step_unit: Literal["optimizer", "batch"]) -> None:
        if not self._profiling_is_enabled():
            return
        if self.cfg.profiling.step_unit != step_unit:
            return
        if self._should_open_profile_window():
            self._open_profile_window()

        if self._profiling_window_active and self._profiler is not None:
            self._profiler.step()
            self._profiling_window_steps += 1
            total_steps = int(self.cfg.profiling.warmup) + int(self.cfg.profiling.active)
            if self._profiling_window_steps >= total_steps:
                self._close_profile_window()

        self._profiling_step_count += 1

    def on_fit_start(self) -> None:
        super().on_fit_start()
        self._reset_profiling_state()

    def on_fit_end(self) -> None:
        self._close_profile_window()
        super().on_fit_end()

    def on_exception(self, exception: BaseException) -> None:
        self._close_profile_window()
        super().on_exception(exception)

    def on_train_batch_end(
        self,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._run_manual_profiler_step("batch")
        return super().on_train_batch_end(outputs, batch, batch_idx)

    def optimizer_step(self, *args: Any, **kwargs: Any) -> Any:
        out = super().optimizer_step(*args, **kwargs)
        self._run_manual_profiler_step("optimizer")
        return out

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        if self.should_validate_ema_weights:
            self._load_ema_weights_to_state_dict(checkpoint)

    def _load_ema_weights_to_state_dict(self, checkpoint: dict) -> None:
        """
        Load EMA weights to state dict.
        """
        rank_zero_print(
            cyan(
                "WARNING: should_validate_ema_weights is set to True, but cannot validate with EMA weights."
            ),
            "Please implement '_load_ema_weights_to_state_dict' in your LightningModule to validate with EMA weights.",
        )

    def log_video(
        self,
        key: str,
        video: Union[np.ndarray, torch.Tensor],
        mean: Union[np.ndarray, torch.Tensor, Sequence, float] = None,
        std: Union[np.ndarray, torch.Tensor, Sequence, float] = None,
        fps: int = 12,
        format: str = "mp4",
    ):
        """
        Log video to wandb. WandbLogger in pytorch lightning does not support video logging yet, so we call wandb directly.

        Args:
            video: a numpy array or tensor, either in form (time, channel, height, width) or in the form
                (batch, time, channel, height, width). The content must be be in 0-255 if under dtype uint8
                or [0, 1] otherwise.
            mean: optional, the mean to unnormalize video tensor, assuming unnormalized data is in [0, 1].
            std: optional, the std to unnormalize video tensor, assuming unnormalized data is in [0, 1].
            key: the name of the video.
            fps: the frame rate of the video.
            format: the format of the video. Can be either "mp4" or "gif".
        """

        if isinstance(video, torch.Tensor):
            video = video.detach().cpu().numpy()

        expand_shape = [1] * (len(video.shape) - 2) + [3, 1, 1]
        if std is not None:
            if isinstance(std, (float, int)):
                std = [std] * 3
            if isinstance(std, torch.Tensor):
                std = std.detach().cpu().numpy()
            std = np.array(std).reshape(*expand_shape)
            video = video * std
        if mean is not None:
            if isinstance(mean, (float, int)):
                mean = [mean] * 3
            if isinstance(mean, torch.Tensor):
                mean = mean.detach().cpu().numpy()
            mean = np.array(mean).reshape(*expand_shape)
            video = video + mean

        if video.dtype != np.uint8:
            video = np.clip(video, a_min=0, a_max=1) * 255
            video = video.astype(np.uint8)

        self.logger.experiment.log(
            {
                key: wandb.Video(video, fps=fps, format=format),
            },
            step=self.global_step,
        )

    def log_image(
        self,
        key: str,
        image: Union[np.ndarray, torch.Tensor, Image.Image, Sequence[Image.Image]],
        mean: Union[np.ndarray, torch.Tensor, Sequence, float] = None,
        std: Union[np.ndarray, torch.Tensor, Sequence, float] = None,
        **kwargs: Any,
    ):
        """
        Log image(s) using WandbLogger.
        Args:
            key: the name of the video.
            image: a single image or a batch of images. If a batch of images, the shape should be (batch, channel, height, width).
            mean: optional, the mean to unnormalize image tensor, assuming unnormalized data is in [0, 1].
            std: optional, the std to unnormalize tensor, assuming unnormalized data is in [0, 1].
            kwargs: optional, WandbLogger log_image kwargs, such as captions=xxx.
        """
        if isinstance(image, Image.Image):
            image = [image]
        elif len(image) and not isinstance(image[0], Image.Image):
            if isinstance(image, torch.Tensor):
                image = image.detach().cpu().numpy()

            if len(image.shape) == 3:
                image = image[None]

            if image.shape[1] == 3:
                if image.shape[-1] == 3:
                    warnings.warn(
                        f"Two channels in shape {image.shape} have size 3, assuming channel first."
                    )
                image = einops.rearrange(image, "b c h w -> b h w c")

            if std is not None:
                if isinstance(std, (float, int)):
                    std = [std] * 3
                if isinstance(std, torch.Tensor):
                    std = std.detach().cpu().numpy()
                std = np.array(std)[None, None, None]
                image = image * std
            if mean is not None:
                if isinstance(mean, (float, int)):
                    mean = [mean] * 3
                if isinstance(mean, torch.Tensor):
                    mean = mean.detach().cpu().numpy()
                mean = np.array(mean)[None, None, None]
                image = image + mean

            if image.dtype != np.uint8:
                image = np.clip(image, a_min=0.0, a_max=1.0) * 255
                image = image.astype(np.uint8)
                image = [img for img in image]

        self.logger.log_image(key=key, images=image, **kwargs)

    def log_gradient_stats(self):
        """Log gradient statistics such as the mean or std of norm."""

        with torch.no_grad():
            grad_norms = []
            gpr = []  # gradient-to-parameter ratio
            for param in self.parameters():
                if param.grad is not None:
                    grad_norms.append(torch.norm(param.grad).item())
                    gpr.append(torch.norm(param.grad) / torch.norm(param))
            if len(grad_norms) == 0:
                return
            grad_norms = torch.tensor(grad_norms)
            gpr = torch.tensor(gpr)
            self.log_dict(
                {
                    "train/grad_norm/min": grad_norms.min(),
                    "train/grad_norm/max": grad_norms.max(),
                    "train/grad_norm/std": grad_norms.std(),
                    "train/grad_norm/mean": grad_norms.mean(),
                    "train/grad_norm/median": torch.median(grad_norms),
                    "train/gpr/min": gpr.min(),
                    "train/gpr/max": gpr.max(),
                    "train/gpr/std": gpr.std(),
                    "train/gpr/mean": gpr.mean(),
                    "train/gpr/median": torch.median(gpr),
                }
            )

    def gather_data(
        self, data: Union[torch.Tensor, Dict, List, Tuple], batch_dim: int = 0
    ):
        """
        Gather tensors or collections of tensors from all devices,
        and stack them along the batch dimension.
        Args:
            data: tensor or collection of tensors to gather
            batch_dim: the batch dimension of the original tensor
        """
        # if not ddp, skip gathering and return the original data
        if self.trainer.world_size == 1:
            return apply_to_collection(data, torch.Tensor, lambda x: x.to(self.device))

        # synchronize before gathering
        torch.distributed.barrier()
        gathered_data = self.all_gather(data)

        # (r ... b ...) -> (... (r b) ...)
        rearrange_fn = (
            lambda x: x.permute(
                list(range(1, batch_dim + 1))
                + [0]
                + list(range(batch_dim + 1, x.dim()))
            )
            .reshape(*x.shape[1 : batch_dim + 1], -1, *x.shape[batch_dim + 2 :])
            .contiguous()
        )

        return apply_to_collection(gathered_data, torch.Tensor, rearrange_fn)

    def register_data_mean_std(
        self,
        mean: Union[str, float, Sequence],
        std: Union[str, float, Sequence],
        namespace: str = "data",
    ):
        """
        Register mean and std of data as tensor buffer.

        Args:
            mean: the mean of data.
            std: the std of data.
            namespace: the namespace of the registered buffer.
        """
        for k, v in [("mean", mean), ("std", std)]:
            if isinstance(v, str):
                if v.endswith(".npy"):
                    v = torch.from_numpy(np.load(v))
                elif v.endswith(".pt"):
                    v = torch.load(v)
                else:
                    raise ValueError(f"Unsupported file type {v.split('.')[-1]}.")
            else:
                v = torch.tensor(v)
            self.register_buffer(f"{namespace}_{k}", v.float().to(self.device))
