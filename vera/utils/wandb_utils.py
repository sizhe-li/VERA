"""
This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research
template [repo](https://github.com/buoyancy99/research-template).
By its MIT license, you must keep the above sentence in `README.md`
and the `LICENSE` file to credit the author.
"""

import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Tuple,
    Union,
)

import wandb
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.loggers.wandb import (
    ModelCheckpoint,
    Tensor,
    WandbLogger,
    _scan_checkpoints,
)
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from tqdm import tqdm
from typing_extensions import override
from wandb.apis.public.runs import Run
from wandb_osh.hooks import TriggerWandbSyncHook

from vera.utils.print_utils import cyan

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from wandb.sdk.lib import RunDisabled
    from wandb.wandb_run import Run


def _cfg_select(cfg: "DictConfig", key: str, default: Any = None) -> Any:
    """Select a nested config key with a default if missing (OmegaConf.select has no default in some versions)."""
    from omegaconf import OmegaConf

    try:
        out = OmegaConf.select(cfg, key)
        return out if out is not None else default
    except (AttributeError, KeyError, TypeError):
        return default


def _resolution_from_cfg(cfg: "DictConfig") -> Any:
    """Resolution from dataset.resolution or dataset.camera.image_size (for action/jacobian configs)."""
    res = _cfg_select(cfg, "dataset.resolution", None)
    if res is not None:
        return res
    img_size = _cfg_select(cfg, "dataset.camera.image_size", None)
    if img_size is not None and len(img_size) > 0:
        return img_size[0]
    return None


def _model_size_label(cfg: "DictConfig") -> str:
    """
    Size label for algorithm.model (e.g. DPT): from neck_preset (S/M/L/XL)
    or backbone_preset (small->S, base->B, large->L, giant->XL).
    """
    neck = _cfg_select(cfg, "algorithm.model.neck_preset", None)
    if neck is not None:
        return str(neck)
    preset = _cfg_select(cfg, "algorithm.model.backbone_preset", None)
    if preset is not None:
        preset_map = {"small": "S", "base": "B", "large": "L", "giant": "XL"}
        return preset_map.get(str(preset).lower(), str(preset))
    return ""


def _ablation_tokens(cfg: "DictConfig") -> List[str]:
    tokens: List[str] = []

    backbone_preset = _cfg_select(cfg, "algorithm.model.backbone_preset", None)
    if backbone_preset is not None:
        tokens.append(f"bb{str(backbone_preset).lower()}")

    train_batch_size = _cfg_select(cfg, "experiment.training.batch_size", None)
    if train_batch_size is not None:
        tokens.append(f"bs{int(train_batch_size)}")

    num_workers = _cfg_select(cfg, "experiment.training.data.num_workers", None)
    if num_workers is not None:
        tokens.append(f"nw{int(num_workers)}")

    prefetch_factor = _cfg_select(cfg, "experiment.training.data.prefetch_factor", None)
    if prefetch_factor is not None:
        tokens.append(f"pf{int(prefetch_factor)}")

    cache_prefixes = {
        "experiment.training.data.cache.video": "cv",
        "experiment.training.data.cache.zarr": "cz",
        "experiment.training.data.cache.droid_h5": "ch",
    }
    for key, prefix in cache_prefixes.items():
        value = _cfg_select(cfg, key, None)
        if value is not None:
            tokens.append(f"{prefix}{int(value)}")

    return tokens


def get_wandb_run_name(cfg: "DictConfig", choices: Optional[Dict[str, Any]] = None) -> str:
    """
    Generate a concise wandb run name from config flags (no parsing of wandb.name).
    Supports both backbone-based algos (DFoT: algorithm.backbone) and model-based algos (Jacobian: algorithm.model).
    Pattern: {experiment_short}/{dataset_short}/{model_or_backbone}_{size?}_{resolution?}/{supervision?}[_linearize_N]
    """
    choices = choices or {}
    exp = choices.get("experiment") or _cfg_select(cfg, "experiment._name", "exp")
    dset = choices.get("dataset") or _cfg_select(cfg, "dataset._name", "dataset")
    algo = choices.get("algorithm") or _cfg_select(cfg, "algorithm._name", "algo")

    # Short experiment slug (e.g. motion_policy_learning -> motion_policy, jacobian_learning -> jacobian)
    exp_short = str(exp).replace("_learning", "").replace("_generation", "").replace("_preprocessing", "")

    # Short dataset slug: strip experiment-like prefix and suffix
    dset_str = str(dset)
    for prefix in ("motion_policy_", "video_", "action_"):
        if dset_str.startswith(prefix):
            dset_str = dset_str[len(prefix) :]
            break
    for suffix in ("_engaging", "_base", "_noise", "_srg1", "_full"):
        if dset_str.endswith(suffix):
            dset_str = dset_str[: -len(suffix)]
            break
    dset_short = dset_str or "dataset"

    resolution = _resolution_from_cfg(cfg)

    # Third segment: backbone (DFoT) or model (Jacobian/DPT)
    backbone = _cfg_select(cfg, "algorithm.backbone.name", None)
    model_name = _cfg_select(cfg, "algorithm.model.name", None)

    if backbone is not None:
        # Backbone-based (e.g. dfot_motion_policy_joint + u_net3d_lester/S)
        network_size = _cfg_select(cfg, "algorithm.backbone.network_size", None)
        size_label = ""
        if network_size is not None:
            size_map = {8: "S", 16: "N", 32: "L", 48: "XL"}
            size_label = size_map.get(int(network_size), str(network_size))
        model_part = backbone
        if size_label:
            model_part = f"{backbone}_{size_label}"
        if resolution is not None:
            model_part = f"{model_part}_{resolution}"
    elif model_name is not None:
        # Model-based (e.g. image_jacobian + algorithm/model=robomimic_lift, optional @DPT/S)
        size_label = _model_size_label(cfg)
        model_part = str(model_name)
        if size_label:
            model_part = f"{model_part}_{size_label}"
        if resolution is not None:
            model_part = f"{model_part}_{resolution}"
    else:
        model_part = algo or "algo"
        if resolution is not None:
            model_part = f"{model_part}_{resolution}"

    # Optional supervision (e.g. flow+tracks -> flow+track for display)
    supervision = _cfg_select(cfg, "algorithm.supervision", None)
    supervision_short = None
    if supervision:
        supervision_short = str(supervision).replace("+tracks", "+track")

    parts = [exp_short, dset_short, model_part]
    if supervision_short:
        parts.append(supervision_short)

    # Optional linearize suffix (e.g. jacobian with dataset.sampling.linearize_time_length=1)
    linearize = _cfg_select(cfg, "dataset.sampling.linearize_time_length", None)
    if linearize is not None and linearize != 0:
        parts.append(f"linearize_{linearize}")

    ablation_tokens = _ablation_tokens(cfg)
    if ablation_tokens:
        parts.append("abl_" + "-".join(ablation_tokens))

    return "/".join(parts)


def get_wandb_tags(cfg: "DictConfig", choices: Optional[Dict[str, Any]] = None) -> List[str]:
    """Generate wandb tags from config flags for filtering and grouping."""
    choices = choices or {}
    tags: List[str] = []
    exp = choices.get("experiment") or _cfg_select(cfg, "experiment._name", None)
    if exp:
        tags.append(str(exp))
    dset = choices.get("dataset") or _cfg_select(cfg, "dataset._name", None)
    if dset:
        tags.append(str(dset))
    algo = choices.get("algorithm") or _cfg_select(cfg, "algorithm._name", None)
    if algo:
        tags.append(str(algo))
    backbone = _cfg_select(cfg, "algorithm.backbone.name", None)
    if backbone:
        tags.append(str(backbone))
    model_name = _cfg_select(cfg, "algorithm.model.name", None)
    if model_name:
        tags.append(str(model_name))
    model_choice = choices.get("model", None)
    if model_choice:
        tags.append(str(model_choice))
    resolution = _cfg_select(cfg, "dataset.resolution", None)
    if resolution is None:
        img_size = _cfg_select(cfg, "dataset.camera.image_size", None)
        if img_size is not None and len(img_size) > 0:
            resolution = img_size[0]
    if resolution is not None:
        tags.append(f"res{resolution}")
    supervision = _cfg_select(cfg, "algorithm.supervision", None)
    if supervision:
        tags.append(str(supervision))
    return tags


class SpaceEfficientWandbLogger(WandbLogger):
    """
    A wandb logger that by default overrides artifacts to save space, instead of creating new version.
    A variable expiration_days can be set to control how long older versions of artifacts are kept.
    By default, the latest version is kept indefinitely, while older versions are kept for 1 days.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        save_dir: _PATH = ".",
        version: Optional[str] = None,
        offline: bool = False,
        dir: Optional[_PATH] = None,
        id: Optional[str] = None,
        anonymous: Optional[bool] = None,
        project: Optional[str] = None,
        log_model: Union[Literal["all"], bool] = False,
        experiment: Union["Run", "RunDisabled", None] = None,
        prefix: str = "",
        checkpoint_name: Optional[str] = None,
        expiration_days: Optional[int] = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            save_dir=save_dir,
            version=version,
            offline=offline,
            dir=dir,
            id=id,
            anonymous=anonymous,
            project=project,
            log_model=log_model,
            experiment=experiment,
            prefix=prefix,
            checkpoint_name=checkpoint_name,
            **kwargs,
        )
        self.expiration_days = expiration_days
        self._last_artifacts = []

    @staticmethod
    def _resolve_checkpoint_artifact_name(experiment: Any, fallback: str = "model-run") -> str:
        run_id = getattr(experiment, "id", None)
        if callable(run_id):
            try:
                run_id = run_id()
            except TypeError:
                run_id = None
        if not isinstance(run_id, str) or len(run_id.strip()) == 0:
            run_id = fallback
        # W&B allows only alnum, dashes, underscores and dots in artifact names.
        safe_id = re.sub(r"[^A-Za-z0-9._-]+", "-", run_id).strip("-")
        if len(safe_id) == 0:
            safe_id = fallback
        return f"model-{safe_id}"

    @rank_zero_only
    def _scan_and_log_checkpoints(self, checkpoint_callback: ModelCheckpoint) -> None:
        import wandb

        # get checkpoints to be saved with associated score
        checkpoints = _scan_checkpoints(checkpoint_callback, self._logged_model_time)

        # log iteratively all new checkpoints
        artifacts = []
        for t, p, s, tag in checkpoints:
            metadata = {
                "score": s.item() if isinstance(s, Tensor) else s,
                "original_filename": Path(p).name,
                checkpoint_callback.__class__.__name__: {
                    k: getattr(checkpoint_callback, k)
                    for k in [
                        "monitor",
                        "mode",
                        "save_last",
                        "save_top_k",
                        "save_weights_only",
                        "_every_n_train_steps",
                    ]
                    # ensure it does not break if `ModelCheckpoint` args change
                    if hasattr(checkpoint_callback, k)
                },
            }
            if not self._checkpoint_name:
                self._checkpoint_name = self._resolve_checkpoint_artifact_name(
                    self.experiment
                )

            try:
                artifact = wandb.Artifact(
                    name=self._checkpoint_name, type="model", metadata=metadata
                )
            except ValueError as exc:
                print(
                    (
                        "[WANDB] Failed to create checkpoint artifact "
                        f"with name '{self._checkpoint_name}': {exc}. "
                        "Skipping artifact upload for this checkpoint."
                    ),
                    flush=True,
                )
                continue
            try:
                artifact.add_file(p, name="model.ckpt")
            except (PermissionError, OSError) as exc:
                print(
                    (
                        "[WANDB] Failed to stage checkpoint artifact "
                        f"from {p}: {exc}. Skipping artifact upload for this checkpoint."
                    ),
                    flush=True,
                )
                continue
            aliases = (
                ["latest", "best"]
                if p == checkpoint_callback.best_model_path
                else ["latest"]
            )
            self.experiment.log_artifact(artifact, aliases=aliases)
            # remember logged models - timestamp needed in case filename didn't change (lastkckpt or custom name)
            self._logged_model_time[p] = t
            artifacts.append(artifact)

        for artifact in self._last_artifacts:
            if not self._offline:
                artifact.wait()
            artifact.ttl = timedelta(days=self.expiration_days)
            artifact.save()
        self._last_artifacts = artifacts


class OfflineWandbLogger(SpaceEfficientWandbLogger):
    """
    Wraps WandbLogger to trigger offline sync hook occasionally.
    This is useful when running on slurm clusters, many of which
    only has internet on login nodes, not compute nodes.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        save_dir: _PATH = ".",
        version: Optional[str] = None,
        offline: bool = False,
        dir: Optional[_PATH] = None,
        id: Optional[str] = None,
        anonymous: Optional[bool] = None,
        project: Optional[str] = None,
        log_model: Union[Literal["all"], bool] = False,
        experiment: Union["Run", "RunDisabled", None] = None,
        prefix: str = "",
        checkpoint_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            save_dir=save_dir,
            version=version,
            offline=False,
            dir=dir,
            id=id,
            anonymous=anonymous,
            project=project,
            log_model=log_model,
            experiment=experiment,
            prefix=prefix,
            checkpoint_name=checkpoint_name,
            **kwargs,
        )
        self._offline = offline
        communication_dir = Path(".wandb_osh_command_dir")
        communication_dir.mkdir(parents=True, exist_ok=True)
        self.trigger_sync = TriggerWandbSyncHook(communication_dir)
        self.last_sync_time = 0.0
        self.min_sync_interval = 60
        self.wandb_dir = os.path.join(self._save_dir, "wandb/latest-run")

    @override
    @rank_zero_only
    def log_metrics(
        self, metrics: Mapping[str, float], step: Optional[int] = None
    ) -> None:
        out = super().log_metrics(metrics, step)
        if time.time() - self.last_sync_time > self.min_sync_interval:
            self.trigger_sync(self.wandb_dir)
            self.last_sync_time = time.time()
        return out


def cleanup_project(
    entity: str,
    project: str,
    log_folder: Optional[str] = None,
    ignore_ttl: bool = False,
):
    """
    cleanup the project by applying TTL policy to the model artifacts
    """
    num_deleted = 0
    total_size = 0
    log_file = Path(log_folder) / f"{datetime.now().strftime('%Y-%m-%d')}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "w") as f:
        f.write(f"[Cleanup] {entity}/{project}\n\n")
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}", order="-created_at")
    tbar = tqdm(runs)
    for run in tbar:
        versions, size = cleanup_run(run, ignore_ttl)
        num_deleted += len(versions)
        total_size += size
        tbar.set_postfix(
            num_deleted=num_deleted,
            saved=f"{total_size:.2f} GB",
        )
        if len(versions) > 0:
            with open(log_file, "a") as f:
                f.write(f"{run.id}\n{run.name}\n{versions}\n{size:.2f} GB\n\n")

    print(cyan(f"Deleted {num_deleted} models, saved {total_size:.2f} GB"))


def cleanup_run(run: Run, ignore_ttl: bool = False) -> Tuple[List[str], float]:
    """
    cleanup the models that are not best or latest and have expired
    Returns: size of the deleted artifacts (in GB)
    """
    size = 0
    versions = []
    for artifact in run.logged_artifacts():
        if (
            artifact.type == "model"
            and artifact.state == "COMMITTED"
            and (
                "best" not in artifact.aliases
                and "latest" not in artifact.aliases
                and "backup" not in artifact.aliases
            )
            and (artifact.ttl is not None or ignore_ttl)
        ):
            should_delete = True
            if not ignore_ttl:
                created_at = datetime.strptime(
                    artifact.created_at, "%Y-%m-%dT%H:%M:%SZ"
                )
                current_time = datetime.now()
                should_delete = current_time - created_at > artifact.ttl
            if should_delete:
                versions.append(artifact.version)
                size += artifact.size / 1024**3
                artifact.delete()
    return versions, size


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ignore-ttl",
        action="store_true",
        help="Ignore TTL policy and delete non-best, non-latest models",
    )
    args = parser.parse_args()
    cleanup_project(
        "scene-representation-group",
        "video_diffusion",
        "wandb_cleanup",
        ignore_ttl=args.ignore_ttl,
    )
