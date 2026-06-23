from pathlib import Path
import wandb
import torch

_FSDP_METADATA_FILENAME = "meta.pt"


def _is_sharded_checkpoint_dir(path: Path) -> bool:
    return path.is_dir() and (path / _FSDP_METADATA_FILENAME).is_file()


def resolve_load_dir(load_dir: Path):
    """
    Given a previous run's output directory, return (checkpoint_path, wandb_run_id).
    Checkpoint is searched under {load_dir}/checkpoints and can be either:
      - legacy single-file ckpt (last.ckpt / latest.ckpt / *.ckpt)
      - sharded FSDP directory ckpt (last.ckpt / latest.ckpt / dir/ with meta.pt)
    Wandb run ID: extracted from the most-recent {load_dir}/wandb/run-*-{id}/ directory.
    """
    load_dir = Path(load_dir)
    checkpoint_root = load_dir / "checkpoints"
    checkpoint_path = None

    if checkpoint_root.exists():
        # Prefer highest step=N checkpoint.
        def _parse_step(p: Path):
            name = p.name
            if name.startswith("step="):
                try:
                    return int(name[len("step="):].split(".")[0])
                except ValueError:
                    pass
            return -1

        step_candidates = []
        for candidate in checkpoint_root.iterdir():
            if candidate.name in ("last.ckpt", "latest.ckpt"):
                continue
            if (candidate.is_file() and candidate.suffix == ".ckpt") or _is_sharded_checkpoint_dir(candidate):
                step = _parse_step(candidate)
                if step >= 0:
                    step_candidates.append((step, candidate))

        if step_candidates:
            checkpoint_path = max(step_candidates, key=lambda x: x[0])[1]

    # Fall back to last/latest if no step checkpoint found.
    if checkpoint_path is None and checkpoint_root.exists():
        for name in ("last.ckpt", "latest.ckpt"):
            candidate = checkpoint_root / name
            if candidate.exists() and (candidate.is_file() or _is_sharded_checkpoint_dir(candidate)):
                checkpoint_path = candidate
                break

    if checkpoint_path is None and checkpoint_root.exists():
        # Final fallback: newest candidate by mtime.
        candidates = []
        for candidate in checkpoint_root.iterdir():
            if candidate.is_file() and candidate.suffix == ".ckpt":
                candidates.append(candidate)
            elif _is_sharded_checkpoint_dir(candidate):
                candidates.append(candidate)

        if candidates:
            checkpoint_path = max(candidates, key=lambda p: p.stat().st_mtime)

    wandb_run_id = None
    wandb_dir = load_dir / "wandb"
    if wandb_dir.exists():
        wandb_run_dirs = sorted(wandb_dir.glob("run-*"), key=lambda p: p.stat().st_mtime)
        if wandb_run_dirs:
            # dir names are like "run-20260310_175811-y7xr1g7s"; ID is the last segment
            wandb_run_id = wandb_run_dirs[-1].name.rsplit("-", 1)[-1]

    return checkpoint_path, wandb_run_id


def consolidate_sharded_checkpoint(src_dir: Path, dst: Path) -> None:
    """Consolidate a Lightning FSDP sharded checkpoint directory into a single .ckpt file.

    The sharded directory layout (set by Lightning's FSDPStrategy with state_dict_type="sharded"):
      meta.pt          — Lightning metadata (epoch, global_step, etc.; no tensors)
      .metadata + shard files — PyTorch DCP tensors saved as {"model.*", "optimizer_N.*"}

    The output is a Lightning-compatible .ckpt with only the model state_dict (optimizer
    states dropped), loadable by torch.load() and compatible with _load_tuned_state_dict.
    """
    try:
        from torch.distributed.checkpoint import FileSystemReader
        from torch.distributed.checkpoint.format_utils import _EmptyStateDictLoadPlanner
        from torch.distributed.checkpoint.state_dict_loader import _load_state_dict
    except ImportError as e:
        raise ImportError(
            "torch.distributed.checkpoint is required. Please upgrade to PyTorch >= 2.2."
        ) from e

    src_dir = Path(src_dir)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Load Lightning non-tensor metadata (epoch, global_step, hyperparams, etc.)
    # meta.pt contains everything except state_dict and optimizer_states (those were popped
    # by Lightning before saving to DCP).
    meta = torch.load(
        src_dir / _FSDP_METADATA_FILENAME,
        map_location="cpu",
        weights_only=False,
    )

    # Load only model weight chunks from DCP shards — skips optimizer states entirely,
    # reducing the read from the full checkpoint size to just the model weights (~1/4-1/8).
    # _EmptyStateDictLoadPlanner reconstructs: {"model": {param_name: tensor}, "optimizer_0": ...}
    # Passing keys={"model"} tells it to only read chunks belonging to the model.
    shard_files = [f for f in src_dir.iterdir() if f.name not in (_FSDP_METADATA_FILENAME, ".metadata")]
    total_mb = sum(f.stat().st_size for f in shard_files) / 1e6
    print(f"  Reading {len(shard_files)} shard files ({total_mb:.0f} MB total, loading model weights only) ...")
    dcp_data: dict = {}
    _load_state_dict(
        dcp_data,
        storage_reader=FileSystemReader(src_dir),
        planner=_EmptyStateDictLoadPlanner(keys={"model"}),
        no_dist=True,
    )
    print(f"  Model weights loaded. Writing consolidated checkpoint ...")

    # dcp_data["model"] is a flat dict {param_name: tensor} matching model.state_dict() keys.
    # Prefix with "model." so _load_tuned_state_dict(prefix="model.") can strip it correctly.
    model_sd = dcp_data.get("model", {})
    state_dict = {f"model.{k}": v for k, v in model_sd.items()}

    torch.save({**meta, "state_dict": state_dict}, dst)
    print(f"  Done.")


def strip_optimizer_states(src: str | Path, dst: str | Path) -> None:
    """Write a copy of a Lightning .ckpt containing only the state_dict.

    The result is sufficient for inference and typically 3-10x smaller.
    """
    src, dst = Path(src), Path(dst)
    ckpt = torch.load(src, map_location="cpu", weights_only=False, mmap=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": ckpt["state_dict"]}, dst)


def is_run_id(run_id: str) -> bool:
    """Check if a string is a run ID."""
    return len(run_id) == 8 and run_id.isalnum()


def version_to_int(artifact) -> int:
    """Convert versions of the form vX to X. For example, v12 to 12."""
    return int(artifact.version[1:])


def download_latest_checkpoint(run_path: str, download_dir: Path) -> Path:
    api = wandb.Api()
    run = api.run(run_path)

    # Find the latest saved model checkpoint.
    latest = None
    for artifact in run.logged_artifacts():
        if artifact.type != "model" or artifact.state != "COMMITTED":
            continue

        if latest is None or version_to_int(artifact) > version_to_int(latest):
            latest = artifact

    # Download the checkpoint.
    download_dir.mkdir(exist_ok=True, parents=True)
    root = download_dir / run_path
    latest.download(root=root)
    return root / "model.ckpt"


def extract_flow_decoder_vae_state_dict(ckpt_path: str) -> dict:
    """Return a WanVAE-compatible state_dict from raw or Lightning checkpoints.

    Supports:
    - raw VAE state dict keys (already matching WanVAE_)
    - Lightning checkpoints with nested `state_dict`
    - common module prefixes: `vae.`, `model.vae.`, `module.vae.`, `module.model.vae.`
    - `decoder_head.*` alias from WanDecoder training checkpoints
    """
    raw = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if not isinstance(raw, dict):
        return {}

    state_dict = raw.get("state_dict", raw)
    if not isinstance(state_dict, dict):
        return {}

    remapped = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("module.model.vae.", "model.vae.", "module.vae.", "vae."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
                break
        if new_key.startswith("decoder_head."):
            new_key = "decoder.head.2." + new_key[len("decoder_head."):]
        remapped[new_key] = value

    return remapped
