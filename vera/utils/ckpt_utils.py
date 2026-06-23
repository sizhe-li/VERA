"""
This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research
template [repo](https://github.com/buoyancy99/research-template).
By its MIT license, you must keep the above sentence in `README.md`
and the `LICENSE` file to credit the author.
"""

import random
import string
import os
from pathlib import Path
from typing import Literal, Optional, Tuple

import wandb
from omegaconf import DictConfig

from .huggingface_utils import download_from_hf
from .logging import cyan

_USER_CKPT_CACHE_DIR = (
    Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    / "okto"
    / "checkpoints"
)

# Anchor "outputs/downloaded" to the repo root, not cwd — submission host and
# compute node run with different cwds, and a relative path makes them disagree.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DOWNLOADED_ROOT = _REPO_ROOT / "outputs" / "downloaded"


def _resolve_download_dir(download_dir: Path) -> Path:
    """Prefer the requested download dir; fallback to user-local cache if unwritable."""
    preferred = Path(download_dir)
    try:
        preferred.mkdir(exist_ok=True, parents=True)
        if os.access(preferred, os.W_OK):
            return preferred
    except OSError:
        pass

    _USER_CKPT_CACHE_DIR.mkdir(exist_ok=True, parents=True)
    print(
        cyan(
            f"Checkpoint download dir is not writable ({preferred}); "
            f"falling back to {_USER_CKPT_CACHE_DIR}."
        )
    )
    return _USER_CKPT_CACHE_DIR


def is_run_id(run_id: str) -> bool:
    """Check if a string is a run ID."""
    return len(run_id) == 8 and run_id.isalnum()


def generate_run_id() -> str:
    """Generate a random 8-character alphanumeric string."""
    chars = string.ascii_lowercase + string.digits
    return "".join(random.choice(chars) for _ in range(8))


def generate_unexisting_run_id(entity: str, project: str) -> str:
    """Generate a random 8-character alphanumeric string that does not exist in the project."""
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    existing_ids = {run.id for run in runs}
    while True:
        run_id = generate_run_id()
        if run_id not in existing_ids:
            return run_id


def parse_load(load: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse load into run_id and download option.
    (for load=xxxxxxxx in configurations)
    - If load_id is a run_id, return the run_id and None.
    - If load_id is of the form run_id:option, return run_id and option.
    - Otherwise, return None, None.
    """
    split = load.split(":")
    if 1 <= len(split) <= 2 and is_run_id(split[0]):
        return split[0], split[1] if len(split) == 2 else None
    return None, None


def version_to_int(artifact) -> int:
    """Convert versions of the form vX to X. For example, v12 to 12."""
    return int(artifact.version[1:])


def is_existing_run(run_path: str) -> bool:
    """Check if a run exists."""
    api = wandb.Api()
    try:
        _ = api.run(run_path)
        return True
    except wandb.errors.CommError:
        return False
    return False


def has_checkpoint(run_path: str) -> bool:
    """Check if a run has a committed model checkpoint."""
    api = wandb.Api()
    try:
        run = api.run(run_path)
        for artifact in run.logged_artifacts():
            if artifact.type == "model" and artifact.state == "COMMITTED":
                return True
        return False
    except wandb.errors.CommError:
        return False
    return False


def download_checkpoint(
    run_path: str,
    download_dir: Path,
    option: Literal["latest", "best"] = "latest",
    return_config: bool = False,
    force_redownload: bool = False,
) -> Path:
    api = wandb.Api()
    run = api.run(run_path)

    # Find the latest saved model checkpoint.
    # When option is "latest", pick the artifact with the highest version so we
    # respect the true latest (not just the first match in iteration order).
    checkpoint = None
    candidates = []
    for artifact in run.logged_artifacts():
        if artifact.type != "model" or artifact.state != "COMMITTED":
            continue
        if option in artifact.aliases or option == artifact.version:
            candidates.append(artifact)
    if option == "latest" and len(candidates) > 1:
        # Prefer highest version (e.g. v2 over v1).
        try:
            checkpoint = max(candidates, key=lambda a: version_to_int(a))
        except (ValueError, IndexError):
            checkpoint = candidates[-1] if candidates else None
    elif candidates:
        checkpoint = candidates[0]

    if checkpoint is None:
        # Fallback: pick the highest-version committed model artifact from the
        # run, ignoring aliases. Covers runs where no "latest"/"best" alias
        # was ever assigned (e.g. 2seo56q5).
        all_models = [
            a for a in run.logged_artifacts()
            if a.type == "model" and a.state == "COMMITTED"
        ]
        if all_models:
            try:
                checkpoint = max(all_models, key=lambda a: version_to_int(a))
            except (ValueError, IndexError):
                checkpoint = all_models[-1]
            print(f"No '{option}' alias found; falling back to {checkpoint.name}:{checkpoint.version}")
    if checkpoint is None:
        print(f"No {option} model checkpoint found in {run_path}.")

    # Download the checkpoint.
    effective_download_dir = _resolve_download_dir(download_dir)
    root = effective_download_dir / run_path

    if force_redownload or not (root / "model.ckpt").exists():
        try:
            checkpoint.download(root=root)
        except PermissionError:
            # Retry once in the user-local cache in case a shared workspace path
            # is mounted read-only for this user.
            fallback_root = _USER_CKPT_CACHE_DIR / run_path
            if fallback_root != root:
                print(
                    cyan(
                        "Permission denied while downloading checkpoint; "
                        f"retrying in {_USER_CKPT_CACHE_DIR}."
                    )
                )
                checkpoint.download(root=fallback_root)
                root = fallback_root
            else:
                raise

    if not return_config:
        return root / "model.ckpt"
    else:
        return root / "model.ckpt", run.config


def download_pretrained(
    name: str,
) -> str:
    """
    Download a pretrained model from the DFoT Hugging Face model hub.
    Set is_full to True to download the full model
    (including optimizer states and non-EMA weights).
    """
    prefix, name = name.split(":")
    download_from_hf(filename="config.json")
    return download_from_hf(filename=f"{prefix}_models/{name}")


def is_wandb_run_path(run_path: str) -> bool:
    split = run_path.split("/")
    return len(split) == 3 and is_run_id(split[-1])


def is_hf_path(path: str) -> bool:
    return path.startswith("pretrained:") or path.startswith("full:")


def download_vae_checkpoints(
    cfg: DictConfig,
):
    pretrained_paths = []
    vae = cfg.algorithm.get("vae", None)
    if vae and vae.get("pretrained_path", None):
        pretrained_paths.append(vae.pretrained_path)

    pretrained_path = cfg.algorithm.get("pretrained_path", None)
    if pretrained_path:
        pretrained_paths.append(pretrained_path)

    wandb_pretrained_paths = [
        path for path in pretrained_paths if is_wandb_run_path(path)
    ]
    hf_pretrained_paths = [path for path in pretrained_paths if is_hf_path(path)]

    for path in wandb_pretrained_paths:
        print(cyan("Downloading pretrained VAE from Wandb:"), path)
        download_checkpoint(path, _DOWNLOADED_ROOT, option="best")

    for path in hf_pretrained_paths:
        print(cyan("Downloading pretrained VAE from Hugging Face:"), path)
        download_pretrained(path)


def wandb_to_local_path(run_path: str) -> Path:
    return _DOWNLOADED_ROOT / run_path / "model.ckpt"
