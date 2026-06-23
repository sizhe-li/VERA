"""
This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research
template [repo](https://github.com/buoyancy99/research-template).
By its MIT license, you must keep the above sentence in `README.md`
and the `LICENSE` file to credit the author.

Main file for the project. This will create and run new experiments and load checkpoints from wandb.
Borrowed the wandb code from David Charatan and wandb.ai.
"""

import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from omegaconf.omegaconf import open_dict

from .utils.ckpt_utils import (
    _DOWNLOADED_ROOT,
    download_checkpoint,
    download_pretrained,
    download_vae_checkpoints,
    generate_unexisting_run_id,
    has_checkpoint,
    is_existing_run,
    is_hf_path,
    is_run_id,
    parse_load,
    wandb_to_local_path,
)
from .utils.cluster_utils import submit_slurm_job
from .utils.distributed_utils import is_rank_zero
from .utils.hydra_utils import unwrap_shortcuts
from .utils.print_utils import cyan
from .utils.slurm_requeue import install_slurm_requeue_handler


def wandb_net_probe(tag):
    try:
        socket.create_connection(("api.wandb.ai", 443), timeout=3)
        print(f"[NET OK] {tag}", flush=True)
        return True
    except Exception as e:
        print(f"[NET FAIL] {tag}: {repr(e)}", flush=True)
        return False


def _bootstrap_debug_enabled(cfg: DictConfig) -> bool:
    if bool(cfg.get("debug", False)):
        return True
    env_flag = os.environ.get("OKTO_VERBOSE_BOOTSTRAP", "").strip().lower()
    return env_flag in {"1", "true", "yes", "on"}


def _bootstrap_debug_print(cfg: DictConfig, message: str) -> None:
    if _bootstrap_debug_enabled(cfg):
        print(message, flush=True)


def _is_writable_directory(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path, delete=True):
            pass
        return True
    except OSError:
        return False


def _ensure_writable_wandb_dirs(output_dir: Path) -> list[str]:
    """Ensure wandb runtime dirs are writable without forcing global TMPDIR."""
    warnings: list[str] = []
    runtime_root = output_dir / "wandb_runtime"
    fallback_map = {
        "WANDB_DIR": runtime_root / "wandb",
        "WANDB_DATA_DIR": runtime_root / "data",
        "WANDB_CACHE_DIR": runtime_root / "cache",
    }

    for key, fallback in fallback_map.items():
        existing = os.environ.get(key, "").strip()
        if existing:
            existing_path = Path(existing).expanduser()
            if _is_writable_directory(existing_path):
                resolved = existing_path
            else:
                resolved = fallback
                warnings.append(
                    f"{key}={existing_path} is not writable; using {resolved}."
                )
        else:
            resolved = fallback
        resolved.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(resolved)

    # Only touch TMPDIR when user explicitly opts in, or when an existing TMPDIR
    # is invalid. Use node-local tmp fallback to avoid shared-fs issues.
    force_tmp = (os.environ.get("OKTO_FORCE_TMPDIR_FOR_WANDB") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    tmpdir_existing = os.environ.get("TMPDIR", "").strip()
    tmp_job = os.environ.get("SLURM_JOB_ID", "local")
    tmp_user = os.environ.get("USER", "user")
    tmp_fallback = Path(tempfile.gettempdir()) / tmp_user / f"okto_tmp_{tmp_job}"

    should_set_tmp = force_tmp or (
        len(tmpdir_existing) > 0
        and not _is_writable_directory(Path(tmpdir_existing).expanduser())
    )
    if should_set_tmp:
        if len(tmpdir_existing) > 0:
            warnings.append(
                f"TMPDIR={Path(tmpdir_existing).expanduser()} is not writable; using {tmp_fallback}."
            )
        else:
            warnings.append(
                f"OKTO_FORCE_TMPDIR_FOR_WANDB is enabled; using TMPDIR={tmp_fallback}."
            )
        tmp_fallback.mkdir(parents=True, exist_ok=True)
        os.environ["TMPDIR"] = str(tmp_fallback)

        # Keep tempfile aliases aligned only when we actively manage TMPDIR.
        tmpdir = os.environ["TMPDIR"]
        for alias in ("TEMP", "TMP"):
            alias_existing = os.environ.get(alias, "").strip()
            if alias_existing:
                alias_path = Path(alias_existing).expanduser()
                if _is_writable_directory(alias_path):
                    continue
                warnings.append(f"{alias}={alias_path} is not writable; using {tmpdir}.")
            os.environ[alias] = tmpdir

    return warnings


def run_local(cfg: DictConfig):
    # delay some imports in case they are not needed in non-local envs for submission
    from .experiments import build_experiment
    from .utils.wandb_utils import (
        OfflineWandbLogger,
        SpaceEfficientWandbLogger,
        get_wandb_run_name,
        get_wandb_tags,
    )

    # ------------------------------------------------------------------
    # PyTorch multiprocessing robustness (SLURM / shared memory)
    # ------------------------------------------------------------------
    # DataLoader uses shared-memory tensor passing between worker -> main.
    # On some nodes (small /dev/shm or many workers), allocation can fail with
    # "RuntimeError: unable to allocate shared memory(shm) for file".
    # file_system can help with /dev/shm limits but may be unstable on some shared
    # filesystems. Keep PyTorch default unless user explicitly requests file_system.
    try:
        import torch
        from omegaconf.listconfig import ListConfig

        # PyTorch 2.6+ defaults to weights_only=True in torch.load; Lightning
        # checkpoints can contain OmegaConf configs. Allowlist them so resume works.
        torch.serialization.add_safe_globals([ListConfig, DictConfig])

        sharing = (os.environ.get("OKTO_TORCH_SHARING_STRATEGY") or "").strip().lower()
        if sharing in ("", "default", "file_descriptor", "fd"):
            # Keep PyTorch default (can still hit shm limits)
            pass
        elif sharing in ("file_system", "filesystem"):
            torch.multiprocessing.set_sharing_strategy("file_system")
        elif is_rank_zero():
            print(
                f"[PyTorch] Unknown OKTO_TORCH_SHARING_STRATEGY='{sharing}', using default.",
                flush=True,
            )
    except Exception:
        # Best-effort; don't crash if torch isn't available yet
        pass

    # Drop any stale service token inherited from the submitting shell — if an
    # earlier wandb process died without clean teardown, WANDB_SERVICE leaks into
    # the shell and every sbatch from it, causing wandb.init to dial a dead port.
    os.environ.pop("WANDB_SERVICE", None)
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    # WANDB_DIR can be set in the environment to redirect wandb run/artifact dirs.

    # Get yaml names
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    cfg_choice = OmegaConf.to_container(hydra_cfg.runtime.choices)

    with open_dict(cfg):
        if cfg_choice["experiment"] is not None:
            cfg.experiment._name = cfg_choice["experiment"]
        if cfg_choice["dataset"] is not None:
            cfg.dataset._name = cfg_choice["dataset"]
        if cfg_choice["algorithm"] is not None:
            cfg.algorithm._name = cfg_choice["algorithm"]

    # Set up the output directory.
    output_dir = Path(hydra_cfg.runtime.output_dir)
    if is_rank_zero():
        print(cyan("Outputs will be saved to:"), output_dir)
        _tmp_link = output_dir.parents[1] / f".latest-run-{os.getpid()}"
        _tmp_link.symlink_to(output_dir, target_is_directory=True)
        os.replace(_tmp_link, output_dir.parents[1] / "latest-run")

    wandb_dir_warnings = _ensure_writable_wandb_dirs(output_dir)
    if is_rank_zero() and len(wandb_dir_warnings) > 0:
        print("[WANDB] Runtime directory overrides:", flush=True)
        for warning in wandb_dir_warnings:
            print(f"  - {warning}", flush=True)

    requeue = cfg.get("requeue", None)
    requeue_path = (
        f"{cfg.wandb.entity}/{cfg.wandb.project}/{requeue}" if requeue else None
    )
    requeue_has_checkpoint = requeue is not None and has_checkpoint(requeue_path)
    requeue_is_existing_run = requeue is not None and is_existing_run(requeue_path)

    # Set up logging with wandb.
    if cfg.wandb.mode != "disabled":
        # Run name: use cfg.name if set, else auto-generate from flags.
        display_name = cfg.get("name") or get_wandb_run_name(cfg, cfg_choice)
        resume = cfg.get("resume", None)
        # If resuming, merge into the existing run on wandb.
        name = (
            f"{display_name} ({output_dir.parent.name}/{output_dir.name})"
            if resume is None and not requeue_is_existing_run
            else None
        )

        if "_on_compute_node" in cfg and cfg.cluster.is_compute_node_offline:
            logger_cls = OfflineWandbLogger
        else:
            logger_cls = SpaceEfficientWandbLogger

        # --- WANDB robustness logic ---
        restart_count = int(os.environ.get("SLURM_RESTART_COUNT", "0"))
        net_ok = wandb_net_probe("before wandb init") if is_rank_zero() else True

        # Only disable when network is unavailable. Requeued runs log checkpoints too.
        disable_artifacts = not net_ok

        offline = cfg.wandb.mode != "online"
        wandb_kwargs = {
            k: v
            for k, v in OmegaConf.to_container(cfg.wandb, resolve=True).items()
            if k != "mode"
        }
        # Tags: auto-generate from flags, then merge with any wandb.tags from config.
        wandb_kwargs["tags"] = get_wandb_tags(cfg, cfg_choice) + (
            wandb_kwargs.get("tags") or []
        )

        if disable_artifacts:
            print(
                "[WANDB] Disabling artifact logging (net_ok=False). "
                f"restart_count={restart_count}",
                flush=True,
            )
            wandb_kwargs["log_model"] = False

        # offline = cfg.wandb.mode != "online"
        # wandb_kwargs = {
        #     k: v
        #     for k, v in OmegaConf.to_container(cfg.wandb, resolve=True).items()
        #     if k != "mode"
        # }

        logger = logger_cls(
            name=name,
            save_dir=str(output_dir),
            offline=offline,
            # log_model="all" if not offline else False,
            log_model=wandb_kwargs.pop("log_model", ("all" if not offline else False)),
            config=OmegaConf.to_container(cfg),
            id=resume or requeue,
            **wandb_kwargs,
        )
    else:
        logger = None

    # Load ckpt
    resume = cfg.get("resume", None)
    if requeue_has_checkpoint:
        if is_rank_zero():
            print(cyan(f"Resuming from requeued run: {requeue}"))
            download_checkpoint(
                f"{cfg.wandb.entity}/{cfg.wandb.project}/{requeue}",
                _DOWNLOADED_ROOT,
                "latest",
            )
        resume = requeue

    load = cfg.get("load", None)
    checkpoint_path = None
    load_id = None
    if resume:
        load_id = resume
    elif load:
        load_id = parse_load(load)[0]
        if load_id is None:
            checkpoint_path = load

    if load_id:
        run_path = f"{cfg.wandb.entity}/{cfg.wandb.project}/{load_id}"
        checkpoint_path = wandb_to_local_path(run_path)
    elif load and is_hf_path(load):
        checkpoint_path = download_pretrained(load)

    # launch experiment
    experiment = build_experiment(cfg, logger, checkpoint_path)
    for task in cfg.experiment.tasks:
        experiment.exec_task(task)


def run_slurm(cfg: DictConfig):
    python_args = (
        " ".join(
            [
                (
                    f"'+requeue={generate_unexisting_run_id(cfg.wandb.entity, cfg.wandb.project)}'"
                    if (arg.startswith("+requeue") and not is_run_id(arg.split("=")[1]))
                    else f"'{arg}'"
                )
                for arg in sys.argv[1:]
            ]
        )
        + " +_on_compute_node=True"
    )

    project_root = Path.cwd()
    while not (project_root / ".git").exists():
        project_root = project_root.parent
        if project_root == Path("/"):
            raise Exception("Could not find repo directory!")

    slurm_log_dir = submit_slurm_job(
        cfg,
        python_args,
        project_root,
    )

    if (
        "cluster" in cfg
        and cfg.cluster.is_compute_node_offline
        and cfg.wandb.mode == "online"
    ):
        print(
            "Job submitted to a compute node without internet. This requires manual syncing on login node."
        )
        osh_command_dir = project_root / ".wandb_osh_command_dir"

        osh_proc = None
        # if click.confirm("Do you want us to run the sync loop for you?", default=True):
        osh_proc = subprocess.Popen(["wandb-osh", "--command-dir", osh_command_dir])
        print(f"Running wandb-osh in background... PID: {osh_proc.pid}")
        print(f"To kill the sync process, run 'kill {osh_proc.pid}' in the terminal.")
        print(
            "You can manually start a sync loop later by running the following:",
            cyan(f"wandb-osh --command-dir {osh_command_dir}"),
        )

    print(
        "Once the job gets allocated and starts running, we will print a command below "
        "for you to trace the errors and outputs: (Ctrl + C to exit without waiting)"
    )
    msg = f"tail -f {slurm_log_dir}/* \n"
    try:
        while not list(slurm_log_dir.glob("*.out")) and not list(
            slurm_log_dir.glob("*.err")
        ):
            time.sleep(1)
        print(cyan("To trace the outputs and errors, run the following command:"), msg)
    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Exiting...")
        print(
            cyan(
                "To trace the outputs and errors, manually wait for the job to start and run the following command:"
            ),
            msg,
        )


@hydra.main(
    version_base=None,
    config_path="configurations",
    config_name="config",
)
def run(cfg: DictConfig):
    _bootstrap_debug_print(cfg, ">>> entered run()")

    install_slurm_requeue_handler(cfg)

    _bootstrap_debug_print(cfg, ">>> after install_slurm_requeue_handler")

    restart_count = int(os.environ.get("SLURM_RESTART_COUNT", "0"))
    if restart_count > 0:
        print(f"[REQUEUE] SLURM_RESTART_COUNT={restart_count}", flush=True)

    if "_on_compute_node" in cfg and cfg.cluster.is_compute_node_offline:
        _bootstrap_debug_print(cfg, ">>> compute node offline check")

        with open_dict(cfg):
            if cfg.cluster.is_compute_node_offline and cfg.wandb.mode == "online":
                cfg.wandb.mode = "offline"

    _bootstrap_debug_print(cfg, ">>> before name check")

    # name is optional: if not set, auto-generate from experiment/dataset/algorithm/backbone/resolution/supervision (used for wandb and SLURM job name).
    if not cfg.get("name"):
        from .utils.wandb_utils import get_wandb_run_name
        hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        cfg_choice = OmegaConf.to_container(hydra_cfg.runtime.choices)
        with open_dict(cfg):
            if cfg_choice.get("experiment"):
                cfg.experiment._name = cfg_choice["experiment"]
            if cfg_choice.get("dataset"):
                cfg.dataset._name = cfg_choice["dataset"]
            if cfg_choice.get("algorithm"):
                cfg.algorithm._name = cfg_choice["algorithm"]
            cfg.name = get_wandb_run_name(cfg, cfg_choice)

    _bootstrap_debug_print(cfg, ">>> before wandb.project set")

    if not cfg.wandb.get("entity", None):
        raise ValueError(
            "must specify wandb entity in 'configurations/config.yaml' or with command line"
            " argument 'wandb.entity=[entity]' \n An entity is your wandb user name or group"
            " name. This is used for logging. If you don't have an wandb account, please signup at https://wandb.ai/"
        )

    if cfg.wandb.project is None:
        cfg.wandb.project = str(Path(__file__).parent.name)

    runtime_output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    wandb_dir_warnings = _ensure_writable_wandb_dirs(runtime_output_dir)
    if is_rank_zero() and len(wandb_dir_warnings) > 0:
        print("[WANDB] Runtime directory overrides:", flush=True)
        for warning in wandb_dir_warnings:
            print(f"  - {warning}", flush=True)

    _bootstrap_debug_print(cfg, ">>> before download logic")

    # If resuming or loading a wandb ckpt and not on a compute node, download the checkpoint.
    resume = cfg.get("resume", None)
    load = cfg.get("load", None)
    load_id = None

    if resume and load:
        raise ValueError(
            "When resuming a wandb run with `resume=[wandb id]`, checkpoint will be loaded from the cloud"
            "and `load` should not be specified."
        )

    option = None
    if resume:
        load_id = resume
        option = "latest"
    elif load:
        load_id, option = parse_load(load)
        option = "best" if option is None else option

    if "skip_download" not in cfg:
        if load_id and "_on_compute_node" not in cfg:
            run_path = f"{cfg.wandb.entity}/{cfg.wandb.project}/{load_id}"
            # When resuming, always re-download so we get the actual latest checkpoint
            # (otherwise a cached model.ckpt from an earlier run can be reused and
            # global_step will be wrong, e.g. starting at 5k instead of 115k).
            force_redownload = resume
            download_checkpoint(
                run_path,
                _DOWNLOADED_ROOT,
                option=option,
                force_redownload=force_redownload,
            )
        if "_on_compute_node" not in cfg and is_rank_zero():
            download_vae_checkpoints(cfg)
        if load and is_hf_path(load) and "_on_compute_node" not in cfg:
            download_pretrained(load)

    _bootstrap_debug_print(cfg, ">>> before slurm/local branch")

    if "cluster" in cfg and "_on_compute_node" not in cfg:

        print(
            cyan(
                "Slurm detected, submitting to compute node instead of running locally..."
            )
        )

        _bootstrap_debug_print(cfg, ">>> calling run_slurm()")
        run_slurm(cfg)
    else:
        _bootstrap_debug_print(cfg, ">>> calling run_local()")
        run_local(cfg)


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CONFIG_DIR = os.path.join(BASE_DIR, "configurations")  # or whatever your folder is

    sys.argv = unwrap_shortcuts(sys.argv, config_path=CONFIG_DIR, config_name="config")
    run()  # pylint: disable=no-value-for-parameter
