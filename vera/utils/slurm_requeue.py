"""
Slurm requeue utilities.

Supports:
- Native Slurm requeue via `scontrol requeue` (preferred, same JobId)
- Fallback self-resubmit via `sbatch` (new JobId)

Designed for:
- ou_bcs_low (Engaging BCS)
- DDP-safe (rank0 only)
- Hydra-compatible
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .cluster_utils import submit_slurm_job
from .distributed_utils import is_rank_zero

# -----------------------------------------------------------------------------
# Basic Slurm environment helpers
# -----------------------------------------------------------------------------


def in_slurm_allocation() -> bool:
    return "SLURM_JOB_ID" in os.environ


def slurm_job_id() -> Optional[str]:
    return os.environ.get("SLURM_JOB_ID")


def slurm_submit_dir() -> Path:
    return Path(os.environ.get("SLURM_SUBMIT_DIR", os.getcwd())).resolve()


def slurm_restart_count() -> int:
    """
    Best-effort restart count.
    NOTE: On Engaging ou_bcs_low, this is NOT always reliable.
    Prefer `scontrol show job` → Restarts=.
    """
    return int(os.environ.get("SLURM_RESTART_COUNT", "0"))


# -----------------------------------------------------------------------------
# Command-line reconstruction helpers
# -----------------------------------------------------------------------------


def read_original_python_cmdline() -> Optional[List[str]]:
    """
    Preferred: store this at submission time:
      export OKTO_PYTHON_CMDLINE="python -m vera.main ..."

    Fallback: None (we reconstruct from sys.argv).
    """
    s = os.environ.get("OKTO_PYTHON_CMDLINE")
    if not s:
        return None
    return shlex.split(s)


def ensure_requeue_override(argv: List[str], run_id: str) -> List[str]:
    """
    Remove resume/load/requeue flags and force +requeue=<run_id>.
    """

    def keep(a: str) -> bool:
        bad = ("resume=", "+requeue=", "requeue=", "load=")
        return not any(a.startswith(p) for p in bad)

    out = [a for a in argv if keep(a)]
    out.append(f"+requeue={run_id}")
    return out


def extract_hydra_args(python_cmdline: List[str]) -> List[str]:
    """
    Extract args that Hydra should see (everything after '-m module').
    """
    if "-m" in python_cmdline:
        i = python_cmdline.index("-m")
        return python_cmdline[i + 2 :]
    return python_cmdline[1:]


# -----------------------------------------------------------------------------
# Native Slurm requeue (preferred)
# -----------------------------------------------------------------------------


def slurm_native_requeue(job_id: str) -> bool:
    """
    Request native Slurm requeue.
    Returns True if command was issued successfully.
    """
    try:
        subprocess.run(
            ["scontrol", "requeue", job_id],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return False


# -----------------------------------------------------------------------------
# Self-resubmit fallback (sbatch from compute node)
# -----------------------------------------------------------------------------


def slurm_resubmit_same_job(
    cfg,
    run_id: str,
) -> None:
    """
    Submit a new job via sbatch with the same python args,
    forcing +requeue=<run_id>.

    This is a FALLBACK when native requeue is unavailable.
    """

    if not in_slurm_allocation():
        print("[REQUEUE] Not in Slurm allocation; skipping resubmit.", flush=True)
        return

    # Reconstruct python command line
    python_cmdline = read_original_python_cmdline()
    if python_cmdline is None:
        python_cmdline = [
            sys.executable,
            "-m",
            "vera.main",
            *sys.argv[1:],
        ]

    python_cmdline = ensure_requeue_override(python_cmdline, run_id)
    hydra_args = extract_hydra_args(python_cmdline)

    python_args = (
        " ".join(shlex.quote(a) for a in hydra_args) + " +_on_compute_node=True"
    )

    submit_dir = slurm_submit_dir()

    # Locate project root (same logic as run_slurm)
    project_root = submit_dir
    while not (project_root / ".git").exists():
        project_root = project_root.parent
        if project_root == Path("/"):
            raise RuntimeError("Could not locate project root for requeue resubmit.")

    print(
        f"[REQUEUE] Self-resubmitting at {datetime.now().isoformat()} "
        f"with +requeue={run_id}",
        flush=True,
    )

    submit_slurm_job(cfg, python_args, project_root)


# -----------------------------------------------------------------------------
# Unified signal handler installer
# -----------------------------------------------------------------------------


def install_slurm_requeue_handler(cfg) -> None:
    """
    Install SIGUSR1 handler that:
    1. Checkpoints (your code should do this upstream)
    2. Native requeue if possible
    3. Fallback to self-resubmit
    4. Exits cleanly

    Safe for DDP (rank0 only).
    """

    if not in_slurm_allocation():
        return

    def handler(signum, frame):
        job_id = slurm_job_id()
        host = os.environ.get("SLURMD_NODENAME", "unknown")

        print(
            f"[REQUEUE] Signal {signum} received " f"(job {job_id} on {host})",
            flush=True,
        )

        # One-shot guard (USR1 + TERM can both fire)
        guard = "OKTO_REQUEUE_DONE"
        if os.environ.get(guard) == "1":
            print("[REQUEUE] Already handled; exiting.", flush=True)
            os._exit(0)
        os.environ[guard] = "1"

        if is_rank_zero:
            # Determine stable run id
            run_id = cfg.get("requeue") or cfg.get("resume")

            # Preferred: native Slurm requeue
            if job_id and slurm_native_requeue(job_id):
                print(f"[REQUEUE] Native requeue requested for {job_id}", flush=True)
            else:
                # Fallback: self-resubmit
                if run_id:
                    print("[REQUEUE] Falling back to self-resubmit", flush=True)
                    slurm_resubmit_same_job(cfg, run_id)
                else:
                    print(
                        "[REQUEUE] No run_id available; cannot resubmit.",
                        flush=True,
                    )

        # Give stdout time to flush, then exit.
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass

        time.sleep(1)
        os._exit(0)

    signal.signal(signal.SIGUSR1, handler)

    # Optional:
    # If you want checkpoint-on-cancel behavior, uncomment,
    # BUT do NOT call scontrol requeue on SIGTERM or you can loop.
    # signal.signal(signal.SIGTERM, handler)
