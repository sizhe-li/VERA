"""VeraServerConfig — the metadata the server declares on connect, + provenance.

Extends DreamZero's PolicyServerConfig (image/camera/action_space) with: the two-stage
model identity (planner + idm), control geometry (action_horizon, control_dt, gripper_is_raw),
and PROVENANCE (git head/diff/hostname/argv/run_dir) — the gap that made DreamZero rollouts
unrecoverable. See SERVER_PROTOCOL_SPEC.md §3 and SERVER_BUG_ANALYSIS.md.
"""
from __future__ import annotations

import dataclasses
import hashlib
import socket
import subprocess
import sys
from typing import List, Optional, Tuple

PROTOCOL_VERSION = 1


@dataclasses.dataclass
class VeraServerConfig:
    # --- observation contract (follows the OLD vera naming: width-concat rgb + view meta) ---
    image_resolution: Optional[Tuple[int, int]]      # (H, per_view_W); client resizes per view before width-concat
    view_keys: List[str]                             # e.g. ["ext1","ext2","wrist"] (DROID) — order of the width-concat
    view_widths: List[int]                           # per-view widths summing to rgb width
    proprio_keys: List[str]                          # e.g. ["q_robot","eef_pos","eef_quat","gripper_qpos"]
    needs_prompt: bool
    needs_session_id: bool = True
    # --- action / control contract ---
    action_space: str = "joint_position"             # joint_position|joint_velocity|cartesian_position|...
    action_horizon: int = 10                         # H — actions per infer (controller plays all H)
    context_frames: int = 9                          # pixel context frames the WAN needs (1+(N-1)*stride); client sends this many
    action_dim: int = 8                              # D — used dims (e.g. 7 joints + 1 gripper)
    control_dt: float = 1.0 / 15.0                   # s/action; H*control_dt = motion bought per call
    gripper_is_raw: bool = True                      # server sends raw float; client binarizes >0.5 -> close
    # --- action denormalization contract (the denorm-must-match-training rule) ---
    actions_already_metric: bool = False             # True: server already emits physical du; client MUST NOT re-scale
    action_abs_scale: List[float] = dataclasses.field(default_factory=list)  # per-dim denorm scale (only if NOT already metric)
    gripper_dim_index: int = -1                      # which action dim is the gripper
    embodiment: str = "droid"                        # selects the client-side action adapter
    # --- model identity (two-stage) ---
    planner_model: str = ""                          # WAN planner name + ckpt id
    idm_model: str = ""                              # jacobian/idm name + ckpt id
    is_causal: bool = False                          # AR (KV-cache) vs bidirectional; informational
    # --- provenance (filled by from_runtime) ---
    protocol_version: int = PROTOCOL_VERSION
    git_head: str = ""
    git_dirty: bool = False
    git_diff_sha: str = ""
    hostname: str = ""
    argv: List[str] = dataclasses.field(default_factory=list)
    run_dir: str = ""

    @classmethod
    def from_runtime(cls, *, repo_dir: str, run_dir: str, **fields) -> "VeraServerConfig":
        """Build a config and stamp git/host/argv provenance from the running process."""
        def _git(*args):
            try:
                return subprocess.run(
                    ["git", "-C", repo_dir, *args],
                    capture_output=True, text=True, timeout=10,
                ).stdout.strip()
            except Exception:
                return ""

        head = _git("rev-parse", "HEAD")
        diff = _git("diff")
        cfg = cls(run_dir=run_dir, **fields)
        cfg.git_head = head
        cfg.git_dirty = bool(diff)
        cfg.git_diff_sha = hashlib.sha256(diff.encode()).hexdigest()[:16] if diff else ""
        cfg.hostname = socket.gethostname()
        cfg.argv = list(sys.argv)
        return cfg
