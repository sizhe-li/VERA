"""Build a MotionPolicy for PushT (sim) — DFoT planner + Jacobian IDM, served over the protocol.

This is the SERVER side of the PushT client-server path. It constructs the exact two-stage policy
the in-process notebook proved (examples/pusht_dfot_stack.ipynb, cells 2-4): a DFoT motion planner
(run dvxixf6d) + a Jacobian inverse-dynamics model (run j1j59qzz), with the canonical PushT recipe
(motion_plan_scale=30, action_scale=8, lam=5, chunk/exec=3). Both checkpoints are LOCAL files with
config sidecars next to them — no wandb / okto access at runtime, no WAN prefix patch (DFoT needs
none). The policy is the gripperless base ``MotionPolicy`` (single view, 2D position-delta du).

    python -m vera.server.start_vera_server --embodiment pusht --port 8820 --vis-port 8821

The PushT runner (vera.env_runner.pusht_runner) integrates the returned du and applies its own
``actions_vel_scale`` to reach env-pixel-velocity units, so the policy emits the raw solved du
(actions_already_metric=False) — the client must NOT re-scale.
"""

import logging
import os
from pathlib import Path

import torch

from vera.policy.motion_policy import MotionPolicy
from vera.policy.motion_policy_types import (
    ControllerCfg,
    DynamicsCfg,
    MotionPolicyCfg,
    PlannerCfg,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

# ── checkpoints (env-overridable; defaults are the proven local PushT runs) ──────────
#   planner = DFoT (dvxixf6d): model.ckpt + run_config.yaml sidecar.
#   IDM     = Jacobian (j1j59qzz): model.ckpt + config.yaml sidecar.
DEFAULT_PLANNER_CKPT = os.environ.get(
    "VERA_PUSHT_PLANNER_CKPT",
    "/path/to/data/jacobian/pusht_run_checkpoints/your-wandb-entity/jacobian/dvxixf6d/model.ckpt",
)
DEFAULT_DYNAMICS_CKPT = os.environ.get(
    "VERA_PUSHT_DYNAMICS_CKPT",
    "/path/to/data/jacobian/pusht_run_checkpoints/your-wandb-entity/jacobian/j1j59qzz/model.ckpt",
)

# adapter_factory references mod.DEFAULT_DYNAMICS_RUN_ID for the idm_model handshake label.
DEFAULT_DYNAMICS_RUN_ID = "j1j59qzz"

# ── canonical PushT recipe (verbatim from the in-process notebook) ──────────────────
MOTION_PLAN_SCALE = float(os.environ.get("VERA_PUSHT_MOTION_PLAN_SCALE", "30.0"))
ACTION_SCALE = float(os.environ.get("VERA_PUSHT_ACTION_SCALE", "8.0"))
LAM = float(os.environ.get("VERA_PUSHT_LAM", "5.0"))
CLIP_DU = float(os.environ.get("VERA_PUSHT_CLIP_DU", "10000.0"))
SMOOTHING = float(os.environ.get("VERA_PUSHT_SMOOTHING", "0.0"))
ACTION_CHUNK_HORIZON = int(os.environ.get("VERA_PUSHT_ACTION_CHUNK_HORIZON", "3"))
N_ACTION_STEPS = int(os.environ.get("VERA_PUSHT_N_ACTION_STEPS", "3"))
DEFAULT_PLANNER_STEPS = int(os.environ.get("VERA_PUSHT_PLANNER_STEPS", "100"))


def build_policy(
    device: torch.device,
    *,
    algo_config_path: str | None = None,   # unused for PushT (kept for factory signature parity)
    dynamics_run_id: str | None = None,    # informational label only; the IDM is a local ckpt
    sample_steps: int | None = None,       # factory kwarg name
    sample_steps_override: int | None = None,  # start_server CLI kwarg name
    planner_ckpt: str | None = None,
    dynamics_ckpt: str | None = None,
    **_ignored,
) -> MotionPolicy:
    """Construct the proven two-stage PushT policy (DFoT planner + Jacobian IDM).

    ``algo_config_path`` / ``dynamics_run_id`` are accepted for factory-signature parity but the
    PushT path uses LOCAL checkpoints (no wandb run resolution). ``sample_steps`` overrides the
    DFoT diffusion sampling timesteps (default 100). Extra factory kwargs are ignored.
    """
    planner_ckpt = planner_ckpt or DEFAULT_PLANNER_CKPT
    dynamics_ckpt = dynamics_ckpt or DEFAULT_DYNAMICS_CKPT
    steps = sample_steps or sample_steps_override or DEFAULT_PLANNER_STEPS

    for p in (planner_ckpt, dynamics_ckpt):
        if not Path(p).exists():
            raise FileNotFoundError(f"PushT checkpoint missing: {p}")
        sidecars = [Path(p).parent / n for n in ("config.yaml", "run_config.yaml")]
        if not any(s.exists() for s in sidecars):
            raise FileNotFoundError(f"PushT config sidecar missing next to {p}")

    logging.info(
        "PUSHT policy: DFoT planner=%s (steps=%d) + Jacobian IDM=%s | "
        "scale=%.1f action_scale=%.1f lam=%.1f H=%d exec=%d",
        planner_ckpt, steps, dynamics_ckpt,
        MOTION_PLAN_SCALE, ACTION_SCALE, LAM, ACTION_CHUNK_HORIZON, N_ACTION_STEPS,
    )

    planner_cfg = PlannerCfg(
        ckpt_path=planner_ckpt,
        diffusion_sampling_timesteps=steps,
    )
    dynamics_cfg = DynamicsCfg(ckpt_path=dynamics_ckpt)
    controller_cfg = ControllerCfg(
        lam=LAM, action_scale=ACTION_SCALE, clip_du=CLIP_DU,
        smoothing=SMOOTHING, weight_flow_thresh=0.0,
    )
    cfg = MotionPolicyCfg(
        name="motion_policy",
        motion_planner=planner_cfg,
        dynamics_model=dynamics_cfg,
        controller=controller_cfg,
        motion_plan_scale=MOTION_PLAN_SCALE,
        action_chunk_horizon=ACTION_CHUNK_HORIZON,
        n_action_steps=N_ACTION_STEPS,
        verbose=True,
    )
    # Server hot-path: keep raw artifact dumping off unless explicitly enabled later.
    setattr(cfg, "save_artifacts", False)
    return MotionPolicy(cfg, device=device)
