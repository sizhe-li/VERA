"""Per-embodiment adapter factory + swap-to-omni hook.

``make_adapter(embodiment=...)`` builds the production two-stage policy for one embodiment
(DROID FR3 or Allegro hand), reads its on-the-wire metadata, and wraps it in a
``VeraPolicyAdapter`` ready to hand to ``WebsocketPolicyServer``. Each embodiment carries its
own WAN planner ckpt (via ``algo_config_path``) and jacobian/IDM ckpt (via ``dynamics_run_id``).

Swapping the planner to the omni model is **one argument**: pass ``algo_config_path=<omni
algo_config.yaml>`` (and, if it ships its own IDM, ``dynamics_run_id=<omni-idm>``). Nothing
else changes — the adapter/transport are model-agnostic. See SERVER_PROTOCOL_SPEC.md §7.

    from vera.server.protocol.adapter_factory import make_adapter
    adapter = make_adapter("droid", text="pick up the red block")          # per-embodiment WAN
    adapter = make_adapter("droid", algo_config_path=OMNI_ALGO_CONFIG)      # <- swap to omni
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Dict, List, Optional

from vera.server.protocol.server_config import VeraServerConfig
from vera.server.protocol.vera_policy_adapter import VeraPolicyAdapter

# Repo root (…/third_party/vera) — for provenance stamping.
_REPO_DIR = str(Path(__file__).resolve().parents[3])


@dataclasses.dataclass(frozen=True)
class _EmbodimentSpec:
    builder_module: str          # module exposing build_policy(device, ...) + _patch fn
    view_keys: List[str]         # width-concat order advertised to the controller
    proprio_keys: List[str]      # informational: what proprio the controller should send
    action_space: str            # default action_space label (overridden by wire meta if present)
    control_dt: float            # s/action — controller cadence (15 Hz -> 1/15)
    is_causal: bool              # AR (KV-cache) vs bidirectional planner — informational
    needs_patch: bool = True     # call the module's WAN-prefix patch before building


# DROID FR3 (native du) and Allegro hand. Both build_policy() share the same core kwargs and
# both policies expose get_wire_metadata() + predict_action_chunk().
_EMBODIMENTS: Dict[str, _EmbodimentSpec] = {
    "droid": _EmbodimentSpec(
        builder_module="vera.server.start_server_droid",
        view_keys=["varied_1", "varied_2", "hand"],
        proprio_keys=["q_robot", "eef_pos", "eef_quat", "gripper_qpos"],
        action_space="cartesian_delta",
        control_dt=1.0 / 15.0,
        is_causal=False,
    ),
    "allegro": _EmbodimentSpec(
        builder_module="vera.server.start_server_allegro",
        view_keys=[f"camera_{i}" for i in range(12)],
        proprio_keys=["q_robot"],
        action_space="joint_position",
        control_dt=1.0 / 15.0,
        is_causal=False,
    ),
    # MIMICGEN sim: omni WAN planner + mimicgen Jacobian (MotionPolicyGripper). 2 views
    # (agentview + wrist), context padded 256->576 black-right by the policy. eef-delta actions.
    "mimicgen": _EmbodimentSpec(
        builder_module="vera.server.start_server_mimicgen",
        view_keys=["agentview_image", "robot0_eye_in_hand_image"],
        proprio_keys=["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
        action_space="eef_delta",
        control_dt=1.0 / 20.0,
        is_causal=False,
    ),
    # PUSHT sim: DFoT planner + Jacobian IDM (gripperless base MotionPolicy). Single image view,
    # 2D position-delta du (no gripper -> gripper_dim_index=-1). LOCAL ckpts, no WAN prefix patch
    # (needs_patch=False). The runner integrates du -> position-velocity (actions_vel_scale).
    "pusht": _EmbodimentSpec(
        builder_module="vera.server.start_server_pusht",
        view_keys=["image"],
        proprio_keys=[],
        action_space="pos",
        control_dt=1.0 / 10.0,
        is_causal=False,
        needs_patch=False,
    ),
}


def available_embodiments() -> List[str]:
    return sorted(_EMBODIMENTS)


def _build_server_config(
    policy: Any,
    spec: _EmbodimentSpec,
    *,
    embodiment: str,
    planner_model: str,
    idm_model: str,
    needs_prompt: bool,
    run_dir: str,
    action_horizon: int,
) -> VeraServerConfig:
    """Translate the policy's on-the-wire metadata into the declared VeraServerConfig.

    ``action_horizon`` is the DEPLOY chunk size the controller plays per call (default 10 @ 15Hz),
    which is independent of — and usually smaller than — the planner's full future-frame budget
    (``action_chunk_horizon``, e.g. 24). The planner still predicts its full budget; we execute
    the first ``action_horizon`` actions, then refill. ``planner_budget`` is advertised for info.
    """
    meta = policy.get_wire_metadata()
    action_dim = int(meta.get("dim_u") or len(meta.get("action_abs_scale", [])) or 8)
    abs_scale = [float(x) for x in meta.get("action_abs_scale", [])]
    return VeraServerConfig.from_runtime(
        repo_dir=_REPO_DIR,
        run_dir=run_dir,
        image_resolution=None,                       # client resizes per view before width-concat
        view_keys=list(spec.view_keys),
        view_widths=[],                              # per-view widths are client-declared
        proprio_keys=list(spec.proprio_keys),
        needs_prompt=needs_prompt,
        action_space=str(meta.get("action_mode") or spec.action_space),
        action_horizon=action_horizon,
        context_frames=int(meta.get("context_frames", 9)),
        action_dim=action_dim,
        control_dt=spec.control_dt,
        gripper_is_raw=True,
        actions_already_metric=bool(meta.get("actions_already_metric", False)),
        action_abs_scale=abs_scale,
        gripper_dim_index=int(meta.get("gripper_dim_index", -1)),
        embodiment=embodiment,
        planner_model=planner_model,
        idm_model=idm_model,
        is_causal=spec.is_causal,
    )


def make_adapter(
    embodiment: str,
    *,
    device: Any = None,
    algo_config_path: Optional[str] = None,     # <- per-embodiment WAN, or OMNI algo_config.yaml
    dynamics_run_id: Optional[str] = None,      # jacobian/IDM ckpt (wandb run id)
    text: Optional[str] = None,
    sample_steps: Optional[int] = None,
    action_horizon: int = 10,                   # deploy chunk size (actions/call); user default 10
    run_dir: str = "",
    **build_kwargs: Any,
) -> VeraPolicyAdapter:
    """Build the production policy for ``embodiment`` and wrap it in a VeraPolicyAdapter.

    ``algo_config_path`` selects the WAN planner ckpt — point it at the omni algo_config.yaml to
    swap to the omni model with no other change. ``action_horizon`` is the chunk the controller
    plays per call (default 10 @ 15 Hz); the planner still predicts its full future budget and we
    execute the first ``action_horizon`` actions. Extra ``build_kwargs`` pass straight to the
    embodiment's ``build_policy`` (e.g. ``tracker_backend``, ``lang_guidance_override``).
    """
    import importlib

    if embodiment not in _EMBODIMENTS:
        raise KeyError(f"unknown embodiment {embodiment!r}; have {available_embodiments()}")
    spec = _EMBODIMENTS[embodiment]
    mod = importlib.import_module(spec.builder_module)

    if device is None:
        import torch
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Apply the module's WAN tuned-state-dict prefix patch (auto-detects model.model. vs model.)
    if spec.needs_patch and hasattr(mod, "_patch_wan_tuned_state_dict_prefix"):
        mod._patch_wan_tuned_state_dict_prefix()

    policy = mod.build_policy(
        device,
        algo_config_path=algo_config_path,
        dynamics_run_id=dynamics_run_id or mod.DEFAULT_DYNAMICS_RUN_ID,
        text_conditioning=text,
        sample_steps_override=sample_steps,
        **build_kwargs,
    )

    config = _build_server_config(
        policy, spec,
        embodiment=embodiment,
        planner_model=str(algo_config_path or "default-wan"),
        idm_model=str(dynamics_run_id or mod.DEFAULT_DYNAMICS_RUN_ID),
        needs_prompt=text is not None,
        run_dir=run_dir,
        action_horizon=action_horizon,
    )
    return VeraPolicyAdapter(policy, config, default_execute_horizon=action_horizon)
