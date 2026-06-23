"""Build a MotionPolicyGripper policy for MIMICGEN (sim) — omni WAN planner + mimicgen Jacobian IDM.

Used to iterate the controller/client/server design in simulation before the real robot. The omni
WAN (combined_4env) was trained with a unified 128x576 canvas; mimicgen contributes 2 views
(agentview + wrist) = 256 wide, padded with black on the RIGHT to 576 (the policy/planner handles
the pad internally). The mimicgen Jacobian (x21o0cwe, dpt_vggt_fusion_M_128) consumes the same 2
views. Mirrors start_server_droid.build_policy but uses MotionPolicyGripper (gripper gating).

    python -m vera.server.start_vera_server --embodiment mimicgen --port 8800 --vis-port 8801 --sample-steps 10
"""

import gc as _gc
import logging
import os
from pathlib import Path

import torch
from omegaconf import OmegaConf

from vera.policy.motion_policy import (
    ControllerCfg,
    DynamicsCfg,
    ModelCheckpoint,
    PlannerCfg,
    _load_algorithm_config_from_path,
)
from vera.policy.motion_policy_gripper import MotionPolicyGripper, MotionPolicyGripperCfg
from vera.policy.cartesian_policy_support import AdaptiveControllerCfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

# ── omni WAN planner (combined_4env, exported from the live training step=9000) ──
DEFAULT_WAN_ALGO_CONFIG_PATH = Path(
    "/path/to/data/jacobian/exported/"
    "omni_combined_4env_step9000/algo_config.yaml"
)
DEFAULT_FLOW_PLANNER_DATA_ROOT = Path(
    "/path/to/data/kitti/jacobian_world_model/flow_planner"
)

# ── mimicgen Jacobian IDM (env-overridable to swap checkpoints/arch) ──
#   default = x21o0cwe (dpt_vggt_fusion, jacobian-learning).
#   VGGT v3 normfix = run 285ouq1q in project jacobian-mimicgen (vggt_jacobian arch).
#   The loader (motion_policy_loading.load_checkpoint) is arch-agnostic: it reads the run's
#   model_cfg + dataset normalization, so swapping run/project is enough.
DYNAMICS_ENTITY = os.environ.get("VERA_DYNAMICS_ENTITY", "your-wandb-entity")
DYNAMICS_PROJECT = os.environ.get("VERA_DYNAMICS_PROJECT", "jacobian-learning")
DEFAULT_DYNAMICS_RUN_ID = os.environ.get("VERA_DYNAMICS_RUN_ID", "x21o0cwe")

# ── mimicgen dualview ──
MIMICGEN_VIEW_KEYS = ["agentview_image", "robot0_eye_in_hand_image"]
DIFFUSION_SAMPLING_TIMESTEPS = 40
MOTION_PLAN_SCALE = 1.0   # old-best (run_mimicgen_eval.sh); 3.0 overshoots, 2.0 retreats. Verified 2026-06-10.


def _patch_wan_tuned_state_dict_prefix() -> None:
    """Auto-detect Lightning ckpt prefix (model.model. vs model.) for the consolidated omni WAN."""
    from vera.video_model.algorithms.wan.wan_t2v import WanTextToVideo, _load_checkpoint_weights_only

    def _load_tuned_state_dict(self, prefix=None):
        ckpt = _load_checkpoint_weights_only(self.cfg.model.tuned_ckpt_path, mmap=True, map_location="cpu")
        sd = ckpt["state_dict"]
        for try_prefix in ("model.model.", "model."):
            filtered = {k[len(try_prefix):]: v for k, v in sd.items() if k.startswith(try_prefix)}
            if filtered:
                logging.info("[wan] loaded %d weights via prefix '%s'", len(filtered), try_prefix)
                del ckpt; _gc.collect()
                return filtered
        del ckpt; _gc.collect()
        raise RuntimeError(f"No model.model./model. prefix in {self.cfg.model.tuned_ckpt_path}")

    WanTextToVideo._load_tuned_state_dict = _load_tuned_state_dict


def build_policy(
    device: torch.device,
    algo_config_path: str | None = None,
    flow_planner_data_root: str | None = None,
    dynamics_run_id: str = DEFAULT_DYNAMICS_RUN_ID,
    text_conditioning: str | None = None,
    sample_steps_override: int | None = None,
    lang_guidance_override: float | None = None,
    hist_guidance_override: float | None = None,
    tracker_backend: str = "cotracker",   # old-best; alltracker produces wrong-direction flow on mimicgen (arm flees blocks). Verified 2026-06-10.
    control_view_keys: list[str] | None = None,
    **_ignored,
) -> MotionPolicyGripper:
    algo_config_path = algo_config_path or str(DEFAULT_WAN_ALGO_CONFIG_PATH)
    flow_planner_data_root = flow_planner_data_root or str(DEFAULT_FLOW_PLANNER_DATA_ROOT)
    view_keys = list(control_view_keys) if control_view_keys else MIMICGEN_VIEW_KEYS

    # --- overnight sweep overrides (env): match old-best mimicgen params without CLI plumbing ---
    #  VERA_TRACKER_BACKEND   (default alltracker; old-best = cotracker)
    #  VERA_MOTION_PLAN_SCALE (default 3.0;        old-best = 1.0)
    #  VERA_N_ACTION_STEPS    (default 10;         old-best exec_horizon = 6)
    tracker_backend = os.environ.get("VERA_TRACKER_BACKEND", tracker_backend)
    motion_plan_scale = float(os.environ.get("VERA_MOTION_PLAN_SCALE", MOTION_PLAN_SCALE))
    n_action_steps = int(os.environ.get("VERA_N_ACTION_STEPS", "10"))

    wan_cfg = OmegaConf.create(_load_algorithm_config_from_path(algo_config_path)).algorithm
    stride = int(wan_cfg.vae.stride[0])
    n_latent = int(wan_cfg.diffusion_forcing.N)
    m_latent = int(wan_cfg.diffusion_forcing.M)
    eff_steps = sample_steps_override if sample_steps_override is not None else int(
        getattr(wan_cfg, "sample_steps", DIFFUSION_SAMPLING_TIMESTEPS))
    future_pixel_frames = m_latent * stride
    # context length MUST match the trained N: required_pixel_frames = 1 + (N-1)*stride. This is the
    # number of context frames the WAN was trained on (decodes to N context latents under the VAE
    # stride). The eval (_load_wan_runtime_info) uses exactly this. (The old `1 + 2*stride` fallback
    # under-contexted the model badly — e.g. 9 instead of 21 for N=6,stride=4 — degrading the dream.)
    ctx_field = OmegaConf.select(wan_cfg, "inference.context_pixel_frames", default=None)
    context_pixel_frames = int(ctx_field) if ctx_field is not None else 1 + max(n_latent - 1, 0) * stride

    logging.info(
        "MIMICGEN WAN budget: N=%d M=%d stride=%d ctx=%d future=%d steps=%d views=%s dyn=%s",
        n_latent, m_latent, stride, context_pixel_frames, future_pixel_frames,
        eff_steps, view_keys, dynamics_run_id,
    )
    logging.info("MIMICGEN sweep overrides: motion_plan_scale=%.2f n_action_steps=%d tracker=%s",
                 motion_plan_scale, n_action_steps, tracker_backend)

    planner_cfg = PlannerCfg(
        ckpt=None, ckpt_path=None,
        algorithm_config_path=algo_config_path,
        diffusion_sampling_timesteps=eff_steps,
        flow_planner_data_root=flow_planner_data_root,
        tracker_backend=tracker_backend,
        tracker_enabled=True, tracker_return_visualization=True,
        alltracker_enabled=True, alltracker_return_visualization=True,
        alltracker_rate=2, alltracker_query_frame=0,
        alltracker_inference_iters=4, alltracker_conf_thr=0.6, alltracker_bkg_opacity=0.0,
    )
    dynamics_cfg = DynamicsCfg(
        ckpt=ModelCheckpoint(
            entity=DYNAMICS_ENTITY, project=DYNAMICS_PROJECT,
            run_id=dynamics_run_id, option="latest", force_redownload=False,
        ),
    )
    controller_cfg = ControllerCfg(
        lam=0.0, clip_du=10000.0, action_scale=1.0, smoothing=0.0, weight_flow_thresh=0.0,
    )
    cfg = MotionPolicyGripperCfg(
        name="motion_policy_gripper",
        motion_planner=planner_cfg,
        dynamics_model=dynamics_cfg,
        controller=controller_cfg,
        motion_plan_scale=motion_plan_scale,
        # action_chunk_horizon = the dream's future frames (M*stride=16, == the eval's task_horizon);
        # n_action_steps = exec_horizon (commit this many of the planned chunk per step).
        action_chunk_horizon=future_pixel_frames,
        n_action_steps=n_action_steps,
        context_frames=context_pixel_frames,
        control_view_keys=view_keys,
        debug_dump_model_name="omni_combined_4env_step9000",
        # ── PROVEN stack_d0 recipe (okto eval 2026-04-21: 25/30 = 83%, top5 47/50 = 94%) ──
        # The full recipe = mimicgen-specialist WAN (mimicgen_dualview_all_in_one) + IDM
        # 37oa162u + cotracker + this gated gripper + the adaptive state_delta_grouped
        # controller below + exec=6. Verbatim from the winning run's policy_cfg.
        # Supersedes the 2026-06-11 June overrides (focus_gain=-1, close=1.0, exec=10),
        # which diverged from this and regressed mimicgen SR toward ~0.
        gripper_command_mode="gated",
        gripper_thresh=0.18,
        gripper_close_thresh=0.18,
        gripper_open_thresh=0.18,
        gripper_deadband_thresh=0.12,
        gripper_min_hold_steps=6,
        gripper_jacobian_focus_gain=-10.0,
        gripper_jacobian_focus_bottom_margin_ratio=0.08984375,
        gripper_realign_on_step0=True,
        gripper_fixed_value=None,
        action_scale_translation=0.12,
        action_scale_yaw=1.0 / 45.0,  # 0.0222
        adaptive_controller=AdaptiveControllerCfg(
            enabled=True,
            mode="state_delta_grouped",
            ema_alpha=0.2,
            invalid_track_penalty=0.25,
            lam_max_scale=4.0,
            action_gain_min_scale=0.35,
            action_gain_max_scale=1.0,
            use_track_mismatch_for_lam=False,
            grouped_action_ema_alpha=0.18,
            grouped_action_step_up=0.08,
            grouped_action_step_down=0.18,
            translation_gain_min_scale=0.45,
            translation_gain_max_scale=1.2,
            rotation_gain_min_scale=0.45,
            rotation_gain_max_scale=1.2,
            gripper_gain_min_scale=0.85,
            gripper_gain_max_scale=1.2,
            enable_gripper_channel_adaptation=True,
        ),
    )
    return MotionPolicyGripper(cfg, device=device)
