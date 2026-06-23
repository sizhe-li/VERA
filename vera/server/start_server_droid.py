"""Start a MotionPolicyDroidNative server for FR3 + DROID.

Emits training-normalized [-1, 1] du directly. The DROID bridge on nora
denormalizes via action_abs_scale advertised on the websocket handshake.

Supports both SE3-delta (dim_u=7) and joint-delta (dim_u=8) dynamics
checkpoints — action_mode is auto-discovered from checkpoint metadata.

Usage:
    # SE3-delta default (dim_u=7, ws :8765, vis :8766)
    python -m vera.server.start_server_droid

    # joint-delta (dim_u=8, 7 Panda joints + gripper)
    python -m vera.server.start_server_droid --dynamics-run-id 2seo56q5

    # With text conditioning
    python -m vera.server.start_server_droid --text "pick up the red block"

    # Disable vis dashboard
    python -m vera.server.start_server_droid --vis-port 0

To connect from nora (the DROID machine):
    # Option 1 — SSH tunnel (recommended):
    ssh -N -L 8765:localhost:8765 user@gpu-cluster
    python -m droid.okto_bridge.droid_ws_runner --host localhost --port 8765

    # Option 2 — Direct (if network allows):
    python -m droid.okto_bridge.droid_ws_runner --host <cluster-ip> --port 8765
"""

import gc as _gc
import logging
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from vera.policy.motion_policy import (
    ControllerCfg,
    DynamicsCfg,
    ModelCheckpoint,
    PlannerCfg,
    _load_algorithm_config_from_path,
)
from vera.policy.motion_policy_droid_native import (
    MotionPolicyDroidNative,
    MotionPolicyDroidNativeCfg,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

# ── WAN planner config (DROID 8C 6T fullsteps) ──────────────────────
DEFAULT_WAN_ALGO_CONFIG_PATH = Path(
    "/path/to/data/acid/evan_checkpoints/exported_checkpoints/"
    "droid_8C_6T_fullsteps/algo_config.yaml"
)
DEFAULT_FLOW_PLANNER_DATA_ROOT = Path(
    "/path/to/data/kitti/jacobian_world_model/flow_planner"
)

# ── Dynamics (Jacobian) checkpoints ──────────────────────────────────
# Production DROID runs (droid_okto_packed_megaflow_full70k):
#   7wohna95 — SE3-delta,   dim_u=7  (tx,ty,tz,rx,ry,rz,gripper)
#   2seo56q5 — joint-delta, dim_u=8  (j0..j6,gripper)
DYNAMICS_ENTITY = "your-wandb-entity"
DYNAMICS_PROJECT = "jacobian-learning"
DEFAULT_DYNAMICS_RUN_ID = "7wohna95"

# ── Runtime defaults ────────────────────────────────────────────────
DIFFUSION_SAMPLING_TIMESTEPS = 40
MOTION_PLAN_SCALE = 3.0  # amplify tracker target to get larger du

# ── DROID camera views ──────────────────────────────────────────────
DROID_VIEW_KEYS = ["varied_1", "varied_2", "hand"]

# ── Wire contract version advertised to the DROID bridge ────────────
WIRE_CONTRACT_VERSION = 1


def _patch_wan_tuned_state_dict_prefix() -> None:
    """Auto-detect Lightning ckpt prefix when loading WAN tuned weights.

    Supports both layouts:
      - 'model.model.<wan_key>'  (legacy: Evan exports + flow-planner third_party default)
      - 'model.<wan_key>'        (current: the original flow-planner export)

    Tries the double-prefix layout first to preserve legacy behavior; if it
    yields zero matches, falls back to the single-prefix layout. Raises if
    neither layout matches so silent zero-weight loads cannot happen.
    """
    from vera.video_model.algorithms.wan.wan_t2v import WanTextToVideo, _load_checkpoint_weights_only

    def _load_tuned_state_dict(self, prefix: str | None = None):
        ckpt = _load_checkpoint_weights_only(
            self.cfg.model.tuned_ckpt_path, mmap=True, map_location="cpu"
        )
        sd = ckpt["state_dict"]
        for try_prefix in ("model.model.", "model."):
            filtered = {
                k[len(try_prefix):]: v
                for k, v in sd.items()
                if k.startswith(try_prefix)
            }
            if filtered:
                logging.info(
                    f"[wan_t2v] loaded {len(filtered)} weights from "
                    f"{self.cfg.model.tuned_ckpt_path} via prefix '{try_prefix}'"
                )
                del ckpt
                _gc.collect()
                return filtered
        del ckpt
        _gc.collect()
        raise RuntimeError(
            f"No keys matching 'model.model.' or 'model.' prefix found in "
            f"{self.cfg.model.tuned_ckpt_path}; cannot load WAN weights"
        )

    WanTextToVideo._load_tuned_state_dict = _load_tuned_state_dict


def _infer_debug_dump_model_name(
    algo_config_path: str,
    wan_algo_cfg: Any,
) -> str:
    explicit = OmegaConf.select(
        wan_algo_cfg,
        "debug_dump_model_name",
        default=None,
    )
    if explicit:
        return str(explicit)
    cfg_stem = Path(str(algo_config_path)).stem
    if cfg_stem and not cfg_stem.startswith("tmp"):
        return cfg_stem
    tuned = OmegaConf.select(
        wan_algo_cfg,
        "model.tuned_ckpt_path",
        default=None,
    )
    if tuned:
        ckpt = Path(str(tuned))
        if ckpt.name in {"video_model.ckpt", "model.ckpt", "latest.ckpt", "last.ckpt"}:
            return ckpt.parent.name
        return ckpt.stem
    return cfg_stem or "model_unknown"


def build_policy(
    device: torch.device,
    algo_config_path: str | None = None,
    flow_planner_data_root: str | None = None,
    dynamics_run_id: str = DEFAULT_DYNAMICS_RUN_ID,
    text_conditioning: str | None = None,
    sample_steps_override: int | None = None,
    lang_guidance_override: float | None = None,
    hist_guidance_override: float | None = None,
    tracker_backend: str = "alltracker",
    megaflow_model_name: str = "megaflow-track",
    megaflow_num_reg_refine: int = 8,
    cotracker_grid_size: int = 15,
) -> MotionPolicyDroidNative:
    algo_config_path = algo_config_path or str(DEFAULT_WAN_ALGO_CONFIG_PATH)
    flow_planner_data_root = flow_planner_data_root or str(
        DEFAULT_FLOW_PLANNER_DATA_ROOT
    )

    wan_cfg = OmegaConf.create(
        _load_algorithm_config_from_path(algo_config_path)
    )
    wan_algo_cfg = wan_cfg.algorithm
    wan_stride = int(wan_algo_cfg.vae.stride[0])
    wan_n_latent = int(wan_algo_cfg.diffusion_forcing.N)
    wan_m_latent = int(wan_algo_cfg.diffusion_forcing.M)

    # ── Resolve runtime knobs: CLI override > yaml value > sane fallback ──
    # sample_steps falls back to 40 (the historical hardcode) if a recipe yaml
    # somehow lacks the field. lang/hist default to 0 (CFG disabled) if absent.
    yaml_sample_steps = int(getattr(wan_algo_cfg, "sample_steps", DIFFUSION_SAMPLING_TIMESTEPS))
    yaml_lang_g = float(getattr(wan_algo_cfg, "lang_guidance", 0.0))
    yaml_hist_g = float(getattr(wan_algo_cfg, "hist_guidance", 0.0))
    eff_sample_steps = (
        sample_steps_override if sample_steps_override is not None else yaml_sample_steps
    )
    eff_lang_g = (
        lang_guidance_override if lang_guidance_override is not None else yaml_lang_g
    )
    eff_hist_g = (
        hist_guidance_override if hist_guidance_override is not None else yaml_hist_g
    )

    # If lang/hist CLI overrides differ from yaml, write a temp yaml so
    # MotionPolicy.load_motion_planner picks up the new values when it reloads
    # the algorithm config. (sample_steps is plumbed through PlannerCfg below.)
    if (eff_lang_g != yaml_lang_g) or (eff_hist_g != yaml_hist_g):
        OmegaConf.update(wan_cfg, "algorithm.lang_guidance", eff_lang_g, merge=True)
        OmegaConf.update(wan_cfg, "algorithm.hist_guidance", eff_hist_g, merge=True)
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False)
        tmp.close()
        OmegaConf.save(wan_cfg.algorithm, tmp.name)
        algo_config_path = tmp.name
        print(
            f"[runtime override] wrote {algo_config_path}: "
            f"lang_guidance={eff_lang_g} (yaml {yaml_lang_g}), "
            f"hist_guidance={eff_hist_g} (yaml {yaml_hist_g})"
        )

    required_pixel_frames = 1 + (wan_n_latent - 1) * wan_stride
    future_pixel_frames = wan_m_latent * wan_stride

    # Context length is a per-model property — recipes are expected to set
    # `inference.context_pixel_frames` to match the trained N. Recipes that
    # lack the field (e.g. legacy Evan exports) fall back to the historical
    # 9-frame default (= 1 + 2*stride with stride=4) so old behavior is
    # preserved exactly.
    legacy_context_default = 1 + 2 * wan_stride
    _ctx_field = OmegaConf.select(
        wan_algo_cfg,
        "inference.context_pixel_frames",
        default=None,
    )
    if _ctx_field is None:
        context_pixel_frames = legacy_context_default
        context_source = f"legacy default (1 + 2*stride = {legacy_context_default})"
    else:
        context_pixel_frames = int(_ctx_field)
        context_source = "yaml inference.context_pixel_frames"

    # Reject a temporally-invalid context up front (N6): the VAE encodes frames into
    # latents in groups of `stride`, so the context must tile cleanly as 1 + k*stride.
    # A typo'd value would otherwise be silently trimmed to the nearest valid length.
    if (context_pixel_frames - 1) % wan_stride != 0:
        valid = ", ".join(str(1 + k * wan_stride) for k in (2, 5, wan_n_latent - 1))
        raise ValueError(
            f"context_pixel_frames={context_pixel_frames} is not VAE-temporally valid for "
            f"stride={wan_stride}: (context_pixel_frames - 1) must be divisible by "
            f"{wan_stride}. Use one of: {valid} (= max trained {required_pixel_frames})."
        )
    if context_pixel_frames > required_pixel_frames:
        print(
            f"[warn] context_pixel_frames={context_pixel_frames} exceeds the model's "
            f"trained max {required_pixel_frames}; the planner will trim it."
        )

    policy_action_chunk_horizon = future_pixel_frames
    policy_n_action_steps = 16

    print(
        f"WAN budget: N={wan_n_latent}, M={wan_m_latent}, stride={wan_stride}\n"
        f"  context_frames={context_pixel_frames}  [source: {context_source}; "
        f"max trained={required_pixel_frames}]\n"
        f"  future_frames={future_pixel_frames}\n"
        f"  action_chunk_horizon={policy_action_chunk_horizon}, "
        f"n_action_steps={policy_n_action_steps}\n"
        f"  sample_steps={eff_sample_steps}  "
        f"[yaml={yaml_sample_steps}, override={sample_steps_override}]\n"
        f"  lang_guidance={eff_lang_g}  "
        f"[yaml={yaml_lang_g}, override={lang_guidance_override}]\n"
        f"  hist_guidance={eff_hist_g}  "
        f"[yaml={yaml_hist_g}, override={hist_guidance_override}]\n"
        f"  control_view_keys={DROID_VIEW_KEYS}\n"
        f"  dynamics_run_id={dynamics_run_id}"
    )

    motion_planner_cfg = PlannerCfg(
        ckpt=None,
        ckpt_path=None,
        algorithm_config_path=algo_config_path,
        diffusion_sampling_timesteps=eff_sample_steps,
        flow_planner_data_root=flow_planner_data_root,
        tracker_backend=tracker_backend,
        tracker_enabled=True,
        tracker_return_visualization=True,
        alltracker_enabled=True,
        alltracker_return_visualization=True,
        alltracker_chunk_size=None,
        alltracker_rate=2,
        alltracker_query_frame=0,
        alltracker_inference_iters=4,
        alltracker_conf_thr=0.6,
        alltracker_bkg_opacity=0.0,
        cotracker_grid_size=cotracker_grid_size,
        megaflow_model_name=megaflow_model_name,
        megaflow_num_reg_refine=megaflow_num_reg_refine,
    )

    dynamics_model_cfg = DynamicsCfg(
        ckpt=ModelCheckpoint(
            entity=DYNAMICS_ENTITY,
            project=DYNAMICS_PROJECT,
            run_id=dynamics_run_id,
            option="latest",
            force_redownload=False,
        ),
    )

    controller_cfg = ControllerCfg(
        lam=0.0,
        clip_du=10000.0,
        action_scale=1.0,
        smoothing=0.0,
        weight_flow_thresh=0.0,
    )

    cfg = MotionPolicyDroidNativeCfg(
        name="motion_policy_droid_native",
        motion_planner=motion_planner_cfg,
        dynamics_model=dynamics_model_cfg,
        controller=controller_cfg,
        motion_plan_scale=MOTION_PLAN_SCALE,
        action_chunk_horizon=policy_action_chunk_horizon,
        n_action_steps=policy_n_action_steps,
        context_frames=context_pixel_frames,
        control_view_keys=DROID_VIEW_KEYS,
        text_conditioning=text_conditioning,
        debug_dump_model_name=_infer_debug_dump_model_name(
            algo_config_path,
            wan_algo_cfg,
        ),
    )

    return MotionPolicyDroidNative(cfg, device=device)


