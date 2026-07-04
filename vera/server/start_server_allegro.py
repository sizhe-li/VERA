"""Start a MotionPolicyAllegro server for the Allegro hand setup.

Cluster side of the 2-machine Allegro deploy architecture:

  +------------------+       ws (msgpack)        +-----------------+
  |  GPU cluster     |  <----------------------> |  hand-host      |
  |  WAN + VGGT-J    |   port 8767 (default)     |  AllegroHwEnv   |
  |                  |                           |  (LCM internally)|
  +------------------+                           +-----------------+

Emits training-normalized [-1, 1] du; MotionPolicyAllegro denormalizes via
the dynamics-checkpoint normalization metadata before returning, so what
goes over the wire is already a physical 16-DOF joint delta in radians.
The hand-host runner adds it to the current q to get the absolute joint
target and sends a chunk to ``AllegroHardwareEnv.step``.

Usage (cluster):
    python -m vera.server.start_server_allegro \\
        --algo-config /path/to/flow-planner/data/exported_models/allegro_mv_1B/better_algo_config.yaml \\
        --dynamics-run-id 3apgu9s9

To connect from the hand-host:
    # SSH tunnel (recommended):
    ssh -N -L 8767:localhost:8767 user@cluster
    python -m tasks_diffusion_policy.neural_jacobian.allegro.scripts.examples.allegro_ws_runner --host localhost --port 8767
"""

import faulthandler
import gc as _gc
import logging
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
from vera.policy.motion_policy_allegro import (
    MotionPolicyAllegro,
    MotionPolicyAllegroCfg,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

faulthandler.enable(all_threads=True)

# ── WAN planner config (allegro multi-view 1B) ─────────────────────
DEFAULT_WAN_ALGO_CONFIG_PATH = Path(
    "/path/to/flow-planner/data/exported_models/"
    "allegro_mv_1B/better_algo_config.yaml"
)
DEFAULT_FLOW_PLANNER_DATA_ROOT = Path(
    "/path/to/flow-planner/data"
)

# ── Dynamics (Jacobian) checkpoints ────────────────────────────────
# 3apgu9s9 = mega 4-dataset VGGT-Jacobian run (current).
# Override via --dynamics-run-id when a newer run becomes preferred.
DYNAMICS_ENTITY = "your-wandb-entity"
DYNAMICS_PROJECT = "jacobian-learning"
DEFAULT_DYNAMICS_RUN_ID = "3apgu9s9"

# ── Runtime defaults ───────────────────────────────────────────────
DIFFUSION_SAMPLING_TIMESTEPS = 40
MOTION_PLAN_SCALE = 3.0

# ── Allegro camera views (mega training: 12-view union; runner sends a
# subset, server matches by name) ──────────────────────────────────
ALLEGRO_VIEW_KEYS = [f"camera_{i}" for i in range(12)]

# ── Wire contract version advertised on the websocket handshake ────
WIRE_CONTRACT_VERSION = 1


def _patch_wan_tuned_state_dict_prefix() -> None:
    """Auto-detect Lightning ckpt prefix when loading WAN tuned weights."""
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
    alltracker_temporal_stride: int = 4,
    motion_plan_scale: float = MOTION_PLAN_SCALE,
    n_action_steps: int = 8,
    control_view_keys: list[str] | None = None,
) -> MotionPolicyAllegro:
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

    policy_action_chunk_horizon = future_pixel_frames
    policy_n_action_steps = int(n_action_steps)

    resolved_view_keys = list(control_view_keys) if control_view_keys else ALLEGRO_VIEW_KEYS

    print(
        f"WAN budget: N={wan_n_latent}, M={wan_m_latent}, stride={wan_stride}\n"
        f"  context_frames={context_pixel_frames}  [source: {context_source}; "
        f"max trained={required_pixel_frames}]\n"
        f"  future_frames={future_pixel_frames}\n"
        f"  action_chunk_horizon={policy_action_chunk_horizon}, "
        f"n_action_steps={policy_n_action_steps}\n"
        f"  sample_steps={eff_sample_steps}  "
        f"[yaml={yaml_sample_steps}, override={sample_steps_override}]\n"
        f"  lang_guidance={eff_lang_g}\n"
        f"  hist_guidance={eff_hist_g}\n"
        f"  alltracker_temporal_stride={max(1, int(alltracker_temporal_stride))}\n"
        f"  motion_plan_scale={motion_plan_scale}\n"
        f"  control_view_keys={resolved_view_keys}\n"
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
        alltracker_temporal_stride=max(1, int(alltracker_temporal_stride)),
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

    cfg = MotionPolicyAllegroCfg(
        name="motion_policy_allegro",
        motion_planner=motion_planner_cfg,
        dynamics_model=dynamics_model_cfg,
        controller=controller_cfg,
        motion_plan_scale=motion_plan_scale,
        action_chunk_horizon=policy_action_chunk_horizon,
        n_action_steps=policy_n_action_steps,
        context_frames=context_pixel_frames,
        control_view_keys=resolved_view_keys,
        text_conditioning=text_conditioning,
    )

    return MotionPolicyAllegro(cfg, device=device)


