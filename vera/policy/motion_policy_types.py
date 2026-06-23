from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Tuple

import numpy as np
import torch
from torch import Tensor

from .base_policy import BasePolicyCfg
from .cartesian_policy_support import AdaptiveControllerCfg


@dataclass
class ControllerCfg:
    lam: float = 3.0
    action_scale: float = 1.0
    clip_du: float = 1.0
    smoothing: float = 0.0
    weight_flow_thresh: float = 0.0


def tikhonov_solve(
    J: Tensor,  # [B, M, C]
    y: Tensor,  # [B, M] or [B, M, K]
    lam: float,
    eps: float = 1e-12,
) -> Tensor:
    # This controller runs in the deployment loop. Do not let tracker/model
    # NaNs or infinities reach BLAS/LAPACK, where some builds can abort instead
    # of raising a Python exception.
    J = torch.nan_to_num(J, nan=0.0, posinf=1.0e6, neginf=-1.0e6).clamp(
        -1.0e6, 1.0e6
    ).contiguous()
    y = torch.nan_to_num(y, nan=0.0, posinf=1.0e6, neginf=-1.0e6).clamp(
        -1.0e6, 1.0e6
    ).contiguous()
    if y.ndim == 2:
        y = y.unsqueeze(-1)

    device = J.device
    on_cuda = device.type == "cuda"
    if on_cuda:
        J = J.cpu()
        y = y.cpu()

    JT = J.transpose(-1, -2)
    JTJ = JT @ J
    JTy = JT @ y

    _, C, _ = JTJ.shape
    I = torch.eye(C, device=J.device, dtype=J.dtype).view(1, C, C)
    A = JTJ + (lam * lam + eps) * I
    u = torch.linalg.solve(A, JTy)

    if on_cuda:
        u = u.to(device)
    return u.squeeze(-1)


class JacobianController:
    """Pure controller: solves J u ~= y with optional weighting and smoothing."""

    def __init__(self, cfg: ControllerCfg):
        self.cfg = cfg
        self._du_prev: Tensor | None = None

    def reset(self):
        self._du_prev = None

    def solve(
        self,
        J: Tensor,
        y: Tensor,
        weights: Tensor | None = None,
        lam_override: float | None = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        if weights is not None:
            J_eff = J * weights[..., None]
            y_eff = y * weights
        else:
            J_eff, y_eff = J, y

        lam = float(self.cfg.lam if lam_override is None else lam_override)
        du = tikhonov_solve(J_eff, y_eff, lam=lam)

        if self.cfg.smoothing > 0.0 and self._du_prev is not None:
            du = self.cfg.smoothing * self._du_prev + (1.0 - self.cfg.smoothing) * du

        self._du_prev = du

        Ju = torch.bmm(J, du.unsqueeze(-1)).squeeze(-1)
        residual = Ju - y
        metrics = {
            "residual_l2": residual.norm(dim=1).mean().item(),
            "plan_l2": y.norm(dim=1).mean().item(),
            "relative_residual": (
                residual.norm(dim=1).mean() / (y.norm(dim=1).mean() + 1e-6)
            ).item(),
            "lam": lam,
        }
        return du, metrics


@dataclass
class ModelCheckpoint:
    entity: str
    project: str
    run_id: str
    option: Literal["latest", "best"] = "latest"
    force_redownload: bool = False


@dataclass
class PlannerCfg:
    ckpt: ModelCheckpoint | None = None
    ckpt_path: str | None = None
    algorithm_config_path: str | None = None
    diffusion_sampling_timesteps: int = 100
    flow_decoder_ckpt: ModelCheckpoint | None = None
    flow_decoder_ckpt_path: str | None = None
    flow_planner_data_root: str | None = None
    tracker_backend: Literal["alltracker", "cotracker", "megaflow"] = "alltracker"
    tracker_enabled: bool = True
    tracker_return_visualization: bool = True
    cotracker_model_name: str = "cotracker3_offline"
    cotracker_grid_size: int = 15
    alltracker_enabled: bool = True
    alltracker_return_visualization: bool = True
    alltracker_chunk_size: int | None = None
    alltracker_rate: int = 2
    alltracker_query_frame: int = 0
    alltracker_inference_iters: int = 4
    alltracker_conf_thr: float = 0.60
    alltracker_bkg_opacity: float = 0.0
    alltracker_temporal_stride: int = 1
    megaflow_model_name: str = "megaflow-track"
    megaflow_num_reg_refine: int = 8
    megaflow_query_frame: int = 0
    megaflow_rate: int = 4
    megaflow_autocast_dtype: str = "bfloat16"
    megaflow_vis_from_flow_mag: bool = False
    megaflow_vis_flow_mag_thresh: float = 96.0
    megaflow_bkg_opacity: float = 0.0


@dataclass
class DynamicsCfg:
    ckpt: ModelCheckpoint | None = None
    ckpt_path: str | None = None


@dataclass
class MotionPolicyCfg(BasePolicyCfg):
    name: Literal["motion_policy"]
    motion_planner: PlannerCfg
    dynamics_model: DynamicsCfg
    controller: ControllerCfg
    adaptive_controller: AdaptiveControllerCfg = field(
        default_factory=AdaptiveControllerCfg
    )
    motion_plan_scale: float = 1.0
    action_chunk_horizon: int = 16
    n_action_steps: int = 8
    context_frames: int | None = None
    control_view_keys: list[str] | None = None
    max_track_abs_dx: float | None = None
    max_track_disp_norm: float | None = None
    vis_track_sparsity: int = 100
    vis_track_sparsity_joint: int = 50
    jacobian_vis_abs_threshold: float = 0.0
    debug_dump_enabled: bool = False
    debug_dump_dir: str = "/path/to/data/jacobian/allegro_realworld_exp_log"
    debug_dump_model_name: str | None = None
    debug_dump_task_name: str | None = None
    debug_dump_max_chunks: int = 200
    debug_dump_max_bytes: int = 5 * 1024**3
    debug_dump_min_free_bytes: int = 2 * 1024**3
    text_conditioning: str | list[str] | None = None
    normalization_compatibility_mode: Literal["off", "warn", "error"] = "warn"
    verbose: bool = True
