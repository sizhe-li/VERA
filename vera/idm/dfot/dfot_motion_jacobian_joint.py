from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import cv2
from einops import rearrange, einsum
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor
from torchvision.utils import flow_to_image

# NOTE: import locally within dfot package to avoid type-checker confusion
# from circular imports through `vera.idm.dfot.__init__`.
from .dfot_motion_policy import (
    DFoTMotionPolicy,
    flatten_time_dim,
    flatten_time_view_dim,
)
from .backbones import Unet3D, DiT3D
from vera.idm.registry import register_algorithm
from vera.datasets.normalization import (
    denormalize_flow_tensor,
    denormalize_jacobian_tensor,
)
from vera.utils import jacobian_utils
from vera.utils.logging_utils import get_sanity_metrics


class TemporalJacobianDecoder(nn.Module):
    """
    Simple 3D conv decoder that maps a video tensor to dense Jacobian fields.

    Input:  x  - [B, T, C_in, H, W]
    Output: Jf - [B, T, C_cmd * C_spatial, H, W]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        mid = max(in_channels // 2, 8)
        mid2 = max(mid // 2, 8)

        self.net = nn.Sequential(
            nn.Conv3d(in_channels, mid, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid, mid2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid2, out_channels, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        # [B, T, C, H, W] -> [B, C, T, H, W]
        x = rearrange(x, "b t c h w -> b c t h w").contiguous()
        x = self.net(x)
        # -> [B, T, C_out, H, W]
        x = rearrange(x, "b c t h w -> b t c h w")
        return x


def _init_weights(m: nn.Module, std: float = 1e-4) -> None:
    if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear)):
        if m.weight is not None:
            torch.nn.init.normal_(m.weight, mean=0.0, std=std)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias, mean=0.0, std=std)


@dataclass
class JointJacobianCfg:
    """
    Lightweight config wrapper for the joint Jacobian head.

    All fields are optional on the Hydra side; sensible defaults are used when missing.
    """

    enable: bool = True
    # Number of spatial dimensions in the flow (e.g. 2 for (u, v)).
    spatial_dim: int = 2
    # Optional explicit command dimension; if <= 0, we infer from dataset metadata
    # (len(jacobian_action_scales) or du.shape[-1]).
    command_dim: int = -1
    # Weight for the Jacobian loss relative to diffusion loss.
    loss_weight: float = 1.0
    # Supervision type; currently only "flow" is supported, but we keep this for parity
    # with the standalone Jacobian algorithm.
    supervision: str = "flow"


# NOTE: `register_algorithm` typing is strict; DFoT algos are dynamically configured.
@register_algorithm("dfot_motion_jacobian_joint", cfg_cls=None)  # type: ignore
class DFoTJointMotionJacobian(DFoTMotionPolicy):
    """
    Joint video model that:
      - trains the DFoT diffusion model on dense flow sequences (as in DFoTMotionPolicy)
      - in parallel trains a temporal Jacobian head J(x_t) such that J @ du ≈ flow_t

    This is intentionally implemented as a NEW algorithm so that the existing
    `dfot_motion_policy` remains completely unchanged and backward compatible.
    """

    def __init__(self, cfg) -> None:  # cfg is a DictConfig
        super().__init__(cfg)  # type: ignore[call-arg]

        # Joint Jacobian configuration (all fields optional on the Hydra side).
        jj_cfg_dict = getattr(self.cfg, "joint_jacobian", {}) or {}  # type: ignore[attr-defined]
        # OmegaConf objects are Mapping-like but not plain dicts; normalize.
        self.joint_jacobian_cfg = JointJacobianCfg()
        for k, v in dict(jj_cfg_dict).items():
            if hasattr(self.joint_jacobian_cfg, k):
                setattr(self.joint_jacobian_cfg, k, v)

        self._jacobian_head: Optional[TemporalJacobianDecoder] = None
        self._jacobian_cmd_dim: Optional[int] = None
        self._jacobian_spatial_dim: int = self.joint_jacobian_cfg.spatial_dim
        self._jacobian_head_in_channels: Optional[int] = None

        # Store the *raw* control batch from the dataloader for Jacobian supervision.
        self._latest_control_batch: Optional[Dict[str, Any]] = None
        self._skip_jacobian: bool = False

        # Backbone feature extraction (UNet3D / DiT3D) for Jacobian decoding.
        self._backbone_feature_hook_handle: Optional[
            torch.utils.hooks.RemovableHandle
        ] = None
        self._latest_backbone_features: Optional[Tensor] = None

    # -------------------------------------------------------------------------
    # Model & Optimizer
    # -------------------------------------------------------------------------

    def _build_model(self) -> None:
        """
        Reuse DFoTMotionPolicy diffusion model; Jacobian head is lazy-built once we
        know command_dim from dataset metadata / first batch.
        """
        super()._build_model()  # type: ignore[misc]

    def _build_jacobian_head_if_needed(
        self,
        backbone_features: Tensor,
        control_batch: Dict[str, Any],
    ) -> None:
        """
        Lazily build the TemporalJacobianDecoder once we know the command dimension.
        """
        if self._jacobian_head is not None:
            return

        if not self.joint_jacobian_cfg.enable:
            return

        # Infer command_dim
        cmd_dim_cfg = self.joint_jacobian_cfg.command_dim
        cmd_dim: Optional[int] = None
        if cmd_dim_cfg and cmd_dim_cfg > 0:
            cmd_dim = cmd_dim_cfg
        else:
            # Try dataset metadata
            meta = getattr(self, "dataset_metadata", None) or {}
            action_scales = meta.get("jacobian_action_scales")
            if action_scales is not None:
                cmd_dim = len(action_scales)
            else:
                # Prefer already-aligned supervision (covers max_frames==1 time->batch collapse)
                sup = getattr(self, "_latest_supervision", None) or {}
                du_flat = sup.get("du_flat") if isinstance(sup, dict) else None
                if du_flat is not None:
                    cmd_dim = int(du_flat.shape[-1])
                else:
                    # Fallback to du tensor from raw batch
                    du = control_batch.get("du")
                    if du is not None:
                        cmd_dim = int(du.shape[-1])

        if cmd_dim is None or cmd_dim <= 0:
            # If we cannot infer yet, skip building for now (do NOT disable permanently).
            if getattr(self, "global_rank", 0) == 0:
                print(
                    "[DFoTJointMotionJacobian] Could not infer command_dim yet; "
                    "skipping Jacobian head build for this step."
                )
            return

        self._jacobian_cmd_dim = cmd_dim

        # Input channels: use backbone features as input to the decoder.
        in_channels = int(backbone_features.shape[2])
        out_channels = cmd_dim * self._jacobian_spatial_dim

        self._jacobian_head = TemporalJacobianDecoder(
            in_channels=in_channels,
            out_channels=out_channels,
        ).to(
            self.device  # type: ignore[attr-defined]
        )
        self._jacobian_head.apply(_init_weights)
        self._jacobian_head_in_channels = in_channels
        if getattr(self, "global_rank", 0) == 0:
            print(
                "[DFoTJointMotionJacobian] Built Jacobian head "
                f"(in_channels={in_channels}, cmd_dim={cmd_dim}, "
                f"spatial_dim={self._jacobian_spatial_dim})"
            )

    def configure_optimizers(self):
        """
        Extend DFoT optimizer to include Jacobian head parameters.
        """
        base = super().configure_optimizers()  # type: ignore[misc]
        if self._jacobian_head is None:
            return base

        # DFoTMotionPolicy already groups all diffusion parameters in one optimizer.
        # To keep things simple and backward compatible, we just add Jacobian head
        # params into the same optimizer.
        optimizer: torch.optim.Optimizer = base["optimizer"]
        optimizer.add_param_group({"params": self._jacobian_head.parameters()})
        return base

    # -------------------------------------------------------------------------
    # Data Preprocessing
    # -------------------------------------------------------------------------

    def on_after_batch_transfer(
        self, batch: Dict[str, Any], dataloader_idx: int
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Tensor, Optional[Tensor]]:
        """
        Keep DFoTMotionPolicy preprocessing, but also stash the *raw* control batch
        (with `du`, `flow`, and optional `tracks`) for Jacobian supervision.
        """
        self._latest_control_batch = batch
        xs, xs_mask, conditions, masks, rgb_processed = super().on_after_batch_transfer(  # type: ignore[misc]
            batch, dataloader_idx
        )

        # Prepare du/flow aligned to DFoT's possible view-flatten + time->batch collapse.
        self._latest_supervision = self._prepare_supervision_like_dfot(
            raw_batch=batch,
            rgb_processed=rgb_processed,
        )

        return xs, xs_mask, conditions, masks, rgb_processed

    def _prepare_supervision_like_dfot(
        self,
        raw_batch: Dict[str, Any],
        rgb_processed: Tensor,
    ) -> Dict[str, Tensor | None]:
        """
        Make (du, flow) match the same flattening/collapse semantics as DFoT's
        `on_after_batch_transfer`, so Jacobian training sees consistent shapes.

        Returns a dict with:
          - du_flat:   [B_eff, cmd_dim] (one per DFoT token if max_frames==1)
          - flow_flat: [B_eff, s_dim, H, W]
        """
        du = raw_batch.get("du")
        flow = raw_batch.get("flow")
        rgb = raw_batch.get("rgb")
        if du is None or flow is None or rgb is None:
            return {"du_flat": None, "flow_flat": None}

        # Step 1: handle multi-view (DFoT flattens batch+view, keeps time)
        if rgb.ndim == 6:  # (B,T,V,C,H,W)
            B, T, V, *_ = rgb.shape
            # du: (B,T,cmd) -> (B,T,V,cmd) -> (B*V,T,cmd)
            du = du[:, :, None, :].expand(-1, -1, V, -1)
            du = flatten_time_view_dim(du)
            # flow: (B,T,V,s,H,W) -> (B*V,T,s,H,W)
            flow = flatten_time_view_dim(flow)
        else:
            # keep (B,T,...) as-is
            B, T = rgb.shape[0], rgb.shape[1]

        # Step 2: match DFoT's "collapse_time_into_batch" behavior
        # DFoT collapses if max_frames==1 and dataset provides T>1.
        collapse_time_into_batch = self.max_frames == 1 and T > self.max_frames  # type: ignore[attr-defined]
        if collapse_time_into_batch:
            # du: (B_eff,T,cmd) -> (B_eff*T,1,cmd) -> squeeze time
            du = rearrange(du, "b t c -> (b t) 1 c")[:, 0]
            # flow: (B_eff,T,s,H,W) -> (B_eff*T,1,s,H,W) -> squeeze time
            flow = rearrange(flow, "b t s h w -> (b t) 1 s h w")[:, 0]
        else:
            # Use the last frame (or you could supervise all frames; start simple)
            du = du[:, -1]
            flow = flow[:, -1]

        # At this point, du: [B_eff, cmd], flow: [B_eff, s, H, W]
        return {"du_flat": du, "flow_flat": flow}

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    def training_step(
        self, batch, batch_idx, namespace: str = "training"
    ) -> STEP_OUTPUT:
        """
        Wrap the parent training_step to add a Jacobian loss term.
        """
        # Install backbone feature hooks (once) so the diffusion forward pass
        # inside DFoTMotionPolicy.training_step populates `_latest_backbone_features`.
        self._ensure_backbone_feature_hooks()
        self._latest_backbone_features = None

        # First, run the original DFoT motion policy training.
        base_out = super().training_step(batch, batch_idx, namespace)  # type: ignore[misc]

        if (
            not self.joint_jacobian_cfg.enable
            or self._skip_jacobian
            or self._latest_control_batch is None
            or namespace != "training"
        ):
            return base_out

        xs, xs_mask, conditions, masks, rgb_tokens = batch
        control_batch = self._latest_control_batch

        # Lazily build Jacobian head once we have dataset info.
        # Decide which feature source to use:
        # - Prefer backbone features when available and channel-compatible.
        # - Otherwise, fall back to DFoT RGB tokens.
        backbone_features = self._latest_backbone_features
        if backbone_features is None:
            backbone_features = rgb_tokens

        # If we already have a head, respect its expected channel count.
        if (
            self._jacobian_head is not None
            and self._jacobian_head_in_channels is not None
        ):
            if backbone_features.shape[2] != self._jacobian_head_in_channels:
                backbone_features = rgb_tokens

        if getattr(self, "global_rank", 0) == 0 and batch_idx == 0:
            src = (
                "backbone_features"
                if backbone_features is self._latest_backbone_features
                and backbone_features is not None
                else "rgb_tokens"
            )
            print(
                "[DFoTJointMotionJacobian] Using feature source "
                f"{src} for Jacobian head: "
                f"shape={tuple(backbone_features.shape)} "
                f"(expected C={self._jacobian_head_in_channels or 'N/A'})"
            )

        self._build_jacobian_head_if_needed(backbone_features, control_batch)
        if self._jacobian_head is None:
            return base_out

        # Compute Jacobian loss
        jacobian_loss = self._compute_jacobian_loss_from_batch(
            backbone_features=backbone_features,
            control_batch=control_batch,
        )

        if jacobian_loss is None:
            return base_out

        total_loss = (
            base_out["loss"] + self.joint_jacobian_cfg.loss_weight * jacobian_loss
        )
        base_out["loss"] = total_loss
        base_out["jacobian_loss"] = jacobian_loss.detach()

        # Log Jacobian loss
        if batch_idx % self.cfg.logging.loss_freq == 0:  # type: ignore[attr-defined]
            self.log(  # type: ignore[attr-defined]
                f"{namespace}/loss_jacobian",
                jacobian_loss,
                on_step=namespace == "training",
                on_epoch=namespace != "training",
                sync_dist=True,
            )

            # Sanity metrics for Jacobian branch
            for k, v in get_sanity_metrics(control_batch).items():
                self.log(  # type: ignore[attr-defined]
                    f"sanity/jac_input_{k}",
                    v,
                    on_step=True,
                    on_epoch=False,
                    sync_dist=True,
                )

        return base_out

    def _compute_jacobian_loss_from_batch(
        self,
        backbone_features: Tensor,  # candidate features from backbone or DFoT RGB tokens: [B_eff, T_tok, C, H, W]
        control_batch: Dict[str, Any],  # raw batch (kept for metadata fallbacks)
    ) -> Optional[Tensor]:
        """
        Supervise the Jacobian head so that J @ du ≈ flow, similar to JacobianVanilla.
        """
        # Use supervision tensors aligned to DFoT preprocessing
        sup = getattr(self, "_latest_supervision", None) or {}
        du_flat = sup.get("du_flat")
        flow_gt = sup.get("flow_flat")
        if du_flat is None or flow_gt is None:
            return None

        # Clamp context length: many configs use context_length=-1 to mean "use all";
        # but for slicing it would create an empty tensor. For this head we need >=1.
        n_ctx = max(int(self.n_context_frames), 1)  # type: ignore[attr-defined]

        # backbone_features: [B_eff, T_tok, C, H, W] (UNet3D mid-block / DiT3D tokens
        # decoded to frames, or DFoT RGB tokens as a fallback).
        # Ensure channel compatibility with the head; if not, just slice on time.
        if self._jacobian_head_in_channels is not None:
            if backbone_features.shape[2] != self._jacobian_head_in_channels:
                # This should not normally happen because the caller filters
                # features, but guard defensively.
                backbone_features = backbone_features[
                    :, :, : self._jacobian_head_in_channels
                ]

        feat_ctx = backbone_features[:, :n_ctx]

        # Predict flattened Jacobian fields: [B_eff, T_ctx, cmd_dim * s_dim, H, W]
        J_flat = self._jacobian_head(feat_ctx)  # type: ignore[arg-type]

        cmd_dim = self._jacobian_cmd_dim or int(du_flat.shape[-1])
        s_dim = self._jacobian_spatial_dim

        # We only use the last context token for supervision to match one-step du,flow.
        J_last = J_flat[:, -1]  # [B_eff, cmd_dim * s_dim, H, W]
        J = rearrange(
            J_last,
            "b (c s) h w -> b c s h w",
            c=cmd_dim,
            s=s_dim,
        )  # [B_eff, cmd_dim, s_dim, H, W]

        # Compose flow from Jacobian and actions: J @ du
        #   du: [B_flat, cmd_dim]
        #   J : [B_flat, cmd_dim, s_dim, H, W]
        flow_pred = einsum(
            J,
            du_flat,
            "b c s h w, b c -> b s h w",
        )

        # Match resolution (if needed)
        if flow_pred.shape[-2:] != flow_gt.shape[-2:]:
            flow_pred = F.interpolate(
                flow_pred, size=flow_gt.shape[-2:], mode="bilinear", align_corners=False
            )

        # Standard MSE in normalized flow units (dataset already applied oflow_scale).
        loss = F.mse_loss(flow_pred, flow_gt)
        return loss

    # -------------------------------------------------------------------------
    # Validation: visualization (Jacobian + flow), mirroring JacobianVanilla
    # -------------------------------------------------------------------------

    @staticmethod
    def _get_tracks_at_frame(
        tracks: Any,
        b: int,
        v: int,
        t: int,
        V: int,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Return (xy_src, pixel_selector, gt_disp, visibility) for batch b, view v, time t.
        This is copied from `JacobianVanilla` so the visualization matches exactly.
        """
        if isinstance(tracks, list):
            tr = tracks[b]
            idx_src = tr["idx_src"]
            if idx_src.ndim == 3:  # (T,V,N)
                xy_src = tr["xy_src"][t, v]
                pixel_selector = tr["idx_src"][t, v]
                gt_disp = tr["disp"][t, v]
                visibility = tr["valid"][t, v]
            elif idx_src.ndim == 2:  # (T,N)
                xy_src = tr["xy_src"][t]
                pixel_selector = tr["idx_src"][t]
                gt_disp = tr["disp"][t]
                visibility = tr["valid"][t]
            else:
                raise ValueError(f"Unexpected tracks idx_src shape: {idx_src.shape}")
        else:
            tr = tracks
            idx_src = tr["idx_src"]
            if idx_src.ndim == 4:  # (B,T,V,N)
                xy_src = tr["xy_src"][b, t, v]
                pixel_selector = tr["idx_src"][b, t, v]
                gt_disp = tr["disp"][b, t, v]
                visibility = tr["valid"][b, t, v]
            elif idx_src.ndim == 3:  # (B,T,N)
                xy_src = tr["xy_src"][b, t]
                pixel_selector = tr["idx_src"][b, t]
                gt_disp = tr["disp"][b, t]
                visibility = tr["valid"][b, t]
            elif idx_src.ndim == 2:  # (T,N)
                xy_src = tr["xy_src"][t]
                pixel_selector = tr["idx_src"][t]
                gt_disp = tr["disp"][t]
                visibility = tr["valid"][t]
            else:
                raise ValueError(f"Unexpected tracks idx_src shape: {idx_src.shape}")
        return xy_src, pixel_selector, gt_disp, visibility

    @staticmethod
    def _compute_track_keep_indices(
        xy_src: Tensor,  # (N,2)
        H: int,
        W: int,
        sparsity: int = 100,
    ) -> Tensor:
        """
        Choose a fixed subset of track indices using spatial grid (one per cell).
        Copied from `JacobianVanilla` for temporal continuity.
        """
        N = xy_src.shape[0]
        base_area = 8 * 8
        approx_arrows = max(int((H * W) / base_area / max(sparsity / 100.0, 1e-3)), 1)
        approx_arrows = min(approx_arrows, N)
        grid_size = max(int(np.sqrt(approx_arrows)), 1)
        gx = max(W // grid_size, 1)
        gy = max(H // grid_size, 1)
        x = xy_src[:, 0]
        y = xy_src[:, 1]
        cell_x = torch.div(x, gx, rounding_mode="floor").long()
        cell_y = torch.div(y, gy, rounding_mode="floor").long()
        cell_x = cell_x.clamp(0, W // gx)
        cell_y = cell_y.clamp(0, H // gy)
        num_x = (W // gx) + 1
        cell_id = cell_y * num_x + cell_x  # (N,)
        max_cell = cell_id.max().item() + 1
        first_idx = torch.full(
            (int(max_cell),), int(N), device=cell_id.device, dtype=torch.long
        )
        idx = torch.arange(N, device=cell_id.device)
        first_idx.scatter_reduce_(0, cell_id, idx, reduce="amin", include_self=True)
        unique_indices = first_idx[first_idx < N]
        keep = torch.zeros(N, dtype=torch.bool, device=cell_id.device)
        keep[unique_indices] = True
        return torch.where(keep)[0]

    def visualize_pixel_motion(
        self,
        xy_src: Tensor,  # (N,2)
        gt_pixel_motion: Tensor,  # (N,2)
        pred_pixel_motion: Tensor,  # (N,2)
        gt_pixel_visibility: Tensor,  # (N,)
        rgb: Tensor,  # (3,H,W)
        sparsity: int = 100,
        motion_thresh: float = 0.05,
        keep_indices: Optional[Tensor] = None,
    ) -> np.ndarray:
        """
        Lagrangian track visualization.
        Red = GT motion, Blue = predicted motion.
        Copied from `JacobianVanilla` so visuals match.
        """
        dataset_meta = getattr(self, "dataset_metadata", {}) or {}
        gt_motion = denormalize_flow_tensor(gt_pixel_motion, dataset_meta)
        pred_motion = denormalize_flow_tensor(pred_pixel_motion, dataset_meta)

        H, W = rgb.shape[1], rgb.shape[2]

        if keep_indices is not None:
            N_cur = xy_src.shape[0]
            valid = keep_indices < N_cur
            keep_indices = keep_indices[valid]
            xy_src = xy_src[keep_indices]
            gt_motion = gt_motion[keep_indices]
            pred_motion = pred_motion[keep_indices]
            visibility = gt_pixel_visibility[keep_indices]
        else:
            base_area = 8 * 8
            approx_arrows = max(
                int((H * W) / base_area / max(sparsity / 100.0, 1e-3)), 1
            )
            approx_arrows = min(approx_arrows, int(xy_src.shape[0]))
            grid_size = max(int(np.sqrt(approx_arrows)), 1)
            gx = max(W // grid_size, 1)
            gy = max(H // grid_size, 1)
            x = xy_src[:, 0]
            y = xy_src[:, 1]
            cell_x = torch.div(x, gx, rounding_mode="floor").long()
            cell_y = torch.div(y, gy, rounding_mode="floor").long()
            cell_x = cell_x.clamp(0, W // gx)
            cell_y = cell_y.clamp(0, H // gy)
            num_x = (W // gx) + 1
            cell_id = cell_y * num_x + cell_x  # (N,)
            N = cell_id.shape[0]
            max_cell = cell_id.max().item() + 1
            first_idx = torch.full(
                (int(max_cell),), int(N), device=cell_id.device, dtype=torch.long
            )
            idx = torch.arange(N, device=cell_id.device)
            first_idx.scatter_reduce_(0, cell_id, idx, reduce="amin", include_self=True)
            unique_indices = first_idx[first_idx < N]
            keep = torch.zeros(N, dtype=torch.bool, device=cell_id.device)
            keep[unique_indices] = True
            xy_src = xy_src[keep]
            gt_motion = gt_motion[keep]
            pred_motion = pred_motion[keep]
            visibility = gt_pixel_visibility[keep]

        rgb_np = np.ascontiguousarray(
            (rgb.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        )

        short_side = float(min(H, W))
        scale_canvas = 3 if short_side <= 256 else 1
        if scale_canvas > 1:
            canvas = cv2.resize(
                rgb_np,
                (W * scale_canvas, H * scale_canvas),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            canvas = rgb_np

        viz_scale = 1.0

        def to_canvas(pt):
            return (int(pt[0] * scale_canvas), int(pt[1] * scale_canvas))

        for i in range(xy_src.shape[0]):
            if visibility[i] <= 0.5:
                continue
            x0, y0 = xy_src[i]
            gt_dx, gt_dy = gt_motion[i]
            pr_dx, pr_dy = pred_motion[i]

            gt_norm = torch.sqrt(gt_dx**2 + gt_dy**2)
            pr_norm = torch.sqrt(pr_dx**2 + pr_dy**2)
            if gt_norm <= motion_thresh and pr_norm <= motion_thresh:
                continue

            start = to_canvas((x0, y0))
            gt_end = to_canvas((x0 + viz_scale * gt_dx, y0 + viz_scale * gt_dy))
            pr_end = to_canvas((x0 + viz_scale * pr_dx, y0 + viz_scale * pr_dy))

            gt_color = (255, 0, 0)
            pr_color = (0, 0, 255)
            outline_color = (0, 0, 0)

            def draw_arrow(end_pt, color):
                cv2.arrowedLine(canvas, start, end_pt, outline_color, 4, tipLength=0.35)
                cv2.arrowedLine(canvas, start, end_pt, color, 2, tipLength=0.25)

            draw_arrow(gt_end, gt_color)
            draw_arrow(pr_end, pr_color)

        if scale_canvas > 1:
            canvas = cv2.resize(canvas, (W, H), interpolation=cv2.INTER_AREA)

        return rearrange(canvas, "h w c -> c h w")

    @torch.no_grad()
    def validation_step(
        self, batch, batch_idx, namespace: str = "validation"
    ) -> STEP_OUTPUT:
        """
        Extend DFoTMotionPolicy validation with Jacobian visualizations.
        """
        # Let DFoT handle its usual validation (denoising + conditional generation).
        # DFoTMotionPolicy.on_after_batch_transfer returns a 5-tuple, so Lightning
        # will pass that into validation_step. Keep compatibility with both:
        # - batch is a dict (rare, if called manually)
        # - batch is a tuple (normal path through Lightning)
        out = super().validation_step(batch, batch_idx, namespace=namespace)  # type: ignore[misc]

        # Guard visualization writes to global rank zero in DDP.
        if not (getattr(self.trainer, "is_global_zero", False) and self.logger):  # type: ignore[attr-defined]
            return out
        if batch_idx != 0:
            # Keep logging volume reasonable (JacobianVanilla also focuses on first batch)
            return out

        # Get DFoT-processed rgb tokens (the 5th element of the tuple).
        if isinstance(batch, tuple) and len(batch) == 5:
            xs, xs_mask, conditions, masks, rgb_tokens = batch
        else:
            # Fallback: if we were passed a dict, process it once.
            xs, xs_mask, conditions, masks, rgb_tokens = super().on_after_batch_transfer(  # type: ignore[misc]
                batch, dataloader_idx=0
            )

        # Use the raw dataloader batch (stashed in on_after_batch_transfer) for du/flow/tracks.
        raw = getattr(self, "_latest_control_batch", None)
        if raw is None or not isinstance(raw, dict):
            return out
        du = raw.get("du")
        rgb = raw.get("rgb")
        flow_gt = raw.get("flow")
        tracks = raw.get("tracks")

        if du is None or rgb is None:
            return out

        # Lazily ensure head is built (use raw batch for cmd_dim fallback).
        self._build_jacobian_head_if_needed(rgb_tokens, raw)
        if self._jacobian_head is None:
            return out
        if not self.joint_jacobian_cfg.enable:
            return out

        # For visualization, only use a small subset: K batches, per-view.
        if rgb.ndim == 6:
            B, T, V, C, H, W = rgb.shape
        else:
            B, T, C, H, W = rgb.shape
            V = 1

        K = min(B, 3)

        dataset_meta = getattr(self, "dataset_metadata", {}) or {}

        # Detect DFoT's time->batch collapse (common when dataset T>1 but max_frames==1)
        collapse_time_into_batch = self.max_frames == 1 and T > 1  # type: ignore[attr-defined]

        def token_index(b: int, t: int, v: int) -> int:
            base = b * V + v
            return base * T + t if collapse_time_into_batch else base

        # Clamp context length for head (avoid context_length=-1 producing empty slices)
        n_ctx = max(int(self.n_context_frames), 1)  # type: ignore[attr-defined]

        view_names = None
        if isinstance(dataset_meta, dict):
            view_names = dataset_meta.get("views") or dataset_meta.get("camera_views")

        for b in range(K):
            for v in range(V):
                view_name = (
                    view_names[v]
                    if isinstance(view_names, list) and v < len(view_names)
                    else f"view{v}"
                )

                vis_dict: Dict[str, list[np.ndarray]] = {
                    "video/context": [],
                    "video/jacobian": [],
                    "video/pred_flow": [],
                    "video/gt_flow": [],
                    "video/track_motion": [],
                }

                track_keep_indices: Optional[Tensor] = None
                if tracks is not None:
                    xy_src_0, _, _, _ = self._get_tracks_at_frame(tracks, b, v, 0, V)
                    track_keep_indices = self._compute_track_keep_indices(
                        xy_src_0, H, W, sparsity=300
                    )

                for t in range(T):
                    rgb_btv = rgb[b, t, v] if V > 1 else rgb[b, t]
                    du_bt = du[b, t]
                    flow_gt_bt = None
                    if flow_gt is not None:
                        flow_gt_bt = flow_gt[b, t, v] if V > 1 else flow_gt[b, t]

                    idx = token_index(b, t, v)
                    rgb_tok_bt = rgb_tokens[idx : idx + 1]  # [1, T_tok, C, H, W]

                    # Predict Jacobian for this (b,t,v) using a feature source that
                    # matches the head's expected channel count. Prefer backbone
                    # features when available and channel-compatible; otherwise
                    # fall back to DFoT RGB tokens.
                    ctx_T = max(1, min(n_ctx, rgb_tok_bt.shape[1]))
                    if (
                        self._latest_backbone_features is not None
                        and self._jacobian_head_in_channels is not None
                        and self._latest_backbone_features.shape[2]
                        == self._jacobian_head_in_channels
                    ):
                        feat_bt = self._latest_backbone_features[idx : idx + 1][
                            :, :ctx_T
                        ]
                    else:
                        feat_bt = rgb_tok_bt[:, :ctx_T]

                    J_flat = self._jacobian_head(feat_bt)  # type: ignore[operator]
                    J_last = J_flat[:, -1]  # [1, cmd_dim*s_dim, H, W]

                    cmd_dim = self._jacobian_cmd_dim or du_bt.shape[-1]
                    s_dim = self._jacobian_spatial_dim
                    J = rearrange(
                        J_last,
                        "b (c s) h w -> b c s h w",
                        c=cmd_dim,
                        s=s_dim,
                    )

                    flow_pred_bt = einsum(
                        J,
                        du_bt[None],
                        "b c s h w, b c -> b s h w",
                    )[0]

                    vis_pred_flow = flow_to_image(
                        denormalize_flow_tensor(
                            flow_pred_bt.float(),
                            dataset_meta,
                        ).cpu()
                    ).numpy()
                    vis_gt_flow = (
                        flow_to_image(
                            denormalize_flow_tensor(
                                flow_gt_bt.float(),
                                dataset_meta,
                            ).cpu()
                        ).numpy()
                        if flow_gt_bt is not None
                        else np.zeros_like(vis_pred_flow)
                    )

                    denorm_jac = denormalize_jacobian_tensor(
                        J,
                        dataset_meta,
                        cmd_dim=cmd_dim,
                    )

                    vis_jac = jacobian_utils.visualize_jacobian(
                        denorm_jac,
                        robot_name=dataset_meta.get("robot_name", ""),
                        flow_scale=0.2,
                    )

                    # Track visualization (if available)
                    if tracks is not None and flow_gt_bt is not None:
                        xy_src, pixel_selector, gt_disp, visibility = (
                            self._get_tracks_at_frame(tracks, b, v, t, V)
                        )
                        flow_pred_hw = rearrange(
                            flow_pred_bt[None], "b c h w -> b h w c"
                        )  # (1,H,W,2)
                        flow_flat = flow_pred_hw.view(1, -1, 2)
                        pred_disp = torch.gather(
                            flow_flat,
                            dim=1,
                            index=pixel_selector[None, :, None].long().expand(1, -1, 2),
                        )[0]
                        track_vis = self.visualize_pixel_motion(
                            xy_src=xy_src,
                            gt_pixel_motion=gt_disp,
                            pred_pixel_motion=pred_disp,
                            gt_pixel_visibility=visibility,
                            rgb=rgb_btv,
                            sparsity=300,
                            keep_indices=track_keep_indices,
                        )
                    else:
                        track_vis = np.zeros((3, H, W), dtype=np.uint8)

                    rgb_np = (rgb_btv.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

                    vis_dict["video/context"].append(rgb_np)
                    vis_dict["video/jacobian"].append(vis_jac)
                    vis_dict["video/pred_flow"].append(vis_pred_flow)
                    vis_dict["video/gt_flow"].append(vis_gt_flow)
                    vis_dict["video/track_motion"].append(track_vis)

                log_dict = {
                    f"joint_jacobian_vis/{k}/b{b}/{view_name}": wandb.Video(
                        np.stack(vframes, axis=0),
                        fps=12,
                        format="mp4",
                        caption=(
                            "red: gt | blue: predicted"
                            if k == "video/track_motion"
                            else ""
                        ),
                    )
                    for k, vframes in vis_dict.items()
                }

                self.logger.experiment.log(  # type: ignore[attr-defined]
                    {**log_dict, "trainer/global_step": self.global_step}  # type: ignore[attr-defined]
                )

        return out

    # -------------------------------------------------------------------------
    # Ensure we don't accidentally apply Jacobian loss during denoising eval
    # (where training_step is called internally on synthetic batches).
    # -------------------------------------------------------------------------

    def _eval_denoising(self, batch, batch_idx, namespace: str = "training") -> None:
        self._skip_jacobian = True
        try:
            return super()._eval_denoising(batch, batch_idx, namespace=namespace)  # type: ignore[misc]
        finally:
            self._skip_jacobian = False

    # -------------------------------------------------------------------------
    # Backbone feature hooks (UNet3D mid-block / DiT3D deep tokens)
    # -------------------------------------------------------------------------

    def _ensure_backbone_feature_hooks(self) -> None:
        """
        Install lightweight forward hooks on the DFoT diffusion backbone so that
        mid-level features are available for the Jacobian decoder, without
        changing the backbone API or behavior for other algorithms.
        """
        if self._backbone_feature_hook_handle is not None:
            return

        diffusion = getattr(self, "diffusion_model", None)
        if diffusion is None:
            return

        backbone = getattr(diffusion, "model", None)
        if backbone is None:
            if getattr(self, "global_rank", 0) == 0:
                print(
                    "[DFoTJointMotionJacobian] diffusion_model has no .model; "
                    "backbone feature hooks not installed."
                )
            return

        # UNet3D: hook the bottleneck (mid_block) output: [B, C, T, H', W'].
        if isinstance(backbone, Unet3D):

            def _unet_mid_hook(module, inputs, output):
                # output: [B, C, T, H, W] -> store as [B, T, C, H, W]
                with torch.no_grad():
                    feat = rearrange(output, "b c t h w -> b t c h w")
                    self._latest_backbone_features = feat

            self._backbone_feature_hook_handle = (
                backbone.mid_block.register_forward_hook(_unet_mid_hook)
            )
            if getattr(self, "global_rank", 0) == 0:
                print(
                    "[DFoTJointMotionJacobian] Installed UNet3D mid_block hook "
                    f"(network_size={backbone.cfg.network_size})."
                )
            return

        # DiT3D: hook the final token representation before unpatchify.
        if isinstance(backbone, DiT3D):

            def _dit_final_hook(module, inputs, output):
                # output: [B, N, OC] where N = T * num_patches, OC = patch_size^2 * C
                # Decode to [B, T, C, H, W] using the backbone's unpatchify helper.
                with torch.no_grad():
                    tokens = output
                    if tokens is None:
                        return
                    B, N, C_tok = tokens.shape
                    num_patches = backbone.num_patches
                    if num_patches is None or N % num_patches != 0:
                        return
                    T = N // num_patches
                    tokens_btpc = rearrange(
                        tokens, "b (t p) c -> (b t) p c", p=num_patches
                    )
                    frames_bt_hw_c = backbone.unpatchify(tokens_btpc)
                    feat = rearrange(
                        frames_bt_hw_c,
                        "(b t) h w c -> b t c h w",
                        b=B,
                        t=T,
                    )
                    self._latest_backbone_features = feat

            self._backbone_feature_hook_handle = (
                backbone.dit_base.final_layer.register_forward_hook(_dit_final_hook)
            )
            if getattr(self, "global_rank", 0) == 0:
                print(
                    "[DFoTJointMotionJacobian] Installed DiT3D final_layer hook "
                    f"(hidden_size={backbone.cfg.hidden_size}, "
                    f"depth={backbone.cfg.depth}, num_heads={backbone.cfg.num_heads})."
                )
            return

        # For unsupported backbones, we simply skip hooks and fall back to DFoT
        # RGB tokens (handled in training_step).
        if getattr(self, "global_rank", 0) == 0:
            print(
                "[DFoTJointMotionJacobian] Backbone type does not support hooks; "
                "falling back to DFoT RGB tokens for Jacobian head."
            )
