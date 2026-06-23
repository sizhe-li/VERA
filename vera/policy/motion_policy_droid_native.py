"""Clean-slate MotionPolicy for deploying okto dynamics models on the
Franka Research 3 via the DROID bridge.

Contract: the controller's raw du lives in training-normalized [-1, 1]
space (symmetric_percentile, 90th percentile per dim). This subclass
emits du unchanged. The DROID runner on NORA denormalizes using the
action_abs_scale advertised on the websocket handshake and applies its
own safety clipping before sending to polymetis.

Supports both SE3-delta (dim_u=7) and joint-delta (dim_u=8) dynamics
checkpoints. The action_mode is discovered from the dynamics model's
_loaded_normalization_metadata at init and echoed in get_wire_metadata().
"""

from dataclasses import dataclass
import json
from typing import Any

import torch
from einops import rearrange
from torch import Tensor

from vera.datasets.normalization import compute_jacobian_action_scales

from .base_policy import PolicyObservation
from .motion_policy import MotionPolicy, MotionPolicyCfg


# Training-time view id mapping for DROID image_jacobian checkpoints.
# The training dataset (droid_se3_normalized_merged) stacks cameras in the
# order `camera.views: ["wrist", "exterior_1", "exterior_2"]` (see
# okto/configurations/dataset/camera/droid.yaml), so the image_jacobian's
# `view_ids = arange(V)` assignment bakes in:
#   view_id 0 → wrist  (hand camera)
#   view_id 1 → exterior_1 (varied camera 1)
#   view_id 2 → exterior_2 (varied camera 2)
# The server-side client keys use shorter aliases — we accept both forms
# so downstream DROID bridges aren't locked to one spelling.
_DROID_VIEW_ID_LOOKUP: dict[str, int] = {
    "wrist": 0,
    "hand": 0,
    "exterior_1": 1,
    "varied_1": 1,
    "exterior_2": 2,
    "varied_2": 2,
}

_DROID_VIEW_ALIAS_GROUPS: tuple[tuple[str, ...], ...] = (
    ("wrist", "hand"),
    ("exterior_1", "varied_1"),
    ("exterior_2", "varied_2"),
)

_DROID_VIEW_CANONICAL_LOOKUP: dict[str, str] = {
    alias: group[0] for group in _DROID_VIEW_ALIAS_GROUPS for alias in group
}


@dataclass
class MotionPolicyDroidNativeCfg(MotionPolicyCfg):
    pass


class MotionPolicyDroidNative(MotionPolicy):
    """MotionPolicy variant that emits training-normalized du directly.

    The whole point of this subclass is to *not* apply hand-crafted scale
    knobs on top of the controller output. All deployment-specific
    denormalization and clipping lives on the DROID bridge side, which
    reads the normalization metadata from the websocket handshake.
    """

    cfg: MotionPolicyDroidNativeCfg

    def __init__(self, cfg: MotionPolicyDroidNativeCfg, device: torch.device):
        super().__init__(cfg, device)
        self._per_view_jacobian_zero_action_dims: dict[str, tuple[int, ...]] = {}
        self._validate_dynamics_metadata()

    def _validate_dynamics_metadata(self) -> None:
        # FSDP workers (rank > 0) skipped dynamics loading in MotionPolicy.__init__,
        # so there is nothing to validate — they only participate in WAN collectives.
        if self.dynamics_model is None:
            return
        meta = self.get_dynamics_normalization_metadata()
        action_mode = meta.get("action_mode")
        if action_mode not in ("se3_delta", "joint_delta"):
            raise ValueError(
                f"MotionPolicyDroidNative requires dynamics checkpoint with "
                f"action_mode in {{se3_delta, joint_delta}}, got {action_mode!r}"
            )
        action_abs_scale = meta.get("action_abs_scale")
        if not action_abs_scale:
            raise ValueError(
                "MotionPolicyDroidNative requires dynamics checkpoint with "
                "action_abs_scale metadata (for handshake advertising)."
            )
        expected_dim = 7 if action_mode == "se3_delta" else 8
        if len(action_abs_scale) != expected_dim:
            raise ValueError(
                f"action_abs_scale length {len(action_abs_scale)} does not match "
                f"action_mode={action_mode} (expected {expected_dim})."
            )
        norm_mode = meta.get("action_normalization_mode")
        if norm_mode != "symmetric_percentile":
            raise ValueError(
                f"MotionPolicyDroidNative expects action_normalization_mode="
                f"symmetric_percentile, got {norm_mode!r}"
            )

    # ── Optional: zero out rx/ry Jacobian columns ─────────────────────
    # SE3 action layout: [tx, ty, tz, rx, ry, rz, gripper].
    # Set this to (3, 4) to disable rx/ry and only allow rz rotation.
    # Leave empty () for full 7-DOF solve. Does NOT apply when the
    # dynamics checkpoint uses joint_delta (dim_u=8).
    _jacobian_zero_action_dims: tuple[int, ...] = ()

    @staticmethod
    def _canonical_droid_view_key(key: str) -> str:
        return _DROID_VIEW_CANONICAL_LOOKUP.get(str(key).lower(), str(key).lower())

    @classmethod
    def _expand_alias_zero_dims(
        cls, config: dict[str, tuple[int, ...]]
    ) -> dict[str, list[int]]:
        expanded: dict[str, list[int]] = {}
        for group in _DROID_VIEW_ALIAS_GROUPS:
            canonical = group[0]
            if canonical not in config:
                continue
            dims = list(config[canonical])
            for alias in group:
                expanded[alias] = dims
        for key, dims in config.items():
            if key not in expanded:
                expanded[key] = list(dims)
        return expanded

    def _normalize_per_view_jacobian_zero_action_dims(
        self, value: Any
    ) -> dict[str, tuple[int, ...]]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise TypeError(
                "per_view_jacobian_zero_action_dims must be a dict[str, list[int]]"
            )
        normalized: dict[str, tuple[int, ...]] = {}
        for raw_key, raw_dims in value.items():
            key = self._canonical_droid_view_key(str(raw_key))
            if raw_dims is None:
                dims: tuple[int, ...] = ()
            else:
                dims = tuple(sorted({int(d) for d in raw_dims}))
            normalized[key] = dims
        return normalized

    def _effective_jacobian_zero_dims_for_view(self, view_key: str | None) -> tuple[int, ...]:
        if view_key is not None:
            canonical = self._canonical_droid_view_key(str(view_key))
            if canonical in self._per_view_jacobian_zero_action_dims:
                return self._per_view_jacobian_zero_action_dims[canonical]
        return self._jacobian_zero_action_dims

    def _effective_jacobian_zero_dims_by_view(
        self, view_keys: list[str] | None, num_views: int | None = None
    ) -> dict[str, list[int]]:
        if view_keys is None:
            count = 0 if num_views is None else int(num_views)
            return {
                f"view_{idx}": list(self._jacobian_zero_action_dims)
                for idx in range(count)
            }
        return {
            str(key): list(self._effective_jacobian_zero_dims_for_view(str(key)))
            for key in view_keys
        }

    def _mask_jacobian(
        self,
        jac: Tensor,
        *,
        view_keys: list[str] | None = None,
        num_views: int | None = None,
    ) -> Tensor:
        """Zero out configured action channels in raw jacobian (b c s h w)."""
        has_global = bool(self._jacobian_zero_action_dims)
        has_per_view = bool(self._per_view_jacobian_zero_action_dims)
        if not has_global and not has_per_view:
            return jac
        meta = self.get_dynamics_normalization_metadata()
        if meta.get("action_mode") != "se3_delta":
            return jac
        if num_views is None or int(num_views) <= 0:
            num_views = 1
        jac = jac.clone()
        if view_keys is None or len(view_keys) != int(num_views):
            dims = self._effective_jacobian_zero_dims_for_view(None)
            for ch in dims:
                if 0 <= ch < jac.shape[1]:
                    jac[:, ch, :, :, :] = 0.0
            return jac

        for view_idx, view_key in enumerate(view_keys):
            dims = self._effective_jacobian_zero_dims_for_view(str(view_key))
            if not dims:
                continue
            for ch in dims:
                if 0 <= ch < jac.shape[1]:
                    jac[view_idx::int(num_views), ch, :, :, :] = 0.0
        return jac

    def get_final_action(self, du: Tensor) -> Tensor:
        # Pass-through: the Jacobian is denormalized in
        # _compute_jacobian_from_rgb_tensor, so the solve already produces
        # du in physical units (meters / radians per step). The client
        # should NOT apply action_abs_scale again.
        return du

    def _denormalize_jacobian_for_solve(
        self, J_flat: Tensor, state_dim: int = 2
    ) -> Tensor:
        """Denormalize Jacobian from training-normalized to physical units.

        J_phys = J_norm * oflow_abs_scale / action_abs_scale

        With J_phys the solve `J_phys @ du = y_pixels` produces du in
        physical units directly. Same math as _denormalize_vis_jacobian_flat.
        """
        if J_flat.ndim != 3 or J_flat.shape[-2] % state_dim != 0:
            return J_flat
        meta = self.get_dynamics_normalization_metadata()
        action_abs_scale = meta.get("action_abs_scale") if meta else None
        if not action_abs_scale:
            return J_flat

        cmd_dim = int(J_flat.shape[-1])
        J = rearrange(J_flat, "b (n s) c -> b n s c", s=state_dim)

        action_scales_list = compute_jacobian_action_scales(
            action_dim=cmd_dim,
            du_scale=float(meta.get("du_scale", 1.0)),
            action_abs_scale=action_abs_scale,
            action_mean=meta.get("action_mean"),
            action_std=meta.get("action_std"),
            action_min=meta.get("action_min"),
            action_max=meta.get("action_max"),
        )
        action_scale = torch.as_tensor(
            action_scales_list, dtype=J.dtype, device=J.device
        ).view(1, 1, 1, cmd_dim)

        oflow_abs_scale = meta.get("oflow_abs_scale") if meta else None
        if oflow_abs_scale is not None:
            flow_scale = torch.as_tensor(
                oflow_abs_scale, dtype=J.dtype, device=J.device
            ).view(1, 1, state_dim, 1)
            J = J * flow_scale * action_scale
        else:
            J = J * action_scale

        return rearrange(J, "b n s c -> b (n s) c")

    def _compute_jacobian_from_rgb_tensor(self, rgb, source_view_widths, view_keys=None):
        resized_rgb, jacobian_flat, jacobian_pixel, target_size, target_view_widths = (
            super()._compute_jacobian_from_rgb_tensor(rgb, source_view_widths, view_keys=view_keys)
        )
        # Denormalize both flat and pixel Jacobians so the solve produces
        # physical du directly. No client-side denorm needed.
        jacobian_flat = self._denormalize_jacobian_for_solve(jacobian_flat)
        b, n, s, c = jacobian_pixel.shape
        jp_flat = rearrange(jacobian_pixel, "b n s c -> b (n s) c")
        jp_flat = self._denormalize_jacobian_for_solve(jp_flat)
        jacobian_pixel = rearrange(jp_flat, "b (n s) c -> b n s c", s=s)
        return resized_rgb, jacobian_flat, jacobian_pixel, target_size, target_view_widths

    def _resolve_view_ids(
        self, view_keys: list[str] | None, num_views: int
    ) -> Tensor | None:
        # The checkpoint's image_jacobian model (config_jacobian_droid_giant_view_embedding,
        # use_view_embedding=true, num_view_embeddings=3) was trained with camera
        # order ["wrist", "exterior_1", "exterior_2"]. Map inference-time view
        # keys to those training ids; fall back to None (all-zeros / view 0) if
        # view_keys is missing or any key is unknown — never silently misassign.
        if view_keys is None or len(view_keys) != num_views:
            return None
        try:
            ids = [_DROID_VIEW_ID_LOOKUP[str(k).lower()] for k in view_keys]
        except KeyError as exc:
            raise ValueError(
                f"MotionPolicyDroidNative cannot map view_key {exc.args[0]!r} to a "
                f"training view_id. Known keys: {sorted(_DROID_VIEW_ID_LOOKUP)}"
            ) from exc
        return torch.tensor(ids, dtype=torch.long)

    AVAILABLE_TRACKER_BACKENDS: tuple[str, ...] = ("alltracker", "cotracker", "megaflow")

    def get_wire_metadata(self) -> dict[str, Any]:
        meta = self.get_dynamics_normalization_metadata()
        action_abs_scale = [float(x) for x in meta["action_abs_scale"]]
        return {
            "action_mode": str(meta["action_mode"]),
            "action_abs_scale": action_abs_scale,
            "action_normalization_mode": str(
                meta.get("action_normalization_mode", "symmetric_percentile")
            ),
            "action_percentile": float(meta.get("action_percentile", 90.0)),
            "dim_u": len(action_abs_scale),
            "gripper_dim_index": -1,
            "robot_name": str(
                getattr(self, "robot_name", meta.get("robot_name", ""))
            ),
            "actions_already_metric": True,
            # Per-ckpt runtime budget — Nora reads these to size its
            # refill-between-chunks and avoid hardcoding 9/16/24 per recipe.
            "context_frames": int(self.context_frames),
            "action_chunk_horizon": int(self.cfg.action_chunk_horizon),
            "n_action_steps": int(self.cfg.n_action_steps),
            "available_tracker_backends": list(self.AVAILABLE_TRACKER_BACKENDS),
            "current_tracker_backend": str(self.cfg.motion_planner.tracker_backend),
            "jacobian_zero_action_dims": list(self._jacobian_zero_action_dims),
            "per_view_jacobian_zero_action_dims": self._expand_alias_zero_dims(
                self._per_view_jacobian_zero_action_dims
            ),
        }

    # ── Per-view Jacobian solve weights ──────────────────────────────────
    # Downweight exterior cameras relative to the wrist camera so the
    # least-squares solver favours the wrist view's gradients.
    # Keys follow _DROID_VIEW_ID_LOOKUP; default weight is 1.0 (wrist).
    _DROID_VIEW_SOLVE_WEIGHTS: dict[str, float] = {
        "wrist": 1.0,
        "hand": 1.0,
        # 10x boost (was 30): with the measured ‖J_ext‖ ~= 2.5 and typical
        # ext ‖y‖ ~= 0.1 px vs wrist ‖J‖ ~= 80, ‖y‖ ~= 10 px, the
        # contribution to JᵀW²y scales as w²·‖J‖·‖y‖. A 10x ext weight
        # keeps ext from being amplified by noise while still letting it
        # override the wrist when its signal is real. For a more
        # principled fix, toggle _per_view_flow_normalize=True below.
        "exterior_1": 10.0,
        "varied_1": 10.0,
        "exterior_2": 10.0,
        "varied_2": 10.0,
    }

    def _view_weight_for_key(self, key: str) -> float:
        return self._DROID_VIEW_SOLVE_WEIGHTS.get(str(key).lower(), 1.0)

    # Per-view flow normalization: if True, rescale each view's target `y`
    # so that its L2 norm = 1 (per-batch). Decouples the solver's treatment
    # of a view from that view's proximity to the scene. With this on, the
    # _DROID_VIEW_SOLVE_WEIGHTS become pure "how much do I trust this view"
    # and no longer need to compensate for depth asymmetry.
    _per_view_flow_normalize: bool = False

    def _preprocess_solve_inputs(
        self,
        J,
        y,
        weights,
        *,
        path_kind: str,
        target_view_widths=None,
        track_xs=None,
    ):
        if not self._per_view_flow_normalize:
            return J, y, weights
        if target_view_widths is None:
            return J, y, weights
        if path_kind == "track" and track_xs is None:
            return J, y, weights

        # Build a per-element "view id" for y (flat [B, N*S]).
        B, N_flat = y.shape
        device, dtype = y.device, y.dtype
        if path_kind == "flow":
            # y layout: [H, W_total, S] flattened with S=2 (xy interleaved
            # per column — see "b c h w -> b (h w c)" upstream). We need
            # a per-element view id.
            H = N_flat // (sum(target_view_widths) * 2)
            # If the layout doesn't match, skip normalization rather than
            # corrupt the solve.
            if H * sum(target_view_widths) * 2 != N_flat:
                return J, y, weights
            view_ids = torch.zeros(N_flat, device=device, dtype=torch.long)
            # Build per-column view ids then tile across H and interleave S.
            col_view = torch.zeros(sum(target_view_widths), device=device, dtype=torch.long)
            start = 0
            for v_idx, w in enumerate(target_view_widths):
                col_view[start:start+w] = v_idx
                start += w
            per_hw = col_view.unsqueeze(0).expand(H, -1).reshape(-1)
            view_ids = per_hw.repeat_interleave(2)  # interleave [x,y] pair
        else:  # track
            # y layout: [N, S] flattened to [N*S] interleaved
            xs = track_xs[0] if track_xs.ndim > 1 else track_xs
            xs = xs.to(device)
            col_starts = torch.zeros(len(target_view_widths)+1, device=device, dtype=torch.long)
            for i, w in enumerate(target_view_widths):
                col_starts[i+1] = col_starts[i] + w
            track_view = torch.zeros(len(xs), device=device, dtype=torch.long)
            for v_idx in range(len(target_view_widths)):
                m = (xs >= col_starts[v_idx].item()) & (xs < col_starts[v_idx+1].item())
                track_view = torch.where(m, torch.full_like(track_view, v_idx), track_view)
            view_ids = track_view.repeat_interleave(2)  # N*S

        # Compute per-view L2 norm of y (one scalar per view per batch).
        y_scaled = y.clone()
        for v_idx in range(len(target_view_widths)):
            mask = (view_ids == v_idx)
            if not mask.any():
                continue
            y_v = y[:, mask]
            denom = torch.linalg.vector_norm(y_v, dim=-1, keepdim=True).clamp_min(1e-6)
            y_scaled[:, mask] = y_v / denom
        return J, y_scaled, weights

    _TRACKER_BACKEND_KEYS: tuple[str, ...] = (
        "tracker_backend",
        "alltracker_rate",
        "alltracker_query_frame",
        "alltracker_inference_iters",
        "alltracker_conf_thr",
        "alltracker_bkg_opacity",
        "cotracker_model_name",
        "cotracker_grid_size",
        "megaflow_model_name",
        "megaflow_num_reg_refine",
        "megaflow_query_frame",
        "megaflow_vis_from_flow_mag",
        "megaflow_vis_flow_mag_thresh",
        "megaflow_autocast_dtype",
    )

    def configure_runtime(self, **kwargs) -> dict[str, Any]:
        """Extend base configure_runtime with DROID-specific knobs:
        - view_solve_weights: dict[str, float] — per-view solve weights
          (keys use _DROID_VIEW_ID_LOOKUP names: wrist/hand, exterior_1,
          exterior_2 — aliases varied_1/varied_2 also accepted).
        - jacobian_zero_action_dims: list[int] — dims to zero in J before
          solve. For SE3: dims 3,4 = rx,ry. No effect for joint_delta.
        - per_view_jacobian_zero_action_dims: dict[str, list[int]] —
          per-view override for jacobian_zero_action_dims. View aliases:
          wrist=hand, exterior_1=varied_1, exterior_2=varied_2.
        - debug_dump_enabled: bool — toggle per-chunk .npz dumps.
        - debug_dump_dir: str — override dump directory path.
        - tracker_backend / alltracker_* / cotracker_* / megaflow_* — hot-swap
          the feedback tracker. Drops the cached instance so the next infer
          rebuilds it via `build_motion_tracker` with the new config.
        """
        view_weights = kwargs.pop("view_solve_weights", None)
        zero_dims = kwargs.pop("jacobian_zero_action_dims", None)
        per_view_zero_dims = kwargs.pop("per_view_jacobian_zero_action_dims", None)
        # Read but do NOT pop these — let super() configure the rich-dump
        # (Mechanism B: per-run subdir, full rgb / q_robot / per-frame
        # jacobians via _dump_policy_chunk_trajectory). The DROID-only thin
        # dump (Mechanism A) is disabled by default; left as opt-in below.
        debug_dump = kwargs.get("debug_dump_enabled", None)
        debug_dump_dir = kwargs.get("debug_dump_dir", None)
        pvfn = kwargs.pop("per_view_flow_normalize", None)

        # Pull tracker-related keys out before delegating to super so we can
        # mutate `self.cfg.motion_planner` and rebuild the cached tracker.
        tracker_kwargs: dict[str, Any] = {}
        for key in self._TRACKER_BACKEND_KEYS:
            if key in kwargs:
                tracker_kwargs[key] = kwargs.pop(key)

        applied = super().configure_runtime(**kwargs)

        if tracker_kwargs:
            requested_backend = tracker_kwargs.get("tracker_backend")
            if requested_backend is not None and requested_backend not in self.AVAILABLE_TRACKER_BACKENDS:
                raise ValueError(
                    f"Unknown tracker_backend {requested_backend!r}; "
                    f"available: {list(self.AVAILABLE_TRACKER_BACKENDS)}"
                )
            planner_cfg = self.cfg.motion_planner
            tracker_applied: dict[str, Any] = {}
            for key, value in tracker_kwargs.items():
                if hasattr(planner_cfg, key):
                    setattr(planner_cfg, key, value)
                    tracker_applied[key] = value
            # Drop the cached feedback tracker so the next call rebuilds with
            # the new backend / params. `_get_feedback_tracker` is the single
            # consumer (motion_policy_adaptive.py:424-434).
            prev_tracker = getattr(self, "_feedback_tracker", None)
            if prev_tracker is not None:
                tracker_applied["tracker_rebuilt_from"] = type(prev_tracker).__name__
                self._feedback_tracker = None
                del prev_tracker
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            tracker_applied["current_tracker_backend"] = str(planner_cfg.tracker_backend)
            applied["tracker"] = tracker_applied
            print(f"[configure_runtime] tracker swapped: {tracker_applied}", flush=True)

        if view_weights is not None:
            # Copy the class-level dict into an instance-level override so
            # we can mutate without affecting other instances.
            merged = dict(MotionPolicyDroidNative._DROID_VIEW_SOLVE_WEIGHTS)
            for k, v in dict(view_weights).items():
                merged[str(k).lower()] = float(v)
            self._DROID_VIEW_SOLVE_WEIGHTS = merged  # shadows class attr
            applied["view_solve_weights"] = merged

        if zero_dims is not None:
            self._jacobian_zero_action_dims = tuple(int(d) for d in zero_dims)
            applied["jacobian_zero_action_dims"] = list(self._jacobian_zero_action_dims)

        if per_view_zero_dims is not None:
            self._per_view_jacobian_zero_action_dims = (
                self._normalize_per_view_jacobian_zero_action_dims(per_view_zero_dims)
            )
            applied["per_view_jacobian_zero_action_dims"] = self._expand_alias_zero_dims(
                self._per_view_jacobian_zero_action_dims
            )

        # Mechanism A (thin per-chunk summary) is disabled by default — its
        # payload is a strict subset of Mechanism B's rich dump. Kept here
        # purely as a no-op pass-through for the configure response.
        if debug_dump is not None:
            applied["debug_dump_enabled"] = bool(debug_dump)
        if debug_dump_dir is not None:
            applied["debug_dump_dir"] = str(debug_dump_dir)

        if pvfn is not None:
            self._per_view_flow_normalize = bool(pvfn)
            applied["per_view_flow_normalize"] = bool(pvfn)

        return applied

    def _log_effective_jacobian_zero_dims(self, *, kind: str, obs: PolicyObservation) -> None:
        view_keys = list(obs.view_keys or [])
        payload = {
            "kind": kind,
            "view_keys": view_keys,
            "effective_zero_dims_by_view": self._effective_jacobian_zero_dims_by_view(
                view_keys,
                len(view_keys),
            ),
            "global_jacobian_zero_action_dims": list(self._jacobian_zero_action_dims),
            "per_view_jacobian_zero_action_dims": self._expand_alias_zero_dims(
                self._per_view_jacobian_zero_action_dims
            ),
        }
        text = json.dumps(payload, sort_keys=True)
        print(f"[jacobian-zero-dims] {text}", flush=True)

    def _control_motion_mask(
        self,
        obs,
        height: int,
        view_widths: list[int],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Binary control mask weighted by per-view solve weights."""
        base_mask = super()._control_motion_mask(
            obs, height, view_widths, device=device, dtype=dtype,
        )
        if obs.view_keys is None or len(obs.view_keys) != len(view_widths):
            return base_mask
        # Build a per-pixel weight vector matching the stitched width.
        col_weights = torch.ones(sum(view_widths), device=device, dtype=dtype)
        start = 0
        for key, w in zip(obs.view_keys, view_widths):
            col_weights[start : start + w] = self._view_weight_for_key(key)
            start += w
        # Expand to (H*W*2,) — same layout as base_mask (repeat_interleave(2) for xy).
        pixel_weights = col_weights.unsqueeze(0).expand(height, -1).reshape(-1)
        pixel_weights = pixel_weights.repeat_interleave(2)
        return base_mask * pixel_weights

    def _control_track_mask(
        self,
        obs,
        x_coords: Tensor,
        view_widths: list[int],
    ) -> Tensor:
        """Binary control track mask weighted by per-view solve weights."""
        base_mask = super()._control_track_mask(obs, x_coords, view_widths)
        if obs.view_keys is None or len(obs.view_keys) != len(view_widths):
            return base_mask
        # x_coords: [B, N] pixel x-positions. Assign each track point
        # the weight of the view it falls into.
        weights = torch.ones_like(base_mask)
        start = 0
        for key, w in zip(obs.view_keys, view_widths):
            end = start + w
            view_w = self._view_weight_for_key(key)
            in_view = (x_coords >= float(start)) & (x_coords < float(end))
            weights = torch.where(in_view, torch.full_like(weights, view_w), weights)
            start = end
        return base_mask * weights

    def _denormalize_vis_jacobian_flat(
        self, J_flat: Tensor, state_dim: int = 2
    ) -> Tensor:
        # Manual-broadcast denormalization mirroring
        # motion_policy_allegro._denormalize_allegro_jacobian_pixel — delegating
        # to normalization.denormalize_jacobian_tensor does not work here because
        # that helper's _flow_channel_stat_tensor places the flow-scale vector at
        # shape[-3], which for a 4D [b, n, s, c] tensor points at the pixel-count
        # axis (n), not the flow-component axis (s), and the broadcast fails.
        if J_flat.ndim != 3 or J_flat.shape[-2] % state_dim != 0:
            return J_flat
        meta = self.get_dynamics_normalization_metadata()
        action_abs_scale = meta.get("action_abs_scale") if meta else None
        if not action_abs_scale:
            return J_flat

        cmd_dim = int(J_flat.shape[-1])
        # [b, n*s, c] → [b, n, s, c]
        J = rearrange(J_flat, "b (n s) c -> b n s c", s=state_dim)

        # J_phys = J_model * (oflow_abs_scale / action_abs_scale).
        # `compute_jacobian_action_scales` already returns the reciprocal
        # `1/action_abs_scale` (scaled by any du_scale pre-factor), so we
        # multiply by it and by oflow_abs_scale directly.
        action_scales_list = compute_jacobian_action_scales(
            action_dim=cmd_dim,
            du_scale=float(meta.get("du_scale", 1.0)),
            action_abs_scale=action_abs_scale,
            action_mean=meta.get("action_mean"),
            action_std=meta.get("action_std"),
            action_min=meta.get("action_min"),
            action_max=meta.get("action_max"),
        )
        action_scale = torch.as_tensor(
            action_scales_list, dtype=J.dtype, device=J.device
        ).view(1, 1, 1, cmd_dim)

        oflow_abs_scale = meta.get("oflow_abs_scale") if meta else None
        if oflow_abs_scale is not None:
            flow_scale = torch.as_tensor(
                oflow_abs_scale, dtype=J.dtype, device=J.device
            ).view(1, 1, state_dim, 1)
            J = J * flow_scale * action_scale
        else:
            J = J * action_scale

        return rearrange(J, "b n s c -> b (n s) c")

    def _denormalize_vis_frames(
        self, vis_frames: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        # No-op: the Jacobian is already denormalized in
        # _compute_jacobian_from_rgb_tensor, so vis frames contain
        # physical-unit Jacobians. No further denormalization needed.
        if not getattr(self, "_printed_jacobian_stats", False):
            for frame in vis_frames:
                J = frame.get("jacobian")
                if J is None:
                    continue
                print(
                    f"[MotionPolicyDroidNative] jacobian (already physical):\n"
                    f"  shape={tuple(J.shape)} "
                    f"min={float(J.min()):+.5f} max={float(J.max()):+.5f} "
                    f"abs_mean={float(J.abs().mean()):.5f}",
                    flush=True,
                )
                self._printed_jacobian_stats = True
                break
        return vis_frames

    def _flow_rgb_chunk_to_actions(
        self,
        obs: PolicyObservation,
        source_rgb: Tensor,
        target_rgb: Tensor,
        flow: Tensor,
        source_view_widths: list[int],
        lam_override: float | None = None,
    ):
        self._log_effective_jacobian_zero_dims(kind="flow", obs=obs)
        actions, vis_frames = super()._flow_rgb_chunk_to_actions(
            obs,
            source_rgb,
            target_rgb,
            flow,
            source_view_widths,
            lam_override=lam_override,
        )
        self._denormalize_vis_frames(vis_frames)
        # Mechanism A (thin DROID-only dump) removed — Mechanism B's
        # _dump_policy_chunk_trajectory is the canonical recorder
        # (rgb + q_robot + per-frame jacobians, per-run subdir).
        return actions, vis_frames

    def _track_rgb_chunk_to_actions(
        self,
        obs: PolicyObservation,
        source_rgb: Tensor,
        tracks: dict,
        source_view_widths: list[int],
        target_rgb: Tensor | None = None,
        lam_override: float | None = None,
    ):
        self._log_effective_jacobian_zero_dims(kind="track", obs=obs)
        actions, vis_frames = super()._track_rgb_chunk_to_actions(
            obs,
            source_rgb,
            tracks,
            source_view_widths,
            target_rgb=target_rgb,
            lam_override=lam_override,
        )
        self._denormalize_vis_frames(vis_frames)
        # Mechanism A (thin DROID-only dump) removed — Mechanism B's
        # _dump_policy_chunk_trajectory is the canonical recorder.
        return actions, vis_frames

    # ── Debug dump ────────────────────────────────────────────────────
    # Mechanism A (thin per-chunk dump) is deprecated; default off. Mechanism
    # B (motion_policy._dump_policy_chunk_trajectory) is the canonical recorder
    # — per-run `run_<ts>_pid<pid>/` subdir + rgb / q_robot / per-frame
    # jacobians + budget tracking. Configured via cfg.debug_dump_enabled +
    # cfg.debug_dump_dir (CLI in start_server_droid.py defaults both on).
    _DEBUG_DUMP_DIR = None
    _debug_chunk_idx = 0

    def _dump_debug_chunk(
        self, path_kind: str, actions: Tensor, vis_frames: list[dict], obs: Any = None
    ) -> None:
        """Save per-chunk tensors (flow/disp, jacobian, tracks, du) to disk
        for post-hoc inspection. One .npz per chunk. Disable by setting
        _DEBUG_DUMP_DIR to None."""
        import os
        import numpy as np

        if not self._DEBUG_DUMP_DIR:
            return
        os.makedirs(self._DEBUG_DUMP_DIR, exist_ok=True)

        idx = MotionPolicyDroidNative._debug_chunk_idx
        MotionPolicyDroidNative._debug_chunk_idx += 1

        payload: dict[str, Any] = {
            "path_kind": path_kind,
            "actions": actions.detach().cpu().numpy(),  # [B, T, dim_u] physical du
            "view_keys": list(obs.view_keys) if (obs is not None and obs.view_keys) else [],
            "view_weights": [self._view_weight_for_key(k) for k in (obs.view_keys or [])] if obs is not None else [],
        }
        if vis_frames:
            frame0 = vis_frames[0]
            # Jacobian: already physical after _denormalize_vis_frames (no-op now)
            J = frame0.get("jacobian")
            if J is not None:
                payload["jacobian_first"] = J.detach().cpu().numpy()
            # Flow or disp
            if "flow" in frame0 and frame0["flow"] is not None:
                payload["flow_first"] = frame0["flow"].detach().cpu().numpy()
            if "curr_track" in frame0 and frame0["curr_track"] is not None:
                payload["curr_track_first"] = frame0["curr_track"].detach().cpu().numpy()
            if "trgt_track" in frame0 and frame0["trgt_track"] is not None:
                payload["trgt_track_first"] = frame0["trgt_track"].detach().cpu().numpy()
            # Per-timestep action summary stats (cheaper than full tensors)
            payload["num_vis_frames"] = len(vis_frames)

        # Summaries for quick grep
        acts_np = payload["actions"]
        action_stats = {
            "dim": int(acts_np.shape[-1]),
            "T": int(acts_np.shape[1]) if acts_np.ndim >= 2 else 1,
            "abs_mean_per_dim": acts_np.reshape(-1, acts_np.shape[-1]).__abs__().mean(0).tolist(),
            "mean_per_dim": acts_np.reshape(-1, acts_np.shape[-1]).mean(0).tolist(),
            "max_abs_per_dim": acts_np.reshape(-1, acts_np.shape[-1]).__abs__().max(0).tolist(),
        }
        payload["action_stats"] = action_stats

        out_path = os.path.join(
            self._DEBUG_DUMP_DIR, f"chunk_{idx:05d}_{path_kind}.npz"
        )
        try:
            np.savez_compressed(out_path, **{k: np.asarray(v, dtype=object) if isinstance(v, (list, dict)) else v for k, v in payload.items()})
            # Print a concise one-liner so it also shows up in tmux log
            print(
                f"[debug-dump] chunk={idx:05d} kind={path_kind} "
                f"du_abs_mean={np.mean(action_stats['abs_mean_per_dim']):.4f} "
                f"du_mean_per_dim={['%+.4f' % v for v in action_stats['mean_per_dim']]} "
                f"view_keys={payload['view_keys']} "
                f"view_weights={payload['view_weights']} "
                f"-> {out_path}",
                flush=True,
            )
        except Exception as exc:  # pragma: no cover
            print(f"[debug-dump] WARN failed to save {out_path}: {exc}", flush=True)
