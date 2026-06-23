"""Per-embodiment inverse-dynamics action models (``du`` derivation).

This is the SELF-CONTAINED, REDUCED port of okto's ``ActionModel`` family
(``project/okto/datasets/action/loaders/action_loader.py``). okto has ~12 action
classes selected per-dataset by ``RobomimicDataset._resolve_action_mode`` /
``DroidDataset`` / ``DrakeAllegroDataset`` / ``PushTDataset``; for public release we
keep ONLY the models actually used for the four supported embodiments, behind a
single tiny :class:`ActionModel` ABC and a config-driven :func:`resolve_action_model`
selector (mirroring okto's per-dataset ``self.action_model = ...`` choice).

The math in each ``compute`` is a faithful port of the cited okto class — the only
structural change is that trajectory reads go through a ``loader`` object
(``loader.load_trajectory(episode, key, row_indices=None) -> np.ndarray``) instead of
okto's ``meta``-keyed packed/h5 IO, and trajectory KEYS + SCALES are read from the
dataset ``cfg`` rather than hardcoded inside the class. This keeps the models general
over embodiment: nothing okto-specific (no ``import okto``) leaks into the data path.

Returns: each ``compute`` yields a ``np.ndarray`` of shape ``[N, action_dim]`` (float64;
the dataset casts to torch float32 and applies ``du_scale`` + action normalization).
"""

from __future__ import annotations

import abc
from typing import Any, List, Optional, Sequence

import numpy as np


# --------------------------------------------------------------------------
# SE(3) geometry helpers — ported verbatim from okto action_loader.py
#   pose_to_matrix       (action_loader.py:418)
#   euler_pose_to_matrix (action_loader.py:426)
#   matrix_to_twist      (action_loader.py:439)
# (Previously lived in vera/datasets/base.py as _pose_to_matrix/_matrix_to_twist;
#  de-duplicated here — they belong with the SE3 action models.)
# --------------------------------------------------------------------------
def pose_to_matrix(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """(3,), (4,) -> (4,4) homogeneous transform. okto ``pose_to_matrix``."""
    from scipy.spatial.transform import Rotation as R

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_quat(quat).as_matrix()
    T[:3, 3] = pos
    return T


def euler_pose_to_matrix(cartesian_pose: np.ndarray) -> np.ndarray:
    """(6,) xyz + euler_xyz -> (4,4) homogeneous transform. okto ``euler_pose_to_matrix``."""
    from scipy.spatial.transform import Rotation as R

    cartesian_pose = np.asarray(cartesian_pose, dtype=np.float64)
    if cartesian_pose.shape[-1] != 6:
        raise ValueError(f"Expected 6D DROID cartesian pose, got shape {cartesian_pose.shape}")
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_euler("xyz", cartesian_pose[3:6]).as_matrix()
    T[:3, 3] = cartesian_pose[:3]
    return T


def matrix_to_twist(dT: np.ndarray) -> np.ndarray:
    """SE(3) delta -> (6,) [v, omega(rotvec)]. okto ``matrix_to_twist``."""
    from scipy.spatial.transform import Rotation as R

    omega = R.from_matrix(dT[:3, :3]).as_rotvec()
    v = dT[:3, 3]
    return np.concatenate([v, omega], axis=0)


# Backward-compat aliases for the names previously exported from base.py.
_pose_to_matrix = pose_to_matrix
_matrix_to_twist = matrix_to_twist


# --------------------------------------------------------------------------
# DROID transition-pair selection (fixed + time-aware).
# Ported from okto action_loader.py:803 (_resolve_fixed_transition_pairs) and
# :819 (_resolve_time_aware_transition_pairs).
# --------------------------------------------------------------------------
def _resolve_fixed_transition_pairs(timesteps, horizon: int, linearize: int):
    ts = np.asarray(timesteps, dtype=np.int64)
    ts1 = ts + int(linearize)
    if ts.size == 0:
        raise ValueError("timesteps is empty")
    if ts.min() < 0:
        raise ValueError(f"Negative timestep encountered: min={ts.min()}")
    if ts1.max() >= horizon:
        raise RuntimeError(
            f"Timesteps out of bounds: max(t+linearize)={ts1.max()}, horizon={horizon}"
        )
    return ts, ts1


def _resolve_time_aware_transition_pairs(
    timesteps, horizon: int, linearize: int, timestamp_ns: np.ndarray, target_delta_sec
):
    ts = np.asarray(timesteps, dtype=np.int64)
    if ts.size == 0:
        raise ValueError("timesteps is empty")
    if ts.min() < 0:
        raise ValueError(f"Negative timestep encountered: min={ts.min()}")
    timestamp_ns = np.asarray(timestamp_ns, dtype=np.int64).reshape(-1)
    if timestamp_ns.shape[0] != horizon:
        raise RuntimeError(
            "Timestamp horizon mismatch: "
            f"timestamp_ns.shape[0]={timestamp_ns.shape[0]}, horizon={horizon}"
        )
    if target_delta_sec is None:
        diffs = np.diff(timestamp_ns)
        positive_diffs = diffs[diffs > 0]
        if positive_diffs.size == 0:
            raise RuntimeError("Cannot infer target delta from non-increasing timestamps")
        target_delta_ns = int(np.median(positive_diffs)) * int(linearize)
    else:
        target_delta_ns = int(float(target_delta_sec) * 1e9)
    if target_delta_ns <= 0:
        raise ValueError(f"target_delta_sec must be positive, got {target_delta_sec}")
    target_timestamps = timestamp_ns[ts] + np.int64(target_delta_ns)
    ts1 = np.searchsorted(timestamp_ns, target_timestamps, side="left")
    ts1 = np.maximum(ts1, ts + 1)
    if ts1.max() >= horizon:
        raise RuntimeError(
            f"Time-aware timesteps out of bounds: max(target index)={ts1.max()}, horizon={horizon}"
        )
    return ts, ts1


# Timestamp keys for DROID time-aware delta (okto _load_droid_timestamp_ns_from_packed).
_DROID_TIMESTAMP_KEYS = {
    "control_step_start": "observation/timestamp/control/step_start",
    "control_step_end": "observation/timestamp/control/step_end",
    "control_start": "observation/timestamp/control/control_start",
    "policy_start": "observation/timestamp/control/policy_start",
    "sleep_start": "observation/timestamp/control/sleep_start",
}


# --------------------------------------------------------------------------
# ABC
# --------------------------------------------------------------------------
class ActionModel(abc.ABC):
    """Derive the inverse-dynamics target ``du`` for one episode + window.

    ``compute`` reads trajectory arrays via ``loader.load_trajectory(episode, key)``
    and returns ``[N, action_dim]`` float64 (raw; the dataset applies ``du_scale`` and
    action normalization afterwards). All embodiment specifics (keys, scales, joint
    indices) come from ``cfg`` — see :func:`resolve_action_model`.
    """

    @abc.abstractmethod
    def compute(self, loader: Any, episode: Any, timesteps: Sequence[int], cfg: Any) -> np.ndarray:
        ...


# --------------------------------------------------------------------------
# SE3-quat (Panda eef): okto RobomimicLiftSE3DeltaAction (action_loader.py:884)
# --------------------------------------------------------------------------
class SE3QuatDeltaAction(ActionModel):
    """Panda eef SE(3)-quat finite-diff twist + left-finger gripper delta.

    du = [Δx, Δθ(rotvec), Δg_left];  Δx,Δθ from SE(3) finite diff of (pos, quat),
    Δg_left = gripper_qpos[t+1, 0] - gripper_qpos[t, 0].

    Faithful port of okto ``RobomimicLiftSE3DeltaAction.compute`` (action_loader.py:904).
    Trajectory keys + scales read from cfg (action_pos_key/action_quat_key/
    action_gripper_key, se3_scale, gripper_scale).
    """

    def compute(self, loader, episode, timesteps, cfg):
        ts = np.asarray(timesteps, dtype=np.int64)
        ts1 = ts + int(cfg.linearize)
        pos = np.asarray(loader.load_trajectory(episode, cfg.action_pos_key), dtype=np.float64)
        quat = np.asarray(loader.load_trajectory(episode, cfg.action_quat_key), dtype=np.float64)
        grip = np.asarray(loader.load_trajectory(episode, cfg.action_gripper_key), dtype=np.float64)
        if ts.size == 0:
            raise ValueError("timesteps is empty")
        if ts1.max() >= pos.shape[0]:
            raise RuntimeError("Timesteps out of bounds in SE3QuatDeltaAction")
        twists = np.zeros((len(ts), 6), dtype=np.float64)
        dgrips = np.zeros((len(ts), 1), dtype=np.float64)
        for i, (t, t1) in enumerate(zip(ts.tolist(), ts1.tolist())):
            T0 = pose_to_matrix(pos[t], quat[t])
            T1 = pose_to_matrix(pos[t1], quat[t1])
            dT = T1 @ np.linalg.inv(T0)
            twists[i] = matrix_to_twist(dT)
            dgrips[i, 0] = float(grip[t1, 0] - grip[t, 0])  # left finger only
        du = np.concatenate([twists, dgrips], axis=-1)  # (N, 7)
        du[:, :6] *= float(cfg.se3_scale)
        du[:, 6:] *= float(cfg.gripper_scale)
        return du


# --------------------------------------------------------------------------
# DROID SE3 (euler-6D): okto DroidSE3DeltaAction (action_loader.py:1105)
# --------------------------------------------------------------------------
class DroidSE3DeltaAction(ActionModel):
    """DROID observed-state SE(3) delta from euler-6D cartesian + scalar gripper.

    du = [Δx, Δθ(rotvec), Δg]; cartesian state is [x,y,z, euler_xyz] and the relative
    rotation is computed geometrically via SE(3) (not euler subtraction). Optional
    time-aware delta resolves t+1 by timestamp instead of fixed +linearize.

    Faithful port of okto ``DroidSE3DeltaAction.compute`` (action_loader.py:1123).
    Keys: cfg.droid_cartesian_key / cfg.droid_gripper_key (+ time-aware via
    cfg.use_time_aware_delta / time_delta_source / target_delta_sec).
    """

    def compute(self, loader, episode, timesteps, cfg):
        cart = np.asarray(
            loader.load_trajectory(episode, cfg.droid_cartesian_key), dtype=np.float64
        )
        grip = np.asarray(
            loader.load_trajectory(episode, cfg.droid_gripper_key), dtype=np.float64
        ).reshape(-1)
        horizon = cart.shape[0]
        if cfg.use_time_aware_delta:
            ts_key = _DROID_TIMESTAMP_KEYS.get(cfg.time_delta_source)
            if cfg.time_delta_source == "robot_state":
                secs = np.asarray(
                    loader.load_trajectory(
                        episode, "observation/timestamp/robot_state/robot_timestamp_seconds"
                    ),
                    dtype=np.int64,
                )
                nanos = np.asarray(
                    loader.load_trajectory(
                        episode, "observation/timestamp/robot_state/robot_timestamp_nanos"
                    ),
                    dtype=np.int64,
                )
                timestamp_ns = secs * np.int64(1_000_000_000) + nanos
            else:
                timestamp_ns = np.asarray(
                    loader.load_trajectory(episode, ts_key), dtype=np.int64
                )
            ts, ts1 = _resolve_time_aware_transition_pairs(
                timesteps, horizon, cfg.linearize, timestamp_ns, cfg.target_delta_sec
            )
        else:
            ts, ts1 = _resolve_fixed_transition_pairs(timesteps, horizon, cfg.linearize)

        twists = np.zeros((len(ts), 6), dtype=np.float64)
        dgrips = np.zeros((len(ts), 1), dtype=np.float64)
        for i, (t, t1) in enumerate(zip(ts.tolist(), ts1.tolist())):
            T0 = euler_pose_to_matrix(cart[t])
            T1 = euler_pose_to_matrix(cart[t1])
            dT = T1 @ np.linalg.inv(T0)
            twists[i] = matrix_to_twist(dT)
            dgrips[i, 0] = float(grip[t1] - grip[t])
        du = np.concatenate([twists, dgrips], axis=-1)
        du[:, :6] *= float(cfg.se3_scale)
        du[:, 6:] *= float(cfg.gripper_scale)
        return du


# --------------------------------------------------------------------------
# Allegro qpos: okto DrakeAllegroQposDeltaAction (action_loader.py:1195)
# --------------------------------------------------------------------------
class QposDeltaAction(ActionModel):
    """Joint-space qpos finite-difference action: du = qpos[t+Δ] - qpos[t].

    Used for the Drake Allegro hand (16-DOF) or any traj with a ``qpos`` key.
    Optionally restricts to cfg.qpos_indices.

    Faithful port of okto ``DrakeAllegroQposDeltaAction.compute`` (action_loader.py:1208).
    Keys/scale: cfg.qpos_key (default "qpos"), cfg.qpos_indices, cfg.action_scale.
    """

    def compute(self, loader, episode, timesteps, cfg):
        ts = np.asarray(timesteps, dtype=np.int64)
        ts1 = ts + int(cfg.linearize)
        qpos = np.asarray(loader.load_trajectory(episode, cfg.qpos_key), dtype=np.float64)
        if ts1.max() >= qpos.shape[0]:
            raise RuntimeError(
                "Timesteps out of bounds in QposDeltaAction: "
                f"max(t+linearize)={ts1.max()}, qpos.shape[0]={qpos.shape[0]}"
            )
        if cfg.qpos_indices is not None:
            ji = np.asarray(cfg.qpos_indices, dtype=np.int64)
            q0 = qpos[ts[:, None], ji[None, :]]
            q1 = qpos[ts1[:, None], ji[None, :]]
        else:
            q0 = qpos[ts]
            q1 = qpos[ts1]
        du = (q1 - q0) * float(cfg.action_scale)
        return du


# --------------------------------------------------------------------------
# PushT pos-command delta: okto PushTPosCmdDeltaAction (action_loader.py:138)
# NOTE: PushT actions live in the zarr, NOT the packed NPZ. This model requires a
# loader exposing the zarr action/state arrays + a per-episode (start,end) global
# index map; the packed format alone is insufficient (see okto docstring). It is
# implemented for completeness but will raise a clear error if the loader cannot
# provide zarr actions — so it does not silently produce wrong du.
# --------------------------------------------------------------------------
class PushTPosCmdDeltaAction(ActionModel):
    """PushT command-frame delta: du = (action[t] - action[t-1]) / action_abs_max,
    clipped to [-1, 1]; first-frame (t=0) falls back to the state delta normalized to
    [-1, 1]. Faithful port of okto ``PushTPosCmdDeltaAction.compute`` (action_loader.py:199).

    REQUIRES zarr actions (not in the packed NPZ). The loader must expose
    ``load_pusht_zarr()`` returning an object with ``action``/``state`` arrays,
    ``episode_ends``, plus precomputed ``action_abs_max``/``q_min``/``q_max``. If
    absent, raises NotImplementedError rather than emitting wrong du.
    """

    def compute(self, loader, episode, timesteps, cfg):
        get_zarr = getattr(loader, "load_pusht_zarr", None)
        if get_zarr is None:
            raise NotImplementedError(
                "PushTPosCmdDeltaAction requires a loader with load_pusht_zarr(): PushT "
                "actions are stored in the zarr, not the packed NPZ. Wire a PushtZarr "
                "loader (cfg.pusht_zarr_root) before selecting action_mode=pusht_pos_cmd."
            )
        z = get_zarr()
        ts = np.asarray(timesteps, dtype=np.int64)
        ts1 = ts + int(cfg.linearize)
        ji = np.asarray(cfg.qpos_indices, dtype=np.int64)
        ep_idx = int(episode.episode_id) if str(episode.episode_id).isdigit() else int(
            getattr(episode, "episode_index", episode.episode_id)
        )
        ends = np.asarray(z.episode_ends)
        start = int(0 if ep_idx == 0 else ends[ep_idx - 1])
        actions = z.action
        states = z.state
        ts_g = ts + start
        ts1_g = ts1 + start
        ts_prev_g = (ts - 1) + start
        a_curr = actions[ts_g[:, None], ji[None, :]]
        valid_prev = ts >= 1
        a_prev = np.zeros_like(a_curr)
        if valid_prev.any():
            a_prev[valid_prev] = actions[ts_prev_g[valid_prev][:, None], ji[None, :]]
        abs_max = np.asarray(z.action_abs_max)
        du = (a_curr - a_prev) / (abs_max + 1e-8)
        du = np.clip(du, -1.0, 1.0)
        if (~valid_prev).any():
            mask = ~valid_prev
            q0 = states[ts_g[mask][:, None], ji[None, :]]
            q1 = states[ts1_g[mask][:, None], ji[None, :]]
            q_min = np.asarray(z.q_min)
            q_max = np.asarray(z.q_max)
            q0n = 2 * (q0 - q_min) / (q_max - q_min + 1e-8) - 1.0
            q1n = 2 * (q1 - q_min) / (q_max - q_min + 1e-8) - 1.0
            du[mask] = (q1n - q0n)
        return du * float(cfg.action_scale)


# --------------------------------------------------------------------------
# Selector — mirrors okto's per-dataset action_model choice
#   robomimic/mimicgen -> RobomimicLiftSE3DeltaAction (robomimic_dataset.py:166)
#   droid              -> DroidSE3DeltaAction (droid_dataset.py:397)
#   allegro            -> DrakeAllegroQposDeltaAction (drake_allegro_dataset.py:146)
#   pusht              -> PushTPosCmdDeltaAction (pusht_dataset.py:137)
# --------------------------------------------------------------------------
_ACTION_MODELS = {
    "se3_quat": SE3QuatDeltaAction,
    "droid_se3": DroidSE3DeltaAction,
    "qpos_delta": QposDeltaAction,
    "pusht_pos_cmd": PushTPosCmdDeltaAction,
}


def resolve_action_model(cfg: Any) -> ActionModel:
    """Select the ActionModel from ``cfg.action_mode`` (default ``se3_quat``).

    Default preserves current behavior: robomimic/mimicgen packed datasets that do
    not set ``action_mode`` get the SE3-quat model, identical to the old hardcoded
    base.py path. Mirrors okto's per-dataset ``self.action_model = ...`` selection.
    """
    mode = str(getattr(cfg, "action_mode", None) or "se3_quat")
    cls = _ACTION_MODELS.get(mode)
    if cls is None:
        raise ValueError(
            f"Unknown action_mode '{mode}'. Available: {sorted(_ACTION_MODELS)}"
        )
    return cls()
