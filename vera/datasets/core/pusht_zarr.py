"""PushT zarr action provider — the one source whose actions are NOT in the packed NPZ.

PushT (the human-teleop dataset) stores its recorded *command* actions and proprio
state in a single zarr replay buffer (``pusht_cchi_v7_replay.zarr``), with all episodes
concatenated and split by ``meta/episode_ends``. The packed NPZ carries only RGB / flow
/ per-episode state bounds — the action stream lives exclusively in the zarr. So
``PushTPosCmdDeltaAction`` (vera/datasets/core/actions.py) reads actions/state through a
``loader.load_pusht_zarr()`` provider instead of ``loader.load_trajectory``.

This is the SELF-CONTAINED port of okto ``PushTPosCmdDeltaAction.__init__``
(project/okto/datasets/action/loaders/action_loader.py:158-197): open the zarr once,
slice ``data/action`` and ``data/state`` to ``joint_indices``, compute the per-channel
``action_abs_max`` as the ``action_percentile`` (default 99.5) of |consecutive action
delta| over each episode (clipped to >= 1e-3), and the state ``q_min``/``q_max`` (from
cfg if pre-computed, else the zarr min/max). The provider object exposes exactly the
attributes the vera action model reads: ``action``, ``state``, ``episode_ends``,
``action_abs_max``, ``q_min``, ``q_max``. NO ``import okto``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np


class PushtZarr:
    """Opened-zarr view exposing the arrays + precomputed stats the PushT action model
    reads. Attributes mirror the synthetic ``_Zarr`` fixture in the action-model parity
    test so the same ``PushTPosCmdDeltaAction.compute`` path serves both."""

    def __init__(
        self,
        zarr_root: str | Path,
        joint_indices: Sequence[int],
        *,
        q_min: Optional[Sequence[float]] = None,
        q_max: Optional[Sequence[float]] = None,
        action_abs_max: Optional[Sequence[float]] = None,
        action_percentile: float = 99.5,
    ):
        import zarr

        self._z = zarr.open(str(zarr_root), mode="r")
        self.joint_indices = list(joint_indices)
        ji = np.asarray(self.joint_indices, dtype=np.int64)

        # episode boundaries (cumulative ends; episode k spans [ends[k-1], ends[k])).
        self.episode_ends = np.asarray(self._z["meta"]["episode_ends"])

        # The action/state full arrays are exposed lazily-sliced. The action model
        # indexes with [ts_global[:, None], ji[None, :]] (fancy indexing), so wrap the
        # zarr arrays in a small ndarray-like that supports that. We materialize the
        # joint-index columns up front (PushT is 2-DOF, ~25k rows -> trivial).
        self.action = np.asarray(self._z["data"]["action"])  # (N_total, A)
        self.state = np.asarray(self._z["data"]["state"])    # (N_total, A_state)

        # state bounds: cfg-precomputed if given (okto pre_q_min/pre_q_max), else zarr.
        if q_min is not None and q_max is not None:
            self.q_min = np.asarray(q_min, dtype=np.float32)
            self.q_max = np.asarray(q_max, dtype=np.float32)
        else:
            sj = self.state[:, ji]
            self.q_min = sj.min(axis=0).astype(np.float32)
            self.q_max = sj.max(axis=0).astype(np.float32)

        # action_abs_max: per-channel percentile of |consecutive action delta| within
        # each episode (okto action_loader.py:181-193). cfg override wins.
        if action_abs_max is not None:
            self.action_abs_max = np.asarray(action_abs_max, dtype=np.float32)
        else:
            actions_j = self.action[:, ji]
            starts = np.concatenate([[0], self.episode_ends[:-1]])
            deltas = []
            for s, e in zip(starts.tolist(), self.episode_ends.tolist()):
                if e - s > 1:
                    deltas.append(actions_j[s + 1 : e] - actions_j[s : e - 1])
            deltas = np.concatenate(deltas, axis=0)  # (N_pairs, J)
            self.action_abs_max = (
                np.percentile(np.abs(deltas), action_percentile, axis=0)
                .clip(min=1e-3)
                .astype(np.float32)
            )


def build_pusht_zarr(cfg) -> PushtZarr:
    """Construct a :class:`PushtZarr` from a dataset cfg.

    Resolves the replay-buffer path as ``cfg.pusht_zarr_root / pusht_cchi_v7_replay.zarr``
    (okto pusht_dataset.py:138), joint indices from ``cfg.qpos_indices`` ([0, 1]),
    and the state bounds from ``cfg.state_q_min/state_q_max`` if present (else zarr).
    The ``action_abs_max`` uses the model's 99.5-percentile default (NOT cfg
    ``action_percentile``, which drives the separate symmetric_percentile action
    normalization step)."""
    zarr_root = getattr(cfg, "pusht_zarr_root", None)
    if zarr_root is None or str(zarr_root) in ("", "???", "."):
        raise NotImplementedError(
            "PushT pos-cmd action requires cfg.pusht_zarr_root (the replay-buffer "
            "directory containing pusht_cchi_v7_replay.zarr); actions are not in the "
            "packed NPZ."
        )
    root = Path(str(zarr_root))
    # Accept either the parent dir or a direct *.zarr path.
    if root.suffix != ".zarr":
        root = root / "pusht_cchi_v7_replay.zarr"
    ji = getattr(cfg, "qpos_indices", None) or [0, 1]
    return PushtZarr(
        root,
        ji,
        q_min=getattr(cfg, "state_q_min", None),
        q_max=getattr(cfg, "state_q_max", None),
    )
