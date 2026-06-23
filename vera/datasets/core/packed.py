"""Packed-NPZ codec (one packed episode read by the unified loader).

okto packs each episode into a single ``.npz`` with three families of entries:

  * ``traj_<key>``                     — low-dim trajectory arrays (e.g.
    ``traj_low_dim__robot0_eef_pos``), float32, shape ``(T, D)``.
  * ``frame_{i:06d}_view_{view}``      — per-frame JPEG bytes (uint8) per view.
  * ``flow_{i:06d}_view_{view}``       — per-frame optical-flow payload per view,
    ``qint8_zstd_npz`` quantized.
  * ``__packed_episode_metadata__``    — JSON blob (uint8 bytes) with
    ``num_frames``, ``views``, ``rgb_entries``, ``flow_entries``,
    ``trajectory_entries``, ``episode_id``, ``source_relative_path``.

This module ports the decode bodies from okto so the unified core reads the EXACT
same bytes/values the okto loaders produced:

  * trajectory load  <- okto ``action_loader._load_packed_dataset`` / ``_packed_entry_key``
  * JPEG RGB decode  <- okto ``loaders/rgb_loader.load_rgb_frames_from_packed_npz``
  * qint8 flow decode<- okto ``loaders/optical_flow_loader._load_packed_optical_flow_impl``
                        (+ ``okto.utils.droid_packed_format.decode_flow_payload``)

It is GENERAL over entry keys / views / dataset — the mimicgen specifics (2 views,
action dim, normalization) come from config + per-episode packed metadata, not here.
"""

from __future__ import annotations

import io
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# vera ships the flow codec (qint8_zstd_npz) under utils.droid_packed_format.
from vera.utils.droid_packed_format import decode_flow_payload

PACKED_METADATA_KEY = "__packed_episode_metadata__"


def packed_entry_key(key: str) -> str:
    """Canonical packed-NPZ entry key (matches okto ``_packed_entry_key``)."""
    return "traj_" + key.replace("/", "__")


def rgb_entry_key(frame_index: int, view: str) -> str:
    """Matches okto ``rgb_loader._packed_rgb_key``."""
    return f"frame_{int(frame_index):06d}_view_{view}"


def flow_entry_key(frame_index: int, view: str) -> str:
    """Matches okto ``optical_flow_loader._packed_flow_key``."""
    return f"flow_{int(frame_index):06d}_view_{view}"


# --------------------------------------------------------------------------
# NPZ open cache (process-local). okto used ``open_npz_cached``; the unified
# core keeps a tiny LRU here so repeated reads of the same episode (same worker)
# don't re-open the zip. Disabled (maxsize bypass) is not needed — npz handles
# are lazy, so the cache holds the ``NpzFile`` object, not the decoded arrays.
# --------------------------------------------------------------------------
@lru_cache(maxsize=256)
def _open_npz_cached(npz_path: str):
    # allow_pickle=False: packed entries are plain arrays / uint8 byte blobs.
    return np.load(npz_path, allow_pickle=False)


def open_packed_npz(npz_path: str | Path):
    """Open a packed NPZ (cached). Returns a lazy ``numpy.lib.npyio.NpzFile``."""
    return _open_npz_cached(str(npz_path))


def load_packed_metadata(npz_path: str | Path) -> Dict[str, Any]:
    """Decode the per-episode JSON metadata embedded in the packed NPZ.

    Ported from okto ``metadata_builder._load_packed_episode_metadata``.
    """
    npz = open_packed_npz(npz_path)
    if PACKED_METADATA_KEY not in npz:
        raise KeyError(f"Missing {PACKED_METADATA_KEY} in packed episode {npz_path}")
    payload = np.asarray(npz[PACKED_METADATA_KEY], dtype=np.uint8).tobytes()
    return json.loads(payload.decode("utf-8"))


def load_packed_array(
    npz_path: str | Path,
    key: str,
    row_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Load a trajectory array from a packed NPZ.

    Ported from okto ``action_loader._load_packed_dataset``: resolve the
    ``traj_<key>`` entry and (optionally) slice rows, preferring a contiguous
    ``[start:stop]`` slice when the requested indices are consecutive.

    ``key`` is the logical trajectory key (e.g. ``low_dim/robot0_eef_pos``); the
    ``traj_``/``__`` mangling is applied internally so callers pass the same key
    okto used.
    """
    npz = open_packed_npz(npz_path)
    entry_key = packed_entry_key(key)
    if entry_key not in npz:
        raise KeyError(f"Missing packed entry '{entry_key}' in {npz_path}")
    values = np.asarray(npz[entry_key])
    if row_indices is None:
        return values
    row_indices = np.asarray(row_indices, dtype=np.int64).reshape(-1)
    if row_indices.size == 0:
        return np.empty((0, *values.shape[1:]), dtype=values.dtype)
    contiguous = bool(np.all(row_indices[1:] == row_indices[:-1] + 1))
    if contiguous:
        start = int(row_indices[0])
        stop = int(row_indices[-1]) + 1
        return values[start:stop]
    return values[row_indices]


def decode_packed_rgb_frame(npz, frame_index: int, view: str) -> np.ndarray:
    """Decode one JPEG-packed RGB frame -> ``(H, W, 3)`` uint8.

    Ported from okto ``rgb_loader.load_rgb_frames_from_packed_npz`` (the inner
    PIL decode of the per-frame JPEG byte blob).
    """
    from PIL import Image  # local import: only the packed path needs PIL

    key = rgb_entry_key(frame_index, view)
    payload = np.asarray(npz[key], dtype=np.uint8).tobytes()
    with Image.open(io.BytesIO(payload)) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8).copy()


def decode_packed_flow_frame(npz, frame_index: int, view: str, codec: str) -> np.ndarray:
    """Decode one packed optical-flow frame -> ``(2, H, W)`` or ``(H, W, 2)`` float32.

    Ported from okto ``optical_flow_loader._load_packed_optical_flow_impl`` (the
    per-frame ``decode_flow_payload`` call). The channel-order fix to ``(2,H,W)``
    is applied by the caller (``view_loader``), mirroring okto.
    """
    key = flow_entry_key(frame_index, view)
    payload = np.asarray(npz[key], dtype=np.uint8).tobytes()
    return decode_flow_payload(payload, codec=codec)
