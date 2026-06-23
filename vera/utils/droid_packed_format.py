from __future__ import annotations

import io
import json
import os
import tempfile
import zipfile
import zlib
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
from PIL import Image

try:
    import numcodecs
except ImportError:  # pragma: no cover
    numcodecs = None


PACKED_DROID_FORMAT_VERSION = 1
DEFAULT_SHARD_SIZE = 1000
DEFAULT_RGB_CODEC = "jpeg"
DEFAULT_RGB_QUALITY = 90
DEFAULT_FLOW_CODEC = "zstd_npy"
DEFAULT_MOTION_CODEC = "zstd_npz"
DEFAULT_FLOW_SPATIAL_DOWNSAMPLE = 1


@dataclass
class PackedMotionTrackChunkMeta:
    chunk_id: int
    key: str
    start: int
    end: int
    num_frames: int
    codec: str
    tracks_shape: list[int]
    tracks_dtype: str
    visibility_shape: list[int]
    visibility_dtype: str


@dataclass
class PackedEpisodeMetadata:
    format_version: int = PACKED_DROID_FORMAT_VERSION
    episode_id: str = ""
    source_relative_path: str = ""
    num_frames: int = 0
    views: list[str] = field(default_factory=list)
    rgb_entries: dict[str, dict[str, Any]] = field(default_factory=dict)
    flow_entries: dict[str, dict[str, Any]] = field(default_factory=dict)
    motion_tracks: dict[str, dict[str, Any]] = field(default_factory=dict)
    trajectory_entries: dict[str, dict[str, Any]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def shard_dir_name(episode_index: int, shard_size: int = DEFAULT_SHARD_SIZE) -> str:
    shard_idx = max(0, int(episode_index)) // int(shard_size)
    return f"shard_{shard_idx:06d}"


def shard_relative_npz_path(
    *, episode_index: int, episode_id: str, shard_size: int = DEFAULT_SHARD_SIZE
) -> Path:
    return Path(shard_dir_name(episode_index, shard_size=shard_size)) / f"{episode_id}.npz"


def np_uint8_from_bytes(payload: bytes) -> np.ndarray:
    return np.frombuffer(payload, dtype=np.uint8)


def bytes_from_np_uint8(arr: np.ndarray) -> bytes:
    arr = np.asarray(arr, dtype=np.uint8)
    return arr.tobytes()


def encode_image_bytes(
    image: np.ndarray,
    *,
    codec: str = DEFAULT_RGB_CODEC,
    quality: int = DEFAULT_RGB_QUALITY,
) -> bytes:
    image = np.asarray(image)
    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got {image.dtype}")
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"Expected HWC image with 3 or 4 channels, got {image.shape}")

    codec = codec.lower()
    pil_image = Image.fromarray(image[..., :3])
    kwargs: dict[str, Any] = {}
    if codec in {"jpg", "jpeg"}:
        fmt = "JPEG"
        kwargs["quality"] = int(quality)
        kwargs["optimize"] = True
    elif codec == "webp":
        fmt = "WEBP"
        kwargs["quality"] = int(quality)
        kwargs["method"] = 6
    elif codec == "webp_lossless":
        fmt = "WEBP"
        kwargs["lossless"] = True
        kwargs["method"] = 6
    elif codec == "png":
        fmt = "PNG"
        kwargs["optimize"] = True
    else:
        raise ValueError(f"Unsupported image codec: {codec}")

    buf = io.BytesIO()
    pil_image.save(buf, format=fmt, **kwargs)
    return buf.getvalue()


def serialize_array_to_npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.asarray(arr), allow_pickle=False)
    return buf.getvalue()


def _write_named_arrays_to_zip_bytes(named_arrays: dict[str, np.ndarray]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_STORED) as zf:
        for key, value in named_arrays.items():
            with zf.open(f"{key}.npy", mode="w") as fh:
                np.lib.format.write_array(fh, np.asarray(value), allow_pickle=False)
    return buf.getvalue()


def serialize_named_arrays(
    named_arrays: dict[str, np.ndarray],
    *,
    codec: str,
) -> bytes:
    codec = codec.lower()
    payload = _write_named_arrays_to_zip_bytes(named_arrays)
    if codec == "raw_npz":
        return payload
    if codec == "zlib_npz":
        return zlib.compress(payload, level=6)
    if codec == "zstd_npz":
        if numcodecs is None:
            return zlib.compress(payload, level=6)
        return numcodecs.Zstd(level=3).encode(payload)
    raise ValueError(f"Unsupported named-array codec: {codec}")


def deserialize_named_arrays(
    payload: bytes,
    *,
    codec: str,
) -> dict[str, np.ndarray]:
    codec = codec.lower()
    if codec == "raw_npz":
        raw = payload
    elif codec == "zlib_npz":
        raw = zlib.decompress(payload)
    elif codec == "zstd_npz":
        if numcodecs is None:
            raw = zlib.decompress(payload)
        else:
            try:
                raw = numcodecs.Zstd().decode(payload)
            except Exception:
                # Backward-compat: payload may have been zlib-compressed when
                # numcodecs was unavailable at encode time.
                raw = zlib.decompress(payload)
    else:
        raise ValueError(f"Unsupported named-array codec: {codec}")

    named_arrays: dict[str, np.ndarray] = {}
    with zipfile.ZipFile(io.BytesIO(raw), mode="r") as zf:
        for name in zf.namelist():
            key = name[:-4] if name.endswith(".npy") else name
            with zf.open(name, mode="r") as fh:
                named_arrays[key] = np.load(fh, allow_pickle=False)
    return named_arrays


def compress_array_payload(arr: np.ndarray, *, codec: str) -> bytes:
    codec = codec.lower()
    payload = serialize_array_to_npy_bytes(arr)
    if codec == "raw_npy":
        return payload
    if codec == "zlib_npy":
        return zlib.compress(payload, level=6)
    if codec == "zstd_npy":
        if numcodecs is None:
            return zlib.compress(payload, level=6)
        return numcodecs.Zstd(level=3).encode(payload)
    raise ValueError(f"Unsupported array codec: {codec}")


def decompress_array_payload(payload: bytes, *, codec: str) -> np.ndarray:
    codec = codec.lower()
    if codec == "raw_npy":
        raw = payload
    elif codec == "zlib_npy":
        raw = zlib.decompress(payload)
    elif codec == "zstd_npy":
        if numcodecs is None:
            raw = zlib.decompress(payload)
        else:
            try:
                raw = numcodecs.Zstd().decode(payload)
            except Exception:
                # Backward-compat: payload may have been zlib-compressed when
                # numcodecs was unavailable at encode time.
                raw = zlib.decompress(payload)
    else:
        raise ValueError(f"Unsupported array codec: {codec}")
    return np.load(io.BytesIO(raw), allow_pickle=False)


def encode_flow_payload(
    flow: np.ndarray,
    *,
    codec: str,
) -> bytes:
    codec = codec.lower()
    flow = np.asarray(flow)
    if codec in {"raw_npy", "zlib_npy", "zstd_npy"}:
        return compress_array_payload(flow, codec=codec)
    if codec == "qint8_zstd_npz":
        flow_f32 = flow.astype(np.float32, copy=False)
        max_abs = float(np.max(np.abs(flow_f32))) if flow_f32.size else 0.0
        scale = np.float32(max(max_abs / 127.0, 1e-6))
        quantized = np.clip(np.round(flow_f32 / scale), -127, 127).astype(np.int8)
        return serialize_named_arrays(
            {
                "q": quantized,
                "scale": np.asarray(scale, dtype=np.float32),
            },
            codec="zstd_npz",
        )
    raise ValueError(f"Unsupported flow codec: {codec}")


def decode_flow_payload(
    payload: bytes,
    *,
    codec: str,
) -> np.ndarray:
    codec = codec.lower()
    if codec in {"raw_npy", "zlib_npy", "zstd_npy"}:
        return decompress_array_payload(payload, codec=codec)
    if codec == "qint8_zstd_npz":
        named = deserialize_named_arrays(payload, codec="zstd_npz")
        quantized = np.asarray(named["q"], dtype=np.int8)
        scale = float(np.asarray(named["scale"], dtype=np.float32).reshape(()))
        return quantized.astype(np.float32) * scale
    raise ValueError(f"Unsupported flow codec: {codec}")


def rgb_entry_key(frame_idx: int, view: str) -> str:
    return f"frame_{int(frame_idx):06d}_view_{view}"


def flow_entry_key(frame_idx: int, view: str) -> str:
    return f"flow_{int(frame_idx):06d}_view_{view}"


def motion_track_entry_key(chunk_idx: int, view: str) -> str:
    return f"tracks_chunk_{int(chunk_idx):04d}_view_{view}"


def trajectory_entry_key(dataset_key: str) -> str:
    return "traj_" + dataset_key.replace("/", "__")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def metadata_to_uint8_array(metadata: PackedEpisodeMetadata | dict[str, Any]) -> np.ndarray:
    payload = json.dumps(metadata, default=_json_default, sort_keys=True).encode("utf-8")
    return np_uint8_from_bytes(payload)


def write_npz_stored(
    output_path: Path,
    entries: Iterable[tuple[str, np.ndarray]],
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output_path,
        mode="w",
        compression=zipfile.ZIP_STORED,
        allowZip64=True,
    ) as zf:
        for key, arr in entries:
            key = str(key)
            if key.endswith(".npy"):
                raise ValueError(f"Entry key should not include .npy suffix: {key}")
            with zf.open(f"{key}.npy", mode="w") as fh:
                np.lib.format.write_array(fh, np.asarray(arr), allow_pickle=False)


def write_npz_stored_atomic(
    output_path: Path,
    entries: Iterable[tuple[str, np.ndarray]],
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=output_path.stem + ".",
        suffix=".tmp",
        dir=str(output_path.parent),
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        write_npz_stored(tmp_path, entries)
        os.replace(tmp_path, output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def iter_index_manifest_entries(npz_paths: Iterable[Path], root: Path) -> Iterator[str]:
    root = Path(root).resolve()
    for path in npz_paths:
        yield str(Path(path).resolve().relative_to(root))
