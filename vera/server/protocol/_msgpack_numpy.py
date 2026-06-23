"""Vendored msgpack codec with numpy support (no openpi_client dependency).

Wire-compatible with DreamZero / openpi_client's ``msgpack_numpy``: ndarrays and numpy
scalars are encoded as tagged maps so dicts-of-arrays pack/unpack directly. BSD-style,
~self-contained. See SERVER_PROTOCOL_SPEC.md §2.
"""
from __future__ import annotations

import msgpack
import numpy as np


def _encode(obj):
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in ("V", "O"):  # void/object arrays are not wire-safe
            raise ValueError(f"cannot serialize ndarray of dtype {obj.dtype!r}")
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": list(obj.shape),
        }
    if isinstance(obj, np.generic):
        return {b"__npscalar__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _decode(obj):
    if b"__ndarray__" in obj:
        return np.frombuffer(obj[b"data"], dtype=np.dtype(obj[b"dtype"])).reshape(
            tuple(obj[b"shape"])
        ).copy()  # copy: frombuffer is read-only over the (transient) bytes
    if b"__npscalar__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


def Packer() -> msgpack.Packer:
    """A reusable packer (matches openpi_client.msgpack_numpy.Packer)."""
    return msgpack.Packer(default=_encode, use_bin_type=True)


def packb(obj) -> bytes:
    return msgpack.packb(obj, default=_encode, use_bin_type=True)


def unpackb(data: bytes):
    return msgpack.unpackb(data, object_hook=_decode, raw=False, strict_map_key=False)
