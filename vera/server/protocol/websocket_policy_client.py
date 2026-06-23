"""Thin websocket client. Port of DreamZero's eval_utils/policy_client.py.

Reads the server's VeraServerConfig on connect (stored as ``server_metadata``), then talks
``infer``/``reset`` over one long-lived connection. A *string* response is the error sentinel
(raise). Long ping interval/timeout because WAN forward passes are slow.
See SERVER_PROTOCOL_SPEC.md §2/§4.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import websockets.sync.client

from .base_policy import BasePolicy
from . import _msgpack_numpy as msgpack_numpy

PING_INTERVAL_SECS = 60
PING_TIMEOUT_SECS = 600


class WebsocketClientPolicy(BasePolicy):
    def __init__(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self._uri = f"ws://{host}:{port}"
        self._packer = msgpack_numpy.Packer()
        self._ws, self._server_metadata = self._connect()

    def get_server_metadata(self) -> Dict[str, Any]:
        return self._server_metadata

    def _connect(self) -> Tuple[Any, Dict[str, Any]]:
        logging.info("connecting to %s ...", self._uri)
        # websockets>=14 sync client does not accept ping_interval/timeout (they belong to
        # the asyncio server, set there). Slow forward passes are kept alive by the server's
        # long ping config + a generous open_timeout here.
        conn = websockets.sync.client.connect(
            self._uri, compression=None, max_size=None, open_timeout=PING_TIMEOUT_SECS,
        )
        return conn, msgpack_numpy.unpackb(conn.recv())

    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        obs = {**obs, "endpoint": "infer"}
        self._ws.send(self._packer.pack(obs))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"inference server error:\n{response}")
        return msgpack_numpy.unpackb(response)

    def reset(self, reset_info: Dict[str, Any] | None = None) -> None:
        msg = {**(reset_info or {}), "endpoint": "reset"}
        self._ws.send(self._packer.pack(msg))
        resp = self._ws.recv()
        if isinstance(resp, str) and resp != "reset successful":
            raise RuntimeError(f"reset server error:\n{resp}")

    def observe(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Post-chunk visual update: send the current rolling context so the server retro-renders
        the just-executed chunk's viewer panel immediately (instead of at the next infer)."""
        msg = {**obs, "endpoint": "observe"}
        self._ws.send(self._packer.pack(msg))
        resp = self._ws.recv()
        if isinstance(resp, str):
            raise RuntimeError(f"observe server error:\n{resp}")
        return msgpack_numpy.unpackb(resp)

    def configure(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Live-tune server runtime knobs (motion_plan_scale, sample_steps, lang/hist_guidance...)
        WITHOUT a model rebuild. Returns the server's {"applied": {...}} echo of effective values."""
        msg = {**params, "endpoint": "configure"}
        self._ws.send(self._packer.pack(msg))
        resp = self._ws.recv()
        if isinstance(resp, str):
            raise RuntimeError(f"configure server error:\n{resp}")
        return msgpack_numpy.unpackb(resp)

    def reload(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Hot-swap the server's model (embodiment / WAN / IDM ckpt) at runtime. ``params`` may set
        ``embodiment``, ``algo_config`` (WAN), ``dynamics_run_id`` (IDM), ``sample_steps``,
        ``action_horizon``. Returns + caches the server's NEW config (handshake metadata)."""
        msg = {**params, "endpoint": "reload"}
        self._ws.send(self._packer.pack(msg))
        resp = self._ws.recv()
        if isinstance(resp, str):
            raise RuntimeError(f"reload server error:\n{resp}")
        self._server_metadata = msgpack_numpy.unpackb(resp)
        return self._server_metadata

    def close(self) -> None:
        try:
            self._ws.close()
        except Exception:
            pass
