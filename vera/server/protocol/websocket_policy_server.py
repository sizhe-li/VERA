"""Thin websocket transport that serves a BasePolicy.

A near-1:1 port of DreamZero's eval_utils/policy_server.py (roboarena transport) — same
metadata-on-connect handshake, same endpoint-multiplexed infer/reset, same string-response
error sentinel — PLUS the gaps that audit flagged: a SIGTERM/SIGINT/atexit flush so killing
mid-episode doesn't lose data. State stays in the policy; this layer is stateless.
See SERVER_PROTOCOL_SPEC.md §2/§4/§8.
"""
from __future__ import annotations

import asyncio
import atexit
import dataclasses
import logging
import signal
import traceback

import websockets.asyncio.server
import websockets.frames

from .base_policy import BasePolicy
from .server_config import VeraServerConfig
from . import _msgpack_numpy as msgpack_numpy

logger = logging.getLogger("vera.server")


class WebsocketPolicyServer:
    def __init__(
        self,
        policy: BasePolicy,
        config: VeraServerConfig,
        host: str = "0.0.0.0",
        port: int = 8000,
        reload_fn=None,
    ) -> None:
        self._policy = policy
        self._config = config
        self._host = host
        self._port = port
        # reload_fn(params: dict, old_policy) -> (new_policy, new_config). When provided, a client
        # can hot-swap the model (embodiment / WAN / IDM ckpt) at runtime without restarting the
        # process. Responsible for freeing the old model's GPU memory before building the new one.
        self._reload_fn = reload_fn
        # Last-applied configure kwargs, accumulated across configure calls. Replayed
        # onto the new policy after a reload so a planner/IDM hot-swap does NOT silently
        # revert operator tuning (guidance, gripper gate, debug-dump re-enable, ...).
        # The legacy DreamZero policy_server kept this; the new transport had dropped it,
        # which is why dumps came up disabled + tuning was lost on every reload (H1).
        self._applied_config: dict = {}
        self._install_flush_handlers()

    # --- lifecycle: flush on kill (DreamZero's missing piece) -----------------
    def _flush(self, *_a):
        flush = getattr(self._policy, "flush", None)
        if callable(flush):
            try:
                flush()
                logger.info("policy.flush() ran on shutdown")
            except Exception:
                logger.exception("policy.flush() failed on shutdown")

    def _install_flush_handlers(self):
        atexit.register(self._flush)
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, lambda *_: (self._flush(), exit(0)))
            except (ValueError, OSError):
                pass  # not main thread / not supported

    # --- serve ----------------------------------------------------------------
    def serve_forever(self) -> None:
        logger.info(
            "vera policy server on %s:%d | %s + %s | H=%d dt=%.4f | host=%s git=%s%s",
            self._host, self._port, self._config.planner_model, self._config.idm_model,
            self._config.action_horizon, self._config.control_dt,
            self._config.hostname, self._config.git_head[:8],
            " (dirty)" if self._config.git_dirty else "",
        )
        asyncio.run(self._run())

    async def _run(self):
        async with websockets.asyncio.server.serve(
            self._handler, self._host, self._port,
            compression=None, max_size=None,
            # long ping so slow WAN forward passes don't trip the keepalive
            ping_interval=60, ping_timeout=600,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket):
        logger.info("connection from %s opened", websocket.remote_address)
        packer = msgpack_numpy.Packer()
        # handshake: declare the config so the client adapts (no hardcoding)
        await websocket.send(packer.pack(dataclasses.asdict(self._config)))
        while True:
            try:
                msg = msgpack_numpy.unpackb(await websocket.recv())
                endpoint = msg.pop("endpoint", "infer")
                if endpoint == "reset":
                    self._policy.reset(msg)
                    await websocket.send("reset successful")
                elif endpoint == "infer":
                    action = self._policy.infer(msg)
                    await websocket.send(packer.pack(action))
                elif endpoint == "observe":
                    # executed-frame report -> immediate retro-render of the just-played
                    # chunk's executed-vs-dream panel (no inference, no episode state change)
                    if hasattr(self._policy, "observe"):
                        await websocket.send(packer.pack(self._policy.observe(msg)))
                    else:
                        await websocket.send("observe not supported by this policy")
                elif endpoint == "configure":
                    # Live runtime tuning (lang/hist guidance, motion_plan_scale,
                    # gripper gate params, debug dumps, ...): forwards kwargs to the
                    # underlying policy's configure_runtime; returns the applied dict.
                    try:
                        inner = getattr(self._policy, "_policy", self._policy)
                        applied = inner.configure_runtime(**msg)
                        # Remember the operator's intent so a later reload can replay it.
                        # configure_runtime mutates cfg field-by-field (not transactional),
                        # but we only reach here on a full success, so every key in msg took.
                        self._applied_config.update(msg)
                        await websocket.send(packer.pack({"applied": applied}))
                    except Exception as e:  # noqa: BLE001
                        await websocket.send(f"configure failed: {e!r}")
                elif endpoint == "reload":
                    # hot-swap the model (embodiment / WAN / IDM ckpt). Requests are serialized on
                    # the connection, so no infer runs concurrently. The reload_fn frees the old
                    # model before building the new one. Re-advertise the new config so the client
                    # adapts (new views / action_dim / provenance).
                    if self._reload_fn is None:
                        await websocket.send("reload not supported by this server")
                    else:
                        logger.info("reload requested: %s", {k: msg[k] for k in sorted(msg)})
                        new_policy, new_config = self._reload_fn(msg, self._policy)
                        self._policy = new_policy
                        self._config = new_config
                        # Replay operator configure state onto the rebuilt policy so the
                        # hot-swap doesn't revert tuning / silently disable dumps (H1).
                        # Key-by-key so a single key the new policy rejects (e.g. gripper
                        # params after a mimicgen->droid embodiment change) doesn't drop
                        # ALL the other (compatible) tuning — configure is not transactional,
                        # so one bad key in a bulk call would abort the rest.
                        if self._applied_config:
                            inner = getattr(new_policy, "_policy", new_policy)
                            replayed, skipped = [], []
                            for k, v in self._applied_config.items():
                                try:
                                    inner.configure_runtime(**{k: v})
                                    replayed.append(k)
                                except Exception as exc:  # noqa: BLE001
                                    skipped.append((k, repr(exc)))
                            logger.info("reload: replayed configure keys %s", replayed)
                            if skipped:
                                logger.warning(
                                    "reload: skipped incompatible configure keys %s",
                                    [k for k, _ in skipped],
                                )
                        logger.info("reload done: %s + %s", new_config.planner_model, new_config.idm_model)
                        await websocket.send(packer.pack(dataclasses.asdict(new_config)))
                else:
                    await websocket.send(f"unknown endpoint: {endpoint!r}")
            except websockets.ConnectionClosed:
                logger.info("connection from %s closed", websocket.remote_address)
                break
            except Exception:
                # string response = error sentinel; client raises on str
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="server error; traceback in previous frame",
                )
                raise
