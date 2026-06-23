"""vera deploy protocol — thin msgpack/websocket transport (DreamZero-derived).

See docs/SERVER_PROTOCOL_SPEC.md. State lives in the policy; this package is model-agnostic
(serves any BasePolicy, AR or bidirectional).
"""
from .base_policy import BasePolicy
from .server_config import VeraServerConfig, PROTOCOL_VERSION
from .websocket_policy_server import WebsocketPolicyServer
from .websocket_policy_client import WebsocketClientPolicy
from . import _msgpack_numpy as msgpack_numpy

__all__ = [
    "BasePolicy",
    "VeraServerConfig",
    "PROTOCOL_VERSION",
    "WebsocketPolicyServer",
    "WebsocketClientPolicy",
    "msgpack_numpy",
]
