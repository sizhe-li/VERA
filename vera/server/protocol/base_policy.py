"""Policy interface the transport serves (replaces openpi_client.base_policy.BasePolicy).

A policy owns ALL episode state (frame/latent history, AR KV-cache). The transport never
holds state. ``infer`` returns a dict (the action chunk lives under ``action``); ``reset``
clears the policy + flushes artifacts. See SERVER_PROTOCOL_SPEC.md §0/§6.
"""
from __future__ import annotations

import abc
from typing import Any, Dict


class BasePolicy(abc.ABC):
    @abc.abstractmethod
    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Run one inference. ``obs`` is the wire dict (sans ``endpoint``). Must return a
        dict; the controller reads ``out["action"]`` of shape ``(H, D)``."""

    @abc.abstractmethod
    def reset(self, reset_info: Dict[str, Any]) -> None:
        """Clear ALL episode state (history + caches) and flush any artifacts."""
