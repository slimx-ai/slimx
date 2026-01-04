"""Low-level SlimX API.

This module is intentionally lazy to avoid circular imports.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, TYPE_CHECKING

_LAZY: dict[str, tuple[str, str]] = {
    "Client": ("slimx.low.client", "Client"),
    "ChatRequest": ("slimx.low.types", "ChatRequest"),
}

__all__ = ["Client", "ChatRequest"]


if TYPE_CHECKING:
    from .client import Client
    from .types import ChatRequest


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        module_name, attr = _LAZY[name]
        return getattr(import_module(module_name), attr)
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LAZY.keys()))
