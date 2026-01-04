"""SlimX — a slim, intuitive, lightweight library for calling LLMs.

This top-level module keeps imports *lazy* to:
- speed up imports
- avoid provider bootstrapping side-effects at import time
- prevent circular imports across `high`, `low`, and `providers`

You can still write:

    from slimx import llm, tool, Message

The symbols resolve on first access.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, TYPE_CHECKING

__version__ = "0.4.1"

_LAZY: dict[str, tuple[str, str]] = {
    # High-level
    "llm": ("slimx.high.api", "llm"),
    "allm": ("slimx.high.api", "allm"),
    "Model": ("slimx.high.api", "Model"),
    "AsyncModel": ("slimx.high.api", "AsyncModel"),

    # Tooling
    "tool": ("slimx.tooling", "tool"),
    "ToolSpec": ("slimx.tooling", "ToolSpec"),

    # Messages & core types
    "Message": ("slimx.messages", "Message"),
    "Result": ("slimx.types", "Result"),
    "StreamEvent": ("slimx.types", "StreamEvent"),
    "Usage": ("slimx.types", "Usage"),
    "ToolCall": ("slimx.types", "ToolCall"),

    # Low-level
    "Client": ("slimx.low.client", "Client"),
    "ChatRequest": ("slimx.low.types", "ChatRequest"),

    # Providers
    "get_provider": ("slimx.providers.registry", "get_provider"),
    "list_providers": ("slimx.providers.registry", "list_providers"),
}

__all__ = [
    # High-level
    "llm",
    "allm",
    "Model",
    "AsyncModel",

    # Tooling
    "tool",
    "ToolSpec",

    # Messages & core types
    "Message",
    "Result",
    "StreamEvent",
    "Usage",
    "ToolCall",

    # Low-level
    "Client",
    "ChatRequest",

    # Providers
    "get_provider",
    "list_providers",

    "__version__",
]


if TYPE_CHECKING:
    # These imports are for type checkers only; runtime is lazy.
    from slimx.high.api import AsyncModel, Model, allm, llm
    from slimx.low.client import Client
    from slimx.low.types import ChatRequest
    from slimx.messages import Message
    from slimx.providers.registry import get_provider, list_providers
    from slimx.tooling import ToolSpec, tool
    from slimx.types import Result, StreamEvent, ToolCall, Usage


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        module_name, attr = _LAZY[name]
        mod = import_module(module_name)
        return getattr(mod, attr)
    raise AttributeError(f"module 'slimx' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LAZY.keys()))
