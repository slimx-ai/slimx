"""Providers package.

Provider implementations live under `slimx.providers.*`.

Important: we do *not* import/register built-in providers at import time.
Defaults are registered lazily in `slimx.providers.registry` when you call
`list_providers()` or `get_provider()`.
"""

from .registry import get_provider, list_providers, load_plugins, register

__all__ = ["register", "get_provider", "load_plugins", "list_providers"]
