"""Model discovery — list the models a provider/server exposes.

`list_models("ollama")` queries Ollama's `/api/tags`; `list_models("oai", ...)`
and `list_models("openai")` query the OpenAI-compatible `/models` endpoint. This
makes a network call (and uses provider credentials where required), so it lives
apart from the import-time-cheap registry helpers.
"""

from __future__ import annotations

from typing import Any, List

from .providers import get_provider


def list_models(provider: str, *, async_mode: bool = False, **kwargs: Any) -> List[str]:
    """Return the model ids/names a provider exposes (best-effort, network call).

    Extra keyword arguments (e.g. ``base_url``, ``api_key``) are forwarded to the
    provider factory; otherwise the usual environment variables are used.
    """
    return list(get_provider(provider, async_mode=async_mode, **kwargs).list_models())
