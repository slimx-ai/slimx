"""Built-in provider factories.

We keep provider imports inside factories so importing SlimX doesn't pull in
provider code unless you actually select/use a provider.

Factories accept keyword overrides where possible:
- OpenAI: api_key, base_url
- Anthropic: api_key, base_url, version
- Ollama: base_url

Async selection is controlled via `async_mode=True`.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict

from ..errors import ProviderAuthError
from .base import Provider

ProviderFactory = Callable[..., Provider]


def openai_factory(*, async_mode: bool = False, **kwargs: Any) -> Provider:
    api_key = kwargs.pop("api_key", None) or os.environ.get("OPENAI_API_KEY")
    base_url = kwargs.pop("base_url", None) or os.environ.get(
        "OPENAI_BASE_URL", "https://api.openai.com/v1"
    )

    if async_mode:
        from .openai_async import OpenAIAsyncProvider as P
    else:
        from .openai import OpenAIProvider as P

    if not api_key:
        raise ProviderAuthError("OPENAI_API_KEY is not set")

    return P(api_key=api_key, base_url=base_url)


def anthropic_factory(*, async_mode: bool = False, **kwargs: Any) -> Provider:
    api_key = kwargs.pop("api_key", None) or os.environ.get("ANTHROPIC_API_KEY")
    base_url = kwargs.pop("base_url", None) or os.environ.get(
        "ANTHROPIC_BASE_URL", "https://api.anthropic.com"
    )
    version = kwargs.pop("version", None) or os.environ.get("ANTHROPIC_VERSION", "2023-06-01")

    if async_mode:
        from .anthropic_async import AnthropicAsyncProvider as P
    else:
        from .anthropic import AnthropicProvider as P

    if not api_key:
        raise ProviderAuthError("ANTHROPIC_API_KEY is not set")

    return P(api_key=api_key, base_url=base_url, version=version)


def ollama_factory(*, async_mode: bool = False, **kwargs: Any) -> Provider:
    base_url = kwargs.pop("base_url", None) or os.environ.get(
        "OLLAMA_BASE_URL", "http://localhost:11434"
    )

    if async_mode:
        from .ollama_async import OllamaAsyncProvider as P
    else:
        from .ollama import OllamaProvider as P

    return P(base_url=base_url)


DEFAULT_FACTORIES: Dict[str, ProviderFactory] = {
    "openai": openai_factory,
    "anthropic": anthropic_factory,
    "ollama": ollama_factory,
}
