"""Built-in provider factories.

We keep provider imports inside factories so importing SlimX doesn't pull in
provider code unless you actually select/use a provider.

Factories accept keyword overrides where possible:
- OpenAI: api_key, base_url
- Anthropic: api_key, base_url, version
- Ollama: base_url
- Google: api_key, base_url
- OAI/OpenAI-compatible: api_key, base_url

Async selection is controlled via `async_mode=True`.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

from .base import Provider

ProviderFactory = Callable[..., Provider]

# Each factory only selects the sync/async class and delegates to that class's
# `from_env(**kwargs)`, which is the single source of truth for env var names,
# defaults, and required-key validation. Keyword overrides (api_key, base_url,
# version, …) win over the environment.


def openai_factory(*, async_mode: bool = False, **kwargs: Any) -> Provider:
    if async_mode:
        from .openai_async import OpenAIAsyncProvider as P
    else:
        from .openai import OpenAIProvider as P
    return P.from_env(**kwargs)


def anthropic_factory(*, async_mode: bool = False, **kwargs: Any) -> Provider:
    if async_mode:
        from .anthropic_async import AnthropicAsyncProvider as P
    else:
        from .anthropic import AnthropicProvider as P
    return P.from_env(**kwargs)


def ollama_factory(*, async_mode: bool = False, **kwargs: Any) -> Provider:
    if async_mode:
        from .ollama_async import OllamaAsyncProvider as P
    else:
        from .ollama import OllamaProvider as P
    return P.from_env(**kwargs)


def google_factory(*, async_mode: bool = False, **kwargs: Any) -> Provider:
    if async_mode:
        from .google_async import GoogleAsyncProvider as P
    else:
        from .google import GoogleProvider as P
    return P.from_env(**kwargs)


def oai_factory(*, async_mode: bool = False, **kwargs: Any) -> Provider:
    if async_mode:
        from .oai_async import OAIAsyncProvider as P
    else:
        from .oai import OAIProvider as P
    return P.from_env(**kwargs)


DEFAULT_FACTORIES: Dict[str, ProviderFactory] = {
    "openai": openai_factory,
    "anthropic": anthropic_factory,
    "ollama": ollama_factory,
    "google": google_factory,
    "oai": oai_factory,
}
