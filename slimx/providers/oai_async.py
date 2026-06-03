from __future__ import annotations

from .openai_async import OpenAIAsyncProvider


class OAIAsyncProvider(OpenAIAsyncProvider):
    """Async OpenAI-compatible provider for local and self-hosted model servers."""

    name = "oai"