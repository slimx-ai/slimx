from __future__ import annotations

from dataclasses import replace

from .openai_async import OpenAIAsyncProvider


class OAIAsyncProvider(OpenAIAsyncProvider):
    """Async OpenAI-compatible provider for local and self-hosted model servers."""

    name = "oai"
    capabilities = replace(OpenAIAsyncProvider.capabilities, image_out=False)
