from __future__ import annotations

import os
from dataclasses import replace

from ..errors import ProviderAuthError
from .openai_async import OpenAIAsyncProvider


class OAIAsyncProvider(OpenAIAsyncProvider):
    """Async OpenAI-compatible provider for local and self-hosted model servers."""

    name = "oai"
    capabilities = replace(
        OpenAIAsyncProvider.capabilities,
        image_out=False,
        image_edit=False,
        hosted_image_tool=False,
        image_partial_streaming=False,
    )

    @classmethod
    def from_env(cls, **overrides):
        """Build from env (`SLIMX_OAI_API_KEY`/`OAI_API_KEY`,
        `SLIMX_OAI_BASE_URL`/`OAI_BASE_URL`); kwargs win."""
        api_key = (
            overrides.get("api_key")
            or os.environ.get("SLIMX_OAI_API_KEY")
            or os.environ.get("OAI_API_KEY")
            or "EMPTY"
        )
        base_url = (
            overrides.get("base_url")
            or os.environ.get("SLIMX_OAI_BASE_URL")
            or os.environ.get("OAI_BASE_URL")
        )
        if not base_url:
            raise ProviderAuthError("SLIMX_OAI_BASE_URL or OAI_BASE_URL is not set")
        return cls(api_key=api_key, base_url=base_url)
