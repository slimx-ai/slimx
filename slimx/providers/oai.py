from __future__ import annotations

import os
from dataclasses import replace

from ..errors import ProviderAuthError
from .openai import OpenAIProvider


class OAIProvider(OpenAIProvider):
    """OpenAI-compatible provider for local and self-hosted model servers.

    This provider speaks the OpenAI Chat Completions API shape but is not
    the official OpenAI provider. Use it for vLLM, LM Studio, llama.cpp
    server, LocalAI, Ollama /v1, and internal OpenAI-compatible gateways.
    """

    name = "oai"
    # OpenAI-compatible servers speak Chat Completions but seldom expose the
    # separate `/images/generations` endpoint, so image-out is not promised here.
    capabilities = replace(OpenAIProvider.capabilities, image_out=False)

    @classmethod
    def from_env(cls, **overrides):
        """Build from env (`SLIMX_OAI_API_KEY`/`OAI_API_KEY`,
        `SLIMX_OAI_BASE_URL`/`OAI_BASE_URL`); kwargs win. A `base_url` is required
        since there's no default server. Single source of truth for `oai` config."""
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
