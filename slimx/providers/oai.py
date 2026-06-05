from __future__ import annotations

from dataclasses import replace

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
