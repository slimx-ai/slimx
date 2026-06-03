from __future__ import annotations

from .openai import OpenAIProvider


class OAIProvider(OpenAIProvider):
    """OpenAI-compatible provider for local and self-hosted model servers.

    This provider speaks the OpenAI Chat Completions API shape but is not
    the official OpenAI provider. Use it for vLLM, LM Studio, llama.cpp
    server, LocalAI, Ollama /v1, and internal OpenAI-compatible gateways.
    """

    name = "oai"