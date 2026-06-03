# Provider Capabilities And Fallbacks

Every provider exposes a small capability object describing whether it supports tools, structured output, streaming, async chat, and async streaming.

Use this in demo diagnostics and production checks to avoid pretending every provider supports every feature equally.

OpenAI currently supports tools, structured output, and streaming. Ollama is treated as a strong local streaming fallback. Anthropic support is available for chat, with streaming currently implemented as a compatibility wrapper.
