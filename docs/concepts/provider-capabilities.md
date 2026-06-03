# Provider Capabilities

Every provider exposes a small, truthful `ProviderCapabilities` object describing
whether it supports tools, structured output, streaming, async chat, and async
streaming. A declared capability is always backed by real behavior — providers
never claim support they don't have.

```python
ProviderCapabilities(
    tools: bool,
    structured_output: bool,
    streaming: bool,
    async_chat: bool,
    async_streaming: bool,
)
```

## Inspecting capabilities

Check what a provider supports **before** runtime — no API key or running server
required:

```python
from slimx.providers import describe_provider

describe_provider("google")
# {'name': 'google', 'native': True, 'tools': True, 'structured_output': True,
#  'streaming': True, 'async_chat': False, 'async_streaming': False}

describe_provider("openai", async_mode=True)["async_streaming"]  # True
```

From a high-level model:

```python
from slimx import llm

m = llm("google:gemini-3.5-flash")
m.capabilities.tools           # True
m.capabilities.structured_output
```

Use this in diagnostics and production checks to avoid pretending every provider
supports every feature equally — and to fail early with a clear message instead
of a confusing provider error at call time.

## Current support

| Provider | Tools | Structured output | Streaming | Async |
| --- | :---: | :---: | :---: | :---: |
| `openai` / `oai` | ✅ | ✅ | ✅ | ✅ |
| `google` | ✅ | ✅ | ✅ | ✅ |
| `anthropic` | ✅ | — | wrapper¹ | chat only |
| `ollama` | — | — | ✅ | ✅ |

¹ Anthropic chat and tool calling are fully supported. Native token streaming is
not yet implemented; `stream()` is a compatibility wrapper that emits the
complete result as a single `text_delta` followed by `done`, so streaming code
still works against Anthropic without special-casing.

These flags are enforced by the conformance suite (`tests/conformance/`): a
provider is only exercised for a capability it declares, and any provider that
declares a capability must satisfy the full contract for it.
