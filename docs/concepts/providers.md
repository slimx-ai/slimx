# Providers

SlimX talks to models through **providers**. You select one with a prefix on the
model string: `provider:model`. When no prefix is given, `openai` is assumed.

```python
from slimx import llm

llm("openai:gpt-4.1-nano")          # OpenAI
llm("anthropic:claude-3-5-haiku")   # Anthropic
llm("google:gemini-3.5-flash")      # Google Gemini
llm("ollama:llama3.2:3b")           # Ollama (native runtime)
llm("oai:Qwen/Qwen2.5-7B-Instruct") # any OpenAI-compatible server
llm("gpt-4.1-nano")                 # no prefix -> openai
```

## Native vs OpenAI-compatible

This split is central to how SlimX is designed. Two providers can reach the same
model server, and you choose based on *which API surface you want to speak*.

| Prefix | Kind | What it talks to |
| --- | --- | --- |
| `openai:` | Native | Official OpenAI API |
| `google:` | Native | Native Gemini API (`generateContent`) |
| `anthropic:` | Native | Native Anthropic Messages API (`/v1/messages`) |
| `ollama:` | Native | Ollama's native `/api/chat` runtime |
| `oai:` | Compatible | Any server exposing OpenAI's `/v1/chat/completions` |

Use a **native** provider when you want that provider's actual API and its native
behaviors (Gemini's `generationConfig`, Anthropic's content blocks, Ollama's
`options`/`keep_alive`, native usage fields).

Use **`oai:`** when your server speaks the OpenAI Chat Completions shape — vLLM,
llama.cpp server, LM Studio, LocalAI, internal gateways, or Ollama's `/v1`
endpoint. See [OpenAI-compatible servers](openai-compatible.md) for setup
recipes.

> Ollama is reachable two ways: `ollama:` for the native runtime, and `oai:` for
> its OpenAI-compatible `/v1` endpoint. Prefer `ollama:` unless you specifically
> need the OpenAI surface.

SlimX intentionally does **not** ship `vllm:`, `llamacpp:`, or `lmstudio:`
providers — those servers speak `/v1/chat/completions`, so they belong behind
`oai:`.

## Configuration

Every provider reads credentials and base URLs from the environment, and you can
override them per call with `provider_kwargs`.

| Provider | Env vars | Notes |
| --- | --- | --- |
| `openai` | `OPENAI_API_KEY`, `OPENAI_BASE_URL` | Default provider |
| `google` | `GOOGLE_API_KEY` or `GEMINI_API_KEY`, `GOOGLE_BASE_URL` | |
| `anthropic` | `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL`, `ANTHROPIC_VERSION` | |
| `ollama` | `OLLAMA_BASE_URL` (default `http://localhost:11434`) | No API key |
| `oai` | `SLIMX_OAI_BASE_URL` / `OAI_BASE_URL`, `SLIMX_OAI_API_KEY` / `OAI_API_KEY` | Base URL required; key defaults to `EMPTY` |

```python
# Per-call override (takes precedence over the environment):
llm(
    "oai:Qwen/Qwen2.5-7B-Instruct",
    provider_kwargs={"base_url": "http://localhost:8000/v1", "api_key": "EMPTY"},
)
```

## The provider contract

Every provider — built-in or third-party plugin — satisfies the same contract,
so you can swap providers without rewriting your code:

- `chat()` returns a normalized `Result` (`text`, `raw`, `usage`, `tool_calls`).
- `stream()` yields normalized `StreamEvent`s ending in a single `done`.
- Errors map to SlimX error types (`ProviderAuthError`, `ProviderRateLimitError`,
  `ProviderError`).
- Tool calls normalize to `ToolCall`; usage normalizes to `Usage` when reported.
- Async (`achat`/`astream`) mirrors sync where the provider supports it.

This contract is enforced by a shared **conformance suite** (`tests/conformance/`)
that every provider must pass. See [Provider Capabilities](provider-capabilities.md)
to inspect what a given provider supports.

## Third-party providers (plugins)

Providers register lazily, and importing `slimx` never loads provider code or
requires keys. Third parties can add a provider through the `slimx.providers`
entry-point group without modifying core — see [Plugins](plugins.md).
