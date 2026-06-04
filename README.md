# SlimX

[![PyPI](https://img.shields.io/pypi/v/slimx.svg)](https://pypi.org/project/slimx/)
[![Python](https://img.shields.io/pypi/pyversions/slimx.svg)](https://pypi.org/project/slimx/)
[![CI](https://github.com/slimx-ai/slimx/actions/workflows/ci.yml/badge.svg)](https://github.com/slimx-ai/slimx/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**The LLM runtime you can actually read.** A tiny, inspectable, vendor-neutral Python
library for calling LLMs — one API across OpenAI, Anthropic, Gemini, Ollama, and any
OpenAI-compatible server.

```python
from slimx import llm

m = llm("anthropic:claude-haiku-4-5")
print(m("Hello, world").text)
```

Change the provider by changing the string — the rest of your code stays the same.

## Why SlimX

- **One API, every model** — OpenAI, Anthropic, Gemini, Ollama, and OpenAI-compatible
  servers (vLLM, llama.cpp, LM Studio, …). No lock-in.
- **See exactly what's sent** — dry-run the precise request before it leaves, hook every
  call, and save reproducible call records. Glass box, not black box.
- **Tiny & readable** — ~3,000 lines of code, one dependency (`httpx`), fully typed. Read
  the whole thing in an afternoon.
- **Call many models at once** — `parallel(...)` to compare answers, race for the fastest,
  or let a judge model pick the best.
- **Explicit, with batteries** — tools, streaming, structured output with auto-repair, a
  two-layer high/low API, conformance-tested providers, and a `slimx` CLI.

```python
# See what SlimX would send — exact URL, headers (secrets redacted), body — no network call:
print(llm("openai:gpt-4.1-nano").inspect("Hello").pretty())

# Ask several models and let one judge the best answer:
from slimx import parallel
best = parallel(
    ["openai:gpt-4.1-mini", "google:gemini-3.5-flash"],
    mode="judge", judge="anthropic:claude-haiku-4-5",
)
print(best("Explain SlimX in one line.").text)
```

> Going deeper: [`ARCHITECTURE.md`](ARCHITECTURE.md) is a diagram-driven tour of the
> runtime; [`DEVELOPMENT.md`](DEVELOPMENT.md) is the engineering charter and Provider
> Contract.

---

## Install

### For users

Create a new project and install SlimX:

```bash
uv init my-project
cd my-project
uv add slimx
```

Run Python through `uv` so it uses the project virtual environment:

```bash
uv run python
```

Or install with pip:

```bash
pip install slimx
```

### For contributors

```bash
git clone https://github.com/slimx-ai/slimx.git
cd slimx
uv sync --all-extras
uv run pytest -q
```

> `uv sync` reads `pyproject.toml` and `uv.lock` when present.

> `uv.lock` is committed to help contributors reproduce the development environment.

---

## Supported providers

| Provider      |       Prefix | Environment variable                 | Notes                                            |
| ------------- | -----------: | ------------------------------------ | ------------------------------------------------ |
| OpenAI        |    `openai:` | `OPENAI_API_KEY`                     | Default provider when no prefix is given         |
| OpenAI-compatible | `oai:` | OpenAI-compatible `/v1/chat/completions` API | vLLM, LM Studio, llama.cpp server, LocalAI, Ollama `/v1`, internal gateways |
| Google Gemini |    `google:` | `GOOGLE_API_KEY` or `GEMINI_API_KEY` | Supports chat, streaming, JSON output, and tools |
| Anthropic     | `anthropic:` | `ANTHROPIC_API_KEY`                  | Claude Messages API; supports chat, JSON output, and tools |
| Ollama        |    `ollama:` | optional `OLLAMA_BASE_URL`           | Local models through Ollama                      |

---

## Inspect provider capabilities

Check what a provider supports before runtime — no API key or running server required:

```python
from slimx.providers import describe_provider

describe_provider("google")
# {'name': 'google', 'native': True, 'tools': True, 'structured_output': True,
#  'streaming': True, 'async_chat': False, 'async_streaming': False}

from slimx import llm
llm("openai:gpt-4.1-nano").capabilities.tools  # True
```

Every provider is checked against a shared conformance suite (`tests/conformance/`),
so declared capabilities always match real behavior. See
[docs: Provider Capabilities](docs/concepts/provider-capabilities.md) and
[docs: OpenAI-compatible servers](docs/concepts/openai-compatible.md).

---

## Configure providers

### OpenAI

```bash
export OPENAI_API_KEY="..."
# optional:
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

### OpenAI-compatible servers

Use `oai:` for local or self-hosted servers that expose an OpenAI-compatible `/v1/chat/completions` API.

```bash
export SLIMX_OAI_BASE_URL="http://localhost:8000/v1"
export SLIMX_OAI_API_KEY="EMPTY"
```
`SLIMX_OAI_API_KEY` can be a real key for authenticated gateways, or EMPTY for local servers that ignore authentication.

### Google Gemini

```bash
export GOOGLE_API_KEY="..."
# or:
export GEMINI_API_KEY="..."

# optional:
export GOOGLE_BASE_URL="https://generativelanguage.googleapis.com/v1beta"
```

### Anthropic

```bash
export ANTHROPIC_API_KEY="..."
# optional:
export ANTHROPIC_BASE_URL="https://api.anthropic.com"
export ANTHROPIC_VERSION="2023-06-01"
```

### Ollama local models

```bash
export OLLAMA_BASE_URL="http://localhost:11434"
```

For Ollama, make sure the server is running and the model is available:

```bash
ollama serve
```

In another terminal:

```bash
ollama pull llama3.2:3b
ollama list
```

---

## Quickstart

### OpenAI

```python
from slimx import llm

m = llm("openai:gpt-4.1-nano", temperature=0.2)
res = m("Write a haiku about fog and streetlights.")

print(res.text)
```


### OpenAI-compatible local/self-hosted server

```python
from slimx import llm

m = llm(
    "oai:Qwen/Qwen2.5-7B-Instruct",
    provider_kwargs={
        "base_url": "http://localhost:8000/v1",
        "api_key": "EMPTY",
    },
    timeout=120,
)

res = m("Explain why compatibility APIs are useful for local model serving.")

print(res.text)
```

### Google Gemini

```python
from slimx import llm

m = llm("google:gemini-3.5-flash", temperature=0.2)
res = m("Write a haiku about small, inspectable AI software.")

print(res.text)
```

### Ollama local model

```python
from slimx import llm

m = llm("ollama:llama3.2:3b", temperature=0.2, timeout=120)
res = m("Explain why small libraries are easier to inspect.")

print(res.text)
```

---

## Response structure

Calling a SlimX model returns a `Result` object.

```python
from slimx import llm

m = llm("ollama:llama3.2:3b", timeout=120)
res = m("Explain why small libraries are easier to inspect.")

print(res.text)
```

A `Result` contains:

```python
Result(
    text="...",          # Normalized assistant text
    raw={...},           # Raw provider response
    usage=Usage(...),    # Token usage when available
    tool_calls=[],       # Tool/function calls requested by the model
    data=None,           # Parsed structured output, used by .json(...)
    trace={...},         # Runtime metadata: provider, model, latency, retries, tools
)
```

Most applications should use:

```python
print(res.text)
```

Use `res.raw` when you need provider-specific details, and `res.trace` when you want runtime diagnostics such as provider name, model name, elapsed time, retries, and tool-call count.


## Streaming

```python
from slimx import llm

m = llm("google:gemini-3.5-flash", temperature=0.2)

for ev in m.stream("Tell a short story in 5 lines."):
    if ev.type == "text_delta":
        print(ev.text, end="", flush=True)

print()
```

---

## Tools

SlimX tools are provider-neutral. The same `@tool` interface can be used across providers that support tool/function calling.

```python
from slimx import llm, tool


@tool
def add(a: int, b: int) -> int:
    "Add two integers."
    return a + b


m = llm("google:gemini-3.5-flash", tools=[add], tool_runtime="auto")
res = m("What is 12 + 30?")

print(res.text)
```

---

## Parallel execution

Fan one prompt out to several models at once with `parallel(...)`. Use `mode="all"` to
compare every answer, or `mode="race"` for the first successful response.

```python
from slimx import parallel

ensemble = parallel(["google:gemini-3.5-flash", "openai:gpt-4.1-nano"])
res = ensemble("Explain SlimX in one paragraph.")

for item in res.results:
    print(item.model, item.result.text if item.ok else item.error)
```

Failures are surfaced in `res.errors` (never swallowed) and each result keeps its raw
provider response. See [docs: Parallel execution](docs/concepts/parallel.md).

---

## Structured output

SlimX can parse structured JSON output into a dataclass.

```python
from dataclasses import dataclass

from slimx import llm


@dataclass
class City:
    name: str
    country: str


m = llm("google:gemini-3.5-flash")
res = m.json("Paris is in France.", schema=City)

print(res.data)
```

---

## Inspectability

See exactly what SlimX does — dry-run a request, observe calls with hooks, and save
reproducible call records. No hosted platform, no extra dependency.

```python
from slimx import llm, CallRecord

m = llm("openai:gpt-4.1-nano")

# 1) Dry-run: the exact request, secrets redacted, without sending it
print(m.inspect("Hello").pretty())

# 2) Hooks: observe every call (log it, push metrics, anything)
traced = llm("openai:gpt-4.1-nano", hooks={"after_call": print})

# 3) Reproducible records: save the whole call to JSON and reload it
res = m("Capital of France?")
res.to_record().save("run.json")
CallRecord.load("run.json")
```

See [docs: Inspectability](docs/concepts/inspectability.md).

---

## CLI & model discovery

Installing SlimX adds a `slimx` command (no extra dependencies):

```bash
slimx doctor              # which keys/servers are configured and reachable
slimx models ollama       # list models a provider exposes (no guessing model strings)
slimx providers           # registered providers + capabilities
```

`slimx doctor` is the fastest way to answer "why isn't my model working?" — usually a
missing key or wrong base URL. The same discovery is available in code via
`list_models(...)`. See [docs: CLI & discovery](docs/concepts/cli.md).

---

## Low-level API

Use the low-level API when you want explicit control over messages, requests, clients, and providers.

```python
from slimx import Message
from slimx.low import ChatRequest, Client
from slimx.providers import get_provider

provider = get_provider("google")
client = Client(provider, timeout=30, retries=2)

req = ChatRequest(
    model="gemini-3.5-flash",
    messages=[Message.user("Explain provider-neutral LLM clients in one paragraph.")],
    temperature=0.2,
)

res = client.chat(req)

print(res.text)
print(res.trace)
```

---

## Provider plugins

SlimX supports third-party provider plugins through the `slimx.providers` entry point group.

Built-in providers are registered lazily, so importing `slimx` does not load provider modules or require API keys.

---

## Stability

As of **1.0**, SlimX commits to semantic versioning. The public API is stable:

- the top-level surface (`llm`, `allm`, `Model`, `AsyncModel`, `tool`, `Message`,
  `Result`, `StreamEvent`, `ToolCall`, `Usage`, `InspectedRequest`, `CallRecord`,
  `parallel`, `list_models`, `describe_provider`, and `slimx.low`'s `Client` /
  `ChatRequest`),
- the **Provider Contract** that every provider implements (see
  [`DEVELOPMENT.md`](DEVELOPMENT.md)), which is enforced by the conformance suite in
  `tests/conformance/`.

Breaking changes to these will only land in a new major version. The package ships type
information (PEP 561), so type checkers see SlimX's types out of the box.

## Troubleshooting

### `ModuleNotFoundError: No module named 'slimx'`

If you installed with `uv add slimx`, run Python through uv:

```bash
uv run python
```

Or activate the virtual environment first:

```bash
source .venv/bin/activate
python
```

### Ollama model not found

Check which models are installed:

```bash
ollama list
```

Pull a model before using it:

```bash
ollama pull llama3.2:3b
```

Then use the exact model name:

```python
m = llm("ollama:llama3.2:3b", timeout=120)
```

### Ollama server not running

Start Ollama:

```bash
ollama serve
```

Then retry your SlimX script.

---

## Development

Run the full validation suite before opening a pull request or tagging a release:

```bash
uv sync --all-extras
uv run ruff check .
uv run pyright
uv run pytest -q
uv run python -m build
```

---

## Repo automation

This repository includes GitHub Actions for:

* CI (`.github/workflows/ci.yml`)
* Docs deployment to GitHub Pages (`docs.yml`)

See `docs/` for more detailed documentation.
