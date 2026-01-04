# SlimX (`slimx`) — v0.4.0

SlimX is a **slim, intuitive, lightweight** Python library for calling LLMs and building LLM systems.

It is intentionally designed around **two clearly separated APIs**:

- **High-level API** (`slimx`) — “1‑minute productivity”: `llm(...)`, `.stream(...)`, `.json(...)`, tools, retries.
- **Low-level API** (`slimx.low`) — “systems builder primitives”: explicit `Client`, `ChatRequest`, `Message`, provider registry, middleware.

SlimX also supports **multiple providers** (OpenAI, Anthropic, Ollama) and **provider plugins** (3rd-party providers without modifying core).

---

## Install (using `uv`)

On Debian/Ubuntu you may hit `externally-managed-environment` (PEP 668) if you try to use system `pip`.
Use **uv**, which manages an isolated environment cleanly.

### Option A — contributors / repo setup (recommended)
```bash
git clone https://github.com/slimx-ai/slimx.git
cd slimx
uv sync --all-extras
```

### Option B — quick test from an extracted zip
```bash
unzip slimx_v0_4.zip
cd slimx_v0_4
uv sync --all-extras
uv run python examples/quickstart_openai.py
```

> `uv sync` reads `pyproject.toml` and (optionally) `uv.lock`.
> If `uv.lock` is present and committed, installs are reproducible.

---

## Configure providers

### OpenAI
```bash
export OPENAI_API_KEY="..."
# optional:
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

### Anthropic
```bash
export ANTHROPIC_API_KEY="..."
# optional:
export ANTHROPIC_BASE_URL="https://api.anthropic.com"
export ANTHROPIC_VERSION="2023-06-01"
```

### Ollama (local)
```bash
export OLLAMA_BASE_URL="http://localhost:11434"
```

---

## Quickstart (high-level)

```python
from slimx import llm
m = llm("openai:gpt-4.1-nano", temperature=0.2)
res = m("Write a haiku about fog and streetlights.")
print(res.text)
```

Streaming:

```python
for ev in m.stream("Tell a short story in 5 lines."):
    if ev.type == "token":
        print(ev.text, end="", flush=True)
print()
```

Tools (auto-loop):

```python
from slimx import llm, tool

@tool
def add(a: int, b: int) -> int:
    "Add two integers."
    return a + b

m = llm("openai:gpt-4.1-nano", tools=[add], tool_runtime="auto")
print(m("What is 12 + 30?").text)
```

Structured output:

```python
from dataclasses import dataclass
from slimx import llm

@dataclass
class City:
    name: str
    country: str

m = llm("openai:gpt-4.1-nano")
res = m.json("Paris is in France.", schema=City)
print(res.data)
```

---

## Repo automation

This bundle includes GitHub Actions:
- CI (`.github/workflows/ci.yml`)
- Docs deploy to GitHub Pages (`docs.yml`)
- PyPI publish on tag (`publish.yml`)

See `README.md` and `docs/` for details.
