# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What SlimX is

A tiny, inspectable, vendor-neutral Python library for calling LLMs (OpenAI, Anthropic, Gemini, Ollama, and any OpenAI-compatible server) with one API. ~3,000 lines, **exactly one required runtime dependency: `httpx`** — never add another (optional extras are allowed in `pyproject.toml`).

`DEVELOPMENT.md` is the engineering charter (identity, fit-test, Provider Contract, roadmap) — read Parts I–VI before adding features. `ARCHITECTURE.md` is a diagram-driven tour of the runtime. New features must pass the five-point fit-test in DEVELOPMENT.md Part I; SlimX is explicitly *not* a gateway/proxy, agent framework, RAG framework, or prompt-management platform.

## Commands

Everything runs through `uv`:

```bash
uv sync --all-extras          # dev setup
uv run pytest -q              # all tests
uv run pytest tests/test_google_provider.py            # one file
uv run pytest tests/test_schema.py::test_name          # one test
uv run pytest tests/conformance/                       # provider conformance suite
uv run ruff check .           # lint (line-length 100)
uv run pyright                # type check (basic mode, py310)
uv run python -m build        # package build (CI runs this too)
uv run python -m mkdocs build --strict                 # docs site
```

CI requires ruff + pyright + pytest + build all green. Tests are fully offline (fakes / `httpx.MockTransport`) — no API keys needed.

## Architecture

Thin layers; each layer has one job. New abstractions compose the layer below, never reach sideways or into provider internals:

```
high/api.py        llm()/allm(), Model/AsyncModel — ergonomics/sugar
_parallel.py, cli.py, record.py, discovery.py — cross-cutting features (live HERE, above the Client)
low/client.py      Client — the ONE orchestration point: retries, tool loop, trace
providers/*        adapters only: wire format ⇄ SlimX primitives (no retries, no cross-provider logic)
types.py, messages.py   primitives: Result, Usage, ToolCall, StreamEvent, Message — the stable contract
```

Other key modules: `schema.py` (structured output + `coerce_dataclass`), `tooling.py` (`@tool`), `content.py` (multimodal parts), `errors.py` (exception hierarchy), `providers/registry.py` + `providers/_defaults.py` (lazy registration, env handling).

### Hard rules (from DEVELOPMENT.md Part II)

- **No import-time side effects.** `slimx/__init__.py` is a lazy facade (`__getattr__` + `_LAZY` map). Importing `slimx` must not import providers, touch the network, or need API keys. New public symbols go in the `_LAZY` map and `__all__`, with a `TYPE_CHECKING` import.
- **Sync/async parity, no drift.** Every capability ships sync *and* async. Shared request/response mapping lives in a helper module the async file imports from — see `providers/_openai_shape.py` (used by `openai`/`oai`) and `anthropic.py` (used by `anthropic_async`). Never copy-paste between the sync and async paths.
- **One retry policy** (`utils/retry.py`), used by both sync and async. Retry only `TRANSIENT_ERRORS` (rate limit, timeout, transport); auth/schema/tool errors fail fast.
- **One error hierarchy** (`slimx/errors.py`): `SlimXError` → `ProviderError`/`SchemaError`/`ToolExecutionError`; `ProviderError` → `ProviderAuthError`/`ProviderRateLimitError`/`ProviderTimeoutError`. Don't invent ad-hoc exceptions — retries key off these types.
- **Trace is the diagnostics channel.** The Client attaches a `trace` dict (`provider`, `model`, `elapsed_ms`, `retries`, `tool_steps`, ...). Extend it rather than inventing a parallel mechanism.

### The Provider Contract

The full spec is DEVELOPMENT.md Part III; the conformance suite (`tests/conformance/`) enforces it offline against every built-in provider. Essentials:

- `capabilities: ProviderCapabilities` must be **truthful** — a declared capability (tools, streaming, vision, ...) must be backed by real behavior; an undeclared modality must raise `UnsupportedModalityError`, never silently drop media.
- `chat()` returns a normalized `Result` (`text` never `None`, `raw` always preserved); `stream()` yields `StreamEvent`s ending in exactly one `done`.
- Error mapping: `401/403 → ProviderAuthError`, `429 → ProviderRateLimitError`, other `≥400 → ProviderError`. **Read the response body before raising, including on streamed responses** (`read()`/`aread()` first).
- Streamed tool-call fragments reassemble by provider **index**, not id.
- Constructor takes `api_key`/`base_url` (normalized with `.rstrip("/")`); `from_env()` reads the same env vars as `providers/_defaults.py`; default timeout 30s.

Adding a provider: implement sync + async with a shared helper module, register in `_defaults.py`, add it to `BUILTINS` in `tests/conformance/test_provider_contract.py`, write provider-specific tests with fake HTTP clients (model after `test_google_provider.py`), add an `examples/` script, update the README provider table and CHANGELOG. **Decision rule:** only add a *native* provider when its API offers something the OpenAI-compatible shape can't express — vLLM/llama.cpp/LM Studio belong behind the existing `oai:` prefix, not as new providers.

### Model-string convention

`llm("provider:model")` — prefixes are `openai:` (default when omitted), `anthropic:`, `google:`, `ollama:`, and `oai:` for any OpenAI-compatible `/v1/chat/completions` server.

## Releasing

Version lives in **three places** that must agree: `pyproject.toml`, `slimx/__init__.py` (`__version__`), and the README header. Refresh `uv lock`, ensure ruff/pyright/pytest/build pass, add a dated CHANGELOG entry. See `RELEASING.md`.

## Notes

- `internal_docs/` is untracked planning material — don't reference it from shipped code or docs.
- Schema conventions: resolve annotations with `typing.get_type_hints`; treat `Optional[T]` and `T | None` the same; `coerce_dataclass` is recursive and best-effort (never raise on a slightly-off field).
