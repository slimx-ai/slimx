# SlimX — Development Guide & Roadmap

> Status: living document. Anchored to the shipped **v0.7.0** codebase.
> Purpose: keep every future change consistent with what SlimX *is*, and decide what to build next.

This guide has two jobs:

1. **Ensure consistency** — define the identity, the architecture, the provider contract, and the conventions that every change must respect, so the runtime stays small and trustworthy as it grows.
2. **Guide development** — an opinionated, pruned roadmap (what to add, what to defer, what to refuse) with a concrete milestone sequence.

Read Parts I–VI before writing code. Use Part VII to pick the next branch.

---

## Part I — Product identity (the north star)

**SlimX is a tiny, inspectable, vendor-neutral LLM runtime** for building AI software across native cloud APIs, local runtimes, and OpenAI-compatible servers.

The whole bet is *not* breadth or orchestration. It is that a developer can hold the entire library in their head, see exactly what goes over the wire, and trust that every provider behaves the same way. We win where the big tools are heavy and opaque:

- smaller than LangChain / LlamaIndex,
- more explicit than LiteLLM,
- narrower than Instructor / Pydantic AI,
- friendlier to local/self-hosted models than the official SDKs.

### Who it is for

Engineers who want simple, explicit, multi-provider LLM calls with a clean high-level API (`llm(...)`, `.stream(...)`, `.json(...)`) and an honest low-level API (`Client`, `ChatRequest`, `Message`, provider registry) underneath — without adopting a framework or a hosted platform.

### Non-goals (say no to these)

SlimX is **not**, and will not become:

- a gateway/proxy with virtual keys and an admin UI (that's LiteLLM proxy),
- an agent/orchestration framework (that's LangChain/LangGraph/Pydantic AI),
- a RAG / data-ingestion framework (that's LlamaIndex; integrate, don't absorb),
- a prompt-management platform,
- a heavy-dependency SDK. The only required runtime dependency is `httpx`.

### The feature fit-test

Before adding *anything*, it must pass all five:

1. **Identity** — does it make SlimX more *inspectable, explicit, or vendor-neutral*? (Not just "more featureful.")
2. **Footprint** — can it ship with zero new required dependencies? (Optional extras are allowed.)
3. **Layer** — does it live in the right layer (Part II) and compose the existing primitives instead of bypassing them?
4. **Parity** — can it be delivered with sync + async parity and full tests?
5. **No magic** — does it keep behavior explicit and debuggable? Anything that hides what happened (implicit resolution, silent fallbacks, opaque defaults) fails this test.

If a feature fails the fit-test, it does not go in core. It becomes a doc, a recipe, or a separate companion package (e.g. `slimx-rag`, `slimx-evals`).

---

## Part II — Architecture & layering

SlimX is a stack of thin layers. Each layer has exactly one job. New abstractions are added by **composing the layer below**, never by reaching sideways or down into provider internals.

```
┌─────────────────────────────────────────────────────────────┐
│  high/        Ergonomics: llm(), Model, .json(), .stream()   │  sugar
├─────────────────────────────────────────────────────────────┤
│  parallel/ prompts/ inspect/ cli/   (NEW features live HERE) │  composition
├─────────────────────────────────────────────────────────────┤
│  low/client.py   Runtime: retries, tool loop, trace          │  orchestration
├─────────────────────────────────────────────────────────────┤
│  providers/*     Adapters: wire format  ⇄  SlimX primitives  │  adapters
├─────────────────────────────────────────────────────────────┤
│  types / messages   Result, Usage, ToolCall, StreamEvent     │  primitives
└─────────────────────────────────────────────────────────────┘
```

### Hard rules

1. **Primitives are the contract.** Everything normalizes to `Result`, `Usage`, `ToolCall`, `StreamEvent`, `Message`. These are stable; change them only with a deprecation path. They are the reason a caller can swap providers without rewriting code.
2. **Providers only translate.** A provider's single job is `wire format ⇄ SlimX primitives`. No cross-provider logic, no retries, no orchestration, no business rules in `providers/*`.
3. **The Client is the one orchestration point** for a single request: retries, the tool loop, and trace attachment all live in `low/client.py`. Don't reimplement them elsewhere.
4. **New cross-cutting features live above the Client.** `parallel`, `prompts`, `inspect`, and the CLI get their own modules (`slimx/parallel.py`, etc.) and call `Client`/`Model`. They must **never** be added inside provider files.
5. **No import-time side effects.** Keep the lazy `__getattr__` pattern in `slimx/__init__.py` and `slimx/low/__init__.py`. Importing `slimx` must not import provider modules, touch the network, or require API keys. Providers register lazily via the registry.
6. **Sync/async parity, no drift.** Every capability ships sync *and* async. Shared mapping logic goes in a helper module so the two paths can't diverge — this is exactly the `providers/_openai_shape.py` pattern (used by `openai`/`oai`) and the shared helpers in `anthropic.py` (used by `anthropic_async`). Follow that pattern for every new provider.
7. **One required dependency.** `httpx`, full stop. Anything else is an optional extra in `pyproject.toml`.

---

## Part III — The Provider Contract (canonical)

This is the spec the **conformance suite** (Part VII, milestone 0.7.1) enforces. Every provider — built-in or third-party plugin — must satisfy it. If a provider can't meet a clause, it must declare that truthfully via `capabilities`, not fake it.

A conformant provider:

1. **Identity & capabilities.** Exposes `name: str` and a `capabilities: ProviderCapabilities` that is *truthful*. `ProviderCapabilities` fields: `tools`, `structured_output`, `streaming`, `async_chat`, `async_streaming`, `vision`, `documents`, `audio_in`, `image_out`. A declared capability must be backed by real behavior. (Regression lesson: pre-0.7.0 Anthropic advertised nothing for tools and silently no-op'd them; 0.7.0 made tools real and set `tools=True`. Capability flags and behavior must always agree.)
2. **`chat()` returns a `Result`** with `text: str` (never `None`), `raw` always preserved, `usage` normalized into `Usage` when the provider reports it, and any tool calls as `ToolCall` objects.
3. **`stream()` yields `StreamEvent`s**, emitting `text_delta` events and terminating with **exactly one** `done`. Tool calls are surfaced as `tool_call` events. The stream is the same normalized event taxonomy across all providers.
4. **Errors map to SlimX errors.** `401/403 → ProviderAuthError`, `429 → ProviderRateLimitError`, other `≥400 → ProviderError`; transport/timeout failures surface as `httpx`/`ProviderTimeoutError` (which the Client treats as transient). **Read the response body before raising — including on streamed responses.** (Regression lesson: pre-0.7.0 the sync OpenAI/Google streams raised `httpx.ResponseNotRead` on error because they touched `.text` on an unread stream. Always `read()`/`aread()` the body first.)
5. **Tool calls normalize to `ToolCall`** with `id`, `name`, and arguments available both as a dict (`arguments`) and canonical JSON (`arguments_json`). Streamed tool-call fragments are reassembled by provider **index**, not by `id` (the first delta carries the id, later deltas don't).
6. **Async mirrors sync.** `achat`/`astream` produce the same normalized results as `chat`/`stream`. Shared request/response mapping lives in a module-level helper, not copy-pasted between the two.
7. **Construction is consistent.** Constructor accepts `api_key`/`base_url`/etc.; `base_url` is normalized with `.rstrip("/")`; a `from_env()` classmethod reads the same env vars as the factory in `providers/_defaults.py`. Default request timeout falls back to 30s.
8. **Multimodal is declared, not faked.** A message can carry non-text `parts` (images, documents, audio). A provider that sets `vision`/`documents`/`audio_in`/`image_out` must serialize the corresponding parts into its native wire shape; one that doesn't must raise `UnsupportedModalityError` (a non-transient `ProviderError`) rather than silently dropping the media. Generated images are surfaced on `Result.images`. The conformance suite's `check_modalities` enforces both directions, sync **and** async. (Same lesson as clause 1: capability flags and behavior must agree.)

### Native vs OpenAI-compatible — when to add a provider

This split is a core part of the identity. Keep it explicit:

| Prefix | Meaning |
| --- | --- |
| `openai:` | Official OpenAI API |
| `google:` | Native Gemini API |
| `anthropic:` | Native Anthropic Messages API |
| `ollama:` | Native Ollama `/api/chat` runtime |
| `oai:` | Any OpenAI-compatible `/v1/chat/completions` server (vLLM, llama.cpp, LM Studio, LocalAI, Ollama `/v1`, internal gateways) |

**Decision rule:** only add a *native* provider when the native API offers something the OpenAI-compatible shape genuinely can't express (or when "official" support is the point). Otherwise, route it through `oai:`. Do **not** add `vllm:`, `llamacpp:`, or `lmstudio:` providers — they speak `/v1/chat/completions` and belong behind `oai:`.

---

## Part IV — Conventions

**Errors.** Use the hierarchy in `slimx/errors.py` (`SlimXError` → `ProviderError`/`SchemaError`/`ToolExecutionError`, with `ProviderError` → `ProviderAuthError`/`ProviderRateLimitError`/`ProviderTimeoutError`). Don't invent ad-hoc exceptions. The retry policy keys off these types.

**Retries.** One policy, shared by sync and async (`utils/retry.py`: `retry` / `async_retry`). Retry **only** transient failures (`TRANSIENT_ERRORS`: rate limit, timeout, transport). Auth/schema/tool errors fail fast. Don't add a second retry implementation anywhere.

**Trace.** The Client attaches a `trace` dict: `provider`, `model`, `elapsed_ms`, `retries`, `tool_steps`, `tool_call_count`, `timeout`. New runtime features should extend this dict (and document new keys here) rather than inventing a parallel diagnostics channel. Trace is the seed of the "inspectable" identity.

**Normalization.** `Usage` carries `prompt_tokens`/`completion_tokens`/`total_tokens` plus the `input_tokens`/`output_tokens` aliases — populate what the provider gives, leave the rest `None`. Never raise because a provider omitted usage.

**Schema/structured output.** Resolve annotations with `typing.get_type_hints` (handles `from __future__ import annotations`); treat both `Optional[T]` and `T | None` as nullable; `coerce_dataclass` is recursive and best-effort (coerce when safe, pass through otherwise — never raise on a slightly-off field).

**Sync/async parity.** Shared logic in a `_shape`/helpers module; the async provider imports from the sync module. New providers follow `openai.py`/`_openai_shape.py` and `anthropic.py`/`anthropic_async.py`.

**Lazy imports.** Add new public symbols to the `_LAZY` map and `__all__` in `slimx/__init__.py`, with a `TYPE_CHECKING` import for type checkers. Don't add eager top-level imports of provider/network code.

**Tooling.** `ruff check`, `pyright`, and `pytest` must all be green before commit. Match the formatting of the file you're editing; prefer the formatted style of `google.py`/`ollama.py`/`_openai_shape.py` for new files.

---

## Part V — Definition of Done

### A new provider is done when…

- [ ] Implements the full Provider Contract (Part III), sync **and** async.
- [ ] `capabilities` are truthful and backed by behavior.
- [ ] Shared mapping logic factored into a helper module (no sync/async duplication).
- [ ] Passes the shared conformance suite **and** has provider-specific tests with fake HTTP clients (model `test_google_provider.py`).
- [ ] Error mapping covers 401/403/429/≥400 and reads the body before raising (incl. streamed errors).
- [ ] Registered in `providers/_defaults.py` with a `from_env()` and env-var documentation.
- [ ] `ruff` + `pyright` + `pytest` green; a runnable `examples/` script; README provider table + CHANGELOG updated.

### A new feature is done when…

- [ ] Passes the Part I fit-test; lives in the correct layer (Part II).
- [ ] Zero new required dependencies (optional extra if needed).
- [ ] Sync + async where applicable; full tests including failure paths.
- [ ] Preserves inspectability — exposes `raw`/`trace`/errors, hides nothing.
- [ ] Public symbols added to the lazy `__init__` map; docs page + example; CHANGELOG updated.

### A release is done when…

- [ ] Version bumped in **both** `pyproject.toml` and `slimx/__init__.py` (`__version__`), plus the README header.
- [ ] `uv lock` refreshed; `uv run ruff check . && uv run pyright && uv run pytest -q && uv run python -m build` all pass.
- [ ] CHANGELOG has a dated entry; conformance suite green for all providers.

---

## Part VI — Feature decisions (opinionated)

Verdicts on the proposed features. The reasoning is the point — apply the same fit-test to anything new.

| Feature | Verdict | Why |
| --- | --- | --- |
| **Provider conformance suite** | **Build first** | The single highest-value item. It is the *mechanism* that makes "every provider behaves the same" true, and it's pure identity. |
| **Capability introspection** (`describe_provider`, `Model.capabilities`) | **Build** | Cheap — `ProviderCapabilities` already exists. Lets callers check support before a runtime failure. |
| **Native vs compatible split + docs/examples** | **Build (docs-first)** | Already implemented in code; just needs `docs/providers.md`, examples, and clear "when to use `ollama:` vs `oai:`" guidance. Near-zero risk. |
| **Inspect mode** (dry-run the exact payload) | **Build** | This *is* the brand promise made literal. Cheap to add as a "build request without sending" path. |
| **Trace hooks** (BYO observability) | **Build** | Extends the existing `trace` dict with `before/after_call` hooks. No SaaS lock-in — fits "inspectable, no platform." |
| **Reproducible call record** (`res.to_record()`) | **Build** | Cheap given we already keep `raw` + `usage` + `trace`. Strong fit for research/regulated use. Bundle with inspect mode. |
| **`slimx doctor` + local model discovery** | **Build** | High onboarding leverage (prevents the "wrong Ollama model string" class of confusion); strong local-first signal. Ships as an optional `[cli]` extra. |
| **Parallel / ensemble runtime** | **Build, minimal first** | Genuinely differentiating and on-identity *if* it never hides what happened. Ship `all` + `race` only; defer `judge`/`compare`/`consensus`/`majority`. No tools/streaming in v1. |
| **Structured output validation + repair** | **Build "good-enough"** | Useful, but this is Instructor's home turf. A no-dependency validate-and-reprompt loop is the ceiling — do not grow a Pydantic-style validation engine. |
| **Fallback profiles** | **Fold in, don't add** | A standalone router overlaps `parallel(mode="race")` and edges toward a gateway. Express failover *through* the parallel runtime, not as a separate abstraction. |
| **Prompt packaging / `slimx.yaml` / `@fast` references** | **Defer / shrink hard** | Fails the no-magic test. Implicit `@profile`/`@prompt` resolution hides what model/prompt actually ran — the opposite of "explicit and inspectable," and it drifts SlimX toward a prompt-management platform (a non-goal). If ever built: a tiny explicit `load_prompt(path).render(**vars)` only — **no** `@`-string magic, no model-profile indirection. |
| **Minimal eval runner** (`slimx eval`) | **Separate package** | Valuable but adjacent. Build it as `slimx-evals` on top of the public API; keep it out of core. |
| **Gateway / proxy** | **Refuse** | Direct non-goal. This is LiteLLM's job. |

---

## Part VII — Roadmap

> **Version reconciliation:** the proposal numbered things from `0.6.1`, but the audit + bug-cleanup it slotted there already shipped in **v0.7.0** (schema/streaming/retry fixes, Anthropic tools, recursive coerce, shared OpenAI shape, 30→57 tests). So the roadmap below starts from the *real* current version, 0.7.0.

| Release | Theme | Scope | Why here |
| --- | --- | --- | --- |
| **0.7.0** ✅ | Foundation (shipped) | Bug audit + fixes, Anthropic tools, shared provider helpers, 57 tests | Trustworthy core before new abstractions |
| **0.7.1** | Provider conformance | Shared conformance suite; `describe_provider` / `Model.capabilities`; capability tests | Make "all providers behave the same" *enforced*, not hoped |
| **0.7.2** | Provider clarity | `docs/providers.md`; `examples/oai_{vllm,llamacpp,lmstudio,ollama_v1}.py`; native-vs-compatible + troubleshooting docs | Docs-only, near-zero risk; makes `oai:` obviously useful |
| **0.8.0** | Inspectability | Inspect mode (dry-run payload with redacted headers); trace hooks (`before/after_call`); `res.to_record()` / `record.save()` | The identity-defining cluster; all cheap, all on-brand |
| **0.9.0** | Local-first UX | `slimx doctor` (+ `doctor ollama`/`oai`); `list_models("ollama")` via `/api/tags`, `list_models("oai")` via `/v1/models`; `[cli]` extra | Onboarding + local-first differentiation |
| **0.10.0** | Parallel runtime | `parallel([...], mode="all" \| "race")`, async-internal, errors/trace preserved; **no** tools/streaming/judge | Flagship differentiator, minimal and inspectable |
| **0.11.0** | Reliable structured output | `.json(..., retries=, on_validation_error="repair")`: validate, reprompt, partial-JSON recovery — no new deps | Practical reliability without becoming Instructor |
| **1.0.0** | Stable contract | Freeze the Provider Contract + public API; publish the conformance suite for third-party plugins | Ecosystem readiness |
| *post-1.0* | Companion packages | `parallel` advanced modes (`judge`/`compare`/`majority`); `slimx-evals`; (only if demanded) a *tiny* explicit prompt loader | Keep core small; grow at the edges |

Each milestone is one focused branch (`feature/<name>`), each shippable on its own, each satisfying the Part V Definition of Done.

### Immediate next step

```bash
git checkout main
git pull --ff-only origin main
git checkout -b feature/provider-conformance-tests
```

Then create `tests/conformance/test_provider_contract.py` and the capability helper. Do **not** start inspect mode, the CLI, or parallel execution until conformance is green — every later feature assumes providers already behave identically.

---

## Part VIII — Design notes for the two flagship features

### 0.7.1 — Provider conformance suite

Add `tests/conformance/` with a parametrized contract test plus a `FakeConformantProvider` (no network). Assert, for every provider under test:

```
has name + capabilities (ProviderCapabilities)
chat() -> Result   (text is str, raw preserved, usage is Usage, tool_calls are ToolCall)
stream() -> StreamEvent*  ending in exactly one `done`
declared capabilities match observed behavior
errors map to SlimX error types (401/403, 429, >=400) and read the body first
async (achat/astream) mirrors sync where capabilities allow
```

Capability helper (cheap, ships same release):

```python
from slimx.providers import describe_provider
describe_provider("google")
# {"name":"google","tools":True,"structured_output":True,
#  "streaming":True,"async_chat":False,"async_streaming":False}
```

Built-in providers use fakes (deterministic, offline). The same suite is what a third-party plugin runs to claim conformance — that's the 1.0 ecosystem story.

### 0.10.0 — Parallel runtime (minimal)

One **SlimX call** fans out to multiple models concurrently. Keep it separate from `llm(...)` so single-model usage stays trivial.

```python
from slimx import parallel

m = parallel(["google:gemini-3.5-flash", "openai:gpt-4.1-nano"])
res = m("Explain SlimX in one paragraph.")        # mode="all"
for item in res.results:
    print(item.model, item.result.text if item.result else item.error)
```

Data shapes (preserve everything — never collapse to just `.text`):

```python
@dataclass
class ParallelItem:
    provider: str
    model: str
    result: Result | None = None
    error: str | None = None
    elapsed_ms: int | None = None

@dataclass
class ParallelResult:
    text: str | None                      # set only when a mode yields one answer (race)
    results: list[ParallelItem]
    errors: list[ParallelItem]
    winner: ParallelItem | None           # race/judge
    trace: dict                           # per-model timings, providers, retries
```

**Inspectability rules (non-negotiable):** failures are surfaced in `errors`, never swallowed; `raw` is preserved on every item; `winner`/`trace` make "what actually happened" obvious. **v1 scope:** `all` + `race`, async internally, per-model timeout, no tools, no streaming, no judge. Advanced modes (`judge`, `compare`, `majority`) come post-1.0 and must *always* expose the underlying candidates — a judge mode that hides disagreement is a bug, not a feature.

---

## Appendix — current module map (v0.7.0)

```
slimx/
  __init__.py            lazy public surface (llm, allm, Model, tool, Message, Client, ...)
  types.py               Result, Usage, ToolCall, StreamEvent      (primitives)
  messages.py            Message
  schema.py              schema_for / coerce_dataclass (get_type_hints, recursive, PEP 604)
  tooling.py             @tool, ToolSpec, execute_tool
  errors.py              SlimXError hierarchy
  high/api.py            llm/allm, Model/AsyncModel  (sugar)
  low/
    client.py            Client: retries (shared policy), tool loop, trace   (orchestration)
    types.py             ChatRequest
  providers/
    base.py              Provider ABC + ProviderCapabilities         (contract)
    registry.py          register / get_provider / list_providers / plugins
    _defaults.py         built-in factories + env handling
    _openai_shape.py     shared OpenAI-shape helpers (sync+async, no drift)
    openai.py / openai_async.py / oai.py / oai_async.py
    anthropic.py / anthropic_async.py     (anthropic.py holds shared helpers)
    google.py / google_async.py
    ollama.py / ollama_async.py
    plugins.py           entry-point provider loading
  utils/
    retry.py             retry / async_retry  (transient-only, shared policy)
    sse.py / sse_async.py / ndjson.py        (stream parsing)
```

When this map changes, update it here — it's the fastest orientation for a new contributor and the quickest way to spot a feature that landed in the wrong layer.
