# Changelog

## v0.9.0 (2026-06-04)

### Added — inspectability
- **Inspect mode (dry-run):** `Model.inspect(prompt, stream=False)` /
  `Client.inspect(req)` return an `InspectedRequest` — the exact method, URL, headers
  (secrets redacted), and JSON payload SlimX would send, without making the call.
  `build_request()` is implemented on every provider (sync + async).
- **Trace hooks:** `Client(provider, hooks=...)` and `llm(model, hooks=...)` accept
  `before_call` / `after_call` callbacks for bring-your-own observability. `after_call`
  receives the trace on success or `{ok: False, error}` on failure; a hook that raises is
  swallowed and never breaks the call.
- **Reproducible call records:** `Result.to_record()` builds a `CallRecord`
  (request snapshot, response, usage, trace, SlimX version) with `.save(path)` /
  `CallRecord.load(path)`. The Client now attaches a request snapshot to every `Result`
  as `result.request`.
- New public exports: `InspectedRequest`, `CallRecord`.

## v0.8.0 (2026-06-03)

### Added
- **Parallel (ensemble) execution**: `parallel(models, mode="all" | "race")` fans one
  prompt out to multiple models concurrently and returns an inspectable
  `ParallelResult` (`results`, `errors`, `winner`, `text`, `trace`). `all` returns
  every result; `race` returns the first success and abandons the rest. Failures are
  surfaced in `errors` (never swallowed) and each result keeps its raw provider
  response. Tools/streaming/judge modes are intentionally out of scope for v1.
  Exposed at the top level: `parallel`, `Parallel`, `ParallelResult`, `ParallelItem`.
- `RELEASING.md`: a repeatable, tag-based release checklist.
- Architecture overview (`ARCHITECTURE.md` + `docs/concepts/architecture.md`) with
  mermaid diagrams; mkdocs now renders mermaid via `pymdownx.superfences`.

## v0.7.2 (2026-06-03)

### Fixed
- **Gemini tool calling (3.x):** the Google provider now captures the
  `thoughtSignature` that Gemini attaches to function-call parts and replays it on
  the next turn. Previously the auto tool loop dropped it, so multi-step tool calls
  against `gemini-3.x` failed with `400 ... Function call is missing a
  thought_signature`. Carried generically through a new `ToolCall.extra` field, so
  it round-trips through the Client tool loop and is only ever sent back to the
  originating provider.

## v0.7.1 (2026-06-03)

### Added
- **Provider conformance suite** (`tests/conformance/`): a reusable set of
  contract checks plus a reference `FakeConformantProvider`, run against every
  built-in provider fully offline via `httpx.MockTransport` (sync + async). This
  is the mechanism that guarantees every provider behaves identically; third-party
  plugins can reuse the same checks to claim conformance.
- **Capability introspection**: `slimx.providers.describe_provider(name,
  async_mode=False)` reports a provider's declared capabilities with no API key
  or running server required; `Model.capabilities` / `AsyncModel.capabilities`
  expose the selected provider's `ProviderCapabilities`.
- Provider documentation: a full `concepts/providers.md` (native vs
  OpenAI-compatible), a new `concepts/openai-compatible.md` (vLLM, llama.cpp,
  LM Studio, Ollama `/v1` recipes + troubleshooting), and runnable
  `examples/oai_{vllm,llamacpp,lmstudio,ollama_v1}.py`.

### Changed
- Corrected the provider-capabilities docs to reflect Anthropic tool support and
  its streaming wrapper. Test suite 57 -> 74.

## v0.7.0 (2026-06-03)

### Fixed
- `schema_for()` now resolves string annotations (PEP 563 / `from __future__ import
  annotations`); previously every field on such dataclasses collapsed to `"string"`,
  corrupting `.json(schema=...)` prompts.
- OpenAI/OAI streaming tool calls are reassembled by `index` instead of `id`, so a
  single streamed call is no longer fragmented into multiple broken calls.
- Sync streaming error paths (OpenAI, Google) now read the response body before
  raising, instead of raising `httpx.ResponseNotRead` and masking the real error.
- `retry()` only retries *transient* failures (rate limits, timeouts, transport
  errors); auth/schema/tool errors now fail fast. Sync and async share one policy.
- `Optional` / `X | None` (PEP 604) fields are correctly treated as nullable.
- NDJSON parsing skips malformed lines instead of aborting the whole Ollama stream.

### Added
- Anthropic tool/function calling: tool definitions, `tool_use` parsing, and
  auto-tool-loop message mapping (sync + async). `capabilities.tools` is now `True`.
- `coerce_dataclass()` recurses into nested dataclasses and `List`/`Dict` fields and
  applies best-effort scalar coercion (e.g. `"3"` -> `3`).
- Shared `providers/_openai_shape.py` so the sync and async OpenAI providers can't drift.

### Changed
- Removed the dead `slimx.low.providers` compatibility shim.
- Expanded the test suite (30 -> 57 tests) covering all of the above.

## v0.4.0 (2026-01-03)

- Multi-provider support (OpenAI, Anthropic, Ollama)
- Provider registry + plugin loading via entry points
- Sync + async clients and streaming
- Tool calling + optional auto tool loop (sync + async)
- Structured JSON output parsing (dataclasses)
- MkDocs docs scaffold + GitHub Actions CI/docs/publish

## v0.5.0 (2026-06-03)

- Added built-in Google Gemini provider support.
- Added synchronous and asynchronous Google provider implementations.
- Added Google Gemini streaming support.
- Added Google Gemini structured JSON output mapping.
- Added Google Gemini tool/function-call mapping.
- Added Google provider tests with fake HTTP clients.
- Added Google Gemini quickstart example.
- Updated provider documentation to include Google Gemini.

## v0.5.1 (2026-06-03)

- Fix Ollama provider code–read Ollama error before raising. 


## v0.6.0

- Added `oai:` provider for OpenAI-compatible `/v1/chat/completions` servers.
- Added sync and async OAI provider wrappers.
- Added OAI provider registration through the built-in provider registry.
- Added OAI tests for registration, environment variables, provider kwargs, and Chat Completions payloads.
- Added OAI quickstart example.
- Updated README with native vs OpenAI-compatible provider guidance.