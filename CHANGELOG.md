# Changelog

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