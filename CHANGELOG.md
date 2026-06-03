# Changelog

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

## v0.5.0 (2026-06-03)

- Fix Ollama provider code–read Ollama error before raising. 