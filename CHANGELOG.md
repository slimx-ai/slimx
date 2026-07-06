# Changelog

## v1.6.2 (2026-07-06)

### Changed

- **`parallel()` reuses the model-string parser.** `Parallel` now derives the
  provider prefix through the same `_parse_model` helper as `llm()`, replacing two
  inline copies of the split logic. `ParallelItem.provider` now always matches the
  provider actually resolved for the call (whitespace around the prefix is stripped).

## v1.6.1 (2026-07-04)

### Fixed

- **`parse_json` tolerates markdown-fenced JSON.** Structured output no longer fails
  when a model (notably Anthropic) wraps its JSON reply in markdown code fences.

## v1.6.0 (2026-07-04)

### Added

- **`parallel()` cooperative cancellation.** Pass a `threading.Event` as
  `cancel_event` (to the constructor or per call). Once set, no new model call
  starts — pending items return `cancelled=True` with an explanatory `error` — and
  judge synthesis is skipped. An in-flight provider request is not aborted mid-HTTP.

### Fixed

- **Anthropic: omit sampling parameters for models that reject them**, instead of
  failing the request.

## v1.5.0 (2026-06-22)

### Added — OpenAI Responses image tool, image editing, image stream events

- **OpenAI Responses API hosted image-generation tool.** A text model (e.g.
  `gpt-5.5`) can now generate real image bytes inline. OpenAI-shaped providers route
  to `/responses` automatically whenever a call carries an `image_generation` config
  (`ChatRequest.image_generation` or an `ImageEditRequest`); plain text/function-tool
  chat stays on `/chat/completions` — fully backward compatible. See
  `docs/concepts/image-generation.md`.
  - New `ImageGenerationOptions` (size/quality/output_format/background/
    output_compression/partial_images/action/force) maps onto the tool object;
    `force` sets `tool_choice` so the model must call the tool.
  - `image_generation_call` outputs are decoded exactly once into `GeneratedImage`
    bytes; text + image outputs in one response are both captured.
- **First-class image editing.** `Model.edit_image(image, instruction, …)` /
  `AsyncModel.edit_image` (+ `Provider.edit_image`/`aedit_image`, `ImageEditRequest`,
  `ImageInput`). Sends the source as an `input_image` and forces the image tool;
  operates on the supplied bytes (durable — survives provider state expiry). Supports
  multiple source images and an optional `previous_response_id` optimization.
- **Normalized image stream events.** `StreamEvent` gains `image_started`,
  `image_partial` (transient base64 preview), and `image_completed` (final
  `GeneratedImage`), surfaced from the Responses SSE stream.
- **Expanded `GeneratedImage`.** Adds `width`/`height`, `provider`/`model`,
  `operation`, `provider_response_id`/`provider_call_id`, `revised_prompt`,
  `source_ids`, `output_index`, `metadata`, and `suggested_extension`. MIME is
  sniffed from the bytes; `content.image_dimensions()` reads header dimensions.
- **Capabilities.** `ProviderCapabilities` adds `image_edit`, `hosted_image_tool`,
  `image_partial_streaming`, and an `image_in` alias; `describe_provider()` reports
  them. The `oai` OpenAI-compatible provider does **not** advertise the hosted image
  tool.

### Compatibility

- Existing `/chat/completions` text/streaming/vision and `/images/generations`
  behaviour is unchanged. New public symbols: `ImageGenerationOptions`, `ImageInput`,
  `ImageEditRequest`.

## v1.4.0 (2026-06-18)

### Added — Local hardware awareness & model recommendations

- **`slimx.local` subpackage (opt-in, stdlib-only, no new dependencies).** A small,
  inspectable layer for local-first GPU/CPU awareness, importable independently of
  provider code. Importing `slimx` does not import it, and it adds nothing beyond the
  existing `httpx` dependency (heavy probes shell out lazily).
  - `hardware.detect()` returns a normalized `HardwareProfile` — OS/arch, CPU RAM, GPUs
    (vendor, name, VRAM total/free, driver), Docker-GPU availability, and a recommended
    runtime. Probes `nvidia-smi`, `rocm-smi`, and Apple Silicon; best-effort and never
    raises (a missing tool yields an empty result).
  - `engines/` defines an `InferenceEngine` abstraction with an `OllamaEngine` that
    reports installed/reachable status, lists local models, and exposes per-model runtime
    placement — fully on GPU, split, or CPU-only — from Ollama's `/api/ps` (`size` vs
    `size_vram`). It can also stream `/api/pull` progress.
  - `catalog.py` (bundled `data/local_models.json`) plus `recommend()` bucket local models
    into `recommended` / `possible` / `not_recommended` for a task, scored on **free** VRAM
    (with a rough KV-cache estimate and a CPU-offload fallback), each with a plain-English
    `why`, `estimated_speed`, and `risk`.
- **CLI.** `slimx doctor --hardware [--json]` adds a local CPU/GPU snapshot and recommended
  runtime; new `slimx models recommend [--task T] [--json]` and `slimx models local [--json]`.
  The existing `slimx models <provider>` behaviour is unchanged.

## v1.3.0 (2026-06-05)

### Added — Multimodal (input + image output)

- **Content parts.** Messages can now carry non-text `parts` alongside `content`.
  New `image()`, `document()`, and `audio()` helpers build parts from a path,
  bytes, file-like object, `data:` URI, or `http(s)://` URL (remote URLs are
  passed through, not downloaded, unless `fetch=True`). MIME type is inferred
  from extension or magic bytes. `Message.user(...)` accepts `images=` /
  `documents=` / `audio=` / `parts=`, and `Message` gains `parts`,
  `content_parts()`, and `is_multimodal()`.
- **All four providers serialize natively** (sync + async, shared mapping):
  OpenAI/`oai` (`image_url` / `file` / `input_audio`), Anthropic (`image` /
  `document` blocks, base64 or URL source), Google (`inlineData` / `fileData`),
  Ollama (message-level `images[]` base64 array).
- **Image-generation output.** New `ImageRequest` and `Model.generate_image()` /
  `inspect_image()` (sync + async). OpenAI is wired to the Images endpoint
  (`/images/generations`); Gemini routes through `generateContent`. Results land
  on `Result.images: list[GeneratedImage]`, and Gemini inline image parts are
  parsed there during ordinary calls too. `oai` truthfully declares
  `image_out=False` (compatible servers rarely expose the endpoint).
- **Truthful capabilities.** `ProviderCapabilities` gains `vision`, `documents`,
  `audio_in`, `image_out`. Sending an undeclared modality raises the new
  `UnsupportedModalityError` (a non-transient `ProviderError`) instead of
  silently dropping media.
- **Inspect/record stay readable.** `inspect().pretty()` and `CallRecord` elide
  large base64 blobs to a short placeholder; the request actually sent keeps the
  real bytes.
- **Contract + conformance.** Provider Contract clause 8 (multimodal) added;
  conformance suite's `check_modalities` enforces both directions across every
  built-in provider, sync and async.
- High-level API threads `images=` / `documents=` / `audio=` / `parts=` through
  `__call__`, `stream`, `json`, and `inspect` (and the async mirrors).

Capability matrix: OpenAI — vision, documents, audio_in, image_out; `oai` —
vision, documents, audio_in; Anthropic — vision, documents; Google — vision,
documents, audio_in, image_out; Ollama — vision (vision-capable models).

### Improved (code-review follow-ups)

- **Multi-turn input.** `Model`/`AsyncModel` `__call__`, `stream`, `json`, and
  `inspect` now accept `str | Sequence[Message]`, so conversation history no longer
  requires dropping to the low-level `ChatRequest` API.
- **Single source of truth for env config.** `from_env(**overrides)` now reads each
  provider's env vars (added to Google and `oai`), and the `providers/_defaults.py`
  factories simply delegate to it — removing the duplicated env handling the factories
  previously carried.

Note: as with tools, actual behavior depends on the chosen model being
multimodal (e.g. `gpt-4o`, `gemini-2.5-flash`, `llava`).

## v1.2.0 (2026-06-04)

### Added — Ollama provider
- **Tool calling** (sync + async): tool definitions are sent to `/api/chat`, returned
  `message.tool_calls` are parsed into `ToolCall`s (streamed too), and the auto-tool-loop
  messages are translated to Ollama's shape (`{"function": {...}}` tool calls and
  `{"role": "tool", "tool_name": …}` results). `capabilities.tools` is now `True`.
- **Structured output**: `.json(...)` now maps to Ollama's native `format: "json"`;
  `capabilities.structured_output` is `True`.
- Shared request/response mapping factored into `ollama.py` and reused by the async
  provider (no sync/async drift). Tool-call support also surfaces in streaming.

Note: actual tool/JSON behavior depends on the pulled model (e.g. `llama3.2`, `qwen2.5`).

## v1.1.0 (2026-06-04)

### Added — Anthropic provider
- **Native token streaming** (sync + async): real Server-Sent-Events parsing of the
  Messages API — incremental `text_delta`s and streamed `tool_use` calls (reassembled
  from `input_json_delta` fragments) — replacing the previous single-shot wrapper.
  `capabilities.streaming` / `async_streaming` are now `True`.
- **`extra` passthrough**: Anthropic-specific request fields (`top_p`, `stop_sequences`,
  `tool_choice`, `metadata`, prompt caching, beta fields, …) flow through
  `ChatRequest.extra`.
- **Model discovery**: `AnthropicProvider.list_models()` (via `/v1/models`), so
  `slimx models anthropic` / `slimx doctor` work for Anthropic.
- Request inspection reflects the stream flag for Anthropic.

## v1.0.0 (2026-06-04)

First stable release. The public API and the Provider Contract are now covered by
semantic versioning — breaking changes will only land in a new major version.

### Changed
- Marked **Production/Stable** and committed to semver for the public surface (top-level
  exports + `slimx.low` + the conformance-tested Provider Contract).
- Shipped a PEP 561 `py.typed` marker so downstream type checkers see SlimX's types.

The 0.7–0.11 line built up to this: bug-fixed core, Anthropic tools, provider conformance
suite + capability introspection, the Gemini `thoughtSignature` fix, parallel/ensemble
execution (all/race/compare/judge), full inspectability (inspect mode, trace hooks,
reproducible call records), the `slimx` CLI + model discovery, and structured-output
repair. See the entries below for details.

## v0.11.0 (2026-06-04)

### Added
- **Structured-output repair:** `.json(prompt, schema=..., repair=N)` re-prompts the
  model with the parse/validation error and asks it to fix its output, up to `N` times.
  `repair=0` (default) keeps the existing fail-fast behavior. No new dependency.
- **Parallel `compare` and `judge` modes:**
  - `parallel(models, mode="compare")` returns a readable side-by-side of every answer in
    `text`.
  - `parallel(models, mode="judge", judge="provider:model")` runs the candidates, then a
    judge model picks or synthesizes the best answer (`text`/`winner`). The candidates are
    always preserved in `results` (also `ParallelResult.candidates`).

## v0.10.0 (2026-06-04)

### Added — CLI & model discovery
- **`slimx` command-line tool** (no extra dependencies):
  - `slimx doctor [provider] [--probe]` — report which keys/base URLs are configured,
    probe local servers (ollama, oai) and list their models; `--probe` also checks cloud
    providers. The fast answer to "why isn't my model working?".
  - `slimx models <provider>` — list the models a provider/server exposes.
  - `slimx providers` — registered providers with capabilities.
  - `slimx version`.
- **Model discovery in code:** `list_models(provider, **kwargs)` (top-level), backed by
  `Provider.list_models()` — implemented for `openai`/`oai` (`/models`) and `ollama`
  (`/api/tags`). `describe_provider` is now exported at the top level too.

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