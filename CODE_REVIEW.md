# SlimX (`slimx`) Code Review — audited against v1.3.0

This refreshes the original **v0.6.0** review against the current tree. Every
finding below was re-checked in the 1.3.0 source. Verified empirically in a clean
Linux env: `pytest` → **180 passed**, `ruff check` → **clean**, `pyright` → **0 errors**.

**Bottom line:** all four confirmed bugs (B1–B4) and all four medium logic issues
(L1–L4) are fixed, across the releases between v0.6.0 and 1.3.0. No high- or
medium-severity issue remains open. What's left is minor polish and one
documentation gap.

---

## Status of the original findings

| # | Finding (v0.6.0) | Severity | Status | Where it was fixed |
|---|------------------|----------|--------|--------------------|
| B1 | `schema_for` wrong under `from __future__ import annotations` | High | ✅ Fixed | `schema.py` resolves via `get_type_hints`; `test_schema_future.py` guards it |
| B2 | OpenAI/OAI streaming tool-call accumulation fragmented | High | ✅ Fixed | `providers/_openai_shape.StreamToolAccumulator` keys by `index`, keeps first `id` |
| B3 | Sync stream 4xx → `ResponseNotRead` | Medium | ✅ Fixed | sync `openai`/`google` `read()` the body before raising; codified as Provider Contract clause 4 |
| B4 | `retry()` retries auth/schema/tool errors | Medium | ✅ Fixed | `utils/retry.py` raises when `not _is_transient(e)`; `TRANSIENT_ERRORS` set |
| L1 | `int \| None` (PEP 604) not treated as optional | Medium | ✅ Fixed | `schema._UNION_ORIGINS` includes `types.UnionType` |
| L2 | `coerce_dataclass` shallow / no coercion | Medium | ✅ Fixed | now recursive + scalar coercion, documented as best-effort |
| L3 | ndjson parser unguarded | Low | ✅ Fixed | `utils/ndjson.py` wraps `json.loads` in try/except |
| L4 | Anthropic tools a silent no-op | Medium | ✅ Fixed | real `tools` payload + `tool_use` parse (0.7.0); native streaming (1.1.0) |
| R1 | Dead `slimx/low/providers` module | Low | ✅ Fixed | module removed |
| R2 | Heavy sync/async copy-paste | Low | ✅ Fixed | `providers/_openai_shape.py` is the shared core; anthropic shares helpers too |
| R3 | Async retry duplicated inline | Low | ✅ Fixed | `client.py` uses `async_retry`; one policy |
| R4 | Dead `for…else` branch | Low | ✅ Fixed | retry loop restructured; unreachable tail documented |
| R5 | Unused `from_env` classmethods / duplicated env reading | Low | ✅ Fixed (1.3.0) | `from_env(**overrides)` is the single source; `_defaults` factories delegate; added to google + oai |
| R6 | `inspect._empty`; oai trailing newline; unknown-kwarg warning | Low | ◑ Mostly fixed | `inspect.Parameter.empty` ✅; trailing newlines ✅; unknown-kwarg warning still open |

### Design suggestions (§5)

| # | Suggestion | Status |
|---|------------|--------|
| 5.1 | One OpenAI-shape core | ✅ `_openai_shape.py` |
| 5.2 | One retry policy (sync + async) | ✅ `utils/retry.py` |
| 5.3 | High-level message history (`str \| list[Message]`) | ✅ Fixed (1.3.0) — `Model`/`AsyncModel` `__call__`/`stream`/`json`/`inspect` accept a message list |
| 5.4 | Document per-provider `response_format` semantics | ◑ Ollama native `format:"json"` wired (1.2.0); cross-provider fallback still under-documented |
| 5.5 | Capability-aware high level | ◑ Modalities raise `UnsupportedModalityError`; tools are now real so no longer silently no-op |
| 5.6 | Regression tests for the bugs | ✅ `test_schema_future`, `test_retry`, provider tests |

---

## Still open (all minor)

1. **R6 — unknown `provider_kwargs` are dropped silently.** `get_provider` / `from_env`
   ignore unrecognized keys. Consider warning on unexpected kwargs so a typo'd
   `temperature=` doesn't vanish.
2. **§4 — ruff rule sets.** Config is still just `line-length` / `target-version`.
   Ruff's defaults (E, F) are active, but the suggested `I`, `UP`, `B` rule sets
   aren't enabled. Optional, but cheap signal.
3. **§5.4 — `response_format` documentation.** The per-provider behavior (OpenAI/
   Google honor `json_object`; Anthropic/Ollama lean on the prompt instruction
   `.json()` injects, with Ollama also mapping to native `format:"json"`) is correct
   but not spelled out in the docs.

None of these block a release.

---

## What's new since the v0.6.0 review (not regressions — context)

The library has grown substantially and the architecture has held up:

- **Conformance suite** (`tests/conformance/`) enforces the Provider Contract across
  every built-in provider, sync and async — including the new multimodal clause 8.
- **Multimodal (1.3.0):** image/document/audio **input** across all four providers and
  image-generation **output** (OpenAI Images endpoint; Gemini via `generateContent`),
  with truthful `vision`/`documents`/`audio_in`/`image_out` capabilities, base64 elision
  in `inspect()`/`CallRecord`, and an `UnsupportedModalityError` gate.
- **CI** now runs `ruff`, `pyright`, and `pytest` (the original review's "no type-checker
  in the verified run" gap is closed).
- Production status: `pyproject.toml` is `5 - Production/Stable` with a semver commitment,
  superseding the original review's `3 - Alpha` note.
