# SlimX Multimodal Execution Plan

> Status: proposal. Scope: full multimodal **in and out** (image / document / audio
> input + image-generation output) across all four built-in providers
> (`openai`, `anthropic`, `google`, `ollama`) and, by inheritance, `oai`.
>
> This plan follows the rules in [`DEVELOPMENT.md`](DEVELOPMENT.md): primitives are
> the contract, every capability ships sync **and** async with shared mapping logic,
> capabilities must be truthful, and `ruff` / `pyright` / `pytest` must be green before
> commit. It is staged into semver-aligned milestones so no single change is a big bang.

---

## Part I — The core problem

Today a message body is a single string. In [`slimx/messages.py`](slimx/messages.py):

```python
@dataclass(frozen=True)
class Message:
    role: str
    content: str          # <- text only
    ...
```

Every constructor (`Message.system/user/assistant/tool`) takes `content: str`, and
`Message.to_dict()` emits `{"role", "content": <str>}`. The low-level
[`ChatRequest.to_dict()`](slimx/low/types.py) serializes each message with `m.to_dict()`,
and the high-level [`Model.__call__`](slimx/high/api.py) only accepts `prompt: str`. There
is no content-parts structure, no image/audio/document handling, and no place on `Result`
for generated media.

So multimodal touches five layers, and the plan is organized around them:

1. **Primitive** — a content-parts model on `Message` (input) and media on `Result` (output).
2. **Helpers / public surface** — `image()`, `document()`, `audio()` constructors and an
   ergonomic way to attach them in both the high and low APIs.
3. **Provider serialization** — each provider's message converter learns to emit parts in
   its native shape; this is the bulk of the work and the part most exposed to drift.
4. **Capabilities & contract** — extend `ProviderCapabilities`, declare truthfully, gate
   unsupported modalities with a fail-fast error, and extend the conformance suite.
5. **Inspect / record hygiene** — base64 blobs must not destroy the "glass box" dry-run and
   reproducible-record values.

---

## Part II — Design decisions (resolve before coding)

These are the choices that shape every milestone. Recommended answer in **bold**.

1. **How do parts coexist with `content: str`?**
   `Message` is a frozen dataclass and `content` is read as a string in many places
   (`anthropic._messages_to_anthropic`, `google._contents_from_messages`,
   `ollama` converter, structured-output repair, etc.). Replacing `content`'s type is a
   breaking change to a contract primitive.
   **Recommendation: keep `content: str` and add a new optional field
   `parts: tuple[Part, ...] = ()`.** `content` continues to hold the text; `parts` holds
   non-text media (plus, optionally, explicit text parts). Add a method
   `content_parts() -> list[Part]` that returns the normalized list (synthesizing a single
   `TextPart(content)` when `parts` is empty). Existing readers of `.content` keep working
   unchanged; new code reads `content_parts()`. This honors "change primitives only with a
   deprecation path."

2. **Do we fetch remote image/audio URLs ourselves?**
   OpenAI, Anthropic, and Gemini accept remote URLs natively; Ollama's `/api/chat` accepts
   **only** base64 bytes in an `images: [...]` array. **Recommendation: pass URLs through
   untouched where the provider supports them; for providers that require bytes (Ollama),
   either lazily fetch+encode the URL or declare URL-sourced media unsupported and raise.**
   Make the fetch opt-in (a flag on the helper) so SlimX never makes a surprise network
   call during request building — this preserves the "see exactly what's sent" guarantee.

3. **Provider-level vs model-level capability.** Vision/audio support depends on the
   *model*, not just the provider (`gpt-4o` vs an old text model; `llava` vs `llama3.2`).
   `ProviderCapabilities` is provider-level. **Recommendation: declare provider-level caps
   for "this provider's API can carry this modality," gate on that, and let genuine
   model-mismatch errors flow through as the provider's `ProviderError`.** Document this
   caveat explicitly (same spirit as the existing Ollama "depends on the pulled model" note
   in the CHANGELOG).

4. **One unsupported-modality error type.** Add `UnsupportedModalityError(ProviderError)`
   so it fails fast (the retry policy in `utils/retry.py` retries only transient errors;
   `ProviderError` is not transient, which is what we want).

5. **Image-gen output lives on `Result`, not in the message loop.** Output images are a
   response concern. **Recommendation: add `Result.images: list[GeneratedImage]`** and parse
   provider responses into it, rather than overloading the tool/text path.

---

## Part III — The content-parts primitive

New module `slimx/content.py` (keeps `messages.py` thin and avoids import cycles):

```python
@dataclass(frozen=True)
class TextPart:
    text: str

@dataclass(frozen=True)
class ImagePart:
    mime_type: str                 # "image/png", "image/jpeg", ...
    data: Optional[bytes] = None   # inline bytes (base64-encoded at serialize time)
    url: Optional[str] = None      # remote URL (provider-native passthrough)
    detail: Optional[str] = None   # OpenAI "low|high|auto" hint, ignored elsewhere

@dataclass(frozen=True)
class DocumentPart:                # milestone 1.4
    mime_type: str                 # "application/pdf", ...
    data: Optional[bytes] = None
    url: Optional[str] = None
    filename: Optional[str] = None

@dataclass(frozen=True)
class AudioPart:                   # milestone 1.5
    mime_type: str                 # "audio/wav", "audio/mp3", ...
    data: Optional[bytes] = None
    url: Optional[str] = None

Part = Union[TextPart, ImagePart, DocumentPart, AudioPart]
```

Source-loading helpers in the same module (lazy, no network unless asked):

```python
def image(src, *, mime_type=None, detail=None, fetch=False) -> ImagePart: ...
def document(src, *, mime_type=None, filename=None) -> DocumentPart: ...
def audio(src, *, mime_type=None) -> AudioPart: ...
```

`src` accepts a filesystem path (`str`/`os.PathLike`), raw `bytes`, a file-like object, or
an `http(s)://`/`data:` URL. MIME type is inferred via `mimetypes` + magic-byte sniffing
when not given. These return frozen part objects; no base64 encoding happens until
serialization, so large blobs aren't duplicated in memory early.

### `Message` changes (`slimx/messages.py`)

- Add field `parts: tuple[Part, ...] = ()`.
- Extend `Message.user(content="", *, images=None, parts=None, ...)` to accept media; keep
  the positional-string form 100% back-compatible.
- Add `content_parts() -> list[Part]`: returns `list(parts)` if non-empty, else
  `[TextPart(content)]` when `content` is truthy, else `[]`.
- `to_dict()` gains a default multimodal serialization (OpenAI Chat shape: `content` becomes
  a list of typed parts when `parts` is present) so any `oai:`-style consumer works without
  a custom converter. Providers with native shapes override via their own converters.

### `Result` changes (`slimx/types.py`) — milestone 1.6

```python
@dataclass(frozen=True)
class GeneratedImage:
    mime_type: str
    data: Optional[bytes] = None
    url: Optional[str] = None

@dataclass
class Result:
    ...
    images: list[GeneratedImage] = field(default_factory=list)
```

`text` stays `str` (never `None`) per contract clause 2; image-only responses simply carry
`text=""` and a populated `images`.

### Public surface (`slimx/__init__.py`)

Add to the `_LAZY` map and `__all__`, with `TYPE_CHECKING` imports (per the lazy-import
convention): `image`, `document`, `audio`, `TextPart`, `ImagePart`, `DocumentPart`,
`AudioPart`, `GeneratedImage`. Target usage:

```python
from slimx import llm, image

m = llm("openai:gpt-4o")
print(m("What's in this picture?", images=[image("diagram.png")]).text)
```

---

## Part IV — Per-provider serialization

This is the core of the contract work. Each provider has a message-converter function that
must learn to emit parts; the **shared mapping must stay in the sync module and be imported
by the async one** (no drift — DEVELOPMENT.md Part III clause 6).

| Provider | Converter (today) | Image input shape | Doc (PDF) | Audio in | Image out |
| --- | --- | --- | --- | --- | --- |
| `openai` / `oai` | `_openai_shape.build_payload` → `Message.to_dict()` | `content: [{"type":"text"}, {"type":"image_url","image_url":{"url": dataURI\|httpURL,"detail"}}]` | `{"type":"file","file":{"file_data": base64}}` | `{"type":"input_audio","input_audio":{"data","format"}}` | Images API / `gpt-image-1` |
| `anthropic` | `_messages_to_anthropic` | block `{"type":"image","source":{"type":"base64"\|"url","media_type","data"\|"url"}}` | `{"type":"document","source":{...}}` | not native → cap `False` | not native → cap `False` |
| `google` | `_contents_from_messages` | part `{"inlineData":{"mimeType","data"}}` or `{"fileData":{"fileUri","mimeType"}}` | inline PDF supported | inline audio supported | image parts in `generateContent` response |
| `ollama` | converter in `ollama.py` | message-level `images: [base64,...]` (no `data:` prefix, no MIME) | not native → cap `False` | not native → cap `False` | not native → cap `False` |

Implementation notes per provider:

- **OpenAI / `oai`.** Because everything flows through `Message.to_dict()` →
  `_openai_shape.build_payload`, most of the work is making `to_dict()` emit the list-content
  shape when `parts` exist. Verify `oai:` targets (vLLM, LM Studio) tolerate the multimodal
  content array; gate behind capability and document model dependence.
- **Anthropic.** Extend `_messages_to_anthropic` so a user turn with parts becomes a list of
  content blocks (text + image/document). Anthropic supports both `base64` and `url` image
  sources; map `ImagePart.data` → base64 source, `ImagePart.url` → url source. Keep the
  existing tool-loop block handling intact.
- **Google.** Extend `_contents_from_messages` to append `inlineData`/`fileData` parts beside
  the existing `{"text": ...}` part on `user` turns. Gemini is the most uniformly multimodal,
  so it's the reference implementation for documents and audio.
- **Ollama.** Different shape: images ride on the message as a sibling `images` array of bare
  base64 strings, not inside `content`. URL-sourced images must be fetched+encoded or rejected
  (decision II.2). Behavior depends on a vision model (`llava`, `llama3.2-vision`).

Each provider's `build_request` (dry-run) automatically reflects the new shapes because it
already calls the same payload builder — but see Part VI on eliding base64.

---

## Part V — Capabilities & the Provider Contract

Extend `ProviderCapabilities` in [`slimx/providers/base.py`](slimx/providers/base.py):

```python
@dataclass(frozen=True)
class ProviderCapabilities:
    tools: bool = False
    structured_output: bool = False
    streaming: bool = False
    async_chat: bool = False
    async_streaming: bool = False
    # new:
    vision: bool = False          # image input
    documents: bool = False       # PDF/file input
    audio_in: bool = False        # audio input
    image_out: bool = False       # image-generation output
```

Truthful declarations (initial target):

| Provider | vision | documents | audio_in | image_out |
| --- | --- | --- | --- | --- |
| `openai` / `oai` | ✅ | ✅ | ✅ | ✅ (Images API / `gpt-image-1`) |
| `anthropic` | ✅ | ✅ | ❌ | ❌ |
| `google` | ✅ | ✅ | ✅ | ✅ (image models) |
| `ollama` | ✅ (vision models) | ❌ | ❌ | ❌ |

Contract updates:

- Add a **clause 8 to the Provider Contract** (DEVELOPMENT.md Part III): "Multimodal is
  declared, not faked. A provider that sets `vision`/`documents`/`audio_in`/`image_out`
  accepts the corresponding parts and emits provider-correct payloads; one that doesn't
  raises `UnsupportedModalityError` rather than silently dropping the media." This mirrors the
  existing clause-1 lesson (Anthropic once advertised tools and no-op'd them).
- Update the `ProviderCapabilities` field list in clause 1 to include the new flags.
- Add `UnsupportedModalityError(ProviderError)` to `slimx/errors.py` and document it in the
  Part IV "Errors" convention.

Gating happens in each provider's converter (or a shared pre-check in the `Client`): if a
message carries an `ImagePart` and `capabilities.vision` is `False`, raise
`UnsupportedModalityError` before building the request.

---

## Part VI — Inspect & record hygiene (don't break the glass box)

Base64 image/audio payloads are large; dumping them verbatim destroys two headline features:
`inspect().pretty()` (dry-run readability) and `CallRecord` (reproducible, diffable records).

- Add a shared `_elide_media(payload)` helper. In
  [`InspectedRequest.pretty()`](slimx/types.py) and in
  [`CallRecord.to_dict()`](slimx/record.py), replace any base64 media value with a stable
  placeholder such as `"<base64 image/png, 48,213 bytes, sha256:ab12…>"`.
- Keep the real bytes in the actual outgoing request; elide **only** in the human-facing /
  serialized views. The request SlimX sends is still exact.
- Record round-tripping: store media as `{mime, size, sha256}` (or an external file ref), not
  inline base64, so records stay small and meaningful.

---

## Part VII — High & low API ergonomics

- **High API** ([`slimx/high/api.py`](slimx/high/api.py)): add `images=`/`parts=` keyword args
  to `Model.__call__`, `stream`, `json`, and `inspect`, plus the `AsyncModel` mirrors. Build
  the user `Message` with parts instead of `Message.user(prompt)`. Keep the string-only path
  unchanged.
- Allow `__call__` to also accept a prebuilt `Message` or list of messages for full control
  (power users assembling mixed turns).
- **Low API**: no signature change needed — callers already build `Message`/`ChatRequest`
  directly; they just gain `parts`.
- **`parallel()`** ([`slimx/_parallel.py`](slimx/_parallel.py)): confirm prompts carrying
  images fan out correctly; add a multimodal case to its tests.
- **CLI** ([`slimx/cli.py`](slimx/cli.py)): add `--image PATH` (repeatable) to the chat path;
  surface the new capability flags in the `providers` / `doctor` output (both already render
  `describe_provider`'s `ProviderCapabilities`).

---

## Part VIII — Testing strategy

Follow the existing layout (`tests/test_<provider>_provider.py`, `tests/conformance/`). All
provider-shape assertions go through `build_request`/`inspect` so they need **no network**.

- **Unit (`tests/test_content.py`, new):** `image()/document()/audio()` from path, bytes,
  file-like, and URL; MIME inference; magic-byte sniffing; `Message.content_parts()`
  synthesis; `Message.to_dict()` multimodal shape.
- **Per-provider:** for each of the four, assert an image message produces the exact native
  payload (OpenAI `image_url`, Anthropic `image` block, Gemini `inlineData`, Ollama `images`
  array). Extend in later milestones to document and audio shapes.
- **Conformance (`tests/conformance/`):** extend `contract.py` /
  `test_provider_contract.py` and `FakeConformantProvider` so that, per declared capability:
  a provider with `vision=True` accepts an `ImagePart` and emits a non-empty media payload; a
  provider with `vision=False` raises `UnsupportedModalityError`. Same pattern for
  `documents`, `audio_in`, `image_out`. This makes "all providers behave the same" enforced,
  per DEVELOPMENT.md Part VII.
- **Inspect/record:** assert base64 is elided in `pretty()` and `CallRecord.to_dict()` while
  the real payload retains the bytes.
- **Async parity:** every provider's async path mirrors sync for at least one multimodal case.
- **Capabilities (`tests/test_provider_capabilities.py`):** assert the new flags match
  observed behavior.

Gate: `ruff check`, `pyright`, `pytest` all green before each milestone merges.

---

## Part IX — Milestones (semver-aligned; current `1.2.0`)

| Version | Theme | Deliverable | Why this slice |
| --- | --- | --- | --- |
| **1.3.0** | **Image input (vision)** | Content-parts primitive, `image()` helper, public exports, all four providers' image serialization, `vision` capability, `UnsupportedModalityError`, inspect/record elision, conformance vision clause, docs + CHANGELOG | The highest-value, most-requested modality; establishes the parts model everything else builds on |
| **1.4.0** | **Document (PDF) input** | `DocumentPart` + `document()`; OpenAI / Anthropic / Gemini support; Ollama declares `documents=False`; conformance doc clause | Reuses the 1.3 machinery; only new payload shapes + caps |
| **1.5.0** | **Audio input** | `AudioPart` + `audio()`; OpenAI + Gemini support; Anthropic/Ollama declare `audio_in=False` | Same pattern, narrower provider set |
| **1.6.0** | **Image-generation output** | `Result.images` + `GeneratedImage`; OpenAI (Images API / `gpt-image-1`) and Gemini image models; response parsing + tests | Output path is independent of the input parts model; isolate its risk |
| **1.7.0** | **Contract freeze** | Finalize the multimodal Provider Contract clauses, publish the extended conformance suite for third-party plugins, lock the public multimodal surface under semver | Ecosystem readiness, mirroring the 1.0.0 contract freeze |

Each milestone's Definition of Done (extends DEVELOPMENT.md Part V):

- [ ] Implements the modality sync **and** async, shared mapping in a helper (no drift).
- [ ] Capabilities are truthful and backed by behavior; conformance clause added/green.
- [ ] Inspect + record stay readable (media elided), real request unchanged.
- [ ] `ruff` / `pyright` / `pytest` green; docs (`docs/concepts/multimodal.md`, README
      example) and `CHANGELOG.md` updated.

---

## Part X — Risks & open questions

- **Frozen-dataclass back-compat.** Adding `parts` to a frozen `Message` is additive and
  safe; changing `content`'s type would not be. Stay additive (decision II.1).
- **Memory.** Large media must not be base64-duplicated early or logged. Encode lazily at
  serialize time; elide in views (Part VI).
- **Model vs provider capability mismatch.** A `vision=True` provider can still get a
  text-only model; that surfaces as a provider error, not a SlimX promise. Document loudly.
- **Ollama URL images.** Requires fetch+encode or rejection — pick per decision II.2; it's the
  one provider that can't pass a URL straight through.
- **`oai:` compatibility variance.** Self-hosted OpenAI-compatible servers vary in multimodal
  support; gate on capability and treat failures as the server's, not SlimX's.
- **Image-gen API surface.** ✅ Resolved: image generation is a dedicated path
  (`ImageRequest` + `Model.generate_image()` / `inspect_image()`), not an overload of
  `chat()`, since OpenAI's Images endpoint takes a prompt rather than a message list. Gemini,
  whose images come back through `generateContent`, routes its `generate_image()` through the
  chat path and still surfaces inline images on `Result.images` during ordinary calls.
