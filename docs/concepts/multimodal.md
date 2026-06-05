# Multimodal

SlimX messages are text-first but can also carry **images**, **documents**, and
**audio**, and can receive **generated images** back. The same call works across
providers — you attach media once and each provider serializes it into its own
native shape.

```python
from slimx import llm, image

m = llm("openai:gpt-4o")
print(m("What's in this picture?", images=[image("diagram.png")]).text)
```

## Attaching media

Build parts with `image()`, `document()`, and `audio()`. Each accepts a
filesystem path, raw `bytes`, a file-like object, a `data:` URI, or an
`http(s)://` URL:

```python
from slimx import image, document, audio

image("photo.jpg")                       # inferred image/jpeg from the bytes
image(png_bytes, mime_type="image/png")  # explicit MIME
image("https://example.com/cat.png")     # remote URL — passed through, not downloaded
document("report.pdf")                    # application/pdf
audio("clip.wav")                         # audio/wav
```

Remote URLs are **passed through untouched** by default, so building a request
never makes a surprise network call. Pass `fetch=True` if you want SlimX to
download and inline the bytes (required for providers like Ollama that only
accept base64 data).

Attach them to a call via keyword, or build a `Message` directly:

```python
from slimx import Message, image

# high-level
m("Compare these", images=[image("a.png"), image("b.png")])

# low-level
Message.user("Compare these", images=[image("a.png"), image("b.png")])
```

## Capabilities

Multimodal support varies by provider (and by model). Each provider declares
truthful capability flags; check them before you call:

```python
from slimx import llm
print(llm("anthropic:claude-sonnet-4-6").capabilities.vision)  # True
```

| Provider | `vision` | `documents` | `audio_in` | `image_out` |
| --- | :---: | :---: | :---: | :---: |
| `openai` | ✅ | ✅ | ✅ | ✅ |
| `oai` | ✅ | ✅ | ✅ | — |
| `anthropic` | ✅ | ✅ | — | — |
| `google` | ✅ | ✅ | ✅ | ✅ |
| `ollama` | ✅ (vision models) | — | — | — |

(`oai` covers generic OpenAI-compatible servers, which speak Chat Completions but
seldom expose the separate image-generation endpoint, so `image_out` is not
promised there.)

If you send media a provider hasn't declared support for, SlimX raises
`UnsupportedModalityError` (a `ProviderError` — it fails fast and is never
retried) instead of silently dropping it.

!!! note "Capability is provider-level, support is model-level"
    A provider declaring `vision=True` means its API can carry an image. The
    *model* you pick still has to be a vision model (e.g. `gpt-4o`, not an older
    text-only model; `llava` rather than a plain `llama3.2` on Ollama). A model
    mismatch surfaces as the provider's own error.

## Generated images (image-out)

Use `generate_image()` to produce images from a prompt; results land on
`Result.images`. The same call works for OpenAI (a dedicated Images endpoint) and
Gemini (which returns images through its normal content endpoint):

```python
from slimx import llm

res = llm("openai:gpt-image-1").generate_image("a red bicycle", size="1024x1024")
for img in res.images:
    print(img.mime_type)              # "image/png"
    open("out.png", "wb").write(img.data)

res = llm("google:gemini-2.5-flash-image").generate_image("a red bicycle")
```

Extra keywords (`n`, `size`, and anything else like `quality` or `style`) are
passed through to the provider. Async models expose `await m.generate_image(...)`,
and `inspect_image(prompt, ...)` dry-runs the request without sending it. Calling
`generate_image()` on a provider that doesn't declare `image_out` raises
`UnsupportedModalityError`.

Gemini image models that also return image inline parts during a normal call
still surface them on `Result.images`, so a plain `m("draw …")` works too.

## Inspect and records stay readable

Base64 media would drown a dry-run or a saved record, so `inspect().pretty()`
and `CallRecord` **elide** large base64 blobs to a short placeholder. The request
SlimX actually sends still carries the real bytes — only the human-facing view is
trimmed:

```python
print(llm("openai:gpt-4o").inspect("hi", images=[image("a.png")]).pretty())
# ... "url": "data:image/png;base64,<48213 base64 chars elided>" ...
```

## Native shapes (what gets sent)

For one image, each provider emits its own wire format:

| Provider | Shape |
| --- | --- |
| `openai` / `oai` | `content: [{"type":"image_url","image_url":{"url": …}}]` |
| `anthropic` | `{"type":"image","source":{"type":"base64"\|"url", …}}` |
| `google` | `parts: [{"inlineData":{"mimeType","data"}}]` (or `fileData` for URLs) |
| `ollama` | message-level `images: ["<base64>"]` |
