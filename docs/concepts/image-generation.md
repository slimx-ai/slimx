# Image generation & editing

SlimX exposes image *output* (and image *editing*) as provider-neutral operations,
normalized onto `Result.images` as `GeneratedImage` objects — never as base64
stuffed into a text string.

There are three ways to produce an image, in increasing order of conversational
power:

| Operation | API | OpenAI endpoint |
|-----------|-----|-----------------|
| Direct generation | `Model.generate_image(prompt)` | `/images/generations` |
| Direct editing | `Model.edit_image(image, instruction)` | `/responses` (hosted image tool) |
| In-conversation generation | `model(prompt, image_generation=ImageGenerationOptions(...))` | `/responses` (hosted image tool) |

## Direct generation (unchanged)

```python
from slimx import llm
m = llm("openai:gpt-image-1")
res = m.generate_image("a red bike", size="1024x1024")
res.images[0].data   # PNG bytes
```

Gated by `capabilities.image_out`; providers that don't support it raise
`UnsupportedModalityError`.

## The OpenAI Responses hosted image tool

A *text* model (e.g. `gpt-5.5`) can call the hosted `image_generation` tool and
return real image bytes inline. SlimX routes an OpenAI call to the **Responses
API** (`/responses`) automatically whenever `ChatRequest.image_generation` is set;
plain text / function-tool chat stays on `/chat/completions` (fully backward
compatible).

```python
from slimx import llm, ImageGenerationOptions

m = llm("openai:gpt-5.5")
res = m("Generate an image of a gray tabby cat hugging an otter",
        image_generation=ImageGenerationOptions(size="1024x1024", action="generate", force=True))
img = res.images[0]
img.data            # decoded PNG bytes (decoded exactly once)
img.revised_prompt  # the model's optimized prompt, if returned
img.provider_response_id, img.provider_call_id  # provider state ids
```

`ImageGenerationOptions` maps onto the tool object: `size`, `quality`,
`output_format`, `background`, `output_compression`, `partial_images`, plus
`action` (`auto` / `generate` / `edit`) and `force` (sets `tool_choice` so the
model *must* call the tool).

## Editing

```python
res = m.edit_image(png_bytes, "change the scarf from orange to blue and add snow")
res.images[0].operation   # "edit"
```

`edit_image` accepts raw `bytes`, an `ImagePart`, an `ImageInput`, a dict, or a
list of those (multiple source images). It sends the source as an `input_image`
to the Responses API and forces the image tool. **Editing operates on the supplied
bytes**, so it stays durable — it does not depend on ephemeral provider
conversation state.

Conversational revisions can additionally pass `previous_response_id=` to continue
from an earlier response, but that is an optimization layered on top of the durable
bytes path, never the only copy of the image.

## Normalized image results

`GeneratedImage` carries inline `data` bytes (or a hosted `url`) plus best-effort
metadata that is safe to persist: `mime_type`, `width`/`height`, `provider`,
`model`, `operation`, `provider_response_id`, `provider_call_id`, `revised_prompt`,
`source_ids`, `output_index`, and a `metadata` dict. The MIME type is sniffed from
the bytes (`suggested_extension` derives the file extension).

## Streaming

`model.stream(prompt, image_generation=...)` emits normalized image events
alongside text:

- `image_started` — generation began;
- `image_partial` — a transient preview frame (`image_partial_b64`), never a final asset;
- `image_completed` — carries the final `GeneratedImage`.

Final images are read from the terminal `response.completed` payload, so they are
captured even when `partial_images=0`. Partial frames are ephemeral and must not be
persisted as the final result.

## Capabilities

`describe_provider("openai")` and `ProviderCapabilities` report:

- `image_in` (alias of `vision`) — image input;
- `image_out` — image-generation output;
- `image_edit` — editing;
- `hosted_image_tool` — the in-conversation Responses image tool;
- `image_partial_streaming` — partial-image stream events.

## Provider limitations

- The hosted image tool is **OpenAI Responses only**. Generic OpenAI-compatible
  servers (the `oai` provider — vLLM, LM Studio, …) do **not** advertise it
  (`hosted_image_tool=False`).
- Anthropic and the `oai` provider declare `image_out=False`/`image_edit=False`
  and raise `NotImplementedError`/`UnsupportedModalityError` rather than silently
  advertising unsupported modalities.
- Google's image path remains the `generateContent` route used by
  `generate_image`.
