# Streaming Event Contract

SlimX normalizes provider streaming into `StreamEvent` records.

Event types:

- `text_delta`
- `tool_call`
- `done`
- `error`

```python
for event in model.stream("Tell a short story."):
    if event.type == "text_delta":
        print(event.text, end="", flush=True)
```

Every stream ends with exactly one `done` event. When the provider sends a final record, the
`done` event carries it as `raw`. Ollama ends a stream with a final line (`"done": true`, with
`done_reason` and the token counts); a stream that ended without it, because the connection
closed or the runtime failed, ends with a `done` event whose `raw` is `None`. An `error` event
reports a failure the provider sent inside the stream, such as Ollama's `{"error": ...}` line;
the stream still ends with `done`.
