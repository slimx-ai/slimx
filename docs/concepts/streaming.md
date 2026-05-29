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
