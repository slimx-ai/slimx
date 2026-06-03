# High-level API

The high-level API is optimized for quick productivity:

- `llm("provider:model")`
- `model(prompt)`
- `model.stream(prompt)`
- `model.json(prompt, schema=...)`

High-level calls still return normalized `Result` objects with text, usage, tool calls, parsed data, and trace metadata.
