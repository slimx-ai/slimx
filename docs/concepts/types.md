# Core types

SlimX is built on a small set of provider-neutral types. These types are used by
providers, the low-level clients, and (indirectly) the high-level API.

## Message

`Message` is the canonical conversation unit:

- `role`: `system | user | assistant | tool`
- `content`: text content
- tool result fields: `tool_name`, `tool_call_id` (used when role is `tool`)
- `metadata`: optional extra fields

Helpers:

```python
from slimx import M

msgs = [
  M.system("You are concise."),
  M.user("Say hello."),
]
````

## Tools

A tool is described by:

* `ToolSpec`: name, description, JSON schema parameters
* `ToolCall`: id, name, `arguments` (JSON string)

Providers return tool calls in different formats; SlimX normalizes to `ToolCall`.

## Result

A completed request returns:

* `text`
* `tool_calls` (if the model requested any tools)
* `usage` (optional)
* `raw` (provider response as a dict)

## StreamEvent

Streaming returns normalized events:

* `text_delta`: incremental text
* `tool_call`: tool call event
* `done`: end of stream
* `error`: stream error
