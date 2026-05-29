# Tool Calling And Safety Boundaries

Use `@tool` to declare explicit, typed tool schemas and pass `tool_runtime="auto"` to allow SlimX to execute requested tools.

```python
from slimx import llm, tool

@tool
def add(a: int, b: int) -> int:
    return a + b

model = llm("openai:gpt-4.1-nano", tools=[add], tool_runtime="auto")
print(model("What is 2 + 3?").text)
```

SlimX keeps tool execution explicit: tools are Python callables, failures raise `ToolExecutionError`, and the final `Result.trace` includes tool-loop metadata.
