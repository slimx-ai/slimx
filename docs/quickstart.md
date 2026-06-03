# Quickstart

```python
from slimx import llm

model = llm("openai:gpt-4.1-nano", temperature=0.2)
result = model("Summarize SlimX in one sentence.")
print(result.text)
print(result.trace)
```

Streaming uses normalized event types:

```python
for event in model.stream("Write three words."):
    if event.type == "text_delta":
        print(event.text, end="")
```
