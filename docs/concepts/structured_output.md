# Structured Output

Use `.json(..., schema=Dataclass)` when a provider supports JSON response formatting.

```python
from dataclasses import dataclass
from slimx import llm

@dataclass
class City:
    name: str
    country: str

result = llm("openai:gpt-4.1-nano").json("Paris is in France.", schema=City)
print(result.data)
```

The raw text remains available as `result.text`, while parsed data is available as `result.data` and `result.parsed`.
