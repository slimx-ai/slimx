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

## Validation + repair

By default, if the model returns invalid JSON or output that doesn't fit the schema,
`.json(...)` raises a `SchemaError`. Pass `repair=N` to automatically re-prompt the model
with the error and ask it to fix its output, up to `N` times:

```python
result = llm("openai:gpt-4.1-nano").json(
    "Extract the city.",
    schema=City,
    repair=2,   # up to 2 corrective retries on bad JSON / wrong shape
)
```

`repair=0` (the default) keeps the original fail-fast behavior. Repair is intentionally
lightweight — it validates by parsing the JSON and constructing the dataclass (catching
malformed JSON and missing required fields) without pulling in a validation framework.

