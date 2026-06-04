# Parallel execution

`parallel(...)` fans **one** SlimX call out to **multiple** models concurrently and
collects the results into a single, inspectable object. It is a thin layer above the
`Model`/`Client` runtime — it composes them and never reaches into provider internals.

```python
from slimx import parallel

ensemble = parallel([
    "google:gemini-3.5-flash",
    "openai:gpt-4.1-nano",
])

res = ensemble("Explain SlimX in one paragraph.")

for item in res.results:
    print(item.model, item.result.text if item.ok else item.error)
```

Single-model usage stays simple with `llm(...)`; reach for `parallel(...)` only when
you actually want several models at once.

## Modes

| Mode | Behavior | `text` | `winner` |
| --- | --- | --- | --- |
| `all` (default) | Run every model; return every result | `None` | `None` |
| `race` | Return the first successful result; abandon the rest | winner's text | first success |
| `compare` | Run all; build a readable side-by-side of every answer | comparison | `None` |
| `judge` | Run all candidates, then a judge model picks/merges the best | judged answer | the judge's result |

```python
# Compare every model side by side:
parallel(models, mode="compare")("...")      # text is a readable comparison

# Let a judge model pick or synthesize the best answer:
res = parallel(models, mode="judge", judge="openai:gpt-4.1-mini")("...")
print(res.text)            # the judge's final answer
print(res.candidates)      # every model's original answer (== res.results)
print(res.winner.model)    # the judge model
```

In `judge` mode the underlying candidates are always preserved in `res.results` (also
available as `res.candidates`) — the judge never hides the disagreement it resolved.

Extra keyword arguments are forwarded to each underlying model, e.g.
`parallel(models, temperature=0.2, timeout=30)`.

## The result object

```python
@dataclass
class ParallelItem:
    provider: str            # e.g. "google"
    model: str               # e.g. "google:gemini-3.5-flash"
    result: Result | None    # the normalized SlimX Result on success
    error: str | None        # "ErrorType: message" on failure
    elapsed_ms: int | None
    # .ok -> True when result is present

@dataclass
class ParallelResult:
    text: str | None             # winner's text for single-answer modes, else None
    results: list[ParallelItem]  # every attempt, in input order
    errors: list[ParallelItem]   # the failed subset
    winner: ParallelItem | None  # the chosen attempt (race)
    trace: dict                  # mode, models, elapsed_ms, ok_count, error_count
```

## Inspectability (the contract)

Parallel execution never hides what happened:

- **Failures are surfaced, not swallowed** — a model that errors becomes a
  `ParallelItem` with `.error` set and appears in `res.errors`; the call does not
  raise just because one model failed.
- **Every result keeps its `raw`** — `item.result.raw` is the untouched provider
  response.
- **`trace` records the run** — mode, the model list, total elapsed time, and ok/error
  counts.

## Cost and latency

Parallel calls hit every model, so they cost more than a single call. `parallel(...)`
is always explicit and never a default. In `all` mode the call waits for the slowest
model; in `race` mode it returns as soon as the first model succeeds (slower calls are
abandoned, though already-running HTTP requests finish in the background).

## Not yet

Parallel calls don't run tools or stream, and `consensus` / `majority` voting modes
aren't implemented yet. As with `judge`, any future aggregating mode will always expose
the underlying candidates rather than hide them.
