# Inspectability

SlimX's core promise is that you can see exactly what it does. Three features make
every call transparent — without a hosted platform or any extra dependency.

## Inspect mode (dry-run)

See the exact HTTP request SlimX would send — method, URL, headers, and JSON payload —
**without making the call**. Secret header values are redacted.

```python
from slimx import llm

m = llm("openai:gpt-4.1-nano", temperature=0.2)
req = m.inspect("Explain SlimX in one line.")

print(req.method, req.url)
print(req.headers)        # {'Authorization': 'Bearer ***', ...}
print(req.payload)        # the exact JSON body
print(req.pretty())       # a readable, secret-free dump
```

`inspect(prompt, stream=True)` shows the streaming endpoint where it differs (e.g.
Gemini's `:streamGenerateContent`). It returns an `InspectedRequest` and never touches
the network — ideal for debugging provider differences or reviewing a payload before
sending. Also available at the low level as `Client.inspect(req)`.

## Trace hooks (bring your own observability)

Every call already carries a `trace` dict (`provider`, `model`, `elapsed_ms`, `retries`,
`tool_steps`, `tool_call_count`, `timeout`). Hooks let you observe calls as they happen —
log them, push metrics, anything — with no SaaS dependency.

```python
from slimx import llm

def log(event):
    print(event)

m = llm("google:gemini-3.5-flash", hooks={
    "before_call": log,   # {'phase': 'before_call', 'provider': ..., 'model': ...}
    "after_call": log,    # the full trace + {'ok': True}, or {'ok': False, 'error': ...}
})
m("Hello")
```

`before_call` fires before each request; `after_call` fires on success (with the trace)
and on failure (with `ok=False` and the error). A hook that raises is swallowed — it can
never break the underlying call. Hooks are also accepted by `Client(provider, hooks=...)`.

## Reproducible call records

Turn any result into a serializable record of the whole call — the request that went out,
the response, usage, trace, and the SlimX version — then save/load it as JSON. Useful for
debugging, audits, evals, and regression fixtures.

```python
from slimx import llm, CallRecord

res = llm("openai:gpt-4.1-nano")("Capital of France?")

record = res.to_record()
record.save("run.json")

# later, anywhere:
loaded = CallRecord.load("run.json")
print(loaded.provider, loaded.model)
print(loaded.request["messages"])
print(loaded.response["text"])
print(loaded.raw)            # the untouched provider response
```

The Client attaches a compact request snapshot to every `Result` (`res.request`), so a
record is fully self-contained.
