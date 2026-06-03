# OpenAI-compatible servers (`oai:`)

Many local and self-hosted runtimes expose an OpenAI-compatible
`/v1/chat/completions` API. The `oai:` provider talks to all of them with one
code path — no per-server provider needed.

```python
from slimx import llm

model = llm(
    "oai:MODEL_NAME",
    provider_kwargs={"base_url": "http://HOST:PORT/v1", "api_key": "EMPTY"},
    timeout=120,
)
print(model("Hello from an OpenAI-compatible server.").text)
```

`SLIMX_OAI_BASE_URL` (or `OAI_BASE_URL`) sets the base URL from the environment;
`SLIMX_OAI_API_KEY` (or `OAI_API_KEY`) sets the key, defaulting to `EMPTY` for
local servers that ignore authentication.

## Recipes

Runnable versions of each are in `examples/`.

### vLLM

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
```

```python
llm("oai:Qwen/Qwen2.5-7B-Instruct",
    provider_kwargs={"base_url": "http://localhost:8000/v1", "api_key": "EMPTY"})
```

The model string after `oai:` is the model vLLM was started with.

### llama.cpp server

```bash
llama-server -m ./models/your-model.gguf --port 8080
```

```python
llm("oai:local-model",
    provider_kwargs={"base_url": "http://localhost:8080/v1", "api_key": "EMPTY"})
```

Usually serves one loaded model, so `local-model` is a fine placeholder.

### LM Studio

Load a model, then start the local server (Developer → Start Server, port 1234).

```python
llm("oai:local-model",
    provider_kwargs={"base_url": "http://localhost:1234/v1", "api_key": "lm-studio"})
```

### Ollama `/v1`

```bash
ollama serve
ollama pull llama3.2:3b
```

```python
llm("oai:llama3.2:3b",
    provider_kwargs={"base_url": "http://localhost:11434/v1", "api_key": "ollama"})
```

Prefer the native `ollama:` provider unless you specifically want the
OpenAI-compatible surface — `ollama:` gives you native usage fields and options
like `num_predict` and `keep_alive`.

## When to use `ollama:` vs `oai:`

- Use `ollama:` for Ollama's native `/api/chat` runtime (native usage, options).
- Use `oai:` for Ollama's OpenAI-compatible `/v1` endpoint, and for vLLM,
  llama.cpp server, LM Studio, LocalAI, and internal gateways.

## Finding the model string

If you are unsure what model name to pass, ask the server's OpenAI-compatible
listing endpoint:

```bash
curl http://localhost:8000/v1/models
```

Use the `id` from the response as the part after `oai:`.

## Troubleshooting

- **Connection refused** — the server isn't running or the port is wrong. Check
  the `base_url` host/port.
- **404 on `/v1/chat/completions`** — your `base_url` is missing the `/v1`
  suffix, or the server uses a different path. Most servers expect the base URL
  to end in `/v1`.
- **401 / auth errors** — pass a non-empty `api_key`. Some gateways require a
  real key even when local servers don't.
- **Timeouts on first call** — local models can be slow to load. Increase
  `timeout` (e.g. `timeout=120`) or warm the model first.
- **Wrong/empty model** — confirm the model name with `GET /v1/models` (above).
