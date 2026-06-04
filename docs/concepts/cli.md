# CLI & model discovery

Installing SlimX adds a `slimx` command for diagnostics and discovery. It needs no
extra dependencies.

## `slimx doctor`

Check what's configured and what's reachable — safe to run with nothing set up.

```bash
slimx doctor
```

```text
SlimX 0.10.0  ·  Python 3.10.8

  anthropic  key missing — set ANTHROPIC_API_KEY
  google     key OPENAI_API_KEY: found
  oai        not configured — set SLIMX_OAI_BASE_URL or OAI_BASE_URL
  ollama     reachable · 3 model(s) — llama3.2:3b, qwen2.5:7b, nomic-embed-text
  openai     key OPENAI_API_KEY: found
```

For local providers (`ollama`, `oai`) it probes the server and lists models. For cloud
providers it reports whether the API key is set — and only makes a network call if you
add `--probe`:

```bash
slimx doctor              # report config; probe local servers only
slimx doctor ollama       # limit to one provider
slimx doctor --probe      # also probe cloud providers (uses your keys)
```

This is the fastest way to answer "why isn't my model working?" — usually a missing key
or the wrong base URL.

## `slimx models`

List the models a provider/server exposes, so you don't have to guess the model string.

```bash
slimx models ollama       # queries Ollama's /api/tags
slimx models oai          # queries the OpenAI-compatible /v1/models
slimx models openai       # needs OPENAI_API_KEY
```

## `slimx providers`

List registered providers (built-in and plugins) with their capabilities.

```bash
slimx providers
```

## In code

The same discovery is available programmatically:

```python
from slimx import list_models, describe_provider

list_models("ollama")                      # ['llama3.2:3b', ...]
list_models("oai", base_url="http://localhost:8000/v1")
describe_provider("google")                # capability flags, no network/key needed
```
