"""Call Ollama's OpenAI-compatible endpoint through the `oai:` provider.

Ollama exposes BOTH:
  - its native `/api/chat` runtime  -> use the `ollama:` provider, and
  - an OpenAI-compatible `/v1`       -> use the `oai:` provider (this example).

Prefer `ollama:` for the native runtime (native usage fields, options like
`num_predict`, `keep_alive`). Use `oai:` when you specifically want the
OpenAI-compatible surface, e.g. to share code paths with other `oai:` servers.

    ollama serve
    ollama pull llama3.2:3b
"""

from slimx import llm


def main() -> None:
    model = llm(
        "oai:llama3.2:3b",
        provider_kwargs={
            "base_url": "http://localhost:11434/v1",
            "api_key": "ollama",  # any non-empty value
        },
        timeout=120,
    )

    result = model("Explain the difference between native Ollama and OpenAI compatibility.")
    print(result.text)


if __name__ == "__main__":
    main()
