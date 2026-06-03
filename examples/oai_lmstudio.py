"""Call an LM Studio server through SlimX's OpenAI-compatible (`oai:`) provider.

In LM Studio: load a model, then start the local server (Developer tab →
Start Server). It defaults to port 1234 and exposes an OpenAI-compatible
`/v1/chat/completions` endpoint.

The model string after `oai:` can be the model's LM Studio identifier, or just
`local-model` if a single model is loaded.
"""

from slimx import llm


def main() -> None:
    model = llm(
        "oai:local-model",
        provider_kwargs={
            "base_url": "http://localhost:1234/v1",
            "api_key": "lm-studio",  # LM Studio accepts any non-empty key
        },
        timeout=120,
    )

    result = model("Explain local-first AI in one paragraph.")
    print(result.text)


if __name__ == "__main__":
    main()
