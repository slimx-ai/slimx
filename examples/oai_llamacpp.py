"""Call a llama.cpp server through SlimX's OpenAI-compatible (`oai:`) provider.

Start the llama.cpp server first, for example:

    llama-server -m ./models/your-model.gguf --port 8080

llama.cpp's server exposes an OpenAI-compatible `/v1/chat/completions` endpoint.
It usually serves a single loaded model, so the exact model string after `oai:`
is not important — `local-model` is a fine placeholder.
"""

from slimx import llm


def main() -> None:
    model = llm(
        "oai:local-model",
        provider_kwargs={
            "base_url": "http://localhost:8080/v1",
            "api_key": "EMPTY",
        },
        timeout=120,
    )

    result = model("Explain GGUF models in one paragraph.")
    print(result.text)


if __name__ == "__main__":
    main()
