"""Call a vLLM server through SlimX's OpenAI-compatible (`oai:`) provider.

Start vLLM first, for example:

    vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000

vLLM exposes an OpenAI-compatible `/v1/chat/completions` endpoint, so use the
`oai:` provider and point it at the server. The model string after `oai:` is the
model name vLLM was started with.
"""

from slimx import llm


def main() -> None:
    model = llm(
        "oai:Qwen/Qwen2.5-7B-Instruct",
        provider_kwargs={
            "base_url": "http://localhost:8000/v1",
            "api_key": "EMPTY",  # vLLM ignores auth unless configured with --api-key
        },
        timeout=120,
    )

    result = model("Explain why OpenAI-compatible local servers are useful.")
    print(result.text)


if __name__ == "__main__":
    main()
