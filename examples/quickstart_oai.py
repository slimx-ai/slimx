from slimx import llm


def main() -> None:
    model = llm(
        "oai:Qwen/Qwen2.5-7B-Instruct",
        provider_kwargs={
            "base_url": "http://localhost:8000/v1",
            "api_key": "EMPTY",
        },
        timeout=120,
    )

    result = model("Explain why OpenAI-compatible APIs are useful for local model serving.")

    print(result.text)


if __name__ == "__main__":
    main()