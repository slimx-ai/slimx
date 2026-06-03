"""Fan one prompt out to several models and compare every answer (mode="all").

`parallel(...)` runs each model concurrently and returns every result — and every
error — without hiding anything.
"""

from slimx import parallel


def main() -> None:
    ensemble = parallel(
        [
            "google:gemini-3.5-flash",
            "openai:gpt-4.1-nano",
        ],
        temperature=0.2,
    )

    res = ensemble("Explain why small, inspectable LLM runtimes are useful, in one paragraph.")

    for item in res.results:
        print(f"\n=== {item.model} ({item.elapsed_ms} ms) ===")
        if item.ok and item.result:
            print(item.result.text)
        else:
            print(f"[error] {item.error}")

    print("\ntrace:", res.trace)


if __name__ == "__main__":
    main()
