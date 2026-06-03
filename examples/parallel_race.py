"""Return whichever model answers first (mode="race") — useful for low latency.

The first successful result wins; slower models are abandoned. Failures are still
recorded in `res.errors` so nothing is hidden.
"""

from slimx import parallel


def main() -> None:
    ensemble = parallel(
        [
            "google:gemini-3.5-flash",
            "openai:gpt-4.1-nano",
            "ollama:llama3.2:3b",
        ],
        mode="race",
        timeout=30,
    )

    res = ensemble("Give a short, punchy tagline for SlimX.")

    if res.winner:
        print(f"winner: {res.winner.model} ({res.winner.elapsed_ms} ms)")
        print(res.text)
    else:
        print("all models failed:")
        for item in res.errors:
            print(f"  {item.model}: {item.error}")

    print("\ntrace:", res.trace)


if __name__ == "__main__":
    main()
