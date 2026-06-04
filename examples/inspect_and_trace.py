"""Inspectability: dry-run a request, observe calls with hooks, save a call record.

Run with an OpenAI key set (OPENAI_API_KEY). The `inspect()` part needs no network.
"""

from slimx import CallRecord, llm


def main() -> None:
    # 1) Dry-run: see the exact request without sending it (secrets redacted).
    m = llm("openai:gpt-4.1-nano", temperature=0.2)
    print("=== inspect (dry-run) ===")
    print(m.inspect("Explain SlimX in one line.").pretty())

    # 2) Trace hooks: observe each call as it happens.
    def log(event):
        print("[hook]", event)

    traced = llm("openai:gpt-4.1-nano", hooks={"before_call": log, "after_call": log})
    res = traced("Give me a one-line tagline for SlimX.")
    print("\nanswer:", res.text)

    # 3) Reproducible call record: save the whole call, reload it later.
    record = res.to_record()
    record.save("slimx_run.json")
    loaded = CallRecord.load("slimx_run.json")
    print("\nsaved record ->", loaded.provider, loaded.model, "| tokens:",
          loaded.response["usage"])


if __name__ == "__main__":
    main()
