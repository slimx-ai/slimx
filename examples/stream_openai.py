from slimx import llm

m = llm("openai:gpt-4.1-nano")
for ev in m.stream("Tell a short story in 5 lines."):
    if ev.type == "token":
        print(ev.text, end="", flush=True)
print()
