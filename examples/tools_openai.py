from slimx import llm, tool

@tool
def add(a: int, b: int) -> int:
    "Add two integers."
    return a + b

m = llm("openai:gpt-4.1-nano", tools=[add], tool_runtime="auto")
res = m("What is 12 + 30?")
print(res.text)
print(res.tool_calls)
