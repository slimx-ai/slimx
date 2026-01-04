from slimx import llm

m = llm("openai:gpt-4.1-nano", temperature=0.2)
res = m("Write a haiku about fog and streetlights.")
print(res.text)
print(res.usage)
