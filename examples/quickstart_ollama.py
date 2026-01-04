from slimx import llm

m = llm("ollama:llama3.2")
print(m("Say hello in Dutch.").text)
