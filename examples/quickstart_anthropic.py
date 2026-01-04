from slimx import llm

m = llm("anthropic:claude-sonnet-4-5", max_tokens=200)
print(m("Explain CRDs in Kubernetes in 2 sentences.").text)
