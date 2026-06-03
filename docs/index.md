# SlimX

SlimX is a slim, intuitive, lightweight Python library for explicit LLM execution.

It is built for systems where provider choice, request shape, streaming events, tool calls, structured outputs, retries, and failures should remain visible and testable.

Core ideas:

- Providers are adapters, not architecture.
- High-level APIs are fast to use, while low-level APIs expose request/response primitives.
- Every production feature should be inspectable enough for RAG, agent, and customer-demo traces.
