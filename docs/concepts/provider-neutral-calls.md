# Provider-neutral Calls

SlimX model IDs use `provider:model` strings. Providers are selected through adapters, while application code keeps using the same request and result shapes.

Examples:

- `openai:gpt-4.1-nano`
- `anthropic:claude-3-5-haiku-latest`
- `ollama:llama3.1`

Provider-neutral calls let demos and deployments switch between hosted API models and local sovereignty fallbacks.
