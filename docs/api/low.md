# Low-level API

Use the low-level API when building systems such as RAG and agents:

```python
from slimx import Client, Message
from slimx.low import ChatRequest
from slimx.providers import get_provider

client = Client(get_provider("openai"), timeout=30)
result = client.chat(ChatRequest(model="gpt-4.1-nano", messages=[Message.user("Hello")]))
```

The low-level API keeps request objects, provider selection, retries, timeouts, and traces explicit.
