# Using SlimX Inside SlimX-RAG

SlimX-RAG handles deterministic retrieval. SlimX handles provider-neutral generation.

The customer-demo flow is:

```text
question -> retrieve chunks -> build grounded prompt -> SlimX model call -> cited answer + trace
```

This keeps the retrieval trace and model trace separate, which makes the system easier to debug and explain to customers.
