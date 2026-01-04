async def aiter_sse_lines(aiter):
    buf=b""
    async for chunk in aiter:
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            yield line.decode("utf-8", errors="replace")
    if buf:
        yield buf.decode("utf-8", errors="replace")

async def aiter_sse_data(aiter):
    async for line in aiter_sse_lines(aiter):
        line=line.strip()
        if not line:
            continue
        if line.startswith("data:"):
            yield line[len("data:"):].strip()
