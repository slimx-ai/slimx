import json
def iter_ndjson(byte_iter):
    buf=b""
    for chunk in byte_iter:
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            s=line.strip()
            if not s: 
                continue
            yield json.loads(s)
    tail=buf.strip()
    if tail:
        yield json.loads(tail)

async def aiter_ndjson(byte_iter):
    buf=b""
    async for chunk in byte_iter:
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            s=line.strip()
            if not s:
                continue
            yield json.loads(s)
    tail=buf.strip()
    if tail:
        yield json.loads(tail)
