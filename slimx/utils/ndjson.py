"""NDJSON (newline-delimited JSON) stream parsing.

Malformed lines are skipped rather than aborting the whole stream: a single bad
frame from a provider should not kill an otherwise-valid response.
"""

import json


def _loads(line: str):
    try:
        return json.loads(line)
    except Exception:
        return None


def iter_ndjson(byte_iter):
    buf = b""
    for chunk in byte_iter:
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            s = line.strip()
            if not s:
                continue
            obj = _loads(s.decode("utf-8", errors="replace"))
            if obj is not None:
                yield obj
    tail = buf.strip()
    if tail:
        obj = _loads(tail.decode("utf-8", errors="replace"))
        if obj is not None:
            yield obj


async def aiter_ndjson(byte_iter):
    buf = b""
    async for chunk in byte_iter:
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            s = line.strip()
            if not s:
                continue
            obj = _loads(s.decode("utf-8", errors="replace"))
            if obj is not None:
                yield obj
    tail = buf.strip()
    if tail:
        obj = _loads(tail.decode("utf-8", errors="replace"))
        if obj is not None:
            yield obj
