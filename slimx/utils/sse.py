def iter_sse_lines(byte_iter):
    buf=b""
    for chunk in byte_iter:
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            yield line.decode("utf-8", errors="replace")
    if buf:
        yield buf.decode("utf-8", errors="replace")

def iter_sse_data(byte_iter):
    for line in iter_sse_lines(byte_iter):
        line=line.strip()
        if not line:
            continue
        if line.startswith("data:"):
            yield line[len("data:"):].strip()
