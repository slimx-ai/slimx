"""Optional refusal diagnostics cannot change status semantics or decode without a bound."""

import asyncio
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys

import httpx
import pytest

from slimx.local.engines.ollama import OllamaEngine


class Body(httpx.SyncByteStream):
    def __init__(self, chunks=(), failure=None):
        self.chunks = chunks
        self.failure = failure
        self.read = 0
        self.closed = False

    def __iter__(self):
        for chunk in self.chunks:
            self.read += len(chunk)
            yield chunk
        if self.failure is not None:
            raise self.failure

    def close(self):
        self.closed = True


def serve(monkeypatch, status, body, headers=None):
    clients, responses = [], []
    real_client = httpx.Client

    def handle(request):
        response = httpx.Response(status, stream=body, headers=headers, request=request)
        responses.append(response)
        return response

    def client(*args, **kwargs):
        instance = real_client(*args, transport=httpx.MockTransport(handle), **kwargs)
        clients.append(instance)
        return instance

    monkeypatch.setattr(httpx, "Client", client)
    return clients, responses


@pytest.mark.parametrize("failure", [httpx.ReadError, httpx.RemoteProtocolError, httpx.ReadTimeout])
def test_unreadable_refusal_preserves_status_and_identity(monkeypatch, failure):
    body = Body([b'{"error":"unfinished'], failure("fixture failure"))
    clients, responses = serve(monkeypatch, 502, body)
    with pytest.raises(httpx.HTTPStatusError) as caught:
        list(OllamaEngine("http://fixture.invalid").pull_or_prepare_model("fixture"))
    assert caught.value.response is responses[0]
    assert caught.value.request is responses[0].request
    assert responses[0].status_code == 502
    assert body.closed and responses[0].is_closed and clients[0].is_closed


@pytest.mark.parametrize("encoding", ["gzip", "GZip", "deflate", "br", "unknown", "identity, gzip"])
def test_encoded_refusal_is_not_consumed_or_decoded(monkeypatch, encoding):
    body = Body([b"not a valid compressed stream"])
    clients, responses = serve(monkeypatch, 502, body, {"Content-Encoding": encoding})
    with pytest.raises(httpx.HTTPStatusError) as caught:
        list(OllamaEngine("http://fixture.invalid").pull_or_prepare_model("fixture"))
    assert caught.value.response is responses[0]
    assert caught.value.request is responses[0].request
    assert caught.value.request.headers["Accept-Encoding"] == "identity"
    assert caught.value.response.status_code == 502
    assert body.read == 0
    assert body.closed and responses[0].is_closed and clients[0].is_closed


@pytest.mark.parametrize("status", [400, 503])
@pytest.mark.parametrize(
    "reason",
    ["pull model manifest: file does not exist", "detail " * 400],
    ids=["short", "bounded"],
)
def test_identity_negotiation_preserves_useful_bounded_refusal(monkeypatch, status, reason):
    # Model an engine that honors negotiation: without the explicit request header, httpx
    # advertises compression and this refusal's useful diagnostic is defensively declined.
    real_client = httpx.Client
    clients, responses, bodies = [], [], []

    def handle(request):
        payload = json.dumps({"error": reason}).encode()
        identity = request.headers.get("Accept-Encoding") == "identity"
        body = Body([payload if identity else gzip.compress(payload)])
        response = httpx.Response(
            status,
            stream=body,
            headers={"Content-Encoding": "identity" if identity else "gzip"},
            request=request,
        )
        bodies.append(body)
        responses.append(response)
        return response

    def client(*args, **kwargs):
        instance = real_client(*args, transport=httpx.MockTransport(handle), **kwargs)
        clients.append(instance)
        return instance

    monkeypatch.setattr(httpx, "Client", client)
    with pytest.raises(httpx.HTTPStatusError) as caught:
        list(OllamaEngine("http://fixture.invalid").pull_or_prepare_model("fixture"))
    expected = reason.strip()
    if len(expected) > 2_000:
        expected = expected[:2_000].rstrip() + "…"
    assert str(caught.value) == f"Ollama refused the pull (HTTP {status}): {expected}"
    assert caught.value.request.headers["Accept-Encoding"] == "identity"
    assert caught.value.request.url.path == "/api/pull"
    assert caught.value.response is responses[0]
    assert caught.value.request is responses[0].request
    assert bodies[0].read > 0
    assert bodies[0].closed and responses[0].is_closed and clients[0].is_closed


@pytest.mark.parametrize("signal", [GeneratorExit, KeyboardInterrupt, SystemExit, asyncio.CancelledError])
def test_process_control_is_not_suppressed(monkeypatch, signal):
    body = Body(failure=signal())
    clients, _ = serve(monkeypatch, 400, body)
    with pytest.raises(signal):
        list(OllamaEngine("http://fixture.invalid").pull_or_prepare_model("fixture"))
    assert body.closed and clients[0].is_closed


def test_successful_stream_transport_error_is_not_a_refusal(monkeypatch):
    body = Body([b'{"status":"pulling manifest"}\n'], httpx.ReadError("lost stream"))
    clients, _ = serve(monkeypatch, 200, body)
    with pytest.raises(httpx.ReadError, match="lost stream"):
        list(OllamaEngine("http://fixture.invalid").pull_or_prepare_model("fixture"))
    assert body.closed and clients[0].is_closed


@pytest.mark.parametrize("encoding", ["identity", "gzip"])
def test_successful_compressed_stream_keeps_original_progress(monkeypatch, encoding):
    payload = b'{"status":"pulling","completed":1,"total":2}\n{"status":"success"}\n'
    body = Body([gzip.compress(payload) if encoding == "gzip" else payload])
    clients, responses = serve(monkeypatch, 200, body, {"Content-Encoding": encoding})
    assert [
        event.to_dict()
        for event in OllamaEngine("http://fixture.invalid").pull_or_prepare_model("fixture")
    ] == [
        {"status": "pulling", "completed": 1, "total": 2},
        {"status": "success", "completed": None, "total": None},
    ]
    assert responses[0].request.headers["Accept-Encoding"] == "identity"
    assert body.closed and responses[0].is_closed and clients[0].is_closed


@pytest.mark.parametrize("status", [307, 400, 502])
def test_identity_diagnostics_ignore_false_length_metadata(monkeypatch, status):
    body = Body([b'{"error":"useful reason"}'])
    _, responses = serve(
        monkeypatch, status, body, {"Content-Encoding": "identity", "Content-Length": "1"}
    )
    with pytest.raises(httpx.HTTPStatusError, match="useful reason") as caught:
        list(OllamaEngine("http://fixture.invalid").pull_or_prepare_model("fixture"))
    assert caught.value.response is responses[0]
    assert caught.value.request is responses[0].request
    assert body.closed


def test_compressed_refusal_allocation_and_consumption_are_bounded_in_child():
    # A hard process deadline protects the worker even if a later change expands/decompresses
    # the body again. Allocate the fixture before tracing; never create a live engine.
    script = r"""
import gzip, tracemalloc
import httpx
from slimx.local.engines.ollama import _bounded_body
payload = gzip.compress(b"x" * (24 * 1024 * 1024))
class Body(httpx.SyncByteStream):
    read = 0
    def __iter__(self):
        self.read += len(payload)
        yield payload
body = Body()
response = httpx.Response(502, headers={"Content-Encoding":"gzip"}, stream=body)
tracemalloc.start()
try:
    assert _bounded_body(response) is None
finally:
    response.close()
peak = tracemalloc.get_traced_memory()[1]
assert body.read == 0, body.read
assert peak < 1024 * 1024, peak
print({"wire_bytes":len(payload),"consumed":body.read,"peak_bytes":peak})
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
