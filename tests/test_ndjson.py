from __future__ import annotations

from slimx.utils.ndjson import iter_ndjson


def _bytes(*lines):
    for line in lines:
        yield line


def test_iter_ndjson_parses_lines():
    objs = list(_iter(b'{"a":1}\n{"b":2}\n'))
    assert objs == [{"a": 1}, {"b": 2}]


def test_iter_ndjson_skips_malformed_lines():
    # A garbage frame in the middle must not abort the stream.
    objs = list(_iter(b'{"a":1}\nnot json\n{"b":2}\n'))
    assert objs == [{"a": 1}, {"b": 2}]


def test_iter_ndjson_handles_split_chunks_and_tail():
    objs = list(_iter(b'{"a":', b'1}\n{"b":2}'))
    assert objs == [{"a": 1}, {"b": 2}]


def _iter(*chunks):
    return iter_ndjson(_bytes(*chunks))
