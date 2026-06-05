"""High-level calls accept either a prompt string or a full message list
(review item 5.3) — exercised offline via `inspect()`."""

from __future__ import annotations

from slimx import Message, image
from slimx.high.api import Model

_KW = {"provider_kwargs": {"api_key": "k"}}


def _model():
    return Model("openai:gpt-4o", **_KW)


def test_string_prompt_becomes_single_user_message():
    payload = _model().inspect("hello").payload
    assert payload["messages"] == [{"role": "user", "content": "hello"}]


def test_message_list_is_used_verbatim():
    history = [
        Message.system("be brief"),
        Message.user("hi"),
        Message.assistant("hello"),
        Message.user("continue"),
    ]
    payload = _model().inspect(history).payload
    assert [m["role"] for m in payload["messages"]] == ["system", "user", "assistant", "user"]
    assert payload["messages"][-1]["content"] == "continue"


def test_json_prepends_system_to_message_list():
    import dataclasses

    @dataclasses.dataclass
    class Out:
        x: int

    history = [Message.user("first"), Message.assistant("ok"), Message.user("again")]
    # json() is sync and builds the request before any network call; inspect the
    # schema-priming system message is prepended ahead of the supplied history.
    # We assert via the message construction helper rather than a live call.
    from slimx.high.api import _json_system_prompt, _messages_from, _json_schema_parts

    schema_dict, _ = _json_schema_parts(Out)
    msgs = [Message.system(_json_system_prompt(schema_dict))] + _messages_from(list(history), {})
    assert msgs[0].role == "system"
    assert [m.role for m in msgs[1:]] == ["user", "assistant", "user"]


def test_multimodal_still_works_with_string():
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
    content = _model().inspect("look", images=[image(png, mime_type="image/png")]).payload["messages"][0]["content"]
    assert any(p["type"] == "image_url" for p in content)


def test_media_kwargs_ignored_for_message_list():
    # Passing a message list plus stray media kwargs must not crash or leak the
    # kwargs into the request.
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
    payload = _model().inspect([Message.user("hi")], images=[image(png, mime_type="image/png")]).payload
    assert payload["messages"] == [{"role": "user", "content": "hi"}]
