from __future__ import annotations

import time

import pytest

from slimx import parallel
from slimx.errors import ProviderError
from slimx.providers import register
from slimx.providers.base import Provider, ProviderCapabilities
from slimx.types import Result, StreamEvent, Usage


class _PTestProvider(Provider):
    name = "ptest"
    capabilities = ProviderCapabilities()

    def chat(self, req, *, tools=(), timeout=None):
        if req.model == "boom":
            raise ProviderError("synthetic failure")
        if req.model.startswith("slow"):
            time.sleep(0.3)
        return Result(text=f"answer:{req.model}", usage=Usage())

    def stream(self, req, *, tools=(), timeout=None):
        yield StreamEvent.done()


@pytest.fixture(autouse=True, scope="module")
def _register_ptest():
    register("ptest", lambda **kw: _PTestProvider())
    yield


def test_all_mode_returns_every_result_in_order():
    res = parallel(["ptest:a", "ptest:b", "ptest:c"])("hi")
    assert res.text is None
    assert [it.model for it in res.results] == ["ptest:a", "ptest:b", "ptest:c"]
    assert all(it.ok for it in res.results)
    assert [it.result.text for it in res.results if it.result] == [
        "answer:a",
        "answer:b",
        "answer:c",
    ]
    assert res.errors == []
    assert res.trace["mode"] == "all"
    assert res.trace["ok_count"] == 3
    assert res.trace["error_count"] == 0
    assert res.trace["model_count"] == 3


def test_all_mode_captures_errors_without_raising():
    res = parallel(["ptest:a", "ptest:boom"])("hi")
    assert len(res.results) == 2
    assert len(res.errors) == 1
    boom = res.errors[0]
    assert boom.model == "ptest:boom"
    assert boom.ok is False
    assert "ProviderError" in (boom.error or "")
    assert res.trace["ok_count"] == 1
    assert res.trace["error_count"] == 1


def test_race_returns_first_success():
    res = parallel(["ptest:slow_one", "ptest:fast"], mode="race")("hi")
    assert res.winner is not None
    assert res.winner.model == "ptest:fast"
    assert res.text == "answer:fast"
    assert res.trace["mode"] == "race"


def test_race_returns_promptly_without_waiting_for_stragglers():
    start = time.perf_counter()
    res = parallel(["ptest:fast", "ptest:slow_one"], mode="race")("hi")
    elapsed = time.perf_counter() - start
    assert res.winner is not None and res.winner.model == "ptest:fast"
    # The 0.3s straggler must not gate the return.
    assert elapsed < 0.25


def test_race_all_fail_returns_no_winner():
    res = parallel(["ptest:boom", "ptest:boom"], mode="race")("hi")
    assert res.winner is None
    assert res.text is None
    assert len(res.errors) == 2


def test_compare_mode_builds_side_by_side_text():
    res = parallel(["ptest:a", "ptest:b"], mode="compare")("hi")
    assert res.trace["mode"] == "compare"
    assert res.text is not None
    assert "### ptest:a" in res.text and "answer:a" in res.text
    assert "### ptest:b" in res.text and "answer:b" in res.text
    assert len(res.results) == 2


def test_judge_mode_uses_judge_model_and_keeps_candidates():
    res = parallel(["ptest:a", "ptest:b"], mode="judge", judge="ptest:judgeX")("hi")
    assert res.trace["mode"] == "judge"
    assert res.trace["judge"] == "ptest:judgeX"
    # candidates (the fanned-out answers) are preserved
    assert [it.model for it in res.candidates] == ["ptest:a", "ptest:b"]
    # winner is the judge's own answer
    assert res.winner is not None and res.winner.model == "ptest:judgeX"
    assert res.text == "answer:judgeX"


def test_judge_mode_requires_judge():
    with pytest.raises(ValueError):
        parallel(["ptest:a"], mode="judge")


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        parallel(["ptest:a"], mode="bogus")


def test_empty_models_raises():
    with pytest.raises(ValueError):
        parallel([])


def test_parallel_is_exposed_at_top_level():
    import slimx

    assert hasattr(slimx, "parallel")
    assert hasattr(slimx, "ParallelResult")
    assert hasattr(slimx, "ParallelItem")


def test_cancel_event_preset_skips_every_call():
    import threading

    evt = threading.Event()
    evt.set()
    res = parallel(["ptest:a", "ptest:b"], cancel_event=evt)("hi")
    assert [it.cancelled for it in res.results] == [True, True]
    assert all(not it.ok for it in res.results)
    assert all("Cancelled" in (it.error or "") for it in res.results)


def test_cancel_event_set_midway_stops_new_calls_but_keeps_finished_results():
    import threading

    evt = threading.Event()
    # One worker: the first (slow) call runs to completion; the event set during it means
    # the second call never starts.
    p = parallel(["ptest:slow-1", "ptest:late"], max_workers=1)

    def set_soon():
        time.sleep(0.05)
        evt.set()

    t = __import__("threading").Thread(target=set_soon)
    t.start()
    res = p("hi", cancel_event=evt)
    t.join()
    assert res.results[0].ok  # finished work is kept
    assert res.results[1].cancelled is True


def test_cancel_event_skips_judge_synthesis():
    import threading

    evt = threading.Event()

    p = parallel(["ptest:a"], mode="judge", judge="ptest:judge")
    evt.set()
    res = p("hi", cancel_event=evt)
    # Candidates were never started either (event pre-set) and no judge winner exists.
    assert res.winner is None
    assert res.trace.get("judge_cancelled") is True


def test_cancel_event_kwarg_never_leaks_into_model_calls():
    import threading

    evt = threading.Event()  # NOT set — calls proceed; kwarg must be stripped
    res = parallel(["ptest:a"])("hi", cancel_event=evt)
    item = res.results[0]
    assert item.ok and item.result is not None
    assert item.result.text == "answer:a"
