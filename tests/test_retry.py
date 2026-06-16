"""Tests for the retry policy and call_with_retries loop."""

from __future__ import annotations

import pytest

from gateway.retry import RetryPolicy, call_with_retries


class _RecordingSleep:
    def __init__(self):
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


def test_returns_first_success_without_sleeping():
    sleeper = _RecordingSleep()
    policy = RetryPolicy(max_attempts=3, base_delay_seconds=0.1, sleep=sleeper)
    calls = {"n": 0}

    def op() -> str:
        calls["n"] += 1
        return "ok"

    assert call_with_retries(op, is_retryable=lambda _: True, policy=policy) == "ok"
    assert calls["n"] == 1
    assert sleeper.calls == []


def test_retries_until_success():
    sleeper = _RecordingSleep()
    policy = RetryPolicy(max_attempts=4, base_delay_seconds=0.1, sleep=sleeper)
    state = {"n": 0}

    def op() -> str:
        state["n"] += 1
        if state["n"] < 3:
            raise RuntimeError("transient")
        return "ok"

    assert call_with_retries(op, is_retryable=lambda _: True, policy=policy) == "ok"
    assert state["n"] == 3
    # Two sleeps before the third (successful) attempt.
    assert len(sleeper.calls) == 2


def test_non_retryable_short_circuits():
    sleeper = _RecordingSleep()
    policy = RetryPolicy(max_attempts=5, base_delay_seconds=0.1, sleep=sleeper)
    state = {"n": 0}

    class CallerBug(Exception):
        pass

    def op() -> str:
        state["n"] += 1
        raise CallerBug("bad input")

    with pytest.raises(CallerBug):
        call_with_retries(op, is_retryable=lambda exc: False, policy=policy)

    assert state["n"] == 1
    assert sleeper.calls == []


def test_exhausting_attempts_re_raises_last_exception():
    sleeper = _RecordingSleep()
    policy = RetryPolicy(max_attempts=3, base_delay_seconds=0.1, sleep=sleeper)
    state = {"n": 0}

    def op() -> str:
        state["n"] += 1
        raise RuntimeError(f"attempt {state['n']}")

    with pytest.raises(RuntimeError, match="attempt 3"):
        call_with_retries(op, is_retryable=lambda _: True, policy=policy)

    assert state["n"] == 3


def test_backoff_is_bounded_by_max_delay():
    policy = RetryPolicy(max_attempts=10, base_delay_seconds=1.0, max_delay_seconds=2.0)
    # Even with full jitter, no sample can exceed max_delay_seconds.
    for attempt in range(1, 11):
        for _ in range(50):
            assert 0.0 <= policy.backoff(attempt) <= 2.0


def test_invalid_max_attempts_raises():
    policy = RetryPolicy(max_attempts=0)
    with pytest.raises(ValueError, match="max_attempts"):
        call_with_retries(lambda: "x", is_retryable=lambda _: True, policy=policy)
