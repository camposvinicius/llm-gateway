"""Tests for the ledger (in-memory SQLite via tmp file)."""

import pytest

from gateway.ledger import Ledger


@pytest.fixture()
def ledger(tmp_path):
    ledger = Ledger(tmp_path / "test.db")
    yield ledger
    ledger.close()


def test_success_roundtrip(ledger):
    request_id = ledger.record_success(
        provider="echo",
        model="echo-1",
        input_tokens=10,
        output_tokens=5,
        cost_micro_usd=42,
        latency_ms=7,
        providers_tried=["echo"],
    )
    entries = ledger.entries()
    assert len(entries) == 1
    entry = entries[0]
    assert entry.id == request_id
    assert entry.status == "ok"
    assert entry.cost_micro_usd == 42
    assert entry.providers_tried == "echo"


def test_failure_roundtrip(ledger):
    ledger.record_failure(
        latency_ms=3, providers_tried=["bedrock", "echo"], error="all failed"
    )
    entry = ledger.entries()[0]
    assert entry.status == "error"
    assert entry.cost_micro_usd is None
    assert entry.providers_tried == "bedrock,echo"
    assert ledger.failure_count() == 1


def test_summary_by_model_sums_exactly(ledger):
    for _ in range(3):
        ledger.record_success(
            provider="echo",
            model="echo-1",
            input_tokens=1000,
            output_tokens=100,
            cost_micro_usd=4500,
            latency_ms=10,
            providers_tried=["echo"],
        )
    ledger.record_success(
        provider="bedrock",
        model="big-model",
        input_tokens=10,
        output_tokens=10,
        cost_micro_usd=99999,
        latency_ms=200,
        providers_tried=["bedrock"],
    )

    summary = ledger.summary_by_model()
    # ordered by total cost, descending
    assert [s["model"] for s in summary] == ["big-model", "echo-1"]
    echo = summary[1]
    assert echo["requests"] == 3
    assert echo["cost_micro_usd"] == 13500  # 3 × 4500, exact
    assert echo["input_tokens"] == 3000


def test_failures_do_not_pollute_summary(ledger):
    ledger.record_failure(latency_ms=1, providers_tried=["echo"], error="x")
    assert ledger.summary_by_model() == []
