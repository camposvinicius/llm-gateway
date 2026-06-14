"""HTTP tests for the FastAPI gateway surface."""

import json

import pytest
from fastapi.testclient import TestClient

from gateway.app import create_app
from gateway.ledger import Ledger
from gateway.metrics import GatewayMetrics
from gateway.pricing import PricingTable
from gateway.providers import EchoProvider
from gateway.router import Router


@pytest.fixture()
def app_client(tmp_path):
    pricing_path = tmp_path / "pricing.json"
    pricing_path.write_text(
        json.dumps(
            {"models": {"echo-1": {"input_per_million": "100", "output_per_million": "200"}}}
        )
    )
    ledger = Ledger(tmp_path / "gateway.db")
    router = Router([EchoProvider(model="echo-1")], PricingTable.from_file(pricing_path), ledger)
    app = create_app(router=router, metrics=GatewayMetrics())
    client = TestClient(app)
    yield client, ledger
    ledger.close()


def test_healthz(app_client):
    client, _ = app_client
    assert client.get("/healthz").json() == {"status": "ok"}


def test_chat_returns_metered_response_and_writes_ledger(app_client):
    client, ledger = app_client
    response = client.post("/v1/chat", json={"messages": [{"role": "user", "content": "hi there"}]})

    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "echo"
    assert body["model"] == "echo-1"
    assert body["text"] == "echo: hi there"
    assert body["usage"] == {"input_tokens": 2, "output_tokens": 3}
    assert body["cost"] == {"micro_usd": 800}
    assert body["routing"] == {"providers_tried": ["echo"]}

    entry = ledger.entries()[0]
    assert entry.id == body["id"]
    assert entry.cost_micro_usd == 800


def test_chat_rejects_missing_messages(app_client):
    client, _ = app_client
    response = client.post("/v1/chat", json={})
    assert response.status_code == 400
    assert response.json()["detail"] == "messages must be a non-empty list"


def test_metrics_endpoint_contains_counters(app_client):
    client, _ = app_client
    client.post("/v1/chat", json={"messages": [{"role": "user", "content": "hi there"}]})

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    text = metrics.text
    assert 'gateway_requests_total{provider="echo",status="ok"} 1.0' in text
    assert 'gateway_tokens_total{direction="input",model="echo-1"} 2.0' in text
    assert 'gateway_cost_micro_usd_total{model="echo-1"} 800.0' in text


def test_all_providers_failed_returns_502_and_records_error(tmp_path):
    pricing_path = tmp_path / "pricing.json"
    pricing_path.write_text(
        json.dumps(
            {"models": {"echo-1": {"input_per_million": "100", "output_per_million": "200"}}}
        )
    )
    ledger = Ledger(tmp_path / "gateway.db")
    failing = EchoProvider(model="echo-1", fail=True)
    router = Router([failing], PricingTable.from_file(pricing_path), ledger)
    client = TestClient(create_app(router=router, metrics=GatewayMetrics()))

    response = client.post("/v1/chat", json={"messages": [{"role": "user", "content": "hi"}]})

    assert response.status_code == 502
    assert ledger.failure_count() == 1
    assert 'gateway_requests_total{provider="none",status="error"} 1.0' in client.get(
        "/metrics"
    ).text
    ledger.close()
