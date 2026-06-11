"""Tests for the router: fallback order, metering, and failure recording."""

import json

import pytest

from gateway.ledger import Ledger
from gateway.pricing import PricingTable
from gateway.providers import EchoProvider
from gateway.router import AllProvidersFailedError, Router


@pytest.fixture()
def pricing(tmp_path):
    path = tmp_path / "pricing.json"
    path.write_text(
        json.dumps(
            {"models": {"echo-1": {"input_per_million": "100", "output_per_million": "200"}}}
        )
    )
    return PricingTable.from_file(path)


@pytest.fixture()
def ledger(tmp_path):
    ledger = Ledger(tmp_path / "router.db")
    yield ledger
    ledger.close()


MESSAGES = [{"role": "user", "content": "two words"}]  # 2 in, 3 out ("echo: two words")


def test_first_provider_wins(pricing, ledger):
    router = Router([EchoProvider(model="echo-1")], pricing, ledger)
    response = router.complete(MESSAGES)

    assert response.provider == "echo"
    assert response.providers_tried == ("echo",)
    # 2 tokens × 100/M = 200µ; 3 tokens × 200/M = 600µ -> 800µUSD... per-million scale:
    # 2 × 100 = 200, 3 × 200 = 600 -> 800
    assert response.cost_micro_usd == 800
    assert ledger.entries()[0].status == "ok"


def test_fallback_fires_on_provider_error(pricing, ledger):
    failing = EchoProvider(model="echo-1", fail=True)
    failing.name = "echo-primary"
    healthy = EchoProvider(model="echo-1")
    healthy.name = "echo-fallback"

    router = Router([failing, healthy], pricing, ledger)
    response = router.complete(MESSAGES)

    assert response.provider == "echo-fallback"
    assert response.providers_tried == ("echo-primary", "echo-fallback")
    # the ledger remembers the whole chain, not just the winner
    assert ledger.entries()[0].providers_tried == "echo-primary,echo-fallback"


def test_all_providers_failing_is_recorded_and_raised(pricing, ledger):
    p1 = EchoProvider(model="echo-1", fail=True)
    p1.name = "a"
    p2 = EchoProvider(model="echo-1", fail=True)
    p2.name = "b"

    router = Router([p1, p2], pricing, ledger)
    with pytest.raises(AllProvidersFailedError) as excinfo:
        router.complete(MESSAGES)

    assert set(excinfo.value.attempts) == {"a", "b"}
    entry = ledger.entries()[0]
    assert entry.status == "error"
    assert entry.providers_tried == "a,b"


def test_unmeterable_completion_is_not_served(tmp_path, ledger):
    # pricing table that does NOT know the echo model
    path = tmp_path / "pricing.json"
    path.write_text(
        json.dumps({"models": {"other": {"input_per_million": "1", "output_per_million": "1"}}})
    )
    pricing = PricingTable.from_file(path)
    router = Router([EchoProvider(model="echo-1")], pricing, ledger)

    from gateway.pricing import PricingError

    with pytest.raises(PricingError, match="No pricing for model"):
        router.complete(MESSAGES)


def test_empty_provider_list_rejected(pricing, ledger):
    with pytest.raises(ValueError, match="at least one provider"):
        Router([], pricing, ledger)
