"""Hand-checkable tests for the pricing core.

The key identity under test: prices are USD per million tokens and costs
are integer micro-USD, so cost_micro_usd = tokens × price_per_million.
1,000 tokens at $3.00/M = exactly 3,000 µUSD.
"""

import json

import pytest

from gateway.pricing import PricingError, PricingTable, format_usd


@pytest.fixture()
def table(tmp_path):
    path = tmp_path / "pricing.json"
    path.write_text(
        json.dumps(
            {
                "models": {
                    "small": {"input_per_million": "0.25", "output_per_million": "1.25"},
                    "large": {"input_per_million": "3.00", "output_per_million": "15.00"},
                }
            }
        )
    )
    return PricingTable.from_file(path)


class TestCost:
    def test_the_identity_tokens_times_price(self, table):
        # 1000 in @ $3/M = 3000 µUSD; 100 out @ $15/M = 1500 µUSD
        assert table.cost_micro_usd("large", 1000, 100) == 4500

    def test_zero_tokens_cost_zero(self, table):
        assert table.cost_micro_usd("small", 0, 0) == 0

    def test_fraction_rounds_half_up(self, table):
        # 1 token @ $0.25/M = 0.25 µUSD -> rounds to 0; 2 tokens = 0.5 -> 1
        assert table.cost_micro_usd("small", 1, 0) == 0
        assert table.cost_micro_usd("small", 2, 0) == 1

    def test_costs_add_exactly_over_many_requests(self, table):
        # one million requests of 1000+100 tokens: integer math never drifts
        one = table.cost_micro_usd("large", 1000, 100)
        assert one * 1_000_000 == 4_500_000_000  # $4,500.00 exactly

    def test_unknown_model_refuses_to_price(self, table):
        with pytest.raises(PricingError, match="No pricing for model 'gpt-x'"):
            table.cost_micro_usd("gpt-x", 10, 10)

    def test_negative_tokens_rejected(self, table):
        with pytest.raises(PricingError, match="cannot be negative"):
            table.cost_micro_usd("small", -1, 0)


class TestLoading:
    def test_missing_file(self, tmp_path):
        with pytest.raises(PricingError, match="not found"):
            PricingTable.from_file(tmp_path / "nope.json")

    def test_invalid_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not json")
        with pytest.raises(PricingError, match="invalid JSON"):
            PricingTable.from_file(path)

    def test_missing_models_key(self, tmp_path):
        path = tmp_path / "empty.json"
        path.write_text("{}")
        with pytest.raises(PricingError, match='non-empty "models"'):
            PricingTable.from_file(path)

    def test_non_numeric_price(self, tmp_path):
        path = tmp_path / "nan.json"
        path.write_text(
            '{"models": {"m": {"input_per_million": "abc", "output_per_million": "1"}}}'
        )
        with pytest.raises(PricingError, match="non-numeric price"):
            PricingTable.from_file(path)

    def test_negative_price(self, tmp_path):
        path = tmp_path / "neg.json"
        path.write_text(
            '{"models": {"m": {"input_per_million": "-1", "output_per_million": "1"}}}'
        )
        with pytest.raises(PricingError, match="negative price"):
            PricingTable.from_file(path)


class TestFormat:
    def test_format_usd(self):
        assert format_usd(3000) == "$0.003000"
        assert format_usd(4_500_000_000) == "$4500.000000"
        assert format_usd(0) == "$0.000000"
