"""Pricing: load a per-model price table and compute request costs.

Two deliberate choices, both of which matter in billing systems:

1. **Money is never a float.** Prices are parsed as ``Decimal`` from strings
   and costs are stored as **integer micro-USD** (1 USD = 1,000,000 µUSD).
   Floats accumulate rounding error across millions of requests; integers
   add up exactly.

2. **There is a pleasant identity that makes the math hand-checkable:**
   prices are quoted in USD per million tokens, so

       cost_in_micro_usd = tokens × price_per_million_usd

   The "per million" and the "micro" cancel out. 1,000 tokens at $3.00/M
   is exactly 3,000 µUSD ($0.003) — you can verify any ledger row mentally.

The pricing file is JSON, loaded from a path given by the environment:

    {"models": {"model-id": {"input_per_million": "3.00",
                             "output_per_million": "15.00"}}}

Prices are strings on purpose: JSON numbers are floats, and we just said
money is never a float.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path


class PricingError(ValueError):
    """Raised when the pricing table is missing, malformed, or incomplete."""


@dataclass(frozen=True)
class ModelPrice:
    input_per_million: Decimal
    output_per_million: Decimal


class PricingTable:
    def __init__(self, prices: dict[str, ModelPrice]):
        if not prices:
            raise PricingError("Pricing table has no models")
        self._prices = prices

    @classmethod
    def from_file(cls, path: Path) -> PricingTable:
        if not path.exists():
            raise PricingError(f"Pricing file not found: {path}")
        try:
            doc = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise PricingError(f"{path}: invalid JSON ({exc})") from exc

        models = doc.get("models")
        if not isinstance(models, dict) or not models:
            raise PricingError(f'{path}: expected a non-empty "models" object')

        prices = {}
        for model_id, entry in models.items():
            try:
                prices[model_id] = ModelPrice(
                    input_per_million=Decimal(entry["input_per_million"]),
                    output_per_million=Decimal(entry["output_per_million"]),
                )
            except KeyError as exc:
                raise PricingError(f"{path}: model {model_id!r} missing field {exc}") from exc
            except InvalidOperation as exc:
                raise PricingError(
                    f"{path}: model {model_id!r} has a non-numeric price"
                ) from exc
            if (
                prices[model_id].input_per_million < 0
                or prices[model_id].output_per_million < 0
            ):
                raise PricingError(f"{path}: model {model_id!r} has a negative price")
        return cls(prices)

    def cost_micro_usd(self, model: str, input_tokens: int, output_tokens: int) -> int:
        """Cost of a request in integer micro-USD.

        A gateway that cannot price a request must fail loudly: unknown
        models raise instead of metering silently at $0 (that is how
        revenue leaks).
        """
        if model not in self._prices:
            known = ", ".join(sorted(self._prices))
            raise PricingError(f"No pricing for model {model!r}. Priced models: {known}")
        if input_tokens < 0 or output_tokens < 0:
            raise PricingError("Token counts cannot be negative")

        price = self._prices[model]
        cost = (
            Decimal(input_tokens) * price.input_per_million
            + Decimal(output_tokens) * price.output_per_million
        )
        return int(cost.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def format_usd(micro_usd: int) -> str:
    """Render integer micro-USD as a dollar string: 3000 -> '$0.003000'."""
    return f"${Decimal(micro_usd) / Decimal(1_000_000):.6f}"
