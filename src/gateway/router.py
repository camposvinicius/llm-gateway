"""Router: try providers in order, meter what succeeded, record everything.

The contract:

1. Providers are tried **in the configured order** (the fallback chain).
2. The first success wins; its usage is priced and written to the ledger
   together with the full list of providers tried.
3. If every provider fails, the request fails — and that failure is also
   written to the ledger. A gateway that only remembers its successes
   cannot answer "how often did fallback fire last week?".
4. Pricing errors are **not** swallowed: serving a completion you cannot
   meter is worse than failing the request (silent revenue leak).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .ledger import Ledger
from .pricing import PricingTable
from .providers import Completion, Provider, ProviderError


class AllProvidersFailedError(RuntimeError):
    def __init__(self, attempts: dict[str, str]):
        self.attempts = attempts
        detail = "; ".join(f"{name}: {err}" for name, err in attempts.items())
        super().__init__(f"All providers failed ({detail})")


@dataclass(frozen=True)
class RoutedResponse:
    request_id: str
    completion: Completion
    provider: str
    providers_tried: tuple[str, ...]
    cost_micro_usd: int
    latency_ms: int


class Router:
    def __init__(self, providers: list[Provider], pricing: PricingTable, ledger: Ledger):
        if not providers:
            raise ValueError("Router needs at least one provider")
        self._providers = providers
        self._pricing = pricing
        self._ledger = ledger

    def complete(self, messages: list[dict]) -> RoutedResponse:
        started = time.monotonic()
        tried: list[str] = []
        errors: dict[str, str] = {}

        for provider in self._providers:
            tried.append(provider.name)
            try:
                completion = provider.complete(messages)
            except ProviderError as exc:
                errors[provider.name] = str(exc)
                continue

            cost = self._pricing.cost_micro_usd(
                completion.model, completion.input_tokens, completion.output_tokens
            )
            latency_ms = int((time.monotonic() - started) * 1000)
            request_id = self._ledger.record_success(
                provider=provider.name,
                model=completion.model,
                input_tokens=completion.input_tokens,
                output_tokens=completion.output_tokens,
                cost_micro_usd=cost,
                latency_ms=latency_ms,
                providers_tried=tried,
            )
            return RoutedResponse(
                request_id=request_id,
                completion=completion,
                provider=provider.name,
                providers_tried=tuple(tried),
                cost_micro_usd=cost,
                latency_ms=latency_ms,
            )

        latency_ms = int((time.monotonic() - started) * 1000)
        error = AllProvidersFailedError(errors)
        self._ledger.record_failure(
            latency_ms=latency_ms, providers_tried=tried, error=str(error)
        )
        raise error
