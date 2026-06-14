"""Metrics for the gateway process.

Prometheus sees micro-USD as a number because the ledger stores exact
integers. Dashboards can convert to dollars by dividing by 1e6, while the
application avoids floating-point money internally.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Histogram


class GatewayMetrics:
    def __init__(self, registry: CollectorRegistry | None = None):
        self.registry = registry or CollectorRegistry()
        self.requests_total = Counter(
            "gateway_requests_total",
            "Total gateway requests by status and provider.",
            ["status", "provider"],
            registry=self.registry,
        )
        self.tokens_total = Counter(
            "gateway_tokens_total",
            "Total tokens by model and direction.",
            ["model", "direction"],
            registry=self.registry,
        )
        self.cost_micro_usd_total = Counter(
            "gateway_cost_micro_usd_total",
            "Total metered cost in integer micro-USD.",
            ["model"],
            registry=self.registry,
        )
        self.fallbacks_total = Counter(
            "gateway_fallbacks_total",
            "Total requests where at least one provider failed before success.",
            ["winner"],
            registry=self.registry,
        )
        self.latency_ms = Histogram(
            "gateway_latency_ms",
            "Gateway request latency in milliseconds.",
            ["status", "provider"],
            buckets=(5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000),
            registry=self.registry,
        )

    def record_success(
        self,
        *,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_micro_usd: int,
        latency_ms: int,
        providers_tried: tuple[str, ...],
    ) -> None:
        self.requests_total.labels(status="ok", provider=provider).inc()
        self.tokens_total.labels(model=model, direction="input").inc(input_tokens)
        self.tokens_total.labels(model=model, direction="output").inc(output_tokens)
        self.cost_micro_usd_total.labels(model=model).inc(cost_micro_usd)
        self.latency_ms.labels(status="ok", provider=provider).observe(latency_ms)
        if len(providers_tried) > 1:
            self.fallbacks_total.labels(winner=provider).inc()

    def record_error(self, *, latency_ms: int) -> None:
        self.requests_total.labels(status="error", provider="none").inc()
        self.latency_ms.labels(status="error", provider="none").observe(latency_ms)
