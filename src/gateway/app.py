"""FastAPI application for the gateway.

Public surface is intentionally small:

- POST /v1/chat     — minimal chat completion endpoint
- GET  /healthz     — cheap liveness probe
- GET  /metrics     — Prometheus metrics

The response includes metering details (tokens and micro-USD cost) because
this is a gateway, not just a proxy: every caller can see what they spent.
"""

from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException, Request
from prometheus_client import generate_latest
from starlette.responses import Response

from .config import ConfigError, Settings
from .factory import build_router
from .metrics import GatewayMetrics
from .pricing import PricingError
from .providers import ProviderError
from .router import AllProvidersFailedError, Router


def create_app(router: Router | None = None, metrics: GatewayMetrics | None = None) -> FastAPI:
    app = FastAPI(title="llm-gateway", version="0.1.0")
    app.state.metrics = metrics or GatewayMetrics()

    if router is None:
        try:
            router = build_router(Settings.from_env())
        except (ConfigError, PricingError) as exc:
            # The app should still import cleanly in tests, but fail fast at
            # startup when misconfigured in production.
            raise RuntimeError(str(exc)) from exc
    app.state.router = router

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/metrics")
    def metrics_endpoint() -> Response:
        data = generate_latest(app.state.metrics.registry)
        return Response(content=data, media_type="text/plain; version=0.0.4")

    @app.post("/v1/chat")
    async def chat(request: Request) -> dict:
        try:
            body = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="invalid JSON body") from exc

        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(status_code=400, detail="messages must be a non-empty list")

        try:
            response = app.state.router.complete(messages)
        except AllProvidersFailedError as exc:
            app.state.metrics.record_error(latency_ms=0)
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except (ProviderError, PricingError, ValueError) as exc:
            app.state.metrics.record_error(latency_ms=0)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        app.state.metrics.record_success(
            provider=response.provider,
            model=response.completion.model,
            input_tokens=response.completion.input_tokens,
            output_tokens=response.completion.output_tokens,
            cost_micro_usd=response.cost_micro_usd,
            latency_ms=response.latency_ms,
            providers_tried=response.providers_tried,
        )

        return {
            "id": response.request_id,
            "provider": response.provider,
            "model": response.completion.model,
            "text": response.completion.text,
            "usage": {
                "input_tokens": response.completion.input_tokens,
                "output_tokens": response.completion.output_tokens,
            },
            "cost": {"micro_usd": response.cost_micro_usd},
            "routing": {"providers_tried": list(response.providers_tried)},
        }

    return app


app = create_app()
