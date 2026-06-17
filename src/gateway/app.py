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
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import generate_latest
from starlette.responses import Response

from .agent import DEFAULT_MAX_STEPS, run_agent
from .config import ConfigError, Settings
from .factory import build_router
from .metrics import GatewayMetrics
from .pricing import PricingError
from .providers import ProviderError
from .router import AllProvidersFailedError, RoutedResponse, Router
from .tools import Tool, build_default_tools


def create_app(
    router: Router | None = None,
    metrics: GatewayMetrics | None = None,
    tools: dict[str, Tool] | None = None,
) -> FastAPI:
    app = FastAPI(title="llm-gateway", version="0.1.0")
    app.state.metrics = metrics or GatewayMetrics()
    # Tools for the research agent (/v1/agent); registry reflects which keys are set.
    app.state.tools = tools if tools is not None else build_default_tools()

    # CORS — allow the bundled chat UI to talk to the gateway in dev.
    # Configurable via GATEWAY_CORS_ORIGINS (comma-separated) for production.
    cors_env = os.environ.get(
        "GATEWAY_CORS_ORIGINS",
        "http://127.0.0.1:3000,http://localhost:3000",
    )
    allow_origins = [origin.strip() for origin in cors_env.split(",") if origin.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

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

        provider_chain = body.get("provider_chain")
        if provider_chain is not None:
            if not isinstance(provider_chain, list) or not all(
                isinstance(provider, str) for provider in provider_chain
            ):
                raise HTTPException(
                    status_code=400,
                    detail="provider_chain must be a list of provider names",
                )

        tools = body.get("tools")
        if tools is not None and not isinstance(tools, list):
            raise HTTPException(status_code=400, detail="tools must be a list")

        try:
            # router.complete does blocking HTTP; run it off the event loop so
            # /healthz and other requests stay responsive during a call.
            response = await run_in_threadpool(
                app.state.router.complete, messages, provider_chain=provider_chain, tools=tools
            )
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

        completion = response.completion
        return {
            "id": response.request_id,
            "provider": response.provider,
            "model": completion.model,
            "text": completion.text,
            "stop_reason": completion.stop_reason,
            "tool_calls": [
                {
                    "id": call.id,
                    "name": call.name,
                    "input": call.input,
                    # Opaque token some providers (Gemini) require echoed back.
                    **({"signature": call.signature} if call.signature else {}),
                }
                for call in completion.tool_calls
            ],
            "usage": {
                "input_tokens": completion.input_tokens,
                "output_tokens": completion.output_tokens,
            },
            "cost": {"micro_usd": response.cost_micro_usd},
            "routing": {"providers_tried": list(response.providers_tried)},
        }

    @app.post("/v1/agent")
    async def agent(request: Request) -> dict:
        try:
            body = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="invalid JSON body") from exc

        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(status_code=400, detail="messages must be a non-empty list")

        provider_chain = body.get("provider_chain")
        if provider_chain is not None and (
            not isinstance(provider_chain, list)
            or not all(isinstance(p, str) for p in provider_chain)
        ):
            raise HTTPException(
                status_code=400, detail="provider_chain must be a list of provider names"
            )

        max_steps = body.get("max_steps", DEFAULT_MAX_STEPS)
        if not isinstance(max_steps, int) or max_steps < 1:
            raise HTTPException(status_code=400, detail="max_steps must be a positive integer")
        max_steps = min(max_steps, 12)  # bound the loop regardless of the request

        def record(routed: RoutedResponse) -> None:
            completion = routed.completion
            app.state.metrics.record_success(
                provider=routed.provider,
                model=completion.model,
                input_tokens=completion.input_tokens,
                output_tokens=completion.output_tokens,
                cost_micro_usd=routed.cost_micro_usd,
                latency_ms=routed.latency_ms,
                providers_tried=routed.providers_tried,
            )

        try:
            # The whole tool-use loop is blocking I/O; run it in a worker thread
            # so a long research run doesn't block the event loop.
            result = await run_in_threadpool(
                run_agent,
                app.state.router,
                messages,
                app.state.tools,
                provider_chain=provider_chain,
                max_steps=max_steps,
                on_model_response=record,
            )
        except AllProvidersFailedError as exc:
            app.state.metrics.record_error(latency_ms=0)
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except (ProviderError, PricingError, ValueError) as exc:
            app.state.metrics.record_error(latency_ms=0)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return {
            "id": result.request_ids[-1] if result.request_ids else None,
            "provider": result.provider,
            "model": result.model,
            "text": result.text,
            "steps": [
                {"tool": step.tool, "arguments": step.arguments, "result": step.result}
                for step in result.steps
            ],
            "usage": {
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
            },
            "cost": {"micro_usd": result.cost_micro_usd},
            "routing": {"providers_tried": list(result.providers_tried)},
            "tools_available": sorted(app.state.tools),
            "stopped_at_max_steps": result.stopped_at_max_steps,
        }

    return app


app = create_app()
