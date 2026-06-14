# llm-gateway

A minimal multi-provider LLM gateway with per-token cost metering, an append-only usage ledger, fallback routing, Prometheus telemetry, and an AWS Bedrock provider.

> The goal is not to hide provider APIs behind a toy wrapper. The goal is to show the production plumbing every LLM platform eventually needs: routing, metering, reconciliation, fallback and observability.

## What it does

- **Multi-provider routing** with a configured fallback chain (`echo`, `bedrock`).
- **Exact cost metering** using integer micro-USD — no floating point money.
- **Append-only SQLite ledger** recording successes and failures.
- **Prometheus metrics** for request count, tokens, cost, latency and fallback events.
- **FastAPI surface**: `POST /v1/chat`, `GET /healthz`, `GET /metrics`.
- **Bedrock integration** via the Converse API, with credentials resolved by the standard AWS chain.

## Why micro-USD?

Provider prices are usually quoted as USD per million tokens. The gateway stores cost as integer micro-USD, so the math becomes hand-checkable:

```text
cost_micro_usd = tokens × price_per_million_usd
```

Example: 1,000 input tokens at $3.00/M = 3,000 µUSD = $0.003000. Across millions of requests, integer arithmetic stays exact; floats do not.

## Quickstart (offline)

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"

GATEWAY_PROVIDERS=echo \
GATEWAY_PRICING_FILE=data/pricing.example.json \
GATEWAY_LEDGER_PATH=/tmp/llm-gateway.db \
.venv/bin/uvicorn gateway.app:app --host 127.0.0.1 --port 8000
```

Request:

```bash
curl -s http://127.0.0.1:8000/v1/chat \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"hello gateway"}]}' | jq
```

Response shape:

```json
{
  "id": "...",
  "provider": "echo",
  "model": "echo-1",
  "text": "echo: hello gateway",
  "usage": {"input_tokens": 2, "output_tokens": 3},
  "cost": {"micro_usd": 1400},
  "routing": {"providers_tried": ["echo"]}
}
```

Inspect the ledger:

```bash
.venv/bin/gateway-report summary --ledger /tmp/llm-gateway.db
```

## Bedrock

```bash
export GATEWAY_PROVIDERS=bedrock,echo
export GATEWAY_BEDROCK_MODEL=<bedrock-model-id>
export GATEWAY_PRICING_FILE=data/pricing.bedrock.example.json
export GATEWAY_LEDGER_PATH=/tmp/llm-gateway-bedrock.db
# AWS credentials/region resolve through the standard AWS chain (AWS_PROFILE, AWS_REGION, instance role, etc.)

.venv/bin/uvicorn gateway.app:app --host 127.0.0.1 --port 8000
```

There is deliberately no default Bedrock model. Model choice affects cost and behavior, so it must be explicit.

## Metrics

`GET /metrics` exposes Prometheus metrics:

- `gateway_requests_total{status,provider}`
- `gateway_tokens_total{model,direction}`
- `gateway_cost_micro_usd_total{model}`
- `gateway_fallbacks_total{winner}`
- `gateway_latency_ms{status,provider}`

Dashboards can convert µUSD to USD by dividing by `1e6`; the application keeps money as integers.

## Design decisions

- **Unknown model pricing fails loud.** Serving a completion you cannot meter is a silent revenue leak.
- **Append-only ledger.** Rows are inserted, never updated or deleted. Corrections should be new rows, not history edits.
- **Failures are recorded too.** The ledger should answer operational questions like "how often did fallback fire?".
- **Echo provider for CI.** The gateway is testable offline with deterministic usage and cost.
- **Provider abstraction is tiny.** Adding OpenAI, Anthropic API, Gemini, or vLLM is one class implementing `complete(messages)`.
- **No provider profile/account names in code.** Cloud credentials resolve through standard environment mechanisms.

## Project layout

```text
src/gateway/
  app.py                 # FastAPI app: /v1/chat, /healthz, /metrics
  config.py              # env-driven settings
  factory.py             # builds router/providers/pricing/ledger
  pricing.py             # Decimal prices, integer micro-USD costs
  ledger.py              # append-only SQLite request ledger
  router.py              # fallback routing + metering
  metrics.py             # Prometheus metrics
  cli.py                 # gateway-report summary
  providers/
    base.py              # Provider protocol + Completion model
    echo.py              # deterministic offline provider
    bedrock.py           # AWS Bedrock Converse provider
```

## Roadmap

- OpenAI / Anthropic API / Gemini providers behind the same Provider protocol.
- Streaming responses.
- Tenant/account ids and prepaid credit ledger.
- Idempotency keys for request replay safety.
- Postgres ledger backend.

## License

MIT
