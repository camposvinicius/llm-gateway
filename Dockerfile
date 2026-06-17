# Gateway image for the observability demo.
#
# It runs in offline `echo` mode by default, so the whole docker-compose stack
# comes up with zero credentials. Override the GATEWAY_* env vars (and pass real
# API keys) to serve the real providers.
FROM python:3.12-slim

WORKDIR /app

# Metadata + source needed for the wheel build. hatchling reads README.md for
# package metadata, so it must be present at build time.
COPY pyproject.toml README.md ./
COPY src ./src
COPY data ./data

RUN pip install --no-cache-dir .

# Offline defaults: no network, no credentials, deterministic cost/tokens.
ENV GATEWAY_PROVIDERS=echo \
    GATEWAY_PRICING_FILE=data/pricing.example.json \
    GATEWAY_LEDGER_PATH=/tmp/llm-gateway.db

EXPOSE 8080

CMD ["uvicorn", "gateway.app:app", "--host", "0.0.0.0", "--port", "8080"]
