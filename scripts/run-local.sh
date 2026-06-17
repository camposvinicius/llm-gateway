#!/usr/bin/env bash
# Launch the gateway + chat UI locally with the premium models the UI advertises.
#
# Secrets come from a repo-local .env (gitignored) that you create from
# .env.example — no key is ever baked into this script or committed.
#
#   cp .env.example .env   # then fill OPENAI_API_KEY / GOOGLE_API_KEY / etc.
#   ./scripts/run-local.sh
#
# Override anything via the environment, e.g.:
#   AWS_PROFILE=my-profile GATEWAY_PORT=8080 ./scripts/run-local.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# --- secrets / overrides from a gitignored .env ---
if [[ -f .env ]]; then
  set -a; source .env; set +a
else
  echo "warning: no .env found — copy .env.example to .env and add your keys" >&2
fi

# --- premium models the UI dropdown promises (override via env if you like) ---
export GATEWAY_PROVIDERS="${GATEWAY_PROVIDERS:-openai,gemini,bedrock}"
export GATEWAY_BEDROCK_MODEL="${GATEWAY_BEDROCK_MODEL:-global.anthropic.claude-opus-4-8}"
export GATEWAY_OPENAI_MODEL="${GATEWAY_OPENAI_MODEL:-gpt-5.5}"
export GATEWAY_GEMINI_MODEL="${GATEWAY_GEMINI_MODEL:-gemini-3.1-pro-preview}"
export GATEWAY_PRICING_FILE="${GATEWAY_PRICING_FILE:-data/pricing.example.json}"
export GATEWAY_LEDGER_PATH="${GATEWAY_LEDGER_PATH:-/tmp/llm-gateway.db}"

GATEWAY_HOST="${GATEWAY_HOST:-127.0.0.1}"
GATEWAY_PORT="${GATEWAY_PORT:-8080}"
UI_PORT="${UI_PORT:-3000}"

PIDS=()
cleanup() { for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done; }
trap cleanup EXIT INT TERM

echo "→ starting gateway on http://${GATEWAY_HOST}:${GATEWAY_PORT}"
.venv/bin/uvicorn gateway.app:app --host "$GATEWAY_HOST" --port "$GATEWAY_PORT" &
PIDS+=("$!")

# wait for the gateway to answer before bringing the UI up
for _ in $(seq 1 30); do
  if curl -fsS "http://${GATEWAY_HOST}:${GATEWAY_PORT}/healthz" >/dev/null 2>&1; then break; fi
  sleep 0.5
done

echo "→ starting chat UI on http://127.0.0.1:${UI_PORT}"
(
  cd examples/chat_ui
  GATEWAY_URL="http://${GATEWAY_HOST}:${GATEWAY_PORT}" npm run dev -- -p "$UI_PORT"
) &
PIDS+=("$!")

echo "→ gateway + UI up. Ctrl-C to stop both."
wait
