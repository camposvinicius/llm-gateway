# llm-gateway

A minimal multi-provider LLM gateway: per-token cost metering, a usage
ledger, fallback routing, and request telemetry.

> Status: under active development. Current phase: cost metering core.

## Development

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

## License

MIT
