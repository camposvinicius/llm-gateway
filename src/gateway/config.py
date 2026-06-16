"""Environment-driven application settings.

The app factory reads these once at startup. Provider choices, model ids,
pricing path and ledger path all come from environment variables — no
runtime decision that affects cost is hardcoded.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


class ConfigError(ValueError):
    """Raised when environment configuration is invalid."""


@dataclass(frozen=True)
class Settings:
    providers: tuple[str, ...]
    echo_model: str
    bedrock_model: str | None
    openai_model: str | None
    gemini_model: str | None
    pricing_file: Path
    ledger_path: Path

    @classmethod
    def from_env(cls) -> Settings:
        providers_raw = os.environ.get("GATEWAY_PROVIDERS", "echo")
        providers = tuple(p.strip().lower() for p in providers_raw.split(",") if p.strip())
        if not providers:
            raise ConfigError("GATEWAY_PROVIDERS must include at least one provider")
        supported = {"echo", "bedrock", "openai", "gemini"}
        unsupported = sorted(set(providers) - supported)
        if unsupported:
            raise ConfigError(f"Unsupported provider(s): {unsupported}")

        echo_model = os.environ.get("GATEWAY_ECHO_MODEL", "echo-1")
        bedrock_model = os.environ.get("GATEWAY_BEDROCK_MODEL", "") or None
        openai_model = os.environ.get("GATEWAY_OPENAI_MODEL", "") or None
        gemini_model = os.environ.get("GATEWAY_GEMINI_MODEL", "") or None
        if "bedrock" in providers and not bedrock_model:
            raise ConfigError("GATEWAY_BEDROCK_MODEL is required when bedrock is enabled")
        if "openai" in providers and not openai_model:
            raise ConfigError("GATEWAY_OPENAI_MODEL is required when openai is enabled")
        if "gemini" in providers and not gemini_model:
            raise ConfigError("GATEWAY_GEMINI_MODEL is required when gemini is enabled")

        pricing_file = Path(os.environ.get("GATEWAY_PRICING_FILE", "data/pricing.example.json"))
        ledger_path = Path(os.environ.get("GATEWAY_LEDGER_PATH", "gateway.db"))

        return cls(
            providers=providers,
            echo_model=echo_model,
            bedrock_model=bedrock_model,
            openai_model=openai_model,
            gemini_model=gemini_model,
            pricing_file=pricing_file,
            ledger_path=ledger_path,
        )
