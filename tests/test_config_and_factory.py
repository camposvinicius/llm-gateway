"""Tests for env-driven settings and provider construction."""

import json

import pytest

from gateway.config import ConfigError, Settings
from gateway.factory import build_providers
from gateway.providers import EchoProvider


def test_default_settings_are_offline_echo(monkeypatch):
    for key in [
        "GATEWAY_PROVIDERS",
        "GATEWAY_ECHO_MODEL",
        "GATEWAY_BEDROCK_MODEL",
        "GATEWAY_PRICING_FILE",
        "GATEWAY_LEDGER_PATH",
    ]:
        monkeypatch.delenv(key, raising=False)

    settings = Settings.from_env()

    assert settings.providers == ("echo",)
    assert settings.echo_model == "echo-1"


def test_unsupported_provider_fails(monkeypatch):
    monkeypatch.setenv("GATEWAY_PROVIDERS", "openai")
    with pytest.raises(ConfigError, match="Unsupported provider"):
        Settings.from_env()


def test_bedrock_requires_model(monkeypatch):
    monkeypatch.setenv("GATEWAY_PROVIDERS", "bedrock")
    monkeypatch.delenv("GATEWAY_BEDROCK_MODEL", raising=False)
    with pytest.raises(ConfigError, match="GATEWAY_BEDROCK_MODEL"):
        Settings.from_env()


def test_build_echo_provider(monkeypatch, tmp_path):
    pricing = tmp_path / "pricing.json"
    pricing.write_text(
        json.dumps({"models": {"echo-test": {"input_per_million": "1", "output_per_million": "1"}}})
    )
    monkeypatch.setenv("GATEWAY_PROVIDERS", "echo")
    monkeypatch.setenv("GATEWAY_ECHO_MODEL", "echo-test")
    monkeypatch.setenv("GATEWAY_PRICING_FILE", str(pricing))

    providers = build_providers(Settings.from_env())

    assert len(providers) == 1
    assert isinstance(providers[0], EchoProvider)
