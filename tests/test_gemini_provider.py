"""Offline tests for the Gemini provider using respx-mocked HTTP."""

from __future__ import annotations

import httpx
import pytest
import respx

from gateway.providers import GeminiProvider, ProviderError
from gateway.retry import RetryPolicy

_BASE = "https://generativelanguage.googleapis.com"
_PATH = f"{_BASE}/v1beta/models/gemini-test:generateContent"

_OK_BODY = {
    "candidates": [
        {"content": {"parts": [{"text": "hi from gemini"}]}}
    ],
    "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 5},
}


def _provider(client: httpx.Client) -> GeminiProvider:
    return GeminiProvider(
        model="gemini-test",
        api_key="test-key",
        client=client,
        retry_policy=RetryPolicy(max_attempts=3, base_delay_seconds=0.0, sleep=lambda _: None),
    )


@respx.mock
def test_successful_completion_maps_usage_metadata():
    respx.post(_PATH).mock(return_value=httpx.Response(200, json=_OK_BODY))

    with httpx.Client() as client:
        provider = _provider(client)
        completion = provider.complete([{"role": "user", "content": "hi"}])

    assert completion.text == "hi from gemini"
    assert completion.model == "gemini-test"
    assert completion.input_tokens == 7
    assert completion.output_tokens == 5


@respx.mock
def test_system_message_goes_to_system_instruction():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = request.read().decode()
        return httpx.Response(200, json=_OK_BODY)

    respx.post(_PATH).mock(side_effect=handler)

    with httpx.Client() as client:
        provider = _provider(client)
        provider.complete(
            [
                {"role": "system", "content": "be concise"},
                {"role": "user", "content": "hi"},
            ]
        )

    assert "systemInstruction" in captured["payload"]
    assert "be concise" in captured["payload"]


@respx.mock
def test_5xx_is_retried_then_succeeds():
    route = respx.post(_PATH).mock(
        side_effect=[
            httpx.Response(502, text="bad gateway"),
            httpx.Response(200, json=_OK_BODY),
        ]
    )

    with httpx.Client() as client:
        provider = _provider(client)
        provider.complete([{"role": "user", "content": "hi"}])

    assert route.call_count == 2


@respx.mock
def test_400_is_not_retried():
    route = respx.post(_PATH).mock(return_value=httpx.Response(400, text="bad model"))

    with httpx.Client() as client:
        provider = _provider(client)
        with pytest.raises(ProviderError, match="HTTP 400"):
            provider.complete([{"role": "user", "content": "hi"}])

    assert route.call_count == 1


@respx.mock
def test_malformed_response_is_provider_error():
    respx.post(_PATH).mock(return_value=httpx.Response(200, json={"unexpected": True}))

    with httpx.Client() as client:
        provider = _provider(client)
        with pytest.raises(ProviderError, match="missing expected fields"):
            provider.complete([{"role": "user", "content": "hi"}])


def test_missing_api_key_rejected(monkeypatch):
    for var in ("GOOGLE_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(ValueError, match="GOOGLE_KEY"):
        GeminiProvider(model="gemini-test")


def test_missing_model_rejected():
    with pytest.raises(ValueError, match="model is required"):
        GeminiProvider(model="", api_key="x")
