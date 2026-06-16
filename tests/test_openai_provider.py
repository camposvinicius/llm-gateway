"""Offline tests for the OpenAI provider using respx-mocked HTTP."""

from __future__ import annotations

import httpx
import pytest
import respx

from gateway.providers import OpenAIProvider, ProviderError
from gateway.retry import RetryPolicy

_BASE = "https://api.openai.com/v1"

_OK_BODY = {
    "id": "chatcmpl-test",
    "choices": [{"message": {"role": "assistant", "content": "hi from openai"}}],
    "usage": {"prompt_tokens": 9, "completion_tokens": 4},
}


def _provider(client: httpx.Client) -> OpenAIProvider:
    return OpenAIProvider(
        model="gpt-test",
        api_key="sk-test",
        client=client,
        retry_policy=RetryPolicy(max_attempts=3, base_delay_seconds=0.0, sleep=lambda _: None),
    )


@respx.mock
def test_successful_completion_carries_usage():
    respx.post(f"{_BASE}/chat/completions").mock(return_value=httpx.Response(200, json=_OK_BODY))

    with httpx.Client() as client:
        provider = _provider(client)
        completion = provider.complete([{"role": "user", "content": "hi"}])

    assert completion.text == "hi from openai"
    assert completion.model == "gpt-test"
    assert completion.input_tokens == 9
    assert completion.output_tokens == 4


@respx.mock
def test_5xx_is_retried_then_succeeds():
    route = respx.post(f"{_BASE}/chat/completions").mock(
        side_effect=[
            httpx.Response(503, text="overloaded"),
            httpx.Response(200, json=_OK_BODY),
        ]
    )

    with httpx.Client() as client:
        provider = _provider(client)
        completion = provider.complete([{"role": "user", "content": "hi"}])

    assert completion.text == "hi from openai"
    assert route.call_count == 2


@respx.mock
def test_429_is_retried():
    route = respx.post(f"{_BASE}/chat/completions").mock(
        side_effect=[
            httpx.Response(429, text="slow down"),
            httpx.Response(200, json=_OK_BODY),
        ]
    )

    with httpx.Client() as client:
        provider = _provider(client)
        completion = provider.complete([{"role": "user", "content": "hi"}])

    assert completion.text == "hi from openai"
    assert route.call_count == 2


@respx.mock
def test_400_is_not_retried_and_surfaced_as_provider_error():
    route = respx.post(f"{_BASE}/chat/completions").mock(
        return_value=httpx.Response(400, text='{"error":"bad model"}')
    )

    with httpx.Client() as client:
        provider = _provider(client)
        with pytest.raises(ProviderError, match="HTTP 400"):
            provider.complete([{"role": "user", "content": "hi"}])

    assert route.call_count == 1


@respx.mock
def test_exhausted_retries_raise_provider_error():
    route = respx.post(f"{_BASE}/chat/completions").mock(
        return_value=httpx.Response(500, text="boom")
    )

    with httpx.Client() as client:
        provider = _provider(client)
        with pytest.raises(ProviderError, match="HTTP 500"):
            provider.complete([{"role": "user", "content": "hi"}])

    assert route.call_count == 3  # max_attempts


@respx.mock
def test_malformed_response_is_provider_error():
    respx.post(f"{_BASE}/chat/completions").mock(return_value=httpx.Response(200, json={"oops": 1}))

    with httpx.Client() as client:
        provider = _provider(client)
        with pytest.raises(ProviderError, match="missing expected fields"):
            provider.complete([{"role": "user", "content": "hi"}])


def test_missing_model_rejected():
    with pytest.raises(ValueError, match="model is required"):
        OpenAIProvider(model="", api_key="sk-test")


def test_missing_api_key_rejected(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIProvider(model="gpt-test")
