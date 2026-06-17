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
def test_thinking_model_skips_thought_parts_and_bills_thought_tokens():
    # Gemini 3.x thinking models interleave a thought part (thoughtSignature, no
    # text) and bill thoughtsTokenCount at the output rate.
    body = {
        "candidates": [
            {"content": {"parts": [{"thoughtSignature": "abc"}, {"text": "final answer"}]}}
        ],
        "usageMetadata": {
            "promptTokenCount": 6,
            "candidatesTokenCount": 2,
            "thoughtsTokenCount": 178,
        },
    }
    respx.post(_PATH).mock(return_value=httpx.Response(200, json=body))

    with httpx.Client() as client:
        completion = _provider(client).complete([{"role": "user", "content": "hi"}])

    assert completion.text == "final answer"
    assert completion.input_tokens == 6
    assert completion.output_tokens == 2 + 178


@respx.mock
def test_function_call_parsed_with_thought_signature():
    body = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "functionCall": {"name": "get_weather", "args": {"location": "Paris"}},
                            "thoughtSignature": "sig123",
                        }
                    ]
                }
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 6,
            "candidatesTokenCount": 0,
            "thoughtsTokenCount": 10,
        },
    }
    route = respx.post(_PATH).mock(return_value=httpx.Response(200, json=body))
    tools = [{"name": "get_weather", "description": "w", "input_schema": {"type": "object"}}]

    with httpx.Client() as client:
        completion = _provider(client).complete(
            [{"role": "user", "content": "weather?"}], tools=tools
        )

    import json

    sent = json.loads(route.calls.last.request.content)
    assert sent["tools"][0]["functionDeclarations"][0]["name"] == "get_weather"
    assert completion.stop_reason == "tool_use"
    call = completion.tool_calls[0]
    assert call.name == "get_weather"
    assert call.input == {"location": "Paris"}
    assert call.signature == "sig123"


@respx.mock
def test_tool_use_echoes_signature_and_result_becomes_function_response():
    route = respx.post(_PATH).mock(return_value=httpx.Response(200, json=_OK_BODY))
    messages = [
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "gemini-0-get_weather",
                    "name": "get_weather",
                    "input": {"location": "Paris"},
                    "signature": "sig123",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "gemini-0-get_weather",
                    "name": "get_weather",
                    "content": "18C",
                }
            ],
        },
    ]
    with httpx.Client() as client:
        _provider(client).complete(messages)

    import json

    contents = json.loads(route.calls.last.request.content)["contents"]
    model_turn = next(c for c in contents if c["role"] == "model")
    fc_part = model_turn["parts"][0]
    assert fc_part["functionCall"]["name"] == "get_weather"
    assert fc_part["thoughtSignature"] == "sig123"
    result_turn = contents[-1]
    assert result_turn["parts"][0]["functionResponse"]["name"] == "get_weather"
    assert result_turn["parts"][0]["functionResponse"]["response"] == {"result": "18C"}


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
