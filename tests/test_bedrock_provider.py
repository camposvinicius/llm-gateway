"""Offline tests for the Bedrock provider using a fake boto3 client."""

import pytest

from gateway.providers import BedrockProvider, ProviderError


class FakeBedrockClient:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def test_converse_request_and_usage_mapping():
    response = {
        "output": {"message": {"content": [{"text": "hello from bedrock"}]}},
        "usage": {"inputTokens": 7, "outputTokens": 3},
    }
    client = FakeBedrockClient(response=response)
    provider = BedrockProvider(model="model-1", client=client)

    completion = provider.complete(
        [
            {"role": "system", "content": "be concise"},
            {"role": "user", "content": "say hello"},
        ]
    )

    assert completion.text == "hello from bedrock"
    assert completion.model == "model-1"
    assert completion.input_tokens == 7
    assert completion.output_tokens == 3

    call = client.calls[0]
    assert call["modelId"] == "model-1"
    assert call["system"] == [{"text": "be concise"}]
    assert call["messages"] == [{"role": "user", "content": [{"text": "say hello"}]}]
    assert call["inferenceConfig"] == {"maxTokens": 2048, "temperature": 0.0}


def test_max_tokens_is_configurable():
    client = FakeBedrockClient(
        response={
            "output": {"message": {"content": [{"text": "ok"}]}},
            "usage": {"inputTokens": 1, "outputTokens": 1},
        }
    )
    provider = BedrockProvider(model="model-1", client=client, max_tokens=256)
    provider.complete([{"role": "user", "content": "hi"}])
    assert client.calls[0]["inferenceConfig"]["maxTokens"] == 256


def test_temperature_omitted_for_sampling_removed_models():
    # Claude Opus 4.8 (and the 4.7+/Fable family) reject `temperature` on
    # Converse — the provider must not send it.
    client = FakeBedrockClient(
        response={
            "output": {"message": {"content": [{"text": "ok"}]}},
            "usage": {"inputTokens": 1, "outputTokens": 1},
        }
    )
    provider = BedrockProvider(model="global.anthropic.claude-opus-4-8", client=client)
    provider.complete([{"role": "user", "content": "hi"}])
    assert "temperature" not in client.calls[0]["inferenceConfig"]


def test_tool_config_sent_and_tool_use_parsed():
    response = {
        "output": {
            "message": {
                "content": [
                    {"text": "Let me check."},
                    {
                        "toolUse": {
                            "toolUseId": "tu1",
                            "name": "get_weather",
                            "input": {"location": "Paris"},
                        }
                    },
                ]
            }
        },
        "usage": {"inputTokens": 5, "outputTokens": 3},
    }
    client = FakeBedrockClient(response=response)
    provider = BedrockProvider(model="model-1", client=client)
    tools = [{"name": "get_weather", "description": "w", "input_schema": {"type": "object"}}]

    completion = provider.complete([{"role": "user", "content": "weather?"}], tools=tools)

    spec = client.calls[0]["toolConfig"]["tools"][0]["toolSpec"]
    assert spec["name"] == "get_weather"
    assert spec["inputSchema"] == {"json": {"type": "object"}}
    assert completion.text == "Let me check."
    assert completion.stop_reason == "tool_use"
    assert completion.tool_calls[0].id == "tu1"
    assert completion.tool_calls[0].input == {"location": "Paris"}


def test_tool_use_and_result_blocks_translate_to_converse():
    client = FakeBedrockClient(
        response={
            "output": {"message": {"content": [{"text": "done"}]}},
            "usage": {"inputTokens": 1, "outputTokens": 1},
        }
    )
    provider = BedrockProvider(model="model-1", client=client)
    messages = [
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tu1",
                    "name": "get_weather",
                    "input": {"location": "Paris"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tu1",
                    "name": "get_weather",
                    "content": "18C",
                }
            ],
        },
    ]
    provider.complete(messages)

    sent = client.calls[0]["messages"]
    assert sent[1]["content"][0]["toolUse"]["toolUseId"] == "tu1"
    assert sent[2]["content"][0]["toolResult"] == {
        "toolUseId": "tu1",
        "content": [{"text": "18C"}],
    }


def test_provider_errors_are_wrapped():
    provider = BedrockProvider(
        model="model-1", client=FakeBedrockClient(error=RuntimeError("boom"))
    )
    with pytest.raises(ProviderError, match="bedrock call failed"):
        provider.complete([{"role": "user", "content": "hi"}])


def test_malformed_response_is_error():
    provider = BedrockProvider(model="model-1", client=FakeBedrockClient(response={"bad": "shape"}))
    with pytest.raises(ProviderError, match="missing expected fields"):
        provider.complete([{"role": "user", "content": "hi"}])


def test_empty_messages_is_error():
    provider = BedrockProvider(model="model-1", client=FakeBedrockClient(response={}))
    with pytest.raises(ProviderError, match="no messages"):
        provider.complete([])


def test_unknown_roles_are_preserved_as_user_content():
    response = {
        "output": {"message": {"content": [{"text": "ok"}]}},
        "usage": {"inputTokens": 1, "outputTokens": 1},
    }
    client = FakeBedrockClient(response=response)
    provider = BedrockProvider(model="model-1", client=client)

    provider.complete([{"role": "tool", "content": "tool output"}])

    assert client.calls[0]["messages"] == [
        {"role": "user", "content": [{"text": "tool output"}]}
    ]
