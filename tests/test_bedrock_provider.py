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
    assert call["inferenceConfig"] == {"maxTokens": 512, "temperature": 0.0}


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
