"""Tests for the echo provider (the deterministic offline provider)."""

import pytest

from gateway.providers import EchoProvider, ProviderError


def test_echoes_last_user_message_with_word_count_usage():
    provider = EchoProvider(model="echo-1")
    completion = provider.complete(
        [
            {"role": "system", "content": "be brief"},          # 2 words
            {"role": "user", "content": "hello there gateway"},  # 3 words
        ]
    )
    assert completion.text == "echo: hello there gateway"
    assert completion.model == "echo-1"
    assert completion.input_tokens == 5   # 2 + 3
    assert completion.output_tokens == 4  # "echo:" + 3 words


def test_uses_last_user_message_not_first():
    provider = EchoProvider(model="echo-1")
    completion = provider.complete(
        [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "echo: first"},
            {"role": "user", "content": "second"},
        ]
    )
    assert completion.text == "echo: second"


def test_no_user_message_is_an_error():
    provider = EchoProvider(model="echo-1")
    with pytest.raises(ProviderError, match="no user message"):
        provider.complete([{"role": "system", "content": "hi"}])


def test_empty_messages_is_an_error():
    provider = EchoProvider(model="echo-1")
    with pytest.raises(ProviderError, match="no messages"):
        provider.complete([])


def test_configured_failure_for_fallback_tests():
    provider = EchoProvider(model="echo-1", fail=True)
    with pytest.raises(ProviderError, match="configured to fail"):
        provider.complete([{"role": "user", "content": "hi"}])
