"""Provider abstraction.

A provider is anything that can turn chat messages into a completion and
report its token usage. The protocol is one method so adding a provider
(Bedrock, Anthropic API, OpenAI, a self-hosted vLLM...) never touches the
router, the ledger, or the app.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    input_tokens: int
    output_tokens: int


class ProviderError(RuntimeError):
    """Raised by a provider when it cannot serve the request."""


class Provider(Protocol):
    name: str

    def complete(self, messages: list[dict]) -> Completion:
        """Serve a chat completion. ``messages`` is [{"role", "content"}, ...]."""
        ...
