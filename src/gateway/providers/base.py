"""Provider abstraction.

A provider is anything that can turn chat messages into a completion and
report its token usage. The protocol is one method so adding a provider
(Bedrock, Anthropic API, OpenAI, a self-hosted vLLM...) never touches the
router, the ledger, or the app.

Tool calling is part of the same one-method contract. Each provider speaks a
different native dialect (OpenAI ``tool_calls``, Gemini ``functionCall``,
Bedrock ``toolUse``); the gateway's job is to normalize all of them to one
canonical shape so callers — and the bundled research agent — write the loop
once. The canonical shapes live here:

- A **tool definition** (request ``tools``): ``{"name", "description", "input_schema"}``.
- A **tool call** (response): :class:`ToolCall` with ``id``/``name``/``input``.
- **Message content** is either a plain string or a list of blocks:
  ``{"type": "text", "text": ...}``,
  ``{"type": "tool_use", "id", "name", "input"}`` (assistant turns),
  ``{"type": "tool_result", "tool_use_id", "name", "content"}`` (user turns).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ToolCall:
    """A tool the model wants the caller to run, in canonical form.

    ``id`` pairs the call with its result (OpenAI/Bedrock use ids natively;
    for Gemini we synthesize one). ``input`` is the already-parsed arguments
    object — never a raw JSON string.

    ``signature`` is an opaque, provider-specific token that must be echoed back
    on the next turn for tools to keep working. Gemini 3.x sets it (its
    ``thoughtSignature``) and rejects history that drops it; other providers
    leave it ``None``.
    """

    id: str
    name: str
    input: dict
    signature: str | None = None


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    # Empty/"end_turn" keep non-tool callers and existing tests unchanged.
    tool_calls: tuple[ToolCall, ...] = ()
    stop_reason: str = "end_turn"


class ProviderError(RuntimeError):
    """Raised by a provider when it cannot serve the request."""


class Provider(Protocol):
    name: str

    def complete(self, messages: list[dict], tools: list[dict] | None = None) -> Completion:
        """Serve a chat completion.

        ``messages`` is ``[{"role", "content"}, ...]`` where ``content`` is a
        string or a list of canonical blocks. ``tools`` is an optional list of
        canonical tool definitions; when present the model may return
        ``tool_calls`` with ``stop_reason == "tool_use"``.
        """
        ...
