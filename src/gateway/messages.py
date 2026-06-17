"""Canonical message/content helpers shared across providers and the agent.

Message ``content`` in this gateway is either a plain string (the common case)
or a list of blocks. Keeping a tiny set of helpers here means every provider
handles both shapes the same way, and the agent loop has one vocabulary for
building assistant tool-call turns and user tool-result turns.

Block shapes (see ``providers/base.py`` for the full contract):

- ``{"type": "text", "text": ...}``
- ``{"type": "tool_use", "id", "name", "input"}``        (assistant)
- ``{"type": "tool_result", "tool_use_id", "name", "content"}``  (user)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid a runtime import cycle (providers import this module)
    from .providers.base import ToolCall


def normalize_content(content) -> list[dict]:
    """Return message content as a list of blocks.

    A plain string becomes a single text block so providers only deal with one
    shape internally.
    """
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        return content
    return [{"type": "text", "text": str(content)}]


def text_of(content) -> str:
    """Concatenate the text blocks of a content value (ignoring tool blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content)


def assistant_tool_use_turn(text: str, tool_calls: tuple[ToolCall, ...]) -> dict:
    """Build the assistant turn that records the model's tool calls.

    This is echoed back to the provider on the next step so the model sees its
    own request alongside the results.
    """
    blocks: list[dict] = []
    if text:
        blocks.append({"type": "text", "text": text})
    for call in tool_calls:
        block = {"type": "tool_use", "id": call.id, "name": call.name, "input": call.input}
        if call.signature:
            block["signature"] = call.signature
        blocks.append(block)
    return {"role": "assistant", "content": blocks}


def tool_result_turn(results: list[dict]) -> dict:
    """Build the user turn carrying tool results.

    Each entry is ``{"tool_use_id", "name", "content"}`` — ``name`` is required
    because Gemini pairs results to calls by function name, not by id.
    """
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": r["tool_use_id"],
                "name": r["name"],
                "content": r["content"],
            }
            for r in results
        ],
    }
