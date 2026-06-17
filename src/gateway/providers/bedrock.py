"""AWS Bedrock provider using the Converse API.

Credentials and region are intentionally not configured here. boto3 resolves
them through the standard AWS chain (AWS_PROFILE, AWS_REGION, instance
roles, etc.). This keeps the repo environment-agnostic and avoids leaking
local profile/account names into code.
"""

from __future__ import annotations

import os

from ..messages import normalize_content
from .base import Completion, ProviderError, ToolCall

# The Claude 4.7+/Fable family removed sampling params: Converse returns a
# ValidationException ("`temperature` is deprecated for this model") if we send
# one. Older Anthropic models on Bedrock still accept temperature, so we only
# omit it for the families that reject it.
_SAMPLING_REMOVED = ("opus-4-7", "opus-4-8", "fable-5", "mythos-5")


def _accepts_temperature(model: str) -> bool:
    return not any(marker in model.lower() for marker in _SAMPLING_REMOVED)


class BedrockProvider:
    name = "bedrock"

    def __init__(self, model: str, client=None, *, max_tokens: int | None = None):
        if not model:
            raise ValueError("model id is required")
        self._model = model
        self._max_tokens = max_tokens or int(
            os.environ.get("GATEWAY_BEDROCK_MAX_TOKENS", "2048")
        )
        if client is None:
            import boto3  # lazy import so tests without bedrock do not need credentials

            client = boto3.client("bedrock-runtime")
        self._client = client

    def complete(self, messages: list[dict], tools: list[dict] | None = None) -> Completion:
        if not messages:
            raise ProviderError("no messages provided")

        inference_config: dict = {"maxTokens": self._max_tokens}
        if _accepts_temperature(self._model):
            inference_config["temperature"] = 0.0

        kwargs: dict = {
            "modelId": self._model,
            "messages": [self._to_bedrock_message(m) for m in messages
                         if m.get("role") != "system"],
            "system": [self._to_system_message(m) for m in messages
                       if m.get("role") == "system"],
            "inferenceConfig": inference_config,
        }
        if tools:
            kwargs["toolConfig"] = {"tools": _to_tool_config(tools)}

        try:
            response = self._client.converse(**kwargs)
        except Exception as exc:  # boto3 raises a family of service/client errors
            raise ProviderError(f"bedrock call failed: {exc}") from exc

        try:
            content_blocks = response["output"]["message"]["content"]
            text = "".join(b["text"] for b in content_blocks if "text" in b)
            tool_calls = _parse_tool_use(content_blocks)
            usage = response["usage"]
            input_tokens = usage["inputTokens"]
            output_tokens = usage["outputTokens"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"bedrock response missing expected fields: {response!r}") from exc

        return Completion(
            text=text,
            model=self._model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_calls=tool_calls,
            stop_reason="tool_use" if tool_calls else "end_turn",
        )

    @staticmethod
    def _to_bedrock_message(message: dict) -> dict:
        content = message.get("content", "")
        if isinstance(content, str):
            blocks = [{"text": content}]
        else:
            blocks = [_to_bedrock_block(b) for b in normalize_content(content)]
            blocks = [b for b in blocks if b]
        return {"role": BedrockProvider._to_bedrock_role(message), "content": blocks}

    @staticmethod
    def _to_system_message(message: dict) -> dict:
        content = message.get("content", "")
        if isinstance(content, str):
            return {"text": content}
        return {"text": "".join(b.get("text", "") for b in normalize_content(content)
                               if b.get("type") == "text")}

    @staticmethod
    def _to_bedrock_role(message: dict) -> str:
        role = message.get("role")
        if role in ("user", "assistant"):
            return role
        # Converse supports system separately; other roles are normalized to user
        # rather than dropped, so the model still sees the content.
        return "user"


def _to_bedrock_block(block: dict) -> dict | None:
    kind = block.get("type")
    if kind == "text":
        return {"text": block.get("text", "")} if block.get("text") else None
    if kind == "tool_use":
        return {
            "toolUse": {
                "toolUseId": block["id"],
                "name": block["name"],
                "input": block.get("input", {}),
            }
        }
    if kind == "tool_result":
        result = block.get("content", "")
        if isinstance(result, dict):
            result_content = [{"json": result}]
        else:
            result_content = [{"text": str(result)}]
        return {"toolResult": {"toolUseId": block["tool_use_id"], "content": result_content}}
    return None


def _to_tool_config(tools: list[dict]) -> list[dict]:
    empty_schema = {"type": "object", "properties": {}}
    return [
        {
            "toolSpec": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "inputSchema": {"json": tool.get("input_schema", empty_schema)},
            }
        }
        for tool in tools
    ]


def _parse_tool_use(content_blocks: list[dict]) -> tuple[ToolCall, ...]:
    calls = []
    for block in content_blocks:
        tool_use = block.get("toolUse")
        if tool_use:
            calls.append(
                ToolCall(
                    id=tool_use["toolUseId"],
                    name=tool_use["name"],
                    input=tool_use.get("input", {}) or {},
                )
            )
    return tuple(calls)
