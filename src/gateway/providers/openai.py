"""OpenAI Chat Completions provider (HTTP, no SDK).

We talk to ``/v1/chat/completions`` directly with httpx so the gateway has
zero SDK coupling and works against any OpenAI-compatible endpoint
(Together, Groq's OpenAI mode, vLLM, llama.cpp server, ...). The base URL
is configurable: same code, different upstream.
"""

from __future__ import annotations

import json
import os

import httpx

from ..messages import normalize_content
from ..retry import RetryPolicy, call_with_retries, is_retryable_http_status
from .base import Completion, ProviderError, ToolCall

_DEFAULT_BASE_URL = "https://api.openai.com/v1"

# Reasoning models (GPT-5.x, o-series) reject `temperature` (only the default
# is allowed) and `max_tokens` (they want `max_completion_tokens`). We branch on
# the model name so the same provider serves both classic chat models and
# reasoning models without per-deployment config.
_REASONING_PREFIXES = ("gpt-5", "gpt-6", "o1", "o3", "o4")


def _is_reasoning_model(model: str) -> bool:
    return model.lower().startswith(_REASONING_PREFIXES)


class OpenAIProvider:
    name = "openai"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_s: float = 30.0,
        max_completion_tokens: int | None = None,
        client: httpx.Client | None = None,
        retry_policy: RetryPolicy | None = None,
    ):
        if not model:
            raise ValueError("model is required")
        self._model = model
        # Reasoning tokens count against this cap, so default generously to
        # avoid an empty visible answer when the model spends its budget thinking.
        self._max_completion_tokens = max_completion_tokens or int(
            os.environ.get("GATEWAY_OPENAI_MAX_TOKENS", "4096")
        )
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        self._base_url = (
            base_url or os.environ.get("OPENAI_BASE_URL") or _DEFAULT_BASE_URL
        ).rstrip("/")
        self._timeout_s = timeout_s
        self._client = client
        self._retry = retry_policy or RetryPolicy.from_env()

    def complete(self, messages: list[dict], tools: list[dict] | None = None) -> Completion:
        if not messages:
            raise ProviderError("no messages provided")

        openai_messages = _to_openai_messages(messages)
        if not openai_messages:
            raise ProviderError("messages must contain role/content entries")

        payload: dict = {"model": self._model, "messages": openai_messages}
        if _is_reasoning_model(self._model):
            payload["max_completion_tokens"] = self._max_completion_tokens
        else:
            payload["temperature"] = 0
        if tools:
            payload["tools"] = _to_openai_tools(tools)
            # One tool call per turn: GPT-5.x otherwise fires many searches at
            # once, which makes the agent's trace long and repetitive. Deliberate,
            # one-at-a-time tool use reads better and is easier to follow.
            payload["parallel_tool_calls"] = False

        try:
            data = call_with_retries(
                lambda: self._post(payload),
                is_retryable=_is_retryable,
                policy=self._retry,
            )
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                f"openai HTTP {exc.response.status_code}: {exc.response.text[:200]}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"openai request failed: {exc}") from exc

        try:
            choice = data["choices"][0]
            message = choice["message"]
            text = message.get("content") or ""
            tool_calls = _parse_tool_calls(message.get("tool_calls"))
            usage = data["usage"]
            input_tokens = int(usage["prompt_tokens"])
            output_tokens = int(usage["completion_tokens"])
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderError(
                f"openai response missing expected fields: {data!r}"
            ) from exc

        return Completion(
            text=text,
            model=self._model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_calls=tool_calls,
            stop_reason="tool_use" if tool_calls else "end_turn",
        )

    def _post(self, payload: dict) -> dict:
        client = self._client or httpx.Client(timeout=self._timeout_s)
        try:
            response = client.post(
                f"{self._base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=payload,
            )
        finally:
            if self._client is None:
                client.close()
        response.raise_for_status()
        return response.json()


def _to_openai_tools(tools: list[dict]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
            },
        }
        for tool in tools
    ]


def _to_openai_messages(messages: list[dict]) -> list[dict]:
    """Translate canonical messages to OpenAI's wire format.

    Text content passes through. A canonical assistant turn with ``tool_use``
    blocks becomes an assistant message with ``tool_calls``; each ``tool_result``
    block becomes its own ``role: "tool"`` message (OpenAI's required shape).
    """
    out: list[dict] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role is None or content is None:
            continue

        # Plain string content (the common, non-tool path).
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue

        blocks = normalize_content(content)
        tool_results = [b for b in blocks if b.get("type") == "tool_result"]
        tool_uses = [b for b in blocks if b.get("type") == "tool_use"]
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")

        if tool_results:
            for block in tool_results:
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": block["tool_use_id"],
                        "content": block.get("content", ""),
                    }
                )
            if text:
                out.append({"role": "user", "content": text})
            continue

        if tool_uses:
            out.append(
                {
                    "role": "assistant",
                    "content": text or None,
                    "tool_calls": [
                        {
                            "id": b["id"],
                            "type": "function",
                            "function": {
                                "name": b["name"],
                                "arguments": json.dumps(b.get("input", {})),
                            },
                        }
                        for b in tool_uses
                    ],
                }
            )
            continue

        out.append({"role": role, "content": text})
    return out


def _parse_tool_calls(raw) -> tuple[ToolCall, ...]:
    if not raw:
        return ()
    calls = []
    for tc in raw:
        function = tc.get("function", {})
        arguments = function.get("arguments") or "{}"
        try:
            parsed = json.loads(arguments)
        except (json.JSONDecodeError, TypeError):
            parsed = {}
        calls.append(ToolCall(id=tc.get("id", ""), name=function.get("name", ""), input=parsed))
    return tuple(calls)


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return is_retryable_http_status(exc.response.status_code)
    return isinstance(exc, httpx.TransportError | httpx.TimeoutException)
