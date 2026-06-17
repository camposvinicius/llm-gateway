"""Google Gemini provider via the Generative Language API (HTTP).

The HTTP API stays stable across SDK churn, so we hit it directly:
``POST /v1beta/models/{model}:generateContent?key=...``. Token usage
comes back under ``usageMetadata`` with ``promptTokenCount`` and
``candidatesTokenCount``.
"""

from __future__ import annotations

import os

import httpx

from ..messages import normalize_content
from ..retry import RetryPolicy, call_with_retries, is_retryable_http_status
from .base import Completion, ProviderError, ToolCall

_DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"


class GeminiProvider:
    name = "gemini"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_s: float = 30.0,
        client: httpx.Client | None = None,
        retry_policy: RetryPolicy | None = None,
    ):
        if not model:
            raise ValueError("model is required")
        self._model = model
        self._api_key = (
            api_key
            or os.environ.get("GOOGLE_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
        )
        if not self._api_key:
            raise ValueError("GOOGLE_KEY/GOOGLE_API_KEY/GEMINI_API_KEY is not set")
        self._base_url = (
            base_url or os.environ.get("GEMINI_BASE_URL") or _DEFAULT_BASE_URL
        ).rstrip("/")
        self._timeout_s = timeout_s
        self._client = client
        self._retry = retry_policy or RetryPolicy.from_env()

    def complete(self, messages: list[dict], tools: list[dict] | None = None) -> Completion:
        if not messages:
            raise ProviderError("no messages provided")

        contents = []
        system_parts: list[str] = []
        for message in messages:
            role = message.get("role")
            if role == "system":
                system_parts.append(_text_only(message.get("content", "")))
                continue
            parts = _to_gemini_parts(message.get("content", ""))
            if parts:
                contents.append(
                    {"role": "user" if role == "user" else "model", "parts": parts}
                )
        if not contents:
            raise ProviderError("messages must include at least one user/assistant turn")

        payload: dict = {
            "contents": contents,
            "generationConfig": {"temperature": 0},
        }
        if system_parts:
            payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}
        if tools:
            payload["tools"] = [{"functionDeclarations": _to_gemini_declarations(tools)}]

        try:
            data = call_with_retries(
                lambda: self._post(payload),
                is_retryable=_is_retryable,
                policy=self._retry,
            )
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                f"gemini HTTP {exc.response.status_code}: {exc.response.text[:200]}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"gemini request failed: {exc}") from exc

        try:
            parts = data["candidates"][0]["content"]["parts"]
            # Thinking models (Gemini 3.x) interleave thought parts that carry a
            # thoughtSignature and no "text"; join only the text parts.
            text = "".join(part["text"] for part in parts if "text" in part)
            tool_calls = _parse_function_calls(parts)
            usage = data["usageMetadata"]
            input_tokens = int(usage["promptTokenCount"])
            # thoughtsTokenCount is billed at the output rate, so it must be
            # metered as output or the cost is understated for thinking models.
            output_tokens = int(usage.get("candidatesTokenCount", 0)) + int(
                usage.get("thoughtsTokenCount", 0)
            )
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderError(
                f"gemini response missing expected fields: {data!r}"
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
                f"{self._base_url}/v1beta/models/{self._model}:generateContent",
                params={"key": self._api_key},
                json=payload,
            )
        finally:
            if self._client is None:
                client.close()
        response.raise_for_status()
        return response.json()


def _text_only(content) -> str:
    if isinstance(content, str):
        return content
    blocks = normalize_content(content)
    return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")


def _to_gemini_parts(content) -> list[dict]:
    """Translate one message's content into Gemini ``parts``.

    Gemini pairs a tool result to its call by **function name**, not an id —
    so the canonical ``tool_result`` block carries ``name`` and we use it here.
    """
    parts: list[dict] = []
    for block in normalize_content(content):
        kind = block.get("type")
        if kind == "text":
            if block.get("text"):
                parts.append({"text": block["text"]})
        elif kind == "tool_use":
            part = {"functionCall": {"name": block["name"], "args": block.get("input", {})}}
            if block.get("signature"):
                part["thoughtSignature"] = block["signature"]
            parts.append(part)
        elif kind == "tool_result":
            result = block.get("content", "")
            response = result if isinstance(result, dict) else {"result": result}
            parts.append(
                {"functionResponse": {"name": block.get("name", ""), "response": response}}
            )
    return parts


def _to_gemini_declarations(tools: list[dict]) -> list[dict]:
    return [
        {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
        }
        for tool in tools
    ]


def _parse_function_calls(parts: list[dict]) -> tuple[ToolCall, ...]:
    calls = []
    for index, part in enumerate(parts):
        function_call = part.get("functionCall")
        if function_call:
            calls.append(
                ToolCall(
                    # Gemini has no native call id; synthesize a unique one so
                    # the agent can pair results even on duplicate tool names.
                    id=f"gemini-{index}-{function_call.get('name', '')}",
                    name=function_call.get("name", ""),
                    input=function_call.get("args", {}) or {},
                    # Gemini 3.x rejects history whose functionCall part is
                    # missing its thoughtSignature — carry it through.
                    signature=part.get("thoughtSignature"),
                )
            )
    return tuple(calls)


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return is_retryable_http_status(exc.response.status_code)
    return isinstance(exc, httpx.TransportError | httpx.TimeoutException)
