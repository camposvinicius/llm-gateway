"""OpenAI Chat Completions provider (HTTP, no SDK).

We talk to ``/v1/chat/completions`` directly with httpx so the gateway has
zero SDK coupling and works against any OpenAI-compatible endpoint
(Together, Groq's OpenAI mode, vLLM, llama.cpp server, ...). The base URL
is configurable: same code, different upstream.
"""

from __future__ import annotations

import os

import httpx

from ..retry import RetryPolicy, call_with_retries, is_retryable_http_status
from .base import Completion, ProviderError

_DEFAULT_BASE_URL = "https://api.openai.com/v1"


class OpenAIProvider:
    name = "openai"

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
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        self._base_url = (
            base_url or os.environ.get("OPENAI_BASE_URL") or _DEFAULT_BASE_URL
        ).rstrip("/")
        self._timeout_s = timeout_s
        self._client = client
        self._retry = retry_policy or RetryPolicy.from_env()

    def complete(self, messages: list[dict]) -> Completion:
        if not messages:
            raise ProviderError("no messages provided")

        payload = {
            "model": self._model,
            "messages": [
                {"role": m["role"], "content": m["content"]}
                for m in messages
                if "role" in m and "content" in m
            ],
            "temperature": 0,
        }
        if not payload["messages"]:
            raise ProviderError("messages must contain role/content entries")

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
            text = data["choices"][0]["message"]["content"] or ""
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


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return is_retryable_http_status(exc.response.status_code)
    return isinstance(exc, httpx.TransportError | httpx.TimeoutException)
