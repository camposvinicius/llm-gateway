"""Google Gemini provider via the Generative Language API (HTTP).

The HTTP API stays stable across SDK churn, so we hit it directly:
``POST /v1beta/models/{model}:generateContent?key=...``. Token usage
comes back under ``usageMetadata`` with ``promptTokenCount`` and
``candidatesTokenCount``.
"""

from __future__ import annotations

import os

import httpx

from ..retry import RetryPolicy, call_with_retries, is_retryable_http_status
from .base import Completion, ProviderError

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

    def complete(self, messages: list[dict]) -> Completion:
        if not messages:
            raise ProviderError("no messages provided")

        contents = []
        system_parts: list[str] = []
        for message in messages:
            role = message.get("role")
            text = message.get("content", "")
            if role == "system":
                system_parts.append(text)
            else:
                contents.append(
                    {
                        "role": "user" if role == "user" else "model",
                        "parts": [{"text": text}],
                    }
                )
        if not contents:
            raise ProviderError("messages must include at least one user/assistant turn")

        payload: dict = {
            "contents": contents,
            "generationConfig": {"temperature": 0},
        }
        if system_parts:
            payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}

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
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            usage = data["usageMetadata"]
            input_tokens = int(usage["promptTokenCount"])
            output_tokens = int(usage.get("candidatesTokenCount", 0))
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderError(
                f"gemini response missing expected fields: {data!r}"
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
                f"{self._base_url}/v1beta/models/{self._model}:generateContent",
                params={"key": self._api_key},
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
