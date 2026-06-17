"""Tool abstraction for the research agent.

A tool is a name + description + JSON Schema (so any provider can advertise it)
plus a ``run(args) -> str`` callable the agent invokes when the model asks for
it. Keeping tools behind this tiny shape means the agent loop never special-cases
a provider or a specific tool.

Tool implementations return a string (what the model sees as the tool result)
and turn their own failures into an error string rather than raising — a failed
tool should let the model recover, not abort the whole request.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import httpx

from ..retry import RetryPolicy, call_with_retries, is_retryable_http_status

DEFAULT_TIMEOUT_S = 12.0


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_schema: dict
    run: Callable[[dict], str]

    def definition(self) -> dict:
        """Canonical tool definition the gateway sends to a provider."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return is_retryable_http_status(exc.response.status_code)
    return isinstance(exc, httpx.TransportError | httpx.TimeoutException)


def request_json(method: str, url: str, *, timeout_s: float = DEFAULT_TIMEOUT_S, **kwargs):
    def _do():
        with httpx.Client(timeout=timeout_s) as client:
            resp = client.request(method, url, **kwargs)
            resp.raise_for_status()
            return resp.json()

    return call_with_retries(_do, is_retryable=_is_retryable, policy=RetryPolicy.from_env())


def request_text(method: str, url: str, *, timeout_s: float = DEFAULT_TIMEOUT_S, **kwargs) -> str:
    def _do():
        with httpx.Client(timeout=timeout_s) as client:
            resp = client.request(method, url, **kwargs)
            resp.raise_for_status()
            return resp.text

    return call_with_retries(_do, is_retryable=_is_retryable, policy=RetryPolicy.from_env())
