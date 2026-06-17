"""Echo provider: deterministic, offline, free.

It exists for the same reason rag-evals defaults to BM25 — the gateway's
own machinery (routing, metering, ledger, telemetry, API) must be testable
in CI with no credentials and no cost. It echoes the last user message and
reports word counts as token usage, so every number downstream is
hand-checkable.
"""

from __future__ import annotations

from ..messages import text_of
from .base import Completion, ProviderError


class EchoProvider:
    name = "echo"

    def __init__(self, model: str, fail: bool = False):
        if not model:
            raise ValueError("model label is required")
        self._model = model
        self._fail = fail  # used in tests to exercise fallback

    def complete(self, messages: list[dict], tools: list[dict] | None = None) -> Completion:
        # Echo never calls tools; it accepts the argument so it stays a drop-in
        # Provider for the tool-aware router and agent loop.
        if self._fail:
            raise ProviderError("echo provider configured to fail")
        if not messages:
            raise ProviderError("no messages provided")

        last_user = next(
            (text_of(m["content"]) for m in reversed(messages) if m.get("role") == "user"), None
        )
        if last_user is None:
            raise ProviderError("no user message found")

        text = f"echo: {last_user}"
        return Completion(
            text=text,
            model=self._model,
            input_tokens=sum(len(text_of(m.get("content", "")).split()) for m in messages),
            output_tokens=len(text.split()),
        )
