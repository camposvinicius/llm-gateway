"""Provider-level retry policy.

Two important boundaries:

- This retry runs **inside** a provider call. The router's fallback chain
  runs **across** providers. They are different layers: a provider
  shouldn't be marked dead because a single 429 came in; conversely, a
  whole provider that's having a regional outage should not retry forever.
- We retry only "retryable" failures: network errors and HTTP 408/429/5xx.
  4xx other than 408/429 are caller bugs and are surfaced immediately.

The default policy is conservative: 3 attempts, exponential backoff with
full jitter, capped at 8s. Operators can tune via env vars without code.
"""

from __future__ import annotations

import os
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


_RETRYABLE_HTTP_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay_seconds: float = 0.25
    max_delay_seconds: float = 8.0
    sleep: Callable[[float], None] = time.sleep

    @classmethod
    def from_env(cls) -> RetryPolicy:
        return cls(
            max_attempts=int(os.environ.get("GATEWAY_RETRY_MAX_ATTEMPTS", "3")),
            base_delay_seconds=float(os.environ.get("GATEWAY_RETRY_BASE_SECONDS", "0.25")),
            max_delay_seconds=float(os.environ.get("GATEWAY_RETRY_MAX_SECONDS", "8.0")),
        )

    def backoff(self, attempt: int) -> float:
        """Full-jitter exponential backoff (AWS Architecture Blog pattern).

        cap   = min(max_delay, base * 2**(attempt-1))
        sleep = uniform(0, cap)
        """
        if attempt < 1:
            return 0.0
        cap = min(self.max_delay_seconds, self.base_delay_seconds * (2 ** (attempt - 1)))
        return random.uniform(0, cap)


def is_retryable_http_status(status: int) -> bool:
    return status in _RETRYABLE_HTTP_STATUS


def call_with_retries(
    operation: Callable[[], T],
    *,
    is_retryable: Callable[[Exception], bool],
    policy: RetryPolicy,
) -> T:
    """Run ``operation`` with retry/backoff for retryable exceptions.

    Returns the first successful result. If every attempt fails, the last
    exception bubbles up — the caller decides how to surface it (the
    BedrockProvider, OpenAIProvider, etc. wrap it as ProviderError).
    """
    if policy.max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    last_exception: Exception | None = None
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return operation()
        except Exception as exc:
            last_exception = exc
            if attempt == policy.max_attempts or not is_retryable(exc):
                raise
            policy.sleep(policy.backoff(attempt))

    # Defensive: unreachable because the loop either returns or re-raises.
    assert last_exception is not None
    raise last_exception
