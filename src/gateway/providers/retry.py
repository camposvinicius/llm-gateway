"""Cross-provider retry helpers.

Providers wrap their own transient errors so the router can keep going.
Within a single provider we still want a few quick retries with
exponential backoff and jitter — provider-side flakiness is more common
than total outage, and falling through to the next provider too eagerly
trades cost stability for tail latency.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import TypeVar

from .base import ProviderError

T = TypeVar("T")


def with_retries(
    operation: Callable[[], T],
    *,
    attempts: int = 3,
    base_delay_s: float = 0.2,
    max_delay_s: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
) -> T:
    """Run ``operation``, retrying on ProviderError up to ``attempts`` times.

    Backoff: ``base_delay_s * 2**(attempt-1)`` capped at ``max_delay_s``,
    with full jitter (uniform 0..delay). Jitter is what stops a thundering
    herd of clients all retrying in lockstep.
    """
    if attempts <= 0:
        raise ValueError("attempts must be positive")
    if base_delay_s < 0 or max_delay_s < base_delay_s:
        raise ValueError("invalid delay configuration")

    rng = rng or random.Random()
    last_error: ProviderError | None = None
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except ProviderError as exc:
            last_error = exc
            if attempt == attempts:
                break
            delay = min(base_delay_s * (2 ** (attempt - 1)), max_delay_s)
            sleep(rng.uniform(0.0, delay))
    assert last_error is not None
    raise last_error
