"""Usage ledger: every request the gateway serves is recorded in SQLite.

Design notes that matter for a billing-adjacent system:

- **Append-only.** Rows are inserted, never updated or deleted. A ledger
  you can rewrite is not a ledger; corrections are new rows (status
  "adjustment") so history stays auditable.
- **Costs are integer micro-USD** (see ``pricing.py``) — summing any
  subset of rows is exact integer arithmetic.
- **Failures are recorded too**, with the provider chain that was tried.
  "What did we attempt and how often did fallback fire?" is an
  operational question the ledger should answer, not just billing.
- SQLite because the project is single-node by design; the schema is
  plain SQL and would port to Postgres unchanged.
"""

from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id            TEXT PRIMARY KEY,
    ts_epoch_ms   INTEGER NOT NULL,
    status        TEXT NOT NULL CHECK (status IN ('ok', 'error')),
    provider      TEXT,
    model         TEXT,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    cost_micro_usd INTEGER,
    latency_ms    INTEGER NOT NULL,
    providers_tried TEXT NOT NULL,
    error         TEXT
);
CREATE INDEX IF NOT EXISTS idx_requests_ts ON requests (ts_epoch_ms);
CREATE INDEX IF NOT EXISTS idx_requests_model ON requests (model);
"""


@dataclass(frozen=True)
class LedgerEntry:
    id: str
    ts_epoch_ms: int
    status: str
    provider: str | None
    model: str | None
    input_tokens: int | None
    output_tokens: int | None
    cost_micro_usd: int | None
    latency_ms: int
    providers_tried: str
    error: str | None


class Ledger:
    def __init__(self, path: Path | str):
        # FastAPI's TestClient and many ASGI servers can execute request
        # handlers from threads different from the startup thread. The lock
        # keeps writes serialized while allowing one shared SQLite connection
        # for this single-process demo gateway.
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def record_success(
        self,
        *,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_micro_usd: int,
        latency_ms: int,
        providers_tried: list[str],
    ) -> str:
        return self._insert(
            status="ok",
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_micro_usd=cost_micro_usd,
            latency_ms=latency_ms,
            providers_tried=providers_tried,
            error=None,
        )

    def record_failure(
        self, *, latency_ms: int, providers_tried: list[str], error: str
    ) -> str:
        return self._insert(
            status="error",
            provider=None,
            model=None,
            input_tokens=None,
            output_tokens=None,
            cost_micro_usd=None,
            latency_ms=latency_ms,
            providers_tried=providers_tried,
            error=error,
        )

    def _insert(self, **fields) -> str:
        request_id = str(uuid.uuid4())
        with self._lock:
            self._conn.execute(
                "INSERT INTO requests (id, ts_epoch_ms, status, provider, model, input_tokens,"
                " output_tokens, cost_micro_usd, latency_ms, providers_tried, error)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    request_id,
                    int(time.time() * 1000),
                    fields["status"],
                    fields["provider"],
                    fields["model"],
                    fields["input_tokens"],
                    fields["output_tokens"],
                    fields["cost_micro_usd"],
                    fields["latency_ms"],
                    ",".join(fields["providers_tried"]),
                    fields["error"],
                ),
            )
            self._conn.commit()
        return request_id

    def entries(self) -> list[LedgerEntry]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, ts_epoch_ms, status, provider, model, input_tokens, output_tokens,"
                " cost_micro_usd, latency_ms, providers_tried, error"
                " FROM requests ORDER BY ts_epoch_ms"
            ).fetchall()
        return [LedgerEntry(*row) for row in rows]

    def summary_by_model(self) -> list[dict]:
        """Aggregate cost/usage per model — the input for `gateway-report`."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT model, COUNT(*), SUM(input_tokens), SUM(output_tokens),"
                " SUM(cost_micro_usd), AVG(latency_ms)"
                " FROM requests WHERE status = 'ok' GROUP BY model"
                " ORDER BY SUM(cost_micro_usd) DESC"
            ).fetchall()
        return [
            {
                "model": model,
                "requests": count,
                "input_tokens": in_tok or 0,
                "output_tokens": out_tok or 0,
                "cost_micro_usd": cost or 0,
                "avg_latency_ms": round(avg_lat or 0),
            }
            for model, count, in_tok, out_tok, cost, avg_lat in rows
        ]

    def failure_count(self) -> int:
        with self._lock:
            (count,) = self._conn.execute(
                "SELECT COUNT(*) FROM requests WHERE status = 'error'"
            ).fetchone()
        return count
