"""Command-line reporting over the SQLite usage ledger."""

from __future__ import annotations

from pathlib import Path

import click

from .ledger import Ledger
from .pricing import format_usd


@click.group()
def main() -> None:
    """Inspect llm-gateway ledgers."""


@main.command()
@click.option("--ledger", type=click.Path(exists=True, path_type=Path), required=True)
def summary(ledger: Path) -> None:
    """Print cost/usage summary grouped by model."""
    db = Ledger(ledger)
    rows = db.summary_by_model()
    failures = db.failure_count()
    db.close()

    if not rows:
        click.echo("No successful requests recorded.")
        if failures:
            click.echo(f"Failures: {failures}")
        return

    click.echo("Model usage summary")
    click.echo("model | requests | input_tokens | output_tokens | cost | avg_latency_ms")
    click.echo("--- | ---: | ---: | ---: | ---: | ---:")
    for row in rows:
        click.echo(
            f"{row['model']} | {row['requests']} | {row['input_tokens']} | "
            f"{row['output_tokens']} | {format_usd(row['cost_micro_usd'])} | "
            f"{row['avg_latency_ms']}"
        )
    if failures:
        click.echo(f"\nFailures: {failures}")
