"""CLI tests for gateway-report."""

from click.testing import CliRunner

from gateway.cli import main
from gateway.ledger import Ledger


def test_summary_for_empty_ledger(tmp_path):
    db_path = tmp_path / "gateway.db"
    ledger = Ledger(db_path)
    ledger.close()

    result = CliRunner().invoke(main, ["summary", "--ledger", str(db_path)])

    assert result.exit_code == 0
    assert "No successful requests recorded" in result.output


def test_summary_groups_by_model_and_formats_usd(tmp_path):
    db_path = tmp_path / "gateway.db"
    ledger = Ledger(db_path)
    ledger.record_success(
        provider="echo",
        model="echo-1",
        input_tokens=1000,
        output_tokens=100,
        cost_micro_usd=4500,
        latency_ms=10,
        providers_tried=["echo"],
    )
    ledger.record_success(
        provider="echo",
        model="echo-1",
        input_tokens=1000,
        output_tokens=100,
        cost_micro_usd=4500,
        latency_ms=20,
        providers_tried=["echo"],
    )
    ledger.close()

    result = CliRunner().invoke(main, ["summary", "--ledger", str(db_path)])

    assert result.exit_code == 0
    assert "echo-1 | 2 | 2000 | 200 | $0.009000 | 15" in result.output


def test_summary_reports_failures(tmp_path):
    db_path = tmp_path / "gateway.db"
    ledger = Ledger(db_path)
    ledger.record_failure(latency_ms=3, providers_tried=["echo"], error="boom")
    ledger.close()

    result = CliRunner().invoke(main, ["summary", "--ledger", str(db_path)])

    assert result.exit_code == 0
    assert "Failures: 1" in result.output
