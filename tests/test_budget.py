import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from typer.testing import CliRunner

from personal_harness.budget import budget_status
from personal_harness.cli import app
from personal_harness.config import load_config
from personal_harness.costs import CostEntry
from personal_harness.ledger import BudgetError, Ledger
from tests import mock_provider


def _online_config(tmp_path: Path, model: str = "test-chat") -> Path:

    path = tmp_path / "online.toml"
    path.write_text(mock_provider.template(model), "utf-8")
    return path


def _unknown(identifier: str) -> CostEntry:
    return CostEntry(
        entry_id=identifier,
        category="model",
        provider="openai-compatible",
        model="test-chat",
        outcome="failed",
        basis="unknown",
        currency="CNY",
        amount=None,
        source="synthetic",
    )


def _external(identifier: str, amount: str | None) -> CostEntry:
    return CostEntry(
        entry_id=identifier,
        category="external_api",
        provider="places-provider",
        model=None,
        outcome="succeeded" if amount is not None else "unknown",
        basis="billed" if amount is not None else "unknown",
        currency="CNY",
        amount=amount,
        source="synthetic external bill",
    )


def _run(ledger: Ledger, run_id: str, *, daily: str = "10", total: str | None = None) -> None:
    ledger.start_run(
        run_id,
        provider="openai-compatible",
        model="test-chat",
        task_limit=Decimal("1"),
        daily_limit=Decimal(daily),
        total_limit=Decimal(total) if total is not None else None,
    )


def test_budget_without_ledger_is_read_only_and_uses_runtime_reservation(tmp_path: Path) -> None:
    config = _online_config(tmp_path)
    result = budget_status(load_config(config), now=datetime(2026, 11, 19, 7, tzinfo=UTC))
    assert result["reservation_cny"] == "0.262656"
    assert result["daily"]["remaining_cny"] == "2.00"
    assert result["new_task_request_fits"]
    assert not result["ledger_exists"]
    assert not (tmp_path / ".personal-harness").exists()


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("test-chat", "0.262656"),
        ("test-reasoner", "0.262656"),
        ("test-large", "2.001024"),
    ],
)
def test_budget_reservation_matches_recorded_model_rates(
    tmp_path: Path, model: str, expected: str
) -> None:
    result = budget_status(load_config(_online_config(tmp_path, model)))
    assert result["reservation_cny"] == expected


def test_unknown_old_request_blocks_and_snapshot_does_not_modify_ledger(tmp_path: Path) -> None:
    config = _online_config(tmp_path)
    path = tmp_path / ".personal-harness" / "ledger.sqlite3"
    ledger = Ledger(path, clock=lambda: datetime(2026, 11, 18, 12, tzinfo=UTC))
    _run(ledger, "old")
    ledger.reserve("old", _unknown("unknown-1"), Decimal("0.262656"), price_json="{}")
    ledger.settle(_unknown("unknown-1"))
    before = hashlib.sha256(path.read_bytes()).hexdigest()

    result = budget_status(load_config(config), now=datetime(2026, 11, 19, 1, tzinfo=UTC))

    assert not result["new_task_request_fits"]
    assert result["reason"] == "unreconciled_cost"
    assert result["daily"]["unknown_requests"] == 0
    assert result["total"]["unknown_requests"] == 1
    assert result["unresolved_runs"] == ["old"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_inflight_old_day_request_still_uses_daily_budget(tmp_path: Path) -> None:
    config = _online_config(tmp_path)
    config.write_text(
        config.read_text("utf-8").replace('max_daily_cny = "2.00"', 'max_daily_cny = "0.30"'),
        "utf-8",
    )
    path = tmp_path / ".personal-harness" / "ledger.sqlite3"
    ledger = Ledger(path, clock=lambda: datetime(2026, 11, 18, 23, 59, tzinfo=UTC))
    _run(ledger, "old", daily="0.3")
    ledger.reserve("old", _unknown("pending"), Decimal("0.3"), price_json="{}")
    result = budget_status(load_config(config), now=datetime(2026, 11, 19, 1, tzinfo=UTC))
    assert result["reason"] == "daily_budget"
    assert result["daily"]["inflight_requests"] == 1
    assert result["daily"]["held_cny"] == "0.3"


def test_external_charge_is_visible_to_readonly_budget_snapshot(tmp_path: Path) -> None:
    config = _online_config(tmp_path)
    path = tmp_path / ".personal-harness" / "ledger.sqlite3"
    ledger = Ledger(path, clock=lambda: datetime(2026, 11, 19, 1, tzinfo=UTC))
    _run(ledger, "places", daily="1", total="0.3")
    ledger.record_external_charge("places", _external("map", "0.2"))

    result = budget_status(load_config(config), now=datetime(2026, 11, 19, 1, tzinfo=UTC))
    assert result["total"]["exposure_cny"] == "0.2"
    assert result["total"]["external_charges"] == 1
    assert result["new_task_request_fits"]

    ledger.record_external_charge("places", _external("map-unknown", None))
    blocked = budget_status(load_config(config), now=datetime(2026, 11, 19, 1, tzinfo=UTC))
    assert blocked["reason"] == "unreconciled_cost"
    assert blocked["total"]["unknown_requests"] == 1


def test_external_spend_and_overnight_hold_share_actual_daily_admission(tmp_path: Path) -> None:
    config = _online_config(tmp_path)
    path = tmp_path / ".personal-harness" / "ledger.sqlite3"
    now = datetime(2026, 11, 18, 23, 59, tzinfo=UTC)
    ledger = Ledger(path, clock=lambda: now)
    _run(ledger, "old", daily="2")
    ledger.reserve_external(
        "old",
        entry_id="inflight",
        provider="places-provider",
        maximum_cny=Decimal("0.9"),
        source="fixed fee bound",
    )
    now = datetime(2026, 11, 19, 0, 1, tzinfo=UTC)
    _run(ledger, "earlier-today", daily="2")
    ledger.record_external_charge("earlier-today", _external("today", "0.9"))
    _run(ledger, "new", daily="2")
    result = budget_status(load_config(config), now=now)
    assert result["reason"] == "daily_budget"
    assert result["daily"]["exposure_cny"] == "1.8"
    with pytest.raises(BudgetError, match="daily_budget"):
        ledger.reserve("new", _unknown("model"), Decimal("0.262656"), price_json="{}")
    with pytest.raises(BudgetError, match="daily_budget"):
        ledger.reserve_external(
            "new",
            entry_id="another",
            provider="places-provider",
            maximum_cny=Decimal("0.3"),
            source="fixed fee bound",
        )


def test_unknown_schema_and_corrupt_records_fail_closed(tmp_path: Path) -> None:
    config = _online_config(tmp_path)
    path = tmp_path / ".personal-harness" / "ledger.sqlite3"
    path.parent.mkdir()
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA user_version=99")
    with pytest.raises(ValueError, match="Cannot inspect budget ledger"):
        budget_status(load_config(config))


def test_cli_budget_json_text_offline_and_exit_codes(tmp_path: Path) -> None:
    runner = CliRunner()
    online = _online_config(tmp_path)
    result = runner.invoke(app, ["budget", "-c", str(online), "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["reservation_cny"] == "0.262656"
    assert not (tmp_path / ".personal-harness").exists()
    text = runner.invoke(app, ["budget", "-c", str(online)])
    assert "One request hold" in text.stdout and "Read-only snapshot" in text.stdout

    offline = tmp_path / "offline.toml"
    from importlib.resources import files

    offline.write_text(
        files("personal_harness").joinpath("templates/personal.toml").read_text("utf-8"),
        "utf-8",
    )
    assert runner.invoke(app, ["budget", "-c", str(offline)]).exit_code == 0

    limited = online.read_text("utf-8").replace('max_task_cny = "0.80"', 'max_task_cny = "0.10"')
    online.write_text(limited, "utf-8")
    blocked = runner.invoke(app, ["budget", "-c", str(online), "--json"])
    assert blocked.exit_code == 1
    assert json.loads(blocked.stdout)["reason"] == "task_budget"


def test_cli_budget_compares_profiles_from_one_shared_ledger_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import personal_harness.budget as budget_module

    runner = CliRunner()
    online = _online_config(tmp_path)
    online.write_text(
        online.read_text("utf-8")
        + '\n[profiles.flash.model]\nmodel = "test-large"\n'
        + 'context_window = 1000000\ninput_per_million_cny = "2"\noutput_per_million_cny = "2"\n'
        + 'cache_read_per_million_cny = "0.2"\n'
        + '[profiles.flash.budget]\nmax_task_cny = "3.00"\nmax_daily_cny = "5.00"\n',
        "utf-8",
    )
    original_snapshot = budget_module._request_snapshot
    snapshots_read = 0

    def count_snapshot(path: Path):
        nonlocal snapshots_read
        snapshots_read += 1
        return original_snapshot(path)

    monkeypatch.setattr(budget_module, "_request_snapshot", count_snapshot)

    result = runner.invoke(app, ["budget", "--all-profiles", "-c", str(online), "--json"])
    assert result.exit_code == 0, result.output
    profiles = json.loads(result.stdout)["profiles"]
    assert [item["profile"] for item in profiles] == ["cowork", "flash"]
    assert [item["model"] for item in profiles] == ["test-chat", "test-large"]
    assert [item["reservation_cny"] for item in profiles] == ["0.262656", "2.001024"]
    assert snapshots_read == 1

    conflict = runner.invoke(app, ["budget", "-p", "cowork", "--all-profiles", "-c", str(online)])
    assert conflict.exit_code == 2
