import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from threading import Barrier

import pytest

from personal_harness.costs import CostEntry, TokenPrices, TokenUsage, cost_from_usage
from personal_harness.ledger import BudgetError, Ledger


def charge(id: str, **updates: object) -> CostEntry:
    return CostEntry.model_validate(
        {
            "entry_id": id,
            "category": "model",
            "provider": "openai-compatible",
            "model": "test-chat",
            "currency": "CNY",
            "outcome": "unknown",
            "basis": "unknown",
            "amount": None,
            "source": "fixture",
            **updates,
        }
    )


def start(
    ledger: Ledger, id: str, task: str = "1", daily: str = "1", total: str | None = None
) -> None:
    ledger.start_run(
        id,
        provider="openai-compatible",
        model="test-chat",
        task_limit=Decimal(task),
        daily_limit=Decimal(daily),
        total_limit=Decimal(total) if total is not None else None,
    )


def reserve(ledger: Ledger, run: str, id: str, amount: str) -> None:
    ledger.reserve(run, charge(id), Decimal(amount), price_json='{"source":"test"}')


def billed(id: str, amount: str) -> CostEntry:
    return CostEntry.model_validate(
        {**charge(id).model_dump(), "basis": "billed", "amount": amount, "source": "test invoice"}
    )


def external(id: str, amount: str | None, *, basis: str = "billed") -> CostEntry:
    return CostEntry.model_validate(
        {
            "entry_id": id,
            "category": "external_api",
            "provider": "places-provider",
            "model": None,
            "currency": "CNY",
            "outcome": "succeeded" if amount is not None else "unknown",
            "basis": basis if amount is not None else "unknown",
            "amount": amount,
            "source": "provider invoice",
        }
    )


def test_settlement_releases_only_unused_reservation_and_preserves_decimal(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    start(ledger, "run", task="0.3", daily="0.3")
    reserve(ledger, "run", "one", "0.3")
    ledger.settle(billed("one", "0.1"))
    reserve(ledger, "run", "two", "0.2")
    ledger.settle(billed("two", "0.2"))
    summary = ledger.summary("run")
    assert Decimal(summary["cost"]) == Decimal("0.3")
    assert summary["cost_complete"] and summary["reserved_cost"] == "0"
    with pytest.raises(BudgetError, match="task_budget"):
        reserve(ledger, "run", "three", "0.000001")
    assert ledger.summary("run")["requests"] == 2
    with pytest.raises(ValueError, match="already settled"):
        ledger.settle(billed("one", "0"))


@pytest.mark.parametrize("schema", [1, 2])
def test_readonly_ledger_reads_without_migration_and_refuses_writes(tmp_path, schema):
    path = tmp_path / "ledger.sqlite3"
    ledger = Ledger(path)
    start(ledger, "run")
    reserve(ledger, "run", "one", "0.2")
    ledger.settle(billed("one", "0.01"))
    reserve(ledger, "run", "two", "0.3")
    if schema == 1:
        with sqlite3.connect(path) as db:
            db.execute("ALTER TABLE runs DROP COLUMN total_limit")
            db.execute("PRAGMA user_version=1")
    before = path.read_bytes()
    reader = Ledger(path, read_only=True)
    summary = reader.summary("run")
    assert summary["cost"] is None and summary["verified_subtotal"] == "0.01"
    assert summary["reserved_cost"] == "0.3" and summary["requests"] == 2
    assert summary["total_limit_cny"] is None
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        reader.finish_run("run", "forbidden")
    assert path.read_bytes() == before
    with sqlite3.connect(path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == schema


def test_readonly_ledger_never_creates_a_missing_file_or_repairs_an_unknown_schema(tmp_path):
    missing = tmp_path / "absent" / "ledger.sqlite3"
    with pytest.raises(sqlite3.Error):
        Ledger(missing, read_only=True)
    assert not missing.parent.exists()
    path = tmp_path / "unsupported.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA user_version=99")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Unsupported"):
        Ledger(path, read_only=True)
    assert path.read_bytes() == before


def test_summary_reports_cached_input_price_difference_from_request_snapshot(
    tmp_path: Path,
) -> None:
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    start(ledger, "cached")
    prices = TokenPrices(
        provider="openai-compatible",
        model="test-chat",
        source="configured price",
        recorded_on=datetime(2026, 11, 18, tzinfo=UTC).date(),
        currency="CNY",
        input_per_million="1",
        output_per_million="1",
        cache_read_per_million="0.1",
    )
    usage = TokenUsage(input_tokens=1000, output_tokens=100, cache_read_tokens=400)
    request = charge("cache-hit")
    ledger.reserve("cached", request, Decimal("1"), price_json=prices.model_dump_json())
    ledger.settle(cost_from_usage("cache-hit", prices, usage))

    summary = ledger.summary("cached")
    assert summary["cache_read_tokens"] == 400
    assert summary["cache_savings_complete"] is True
    assert Decimal(summary["cache_savings_cny"]) == Decimal("0.00036")
    assert Decimal(summary["cost"]) == Decimal("0.00074")


@pytest.mark.parametrize("case", ["billed", "no-price", "invalid-price", "wrong-model", "unknown"])
def test_summary_does_not_claim_cache_savings_without_complete_price_or_usage(
    tmp_path: Path, case: str
) -> None:
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    start(ledger, "incomplete", daily="3")
    prices = TokenPrices(
        provider="openai-compatible",
        model="test-chat",
        source="configured price",
        recorded_on=datetime(2026, 11, 18, tzinfo=UTC).date(),
        currency="CNY",
        input_per_million="1",
        output_per_million="1",
        cache_read_per_million=(None if case == "no-price" else "0.1"),
    )
    usage = TokenUsage(input_tokens=1000, output_tokens=100, cache_read_tokens=400)
    ledger.reserve(
        "incomplete",
        charge("missing-cache-price"),
        Decimal("1"),
        price_json=(
            '{"source":"legacy snapshot"}'
            if case == "invalid-price"
            else prices.model_copy(update={"model": "different"}).model_dump_json()
            if case == "wrong-model"
            else prices.model_dump_json()
        ),
    )
    ledger.settle(
        charge(
            "missing-cache-price",
            basis="billed" if case == "billed" else "usage",
            amount="0.0007",
            usage=usage,
        )
        if case != "unknown"
        else charge("missing-cache-price", usage=usage)
    )
    incomplete = ledger.summary("incomplete")
    assert incomplete["cache_read_tokens"] == 400
    assert incomplete["cache_savings_complete"] is False
    assert incomplete["cache_savings_cny"] is None


def test_zero_cache_hits_need_no_cache_price(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    prices = TokenPrices(
        provider="openai-compatible",
        model="test-chat",
        source="test",
        recorded_on=datetime(2026, 11, 18, tzinfo=UTC).date(),
        currency="CNY",
        input_per_million="1",
        output_per_million="1",
    )

    start(ledger, "no-hit")
    ledger.reserve("no-hit", charge("no-cache"), Decimal("1"), price_json=prices.model_dump_json())
    ledger.settle(
        cost_from_usage("no-cache", prices, TokenUsage(input_tokens=1000, output_tokens=100))
    )
    no_hit = ledger.summary("no-hit")
    assert no_hit["cache_read_tokens"] == 0
    assert no_hit["cache_savings_complete"] is True
    assert Decimal(no_hit["cache_savings_cny"]) == 0


def test_unknown_usage_does_not_become_zero_cache_savings(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    start(ledger, "unknown")
    reserve(ledger, "unknown", "lost", "0.1")
    ledger.settle(charge("lost"))
    snapshot = ledger.summary("unknown")
    assert snapshot["cache_read_tokens"] == 0
    assert snapshot["cache_savings_cny"] is None
    assert not snapshot["cache_savings_complete"]


def test_daily_budget_shared_across_runs_and_atomic_under_concurrency(tmp_path: Path) -> None:
    path = tmp_path / "ledger.sqlite3"
    ledger = Ledger(path)
    start(ledger, "a", daily="0.3")
    start(ledger, "b", daily="0.3")
    barrier = Barrier(2)

    def attempt(id: str) -> str:
        other = Ledger(path)
        barrier.wait()
        try:
            reserve(other, id, id, "0.2")
            return "reserved"
        except BudgetError as error:
            return error.reason

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(attempt, ("a", "b")))
    assert sorted(outcomes) == ["daily_budget", "reserved"]
    assert sum(ledger.summary(id)["requests"] for id in ("a", "b")) == 1


def test_unknown_cost_survives_restart_and_midnight_until_verified(tmp_path: Path) -> None:
    path = tmp_path / "ledger.sqlite3"
    now = datetime(2026, 11, 18, 23, 59, tzinfo=UTC)
    ledger = Ledger(path, clock=lambda: now)
    start(ledger, "old")
    reserve(ledger, "old", "lost", "0.3")
    ledger.settle(charge("lost"))
    now += timedelta(days=1)
    reopened = Ledger(path, clock=lambda: now)
    start(reopened, "new")
    with pytest.raises(BudgetError, match="unreconciled_cost"):
        reserve(reopened, "new", "next", "0.3")
    summary = reopened.summary("old")
    assert summary["cost"] is None and summary["reserved_cost"] == "0.3"
    reopened.reconcile("lost", Decimal("0.02"), source="verified online invoice")
    reserve(reopened, "new", "next", "0.3")
    assert reopened.summary("old")["charges"][0]["basis"] == "billed"


def test_retry_unknown_holds_accumulate_within_run_but_block_other_runs(
    tmp_path: Path,
) -> None:
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    start(ledger, "retry-run", task="1", daily="1")
    reserve(ledger, "retry-run", "attempt-1", "0.3")
    ledger.settle(charge("attempt-1", outcome="failed"))
    reserve(ledger, "retry-run", "attempt-2", "0.3")
    start(ledger, "other-run", task="1", daily="1")
    with pytest.raises(BudgetError, match="unreconciled_cost"):
        reserve(ledger, "other-run", "other-attempt", "0.1")
    assert ledger.summary("retry-run")["reserved_cost"] == "0.6"


def test_inflight_reservations_still_count_after_midnight(tmp_path: Path) -> None:
    now = datetime(2026, 11, 18, 23, 59, tzinfo=UTC)
    ledger = Ledger(tmp_path / "ledger.sqlite3", clock=lambda: now)
    start(ledger, "old", daily="0.3")
    reserve(ledger, "old", "inflight", "0.3")
    now += timedelta(minutes=2)
    start(ledger, "new", daily="0.3")
    with pytest.raises(BudgetError, match="daily_budget"):
        reserve(ledger, "new", "next", "0.1")


def test_usage_above_reservation_blocks_new_calls_until_billing_review(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    start(ledger, "run")
    reserve(ledger, "run", "large", "0.1")
    overrun = {
        **billed("large", "0.2").model_dump(),
        "basis": "usage",
        "usage": dict(input_tokens=200000, output_tokens=0),
    }
    ledger.settle(CostEntry.model_validate(overrun))
    assert ledger.summary("run")["reservation_exceeded"]
    with pytest.raises(BudgetError, match="unreconciled_cost"):
        reserve(ledger, "run", "next", "0.1")
    ledger.reconcile("large", Decimal("0.2"), source="reviewed bill and reservation assumption")
    reserve(ledger, "run", "next", "0.1")


def test_settlement_cannot_change_currency_or_model(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    start(ledger, "run")
    reserve(ledger, "run", "one", "0.2")
    for changes in ({"currency": "USD"}, {"model": "test-large"}):
        with pytest.raises(ValueError, match="identity or currency"):
            ledger.settle(
                CostEntry.model_validate({**billed("one", "0.1").model_dump(), **changes})
            )
    assert ledger.summary("run")["cost"] is None


def test_total_budget_survives_days_restarts_and_includes_older_uncapped_runs(tmp_path):
    path = tmp_path / "ledger.sqlite3"
    now = datetime(2026, 11, 18, 23, 59, tzinfo=UTC)
    ledger = Ledger(path, clock=lambda: now)
    start(ledger, "old")
    reserve(ledger, "old", "paid", "0.3")
    ledger.settle(billed("paid", "0.1"))
    ledger.finish_run("old", "completed")
    now += timedelta(days=1)
    ledger = Ledger(path, clock=lambda: now)
    start(ledger, "limited", total="0.3")
    reserve(ledger, "limited", "held", "0.2")
    assert ledger.total_exposure() == Decimal("0.3")
    now += timedelta(days=1)
    ledger = Ledger(path, clock=lambda: now)
    # A new run cannot reuse a past day's in-flight hold or spending.
    start(ledger, "next", total="0.3")
    with pytest.raises(BudgetError, match="total_budget"):
        reserve(ledger, "next", "blocked", "0.000001")
    assert ledger.summary("next")["requests"] == 0
    assert ledger.summary("limited")["total_limit_cny"] == "0.3"
    ledger.settle(billed("held", "0.05"))
    reserve(ledger, "next", "admitted", "0.15")
    assert ledger.total_exposure() == Decimal("0.30")


def test_total_budget_is_atomic_across_connections(tmp_path):
    path = tmp_path / "ledger.sqlite3"
    ledger = Ledger(path)
    for run in ("a", "b"):
        start(ledger, run, total="0.3")
    barrier = Barrier(2)

    def attempt(run):
        connection = Ledger(path)
        barrier.wait()
        try:
            reserve(connection, run, run, "0.2")
            return "reserved"
        except BudgetError as error:
            return error.reason

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(attempt, ("a", "b")))
    assert sorted(outcomes) == ["reserved", "total_budget"]
    assert ledger.total_exposure() == Decimal("0.2")


def test_total_budget_counts_retry_unknown_and_reconciled_overrun(tmp_path):
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    start(ledger, "retry", total="0.3")
    reserve(ledger, "retry", "first", "0.2")
    ledger.settle(charge("first", outcome="failed"))
    with pytest.raises(BudgetError, match="total_budget"):
        reserve(ledger, "retry", "blocked-retry", "0.2")
    assert ledger.summary("retry")["reserved_cost"] == "0.2"
    ledger.reconcile("first", Decimal("0.05"), source="synthetic verified bill")
    reserve(ledger, "retry", "second", "0.25")
    # A verified bill can exceed its hold; record it rather than hiding actual cost.
    ledger.settle(billed("second", "0.30"))
    with pytest.raises(BudgetError, match="total_budget"):
        reserve(ledger, "retry", "after-bill", "0.000001")
    assert ledger.total_exposure() == Decimal("0.35")


def test_v1_migration_preserves_requests_audits_and_run_policy(tmp_path):
    path = tmp_path / "ledger.sqlite3"
    ledger = Ledger(path)
    start(ledger, "old")
    reserve(ledger, "old", "paid", "0.2")
    ledger.reconcile("paid", Decimal("0.1"), source="synthetic invoice")
    ledger.finish_run("old", "completed", metrics={"requests": 1})
    before = ledger.summary("old")
    # Reconstruct the previous schema, including an existing request and audit trail.
    with sqlite3.connect(path) as db:
        db.execute("ALTER TABLE runs DROP COLUMN total_limit")
        db.execute("PRAGMA user_version=1")
        audit = db.execute("SELECT * FROM reconciliations").fetchall()
    migrated = Ledger(path)
    assert migrated.summary("old") == before
    with sqlite3.connect(path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 2
        assert db.execute("SELECT * FROM reconciliations").fetchall() == audit
    assert Ledger(path).summary("old") == before  # Migration is idempotent.
    start(migrated, "new", total="0.2")
    with pytest.raises(BudgetError, match="total_budget"):
        reserve(migrated, "new", "blocked", "0.11")


@pytest.mark.parametrize("limit", ["-1", "NaN", "Infinity"])
def test_invalid_total_limit_creates_no_run(tmp_path, limit):
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    with pytest.raises(ValueError, match="finite and non-negative"):
        start(ledger, "invalid", total=limit)
    assert ledger.recent() == []


def test_zero_total_budget_blocks_positive_reservation(tmp_path):
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    start(ledger, "zero", total="0")
    with pytest.raises(BudgetError, match="total_budget"):
        reserve(ledger, "zero", "blocked", "0.000001")
    assert ledger.total_exposure() == 0


def test_external_charges_are_separate_from_attempts_but_count_toward_budget(tmp_path):
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    start(ledger, "run", task="1", daily="1", total="0.5")
    reserve(ledger, "run", "model", "0.2")
    ledger.settle(billed("model", "0.1"))
    ledger.record_external_charge("run", external("map", "0.2"))

    summary = ledger.summary("run")
    assert summary["requests"] == 1
    assert len(summary["external_charges"]) == 1
    assert summary["external_cost"] == "0.2"
    assert summary["total_cost"] == "0.3"
    assert ledger.total_exposure() == Decimal("0.3")
    with pytest.raises(BudgetError, match="total_budget"):
        reserve(ledger, "run", "next", "0.21")


def test_estimated_external_charge_is_exposed_but_not_complete_total_cost(tmp_path):
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    start(ledger, "run", task="1", daily="1")
    ledger.record_external_charge("run", external("estimate", "0.2", basis="estimated"))
    summary = ledger.summary("run")
    assert summary["external_cost"] == "0.2"
    assert summary["total_cost"] is None
    assert not summary["total_cost_complete"]


def test_unknown_external_charge_blocks_new_model_requests_until_reconciled(tmp_path):
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    start(ledger, "run", task="1", daily="1")
    ledger.record_external_charge("run", external("map-unknown", None))
    with pytest.raises(BudgetError, match="unreconciled_cost"):
        reserve(ledger, "run", "model", "0.1")
    summary = ledger.summary("run")
    assert summary["external_cost"] is None
    assert summary["total_cost"] is None
    ledger.reconcile("map-unknown", Decimal("0.03"), source="provider invoice")
    assert ledger.summary("run")["external_cost"] == "0.03"
    reserve(ledger, "run", "model-after-reconcile", "0.1")


def test_external_reservation_holds_budget_and_settles_actual_cost(tmp_path):
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    start(ledger, "run", task="1", daily="1", total="0.5")
    pending = ledger.reserve_external(
        "run",
        entry_id="maps-call",
        provider="maps",
        maximum_cny=Decimal("0.4"),
        source="provider tariff upper bound",
    )

    summary = ledger.summary("run")
    assert summary["reserved_cost"] == "0.4"
    assert summary["external_cost"] is None
    assert summary["total_cost"] is None
    with pytest.raises(BudgetError, match="total_budget"):
        reserve(ledger, "run", "model-too-soon", "0.2")

    ledger.settle_external(
        CostEntry.model_validate(
            {
                **pending.model_dump(mode="python"),
                "outcome": "succeeded",
                "basis": "usage",
                "amount": "0.1",
                "source": "provider response usage",
            }
        )
    )
    assert ledger.summary("run")["reserved_cost"] == "0"
    reserve(ledger, "run", "model-after-settlement", "0.2")


def test_external_overrun_fails_closed_until_reconciliation(tmp_path):
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    start(ledger, "run", task="1", daily="1", total="1")
    pending = ledger.reserve_external(
        "run",
        entry_id="maps-call",
        provider="maps",
        maximum_cny=Decimal("0.1"),
        source="provider tariff estimate",
    )
    overrun = CostEntry.model_validate(
        {
            **pending.model_dump(mode="python"),
            "outcome": "succeeded",
            "basis": "usage",
            "amount": "0.2",
            "source": "provider response usage",
        }
    )
    ledger.settle_external(overrun)
    with pytest.raises(BudgetError, match="unreconciled_cost"):
        reserve(ledger, "run", "model-blocked", "0.1")
    with pytest.raises(ValueError, match="already settled"):
        ledger.settle_external(overrun)

    ledger.reconcile("maps-call", Decimal("0.2"), source="platform billing confirmed")
    assert ledger.summary("run")["reservation_exceeded"] is False
    reserve(ledger, "run", "model-after-reconciliation", "0.1")


def test_external_estimate_keeps_preflight_hold_until_invoice(tmp_path):
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    start(ledger, "run", total="0.4")
    ledger.reserve_external(
        "run",
        entry_id="estimate",
        provider="places-provider",
        maximum_cny=Decimal("0.3"),
        source="provider tariff",
    )
    ledger.settle_external(external("estimate", "0.1", basis="estimated"))
    assert ledger.total_exposure() == Decimal("0.3")
    assert ledger.summary("run")["total_cost"] is None
    with pytest.raises(BudgetError, match="total_budget"):
        reserve(ledger, "run", "too-early", "0.2")
    ledger.reconcile("estimate", Decimal("0.1"), source="provider invoice")
    reserve(ledger, "run", "after-invoice", "0.2")
