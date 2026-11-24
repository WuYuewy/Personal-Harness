from decimal import Decimal

from personal_harness.costs import CostEntry, TokenPrices, TokenUsage, cost_from_usage
from personal_harness.display import ledger_cost_receipt
from personal_harness.ledger import Ledger


def test_cost_receipt_shows_measured_cache_usage_and_marks_price_difference_as_estimate(
    tmp_path,
):
    from datetime import UTC, datetime

    ledger = Ledger(tmp_path / "ledger.sqlite3")
    ledger.start_run(
        "run",
        provider="openai-compatible",
        model="test-chat",
        task_limit=Decimal("1"),
        daily_limit=Decimal("1"),
    )
    prices = TokenPrices(
        provider="openai-compatible",
        model="test-chat",
        source="test rate",
        recorded_on=datetime(2026, 11, 18, tzinfo=UTC).date(),
        currency="CNY",
        input_per_million="1",
        output_per_million="1",
        cache_read_per_million="0.1",
    )
    pending = CostEntry(
        entry_id="cache",
        category="model",
        provider="openai-compatible",
        model="test-chat",
        outcome="unknown",
        basis="unknown",
        currency="CNY",
        amount=None,
        source="pending request",
    )
    ledger.reserve("run", pending, Decimal("1"), price_json=prices.model_dump_json())
    ledger.settle(
        cost_from_usage(
            "cache", prices, TokenUsage(input_tokens=1000, output_tokens=100, cache_read_tokens=400)
        )
    )

    receipt = ledger_cost_receipt(ledger.summary("run"))
    assert "Provider-reported cache-read input: 400 tokens" in receipt
    assert "provider cache price difference" in receipt
    assert "not a Personal Harness saving" in receipt
    assert "CNY 0.00036 vs ordinary input at configured prices; not bill-verified" in receipt


def test_receipt_tracks_unknown_to_verified_billing_without_inventing_usage(tmp_path):
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    ledger.start_run(
        "run",
        provider="openai-compatible",
        model="test-chat",
        task_limit=Decimal("1"),
        daily_limit=Decimal("1"),
    )

    def pending(entry_id):
        return CostEntry(
            entry_id=entry_id,
            category="model",
            provider="openai-compatible",
            model="test-chat",
            outcome="unknown",
            basis="unknown",
            currency="CNY",
            amount=None,
            source="test pending request",
        )

    ledger.reserve("run", pending("failed"), Decimal("0.1"), price_json="{}")
    ledger.settle(pending("failed").model_copy(update={"outcome": "failed"}))
    ledger.reserve("run", pending("success"), Decimal("0.1"), price_json="{}")
    ledger.settle(
        CostEntry(
            entry_id="success",
            category="model",
            provider="openai-compatible",
            model="test-chat",
            outcome="succeeded",
            basis="usage",
            currency="CNY",
            amount=Decimal("0.00000001"),
            source="test prices",
            usage=TokenUsage(input_tokens=1, output_tokens=0),
        )
    )
    ledger.finish_run("run", "completed")
    before = ledger.summary("run")
    receipt = ledger_cost_receipt(before)
    assert "Cost: CNY unknown" in receipt
    assert "Known subtotal: CNY 0.00000001" in receipt
    assert "Unknown charges: 1" in receipt
    assert "Budget hold: CNY 0.1 (not a billed amount)" in receipt
    assert "Cost basis: 1 usage-priced, 1 unknown" in receipt
    assert "Attempts: 2; usage complete: no" in receipt
    assert ledger.summary("run") == before
    assert before["cost"] is None and before["charges"][0]["amount"] is None

    ledger.reconcile("failed", Decimal("0.0003"), source="test invoice")
    after = ledger.summary("run")
    reconciled = ledger_cost_receipt(after)
    assert "Cost: CNY 0.00030001" in reconciled
    assert "Cost basis: 1 usage-priced, 1 bill-verified" in reconciled
    assert "Attempts: 2; usage complete: no" in reconciled
    assert "Budget hold" not in reconciled and "Unknown charges" not in reconciled
    assert "unknown" not in reconciled
    assert after["charges"][0]["usage"] is None
    assert ledger.summary("run") == after


def test_receipt_separates_external_api_cost_from_model_attempts(tmp_path):
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    ledger.start_run(
        "run",
        provider="openai-compatible",
        model="test-chat",
        task_limit=Decimal("1"),
        daily_limit=Decimal("1"),
    )
    ledger.record_external_charge(
        "run",
        CostEntry(
            entry_id="places",
            category="external_api",
            provider="places-provider",
            model=None,
            outcome="succeeded",
            basis="billed",
            currency="CNY",
            amount=Decimal("0.02"),
            source="provider invoice",
        ),
    )
    receipt = ledger_cost_receipt(ledger.summary("run"))
    assert "Attempts: 0; usage complete: yes" in receipt
    assert "External API charges: CNY 0.02 (1 recorded" in receipt
    assert "Total task cost including external APIs: CNY 0.02" in receipt
