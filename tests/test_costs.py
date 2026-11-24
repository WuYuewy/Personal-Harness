from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from personal_harness.costs import CostEntry, TokenPrices, TokenUsage, cost_from_usage
from personal_harness.openai_adapter import normalize_usage
from tests.mock_provider import MODELS, prices_for


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("test-large", "0.00148"),
        ("test-large-2", "0.00148"),
        ("test-chat", "0.00074"),
        ("test-reasoner", "0.00074"),
    ],
)
def test_online_rates_do_not_double_charge_cached_input(model: str, expected: str) -> None:
    prices = prices_for(model)
    assert prices is not None and prices.currency == "CNY"
    assert prices.recorded_on == date(2026, 11, 18)
    usage = TokenUsage(input_tokens=1000, cache_read_tokens=400, output_tokens=100)
    charge = cost_from_usage("request-1", prices, usage, outcome="failed")
    assert charge.amount == Decimal(expected)
    assert charge.outcome == "failed" and charge.basis == "usage"
    assert CostEntry.model_validate_json(charge.model_dump_json()) == charge
    assert set(MODELS) == {"test-chat", "test-reasoner", "test-large", "test-large-2"}


def test_unknown_model_has_no_invented_price() -> None:
    assert prices_for("unlisted") is None


def test_hello_price() -> None:
    prices = prices_for("test-chat")
    assert prices is not None
    assert prices.calculate(TokenUsage(input_tokens=13, output_tokens=14)) == Decimal("0.000027")


def test_cache_write_requires_its_own_rate() -> None:
    prices = TokenPrices(
        provider="test",
        model="test",
        currency="USD",
        source="fixture",
        recorded_on=date(2026, 11, 18),
        input_per_million="2",
        output_per_million="4",
    )
    usage = TokenUsage(input_tokens=100, output_tokens=10, cache_write_tokens=40)
    with pytest.raises(ValueError, match="explicit applicable price"):
        prices.calculate(usage)
    prices = prices.model_copy(update={"cache_write_per_million": Decimal("3")})
    assert prices.calculate(usage) == Decimal("0.00028")


@pytest.mark.parametrize(
    "changes",
    [
        {"input_tokens": True},
        {"output_tokens": -1},
        {"cache_read_tokens": 101},
        {"cache_read_tokens": 60, "cache_write_tokens": 50},
    ],
)
def test_invalid_token_buckets_are_rejected(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        TokenUsage.model_validate({"input_tokens": 100, "output_tokens": 0, **changes})


@pytest.mark.parametrize(
    "basis,amount", [("unknown", "0"), ("billed", None), ("offline", "1"), ("estimated", "-1")]
)
def test_cost_basis_cannot_hide_unknown_or_fabricate_free_calls(basis: str, amount: str) -> None:
    with pytest.raises(ValidationError):
        CostEntry.model_validate(
            dict(
                entry_id="x",
                category="model",
                provider="test",
                model="test",
                outcome="failed",
                basis=basis,
                currency="CNY",
                amount=amount,
                source="fixture",
            )
        )


@pytest.mark.parametrize(
    "raw",
    [
        None,
        {},
        {"prompt_tokens": 10, "completion_tokens": 2},
        {"prompt_tokens": True, "completion_tokens": 2, "prompt_cache_hit_tokens": 0},
        {"prompt_tokens": 10, "completion_tokens": 2, "prompt_cache_hit_tokens": 11},
        {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "prompt_cache_hit_tokens": 4,
            "prompt_cache_miss_tokens": 7,
        },
        {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "prompt_cache_hit_tokens": 4,
            "prompt_tokens_details": {"cached_tokens": 3},
        },
        {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "prompt_cache_hit_tokens": 0,
            "cache_creation_input_tokens": 2,
        },
    ],
)
def test_missing_or_inconsistent_usage_stays_unknown(raw: dict[str, object] | None) -> None:
    assert normalize_usage(raw) is None


def test_deepseek_and_standard_cache_usage() -> None:
    base = {"prompt_tokens": 100, "completion_tokens": 20}
    direct = normalize_usage(
        {**base, "prompt_cache_hit_tokens": 30, "prompt_cache_miss_tokens": 70}
    )
    standard = normalize_usage(
        {
            **base,
            "prompt_tokens_details": {"cached_tokens": 30},
            "completion_tokens_details": {"reasoning_tokens": 10},
        }
    )
    assert (
        direct == standard == TokenUsage(input_tokens=100, output_tokens=20, cache_read_tokens=30)
    )
