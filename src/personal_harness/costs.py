"""Explicit usage-based costing; no guessed prices, network access or currency conversion."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

Count = Annotated[int, Field(strict=True, ge=0)]
Amount = Annotated[Decimal, Field(ge=0, allow_inf_nan=False)]
Currency = Literal["USD", "CNY"]


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TokenUsage(_Record):
    """Normalized inclusive input: read/write cache tokens are subsets, not additions.

    Reasoning output, if billed as output, must be included in output_tokens by
    the provider adapter. Other modalities require their own cost entries.
    """

    input_tokens: Count
    output_tokens: Count
    cache_read_tokens: Count = 0
    cache_write_tokens: Count = 0

    @model_validator(mode="after")
    def cache_is_subset(self) -> Self:
        if self.cache_read_tokens + self.cache_write_tokens > self.input_tokens:
            raise ValueError("Cache buckets must be disjoint subsets of input_tokens.")
        return self


class TokenPrices(_Record):
    """Original-currency prices per million tokens from a recorded source."""

    provider: Annotated[str, Field(min_length=1)]
    model: Annotated[str, Field(min_length=1)]
    source: Annotated[str, Field(min_length=1)]
    recorded_on: date
    currency: Currency
    input_per_million: Amount
    output_per_million: Amount
    cache_read_per_million: Amount | None = None
    cache_write_per_million: Amount | None = None

    def calculate(self, usage: TokenUsage) -> Decimal:
        total = (
            Decimal(usage.input_tokens - usage.cache_read_tokens - usage.cache_write_tokens)
            * self.input_per_million
            + Decimal(usage.output_tokens) * self.output_per_million
        )
        for tokens, rate in (
            (usage.cache_read_tokens, self.cache_read_per_million),
            (usage.cache_write_tokens, self.cache_write_per_million),
        ):
            if tokens and rate is None:
                raise ValueError("A non-empty cache bucket requires an explicit applicable price.")
            if rate is not None:
                total += Decimal(tokens) * rate
        return total / Decimal(1_000_000)


class CostEntry(_Record):
    """One request/charge, including retries and requests with unknown billing.

    A billing reconciliation replaces the same entry, never appends another
    charge for the same request. Unsent work belongs in run status, not here.
    """

    entry_id: Annotated[str, Field(min_length=1)]
    category: Literal["model", "external_api"]
    provider: Annotated[str, Field(min_length=1)]
    model: str | None = None
    outcome: Literal["succeeded", "failed", "cancelled", "unknown"]
    basis: Literal["usage", "billed", "estimated", "unknown", "offline"]
    currency: Currency
    amount: Amount | None
    source: Annotated[str, Field(min_length=1)]
    usage: TokenUsage | None = None

    @model_validator(mode="after")
    def amount_matches_basis(self) -> Self:
        if self.basis == "unknown" and self.amount is not None:
            raise ValueError("Unknown cost must have amount=null, never a guessed zero.")
        if self.basis != "unknown" and self.amount is None:
            raise ValueError("Known/estimated cost requires amount.")
        if self.basis == "offline" and self.amount != 0:
            raise ValueError("Offline synthetic execution must have zero external API cost.")
        if self.basis == "usage" and self.category == "model" and self.usage is None:
            raise ValueError("Usage-based model cost requires token usage.")
        return self


def cost_from_usage(
    entry_id: str,
    prices: TokenPrices,
    usage: TokenUsage,
    *,
    outcome: Literal["succeeded", "failed", "cancelled", "unknown"] = "succeeded",
) -> CostEntry:
    """Calculate one charge without silently assuming cache token rates."""
    return CostEntry(
        entry_id=entry_id,
        category="model",
        provider=prices.provider,
        model=prices.model,
        outcome=outcome,
        basis="usage",
        currency=prices.currency,
        amount=prices.calculate(usage),
        source=f"{prices.source} (recorded {prices.recorded_on.isoformat()})",
        usage=usage,
    )


def request_reservation(
    prices: TokenPrices, context_window: int, max_output_tokens: int
) -> Decimal:
    """Conservative full-context hold shared by runtime and offline budget inspection."""
    input_rate = max(prices.input_per_million, prices.cache_read_per_million or Decimal(0))
    return (context_window * input_rate + max_output_tokens * prices.output_per_million) / Decimal(
        1_000_000
    )
