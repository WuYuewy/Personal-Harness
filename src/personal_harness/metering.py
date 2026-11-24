"""Internal per-request cost wrapper; PydanticAI still owns the tool loop."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any
from uuid import uuid4

from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelConnectionError, ModelTimeoutError, UsageLimitExceeded
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage

from personal_harness.context import CURRENT_METRICS
from personal_harness.costs import (
    CostEntry,
    TokenPrices,
    TokenUsage,
    cost_from_usage,
    request_reservation,
)
from personal_harness.ledger import BudgetError, Ledger


class ProviderRequestError(RuntimeError):
    def __init__(self, reason: str, status_code: int | None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.status_code = status_code


def _retryable(error: BaseException) -> tuple[bool, int | None]:
    status = getattr(error, "status_code", None)
    if status in {408, 429, 500, 502, 503, 504}:
        return True, int(status)
    name = type(error).__name__
    if isinstance(error, ModelConnectionError) or name in {
        "APIConnectionError",
        "APITimeoutError",
    }:
        return True, status if isinstance(status, int) else None
    return False, status if isinstance(status, int) else None


def _retry_delay(error: BaseException, attempt: int) -> float:
    retry_after = getattr(error, "retry_after", None)
    if isinstance(retry_after, (int, float)) and retry_after > 0:
        return min(float(retry_after), 5.0)
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    value = headers.get("retry-after") if headers is not None else None
    if value:
        try:
            delay = float(value)
        except ValueError:
            try:
                delay = (parsedate_to_datetime(value) - datetime.now(UTC)).total_seconds()
            except (TypeError, ValueError, OverflowError):
                delay = 0
        if delay > 0:
            return float(min(delay, 5.0))
    return float(min(0.5 * (2**attempt), 2.0))


class MeteredModel(WrapperModel):
    def __init__(
        self,
        wrapped: Model,
        ledger: Ledger,
        prices: TokenPrices,
        context_window: int,
        max_output_tokens: int,
        provider_retries: int,
        max_requests: int | None = None,
    ) -> None:
        super().__init__(wrapped)
        self.ledger = ledger
        self.prices = prices
        # Full-price maximum context, not a guessed tokenizer estimate or assumed cache hit.
        self.reservation = request_reservation(prices, context_window, max_output_tokens)
        self.run_id: ContextVar[str] = ContextVar("personal_harness_run_id")
        self.provider_retries = provider_retries
        self.max_requests = max_requests

    def _reserve_attempt(self) -> CostEntry:
        metrics = CURRENT_METRICS.get()
        if (
            metrics is not None
            and self.max_requests is not None
            and metrics.physical_requests >= self.max_requests
        ):
            raise UsageLimitExceeded("Physical model request limit reached.")
        charge = CostEntry(
            entry_id=uuid4().hex,
            category="model",
            provider=self.prices.provider,
            model=self.prices.model,
            outcome="unknown",
            basis="unknown",
            currency="CNY",
            amount=None,
            source="Pending model API attempt",
        )
        self.ledger.reserve(
            self.run_id.get(), charge, self.reservation, price_json=self.prices.model_dump_json()
        )
        if metrics is not None:
            metrics.physical_requests += 1
        return charge

    def _settle_usage(self, charge: CostEntry, usage: RequestUsage) -> None:
        if usage.details.get("personal_harness_usage_valid") != 1:
            self.ledger.settle(charge.model_copy(update={"outcome": "succeeded"}))
            raise BudgetError("unknown_usage")
        normalized = TokenUsage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cache_write_tokens=usage.cache_write_tokens,
        )
        settled = cost_from_usage(charge.entry_id, self.prices, normalized)
        self.ledger.settle(settled)
        if settled.amount is not None and settled.amount > self.reservation:
            raise BudgetError("reservation_exceeded")

    async def _after_failed_attempt(
        self, charge: CostEntry, error: BaseException, attempt: int, *, opened: bool = False
    ) -> None:
        # A failed send may still be billed; preserve its hold as an unknown charge.
        cancelled = isinstance(error, (asyncio.CancelledError, GeneratorExit))
        retryable, status = _retryable(error)
        detail = f"HTTP {status}" if status is not None else type(error).__name__
        self.ledger.settle(
            charge.model_copy(
                update={
                    "outcome": "cancelled" if cancelled else "failed",
                    "source": f"Model API attempt failed ({detail}); billing unknown",
                }
            )
        )
        if not isinstance(error, Exception) or isinstance(error, (UsageLimitExceeded, BudgetError)):
            raise error
        if opened or not retryable or attempt >= self.provider_retries:
            reason = (
                "provider_rate_limit"
                if status == 429
                else "provider_timeout"
                if status == 408
                or isinstance(error, ModelTimeoutError)
                or type(error).__name__ == "APITimeoutError"
                else "provider_unavailable"
                if status in {500, 502, 503, 504}
                else "provider_connection"
                if status is None
                else "provider_error"
            )
            raise ProviderRequestError(reason, status) from None
        await asyncio.sleep(_retry_delay(error, attempt))

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        for attempt in range(self.provider_retries + 1):
            charge = self._reserve_attempt()
            try:
                response = await self.wrapped.request(
                    messages, model_settings, model_request_parameters
                )
            except BaseException as error:
                await self._after_failed_attempt(charge, error, attempt)
                continue
            self._settle_usage(charge, response.usage)
            return response
        raise AssertionError("Retry loop exited without a response or error")

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncGenerator[StreamedResponse]:
        for attempt in range(self.provider_retries + 1):
            charge = self._reserve_attempt()
            opened = False
            try:
                async with self.wrapped.request_stream(
                    messages, model_settings, model_request_parameters, run_context
                ) as response:
                    opened = True
                    yield response
            except BaseException as error:
                # An opened stream may have already delivered output. Never replay it.
                await self._after_failed_attempt(charge, error, attempt, opened=opened)
                continue
            self._settle_usage(charge, response.usage)
            return
        raise AssertionError("Streaming retry loop exited without a response or error")
