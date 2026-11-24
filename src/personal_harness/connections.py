"""OpenAI-compatible connections, with credentials kept outside resolved config."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import SecretStr

from personal_harness.config import ModelSettings
from personal_harness.costs import TokenPrices

if TYPE_CHECKING:
    from openai import AsyncOpenAI


def model_prices(model: ModelSettings) -> tuple[TokenPrices, int]:
    if model.provider != "openai-compatible":
        raise ValueError("Offline models do not require paid prices.")
    assert model.context_window is not None
    assert model.input_per_million_cny is not None and model.output_per_million_cny is not None
    return TokenPrices(
        provider=model.provider,
        model=model.model,
        source="User-configured CNY rates; verify against provider billing",
        recorded_on=datetime.now(UTC).date(),
        currency="CNY",
        input_per_million=model.input_per_million_cny,
        output_per_million=model.output_per_million_cny,
        cache_read_per_million=(
            model.cache_read_per_million_cny
            if model.cache_read_per_million_cny is not None
            else model.input_per_million_cny
        ),
    ), model.context_window


def compatible_client(
    model: ModelSettings,
    env_file: Path,
    timeout_seconds: int,
    api_key: SecretStr | None = None,
) -> AsyncOpenAI:
    from dotenv import dotenv_values
    from openai import AsyncOpenAI

    key = api_key.get_secret_value() if api_key is not None else os.environ.get(model.api_key_env)
    if not key and env_file.is_file():
        key = dotenv_values(env_file, interpolate=False).get(model.api_key_env)
    if not key or not key.strip():
        raise ValueError("Provide an API key in Model Connections or the configured environment.")
    return AsyncOpenAI(
        api_key=key.strip(),
        base_url=model.base_url,
        timeout=timeout_seconds,
        max_retries=0,
    )


def uniform_input_price(prices: TokenPrices) -> bool:
    return prices.cache_read_per_million in (None, Decimal(prices.input_per_million))
