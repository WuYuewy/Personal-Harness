"""Synthetic model metadata used only by regression tests, never by runtime defaults."""

from datetime import date

from personal_harness.costs import TokenPrices

BASE_URL = "https://model.example.test/v1"
MODELS = ("test-chat", "test-large", "test-large-2", "test-reasoner")


def settings(model="test-chat"):
    large = model in {"test-large", "test-large-2"}
    return dict(
        provider="openai-compatible",
        model=model,
        base_url=BASE_URL,
        context_window=1000000 if large else 262144,
        input_per_million_cny="2" if large else "1",
        output_per_million_cny="2" if large else "1",
        cache_read_per_million_cny="0.2" if large else "0.1",
    )


def prices_for(model):
    if model not in MODELS:
        return None
    m = settings(model)
    return TokenPrices(
        provider="openai-compatible",
        model=model,
        source="Synthetic test fixture",
        recorded_on=date(2026, 11, 18),
        currency="CNY",
        input_per_million=m["input_per_million_cny"],
        output_per_million=m["output_per_million_cny"],
        cache_read_per_million=m["cache_read_per_million_cny"],
    )


def template(model="test-chat"):
    from importlib.resources import files

    s = files("personal_harness").joinpath("templates/openai-compatible.toml").read_text("utf-8")
    meta = settings(model)
    s = s.replace("your-model-id", model).replace("https://api.example.com/v1", BASE_URL)
    s = s.replace("context_window = 32768", f"context_window = {meta['context_window']}")
    s = s.replace(
        'input_per_million_cny = "1"', f'input_per_million_cny = "{meta["input_per_million_cny"]}"'
    )
    s = s.replace(
        'output_per_million_cny = "1"',
        f'output_per_million_cny = "{meta["output_per_million_cny"]}"\n'
        f'cache_read_per_million_cny = "{meta["cache_read_per_million_cny"]}"',
    )
    return s
