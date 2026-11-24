"""Custom endpoints must preserve usage metering, credentials and budget controls."""

import asyncio
import json
from decimal import Decimal

import pytest
from pydantic import SecretStr

from personal_harness import Agent, ConfigError, RunError, load_config
from personal_harness.connections import model_prices
from personal_harness.ledger import Ledger


def configured(path, **changes):
    values = {
        "model": {
            "provider": "openai-compatible",
            "model": "my-model",
            "base_url": "https://custom.test/v1",
            "context_window": 8192,
            "max_output_tokens": 128,
            "input_per_million_cny": "1",
            "output_per_million_cny": "2",
        },
        "runtime": {"offline": False, "provider_retries": 0},
        "budget": {
            "currency": "CNY",
            "max_task_cny": "1",
            "max_daily_cny": "2",
            "max_total_cny": "2",
        },
    }
    for section, patch in changes.items():
        values.setdefault(section, {}).update(patch)
    return load_config(path, overrides=values)


@pytest.mark.parametrize(
    "model",
    [
        {"base_url": "https://user:secret@example.test/v1"},
        {"base_url": "file:///private"},
        {"base_url": "https://example.test?key=secret"},
        {"context_window": None},
        {"input_per_million_cny": None},
        {"api_key_env": "INVALID KEY"},
        {"max_output_tokens": 10000},
    ],
)
def test_custom_model_rejects_unsafe_or_incomplete_configuration(config_file, model):
    with pytest.raises(ConfigError):
        configured(config_file, model=model)


def test_custom_model_price_is_explicit_and_keeps_arbitrary_model_name(config_file):
    resolved = configured(config_file)
    prices, context = model_prices(resolved.config.model)
    assert context == 8192 and prices.model == "my-model"
    assert prices.cache_read_per_million == Decimal("1")
    assert "User-configured" in prices.source


@pytest.mark.parametrize("usage_present", [True, False])
def test_actual_compatible_protocol_metering_and_unknown_usage(
    config_file, monkeypatch, usage_present
):
    httpx = pytest.importorskip("httpx2")
    openai = pytest.importorskip("openai")
    from personal_harness import connections

    recorded = []

    def respond(request):
        recorded.append(json.loads(request.content))
        assert request.headers["authorization"] == "Bearer private-test-key"
        assert request.url.host == "custom.test"
        return httpx.Response(
            200,
            json={
                "id": "mock",
                "object": "chat.completion",
                "created": 0,
                "model": "my-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "OK"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}
                if usage_present
                else None,
            },
        )

    def client(model, env_file, timeout_seconds, api_key):
        return openai.AsyncOpenAI(
            api_key=api_key.get_secret_value(),
            base_url=model.base_url,
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        )

    monkeypatch.setattr(connections, "compatible_client", client)

    async def exercise():
        async with Agent(configured(config_file), api_key=SecretStr("private-test-key")) as agent:
            if usage_present:
                result = await agent.run("hello", tools=[])
                assert result.cost == Decimal("0.00012") and result.cost_complete
                assert result.output == "OK" and "private-test-key" not in json.dumps(
                    result.as_dict()
                )
            else:
                with pytest.raises(RunError, match="unknown_usage"):
                    await agent.run("hello", tools=[])

    asyncio.run(exercise())
    assert len(recorded) == 1
    ledger = Ledger(config_file.parent / ".personal-harness/ledger.sqlite3")
    assert ledger.recent()[0]["charges"][0]["provider"] == "openai-compatible"
    if not usage_present:
        assert ledger.recent()[0]["cost"] is None
