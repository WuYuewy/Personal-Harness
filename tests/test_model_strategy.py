"""Exercise priced fallback through the actual compatible HTTP adapter, offline."""

import asyncio
import json
from decimal import Decimal

import pytest
from pydantic import SecretStr

from personal_harness import Agent, ConfigError, RunError
from personal_harness.ledger import Ledger
from tests.test_connections import configured


def strategy(path, *, max_steps=8, task_budget="1"):
    base = configured(path)
    return configured(
        path,
        fallback_model={
            **base.config.model.model_dump(),
            "model": "upgrade",
            "output_per_million_cny": "4",
        },
        runtime={"max_model_upgrades": 1, "max_steps": max_steps},
        budget={"max_task_cny": task_budget},
    )


@pytest.mark.parametrize("failure", [None, "unknown", "http", "cap", "budget"])
def test_upgrade_accounts_discarded_response_and_obeys_hard_stops(
    config_file, monkeypatch, failure
):
    import httpx2 as httpx
    import openai

    from personal_harness import connections

    calls = []
    clients = []

    def respond(request):
        model = json.loads(request.content)["model"]
        calls.append(model)
        if failure == "http":
            return httpx.Response(429, json={"error": {"message": "busy"}})
        return httpx.Response(
            200,
            json={
                "id": "mock",
                "object": "chat.completion",
                "created": 0,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "OK"},
                        "finish_reason": "length" if model == "my-model" else "stop",
                    }
                ],
                "usage": None
                if failure == "unknown"
                else {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
            },
        )

    def client(model, env_file, timeout_seconds, api_key):
        value = openai.AsyncOpenAI(
            api_key="local-test",
            base_url=model.base_url,
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        )
        clients.append(value)
        return value

    monkeypatch.setattr(connections, "compatible_client", client)

    async def exercise():
        resolved = strategy(
            config_file,
            max_steps=1 if failure == "cap" else 8,
            task_budget="0.0086" if failure == "budget" else "1",
        )
        async with Agent(resolved, api_key=SecretStr("test")) as agent:
            if failure:
                with pytest.raises(RunError) as error:
                    await agent.run("hello", tools=[])
                expected = {
                    "unknown": "unknown_usage",
                    "http": "provider_rate_limit",
                    "cap": "usage_limit",
                    "budget": "task_budget",
                }
                assert error.value.reason == expected[failure]
            else:
                for _ in range(2):  # The next run must start on the primary again.
                    result = await agent.run("hello", tools=[])
                    assert result.output == "OK"
                    assert (
                        result.requests == 2
                        and result.input_tokens == 200
                        and result.output_tokens == 20
                    )
                    assert result.cost == Decimal("0.00026")
                    snapshot = agent.cost_summary(result.run_id)
                    assert snapshot["metrics"]["input_tokens"] == 200
                    assert snapshot["metrics"]["model_upgrades"] == 1

    asyncio.run(exercise())
    assert calls == (["my-model", "upgrade"] * 2 if failure is None else ["my-model"])
    assert all(c.is_closed() for c in clients)
    if failure in {"cap", "budget"}:
        row = Ledger(config_file.parent / ".personal-harness/ledger.sqlite3").recent()[0]
        assert row["metrics"]["input_tokens"] == 100


def test_upgrade_requires_explicit_configuration(config_file):
    with pytest.raises(ConfigError):
        configured(config_file, runtime={"max_model_upgrades": 1})
    with pytest.raises(ConfigError):
        configured(config_file, fallback_model=configured(config_file).config.model.model_dump())


def test_upgrade_persists_across_deferred_approval(config_file, monkeypatch):
    import httpx2 as httpx
    import openai

    from personal_harness import connections, load_config

    calls = []

    def respond(request):
        model = json.loads(request.content)["model"]
        calls.append(model)
        message = {"role": "assistant", "content": "OK"}
        reason = "length" if model == "my-model" else "stop"
        if len(calls) == 2:
            message = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "write-1",
                        "type": "function",
                        "function": {
                            "name": "workspace_write",
                            "arguments": json.dumps(
                                {
                                    "relative_path": "out.md",
                                    "content": "approved",
                                    "expected_sha256": None,
                                }
                            ),
                        },
                    }
                ],
            }
            reason = "tool_calls"
        return httpx.Response(
            200,
            json={
                "id": "mock",
                "object": "chat.completion",
                "created": 0,
                "model": model,
                "choices": [{"index": 0, "message": message, "finish_reason": reason}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
            },
        )

    def client(model, env_file, timeout_seconds, api_key):
        return openai.AsyncOpenAI(
            api_key="local-test",
            base_url=model.base_url,
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        )

    monkeypatch.setattr(connections, "compatible_client", client)
    resolved = load_config(
        config_file,
        overrides={
            **strategy(config_file).config.model_dump(),
            "tools": {"allowed": ["workspace.write"]},
        },
    )

    async def exercise():
        async with Agent(resolved) as agent:
            pending = await agent.run("create out.md")
            assert pending.status == "pending_approval"
            assert not (config_file.parent / "out.md").exists()
            result = await agent.resume(
                pending.run_id, {a.call_id: True for a in pending.approvals}
            )
            assert result.status == "completed" and result.requests == 3
            assert result.input_tokens == 300 and result.output_tokens == 30
            assert agent.cost_summary(result.run_id)["metrics"]["model_upgrades"] == 1
            assert (config_file.parent / "out.md").read_text() == "approved"

    asyncio.run(exercise())
    assert calls == ["my-model", "upgrade", "upgrade"]
