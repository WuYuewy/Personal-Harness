"""Outgoing history compaction over a local OpenAI transport; no paid requests."""

import asyncio
import json
from decimal import Decimal

import pytest

from tests import mock_provider

httpx = pytest.importorskip("httpx2")
pytest.importorskip("dotenv")
openai = pytest.importorskip("openai")

from personal_harness import Agent, RunError, load_config  # noqa: E402
from personal_harness.ledger import Ledger  # noqa: E402


def response(tool=None, arguments=None, call_id="call"):
    message = {"role": "assistant", "content": "Decision processed"}
    if tool is not None:
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": tool,
                        "arguments": json.dumps(arguments),
                    },
                }
            ],
        }
    return {
        "id": "offline-history",
        "object": "chat.completion",
        "created": 0,
        "model": "test-chat",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool else "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_tokens_details": {"cached_tokens": 0},
        },
    }


@pytest.mark.parametrize("approve", [True, False])
def test_compaction_preserves_reads_approvals_trace_and_accounting(
    config_file, tmp_path, monkeypatch, approve
):
    original = (
        "# Evidence\r\n"
        + "".join(f"Source line {i:04d}: unchanged.\r\n" for i in range(300))
        + "Deadline: Friday\r\n"
    ).encode()
    active = ""
    payloads, sizes, results, previews, artifacts, clients = {}, {}, {}, {}, {}, []

    def respond(request):
        body = json.loads(request.content)
        payloads[active].append(body)
        sizes[active].append(len(request.content))
        returned = [m for m in body["messages"] if m["role"] == "tool"]
        reads = [m for m in returned if m["tool_call_id"].startswith("read-")]
        if len(reads) < 3:
            return httpx.Response(
                200,
                json=response(
                    "workspace_read", {"relative_path": "notes.md"}, f"read-{len(reads)}"
                ),
            )
        full = json.loads(reads[-1]["content"])
        assert full["content"].encode() == original
        if len(returned) == 3:
            return httpx.Response(
                200,
                json=response(
                    "workspace_replace_text",
                    {
                        "relative_path": "notes.md",
                        "old_text": "Friday",
                        "new_text": "Monday",
                        "expected_sha256": full["sha256"],
                    },
                    "edit",
                ),
            )
        return httpx.Response(200, json=response())

    def create(model, path, timeout_seconds=45, api_key=None):
        client = openai.AsyncOpenAI(
            api_key="local-placeholder",
            base_url=mock_provider.BASE_URL,
            max_retries=0,
            timeout=timeout_seconds,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        )
        clients.append(client)
        return client

    monkeypatch.setattr("personal_harness.connections.compatible_client", create)

    async def exercise():
        nonlocal active
        for variant in ("baseline", "compact"):
            active = variant
            payloads[variant], sizes[variant] = [], []
            root = tmp_path / variant
            root.mkdir()
            (root / "notes.md").write_bytes(original)
            resolved = load_config(
                config_file,
                overrides={
                    "workspace": {"root": str(root), "read_cache_bytes": 32000},
                    "model": {**mock_provider.settings("test-chat")},
                    "budget": {"currency": "CNY", "max_task_cny": "1", "max_daily_cny": "2"},
                    "runtime": {
                        "offline": False,
                        "provider_retries": 0,
                        "compact_read_history": variant == "compact",
                    },
                    "tools": {"allowed": ["workspace.read", "workspace.replace_text"]},
                },
            )
            async with Agent(resolved) as agent:
                pending = await agent.run("Change only Friday to Monday in notes.md")
                assert pending.status == "pending_approval" and pending.requests == 4
                previews[variant] = pending.approvals[0].preview
                assert (root / "notes.md").read_bytes() == original
                result = await agent.resume(pending.run_id, {pending.approvals[0].call_id: approve})
                results[variant] = result
                assert result.requests == 5 and result.cost_complete
                assert result.cost == Decimal("0.0006")  # Fixed mock usage, not a savings result.
                assert result.read_bytes > len(
                    original
                )  # Approval reads deliberately bypass cache.
                assert result.read_cache_hits == 2
                assert result.compacted_read_results == (5 if variant == "compact" else 0)
                summary = Ledger(config_file.parent / ".personal-harness/ledger.sqlite3").summary(
                    result.run_id
                )
                assert summary["requests"] == 5 and summary["cost_complete"]
                assert summary["metrics"]["compacted_read_bytes"] == result.compacted_read_bytes
                assert summary["metrics"]["execution_trace"] == result.execution_trace.model_dump(
                    mode="json"
                )
            artifacts[variant] = (root / "notes.md").read_bytes()

    asyncio.run(exercise())
    assert all(client.is_closed for client in clients)
    assert payloads["baseline"][:2] == payloads["compact"][:2]
    assert previews["baseline"] == previews["compact"]
    assert (
        artifacts["baseline"]
        == artifacts["compact"]
        == (original.replace(b"Friday", b"Monday") if approve else original)
    )
    assert results["baseline"].execution_trace == results["compact"].execution_trace
    assert results["baseline"].tool_calls == results["compact"].tool_calls
    assert results["baseline"].read_bytes == results["compact"].read_bytes
    final_reads = [
        m
        for m in payloads["compact"][-1]["messages"]
        if m["role"] == "tool" and m["tool_call_id"].startswith("read-")
    ]
    assert all("content_omitted" in json.loads(m["content"]) for m in final_reads[:2])
    assert json.loads(final_reads[-1]["content"])["content"].encode() == original
    assert sum(sizes["compact"]) < sum(sizes["baseline"])
    print(
        json.dumps(
            {
                "approve": approve,
                "source_bytes": len(original),
                "http_request_bytes": {key: sum(value) for key, value in sizes.items()},
                "compacted_result_bytes": results["compact"].compacted_read_bytes,
                "compacted_results": results["compact"].compacted_read_results,
                "requests_per_variant": 5,
            },
            sort_keys=True,
        )
    )


def test_failed_attempt_keeps_unknown_cost_and_does_not_recompact_retries(config_file, monkeypatch):
    (config_file.parent / "notes.md").write_bytes(b"Evidence\r\n" * 200)
    calls, clients = [], []

    def respond(request):
        body = json.loads(request.content)
        calls.append(body)
        if len(calls) <= 2:
            return httpx.Response(
                200,
                json=response("workspace_read", {"relative_path": "notes.md"}, f"r-{len(calls)}"),
            )
        return httpx.Response(
            429, json={"error": {"message": "too many requests"}}, headers={"retry-after": "0.001"}
        )

    def create(model, path, timeout_seconds=45, api_key=None):
        client = openai.AsyncOpenAI(
            api_key="local-placeholder",
            base_url=mock_provider.BASE_URL,
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        )
        clients.append(client)
        return client

    monkeypatch.setattr("personal_harness.connections.compatible_client", create)
    resolved = load_config(
        config_file,
        overrides={
            "model": {**mock_provider.settings("test-chat")},
            "budget": {"currency": "CNY", "max_task_cny": "1", "max_daily_cny": "2"},
            "runtime": {"offline": False, "provider_retries": 1, "compact_read_history": True},
            "tools": {"allowed": ["workspace.read"]},
        },
    )

    async def exercise():
        async with Agent(resolved) as agent:
            with pytest.raises(RunError) as failure:
                await agent.run("Read notes twice")
            assert failure.value.reason == "provider_rate_limit"
            return failure.value

    error = asyncio.run(exercise())
    assert clients[0].is_closed()
    assert len(calls) == 4 and calls[2] == calls[3]
    assert error.cost_summary["requests"] == 4 and not error.cost_summary["cost_complete"]
    assert error.cost_summary["metrics"]["compacted_read_results"] == 1
    assert Decimal(error.cost_summary["reserved_cost"]) > 0
    assert all(c.outcome == "success" for r in error.execution_trace.responses for c in r.calls)


def test_compaction_reduces_repeated_search_request_payload_without_extra_calls(
    config_file, tmp_path, monkeypatch
):
    sizes, outputs, compacted, clients = {}, {}, {}, []
    active = ""

    def respond(request):
        body = json.loads(request.content)
        sizes[active].append(len(request.content))
        search_returns = [
            message
            for message in body["messages"]
            if message["role"] == "tool" and message["tool_call_id"].startswith("search-")
        ]
        if len(search_returns) < 2:
            return httpx.Response(
                200,
                json=response(
                    "workspace_search",
                    {"query": "Evidence", "relative_path": "."},
                    f"search-{len(search_returns)}",
                ),
            )
        return httpx.Response(200, json=response())

    def create(model, path, timeout_seconds=45, api_key=None):
        client = openai.AsyncOpenAI(
            api_key="local-placeholder",
            base_url=mock_provider.BASE_URL,
            max_retries=0,
            timeout=timeout_seconds,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        )
        clients.append(client)
        return client

    monkeypatch.setattr("personal_harness.connections.compatible_client", create)

    async def exercise():
        nonlocal active
        for variant in ("baseline", "compact"):
            active = variant
            sizes[variant] = []
            root = tmp_path / variant
            root.mkdir()
            (root / "evidence.md").write_text(
                "\n".join(f"Evidence line {index}: " + "x" * 100 for index in range(40)),
                "utf-8",
            )
            resolved = load_config(
                config_file,
                overrides={
                    "workspace": {"root": str(root)},
                    "model": {**mock_provider.settings("test-chat")},
                    "budget": {
                        "currency": "CNY",
                        "max_task_cny": "1",
                        "max_daily_cny": "2",
                    },
                    "runtime": {
                        "offline": False,
                        "provider_retries": 0,
                        "compact_read_history": variant == "compact",
                    },
                    "tools": {"allowed": ["workspace.search"]},
                },
            )
            async with Agent(resolved) as agent:
                result = await agent.run("Search for Evidence")
                outputs[variant] = result.output
                compacted[variant] = result.compacted_read_results
                assert result.requests == 3
                assert len(result.tool_calls) == 2

    asyncio.run(exercise())
    assert all(client.is_closed() for client in clients)
    assert outputs["baseline"] == outputs["compact"] == "Decision processed"
    assert compacted == {"baseline": 0, "compact": 1}
    assert sizes["baseline"][:2] == sizes["compact"][:2]
    assert sum(sizes["compact"]) < sum(sizes["baseline"])
