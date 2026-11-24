"""Offline OpenAI-protocol comparison; no gateway requests or model-quality claims."""

import asyncio
import json
from pathlib import Path

import pytest

from tests import mock_provider

httpx = pytest.importorskip("httpx2")
pytest.importorskip("dotenv")
openai = pytest.importorskip("openai")

from personal_harness import Agent, load_config  # noqa: E402
from personal_harness.ledger import Ledger  # noqa: E402


@pytest.mark.parametrize("approve", [True, False])
def test_full_write_and_literal_edit_keep_approval_and_artifact_equivalent(
    config_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, approve: bool
):
    original = (
        "# Report\r\n"
        + "".join(
            f"Unchanged reference {index:04d}: retain this sentence.\r\n" for index in range(300)
        )
        + "Deadline: Friday\r\n"
    ).encode("utf-8")
    payloads, sizes, arguments, previews, outputs, clients = {}, {}, {}, {}, {}, []
    active = ""

    def respond(request):
        body = json.loads(request.content)
        payloads[active].append(body)
        sizes[active].append(len(request.content))
        returned = [message for message in body["messages"] if message["role"] == "tool"]
        tool, args = None, None
        if not returned:
            tool, args = "workspace_read", {"relative_path": "report.md"}
        elif len(returned) == 1:
            source = json.loads(returned[0]["content"])
            tool = "workspace_write" if active == "full" else "workspace_replace_text"
            args = {"relative_path": "report.md", "expected_sha256": source["sha256"]}
            if active == "full":
                args["content"] = source["content"].replace("Friday", "Monday")
            else:
                args.update(old_text="Friday", new_text="Monday")
            arguments[active] = json.dumps(args, ensure_ascii=False).encode()
        message = {"role": "assistant", "content": "Decision processed"}
        if tool is not None:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tool,
                        "type": "function",
                        "function": {"name": tool, "arguments": json.dumps(args)},
                    }
                ],
            }
        return httpx.Response(
            200,
            json={
                "id": "offline-edit",
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
            },
        )

    def create(model, path, timeout_seconds=45, api_key=None):
        client = openai.AsyncOpenAI(
            api_key="local-placeholder",
            base_url=mock_provider.BASE_URL,
            timeout=timeout_seconds,
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        )
        clients.append(client)
        return client

    monkeypatch.setattr("personal_harness.connections.compatible_client", create)

    async def exercise():
        nonlocal active
        for variant in ("full", "edit"):
            active = variant
            payloads[variant], sizes[variant] = [], []
            root = tmp_path / variant
            root.mkdir()
            (root / "report.md").write_bytes(original)
            resolved = load_config(
                config_file,
                overrides={
                    "model": {
                        **mock_provider.settings("test-chat"),
                        "max_output_tokens": 8192,
                    },
                    "budget": {"currency": "CNY", "max_task_cny": "1", "max_daily_cny": "2"},
                    "runtime": {"offline": False, "provider_retries": 0},
                    "workspace": {"root": str(root)},
                    # Both groups receive exactly the same capabilities and schema.
                    "tools": {
                        "allowed": ["workspace.read", "workspace.write", "workspace.replace_text"]
                    },
                },
            )
            async with Agent(resolved) as agent:
                pending = await agent.run("Change only Friday to Monday in report.md")
                assert pending.status == "pending_approval" and pending.requests == 2
                approval = pending.approvals[0]
                previews[variant] = approval.preview
                assert (root / "report.md").read_bytes() == original
                result = await agent.resume(pending.run_id, {approval.call_id: approve})
                outputs[variant] = (root / "report.md").read_bytes()
                assert result.status == "completed" and result.requests == 3
                assert result.tool_calls == (
                    (
                        "workspace.read",
                        "workspace.write" if variant == "full" else "workspace.replace_text",
                    )
                    if approve
                    else ("workspace.read",)
                )
                trace = result.execution_trace.responses[1].calls[0]
                assert trace.outcome == ("success" if approve else "denied")
                ledger = Ledger(config_file.parent / ".personal-harness/ledger.sqlite3")
                summary = ledger.summary(result.run_id)
                assert summary["requests"] == 3 and summary["cost_complete"]
                assert summary["metrics"]["execution_trace"] == result.execution_trace.model_dump(
                    mode="json"
                )
        return

    asyncio.run(exercise())
    assert all(client.is_closed for client in clients)
    assert (
        outputs["full"]
        == outputs["edit"]
        == (original.replace(b"Friday", b"Monday") if approve else original)
    )
    assert previews["full"] == previews["edit"]
    assert payloads["full"][0] == payloads["edit"][0]
    assert len(arguments["full"]) > len(arguments["edit"]) * 20
    assert sum(sizes["edit"]) < sum(sizes["full"])
    # Fixed mock usage validates accounting only; no token or money reduction is inferred.
    print(
        json.dumps(
            {
                "approval": approve,
                "source_bytes": len(original),
                "tool_argument_bytes": {key: len(value) for key, value in arguments.items()},
                "http_request_bytes": {key: sum(value) for key, value in sizes.items()},
                "requests_per_variant": 3,
            },
            sort_keys=True,
        )
    )
