import asyncio
import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import FunctionModel

from personal_harness import Agent, RunError
from personal_harness.trace import ExecutionTrace, ToolCallTrace, summarize_messages


def test_summary_pairs_out_of_order_returns_and_reused_ids_and_redacts_content() -> None:
    secret = "private-file-content-and-user-input"
    source = ToolReturnPart("workspace_read", secret, "reused-id")
    batch = ToolReturnPart("workspace_read_many", '{"complete":false}', "batch-id")
    denied = ToolReturnPart("workspace_write", "Refused", "reused-id", outcome="denied")
    unknown = ToolReturnPart(
        "sensitive-unknown-name", "bad arguments", "unknown", outcome="retried"
    )
    trace = summarize_messages(
        [
            ModelRequest([UserPromptPart(secret)]),
            ModelResponse(
                [
                    ToolCallPart("workspace_read", {"relative_path": secret}, "reused-id"),
                    ToolCallPart("workspace_read_many", {"reads": []}, "batch-id"),
                ]
            ),
            ModelRequest([batch, source]),
            ModelResponse([ToolCallPart("workspace_write", {"content": secret}, "reused-id")]),
            ModelRequest([denied]),
            ModelResponse([ToolCallPart("sensitive-unknown-name", secret, "unknown")]),
            ModelRequest([unknown, ToolReturnPart("workspace_list", "orphan", "missing")]),
            ModelResponse([TextPart(secret)]),
        ]
    )
    assert len(trace.responses) == 4 and trace.multi_call_responses == 1
    assert trace.responses[0].calls[0].result_bytes == len(secret.encode())
    assert trace.responses[0].calls[1].outcome == "success"  # Tool success is not task quality.
    assert trace.responses[1].calls[0].outcome == "denied"
    assert trace.responses[2].calls[0].tool == "unregistered"
    assert trace.responses[2].calls[0].outcome == "retried"
    assert trace.unmatched_tool_returns == 1
    assert trace.tool_result_bytes == sum(
        len(part.model_response_str().encode()) for part in (source, batch, denied, unknown)
    )
    serialized = trace.model_dump_json()
    for hidden in (secret, "reused-id", "sensitive-unknown-name", "relative_path", "Refused"):
        assert hidden not in serialized
    assert ExecutionTrace.model_validate_json(serialized) == trace


def test_interrupted_response_does_not_mean_its_call_executed() -> None:
    trace = summarize_messages(
        [
            ModelResponse(
                [ToolCallPart("workspace_read", {"relative_path": "private"})],
                state="interrupted",
            )
        ]
    )
    assert trace.responses[0].state == "interrupted"
    assert trace.responses[0].calls[0].outcome == "unresolved"
    assert trace.responses[0].calls[0].result_bytes is None and trace.tool_result_bytes == 0


def test_reused_call_id_for_the_same_tool_in_different_responses_is_not_merged() -> None:
    trace = summarize_messages(
        [
            ModelResponse([ToolCallPart("workspace_read", {}, "same")]),
            ModelRequest([ToolReturnPart("workspace_read", "bad", "same", outcome="retried")]),
            ModelResponse([ToolCallPart("workspace_read", {}, "same")]),
            ModelRequest([ToolReturnPart("workspace_read", "ok", "same")]),
        ]
    )
    assert trace.unmatched_tool_returns == 0
    assert [response.calls[0].outcome for response in trace.responses] == ["retried", "success"]
    assert trace.responses[1].calls[0].result_bytes == 2


@pytest.mark.parametrize(
    "outcome,result_bytes", [("success", None), ("unresolved", 0), ("denied", -1)]
)
def test_trace_schema_rejects_inconsistent_outcomes(outcome, result_bytes) -> None:
    with pytest.raises(ValidationError):
        ToolCallTrace(tool="workspace.read", outcome=outcome, result_bytes=result_bytes)


@pytest.mark.parametrize("approve", [True, False])
def test_approval_snapshot_is_rebuilt_without_duplicating_earlier_responses(
    config_file: Path, approve: bool
) -> None:
    def respond(messages, info):
        if any(
            isinstance(part, ToolReturnPart)
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
        ):
            return ModelResponse([TextPart("Decision processed")])
        return ModelResponse(
            [
                ToolCallPart(
                    "workspace_write",
                    {"relative_path": "report.md", "content": "proposed", "expected_sha256": None},
                )
            ]
        )

    async def exercise():
        async with Agent.from_config(
            config_file,
            overrides={"tools": {"allowed": ["workspace.write"]}},
            model=FunctionModel(respond),
        ) as agent:
            pending = await agent.run("Write report")
            before = pending.as_dict()["execution_trace"]
            final = await agent.resume(pending.run_id, {pending.approvals[0].call_id: approve})
        return pending, before, final

    pending, before, final = asyncio.run(exercise())
    assert pending.execution_trace.responses[0].calls[0].outcome == "unresolved"
    assert pending.as_dict()["execution_trace"] == before
    assert len(final.execution_trace.responses) == 2 and final.requests == 2
    assert final.execution_trace.responses[0].calls[0].outcome == (
        "success" if approve else "denied"
    )
    assert final.execution_trace.responses[1].calls == ()
    assert (config_file.parent / "report.md").exists() is approve


def test_concurrent_runs_and_failure_preserve_independent_diagnostics(config_file: Path) -> None:
    (config_file.parent / "notes.md").write_bytes(b"private-source")
    both_started = asyncio.Event()
    started = set()

    async def respond(messages, info):
        task = next(
            part.content
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, UserPromptPart)
        )
        started.add(task)
        if len(started) == 2:
            both_started.set()
        await both_started.wait()
        if task == "plain":
            return ModelResponse([TextPart("No tool")])
        returns = [
            part
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        if returns:
            raise ValueError("private failure detail")
        return ModelResponse([ToolCallPart("workspace_read", {"relative_path": "notes.md"})])

    async def exercise():
        async with Agent.from_config(
            config_file,
            overrides={"tools": {"allowed": ["workspace.read"]}},
            model=FunctionModel(respond),
        ) as agent:
            plain, failure = await asyncio.gather(
                agent.run("plain", tools=[]), agent.run("read"), return_exceptions=True
            )
        return plain, failure

    plain, failure = asyncio.run(exercise())
    assert (
        len(plain.execution_trace.responses) == 1 and not plain.execution_trace.responses[0].calls
    )
    assert isinstance(failure, RunError) and failure.reason == "execution_error"
    assert len(failure.execution_trace.responses) == 1
    call = failure.execution_trace.responses[0].calls[0]
    assert call.outcome == "success" and call.result_bytes > len(b"private-source")
    assert "private" not in json.dumps(failure.execution_trace.model_dump())
