import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from personal_harness import Agent, RunError


@pytest.mark.parametrize("existing", [False, True])
def test_invalid_write_fingerprint_is_corrected_before_approval(config_file, existing):
    target = config_file.parent / "report.md"
    if existing:
        target.write_text("original", "utf-8")
    correct = hashlib.sha256(target.read_bytes()).hexdigest() if existing else None
    attempts = 0

    async def respond(messages, info):
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            assert "Invalid write proposal" in str(messages)
            assert "use null for a new file" in str(messages)
            assert target.read_text("utf-8") == "original" if existing else not target.exists()
        if attempts < 3:
            return ModelResponse(
                [
                    ToolCallPart(
                        "workspace_write",
                        {
                            "relative_path": "report.md",
                            "content": "Reviewed content",
                            "expected_sha256": "a" * 64 if attempts == 1 else correct,
                        },
                    )
                ]
            )
        return ModelResponse([TextPart("Saved after approval.")])

    async def exercise():
        async with Agent.from_config(
            config_file,
            overrides={"tools": {"allowed": ["workspace.write"]}},
            model=FunctionModel(respond),
        ) as agent:
            pending = await agent.run("Write a report")
            assert pending.status == "pending_approval" and pending.requests == 2
            assert pending.approvals[0].preview["expected_sha256"] == correct
            assert target.read_text("utf-8") == "original" if existing else not target.exists()
            final = await agent.resume(pending.run_id, {pending.approvals[0].call_id: True})
            assert final.status == "completed" and final.requests == 3
            assert target.read_text("utf-8") == "Reviewed content"

    asyncio.run(exercise())


def write_model(path: str, content: str, fingerprint: str | None = None) -> FunctionModel:
    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(
            isinstance(part, ToolReturnPart) and part.tool_name == "workspace_write"
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
        ):
            return ModelResponse([TextPart("Write decision processed.")])
        return ModelResponse(
            [
                ToolCallPart(
                    "workspace_write",
                    {
                        "relative_path": path,
                        "content": content,
                        "expected_sha256": fingerprint,
                    },
                )
            ]
        )

    return FunctionModel(respond)


def writer(config_file: Path, fingerprint: str | None = None) -> Agent:
    return Agent.from_config(
        config_file,
        overrides={"tools": {"allowed": ["workspace.write"]}},
        model=write_model("report.md", "Approved contents\n", fingerprint),
    )


def test_actual_framework_write_waits_for_approval_then_executes_once(config_file: Path) -> None:
    async def exercise() -> None:
        agent = writer(config_file)
        pending = await agent.run("Create report")
        target = config_file.parent / "report.md"
        assert pending.status == "pending_approval"
        assert not target.exists()
        assert pending.requests == 1
        approval = pending.approvals[0]
        assert approval.tool == "workspace.write"
        assert approval.preview["content"] == "Approved contents\n"
        assert approval.preview["written"] is False
        # Editing a public preview cannot alter the held tool call arguments.
        approval.preview["content"] = "tampered"
        completed = await agent.resume(pending.run_id, {approval.call_id: True})
        assert completed.status == "completed"
        assert completed.run_id == pending.run_id
        assert completed.requests == 2
        assert completed.tool_calls == ("workspace.write",)
        assert target.read_text("utf-8") == "Approved contents\n"
        with pytest.raises(RunError, match="No pending"):
            await agent.resume(pending.run_id, {approval.call_id: True})

    asyncio.run(exercise())


def test_rejection_never_writes_and_records_denied_event(config_file: Path) -> None:
    async def exercise() -> None:
        agent = writer(config_file)
        pending = await agent.run("Create report")
        final = await agent.resume(pending.run_id, {pending.approvals[0].call_id: False})
        assert not (config_file.parent / "report.md").exists()
        assert final.tool_calls == ()
        assert "tool.workspace.write.denied" in final.events

    asyncio.run(exercise())


@pytest.mark.parametrize("existing", [False, True])
def test_file_changes_during_approval_cannot_be_overwritten(
    config_file: Path, existing: bool
) -> None:
    target = config_file.parent / "report.md"
    fingerprint = None
    if existing:
        target.write_text("original", "utf-8")
        fingerprint = hashlib.sha256(target.read_bytes()).hexdigest()

    async def exercise() -> None:
        agent = writer(config_file, fingerprint)
        pending = await agent.run("Create report")
        target.write_text("user edited while pending", "utf-8")
        with pytest.raises(RunError) as error:
            await agent.resume(pending.run_id, {pending.approvals[0].call_id: True})
        assert error.value.reason == "write_conflict"
        assert target.read_text("utf-8") == "user edited while pending"

    asyncio.run(exercise())


def test_wrong_approval_ids_and_cross_agent_resumes_are_rejected(config_file: Path) -> None:
    async def exercise() -> None:
        agent = writer(config_file)
        pending = await agent.run("Create report")
        with pytest.raises(RunError) as error:
            await agent.resume(pending.run_id, {"unrelated": True})
        assert error.value.reason == "invalid_approval"
        with pytest.raises(RunError):
            await writer(config_file).resume(pending.run_id, {pending.approvals[0].call_id: True})
        final = await agent.resume(pending.run_id, {pending.approvals[0].call_id: False})
        assert final.status == "completed"

    asyncio.run(exercise())


@pytest.mark.parametrize("preview_mode", ["full", "summary"])
def test_disabled_write_tool_cannot_be_called_by_model(
    config_file: Path, preview_mode: str
) -> None:
    async def exercise() -> None:
        agent = Agent.from_config(
            config_file,
            overrides={"tools": {"allowed": [], "preview_mode": preview_mode}},
            model=write_model("report.md", "bad"),
        )
        result = await agent.run("Write without a registered tool")
        assert result.tool_calls == ()
        assert result.approvals == ()
        assert not (config_file.parent / "report.md").exists()

    asyncio.run(exercise())


def test_framework_retries_invalid_tool_arguments_and_rejects_traversal(config_file: Path) -> None:
    attempts = 0

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return ModelResponse([ToolCallPart("workspace_read", {"relative_path": "../secret"})])
        return ModelResponse([TextPart("The workspace boundary rejected this read.")])

    result = asyncio.run(
        Agent.from_config(
            config_file,
            overrides={"tools": {"allowed": ["workspace.read"]}},
            model=FunctionModel(respond),
        ).run("Read external file")
    )
    assert result.requests == 2
    assert result.tool_calls == ()


def test_framework_can_read_then_search_and_preview_without_writing(config_file: Path) -> None:
    (config_file.parent / "notes.md").write_text("TODO: implement profiles\n", "utf-8")
    calls = [
        ("workspace_list", {}),
        ("workspace_read", {"relative_path": "notes.md"}),
        ("workspace_search", {"query": "TODO"}),
        ("workspace_write_preview", {"relative_path": "report.md", "content": "Report\n"}),
    ]

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        returns = [
            part
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        if len(returns) < len(calls):
            name, args = calls[len(returns)]
            return ModelResponse([ToolCallPart(name, args)])
        for part in returns:
            data: dict[str, Any] = json.loads(str(part.content))
            assert data
        return ModelResponse([TextPart("Read, search and preview completed.")])

    result = asyncio.run(
        Agent.from_config(
            config_file,
            overrides={
                "tools": {
                    "allowed": [
                        "workspace.list",
                        "workspace.read",
                        "workspace.search",
                        "workspace.write_preview",
                    ]
                }
            },
            model=FunctionModel(respond),
        ).run("Inspect notes")
    )
    assert result.requests == 5
    assert result.tool_calls == (
        "workspace.list",
        "workspace.read",
        "workspace.search",
        "workspace.write_preview",
    )
    assert not (config_file.parent / "report.md").exists()


def test_call_limit_survives_approval_resume(config_file: Path) -> None:
    async def exercise() -> None:
        agent = Agent.from_config(
            config_file,
            overrides={"tools": {"allowed": ["workspace.write"]}, "runtime": {"max_steps": 1}},
            model=write_model("report.md", "content"),
        )
        pending = await agent.run("Create")
        with pytest.raises(RunError) as error:
            await agent.resume(pending.run_id, {pending.approvals[0].call_id: False})
        assert error.value.reason == "usage_limit"

    asyncio.run(exercise())


def test_timeout_and_cancellation_propagate_without_writes(config_file: Path) -> None:
    async def slow(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        await asyncio.Event().wait()
        return ModelResponse([TextPart("unreachable")])

    async def exercise() -> None:
        agent = Agent.from_config(
            config_file, model=FunctionModel(slow), overrides={"runtime": {"timeout_seconds": 1}}
        )
        with pytest.raises(RunError) as error:
            await agent.run("timeout")
        assert error.value.reason == "timeout"
        task = asyncio.create_task(agent.run("cancel"))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())
