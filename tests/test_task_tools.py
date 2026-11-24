import asyncio
from pathlib import Path

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from personal_harness import Agent, RunError


def prompt(messages: list[ModelMessage]) -> str:
    return next(
        part.content
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart) and isinstance(part.content, str)
    )


def test_concurrent_tasks_keep_independent_tool_subsets(config_file: Path) -> None:
    seen = {}

    async def exercise():
        both_entered = asyncio.Event()

        async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            task = prompt(messages)
            seen[task] = {tool.name for tool in info.function_tools}
            if len(seen) == 2:
                both_entered.set()
            await both_entered.wait()
            return ModelResponse([TextPart(task)])

        async with Agent.from_config(
            config_file,
            overrides={"tools": {"allowed": ["workspace.read", "workspace.write"]}},
            model=FunctionModel(respond),
        ) as agent:
            first, second = await asyncio.gather(
                agent.run("read", tools=["workspace.read"]),
                agent.run("write", tools=["workspace.write"]),
            )
            default = await agent.run("default")
        return first, second, default

    first, second, default = asyncio.run(exercise())
    assert seen == {
        "read": {"workspace_read"},
        "write": {"workspace_write"},
        "default": {"workspace_read", "workspace_write"},
    }
    assert first.available_tools == ("workspace.read",)
    assert second.available_tools == ("workspace.write",)
    assert default.available_tools == ("workspace.read", "workspace.write")
    assert first.as_dict()["available_tools"] == ["workspace.read"]


@pytest.mark.parametrize("tool", ["workspace_read", "workspace_read_many", "workspace_write"])
def test_model_cannot_execute_a_tool_omitted_from_task(config_file: Path, tool: str) -> None:
    calls = 0
    (config_file.parent / "notes.md").write_text("Private source", "utf-8")

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        assert info.function_tools == []
        calls += 1
        if calls == 1:
            args = {"relative_path": "notes.md"}
            if tool == "workspace_read_many":
                args = {"reads": [{"relative_path": "notes.md"}]}
            if tool == "workspace_write":
                args.update(content="Unapproved replacement", expected_sha256=None)
            return ModelResponse([ToolCallPart(tool, args)])
        return ModelResponse([TextPart("Tool unavailable.")])

    async def exercise():
        async with Agent.from_config(
            config_file,
            overrides={
                "tools": {"allowed": ["workspace.read", "workspace.read_many", "workspace.write"]}
            },
            model=FunctionModel(respond),
        ) as agent:
            return await agent.run("No file access for this task", tools=[])

    result = asyncio.run(exercise())
    assert calls == 2
    assert result.tool_calls == result.approvals == result.available_tools == ()
    assert result.read_bytes == 0
    assert (config_file.parent / "notes.md").read_text("utf-8") == "Private source"


def test_task_tool_snapshot_survives_approval_and_interleaved_runs(config_file: Path) -> None:
    seen = []

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        task = prompt(messages)
        seen.append((task, {tool.name for tool in info.function_tools}))
        written = any(
            isinstance(part, ToolReturnPart) and part.tool_name == "workspace_write"
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
        )
        if task == "write" and not written:
            return ModelResponse(
                [
                    ToolCallPart(
                        "workspace_write",
                        {
                            "relative_path": "brief.md",
                            "content": "Approved",
                            "expected_sha256": None,
                        },
                    )
                ]
            )
        return ModelResponse([TextPart("Done")])

    async def exercise():
        async with Agent.from_config(
            config_file,
            overrides={"tools": {"allowed": ["workspace.read", "workspace.write"]}},
            model=FunctionModel(respond),
        ) as agent:
            requested = ["workspace.write"]
            pending = await agent.run("write", tools=requested)
            assert pending.status == "pending_approval"
            requested[:] = ["workspace.read"]
            other = await agent.run("inspect", tools=requested)
            assert other.available_tools == ("workspace.read",)
            return await agent.resume(pending.run_id, {pending.approvals[0].call_id: True})

    result = asyncio.run(exercise())
    assert seen == [
        ("write", {"workspace_write"}),
        ("inspect", {"workspace_read"}),
        ("write", {"workspace_write"}),
    ]
    assert result.available_tools == ("workspace.write",)
    assert result.tool_calls == ("workspace.write",)
    assert (config_file.parent / "brief.md").read_text("utf-8") == "Approved"


@pytest.mark.parametrize("requested", [["workspace.write"], ["unknown"], "context.describe"])
def test_task_tools_cannot_expand_profile_permissions(config_file: Path, requested: object) -> None:
    async def exercise():
        async with Agent.from_config(config_file) as agent:
            with pytest.raises(RunError) as failure:
                await agent.run("Inspect", tools=requested)
            return failure.value

    error = asyncio.run(exercise())
    assert error.reason == "tool_selection" and error.run_id is None
