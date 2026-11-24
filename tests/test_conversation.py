"""Real framework multi-turn protocol, including tools and deferred approval."""

import asyncio

import pytest
from pydantic_ai.messages import (
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import FunctionModel

from personal_harness import Agent, load_config
from personal_harness.conversation import history_status, select_history


def test_history_is_explicit_isolated_and_tools_count_only_current_turn(config_file):
    def respond(messages, info):
        prompts = [
            p.content
            for m in messages
            if isinstance(m, ModelRequest)
            for p in m.parts
            if isinstance(p, UserPromptPart)
        ]
        return ModelResponse([TextPart(" | ".join(prompts))])

    async def exercise():
        async with Agent.from_config(config_file, model=FunctionModel(respond)) as agent:
            first = await agent.run("我的项目叫 Aurora", tools=[])
            second = await agent.run("刚才的项目叫什么？", tools=[], message_history=first.messages)
            isolated = await agent.run("新的会话", tools=[])
            assert "Aurora" in second.output and "Aurora" not in isolated.output
            assert len(first.messages) == len(second.messages) == 2
            assert first.requests == second.requests == isolated.requests == 1
            assert len(second.execution_trace.responses) == 1
            assert "messages" not in second.as_dict()

    asyncio.run(exercise())


def test_history_boundaries_keep_tool_protocol_and_original_history():
    history = [
        ModelRequest([UserPromptPart("first")]),
        ModelResponse([ToolCallPart("workspace_read", {}, tool_call_id="one")]),
        ModelRequest([ToolReturnPart("workspace_read", "data", tool_call_id="one")]),
        ModelResponse([TextPart("done")]),
        ModelRequest([UserPromptPart("second")]),
        ModelResponse([TextPart("answer")]),
    ]
    selected = select_history(history, turns=1, max_chars=10000)
    assert len(selected) == 2 and selected[0].parts[0].content == "second"
    assert len(history) == 6
    assert select_history(history, turns=0, max_chars=10000) == []
    assert select_history(history, turns=2, max_chars=1) == []
    assert len(select_history(history, turns=2, max_chars=10000)) == 6


@pytest.mark.parametrize("turns", [0, 1, 2, 12])
@pytest.mark.parametrize("bound", [1, 1000, 100000])
def test_context_status_matches_whole_turn_selection_with_utf8_and_attachment(turns, bound):
    history = [
        ModelRequest(
            [UserPromptPart('总结 <attachment name="a.pdf">预算：' + "资" * 3000 + "</attachment>")]
        ),
        ModelResponse([TextPart("已读取")]),
        ModelRequest([UserPromptPart("继续")]),
        ModelResponse([ToolCallPart("workspace_read", {}, tool_call_id="read")]),
        ModelRequest([ToolReturnPart("workspace_read", "来源正文", tool_call_id="read")]),
        ModelResponse([TextPart("已完成")]),
    ]
    before = ModelMessagesTypeAdapter.dump_json(history)
    result = history_status(history, turns=turns, max_chars=bound)
    selected = select_history(history, turns=turns, max_chars=bound)
    assert result["total_turns"] == 2
    assert result["selected_turns"] == sum(
        isinstance(m, ModelRequest) and any(isinstance(p, UserPromptPart) for p in m.parts)
        for m in selected
    )
    assert result["omitted_turns"] + result["selected_turns"] == 2
    assert result["selected_bytes"] <= bound
    assert result["latest_turn_bytes"] == len(ModelMessagesTypeAdapter.dump_json(history[2:]))
    assert result["latest_turn_exceeds_limit"] == bool(
        turns and result["latest_turn_bytes"] > bound
    )
    assert ModelMessagesTypeAdapter.dump_json(history) == before
    assert not history_status([], turns=turns, max_chars=bound)["latest_turn_exceeds_limit"]


def test_approval_history_survives_into_next_turn_without_repeating_write(config_file):
    resolved = load_config(config_file, overrides={"tools": {"allowed": ["workspace.write"]}})

    def respond(messages, info):
        prompts = [
            p.content
            for m in messages
            if isinstance(m, ModelRequest)
            for p in m.parts
            if isinstance(p, UserPromptPart)
        ]
        if len(prompts) > 1:
            assert any(
                isinstance(p, ToolReturnPart)
                for m in messages
                if isinstance(m, ModelRequest)
                for p in m.parts
            )
            return ModelResponse([TextPart("之前已写入 saved.md")])
        if any(
            isinstance(p, ToolReturnPart)
            for m in messages
            if isinstance(m, ModelRequest)
            for p in m.parts
        ):
            return ModelResponse([TextPart("saved")])
        return ModelResponse(
            [
                ToolCallPart(
                    "workspace_write",
                    {
                        "relative_path": "saved.md",
                        "content": "Aurora",
                        "expected_sha256": None,
                    },
                    tool_call_id="write-once",
                )
            ]
        )

    async def exercise():
        async with Agent(resolved, model=FunctionModel(respond)) as agent:
            pending = await agent.run("save Aurora")
            assert not (config_file.parent / "saved.md").exists()
            done = await agent.resume(pending.run_id, {pending.approvals[0].call_id: True})
            assert done.tool_calls == ("workspace.write",)
            again = await agent.run("刚才做了什么？", message_history=done.messages)
            assert "saved.md" in again.output and again.tool_calls == ()
            assert again.requests == 1

    asyncio.run(exercise())
