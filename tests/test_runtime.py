import asyncio
import socket
from decimal import Decimal
from pathlib import Path

import pytest

from personal_harness import Agent, RunResult


def test_same_task_uses_each_personal_configuration_without_network(
    config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("Offline rehearsal must not connect to the network")

    async def exercise() -> tuple[RunResult, RunResult]:
        # Windows asyncio creates a local socketpair while starting its event loop.
        # Block connections only during the actual Agent calls, after loop creation.
        with monkeypatch.context() as network:
            network.setattr(socket.socket, "connect", no_network)
            desktop = await Agent.from_config(config_file, profile="desktop").run("Plan my day")
            mobile = await Agent.from_config(
                config_file,
                profile="mobile",
                overrides={"preferences": {"city": "Shanghai"}, "user": {"display_name": "Alex"}},
            ).run("Plan my day")
            return desktop, mobile

    desktop, mobile = asyncio.run(exercise())
    assert desktop.context["preferences"]["response_style"] == "detailed"
    assert mobile.context["preferences"]["response_style"] == "concise"
    assert mobile.context["preferences"]["city"] == "Shanghai"
    assert mobile.context["user"]["display_name"] == "Alex"
    assert desktop.context["instructions"] != mobile.context["instructions"]
    assert desktop.context["budget"] != mobile.context["budget"]
    assert "Alex" in mobile.output and "Shanghai" in mobile.output
    assert desktop.tool_calls == ("context.describe",)
    assert desktop.cost_usd == mobile.cost_usd == Decimal("0")
    assert "OFFLINE DEMO" in desktop.output
    assert desktop.events[-1] == "run.completed"


def test_disallowed_context_tool_is_not_called(config_file: Path) -> None:
    result = asyncio.run(
        Agent.from_config(config_file, overrides={"tools": {"allowed": []}}).run("Hello")
    )
    assert result.tool_calls == ()
    assert "tool.context.describe.completed" not in result.events
    assert "Context tool: disabled" in result.output


def test_user_can_define_a_profile_without_a_runtime_code_branch(config_file: Path) -> None:
    config_file.write_text(
        config_file.read_text("utf-8")
        + '\n[profiles.research]\ninstructions = ["Compare evidence before drawing conclusions."]\n'
        + '[profiles.research.preferences]\npriorities = ["evidence", "control_cost"]\n'
        + "[profiles.research.tools]\nallowed = []\n",
        "utf-8",
    )
    result = asyncio.run(Agent.from_config(config_file, profile="research").run("Review my notes"))
    assert result.profile == "research"
    assert result.context["preferences"]["priorities"] == ["evidence", "control_cost"]
    assert result.context["instructions"] == ["Compare evidence before drawing conclusions."]
    assert result.context["user"]["display_name"] == "You"
    assert result.tool_calls == ()


def test_run_uses_snapshot_until_explicitly_reloaded(config_file: Path) -> None:
    agent = Agent.from_config(config_file)
    config_file.write_text(
        config_file.read_text("utf-8").replace('display_name = "You"', 'display_name = "Changed"'),
        "utf-8",
    )
    assert asyncio.run(agent.run("Hi")).context["user"]["display_name"] == "You"
    assert (
        asyncio.run(Agent.from_config(config_file).run("Hi")).context["user"]["display_name"]
        == "Changed"
    )


def test_blank_task_fails(config_file: Path) -> None:
    with pytest.raises(ValueError, match="empty"):
        asyncio.run(Agent.from_config(config_file).run("  "))


def test_stream_callback_receives_text_without_changing_result(config_file: Path) -> None:
    from pydantic_ai.models.function import DeltaThinkingPart, FunctionModel

    async def stream(messages, info):
        yield {0: DeltaThinkingPart(content="private reasoning")}
        yield {0: DeltaThinkingPart(content=" more reasoning")}
        yield "Hel"
        yield "lo"

    async def exercise() -> tuple[RunResult, list[str]]:
        chunks: list[str] = []

        async def callback(chunk: str) -> None:
            chunks.append(chunk)

        async with Agent.from_config(
            config_file, model=FunctionModel(stream_function=stream)
        ) as agent:
            result = await agent.run("Say hello", on_text_delta=callback)
        return result, chunks

    result, chunks = asyncio.run(exercise())
    assert result.output == "Hello"
    assert chunks == ["Hel", "lo"]


def test_stream_callback_errors_do_not_fail_task(config_file: Path) -> None:
    from pydantic_ai.models.function import FunctionModel

    async def stream(messages, info):
        yield "answer"

    async def callback(chunk: str) -> None:
        raise RuntimeError("presentation failed")

    async def exercise() -> RunResult:
        async with Agent.from_config(
            config_file, model=FunctionModel(stream_function=stream)
        ) as agent:
            return await agent.run("Say answer", on_text_delta=callback)

    result = asyncio.run(exercise())
    assert result.output == "answer"
