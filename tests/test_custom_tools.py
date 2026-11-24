"""Ordinary Python tools use the existing permission and task selection path."""

import asyncio

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from personal_harness import Agent, load_config


def test_custom_function_is_explicitly_authorized_scoped_and_traced(config_file):
    calls = []

    def count_words(text: str) -> int:
        """Count space-separated words locally."""
        calls.append(text)
        return len(text.split())

    def respond(messages, info):
        if len(messages) == 1 and info.function_tools:
            assert [t.name for t in info.function_tools] == ["user_count_words"]
            return ModelResponse([ToolCallPart("user_count_words", {"text": "hello world"}, "c")])
        return ModelResponse([TextPart("done")])

    async def exercise():
        async with Agent.from_config(
            config_file,
            overrides={"tools": {"allowed": ["user.count_words"]}},
            tools=[count_words],
            model=FunctionModel(respond),
        ) as agent:
            result = await agent.run("count")
            assert result.tool_calls == ("user.count_words",) and calls == ["hello world"]
            assert result.execution_trace.responses[0].calls[0].tool == "user.count_words"
            result = await agent.run("disabled", tools=[])
            assert result.tool_calls == () and len(calls) == 1
        async with Agent.from_config(
            config_file,
            tools=[count_words],
            model=FunctionModel(respond),
            overrides={"tools": {"allowed": []}},
        ) as agent:
            await agent.run("no permission")
        assert calls == ["hello world"]

    asyncio.run(exercise())


def test_uninstalled_user_tool_is_a_local_configuration_error(config_file):
    resolved = load_config(config_file, overrides={"tools": {"allowed": ["user.absent"]}})
    with pytest.raises(ValueError, match="Supply Python functions"):
        Agent(resolved)
