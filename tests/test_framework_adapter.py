import asyncio
from pathlib import Path

from personal_harness import Agent


def test_pydantic_ai_executes_allowlisted_tool_and_reports_usage(config_file: Path) -> None:
    result = asyncio.run(Agent.from_config(config_file, profile="desktop").run("Inspect my setup"))
    assert result.provider == "fake"
    assert result.requests == 2
    assert result.input_tokens > 0
    assert result.output_tokens > 0
    assert result.tool_calls == ("context.describe",)
    assert result.events.count("model.response") == 2
    assert result.as_dict()["engine"] == "pydantic-ai"


def test_pydantic_ai_receives_no_tool_when_user_disables_it(config_file: Path) -> None:
    result = asyncio.run(
        Agent.from_config(config_file, overrides={"tools": {"allowed": []}}).run("Inspect my setup")
    )
    assert result.requests == 1
    assert result.tool_calls == ()
    assert "Context tool: disabled" in result.output
