"""Measure actual model/tool phases and bounded native multi-tool execution."""

import asyncio
import threading
import time

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from personal_harness import Agent, ConfigError, RunError, load_config
from personal_harness.workspace import WorkspaceFiles


def test_native_reads_are_bounded_and_timing_excludes_overlap(config_file, monkeypatch):
    for index in range(6):
        (config_file.parent / f"{index}.md").write_text(f"source {index}", "utf-8")
    active = peak = 0
    lock = threading.Lock()
    original = WorkspaceFiles.read

    def read(self, *args, **kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            time.sleep(0.02)
            return original(self, *args, **kwargs)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(WorkspaceFiles, "read", read)

    async def respond(messages, info):
        await asyncio.sleep(0.005)
        if len(messages) == 1:
            return ModelResponse(
                [
                    ToolCallPart("workspace_read", {"relative_path": f"{i}.md"}, str(i))
                    for i in range(6)
                ]
            )
        return ModelResponse([TextPart("sources read")])

    resolved = load_config(
        config_file,
        overrides={
            "tools": {"allowed": ["workspace.read"]},
            "runtime": {"read_concurrency": 2},
        },
    )

    async def exercise():
        async with Agent(resolved, model=FunctionModel(respond)) as agent:
            result = await agent.run("read six files")
        assert result.output == "sources read" and len(result.tool_calls) == 6
        assert peak == 2 and result.timing["peak_read_concurrency"] == 2
        assert result.timing["tool_sum_ms"] > result.timing["tool_wall_ms"] > 0
        assert result.timing["model_phase_ms"] > 0
        assert 0 < result.timing["first_model_response_ms"] < result.duration_ms
        assert (
            abs(
                sum(
                    result.timing[k]
                    for k in ("model_phase_ms", "tool_wall_ms", "local_overhead_ms")
                )
                - result.duration_ms
            )
            < 0.01
        )
        assert result.as_dict()["timing"] == result.timing

    asyncio.run(exercise())


def test_model_cancellation_reaches_async_provider_and_leaves_no_active_work(config_file):
    finished = asyncio.Event()

    async def respond(messages, info):
        try:
            await asyncio.sleep(30)
        finally:
            finished.set()
        return ModelResponse([TextPart("must not finish")])

    async def exercise():
        resolved = load_config(config_file, overrides={"runtime": {"timeout_seconds": 1}})
        async with Agent(resolved, model=FunctionModel(respond)) as agent:
            with pytest.raises(RunError, match="timeout"):
                await agent.run("wait")
        assert finished.is_set()

    asyncio.run(exercise())


@pytest.mark.parametrize("value", [0, 17, True, "2"])
def test_read_concurrency_is_explicit_and_bounded(config_file, value):
    with pytest.raises(ConfigError):
        load_config(config_file, overrides={"runtime": {"read_concurrency": value}})
