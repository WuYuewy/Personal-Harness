"""Framework wrappers for bounded reads and local phase timing; no extra LLM calls."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from time import perf_counter
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import ModelRequestParameters, StreamedResponse
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.abstract import ToolsetTool
from pydantic_ai.toolsets.wrapper import WrapperToolset

from personal_harness.context import CURRENT_METRICS, RunDependencies


class TimedModel(WrapperModel):
    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        metrics = CURRENT_METRICS.get()
        started = perf_counter()
        try:
            response = await self.wrapped.request(
                messages, model_settings, model_request_parameters
            )
            if metrics is not None and metrics.first_response_ms is None:
                metrics.first_response_ms = (
                    metrics.active_seconds + perf_counter() - metrics.phase_started
                ) * 1000
            return response
        finally:
            if metrics is not None:
                metrics.model_seconds += perf_counter() - started

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncGenerator[StreamedResponse]:
        metrics = CURRENT_METRICS.get()
        started = perf_counter()
        try:
            async with self.wrapped.request_stream(
                messages, model_settings, model_request_parameters, run_context
            ) as response:
                if metrics is not None and metrics.first_response_ms is None:
                    metrics.first_response_ms = (
                        metrics.active_seconds + perf_counter() - metrics.phase_started
                    ) * 1000
                yield response
        finally:
            if metrics is not None:
                metrics.model_seconds += perf_counter() - started


class BoundedTools(WrapperToolset[RunDependencies]):
    """Use the framework scheduler; only bound actual read execution locally.

    Write tools retain the framework's sequential barrier and deferred approval.
    Timing uses the union of overlapping tool intervals, not their sum, for residuals.
    """

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[RunDependencies],
        tool: ToolsetTool[RunDependencies],
    ) -> Any:
        metrics = CURRENT_METRICS.get()
        read = name not in {"workspace_write", "workspace_replace_text"}
        if metrics is None:
            return await self.wrapped.call_tool(name, tool_args, ctx, tool)
        if metrics.read_gate is None:
            metrics.read_gate = asyncio.Semaphore(ctx.deps.resolved.config.runtime.read_concurrency)
        gate = metrics.read_gate if read else asyncio.Semaphore(1)
        async with gate:
            started = perf_counter()
            if metrics.active_tools == 0:
                metrics.tool_interval_started = started
            metrics.active_tools += 1
            if read:
                metrics.active_reads += 1
                metrics.peak_reads = max(metrics.peak_reads, metrics.active_reads)
            try:
                return await self.wrapped.call_tool(name, tool_args, ctx, tool)
            finally:
                ended = perf_counter()
                metrics.tool_seconds += ended - started
                metrics.active_tools -= 1
                if read:
                    metrics.active_reads -= 1
                if metrics.active_tools == 0:
                    metrics.tool_wall_seconds += ended - metrics.tool_interval_started


def bounded_tools(tools: list[Any]) -> BoundedTools:
    return BoundedTools(FunctionToolset(tools=tools))
