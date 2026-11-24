"""Compact identical single-file reads in outgoing requests, keeping the run history intact."""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import Any, cast

from pydantic_ai import RunContext
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters, StreamedResponse
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings

from personal_harness.context import CURRENT_METRICS

READ_ONLY_TOOL_NAMES = frozenset(
    {
        "context_describe",
        "workspace_list",
        "workspace_read",
        "workspace_read_many",
        "workspace_search",
        "places_search",
    }
)


def reference_past_reads(
    messages: list[ModelMessage], *, can_read: bool, min_chars: int = 2000
) -> tuple[list[ModelMessage], int, int]:
    """Reference large reads older than the previous turn; leave the transcript intact.

    Only recognized, successful single-file reads with a matching call are eligible.
    Keep the current and most recent preceding user turn in full, as well as errors,
    writes, approvals, user text and assistant messages. No semantic summarization.
    Do not omit evidence if the model cannot request it again with its active tools.
    """
    if not can_read:
        return messages, 0, 0
    boundaries = [
        index
        for index, message in enumerate(messages)
        if isinstance(message, ModelRequest)
        and any(isinstance(part, UserPromptPart) for part in message.parts)
    ]
    if len(boundaries) < 3:
        return messages, 0, 0
    cutoff = boundaries[-2]
    current = messages[boundaries[-1]]
    assert isinstance(current, ModelRequest)
    prompts = [part.content for part in current.parts if isinstance(part, UserPromptPart)]
    # Unfamiliar rich/multimodal inputs may refer to a source we cannot identify locally.
    if any(not isinstance(prompt, str) for prompt in prompts):
        return messages, 0, 0
    current_text = "\n".join(str(prompt) for prompt in prompts).replace("\\", "/").casefold()
    # Ambiguous follow-ups retain all evidence. Only target tasks naming concrete files.
    if not re.search(r"[\w./-]+\.[a-z][a-z0-9]{0,11}\b", current_text):
        return messages, 0, 0
    if re.search(
        r"(?:之前|前面|上次|最初|早先|刚才).{0,12}(?:文件|档案|资料|报告)"
        r"|(?:earlier|previous|prior|above).{0,20}(?:file|document|source|record)",
        current_text,
    ):
        return messages, 0, 0
    calls: dict[str, str | None] = {}
    changed: dict[int, ModelRequest] = {}
    count = saved = 0
    for index, message in enumerate(messages[:cutoff]):
        if isinstance(message, ModelResponse):
            for part in message.parts:
                if isinstance(part, ToolCallPart):
                    try:
                        arguments = part.args_as_dict()
                    except (ValueError, TypeError):
                        arguments = {}
                    path = arguments.get("relative_path")
                    calls[part.tool_call_id] = (
                        path
                        if part.tool_name == "workspace_read" and isinstance(path, str)
                        else None
                    )
        elif isinstance(message, ModelRequest):
            # Mixed user/tool requests are never changed.
            mixed = any(isinstance(part, UserPromptPart) for part in message.parts)
            parts = list(message.parts)
            for offset, returned in enumerate(message.parts):
                if not isinstance(returned, ToolReturnPart):
                    continue
                path = calls.pop(returned.tool_call_id, None)
                if (
                    mixed
                    or path is None
                    or returned.tool_name != "workspace_read"
                    or returned.outcome != "success"
                    or not isinstance(returned.content, str)
                    or len(returned.content) < min_chars
                ):
                    continue
                # Explicit references in the current task stay available in full. This avoids
                # paying for a re-read or relying on the model to recover omitted evidence.
                normalized_path = path.replace("\\", "/").casefold()
                if (
                    normalized_path in current_text
                    or normalized_path.rsplit("/", 1)[-1] in current_text
                ):
                    continue
                try:
                    data = json.loads(returned.content)
                except (ValueError, TypeError):
                    continue
                if (
                    not isinstance(data, dict)
                    or set(data)
                    not in (
                        {"path", "content", "bytes", "sha256"},
                        {"path", "content", "bytes", "sha256", "start_line", "end_line"},
                    )
                    or data.get("path") != path
                    or not isinstance(data.get("content"), str)
                    or not isinstance(data.get("sha256"), str)
                ):
                    continue
                marker = {key: value for key, value in data.items() if key != "content"}
                marker["content_omitted"] = (
                    "Older file-read body omitted from model context; full result remains in "
                    "the local transcript. This fingerprint is historical, not current. "
                    "Use workspace_read to retrieve needed facts before relying on this source."
                )
                replacement = json.dumps(marker, ensure_ascii=False)
                reduction = len(returned.content.encode()) - len(replacement.encode())
                if reduction <= 0:
                    continue
                parts[offset] = replace(cast(ToolReturnPart[Any], returned), content=replacement)
                count += 1
                saved += reduction
            if parts != list(message.parts):
                changed[index] = replace(message, parts=parts)
    return [changed.get(i, message) for i, message in enumerate(messages)], count, saved


class _HistoryProjection(WrapperModel):
    def project(
        self, messages: list[ModelMessage], parameters: ModelRequestParameters
    ) -> list[ModelMessage]:
        raise NotImplementedError

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        outgoing = self.project(messages, model_request_parameters)
        return await self.wrapped.request(outgoing, model_settings, model_request_parameters)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        outgoing = self.project(messages, model_request_parameters)
        async with self.wrapped.request_stream(
            outgoing, model_settings, model_request_parameters, run_context
        ) as response:
            yield response


class PastReadModel(_HistoryProjection):
    """Outgoing-only context reduction, shared by CLI, SDK and persistent UI chats."""

    def project(
        self,
        messages: list[ModelMessage],
        parameters: ModelRequestParameters,
    ) -> list[ModelMessage]:
        outgoing, count, saved = reference_past_reads(
            messages,
            can_read=any(tool.name == "workspace_read" for tool in parameters.function_tools),
        )
        metrics = CURRENT_METRICS.get()
        if metrics is not None:
            metrics.history_reads_referenced += count
            metrics.history_read_bytes_omitted += saved
        return outgoing


def compact_reads(messages: list[ModelMessage]) -> tuple[list[ModelMessage], int, int]:
    """Keep the newest full result for exact repeated successful read-only calls.

    Copy only changed requests/parts. Never remove messages or calls, alter an approval,
    or look outside the most recent user turn. A result is eligible only when its tool name,
    canonical arguments and result all match. This is not a semantic summary or a cache.
    """
    start: int | None = None
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, ModelRequest) and any(
            isinstance(part, UserPromptPart) for part in message.parts
        ):
            start = index
            break
    if start is None:
        return messages, 0, 0

    calls: dict[str, tuple[str, str]] = {}
    for message in messages[start:]:
        if not isinstance(message, ModelResponse):
            continue
        for response_part in message.parts:
            if (
                not isinstance(response_part, ToolCallPart)
                or response_part.tool_name not in READ_ONLY_TOOL_NAMES
            ):
                continue
            try:
                arguments = json.dumps(
                    response_part.args_as_dict(),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            except (TypeError, ValueError):
                continue
            calls[response_part.tool_call_id] = (response_part.tool_name, arguments)

    seen: set[tuple[str, str, str]] = set()
    changed: dict[int, ModelRequest] = {}
    count = saved = 0
    for index in range(len(messages) - 1, start - 1, -1):
        message = messages[index]
        if not isinstance(message, ModelRequest):
            continue
        # A mixed user/tool request is a boundary too; conservatively leave it intact.
        if any(isinstance(part, UserPromptPart) for part in message.parts):
            break
        parts = list(message.parts)
        for offset in range(len(parts) - 1, -1, -1):
            request_part = parts[offset]
            if (
                not isinstance(request_part, ToolReturnPart)
                or request_part.tool_name not in READ_ONLY_TOOL_NAMES
                or request_part.outcome != "success"
                or not isinstance(request_part.content, str)
            ):
                continue
            call = calls.get(request_part.tool_call_id)
            if call is None or call[0] != request_part.tool_name:
                continue
            key = (request_part.tool_name, call[1], request_part.content)
            if key not in seen:
                seen.add(key)
                continue
            marker: dict[str, Any] = {}
            if request_part.tool_name == "workspace_read":
                try:
                    data = json.loads(request_part.content)
                except (TypeError, ValueError):
                    continue
                if not isinstance(data, dict) or set(data) not in (
                    {"path", "content", "bytes", "sha256"},
                    {"path", "content", "bytes", "sha256", "start_line", "end_line"},
                ):
                    continue
                if not all(isinstance(data[key], str) for key in ("path", "content", "sha256")):
                    continue
                marker = {key: value for key, value in data.items() if key != "content"}
            marker["content_omitted"] = (
                "Exact successful read-only result repeated for the same tool and arguments; "
                "the latest identical result remains in this task history."
            )
            replacement = json.dumps(marker, ensure_ascii=False, separators=(",", ":"))
            reduction = len(request_part.content.encode("utf-8")) - len(replacement.encode("utf-8"))
            if reduction <= 0:
                continue
            parts[offset] = replace(cast(ToolReturnPart[Any], request_part), content=replacement)
            count += 1
            saved += reduction
        if parts != list(message.parts):
            changed[index] = replace(message, parts=parts)
    return [changed.get(i, message) for i, message in enumerate(messages)], count, saved


class ReadHistoryModel(_HistoryProjection):
    """Project identical reads out of the payload without mutating chat history."""

    def project(
        self,
        messages: list[ModelMessage],
        parameters: ModelRequestParameters,
    ) -> list[ModelMessage]:
        outgoing, count, saved = compact_reads(messages)
        metrics = CURRENT_METRICS.get()
        if metrics is not None:
            metrics.compacted_read_results += count
            metrics.compacted_read_bytes += saved
        return outgoing
