"""Content-free diagnostics derived from PydanticAI's captured message history."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)

from personal_harness.config import ToolName
from personal_harness.tools import logical_tool_name

Count = Annotated[int, Field(strict=True, ge=0)]


class _TraceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ToolCallTrace(_TraceRecord):
    tool: ToolName | Literal["unregistered"]
    outcome: Literal["unresolved", "success", "failed", "denied", "interrupted", "retried"]
    result_bytes: Count | None = None

    @model_validator(mode="after")
    def return_matches_outcome(self) -> ToolCallTrace:
        if (self.outcome == "unresolved") != (self.result_bytes is None):
            raise ValueError("Only an unresolved tool call has no recorded result size.")
        return self


class ResponseTrace(_TraceRecord):
    state: Literal["complete", "incomplete", "suspended", "interrupted"]
    calls: tuple[ToolCallTrace, ...] = ()


class ExecutionTrace(_TraceRecord):
    schema_version: Literal[1] = 1
    responses: tuple[ResponseTrace, ...] = ()
    unmatched_tool_returns: Count = 0

    @property
    def multi_call_responses(self) -> int:
        return sum(len(response.calls) > 1 for response in self.responses)

    @property
    def tool_result_bytes(self) -> int:
        return sum(call.result_bytes or 0 for response in self.responses for call in response.calls)


def summarize_messages(messages: Sequence[ModelMessage]) -> ExecutionTrace:
    """Count each recorded return once, including feedback; never retain its content.

    IDs/names are used transiently for pairing. Unknown names are redacted and repeated IDs
    in later responses pair with the latest unanswered call, not an earlier completed one.
    """
    calls: list[list[ToolCallTrace]] = []
    states: list[Literal["complete", "incomplete", "suspended", "interrupted"]] = []
    pending: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    unmatched = 0
    for message in messages:
        if isinstance(message, ModelResponse):
            response_index = len(calls)
            calls.append([])
            states.append(message.state)
            for part in message.parts:
                if isinstance(part, ToolCallPart):
                    pending[(part.tool_name, part.tool_call_id)].append(
                        (response_index, len(calls[-1]))
                    )
                    calls[-1].append(
                        ToolCallTrace(
                            tool=logical_tool_name(part.tool_name),
                            outcome="unresolved",
                        )
                    )
        elif isinstance(message, ModelRequest):
            for returned in message.parts:
                if not isinstance(returned, ToolReturnPart):
                    continue
                matches = pending.get((returned.tool_name, returned.tool_call_id))
                if not matches:
                    unmatched += 1
                    continue
                response_index, call_index = matches.pop()
                previous = calls[response_index][call_index]
                calls[response_index][call_index] = ToolCallTrace(
                    tool=previous.tool,
                    outcome=returned.outcome,
                    result_bytes=len(returned.model_response_str().encode("utf-8")),
                )
    return ExecutionTrace(
        responses=tuple(
            ResponseTrace(state=state, calls=tuple(items))
            for state, items in zip(states, calls, strict=True)
        ),
        unmatched_tool_returns=unmatched,
    )
