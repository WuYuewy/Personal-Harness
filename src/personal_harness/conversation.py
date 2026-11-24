"""Bound model context by complete turns without deleting the saved transcript."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import TypedDict

from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    UserPromptPart,
)


def _starts_turn(message: ModelMessage) -> bool:
    return isinstance(message, ModelRequest) and any(
        isinstance(part, UserPromptPart) for part in message.parts
    )


def _history_groups(messages: Sequence[ModelMessage]) -> list[list[ModelMessage]]:
    groups: list[list[ModelMessage]] = []
    for message in messages:
        if _starts_turn(message):
            groups.append([])
        if groups:
            groups[-1].append(message)
    return groups


def _turn_size(messages: list[ModelMessage]) -> int:
    return len(ModelMessagesTypeAdapter.dump_json(messages))


def latest_turn_size(messages: Sequence[ModelMessage]) -> int:
    for index in range(len(messages) - 1, -1, -1):
        if _starts_turn(messages[index]):
            return _turn_size(list(messages[index:]))
    return 0


class HistoryStatus(TypedDict):
    total_turns: int
    selected_turns: int
    selected_messages: int
    selected_bytes: int
    omitted_turns: int
    latest_turn_bytes: int
    latest_turn_exceeds_limit: bool


def _select_groups(
    groups: list[list[ModelMessage]], *, turns: int, max_chars: int
) -> tuple[list[list[ModelMessage]], int]:
    selected: list[list[ModelMessage]] = []
    size = 0
    for group in reversed(groups[-turns:] if turns else []):
        length = _turn_size(group)
        if size + length > max_chars:
            break
        selected.append(group)
        size += length
    selected.reverse()
    return selected, size


def history_status(
    messages: Sequence[ModelMessage], *, turns: int, max_chars: int
) -> HistoryStatus:
    """Describe exact whole-turn selection before a request, using serialized UTF-8 bytes."""
    groups = _history_groups(messages)
    selected, selected_bytes = _select_groups(groups, turns=turns, max_chars=max_chars)
    latest_bytes = _turn_size(groups[-1]) if groups else 0
    return {
        "total_turns": len(groups),
        "selected_turns": len(selected),
        "selected_messages": sum(map(len, selected)),
        "selected_bytes": selected_bytes,
        "omitted_turns": len(groups) - len(selected),
        "latest_turn_bytes": latest_bytes,
        "latest_turn_exceeds_limit": bool(turns and groups and latest_bytes > max_chars),
    }


def select_history(
    messages: Sequence[ModelMessage], *, turns: int, max_chars: int
) -> list[ModelMessage]:
    if not turns:
        return []
    groups, _ = _select_groups(_history_groups(messages), turns=turns, max_chars=max_chars)
    return deepcopy([message for group in groups for message in group])
