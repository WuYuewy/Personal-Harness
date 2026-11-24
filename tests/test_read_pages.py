import asyncio
import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel

from personal_harness import Agent
from personal_harness.config import ToolSettings
from personal_harness.context import CURRENT_METRICS, RunMetrics
from personal_harness.tools import _read_page


@pytest.mark.parametrize("invalid", [-1, True, 1.5, "24000", 2_000_001])
def test_result_limit_rejects_invalid_values(invalid):
    with pytest.raises(ValidationError):
        ToolSettings(max_read_result_chars=invalid)


def test_paging_keeps_complete_unicode_crlf_lines_and_full_file_identity():
    content = "负责人：林\r\n金额：125\r\n结尾无换行"
    original = {
        "path": "notes.md",
        "content": content,
        "bytes": len(content.encode()),
        "sha256": hashlib.sha256(content.encode()).hexdigest(),
        "start_line": 10,
        "end_line": 12,
    }
    metrics = RunMetrics()
    token = CURRENT_METRICS.set(metrics)
    try:
        page = _read_page(original, len("负责人：林\r\n") + 1)
    finally:
        CURRENT_METRICS.reset(token)
    assert page["content"] == "负责人：林\r\n"
    assert page["start_line"] == page["end_line"] == 10
    assert page["next_start_line"] == 11 and page["requested_end_line"] == 12
    assert page["bytes"] == len(page["content"].encode())
    assert page["sha256"] == original["sha256"]
    assert not page["complete"] and page["truncated"]
    assert metrics.paged_read_results == 1
    assert metrics.read_result_bytes_omitted == len(content.encode()) - page["bytes"]
    assert original["content"] == content


def test_oversized_line_does_not_silently_drop_or_split_evidence():
    original = {"path": "large.json", "content": "x" * 100, "bytes": 100, "sha256": "a" * 64}
    page = _read_page(original, 10)
    assert page["content"] == "" and page["end_line"] is None
    assert page["next_start_line"] == 1 and page["oversized_line"]
    assert not page["complete"] and "increase" in page["notice"]


@pytest.mark.parametrize("limit", [0, 12, 20])
def test_agent_recovers_omitted_middle_fact_with_focused_read(config_file: Path, limit: int):
    content = "header\r\n" + "padding\r\n" * 10 + "deadline: Friday\r\n"
    source = config_file.parent / "notes.md"
    source.write_bytes(content.encode())
    observed = []

    def respond(messages, info):
        returns = [
            part
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        if not returns:
            return ModelResponse([ToolCallPart("workspace_read", {"relative_path": "notes.md"})])
        value = json.loads(returns[-1].content)
        observed.append(value)
        if len(returns) == 1 and limit:
            assert value["content"].startswith("header\r\n") and "Friday" not in value["content"]
            assert value["next_start_line"] >= 2 and not value["complete"]
            return ModelResponse(
                [
                    ToolCallPart(
                        "workspace_read",
                        {"relative_path": "notes.md", "start_line": 12, "end_line": 12},
                    )
                ]
            )
        if limit == 12:
            # The exact requested line exceeds 12 characters. It is explicitly unavailable,
            # rather than a misleading fragment that a model might call the full deadline.
            assert value["oversized_line"] and value["next_start_line"] == 12
            return ModelResponse([TextPart("This line requires a larger configured result limit.")])
        assert "deadline: Friday" in value["content"]
        return ModelResponse([TextPart("Friday (notes.md:12)")])

    async def exercise():
        async with Agent.from_config(
            config_file,
            overrides={"tools": {"allowed": ["workspace.read"], "max_read_result_chars": limit}},
            model=FunctionModel(respond),
        ) as agent:
            return await agent.run("Find the deadline in notes.md")

    result = asyncio.run(exercise())
    assert result.paged_read_results == (2 if limit == 12 else 1 if limit else 0)
    assert source.read_bytes() == content.encode()
    assert all(
        value["sha256"] == hashlib.sha256(content.encode()).hexdigest() for value in observed
    )
    assert result.as_dict()["usage"]["paged_read_results"] == result.paged_read_results


def test_batch_shares_body_budget_and_keeps_errors_and_continuations(config_file: Path):
    (config_file.parent / "first.md").write_bytes(b"one\ntwo\n")
    (config_file.parent / "second.md").write_bytes(b"three\nfour\n")

    def respond(messages, info):
        returns = [
            part
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        if not returns:
            return ModelResponse(
                [
                    ToolCallPart(
                        "workspace_read_many",
                        {
                            "reads": [
                                {"relative_path": name}
                                for name in ("first.md", "missing", "second.md")
                            ]
                        },
                    )
                ]
            )
        batch = json.loads(returns[-1].content)
        assert not batch["complete"]
        assert batch["results"][0]["content"] == "one\ntwo\n"
        assert "error" in batch["results"][1]
        assert batch["results"][2]["content"] == ""
        assert batch["results"][2]["next_start_line"] == 1
        assert not batch["results"][2]["oversized_line"]
        assert sum(len(item.get("content", "")) for item in batch["results"]) <= 8
        return ModelResponse([TextPart("Some sources still need a focused read.")])

    async def exercise():
        async with Agent.from_config(
            config_file,
            overrides={"tools": {"allowed": ["workspace.read_many"], "max_read_result_chars": 8}},
            model=FunctionModel(respond),
        ) as agent:
            return await agent.run("Read these sources")

    result = asyncio.run(exercise())
    assert result.paged_read_results == 1
    assert result.read_result_bytes_omitted == len(b"three\nfour\n")
