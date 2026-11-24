"""Past-read reduction must save outgoing context without changing task/approval history."""

import asyncio
import hashlib
import json
from copy import deepcopy
from dataclasses import replace

import pytest
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import FunctionModel

from personal_harness import Agent, ConfigError, load_config
from personal_harness.history import reference_past_reads


def read_result(text="Evidence line.\r\n" * 150, **updates):
    return json.dumps(
        {
            "path": "notes.md",
            "content": text,
            "bytes": len(text.encode()),
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
            **updates,
        }
    )


def history(result):
    return [
        ModelRequest([UserPromptPart("Read notes")]),
        ModelResponse([ToolCallPart("workspace_read", {"relative_path": "notes.md"}, "read-0")]),
        ModelRequest([ToolReturnPart("workspace_read", result, "read-0")]),
    ]


def old_history():
    return [
        *history(read_result()),
        ModelResponse([TextPart("Source inspected; I prefer short answers.")]),
        ModelRequest([UserPromptPart("Second task, retain my preference")]),
        ModelResponse([TextPart("Understood")]),
        ModelRequest([UserPromptPart("Third task: use current.md")]),
    ]


def test_old_reads_preserve_complete_turns_and_do_not_mutate_saved_history():
    original = old_history()
    original[-2].parts.append(ToolCallPart("workspace_write", {}, "pending-approval"))
    snapshot = deepcopy(original)
    outgoing, count, saved = reference_past_reads(original, can_read=True)
    assert original == snapshot and count == 1 and saved > 2000
    assert len(outgoing) == len(original)
    marker = json.loads(outgoing[2].parts[0].content)
    assert marker["path"] == "notes.md" and marker["sha256"]
    assert "historical, not current" in marker["content_omitted"]
    assert "content" not in marker
    assert outgoing[2].parts[0].tool_call_id == original[2].parts[0].tool_call_id
    assert outgoing[2].parts[0].outcome == "success"
    assert all(
        old is new for old, new in zip(original, outgoing, strict=True) if old is not original[2]
    )
    # The same saved transcript may be projected again, without state leaking to another chat.
    assert reference_past_reads(original, can_read=True)[1:] == (count, saved)
    assert reference_past_reads(outgoing, can_read=True) == (outgoing, 0, 0)


def test_current_and_most_recent_previous_turn_stay_in_full():
    original = [*history(read_result()), *history(read_result())]
    assert reference_past_reads(original, can_read=True) == (original, 0, 0)
    assert reference_past_reads(old_history(), can_read=False)[1:] == (0, 0)


@pytest.mark.parametrize(
    "case",
    [
        "failed",
        "denied",
        "retried",
        "write",
        "unknown-tool",
        "orphan",
        "mismatch",
        "short",
        "unrecognized",
        "mixed",
        "invalid-json",
    ],
)
def test_only_known_retrievable_successful_reads_are_referenced(case):
    original = old_history()
    returned = original[2].parts[0]
    if case in {"failed", "denied", "retried"}:
        returned = replace(returned, outcome=case)
    elif case in {"write", "unknown-tool"}:
        returned = replace(returned, tool_name="workspace_write" if case == "write" else "user_x")
    elif case == "orphan":
        returned = replace(returned, tool_call_id="unmatched")
    elif case == "mismatch":
        returned = replace(returned, content=read_result(path="another.md"))
    elif case == "short":
        returned = replace(returned, content=read_result("short"))
    elif case == "unrecognized":
        returned = replace(returned, content=read_result(extra="not a supported result"))
    elif case == "invalid-json":
        returned = replace(returned, content="not-json" * 600)
    original[2].parts[0] = returned
    if case == "mixed":
        original[2].parts.append(UserPromptPart("Preserve mixed request"))
    assert reference_past_reads(original, can_read=True) == (original, 0, 0)


@pytest.mark.parametrize("mode", ["full", "references"])
def test_sdk_real_request_projection_rereads_missing_facts_with_current_permissions(
    config_file, mode
):
    source = config_file.parent / "current.md"
    source.write_text("Current deadline: Monday\n" + "Reference text.\n" * 200, "utf-8")
    seen = []

    def respond(messages, info):
        seen.append(deepcopy(messages))
        returned = [
            p
            for m in messages
            if isinstance(m, ModelRequest)
            for p in m.parts
            if isinstance(p, ToolReturnPart)
        ]
        if len(returned) == 1:
            return ModelResponse(
                [ToolCallPart("workspace_read", {"relative_path": "current.md"}, "fresh")]
            )
        assert "Monday" in json.loads(returned[-1].content)["content"]
        return ModelResponse([TextPart("Monday (notes.md)")])

    original = old_history()[:-1]
    snapshot = deepcopy(original)

    async def exercise():
        async with Agent.from_config(
            config_file,
            overrides={
                "tools": {"allowed": ["workspace.read"]},
                "runtime": {"history_read_mode": mode},
            },
            model=FunctionModel(respond),
        ) as agent:
            result = await agent.run("Read deadline from current.md", message_history=original)
            assert result.output == "Monday (notes.md)" and result.requests == 2
            assert original == snapshot
            assert len(result.messages) == 4
            assert result.history_reads_referenced == (2 if mode == "references" else 0)
            assert (result.history_read_bytes_omitted > 0) == (mode == "references")
            assert result.as_dict()["usage"]["history_reads_referenced"] == (
                result.history_reads_referenced
            )
            assert result.execution_trace.unmatched_tool_returns == 0

    asyncio.run(exercise())
    first_old = json.loads(seen[0][2].parts[0].content)
    assert ("content_omitted" in first_old) == (mode == "references")
    # The newly read source and the persisted original transcript are always full-fidelity.
    assert "Current deadline: Monday" in seen[-1][-1].parts[0].content


def test_history_read_mode_is_strict_and_configurable(config_file):
    for invalid in (True, None, "automatic"):
        with pytest.raises(ConfigError):
            load_config(config_file, overrides={"runtime": {"history_read_mode": invalid}})


def test_current_task_named_source_is_not_omitted():
    original = old_history()
    original[-1] = ModelRequest([UserPromptPart("Use NOTES.md to prepare a report")])
    assert reference_past_reads(original, can_read=True) == (original, 0, 0)


@pytest.mark.parametrize(
    "prompt",
    [
        "继续",
        "Summarize the previous file",
        "总结之前的档案到 out/report.md",
        "参考上次的资料修改 current.md",
        "Please use notes.md",
        "Use folder/NOTES.md",
    ],
)
def test_ambiguous_followups_and_explicit_old_sources_keep_evidence(prompt):
    original = old_history()
    original[-1] = ModelRequest([UserPromptPart(prompt)])
    assert reference_past_reads(original, can_read=True) == (original, 0, 0)


@pytest.mark.parametrize(
    "preference,expected",
    [
        ("prioritize_cost", "references"),
        ("balanced", "full"),
        ("prioritize_quality", "full"),
    ],
)
def test_personal_cost_preference_selects_execution_policy_and_explicit_override_wins(
    config_file, preference, expected
):
    resolved = load_config(config_file, overrides={"experience": {"cost_strategy": preference}})
    assert resolved.config.effective_history_read_mode == expected
    assert resolved.config.runtime.history_read_mode == "auto"
    manual = load_config(
        config_file,
        overrides={
            "experience": {"cost_strategy": preference},
            "runtime": {"history_read_mode": "full"},
        },
    )
    assert manual.config.effective_history_read_mode == "full"
