import asyncio
import hashlib
import json
from copy import deepcopy
from dataclasses import replace

import pytest
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import FunctionModel

from personal_harness import Agent, ConfigError, RunError, load_config
from personal_harness.history import compact_reads
from personal_harness.trace import summarize_messages


def read_result(text="Evidence line.\r\n" * 150, **updates):
    return json.dumps(
        {
            "path": "notes.md",
            "content": text,
            "bytes": len(text.encode()),
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
            **updates,
        },
        ensure_ascii=False,
    )


def history(*returns):
    messages = [ModelRequest([SystemPromptPart("Keep instructions"), UserPromptPart("Read notes")])]
    for index, returned in enumerate(returns):
        call_id = f"read-{index}"
        messages.extend(
            [
                ModelResponse(
                    [ToolCallPart("workspace_read", {"relative_path": "notes.md"}, call_id)]
                ),
                ModelRequest([ToolReturnPart("workspace_read", returned, call_id)]),
            ]
        )
    return messages


def test_exact_duplicates_keep_latest_full_result_pairing_and_source_metadata():
    original = history(
        read_result(start_line=4, end_line=153), read_result(start_line=4, end_line=153)
    )
    snapshot = deepcopy(original)
    compacted, count, saved = compact_reads(original)
    assert count == 1 and saved > 0 and original == snapshot
    assert len(compacted) == len(original)
    assert compacted[0] is original[0] and compacted[-1] is original[-1]
    marker = json.loads(compacted[2].parts[0].content)
    full = json.loads(compacted[-1].parts[0].content)
    assert "content" not in marker and marker.pop("content_omitted")
    assert marker == {key: value for key, value in full.items() if key != "content"}
    for before, after in zip(original, compacted, strict=True):
        if isinstance(before, ModelResponse):
            assert after is before
        else:
            assert [p.part_kind for p in after.parts] == [p.part_kind for p in before.parts]
            for old, new in zip(before.parts, after.parts, strict=True):
                if isinstance(old, ToolReturnPart):
                    assert old.tool_call_id == new.tool_call_id and old.outcome == new.outcome
    # Diagnostics continue to describe actual results, before outgoing-only compaction.
    assert summarize_messages(original) == summarize_messages(snapshot)


@pytest.mark.parametrize(
    "returned",
    [
        "not json" * 200,
        json.dumps(["invalid"] * 200),
        json.dumps({"error": "failed" * 200}),
        read_result(content=123),
        read_result(extra="preserve unfamiliar data"),
        read_result("short"),
        {"content": "structured data" * 200},
    ],
)
def test_unrecognized_and_short_results_remain_intact(returned):
    messages = history(returned, returned)
    compacted, count, saved = compact_reads(messages)
    assert compacted == messages and count == saved == 0


@pytest.mark.parametrize("difference", ["path", "sha256", "content", "lines"])
def test_changed_sources_and_excerpts_are_not_equated(difference):
    changed = {
        "path": read_result(path="other.md"),
        "sha256": read_result(sha256="0" * 64),
        "content": read_result("Changed.\r\n" * 150),
        "lines": read_result(start_line=1, end_line=150),
    }[difference]
    messages = history(read_result(), changed)
    compacted, count, saved = compact_reads(messages)
    assert compacted == messages and count == saved == 0


@pytest.mark.parametrize("outcome", ["failed", "denied", "retried", "interrupted"])
def test_errors_and_other_tools_are_preserved(outcome):
    messages = history(read_result(), read_result())
    messages[-1].parts[0] = replace(messages[-1].parts[0], outcome=outcome)
    assert compact_reads(messages) == (messages, 0, 0)
    messages[-1].parts[0] = replace(
        messages[-1].parts[0], outcome="success", tool_name="workspace_read_many"
    )
    assert compact_reads(messages) == (messages, 0, 0)


def test_compaction_stops_at_user_turn_and_does_not_keep_state():
    messages = history(read_result())
    messages.extend(history(read_result()))
    assert compact_reads(messages) == (messages, 0, 0)
    repeated = history(read_result(), read_result(), read_result())
    compacted, count, saved = compact_reads(repeated)
    assert count == 2 and saved > 0
    assert compact_reads(compacted) == (compacted, 0, 0)
    assert compact_reads(history(read_result()))[1:] == (0, 0)


def test_compaction_requires_an_explicit_current_user_turn_boundary():
    messages = history(read_result(), read_result())[1:]
    assert compact_reads(messages) == (messages, 0, 0)


def test_changed_version_is_retained_in_full_even_with_other_duplicates():
    first = read_result("Original evidence\r\n" * 100)
    changed = read_result("Updated evidence\r\n" * 100)
    messages = history(first, first, changed, changed)
    outgoing, count, saved = compact_reads(messages)
    assert count == 2 and saved > 0
    assert outgoing[4].parts[0].content == first
    assert outgoing[8].parts[0].content == changed
    assert "content_omitted" in outgoing[2].parts[0].content
    assert "content_omitted" in outgoing[6].parts[0].content


def test_same_response_multiple_reads_keep_all_calls_and_one_full_result():
    content = read_result()
    messages = [
        ModelRequest([UserPromptPart("Read notes")]),
        ModelResponse([ToolCallPart("workspace_read", {}, f"r-{i}") for i in range(3)]),
        ModelRequest([ToolReturnPart("workspace_read", content, f"r-{i}") for i in range(3)]),
    ]
    outgoing, count, saved = compact_reads(messages)
    assert count == 2 and saved > 0 and len(outgoing[-1].parts) == 3
    assert outgoing[-1].parts[-1] is messages[-1].parts[-1]
    assert [p.tool_call_id for p in outgoing[-1].parts] == ["r-0", "r-1", "r-2"]


@pytest.mark.parametrize(
    ("tool_name", "arguments", "content"),
    [
        (
            "workspace_search",
            {"query": "deadline", "relative_path": "."},
            json.dumps({"matches": [{"path": "plan.md", "line": 3, "text": "x" * 600}]}),
        ),
        (
            "workspace_read_many",
            {"reads": [{"relative_path": "a.md"}]},
            json.dumps({"complete": True, "results": [{"path": "a.md", "content": "x" * 600}]}),
        ),
        (
            "workspace_list",
            {"relative_path": "."},
            json.dumps({"entries": [f"file-{index}.md" for index in range(100)]}),
        ),
    ],
)
def test_compacts_exact_duplicate_read_only_tool_calls(tool_name, arguments, content):
    messages = [ModelRequest([UserPromptPart("Inspect sources")])]
    for index in range(2):
        call_id = f"read-{index}"
        call_arguments = arguments if index == 0 else dict(reversed(list(arguments.items())))
        messages.extend(
            [
                ModelResponse([ToolCallPart(tool_name, call_arguments, call_id)]),
                ModelRequest([ToolReturnPart(tool_name, content, call_id)]),
            ]
        )

    outgoing, count, saved = compact_reads(messages)

    assert count == 1 and saved > 0
    marker = json.loads(outgoing[-3].parts[0].content)
    assert "content_omitted" in marker
    assert outgoing[-1].parts[0].content == content
    assert messages[-1].parts[0].content == content


def test_does_not_compact_read_only_calls_with_different_arguments():
    messages = [ModelRequest([UserPromptPart("Inspect sources")])]
    for index, query in enumerate(("deadline", "owner")):
        call_id = f"search-{index}"
        messages.extend(
            [
                ModelResponse([ToolCallPart("workspace_search", {"query": query}, call_id)]),
                ModelRequest(
                    [
                        ToolReturnPart(
                            "workspace_search",
                            json.dumps({"matches": [{"text": "x" * 600}]}),
                            call_id,
                        )
                    ]
                ),
            ]
        )

    assert compact_reads(messages) == (messages, 0, 0)


def test_does_not_compact_read_only_calls_with_different_results():
    messages = [ModelRequest([UserPromptPart("Inspect sources")])]
    for index, text in enumerate(("first", "second")):
        call_id = f"search-{index}"
        messages.extend(
            [
                ModelResponse([ToolCallPart("workspace_search", {"query": "deadline"}, call_id)]),
                ModelRequest(
                    [
                        ToolReturnPart(
                            "workspace_search",
                            json.dumps({"matches": [{"text": text + "x" * 600}]}),
                            call_id,
                        )
                    ]
                ),
            ]
        )

    assert compact_reads(messages) == (messages, 0, 0)


def test_personal_configuration_is_explicit_strict_and_tracks_provenance(config_file):
    assert not load_config(config_file).config.runtime.compact_read_history
    configured = load_config(config_file, overrides={"runtime": {"compact_read_history": True}})
    assert configured.config.runtime.compact_read_history
    assert configured.sources["runtime.compact_read_history"] == "override"
    for invalid in (1, "true", None):
        with pytest.raises(ConfigError):
            load_config(config_file, overrides={"runtime": {"compact_read_history": invalid}})


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("approve", [False, True])
def test_sdk_compacts_outgoing_history_only_across_existing_approval(config_file, enabled, approve):
    original = "负责人：林\r\n" + "Evidence\r\n" * 150 + "Deadline: Friday\r\n"
    target = config_file.parent / "notes.md"
    target.write_bytes(original.encode())
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
        reads = [p for p in returned if p.tool_name == "workspace_read"]
        if len(reads) < 3:
            return ModelResponse(
                [
                    ToolCallPart(
                        "workspace_read", {"relative_path": "notes.md"}, f"read-{len(reads)}"
                    )
                ]
            )
        full = json.loads(reads[-1].content)
        assert full["content"] == original
        if len(returned) == 3:
            return ModelResponse(
                [
                    ToolCallPart(
                        "workspace_replace_text",
                        {
                            "relative_path": "notes.md",
                            "old_text": "Friday",
                            "new_text": "Monday",
                            "expected_sha256": full["sha256"],
                        },
                        "edit",
                    )
                ]
            )
        return ModelResponse([TextPart("Decision processed")])

    async def exercise():
        async with Agent.from_config(
            config_file,
            overrides={
                "tools": {"allowed": ["workspace.read", "workspace.replace_text"]},
                "runtime": {"compact_read_history": enabled},
            },
            model=FunctionModel(respond),
        ) as agent:
            pending = await agent.run("Change Friday to Monday in notes.md")
            assert pending.status == "pending_approval" and pending.requests == 4
            assert pending.approvals[0].preview["content"] == original.replace("Friday", "Monday")
            assert target.read_bytes() == original.encode()
            held = agent._pending[pending.run_id].history
            assert all(
                "content_omitted" not in p.content
                for m in held
                if isinstance(m, ModelRequest)
                for p in m.parts
                if isinstance(p, ToolReturnPart)
            )
            result = await agent.resume(pending.run_id, {pending.approvals[0].call_id: approve})
            assert result.status == "completed" and result.requests == 5
            assert result.compacted_read_results == (5 if enabled else 0)
            assert (result.compacted_read_bytes > 0) is enabled
            assert result.as_dict()["usage"]["compacted_read_bytes"] == result.compacted_read_bytes
            # Trace records the three original full reads, not their outgoing markers.
            sizes = [
                response.calls[0].result_bytes for response in result.execution_trace.responses[:3]
            ]
            assert len(set(sizes)) == 1
            with pytest.raises(RunError):
                await agent.resume(pending.run_id, {pending.approvals[0].call_id: True})
            return result

    result = asyncio.run(exercise())
    assert (
        target.read_bytes()
        == (original.replace("Friday", "Monday") if approve else original).encode()
    )
    assert result.execution_trace.unmatched_tool_returns == 0
    if enabled:
        assert '"content_omitted"' in seen[-1][2].parts[0].content
    else:
        assert all(
            "content_omitted" not in p.content
            for m in seen[-1]
            if isinstance(m, ModelRequest)
            for p in m.parts
            if isinstance(p, ToolReturnPart)
        )
