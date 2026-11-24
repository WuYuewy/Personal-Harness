import asyncio
import hashlib
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel

from personal_harness import Agent, RunError, load_config
from personal_harness.context import CURRENT_METRICS, RunDependencies, RunMetrics, task_instructions
from personal_harness.workspace import WorkspaceError, WorkspaceFiles


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def edit_model(arguments: dict) -> FunctionModel:
    def respond(messages, info):
        if any(
            isinstance(part, ToolReturnPart)
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
        ):
            return ModelResponse([TextPart("Edit decision processed")])
        return ModelResponse([ToolCallPart("workspace_replace_text", arguments)])

    return FunctionModel(respond)


def editor(config_file: Path, data: bytes, **arguments) -> Agent:
    (config_file.parent / "report.md").write_bytes(data)
    return Agent.from_config(
        config_file,
        overrides={
            "tools": {"allowed": ["workspace.replace_text"]},
            "workspace": {"read_cache_bytes": 1000},
        },
        model=edit_model(
            {
                "relative_path": "report.md",
                "old_text": "Friday",
                "new_text": "Monday",
                "expected_sha256": sha(data),
                **arguments,
            }
        ),
    )


@pytest.mark.parametrize("approve", [True, False])
def test_edit_requires_exact_approval_and_preserves_other_bytes(config_file, approve):
    original = "负责人：林\r\nDeadline: Friday\r\nDo not reformat.\r\n".encode()
    agent = editor(config_file, original)
    target = config_file.parent / "report.md"

    async def exercise():
        async with agent:
            pending = await agent.run("Change Friday to Monday")
            assert pending.status == "pending_approval" and pending.requests == 1
            approval = pending.approvals[0]
            assert approval.tool == "workspace.replace_text"
            assert approval.preview["content"].encode() == original.replace(b"Friday", b"Monday")
            assert "Friday" in approval.preview["diff"] and "Monday" in approval.preview["diff"]
            assert target.read_bytes() == original
            # A caller cannot change the held request through its public preview.
            approval.preview["content"] = "Tampered"
            final = await agent.resume(pending.run_id, {approval.call_id: approve})
            assert final.status == "completed" and final.requests == 2
            assert final.available_tools == ("workspace.replace_text",)
            assert final.tool_calls == (("workspace.replace_text",) if approve else ())
            call = final.execution_trace.responses[0].calls[0]
            assert call.tool == "workspace.replace_text" and call.outcome == (
                "success" if approve else "denied"
            )
            with pytest.raises(RunError, match="No pending"):
                await agent.resume(pending.run_id, {approval.call_id: approve})

    asyncio.run(exercise())
    assert target.read_bytes() == (original.replace(b"Friday", b"Monday") if approve else original)


@pytest.mark.parametrize(
    ("data", "old", "new", "fingerprint"),
    [
        (b"Friday Friday", "Friday", "Monday", None),
        (b"aaa", "aa", "b", None),
        (b"Friday", "missing", "Monday", None),
        (b"Friday", "", "Monday", None),
        (b"Friday", "Friday", "Friday", None),
        (b"Friday", "Friday", "Monday", "0" * 64),
        (b"Friday", "Friday", "Monday", ""),
        (b"\xffFriday", "Friday", "Monday", None),
        (b"Friday\x00", "Friday", "Monday", None),
        (b"Friday", "Friday", "\x00", None),
    ],
)
def test_invalid_or_ambiguous_edits_never_propose_or_write(
    config_file, data, old, new, fingerprint
):
    hash_value = fingerprint if fingerprint is not None else sha(data)
    agent = editor(config_file, data, old_text=old, new_text=new, expected_sha256=hash_value)

    async def exercise():
        async with agent:
            with pytest.raises(RunError):
                await agent.run("Invalid edit")
            assert not agent._pending

    asyncio.run(exercise())
    assert (config_file.parent / "report.md").read_bytes() == data


def test_missing_hash_is_returned_as_framework_validation_feedback_without_writing(config_file):
    original = b"Deadline: Friday"
    agent = editor(config_file, original, expected_sha256=None)

    async def exercise():
        async with agent:
            result = await agent.run("Invalid missing hash")
            assert not result.approvals and not agent._pending and not result.tool_calls
            assert result.execution_trace.responses[0].calls[0].outcome == "retried"

    asyncio.run(exercise())
    assert (config_file.parent / "report.md").read_bytes() == original


def test_edit_conflict_ignores_warm_cache_and_does_not_send_resume_request(config_file):
    original = b"Deadline: Friday"
    agent = editor(config_file, original)
    target = config_file.parent / "report.md"

    async def exercise():
        async with agent:
            agent._files.read("report.md")
            assert agent._files._read_cache
            pending = await agent.run("Edit")
            target.write_bytes(b"Deadline: Sunday")
            with pytest.raises(RunError) as failure:
                await agent.resume(pending.run_id, {pending.approvals[0].call_id: True})
            assert failure.value.reason == "write_conflict"
            assert failure.value.run_id == pending.run_id and not agent._pending

    asyncio.run(exercise())
    assert target.read_bytes() == b"Deadline: Sunday"


@pytest.mark.parametrize("operation", ["read-limit", "write-limit", "excluded", "missing"])
def test_edit_honors_existing_workspace_limits(config_file, operation):
    target = config_file.parent / "report.md"
    original = b"Deadline: Friday"
    target.write_bytes(original)
    patch = {
        "read-limit": {"max_read_bytes": 2},
        "write-limit": {"max_write_bytes": 2},
        "excluded": {"exclude": ["report.md"]},
        "missing": {},
    }[operation]
    files = WorkspaceFiles(
        load_config(config_file, overrides={"workspace": patch}).config.workspace
    )
    if operation == "missing":
        target.unlink()
    with pytest.raises(WorkspaceError):
        files.preview_replace("report.md", "Friday", "Monday", sha(original))
    assert not target.exists() if operation == "missing" else target.read_bytes() == original


def test_edit_deletes_unique_text_and_invalidates_only_target_cache(config_file):
    target, source = config_file.parent / "report.md", config_file.parent / "source.md"
    target.write_bytes(b"Keep\r\nRemove\r\n")
    source.write_bytes(b"Source")
    files = WorkspaceFiles(
        load_config(
            config_file, overrides={"workspace": {"read_cache_bytes": 1000}}
        ).config.workspace
    )
    metrics = RunMetrics()
    token = CURRENT_METRICS.set(metrics)
    try:
        fingerprint = files.read("report.md")["sha256"]
        files.read("source.md")
        files.replace_text("report.md", "Remove\r\n", "", fingerprint)
        before = metrics.read_bytes
        assert files.read("report.md")["content"] == "Keep\r\n"
        assert metrics.read_bytes == before + 6 and metrics.read_cache_hits == 0
        assert files.read("source.md")["content"] == "Source" and metrics.read_cache_hits == 1
    finally:
        CURRENT_METRICS.reset(token)
    assert target.read_bytes() == b"Keep\r\n"


@pytest.mark.parametrize("path", ["../outside.md", "/absolute.md", ".env", "C:/outside.md"])
def test_replacement_does_not_expand_workspace_paths(config_file, path):
    files = WorkspaceFiles(load_config(config_file).config.workspace)
    with pytest.raises(WorkspaceError):
        files.preview_replace(path, "Friday", "Monday", "0" * 64)


def test_task_can_hide_replacement_and_old_write_permission_does_not_grant_it(config_file):
    async def exercise():
        for allowed, selected in [
            (["workspace.write"], None),
            (["workspace.write", "workspace.replace_text"], ["workspace.write"]),
        ]:
            seen = []

            def respond(messages, info, seen=seen):
                seen.append({t.name for t in info.function_tools})
                return ModelResponse([TextPart("No edit")])

            async with Agent.from_config(
                config_file, overrides={"tools": {"allowed": allowed}}, model=FunctionModel(respond)
            ) as agent:
                result = await agent.run("Inspect", tools=selected)
                assert result.available_tools == ("workspace.write",)
            assert seen == [{"workspace_write"}]
        async with Agent.from_config(config_file) as agent:
            with pytest.raises(RunError) as failure:
                await agent.run("Edit", tools=["workspace.replace_text"])
            assert failure.value.run_id is None

    asyncio.run(exercise())


def test_task_context_explains_explicit_edit_without_enabling_other_writes(config_file):
    resolved = load_config(
        config_file,
        overrides={
            "runtime": {"context_mode": "task"},
            "tools": {"allowed": ["workspace.replace_text"]},
        },
    )
    prompt = task_instructions(RunDependencies(resolved))
    assert "human approval" in prompt and "unique literal excerpt" in prompt
    assert "workspace_replace_text" in prompt
    assert resolved.config.tools.allowed == ("workspace.replace_text",)


@pytest.mark.parametrize("decision", ["y", "n"])
def test_cli_uses_existing_confirmation_for_literal_edits(config_file, monkeypatch, decision):
    import json

    from typer.testing import CliRunner

    from personal_harness import cli

    original = b"Deadline: Friday\r\nKeep unchanged\r\n"
    target = config_file.parent / "report.md"
    target.write_bytes(original)
    config_file.write_text(
        config_file.read_text("utf-8")
        + '\n[profiles.edit.tools]\nallowed = ["workspace.replace_text"]\n',
        "utf-8",
    )
    model = edit_model(
        {
            "relative_path": "report.md",
            "old_text": "Friday",
            "new_text": "Monday",
            "expected_sha256": sha(original),
        }
    )
    monkeypatch.setattr(cli, "Agent", lambda resolved: Agent(resolved, model=model))
    result = CliRunner().invoke(
        cli.app,
        [
            "run",
            "Edit Friday",
            "-c",
            str(config_file),
            "-p",
            "edit",
            "--tool",
            "workspace.replace_text",
            "--json",
        ],
        input=decision + "\n",
    )
    assert result.exit_code == 0, result.output
    record = json.loads(result.stdout)
    assert record["status"] == "completed" and record["usage"]["requests"] == 2
    assert "Apply this exact write?" in result.stderr
    assert target.read_bytes() == (
        original.replace(b"Friday", b"Monday") if decision == "y" else original
    )
