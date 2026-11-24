import asyncio
import hashlib
import json
import os

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel
from typer.testing import CliRunner

from personal_harness import Agent, ConfigError, RunError, load_config
from personal_harness.cli import app
from personal_harness.config import WorkspaceSettings
from personal_harness.context import RunDependencies, model_context, task_instructions
from personal_harness.workspace import WorkspaceError, WorkspaceFiles


@pytest.mark.parametrize(
    "directory",
    [
        "",
        "/out",
        "C:/out",
        "C:out",
        "../out",
        "out/../src",
        "out/NUL.txt",
        "out.",
        "out/*",
        " out",
        True,
    ],
)
def test_invalid_write_directory_fails_config_validation(config_file, directory):
    with pytest.raises(ConfigError, match="write_directory"):
        load_config(config_file, overrides={"workspace": {"write_directory": directory}})


def test_relative_directory_normalization_and_layer_sources(config_file):
    assert load_config(config_file).config.workspace.write_directory is None
    config_file.write_text(
        config_file.read_text("utf-8")
        + '\n[profiles.desktop.workspace]\nwrite_directory = "out"\n',
        "utf-8",
    )
    resolved = load_config(config_file)
    assert resolved.config.workspace.write_directory == "out"
    assert resolved.sources["workspace.write_directory"] == "profile:desktop"
    nested = load_config(config_file, overrides={"workspace": {"write_directory": "out\\发布//."}})
    assert nested.config.workspace.write_directory == "out/发布"
    assert nested.sources["workspace.write_directory"] == "override"
    assert load_config(config_file, profile="mobile").config.workspace.write_directory is None


@pytest.fixture
def bounded(tmp_path):
    (tmp_path / "out" / "nested").mkdir(parents=True)
    (tmp_path / "out-other").mkdir()
    (tmp_path / "source.md").write_text("deadline old", "utf-8")
    (tmp_path / "out" / "draft.md").write_text("deadline old", "utf-8")
    return WorkspaceFiles(WorkspaceSettings(root=str(tmp_path), write_directory="out"))


@pytest.mark.parametrize("path", ["source.md", "out-other/new.md", "out.md", "out"])
def test_directory_boundary_is_component_based_and_does_not_change_sources(bounded, path):
    with pytest.raises(WorkspaceError, match="inside the configured write directory"):
        bounded.preview(path, "changed")
    with pytest.raises(WorkspaceError, match="inside the configured write directory"):
        bounded.write(path, "changed", None)
    assert bounded.read("source.md")["content"] == "deadline old"
    assert bounded.search("deadline")["matches"]
    assert not list(bounded.root.rglob(".harness-*"))


def test_exact_replacement_cannot_modify_readable_source(bounded):
    fingerprint = bounded.read("source.md")["sha256"]
    with pytest.raises(WorkspaceError, match="write directory"):
        bounded.preview_replace("source.md", "old", "new", fingerprint)
    with pytest.raises(WorkspaceError, match="write directory"):
        bounded.replace_text("source.md", "old", "new", fingerprint)
    assert bounded.read("source.md")["content"] == "deadline old"


def test_allowed_new_write_and_exact_edit_keep_conflict_checks(bounded):
    receipt = bounded.write("out/nested/brief.md", "deadline old", None)
    assert receipt["written"]
    preview = bounded.preview_replace("out/nested/brief.md", "old", "new", receipt["sha256"])
    assert preview["content"] == "deadline new"
    bounded.replace_text("out/nested/brief.md", "old", "new", receipt["sha256"])
    with pytest.raises(WorkspaceError, match="changed"):
        bounded.write("out/nested/brief.md", "stale", receipt["sha256"])
    assert bounded.read("out/nested/brief.md")["content"] == "deadline new"


def test_directory_limit_does_not_create_missing_parent_or_override_exclusions(tmp_path):
    workspace = WorkspaceFiles(WorkspaceSettings(root=str(tmp_path), write_directory="missing"))
    with pytest.raises(WorkspaceError, match="parent directory"):
        workspace.write("missing/brief.md", "draft", None)
    assert not (tmp_path / "missing").exists()
    excluded = WorkspaceFiles(WorkspaceSettings(root=str(tmp_path), write_directory=".git"))
    with pytest.raises(WorkspaceError, match="excluded"):
        excluded.preview(".git/brief.md", "draft")


def test_unset_or_root_directory_preserves_whole_workspace_writes(tmp_path):
    for index, directory in enumerate((None, ".")):
        workspace = WorkspaceFiles(WorkspaceSettings(root=str(tmp_path), write_directory=directory))
        assert workspace.write(f"brief-{index}.md", "draft", None)["written"]


@pytest.mark.parametrize("inside", [False, True])
def test_sdk_enforces_directory_before_approval_and_requires_approval_inside(config_file, inside):
    (config_file.parent / "out").mkdir()
    (config_file.parent / "source.md").write_text("source", "utf-8")
    target = "out/brief.md" if inside else "brief.md"

    def respond(messages, info):
        if any(
            isinstance(part, ToolReturnPart)
            for m in messages
            if isinstance(m, ModelRequest)
            for part in m.parts
        ):
            return ModelResponse([TextPart("Done")])
        return ModelResponse(
            [
                ToolCallPart(
                    "workspace_write",
                    {"relative_path": target, "content": "draft", "expected_sha256": None},
                )
            ]
        )

    async def exercise():
        async with Agent.from_config(
            config_file,
            overrides={
                "workspace": {"write_directory": "out"},
                "tools": {"allowed": ["workspace.write"]},
            },
            model=FunctionModel(respond),
        ) as agent:
            if not inside:
                with pytest.raises(RunError, match="write directory"):
                    await agent.run("Prepare brief")
                assert agent._pending == {}
                return
            pending = await agent.run("Prepare brief")
            assert (
                pending.status == "pending_approval" and not (config_file.parent / target).exists()
            )
            assert pending.approvals[0].preview["path"] == target
            final = await agent.resume(pending.run_id, {pending.approvals[0].call_id: True})
            assert final.status == "completed"

    asyncio.run(exercise())
    assert (config_file.parent / target).exists() is inside
    assert (config_file.parent / "source.md").read_text("utf-8") == "source"


@pytest.mark.parametrize("mode", ["configured", "task"])
def test_relative_directory_is_model_guidance_without_absolute_paths(config_file, mode):
    resolved = load_config(
        config_file,
        overrides={
            "runtime": {"context_mode": mode},
            "workspace": {"write_directory": "out"},
            "tools": {"allowed": ["workspace.read", "workspace.write"]},
        },
    )
    assert model_context(resolved, ("workspace.write",))["write_directory"] == "out"
    instructions = task_instructions(RunDependencies(resolved, selected_tools=("workspace.write",)))
    assert "All file changes must stay inside write_directory" in instructions
    assert str(config_file.parent) not in instructions
    if mode == "task":
        assert "write_directory" not in model_context(resolved, ("workspace.read",))


def test_path_case_uses_platform_directory_semantics(bounded):
    if os.name == "nt":
        assert bounded.write("OUT/case.md", "draft", None)["written"]
    else:
        with pytest.raises(WorkspaceError, match="write directory"):
            bounded.preview("OUT/case.md", "draft")


@pytest.mark.parametrize("has_tools,exists", [(True, False), (True, True), (False, False)])
def test_doctor_checks_required_output_directory_without_creating_it(
    config_file, has_tools, exists
):
    config_file.write_text(
        config_file.read_text("utf-8")
        .replace('root = "."', 'root = "."\nwrite_directory = "out"')
        .replace(
            'allowed = ["context.describe"]',
            'allowed = ["workspace.write"]' if has_tools else 'allowed = ["workspace.read"]',
        ),
        "utf-8",
    )
    if exists:
        (config_file.parent / "out").mkdir()
    result = CliRunner().invoke(app, ["doctor", "-c", str(config_file)])
    assert result.exit_code == (1 if has_tools and not exists else 0)
    payload = json.loads(result.stdout)
    assert payload["workspace"]["write_directory"] == {
        "path": "out",
        "available": exists,
        "required": has_tools,
    }
    assert (config_file.parent / "out").exists() is exists


@pytest.mark.parametrize("inside", [False, True])
def test_sdk_exact_edit_obeys_same_directory_and_keeps_crlf(config_file, inside):
    (config_file.parent / "out").mkdir()
    target = "out/draft.md" if inside else "source.md"
    original = b"deadline old\r\nkeep this line\r\n"
    (config_file.parent / target).write_bytes(original)

    def respond(messages, info):
        if any(
            isinstance(part, ToolReturnPart)
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
        ):
            return ModelResponse([TextPart("Done")])
        return ModelResponse(
            [
                ToolCallPart(
                    "workspace_replace_text",
                    {
                        "relative_path": target,
                        "old_text": "old",
                        "new_text": "new",
                        "expected_sha256": hashlib.sha256(original).hexdigest(),
                    },
                )
            ]
        )

    async def exercise():
        async with Agent.from_config(
            config_file,
            overrides={
                "workspace": {"write_directory": "out"},
                "tools": {"allowed": ["workspace.replace_text"]},
            },
            model=FunctionModel(respond),
        ) as agent:
            if inside:
                pending = await agent.run("Update the deadline")
                assert pending.status == "pending_approval"
                assert (config_file.parent / target).read_bytes() == original
                await agent.resume(pending.run_id, {pending.approvals[0].call_id: True})
            else:
                with pytest.raises(RunError, match="write directory"):
                    await agent.run("Update the source")
                assert agent._pending == {}

    asyncio.run(exercise())
    assert (config_file.parent / target).read_bytes() == (
        original.replace(b"old", b"new") if inside else original
    )
