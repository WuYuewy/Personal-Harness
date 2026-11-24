from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from personal_harness import load_config
from personal_harness.reports import (
    ReportRecipe,
    apply_report,
    inspect_report_sources,
    prepare_report,
)
from personal_harness.workspace import WorkspaceError


def resolved(tmp_path: Path):
    config = tmp_path / "personal.toml"
    config.write_text(
        """schema_version = 1
default_profile = "default"

[base.model]
provider = "fake"

[base.workspace]
root = "."
write_directory = "out"
max_batch_read_bytes = 10000
max_read_bytes = 10000

[base.tools]
allowed = ["workspace.read", "workspace.write"]

[profiles.default]
""",
        "utf-8",
    )
    (tmp_path / "out").mkdir()
    (tmp_path / "notes.md").write_text("TODO: 核对预算\n负责人：小周\nTODO: 更新材料\n", "utf-8")
    return load_config(config)


def recipe() -> ReportRecipe:
    return ReportRecipe.model_validate(
        {
            "name": "项目周报",
            "sources": ["notes.md"],
            "target": "out/weekly-brief.md",
            "sections": [
                {"heading": "本周事项", "prefix": "TODO:"},
                {"heading": "负责人", "prefix": "负责人："},
            ],
        }
    )


def test_report_recipe_extracts_current_lines_with_exact_sources_and_no_model_cost(tmp_path):
    config = resolved(tmp_path)
    draft = prepare_report(config, recipe())

    assert draft.content == (
        "# 项目周报\n\n## 本周事项\n\n"
        "- 核对预算（notes.md:1）\n- 更新材料（notes.md:3）\n\n"
        "## 负责人\n\n- 小周（notes.md:2）\n"
    )
    fingerprint = hashlib.sha256((tmp_path / "notes.md").read_bytes()).hexdigest()
    assert draft.sources == (("notes.md", fingerprint),)
    assert draft.public()["model_requests"] == 0
    assert draft.public()["cost_cny"] == "0"


def test_approval_rebuilds_from_current_sources_and_rejects_stale_content(tmp_path):
    config = resolved(tmp_path)
    before = prepare_report(config, recipe())
    (tmp_path / "notes.md").write_text("TODO: 新增的本周工作\n负责人：小周\n", "utf-8")
    with pytest.raises(WorkspaceError, match="来源或输出已变化"):
        apply_report(config, recipe(), before)
    assert not (tmp_path / "out/weekly-brief.md").exists()


def test_approved_report_uses_write_fingerprint_and_refuses_overwrite_race(tmp_path):
    config = resolved(tmp_path)
    prepared = prepare_report(config, recipe())
    (tmp_path / "out/weekly-brief.md").write_text("user content\n", "utf-8")
    with pytest.raises(WorkspaceError, match="变化"):
        apply_report(config, recipe(), prepared)
    assert (tmp_path / "out/weekly-brief.md").read_text("utf-8") == "user content\n"


def test_report_only_reads_configured_sources_and_rejects_unsafe_or_ambiguous_recipes(tmp_path):
    config = resolved(tmp_path)
    (tmp_path / "outside.md").write_text("TODO: secret", "utf-8")
    with pytest.raises(WorkspaceError, match="relative path"):
        prepare_report(config, recipe().model_copy(update={"sources": ("../outside.md",)}))

    for change, error in (
        ({"target": "notes.md"}, "不能与输出文件相同"),
        ({"sections": ({"heading": "空", "prefix": "MISSING:"},)}, "没有匹配到内容"),
    ):
        result = recipe().model_dump()
        result.update(change)
        if change.get("sections"):
            result["sections"] = list(change["sections"])
        with pytest.raises((ValueError, WorkspaceError), match=error):
            prepare_report(config, ReportRecipe.model_validate(result))

    with pytest.raises(ValueError, match="overlap"):
        ReportRecipe.model_validate(
            {
                "name": "bad",
                "sources": ["notes.md"],
                "target": "out/bad.md",
                "sections": [
                    {"heading": "one", "prefix": "TODO"},
                    {"heading": "two", "prefix": "TODO:"},
                ],
            }
        )


def test_report_data_is_literal_and_html_and_markdown_injection_are_escaped(tmp_path):
    config = resolved(tmp_path)
    (tmp_path / "notes.md").write_text(
        "TODO: <script> **heading** [link](x)\n负责人：<admin>\n", "utf-8"
    )
    draft = prepare_report(config, recipe())
    assert "&lt;script&gt;" in draft.content
    assert "\\*\\*heading\\*\\*" in draft.content
    assert "\\[link\\](x)" in draft.content
    assert "&lt;admin&gt;" in draft.content


def test_inspected_markers_build_a_current_cited_report_without_writes(tmp_path):
    config = resolved(tmp_path)
    (tmp_path / "notes.md").write_text(
        "\ufeff  TODO: 核对预算\n- [ ]更新材料\n风险：期限临近\n"
        "https://example.test\n普通文字\n- [x] 已送审\n",
        "utf-8",
    )
    (tmp_path / "next.md").write_text("TODO: 汇报进展\n", "utf-8")
    found = inspect_report_sources(config, ["notes.md", "next.md"])
    candidates = found["candidates"]
    assert [c["prefix"] for c in candidates] == ["TODO:", "- [ ]", "风险：", "- [x]"]
    assert candidates[0] == {
        "heading": "TODO",
        "prefix": "TODO:",
        "count": 2,
        "example": "核对预算",
        "example_truncated": False,
        "path": "notes.md",
        "line": 1,
    }
    assert found["model_requests"] == 0 and found["cost_cny"] == "0"
    assert found["source_count"] == 2 and not found["additional_markers_omitted"]
    generated = ReportRecipe.model_validate(
        {
            "name": "我的事项",
            "sources": ["notes.md", "next.md"],
            "target": "out/current.md",
            "sections": [{"heading": c["heading"], "prefix": c["prefix"]} for c in candidates],
        }
    )
    # Discovery is advisory: preparation reads current source facts again.
    (tmp_path / "next.md").write_text("TODO: 更新后的进展\n", "utf-8")
    draft = prepare_report(config, generated)
    assert "更新后的进展（next.md:1）" in draft.content
    assert "更新材料（notes.md:2）" in draft.content
    assert "已送审（notes.md:6）" in draft.content
    assert not list((tmp_path / "out").iterdir())


def test_marker_discovery_needs_only_read_permission_and_never_relaxes_boundaries(tmp_path):
    config = resolved(tmp_path)
    config = load_config(config.source_file, overrides={"tools": {"allowed": ["workspace.read"]}})
    assert inspect_report_sources(config, ["notes.md"])["candidates"]
    with pytest.raises(WorkspaceError, match="授权"):
        prepare_report(config, recipe())
    for sources in (["../notes.md"], [".env"], ["notes.md", "notes.md"], ["missing.md"]):
        with pytest.raises(WorkspaceError):
            inspect_report_sources(config, sources)
    for sources in ([], ["notes.md"] * 9, ["x" * 241]):
        with pytest.raises(ValueError):
            inspect_report_sources(config, sources)
    config = load_config(config.source_file, overrides={"workspace": {"max_read_bytes": 5}})
    with pytest.raises(WorkspaceError, match="无法完整读取"):
        inspect_report_sources(config, ["notes.md"])
    config = load_config(config.source_file, overrides={"tools": {"allowed": []}})
    with pytest.raises(WorkspaceError, match="授权"):
        inspect_report_sources(config, ["notes.md"])


def test_marker_discovery_bounds_candidates_and_examples_and_handles_plain_sources(tmp_path):
    config = resolved(tmp_path)
    (tmp_path / "notes.md").write_text(
        "\n".join(f"Label{i}: {'v' * 170}" for i in range(30)), "utf-8"
    )
    result = inspect_report_sources(config, ["notes.md"])
    assert len(result["candidates"]) == result["candidate_limit"] == 24
    assert result["additional_markers_omitted"]
    assert all(c["example_truncated"] and len(c["example"]) == 160 for c in result["candidates"])
    (tmp_path / "notes.md").write_text("没有行标记的普通资料\n", "utf-8")
    result = inspect_report_sources(config, ["notes.md"])
    assert not result["candidates"] and not result["additional_markers_omitted"]
