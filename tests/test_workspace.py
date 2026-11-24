import hashlib
import os
import subprocess
from pathlib import Path

import pytest

from personal_harness import load_config
from personal_harness.config import WorkspaceSettings
from personal_harness.workspace import WorkspaceError, WorkspaceFiles


@pytest.fixture
def workspace(tmp_path: Path) -> WorkspaceFiles:
    return WorkspaceFiles(WorkspaceSettings(root=str(tmp_path)))


def test_read_list_search_and_preview_share_the_workspace(workspace: WorkspaceFiles) -> None:
    (workspace.root / "notes").mkdir()
    source = workspace.root / "notes" / "meeting.md"
    source.write_text("# Meeting\nTODO: 整理材料\n", "utf-8")
    assert workspace.list()["entries"] == [{"path": "notes", "type": "directory"}]
    read = workspace.read("notes/meeting.md")
    assert read["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    result = workspace.search("todo")
    assert result["matches"][0] == {"path": "notes/meeting.md", "line": 2, "text": "TODO: 整理材料"}
    preview = workspace.preview("notes/meeting.md", "# Updated\n")
    assert preview["expected_sha256"] == read["sha256"]
    assert "-TODO:" in preview["diff"]
    assert preview["written"] is False
    assert "TODO:" in source.read_text("utf-8")


def test_search_shows_actual_hit_inside_long_line_with_correct_unicode_columns(tmp_path):
    line = "ß" * 600 + " invoice INV-0173 amount=847.50 " + "z" * 1000
    (tmp_path / "invoices.txt").write_text(line, "utf-8")
    result = WorkspaceFiles(WorkspaceSettings(root=str(tmp_path))).search(
        "inv-0173", "invoices.txt"
    )
    match = result["matches"][0]
    assert "INV-0173 amount=847.50" in match["text"]
    assert match["line"] == 1 and result["truncated"]
    start = match["text_start_column"] - 1
    assert match["text"] == line[start : start + len(match["text"])]
    assert len(match["text"]) <= 500


def test_single_file_search_keeps_path_scope_line_numbers_and_limits(tmp_path: Path):
    (tmp_path / "target.csv").write_bytes(b"id,amount\r\nA,10\r\nA,20\r\n")
    (tmp_path / "other.csv").write_bytes(b"A,999\n")
    workspace = WorkspaceFiles(WorkspaceSettings(root=str(tmp_path), max_scan_entries=1))
    result = workspace.search("a,", "target.csv")
    assert result["matches"] == [
        {"path": "target.csv", "line": 2, "text": "A,10"},
        {"path": "target.csv", "line": 3, "text": "A,20"},
    ]
    assert result["files_scanned"] == 1 and result["skipped"] == 0
    assert result["bytes_scanned"] == len((tmp_path / "target.csv").read_bytes())
    assert not result["truncated"]
    bounded = WorkspaceFiles(
        WorkspaceSettings(root=str(tmp_path), max_search_results=1, max_search_result_chars=3)
    ).search("a,", "target.csv")
    assert bounded["matches"] == [{"path": "target.csv", "line": 2, "text": "A,1"}]
    assert bounded["truncated"]


@pytest.mark.parametrize("contents", [b"\x00private", b"\xff", b"abcdef"])
def test_single_file_search_reports_skipped_unreadable_or_oversized_source(tmp_path, contents):
    (tmp_path / "source").write_bytes(contents)
    workspace = WorkspaceFiles(WorkspaceSettings(root=str(tmp_path), max_search_bytes=5))
    result = workspace.search("private", "source")
    assert result["matches"] == [] and result["skipped"] == 1
    assert result["bytes_scanned"] <= 5
    (tmp_path / ".env").write_bytes(b"private")
    with pytest.raises(WorkspaceError, match="excluded"):
        workspace.search("private", ".env")
    with pytest.raises(WorkspaceError, match="existing directory or regular file"):
        workspace.search("private", "missing")


@pytest.mark.parametrize(
    "path",
    [
        "../outside.txt",
        "nested/../../outside",
        "/absolute",
        "C:/absolute",
        "C:relative",
        "\\\\server\\share\\secret",
        "..\\outside",
        "file.txt:stream",
        "NUL",
        "con.txt",
        "file. ",
        "file.",
        "",
        "bad\x00path",
    ],
)
def test_paths_cannot_escape_or_use_windows_special_names(
    workspace: WorkspaceFiles, path: str
) -> None:
    with pytest.raises(WorkspaceError):
        workspace.path(path)


def test_exclusions_apply_to_all_operations_and_nested_paths(workspace: WorkspaceFiles) -> None:
    (workspace.root / ".env.local").write_text("secret-token", "utf-8")
    hidden = workspace.root / "notes" / ".git"
    hidden.mkdir(parents=True)
    (hidden / "data").write_text("secret-token", "utf-8")
    (workspace.root / "visible.txt").write_text("visible", "utf-8")
    assert workspace.search("secret-token")["matches"] == []
    assert ".env.local" not in str(workspace.list())
    for path in (".env.local", "notes/.git/data"):
        with pytest.raises(WorkspaceError, match="excluded"):
            workspace.read(path)
        with pytest.raises(WorkspaceError, match="excluded"):
            workspace.preview(path, "replacement")


def test_custom_profile_limits_and_exclusions_survive_path_resolution(config_file: Path) -> None:
    resolved = load_config(
        config_file,
        overrides={
            "workspace": {
                "max_read_bytes": 3,
                "max_entries": 1,
                "exclude": ["private*"],
            }
        },
    )
    workspace = WorkspaceFiles(resolved.config.workspace)
    (workspace.root / "long.txt").write_text("1234", "utf-8")
    (workspace.root / "private.txt").write_text("private", "utf-8")
    with pytest.raises(WorkspaceError, match="limit"):
        workspace.read("long.txt")
    with pytest.raises(WorkspaceError, match="excluded"):
        workspace.read("private.txt")
    assert len(workspace.list()["entries"]) == 1
    assert workspace.list()["truncated"] is True


def test_search_bounds_and_binary_handling(tmp_path: Path) -> None:
    for number in range(4):
        (tmp_path / f"{number}.txt").write_text("TODO\nTODO\n", "utf-8")
    workspace = WorkspaceFiles(WorkspaceSettings(root=str(tmp_path), max_search_results=2))
    assert len(workspace.search("todo")["matches"]) == 2
    assert workspace.search("todo")["truncated"] is True
    bounded = WorkspaceFiles(
        WorkspaceSettings(root=str(tmp_path), max_search_files=1, max_search_bytes=4)
    )
    report = bounded.search("not-there")
    assert report["files_scanned"] <= 1
    assert report["bytes_scanned"] <= 4
    assert report["truncated"] is True

    (tmp_path / "binary").write_bytes(b"\x00secret")
    with pytest.raises(WorkspaceError, match="Binary"):
        workspace.read("binary")
    (tmp_path / "invalid").write_bytes(b"\xff")
    with pytest.raises(WorkspaceError, match="UTF-8"):
        workspace.read("invalid")


def test_search_keeps_later_matches_after_clipping_a_long_line(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text(
        "needle " + "x" * 500 + "\nneedle: actual answer\nneedle: later detail\n", "utf-8"
    )
    workspace = WorkspaceFiles(WorkspaceSettings(root=str(tmp_path), max_search_result_chars=522))
    report = workspace.search("needle")
    assert [match["line"] for match in report["matches"]] == [1, 2, 3]
    assert report["matches"][1]["text"] == "needle: actual answer"
    assert report["matches"][2]["text"] == "n"
    assert sum(len(match["text"]) for match in report["matches"]) == 522
    assert report["truncated"]


def test_excerpt_keeps_full_file_fingerprint_and_respects_workspace_limits(
    workspace: WorkspaceFiles,
) -> None:
    data = "第一行\r\n第二行\n最后一行".encode()
    (workspace.root / "notes.md").write_bytes(data)
    result = workspace.read("notes.md", 2, 10)
    assert result["content"] == "第二行\n最后一行"
    assert result["bytes"] == len(result["content"].encode("utf-8"))
    assert (result["start_line"], result["end_line"]) == (2, 3)
    assert result["sha256"] == hashlib.sha256(data).hexdigest()
    with pytest.raises(WorkspaceError, match="both line numbers"):
        workspace.read("notes.md", start_line=2)
    with pytest.raises(WorkspaceError, match="past the end"):
        workspace.read("notes.md", 4, 5)
    limited = WorkspaceFiles(WorkspaceSettings(root=str(workspace.root), max_read_bytes=8))
    with pytest.raises(WorkspaceError, match="read limit"):
        limited.read("notes.md", 2, 2)
    with pytest.raises(WorkspaceError, match="excluded"):
        workspace.read(".env", 1, 1)


def test_links_are_not_traversed_by_read_search_list_or_preview(
    workspace: WorkspaceFiles, tmp_path: Path
) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("private-external", "utf-8")
    linked = workspace.root / "linked"
    try:
        linked.symlink_to(outside, target_is_directory=True)
    except OSError:
        if os.name != "nt":
            raise
        # Junction creation needs no Windows Developer Mode and requires no deletion.
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(linked), str(outside)],
            check=True,
            capture_output=True,
        )
    assert workspace.search("private-external")["matches"] == []
    assert workspace.list()["entries"] == []
    for operation in (
        lambda: workspace.read("linked/secret.txt"),
        lambda: workspace.preview("linked/new.txt", "bad"),
    ):
        with pytest.raises(WorkspaceError, match="links|reparse"):
            operation()
    assert (outside / "secret.txt").read_text("utf-8") == "private-external"


def test_create_and_overwrite_require_exact_target_state(workspace: WorkspaceFiles) -> None:
    result = workspace.write("report.md", "first\n", None)
    assert result["written"] is True
    with pytest.raises(WorkspaceError, match="fingerprint|SHA-256"):
        workspace.write("report.md", "second\n", None)
    current = workspace.read("report.md")["sha256"]
    workspace.write("report.md", "second\n", current)
    assert workspace.read("report.md")["content"] == "second\n"
    with pytest.raises(WorkspaceError, match="changed"):
        workspace.write("report.md", "stale\n", current)
    assert workspace.read("report.md")["content"] == "second\n"
    assert not list(workspace.root.glob(".harness-*"))


def test_write_limits_and_preview_truncation(tmp_path: Path) -> None:
    workspace = WorkspaceFiles(
        WorkspaceSettings(root=str(tmp_path), max_write_bytes=6, max_preview_chars=3)
    )
    preview = workspace.preview("new.md", "你好")
    assert preview["bytes"] == 6
    assert preview["diff_truncated"] is True
    assert len(preview["diff"]) == 3
    with pytest.raises(WorkspaceError, match="byte limit"):
        workspace.preview("new.md", "你好!")
    assert not (tmp_path / "new.md").exists()
