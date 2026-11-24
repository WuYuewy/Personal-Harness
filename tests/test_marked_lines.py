import hashlib

import pytest

from personal_harness.config import WorkspaceSettings
from personal_harness.marked_lines import (
    MAX_ACTION_RESULT_BYTES,
    MAX_ACTION_RESULTS,
    extract_marked_lines,
)
from personal_harness.workspace import WorkspaceError, WorkspaceFiles


def test_extracts_only_complete_matching_lines_with_source_and_fingerprint(tmp_path):
    content = (
        "\ufeffTODO: 行动000，负责人王青，截止12月1日。\r\n"
        "归档001：已完成。\r"
        " TODO: 缩进标记不匹配\n"
        "TODO: 行动649，负责人王青，截止12月10日。"
    )
    source = tmp_path / "activity log.md"
    source.write_bytes(content.encode("utf-8"))

    result = extract_marked_lines(
        WorkspaceFiles(WorkspaceSettings(root=str(tmp_path))), "activity log.md", "TODO:"
    )

    assert [(item["path"], item["line"], item["text"]) for item in result["actions"]] == [
        ("activity log.md", 1, "TODO: 行动000，负责人王青，截止12月1日。"),
        ("activity log.md", 4, "TODO: 行动649，负责人王青，截止12月10日。"),
    ]
    assert result["match_count"] == result["displayed_count"] == 2
    assert not result["truncated"] and result["omitted_count"] == 0
    assert result["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()


def test_result_limits_count_every_match_without_returning_partial_lines(tmp_path):
    source = tmp_path / "actions.txt"
    lines = ["TODO: " + "長" * MAX_ACTION_RESULT_BYTES]
    lines.extend("TODO: " + str(index) for index in range(MAX_ACTION_RESULTS + 1))
    source.write_text("\n".join(lines), encoding="utf-8")

    result = extract_marked_lines(
        WorkspaceFiles(WorkspaceSettings(root=str(tmp_path))), "actions.txt", "TODO:"
    )

    assert result["match_count"] == MAX_ACTION_RESULTS + 2
    assert result["displayed_count"] == MAX_ACTION_RESULTS
    assert result["omitted_count"] == 2 and result["truncated"]
    assert all(
        len(item["text"].encode("utf-8")) < MAX_ACTION_RESULT_BYTES for item in result["actions"]
    )


@pytest.mark.parametrize("prefix", ["", "x" * 81, "todo\n"])
def test_rejects_invalid_markers(tmp_path, prefix):
    source = tmp_path / "actions.txt"
    source.write_text("TODO: item\n", encoding="utf-8")
    with pytest.raises(ValueError, match="标记"):
        extract_marked_lines(
            WorkspaceFiles(WorkspaceSettings(root=str(tmp_path))), "actions.txt", prefix
        )


def test_obeys_workspace_exclusions_and_read_limit(tmp_path):
    excluded = tmp_path / ".env"
    excluded.write_text("TODO: secret\n", encoding="utf-8")
    files = WorkspaceFiles(WorkspaceSettings(root=str(tmp_path), max_read_bytes=8))
    with pytest.raises(WorkspaceError, match="excluded"):
        extract_marked_lines(files, ".env", "TODO:")

    source = tmp_path / "long.txt"
    source.write_text("TODO: this file is over limit\n", encoding="utf-8")
    with pytest.raises(WorkspaceError, match="limit"):
        extract_marked_lines(files, "long.txt", "TODO:")
