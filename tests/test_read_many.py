import hashlib
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from personal_harness.config import WorkspaceSettings
from personal_harness.context import CURRENT_METRICS, RunMetrics
from personal_harness.workspace import ReadRequest, WorkspaceError, WorkspaceFiles


@pytest.mark.parametrize("size", [0, 17, True, 1.5])
def test_batch_file_limit_is_strict_and_bounded(size) -> None:
    with pytest.raises(ValidationError):
        WorkspaceSettings(max_batch_files=size)


@pytest.mark.parametrize(
    "arguments",
    [
        {"relative_path": "a", "start_line": 1},
        {"relative_path": "a", "start_line": 2, "end_line": 1},
        {"relative_path": "a", "start_line": True, "end_line": 1},
        {"relative_path": "a", "start_line": 0, "end_line": 1},
        {"relative_path": "a", "start_line": "1", "end_line": 1},
        {"relative_path": "a", "content": "not a write"},
    ],
)
def test_read_request_rejects_ambiguous_or_non_read_arguments(arguments) -> None:
    with pytest.raises(ValidationError):
        ReadRequest.model_validate(arguments)


def test_batch_returns_same_contents_lines_and_full_fingerprints(tmp_path: Path) -> None:
    first = "负责人：林\n截止日期：11月20日\n".encode()
    second = b"Budget: 3000\r\nRisk: missing samples\r\n"
    (tmp_path / "first.md").write_bytes(first)
    (tmp_path / "second.md").write_bytes(second)
    workspace = WorkspaceFiles(WorkspaceSettings(root=str(tmp_path)))
    result = workspace.read_many(
        [
            ReadRequest(relative_path="first.md", start_line=2, end_line=2),
            ReadRequest(relative_path="second.md"),
        ]
    )
    assert result["complete"] and result["bytes_processed"] == len(first) + len(second)
    assert result["results"] == [
        {"index": 0, **workspace.read("first.md", 2, 2)},
        {"index": 1, **workspace.read("second.md")},
    ]
    assert result["results"][0]["sha256"] == hashlib.sha256(first).hexdigest()
    assert (tmp_path / "first.md").read_bytes() == first
    assert (tmp_path / "second.md").read_bytes() == second


def test_batch_reports_missing_denied_binary_and_invalid_sources_without_losing_valid_ones(
    tmp_path: Path,
) -> None:
    (tmp_path / "visible").write_bytes(b"hello")
    (tmp_path / ".env").write_bytes(b"secret")
    (tmp_path / "binary").write_bytes(b"\x00")
    (tmp_path / "invalid").write_bytes(b"\xff")
    workspace = WorkspaceFiles(WorkspaceSettings(root=str(tmp_path)))
    result = workspace.read_many(
        [
            ReadRequest(relative_path=name)
            for name in ("missing", ".env", "../outside", "binary", "invalid", "visible")
        ]
    )
    assert not result["complete"] and result["bytes_processed"] == 7
    assert [item["index"] for item in result["results"]] == list(range(6))
    assert all("error" in item for item in result["results"][:5])
    assert result["results"][-1]["content"] == "hello"
    assert "secret" not in str(result) and str(tmp_path) not in str(result)


@pytest.mark.parametrize("cache_bytes", [0, 1000])
def test_budget_counts_full_files_cache_hits_and_failed_reads(
    tmp_path: Path, cache_bytes: int
) -> None:
    data = b"header\nanswer\n"
    (tmp_path / "a").write_bytes(data)
    (tmp_path / "invalid").write_bytes(b"\xff")
    workspace = WorkspaceFiles(
        WorkspaceSettings(
            root=str(tmp_path),
            read_cache_bytes=cache_bytes,
            max_batch_read_bytes=len(data) + 1,
        )
    )
    if cache_bytes:
        workspace.read("a")
    metrics = RunMetrics()
    token = CURRENT_METRICS.set(metrics)
    try:
        result = workspace.read_many(
            [
                ReadRequest(relative_path="invalid"),
                ReadRequest(relative_path="a", start_line=2, end_line=2),
                ReadRequest(relative_path="a", start_line=2, end_line=2),
            ]
        )
    finally:
        CURRENT_METRICS.reset(token)
    assert not result["complete"] and result["bytes_processed"] == len(data) + 1
    assert result["results"][1]["content"] == "answer\n"
    assert "remaining batch read budget" in result["results"][2]["error"]
    assert metrics.read_bytes + metrics.cached_read_bytes == len(data) + 1
    assert metrics.read_cache_hits == (1 if cache_bytes else 0)


def test_count_rejection_reads_nothing_and_single_file_limit_still_applies(tmp_path: Path) -> None:
    (tmp_path / "large").write_bytes(b"12345")
    workspace = WorkspaceFiles(
        WorkspaceSettings(root=str(tmp_path), max_batch_files=1, max_read_bytes=4)
    )
    for reads in ([], [ReadRequest(relative_path="large")] * 2):
        with pytest.raises(WorkspaceError, match="Batch file count"):
            workspace.read_many(reads)
    result = workspace.read_many([ReadRequest(relative_path="large")])
    assert not result["complete"] and result["bytes_processed"] == 0
    assert "configured read limit" in result["results"][0]["error"]


def test_skipped_oversized_file_does_not_prevent_later_small_source(tmp_path: Path) -> None:
    (tmp_path / "large").write_bytes(b"12345")
    (tmp_path / "small").write_bytes(b"ok")
    workspace = WorkspaceFiles(WorkspaceSettings(root=str(tmp_path), max_batch_read_bytes=2))
    result = workspace.read_many(
        [ReadRequest(relative_path="large"), ReadRequest(relative_path="small")]
    )
    assert not result["complete"] and result["bytes_processed"] == 2
    assert "error" in result["results"][0]
    assert result["results"][1]["content"] == "ok"


def test_change_during_read_reports_error_charges_bytes_and_does_not_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "changing"
    source.write_bytes(b"old")
    (tmp_path / "stable").write_bytes(b"y")
    original_stat = os.fstat
    calls = 0

    def changing_stat(descriptor):
        nonlocal calls
        calls += 1
        if calls == 2:
            info = source.stat()
            os.utime(source, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000_000))
        return original_stat(descriptor)

    workspace = WorkspaceFiles(
        WorkspaceSettings(root=str(tmp_path), max_batch_read_bytes=4, read_cache_bytes=100)
    )
    monkeypatch.setattr(os, "fstat", changing_stat)
    result = workspace.read_many(
        [ReadRequest(relative_path="changing"), ReadRequest(relative_path="stable")]
    )
    assert not result["complete"] and result["bytes_processed"] == 4
    assert "changed during batch read" in result["results"][0]["error"]
    assert result["results"][1]["content"] == "y"
    assert source not in workspace._read_cache
