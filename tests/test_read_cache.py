import hashlib
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from personal_harness.config import WorkspaceSettings
from personal_harness.context import CURRENT_METRICS, RunMetrics
from personal_harness.workspace import WorkspaceError, WorkspaceFiles


@pytest.fixture
def metrics():
    usage = RunMetrics()
    token = CURRENT_METRICS.set(usage)
    try:
        yield usage
    finally:
        CURRENT_METRICS.reset(token)


def cached_workspace(root: Path, capacity: int = 1000, **settings) -> WorkspaceFiles:
    return WorkspaceFiles(WorkspaceSettings(root=str(root), read_cache_bytes=capacity, **settings))


@pytest.mark.parametrize("capacity", [-1, True, 1.5, 16_000_001])
def test_cache_capacity_is_strict_and_bounded(capacity) -> None:
    with pytest.raises(ValidationError):
        WorkspaceSettings(read_cache_bytes=capacity)


def test_read_and_search_share_bytes_but_preserve_results_and_scan_limits(
    tmp_path: Path, metrics: RunMetrics
) -> None:
    data = "header\nneedle: 你好\n".encode()
    (tmp_path / "notes.md").write_bytes(data)
    workspace = cached_workspace(tmp_path)
    original = workspace.read("notes.md")
    excerpt = workspace.read("./notes.md", 2, 2)
    report = workspace.search("needle")
    assert excerpt["content"] == "needle: 你好\n"
    assert original["sha256"] == excerpt["sha256"] == hashlib.sha256(data).hexdigest()
    assert report["matches"] == [{"path": "notes.md", "line": 2, "text": "needle: 你好"}]
    assert report["bytes_scanned"] == len(data)
    assert metrics.read_bytes == len(data)
    assert metrics.read_cache_hits == 2 and metrics.cached_read_bytes == 2 * len(data)

    limited = cached_workspace(tmp_path, max_search_bytes=3)
    limited.read("notes.md")
    before = metrics.read_cache_hits
    report = limited.search("needle")
    assert not report["matches"] and report["skipped"] == 1
    assert report["bytes_scanned"] <= 3 and metrics.read_cache_hits == before


def test_same_size_edit_replacement_and_deletion_do_not_reuse_old_bytes(
    tmp_path: Path, metrics: RunMetrics
) -> None:
    source = tmp_path / "notes.md"
    source.write_bytes(b"old")
    workspace = cached_workspace(tmp_path)
    first = workspace.read("notes.md")
    old_stat = source.stat()
    source.write_bytes(b"new")
    os.utime(source, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns + 1_000_000_000))
    second = workspace.read("notes.md")
    assert second["content"] == "new" and second["sha256"] != first["sha256"]

    replacement = tmp_path / "replacement.md"
    replacement.write_bytes(b"now")
    current = source.stat()
    os.utime(replacement, ns=(current.st_atime_ns, current.st_mtime_ns))
    os.replace(replacement, source)
    assert workspace.read("notes.md")["content"] == "now"
    assert metrics.read_bytes == 9 and metrics.read_cache_hits == 0
    source.unlink()
    with pytest.raises(WorkspaceError, match="regular file"):
        workspace.read("notes.md")


def test_hit_reopens_file_and_new_workspace_permissions_do_not_share_cache(
    tmp_path: Path, metrics: RunMetrics, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "notes.md"
    source.write_bytes(b"hello")
    workspace = cached_workspace(tmp_path)
    workspace.read("notes.md")
    original_open = Path.open

    def deny(path, *args, **kwargs):
        if path == source:
            raise PermissionError("read permission removed")
        return original_open(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", deny)
        with pytest.raises(WorkspaceError, match="Cannot read"):
            workspace.read("notes.md")
    with pytest.raises(WorkspaceError, match="excluded"):
        cached_workspace(tmp_path, exclude=("notes.md",)).read("notes.md")
    with pytest.raises(WorkspaceError, match="read limit"):
        cached_workspace(tmp_path, max_read_bytes=2).read("notes.md")
    cached_workspace(tmp_path).read("notes.md")
    assert metrics.read_bytes == 10 and metrics.read_cache_hits == 0


def test_lru_capacity_disabled_cache_and_oversized_entries(
    tmp_path: Path, metrics: RunMetrics
) -> None:
    for name in ("a", "b", "c"):
        (tmp_path / name).write_bytes(name.encode() * 3)
    workspace = cached_workspace(tmp_path, 6)
    for name in ("a", "b", "a", "c", "a", "b"):
        workspace.read(name)
    assert metrics.read_bytes == 12 and metrics.read_cache_hits == 2
    assert workspace._cache_bytes == 6
    for capacity in (0, 2):
        uncached = cached_workspace(tmp_path, capacity)
        uncached.read("a")
        uncached.read("a")
        assert not uncached._read_cache
    assert metrics.read_bytes == 24 and metrics.read_cache_hits == 2


def test_empty_entries_are_bounded_and_invalid_text_is_never_cached(
    tmp_path: Path, metrics: RunMetrics
) -> None:
    workspace = cached_workspace(tmp_path, 1)
    for number in range(130):
        (tmp_path / str(number)).write_bytes(b"")
        workspace.read(str(number))
    workspace.read("129")
    assert len(workspace._read_cache) == 128 and metrics.read_cache_hits == 1
    for data in (b"\x00", b"\xff"):
        (tmp_path / "invalid").write_bytes(data)
        for _ in range(2):
            with pytest.raises(WorkspaceError):
                workspace.read("invalid")
        assert tmp_path / "invalid" not in workspace._read_cache


def test_approval_reads_bypass_cache_and_successful_writes_invalidate_target(
    tmp_path: Path, metrics: RunMetrics
) -> None:
    source = tmp_path / "notes.md"
    source.write_bytes(b"old")
    workspace = cached_workspace(tmp_path)
    old_hash = workspace.read("notes.md")["sha256"]
    workspace.preview("notes.md", "new", expected_sha256=old_hash)
    assert metrics.read_bytes == 6 and metrics.read_cache_hits == 0
    source.write_bytes(b"changed outside")
    with pytest.raises(WorkspaceError, match="changed"):
        workspace.write("notes.md", "new", old_hash)
    assert source.read_bytes() == b"changed outside"
    current = workspace.read("notes.md")["sha256"]
    workspace.write("notes.md", "new", current)
    assert not workspace._read_cache
    assert workspace.read("notes.md")["content"] == "new"
    workspace.clear_cache()
    assert not workspace._read_cache and workspace._cache_bytes == 0


def test_writing_deliverable_preserves_sources_and_invalidates_only_target(
    tmp_path: Path, metrics: RunMetrics
) -> None:
    (tmp_path / "source.md").write_bytes(b"source")
    (tmp_path / "brief.md").write_bytes(b"old")
    workspace = cached_workspace(tmp_path)
    source = workspace.read("source.md")
    target = workspace.read("brief.md")
    workspace.write("brief.md", "new", target["sha256"])
    assert workspace._cache_bytes == len(b"source")
    before = metrics.read_bytes
    assert workspace.read("source.md") == source
    assert metrics.read_bytes == before and metrics.read_cache_hits == 1
    assert workspace.read("brief.md")["content"] == "new"
    assert metrics.read_bytes == before + 3
    workspace.write("other.md", "new deliverable", None)
    before = metrics.read_bytes
    workspace.read("source.md")
    assert metrics.read_bytes == before and metrics.read_cache_hits == 2
