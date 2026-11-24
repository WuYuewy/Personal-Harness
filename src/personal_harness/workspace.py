"""Bounded local text operations; execution policy lives in the tool adapter."""

from __future__ import annotations

import difflib
import fnmatch
import hashlib
import os
import stat
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from threading import RLock
from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from personal_harness.config import WorkspaceSettings
from personal_harness.context import CURRENT_METRICS


class WorkspaceError(ValueError):
    """A workspace operation was denied or cannot be completed."""


class WriteFingerprintError(WorkspaceError):
    """The proposed write does not match the target's current fingerprint."""


class ReadRequest(BaseModel):
    """Tool arguments for one file; not a new SDK execution abstraction."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    relative_path: Annotated[str, Field(min_length=1)]
    start_line: Annotated[int, Field(ge=1)] | None = None
    end_line: Annotated[int, Field(ge=1)] | None = None

    @model_validator(mode="after")
    def paired_lines(self) -> Self:
        if (self.start_line is None) != (self.end_line is None) or (
            self.start_line is not None
            and self.end_line is not None
            and self.start_line > self.end_line
        ):
            raise ValueError("Supply both line numbers in ascending order, starting at 1.")
        return self


@dataclass
class _ReadBudget:
    remaining: int


class WorkspaceFiles:
    """Application-level path checks for trusted local single-user execution."""

    def __init__(self, settings: WorkspaceSettings) -> None:
        self.settings = settings
        self.root = Path(settings.root).resolve()
        self._read_cache: OrderedDict[Path, tuple[tuple[int, ...], bytes]] = OrderedDict()
        self._cache_bytes = 0
        self._cache_lock = RLock()

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._read_cache.clear()
            self._cache_bytes = 0

    def _forget(self, path: Path) -> None:
        with self._cache_lock:
            previous = self._read_cache.pop(path, None)
            if previous is not None:
                self._cache_bytes -= len(previous[1])

    @staticmethod
    def _identity(info: os.stat_result) -> tuple[int, ...]:
        return (
            info.st_dev,
            info.st_ino,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
            info.st_mode,
            info.st_uid,
            info.st_gid,
        )

    def _cached(self, path: Path, identity: tuple[int, ...]) -> bytes | None:
        with self._cache_lock:
            previous = self._read_cache.pop(path, None)
            if previous is None:
                return None
            if previous[0] != identity:
                self._cache_bytes -= len(previous[1])
                return None
            self._read_cache[path] = previous
            return previous[1]

    def _remember(self, path: Path, identity: tuple[int, ...], data: bytes) -> None:
        with self._cache_lock:
            previous = self._read_cache.pop(path, None)
            if previous is not None:
                self._cache_bytes -= len(previous[1])
            if len(data) > self.settings.read_cache_bytes:
                return
            self._read_cache[path] = (identity, data)
            self._cache_bytes += len(data)
            # Bound entry overhead too, including empty files. Keep most recently used files.
            while self._cache_bytes > self.settings.read_cache_bytes or len(self._read_cache) > 128:
                _, (_, evicted) = self._read_cache.popitem(last=False)
                self._cache_bytes -= len(evicted)

    def _excluded(self, parts: tuple[str, ...]) -> bool:
        return any(
            fnmatch.fnmatchcase(part.casefold(), pattern.casefold())
            for part in parts
            for pattern in self.settings.exclude
        )

    @staticmethod
    def _link(path: Path) -> bool:
        info = path.lstat()
        return stat.S_ISLNK(info.st_mode) or bool(
            getattr(info, "st_file_attributes", 0) & 0x400  # Windows reparse points / junctions
        )

    def path(self, relative_path: str) -> Path:
        value = relative_path.replace("\\", "/")
        windows = PureWindowsPath(value)
        parts = Path(value).parts
        devices = {"con", "prn", "aux", "nul"} | {
            f"{prefix}{index}" for prefix in ("com", "lpt") for index in range(1, 10)
        }
        if (
            not value
            or "\x00" in value
            or Path(value).is_absolute()
            or windows.drive
            or windows.root
            or ".." in parts
            or any(":" in part or part.endswith((".", " ")) for part in parts if part != ".")
            or any(part.split(".")[0].casefold() in devices for part in parts)
        ):
            raise WorkspaceError("Use a relative path without traversal or device names.")
        if self._excluded(parts):
            raise WorkspaceError("Path is excluded by the user's workspace configuration.")
        if not self.root.is_dir():
            raise WorkspaceError("Configured workspace is not an existing directory.")
        candidate = self.root
        try:
            for part in parts:
                candidate /= part
                if candidate.exists() or candidate.is_symlink():
                    if self._link(candidate):
                        raise WorkspaceError(
                            "Symbolic links and reparse points are not accessible."
                        )
            if not candidate.resolve().is_relative_to(self.root):
                raise WorkspaceError("Path escapes the configured workspace.")
        except OSError as error:
            raise WorkspaceError("Cannot inspect the workspace path.") from error
        return candidate

    def _bytes(
        self, path: Path, limit: int, *, cache: bool = False, budget: _ReadBudget | None = None
    ) -> bytes:
        cache = cache and self.settings.read_cache_bytes > 0
        try:
            if not path.is_file():
                raise WorkspaceError("Path is not a regular file.")
            with path.open("rb") as stream:
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode):
                    raise WorkspaceError("Only regular files are supported.")
                if info.st_size > limit:
                    raise WorkspaceError("File exceeds the configured read limit.")
                if budget is not None and info.st_size > budget.remaining:
                    raise WorkspaceError("File exceeds the remaining batch read budget.")
                identity = self._identity(info)
                # Always reopen the file and inspect it before reusing bytes: path checks,
                # current OS read permissions, type and size limits still apply on a hit.
                cached = self._cached(path, identity) if cache else None
                if cached is not None:
                    if budget is not None:
                        budget.remaining -= len(cached)
                    metrics = CURRENT_METRICS.get()
                    if metrics is not None:
                        metrics.read_cache_hits += 1
                        metrics.cached_read_bytes += len(cached)
                    return cached
                data = stream.read(
                    limit + 1 if budget is None else min(limit + 1, budget.remaining)
                )
                if budget is not None:
                    budget.remaining -= len(data)
                metrics = CURRENT_METRICS.get()
                if metrics is not None:
                    metrics.read_bytes += len(data)
                stable = identity == self._identity(os.fstat(stream.fileno()))
        except OSError as error:
            raise WorkspaceError("Cannot read the workspace file.") from error
        if len(data) > limit:
            raise WorkspaceError("File exceeds the configured read limit.")
        if budget is not None and (not stable or len(data) != info.st_size):
            raise WorkspaceError("File changed during batch read; retry with current files.")
        if b"\x00" in data:
            raise WorkspaceError("Binary files are not supported.")
        try:
            data.decode("utf-8")
        except UnicodeError as error:
            raise WorkspaceError("Only UTF-8 text files are supported.") from error
        if cache and stable:
            self._remember(path, identity, data)
        return data

    def _write_path(self, relative_path: str) -> Path:
        path = self.path(relative_path)
        if self.settings.write_directory is not None:
            directory = self.path(self.settings.write_directory)
            if path == directory or not path.is_relative_to(directory):
                raise WorkspaceError("Target must be inside the configured write directory.")
        return path

    def read(
        self, relative_path: str, start_line: int | None = None, end_line: int | None = None
    ) -> dict[str, Any]:
        return self._read(relative_path, start_line, end_line)

    def _read(
        self,
        relative_path: str,
        start_line: int | None = None,
        end_line: int | None = None,
        *,
        budget: _ReadBudget | None = None,
    ) -> dict[str, Any]:
        if (start_line is None) != (end_line is None) or (
            start_line is not None and end_line is not None and not 1 <= start_line <= end_line
        ):
            raise WorkspaceError("Supply both line numbers in ascending order, starting at 1.")
        path = self.path(relative_path)
        data = self._bytes(path, self.settings.max_read_bytes, cache=True, budget=budget)
        content = data.decode("utf-8")
        result = {
            "path": path.relative_to(self.root).as_posix(),
            "content": content,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        if start_line is not None and end_line is not None:
            lines = content.splitlines(keepends=True)
            if start_line > len(lines):
                raise WorkspaceError("The starting line is past the end of the file.")
            last = min(end_line, len(lines))
            snippet = "".join(lines[start_line - 1 : last])
            result.update(
                content=snippet,
                bytes=len(snippet.encode("utf-8")),
                start_line=start_line,
                end_line=last,
            )
        return result

    def read_many(self, reads: list[ReadRequest]) -> dict[str, Any]:
        if not 1 <= len(reads) <= self.settings.max_batch_files:
            raise WorkspaceError("Batch file count must be within the configured limit.")
        budget = _ReadBudget(self.settings.max_batch_read_bytes)
        results: list[dict[str, Any]] = []
        complete = True
        # Local reads are ordered and bounded; PydanticAI still owns the model/tool loop.
        for index, request in enumerate(reads):
            try:
                result = self._read(
                    request.relative_path, request.start_line, request.end_line, budget=budget
                )
                results.append({"index": index, **result})
            except WorkspaceError as error:
                complete = False
                results.append({"index": index, "error": str(error)})
        return {
            "results": results,
            "complete": complete,
            "bytes_processed": self.settings.max_batch_read_bytes - budget.remaining,
        }

    def list(self, relative_path: str = ".") -> dict[str, Any]:
        path = self.path(relative_path)
        entries: list[dict[str, str]] = []
        truncated = False
        try:
            with os.scandir(path) as directory:
                for scanned, entry in enumerate(directory, 1):
                    if scanned > self.settings.max_scan_entries:
                        truncated = True
                        break
                    item = Path(entry.path)
                    if self._excluded(item.relative_to(self.root).parts) or self._link(item):
                        continue
                    kind = "directory" if entry.is_dir(follow_symlinks=False) else "file"
                    if kind == "file" and not entry.is_file(follow_symlinks=False):
                        continue
                    if len(entries) == self.settings.max_entries:
                        truncated = True
                        break
                    entries.append({"path": item.relative_to(self.root).as_posix(), "type": kind})
        except OSError as error:
            raise WorkspaceError("Cannot list the workspace directory.") from error
        entries.sort(key=lambda item: item["path"].casefold())
        return {"entries": entries, "truncated": truncated}

    def search(self, query: str, relative_path: str = ".") -> dict[str, Any]:
        if not query.strip() or "\x00" in query:
            raise WorkspaceError("Search query must be non-empty text.")
        folded_query = query.casefold()
        start = self.path(relative_path)
        directory_search = start.is_dir()
        if not directory_search and not start.is_file():
            raise WorkspaceError("Search path is not an existing directory or regular file.")
        matches: list[dict[str, Any]] = []
        pending = [start] if directory_search else []
        scanned = files = total_bytes = 0
        result_chars = 0
        skipped = 0
        truncated = False
        clipped = False

        def scan_file(path: Path) -> None:
            nonlocal files, total_bytes, skipped, truncated, result_chars, clipped
            if files >= self.settings.max_search_files:
                truncated = True
                return
            files += 1
            remaining = self.settings.max_search_bytes - total_bytes
            if remaining <= 0:
                truncated = True
                return
            # Charge the bounded read attempt too, including binary/invalid files.
            attempt_limit = min(self.settings.max_read_bytes, remaining)
            try:
                total_bytes += min(path.stat(follow_symlinks=False).st_size, attempt_limit)
                data = self._bytes(path, attempt_limit, cache=True)
            except (WorkspaceError, OSError):
                skipped += 1
                return
            for number, line in enumerate(data.decode("utf-8").splitlines(), 1):
                if folded_query in line.casefold():
                    remaining_chars = self.settings.max_search_result_chars - result_chars
                    if remaining_chars <= 0:
                        truncated = True
                        break
                    width = min(500, remaining_chars)
                    text = line[:width]
                    column = 0
                    folded_position = line.casefold().find(folded_query)
                    if folded_position + len(folded_query) > len(text.casefold()):
                        # Casefold can expand a character: folded offsets are not columns.
                        folded_count = 0
                        for offset, char in enumerate(line):
                            if folded_count + len(char.casefold()) > folded_position:
                                before = min(80, max(0, width - len(query)) // 2)
                                column = max(0, offset - before)
                                break
                            folded_count += len(char.casefold())
                        text = line[column : column + width]
                    item = {
                        "path": path.relative_to(self.root).as_posix(),
                        "line": number,
                        "text": text,
                    }
                    if column:
                        item["text_start_column"] = column + 1
                    matches.append(item)
                    result_chars += len(text)
                    clipped |= len(text) < len(line)
                    if len(matches) >= self.settings.max_search_results:
                        truncated = True
                        break

        if not directory_search:
            scan_file(start)
        while pending and not truncated:
            directory = pending.pop()
            try:
                with os.scandir(directory) as entries:
                    for entry in entries:
                        scanned += 1
                        if scanned > self.settings.max_scan_entries:
                            truncated = True
                            break
                        path = Path(entry.path)
                        if self._excluded(path.relative_to(self.root).parts) or self._link(path):
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            pending.append(path)
                            continue
                        if not entry.is_file(follow_symlinks=False):
                            continue
                        scan_file(path)
                        if truncated:
                            break
            except OSError:
                skipped += 1
        return {
            "matches": matches,
            "truncated": truncated or clipped,
            "files_scanned": files,
            "bytes_scanned": total_bytes,
            "skipped": skipped,
        }

    def preview(
        self, relative_path: str, content: str, *, expected_sha256: str | None = None
    ) -> dict[str, Any]:
        path = self._write_path(relative_path)
        encoded = content.encode("utf-8")
        if len(encoded) > self.settings.max_write_bytes or b"\x00" in encoded:
            raise WorkspaceError("Write must be text within the configured byte limit.")
        if not path.parent.is_dir():
            raise WorkspaceError("Target parent directory must already exist.")
        old = self._bytes(path, self.settings.max_read_bytes) if path.exists() else None
        fingerprint = hashlib.sha256(old).hexdigest() if old is not None else None
        if expected_sha256 is not None and expected_sha256 != fingerprint:
            raise WriteFingerprintError("File changed; read it again before proposing a write.")
        target = path.relative_to(self.root).as_posix()
        difference = "".join(
            difflib.unified_diff(
                (old.decode("utf-8") if old is not None else "").splitlines(keepends=True),
                content.splitlines(keepends=True),
                fromfile=f"before/{target}",
                tofile=f"after/{target}",
            )
        )
        return {
            "path": target,
            "bytes": len(encoded),
            "exists": old is not None,
            "expected_sha256": fingerprint,
            "content": content,
            "diff": difference[: self.settings.max_preview_chars],
            "diff_truncated": len(difference) > self.settings.max_preview_chars,
            "requires_confirmation": True,
            "written": False,
        }

    def preview_replace(
        self, relative_path: str, old_text: str, new_text: str, expected_sha256: str
    ) -> dict[str, Any]:
        """Resolve one exact replacement against disk; approvals never use the read cache."""
        if not old_text or old_text == new_text:
            raise WorkspaceError("Provide nonempty old text and a different replacement.")
        if len(expected_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_sha256):
            raise WorkspaceError("An existing file's full lowercase SHA-256 is required.")
        path = self._write_path(relative_path)
        original = self._bytes(path, self.settings.max_read_bytes)
        if hashlib.sha256(original).hexdigest() != expected_sha256:
            raise WorkspaceError("File changed; read it again before proposing a replacement.")
        try:
            text = original.decode("utf-8")
        except UnicodeError as error:
            raise WorkspaceError("Replacement requires a UTF-8 text file.") from error
        if "\x00" in text:
            raise WorkspaceError("Replacement requires a text file without null bytes.")
        # Detect overlapping occurrences too, e.g. "aa" in "aaa" is ambiguous.
        first = text.find(old_text)
        if first < 0 or text.find(old_text, first + 1) >= 0:
            raise WorkspaceError("Old text must match exactly once; use a larger unique excerpt.")
        proposed = text[:first] + new_text + text[first + len(old_text) :]
        return self.preview(relative_path, proposed, expected_sha256=expected_sha256)

    def replace_text(
        self, relative_path: str, old_text: str, new_text: str, expected_sha256: str
    ) -> dict[str, Any]:
        proposal = self.preview_replace(relative_path, old_text, new_text, expected_sha256)
        return self.write(relative_path, proposal["content"], expected_sha256)

    def write(
        self, relative_path: str, content: str, expected_sha256: str | None
    ) -> dict[str, Any]:
        preview = self.preview(relative_path, content, expected_sha256=expected_sha256)
        if preview["expected_sha256"] != expected_sha256:
            raise WorkspaceError("Existing file requires its current SHA-256; no file was changed.")
        path = self._write_path(relative_path)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=path.parent, prefix=".harness-", delete=False
            ) as stream:
                temporary = Path(stream.name)
                stream.write(content.encode("utf-8"))
                stream.flush()
                os.fsync(stream.fileno())
            # Recheck the path boundary and expected state immediately before committing.
            self._write_path(relative_path)
            if expected_sha256 is None:
                os.link(temporary, path)  # Atomic create-if-absent, never overwrites a new file.
            else:
                current = hashlib.sha256(
                    self._bytes(path, self.settings.max_read_bytes)
                ).hexdigest()
                if current != expected_sha256:
                    raise WorkspaceError(
                        "File changed while awaiting approval; no file was changed."
                    )
                os.chmod(temporary, stat.S_IMODE(path.stat().st_mode))
                os.replace(temporary, path)
            # Keep unchanged sources warm when writing a separate deliverable. The target
            # is invalidated; all other reads still reopen and validate their metadata.
            self._forget(path)
            return {
                "path": preview["path"],
                "written": True,
                "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            }
        except OSError as error:
            raise WorkspaceError(
                "Cannot commit the approved write; inspect the target before retrying."
            ) from error
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
