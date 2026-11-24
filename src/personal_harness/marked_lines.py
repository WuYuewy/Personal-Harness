"""Bounded, cited extraction of explicitly marked lines from one workspace file."""

from __future__ import annotations

import json
import re
from typing import Any

from personal_harness.workspace import WorkspaceFiles

MAX_ACTION_RESULTS = 500
MAX_ACTION_RESULT_BYTES = 256_000
_LINE_END = re.compile(r"\r\n?|\n|\Z")


def extract_marked_lines(files: WorkspaceFiles, relative_path: str, prefix: str) -> dict[str, Any]:
    """Return complete matching source lines, with their path and one-based line numbers."""
    if not prefix.strip() or len(prefix) > 80 or not prefix.isprintable():
        raise ValueError("请输入 1–80 个字符的单行标记。")
    source = files.read(relative_path)
    content: str = source["content"]
    actions: list[dict[str, Any]] = []
    match_count = 0
    result_bytes = 0
    line_start = 1 if content.startswith("\ufeff") else 0
    for number, ending in enumerate(_LINE_END.finditer(content), 1):
        start, stop = line_start, ending.start()
        line_start = ending.end()
        if not content.startswith(prefix, start, stop):
            continue
        match_count += 1
        if len(actions) >= MAX_ACTION_RESULTS or stop - start > MAX_ACTION_RESULT_BYTES:
            continue
        item = {"path": source["path"], "line": number, "text": content[start:stop]}
        size = len(json.dumps(item, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) + 1
        if result_bytes + size > MAX_ACTION_RESULT_BYTES:
            continue
        actions.append(item)
        result_bytes += size
    displayed_count = len(actions)
    return {
        "actions": actions,
        "path": source["path"],
        "prefix": prefix,
        "sha256": source["sha256"],
        "match_count": match_count,
        "displayed_count": displayed_count,
        "omitted_count": match_count - displayed_count,
        "truncated": match_count > displayed_count,
    }
