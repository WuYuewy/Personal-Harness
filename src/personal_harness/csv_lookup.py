"""Exact, local CSV lookup with source line ranges."""

from __future__ import annotations

import csv
from io import StringIO
from pathlib import PurePosixPath
from typing import Any

from personal_harness.config import WorkspaceSettings
from personal_harness.workspace import WorkspaceError, WorkspaceFiles

MAX_CSV_BYTES = 16 * 1024 * 1024


def lookup_csv(settings: WorkspaceSettings, path: str, column: str, value: str) -> dict[str, Any]:
    if any(
        not field or len(field) > 240 or not field.isprintable() for field in (path, column, value)
    ):
        raise WorkspaceError("请填写文件、列名和精确值，每项最多 240 个字符。")
    if PurePosixPath(path.replace("\\", "/")).suffix.lower() != ".csv":
        raise WorkspaceError("请指定 UTF-8 CSV 文件。")
    limit = min(settings.max_read_bytes, MAX_CSV_BYTES)
    try:
        snapshot = WorkspaceFiles(settings.model_copy(update={"max_read_bytes": limit})).read(path)
    except WorkspaceError as error:
        raise WorkspaceError(f"无法读取 CSV（上限 {limit:,} 字节）：{error}") from None

    reader = csv.reader(
        StringIO(snapshot["content"].removeprefix("\ufeff"), newline=""), strict=True
    )
    rows: list[dict[str, Any]] = []
    count = shown_chars = 0
    try:
        headers = next(reader, [])
        if (
            not 1 <= len(headers) <= 128
            or any(not item or len(item) > 240 or not item.isprintable() for item in headers)
            or len(set(headers)) != len(headers)
        ):
            raise WorkspaceError("CSV 首行需要有效且不重复的列名（最多 128 列）。")
        if column not in headers:
            raise WorkspaceError(f"CSV 中没有列「{column}」。")
        index = headers.index(column)
        result_chars = min(settings.max_search_result_chars, 32_000)
        while True:
            start = reader.line_num + 1
            row = next(reader, None)
            if row is None:
                break
            if not row:
                continue
            if len(row) != len(headers):
                raise WorkspaceError(f"CSV 第 {start} 行的字段数与表头不符。")
            if row[index] != value:
                continue
            count += 1
            width = sum(map(len, row)) + len(row)
            if len(rows) < settings.max_search_results and shown_chars + width <= result_chars:
                rows.append({"start_line": start, "end_line": reader.line_num, "values": row})
                shown_chars += width
    except csv.Error:
        raise WorkspaceError(f"无法解析 CSV，第 {reader.line_num} 行附近的格式有误。") from None
    return {
        "path": snapshot["path"],
        "headers": headers,
        "rows": rows,
        "match_count": count,
        "truncated": count > len(rows),
    }
