"""Deterministic local CSV numeric summaries."""

from __future__ import annotations

import csv
from decimal import Decimal, InvalidOperation
from io import StringIO
from pathlib import PurePosixPath
from typing import Any

from personal_harness.config import WorkspaceSettings
from personal_harness.workspace import WorkspaceError, WorkspaceFiles

MAX_CSV_BYTES = 16 * 1024 * 1024


def summarize_csv(
    settings: WorkspaceSettings,
    path: str,
    numeric_column: str,
    filter_column: str = "",
    filter_value: str = "",
) -> dict[str, Any]:
    fields = (path, numeric_column, filter_column, filter_value)
    if any(len(field) > 240 or (field and not field.isprintable()) for field in fields):
        raise WorkspaceError("请填写有效的 CSV 汇总条件，每项最多 240 个字符。")
    if bool(filter_column) != bool(filter_value):
        raise WorkspaceError("筛选列和筛选值需要同时填写，或同时留空。")
    if not path or not numeric_column:
        raise WorkspaceError("请填写 CSV 路径和数字列名。")
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
    try:
        headers = next(reader, [])
        if (
            not 1 <= len(headers) <= 128
            or any(not item or len(item) > 240 or not item.isprintable() for item in headers)
            or len(set(headers)) != len(headers)
        ):
            raise WorkspaceError("CSV 首行需要有效且不重复的列名（最多 128 列）。")
        if numeric_column not in headers:
            raise WorkspaceError(f"CSV 中没有数字列「{numeric_column}」。")
        if filter_column and filter_column not in headers:
            raise WorkspaceError(f"CSV 中没有筛选列「{filter_column}」。")
        numeric_index = headers.index(numeric_column)
        filter_index = headers.index(filter_column) if filter_column else None
        count = 0
        total = Decimal(0)
        for row in reader:
            start = reader.line_num
            if not row:
                continue
            if len(row) != len(headers):
                raise WorkspaceError(f"CSV 第 {start} 行的字段数与表头不符。")
            if filter_index is not None and row[filter_index] != filter_value:
                continue
            raw = row[numeric_index].strip()
            try:
                amount = Decimal(raw)
            except InvalidOperation:
                raise WorkspaceError(
                    f"CSV 第 {start} 行的「{numeric_column}」不是有效数字。"
                ) from None
            if not amount.is_finite():
                raise WorkspaceError(f"CSV 第 {start} 行的数字必须是有限值。")
            count += 1
            total += amount
    except csv.Error:
        raise WorkspaceError(f"无法解析 CSV，第 {reader.line_num} 行附近的格式有误。") from None
    average = total / count if count else Decimal(0)
    return {
        "path": snapshot["path"],
        "numeric_column": numeric_column,
        "filter_column": filter_column,
        "filter_value": filter_value,
        "matched_count": count,
        "total": format(total, "f"),
        "average": format(average, "f"),
    }
