from decimal import Decimal

import pytest

from personal_harness.config import WorkspaceSettings
from personal_harness.csv_summary import summarize_csv
from personal_harness.workspace import WorkspaceError


def test_summary_filters_decimal_values_and_preserves_zero_result(tmp_path):
    path = tmp_path / "expenses.csv"
    path.write_text(
        "id,status,amount\nA,待报销,847.50\nB,已报销,12.5\nC,待报销,2\n",
        encoding="utf-8",
    )
    settings = WorkspaceSettings(root=str(tmp_path))
    result = summarize_csv(settings, path.name, "amount", "status", "待报销")
    assert result["matched_count"] == 2
    assert result["total"] == "849.50" and result["average"] == "424.75"
    empty = summarize_csv(settings, path.name, "amount", "status", "不存在")
    assert empty["matched_count"] == 0 and empty["total"] == "0" and empty["average"] == "0"


@pytest.mark.parametrize(
    ("content", "kwargs", "error"),
    [
        ("id,amount\nA,nope\n", {}, "不是有效数字"),
        ("id,amount\nA,1\nB\n", {}, "字段数"),
        ("id,amount\nA,1\n", {"filter_column": "missing", "filter_value": "x"}, "没有筛选列"),
        ("id,amount\nA,1\n", {"filter_column": "status"}, "同时填写"),
    ],
)
def test_summary_refuses_ambiguous_or_invalid_data(tmp_path, content, kwargs, error):
    (tmp_path / "bad.csv").write_text(content, encoding="utf-8")
    with pytest.raises(WorkspaceError, match=error):
        summarize_csv(WorkspaceSettings(root=str(tmp_path)), "bad.csv", "amount", **kwargs)


def test_summary_uses_workspace_limits_and_decimal_sum(tmp_path):
    path = tmp_path / "records.csv"
    path.write_text("id,amount\nA,0.1\nB,0.2\n", encoding="utf-8")
    result = summarize_csv(WorkspaceSettings(root=str(tmp_path)), path.name, "amount")
    assert Decimal(result["total"]) == Decimal("0.3")
    with pytest.raises(WorkspaceError, match="上限 8 字节"):
        summarize_csv(WorkspaceSettings(root=str(tmp_path), max_read_bytes=8), path.name, "amount")
