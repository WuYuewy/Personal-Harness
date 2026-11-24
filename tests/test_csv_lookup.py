import pytest

from personal_harness.config import WorkspaceSettings
from personal_harness.csv_lookup import lookup_csv
from personal_harness.workspace import WorkspaceError


def test_lookup_preserves_quoted_rows_and_reports_all_exact_matches(tmp_path):
    path = tmp_path / "records.csv"
    path.write_bytes(
        b'\xef\xbb\xbfid,note\r\n"A 1","first,\r\nsecond"\r\n\r\nA 1,duplicate\r\na 1,lower\r\n'
    )
    settings = WorkspaceSettings(root=str(tmp_path), max_search_results=1)
    result = lookup_csv(settings, path.name, "id", "A 1")
    assert result["rows"] == [
        {"start_line": 2, "end_line": 3, "values": ["A 1", "first,\r\nsecond"]}
    ]
    assert result["match_count"] == 2 and result["truncated"]
    assert lookup_csv(settings, path.name, "id", "A")["match_count"] == 0


@pytest.mark.parametrize(
    ("content", "error"),
    [
        ("id,id\nA,B\n", "不重复"),
        ("id,note\nA,match\nB\n", "第 3 行"),
        ('id,note\nA,"unclosed', "无法解析"),
    ],
)
def test_lookup_does_not_report_complete_results_from_invalid_csv(tmp_path, content, error):
    (tmp_path / "bad.csv").write_text(content, encoding="utf-8")
    with pytest.raises(WorkspaceError, match=error):
        lookup_csv(WorkspaceSettings(root=str(tmp_path)), "bad.csv", "id", "A")


def test_lookup_observes_workspace_limits(tmp_path):
    (tmp_path / "records.csv").write_text("id,note\nA,example\n", encoding="utf-8")
    settings = WorkspaceSettings(root=str(tmp_path), max_read_bytes=12)
    with pytest.raises(WorkspaceError, match="上限 12 字节"):
        lookup_csv(settings, "records.csv", "id", "A")
    with pytest.raises(WorkspaceError, match="relative path"):
        lookup_csv(settings, "../records.csv", "id", "A")


def test_lookup_finds_exact_invoice_in_long_file(tmp_path):
    rows = ["发票编号,部门,金额元,状态"]
    rows.extend(f"INV-{i:04d},研发,{847.50 if i == 173 else 10:.2f},待报销" for i in range(1, 201))
    (tmp_path / "expenses.csv").write_text("\n".join(rows), "utf-8")
    result = lookup_csv(
        WorkspaceSettings(root=str(tmp_path)), "expenses.csv", "发票编号", "INV-0173"
    )
    assert result["rows"] == [
        {"start_line": 174, "end_line": 174, "values": ["INV-0173", "研发", "847.50", "待报销"]}
    ]
