import pytest

from scripts.audit_release import is_sensitive_path


@pytest.mark.parametrize(
    "path",
    [
        ".env",
        "nested/.ENV.local",
        "personal-harness.toml",
        "personal-harness.online.toml",
        ".personal-harness/reviews/decisions.json",
        "nested/ledger.sqlite3-wal",
        "examples/.files/4/document.pdf",
        "client.pem",
        ".ssh/id_rsa",
    ],
)
def test_release_audit_detects_private_paths(path: str) -> None:
    assert is_sensitive_path(path)


@pytest.mark.parametrize(
    "path",
    [
        ".env.example",
        "src/personal_harness/ledger.py",
        "docs/REAL_TASKS.md",
        "examples/personal-harness.toml.example",
        "tests/test_ledger.py",
    ],
)
def test_release_audit_allows_project_sources_and_templates(path: str) -> None:
    assert not is_sensitive_path(path)


def test_release_audit_is_case_insensitive_and_normalizes_windows_separators() -> None:
    assert is_sensitive_path(r".PERSONAL-HARNESS\reviews\review.json")
