"""Check publishable paths and text for local data and credential-shaped values."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

IGNORED = {
    ".git",
    ".venv",
    ".uv-cache",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "__pycache__",
    "dist",
    "build",
}
KEY = re.compile(rb"(?:sk-[A-Za-z0-9_-]{16,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)")


def is_sensitive_path(path: str) -> bool:
    parts = tuple(p.casefold() for p in path.replace("\\", "/").split("/"))
    name = parts[-1]
    if any(p in {".personal-harness", ".aws", ".ssh", ".chainlit", ".files"} for p in parts):
        return True
    if name == ".env.example":
        return False
    return (
        name == ".env"
        or name.startswith(".env.")
        or name == "personal-harness.toml"
        or (name.startswith("personal-harness.") and name.endswith(".toml"))
        or ".sqlite" in name
        or name == "chainlit.md"
        or name.endswith((".pem", ".p12", ".pfx"))
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tree", type=Path, help="Inspect a clean source directory before Git initialization."
    )
    parser.add_argument(
        "--forbid",
        action="append",
        default=[],
        help="Reject an additional literal text fragment (case-insensitive).",
    )
    args = parser.parse_args()
    root = args.tree.resolve() if args.tree else Path(__file__).resolve().parents[1]
    if args.tree:
        paths = [
            p.relative_to(root).as_posix()
            for p in root.rglob("*")
            if p.is_file() and not any(x in IGNORED for x in p.relative_to(root).parts)
        ]
    else:
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], cwd=root, capture_output=True, text=True
        )
        if top.returncode or Path(top.stdout.strip()).resolve() != root:
            print(
                "No Git index at project root. Use --tree . before initializing Git.",
                file=sys.stderr,
            )
            return 2
        result = subprocess.run(
            ["git", "ls-files", "--cached", "-z"], cwd=root, capture_output=True
        )
        if result.returncode:
            return 2
        paths = [p.decode("utf-8") for p in result.stdout.split(b"\0") if p]
    findings = []
    for name in paths:
        path = root / name
        if is_sensitive_path(name):
            findings.append((name, "private path"))
            continue
        if not path.is_file():
            findings.append((name, "missing indexed file"))
            continue
        data = path.read_bytes()
        if KEY.search(data):
            findings.append((name, "possible credential"))
        lower = data.lower()
        if any(fragment.encode().lower() in lower for fragment in args.forbid):
            findings.append((name, "forbidden text fragment"))
    if findings:
        for name, reason in findings:
            print(f"{name}: {reason}", file=sys.stderr)
        return 1
    print(f"Release audit passed: {len(paths)} paths and file contents checked.")
    print("Heuristic scan; a maintainer must still review the release.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
