"""Turn explicitly marked workspace notes into a reusable, cited Markdown report."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from html import escape
from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, TypeAdapter, model_validator

from personal_harness.config import ResolvedConfig
from personal_harness.workspace import ReadRequest, WorkspaceError, WorkspaceFiles

Label = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=80)]
RelativePath = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=240)
]
SourcePaths = Annotated[tuple[RelativePath, ...], Field(min_length=1, max_length=8)]
_source_paths = TypeAdapter(SourcePaths)
REPORT_TOOLS = frozenset({"workspace.read", "workspace.write"})
MAX_REPORT_BYTES = 48_000
MAX_DISCOVERED_MARKERS = 24


class ReportSection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, hide_input_in_errors=True)
    heading: Label
    prefix: Label


class ReportRecipe(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
    name: Label
    sources: SourcePaths
    target: RelativePath
    sections: Annotated[tuple[ReportSection, ...], Field(min_length=1, max_length=6)]

    @model_validator(mode="after")
    def validate_recipe(self) -> Self:
        if not all(
            value.isprintable()
            for value in (
                self.name,
                *self.sources,
                self.target,
                *(
                    value
                    for section in self.sections
                    for value in (section.heading, section.prefix)
                ),
            )
        ):
            raise ValueError("Use printable, single-line report fields.")
        prefixes = [section.prefix for section in self.sections]
        if any(
            a.startswith(b)
            for i, a in enumerate(prefixes)
            for j, b in enumerate(prefixes)
            if i != j
        ):
            raise ValueError("Section prefixes must be distinct and cannot overlap.")
        if not self.target.lower().endswith(".md"):
            raise ValueError("Choose a Markdown output path ending in .md.")
        return self


def _markdown_text(text: str) -> str:
    return re.sub(r"([\\`*_\[\]~|])", r"\\\1", escape(text, quote=False))


@dataclass(frozen=True)
class ReportDraft:
    workspace_root: str
    path: str
    content: str
    diff: str
    diff_truncated: bool
    destination_sha256: str | None
    sources: tuple[tuple[str, str], ...]
    unchanged: bool

    def public(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "content": self.content,
            "diff": self.diff,
            "diff_truncated": self.diff_truncated,
            "source_count": len(self.sources),
            "unchanged": self.unchanged,
            "model_requests": 0,
            "cost_cny": "0",
        }


def _workspace(resolved: ResolvedConfig) -> WorkspaceFiles:
    if not REPORT_TOOLS.issubset(resolved.config.tools.allowed):
        raise WorkspaceError("请在当前聊天授权 workspace.read 和 workspace.write。")
    # Every preview and approval gets a fresh, bounded read of the current files.
    return WorkspaceFiles(resolved.config.workspace)


def _read_sources(files: WorkspaceFiles, sources: tuple[str, ...]) -> list[dict[str, Any]]:
    source_paths = [files.path(source) for source in sources]
    if len(set(source_paths)) != len(source_paths):
        raise WorkspaceError("来源必须各不相同。")
    batch = files.read_many([ReadRequest(relative_path=path) for path in sources])
    for item in batch["results"]:
        if "error" in item:
            raise WorkspaceError(f"无法完整读取来源 {sources[item['index']]}：{item['error']}")
    results: list[dict[str, Any]] = batch["results"]
    return results


def _source_line(line: str, number: int) -> str:
    return (line.removeprefix("\ufeff") if number == 1 else line).lstrip()


def inspect_report_sources(resolved: ResolvedConfig, sources: list[str]) -> dict[str, Any]:
    """Suggest literal line markers from explicitly chosen files, without a model or write."""
    if "workspace.read" not in resolved.config.tools.allowed:
        raise WorkspaceError("请在当前聊天授权 workspace.read。")
    files = WorkspaceFiles(resolved.config.workspace)
    results = _read_sources(files, _source_paths.validate_python(sources))
    candidates: dict[str, dict[str, Any]] = {}
    omitted = False
    for item in results:
        for number, raw in enumerate(item["content"].splitlines(), 1):
            line = _source_line(raw, number)
            checkbox = re.match(r"^([-*+]\s+\[([ xX])\])\s*(\S.*)$", line)
            marker = re.match(r"^([^:：]{1,60}[:：])\s*(\S.*)$", line)
            if checkbox:
                prefix, state, value = checkbox.groups()
                heading = "未完成事项" if state == " " else "已完成事项"
            elif marker:
                prefix, value = marker.groups()
                heading = re.sub(r"^(?:[-*+]\s+|#{1,6}\s+)", "", prefix[:-1]).strip()
                if value.startswith("//"):
                    continue
            else:
                continue
            if not heading or not prefix.isprintable() or "|" in prefix:
                continue
            if prefix not in candidates:
                if len(candidates) >= MAX_DISCOVERED_MARKERS:
                    omitted = True
                    continue
                candidates[prefix] = {
                    "heading": heading,
                    "prefix": prefix,
                    "count": 0,
                    "example": value[:160],
                    "example_truncated": len(value) > 160,
                    "path": item["path"],
                    "line": number,
                }
            candidates[prefix]["count"] += 1
    return {
        "candidates": list(candidates.values()),
        "source_count": len(results),
        "candidate_limit": MAX_DISCOVERED_MARKERS,
        "additional_markers_omitted": omitted,
        "model_requests": 0,
        "cost_cny": "0",
    }


def prepare_report(resolved: ResolvedConfig, recipe: ReportRecipe) -> ReportDraft:
    files = _workspace(resolved)
    target = files.path(recipe.target)
    source_paths = [files.path(source) for source in recipe.sources]
    if target in source_paths or len(set(source_paths)) != len(source_paths):
        raise WorkspaceError("来源必须各不相同，且不能与输出文件相同。")
    results = _read_sources(files, recipe.sources)

    lines = [f"# {_markdown_text(recipe.name)}", ""]
    matched: list[list[str]] = [[] for _ in recipe.sections]
    remaining = MAX_REPORT_BYTES
    for item in results:
        for number, line in enumerate(item["content"].splitlines(), 1):
            line = _source_line(line, number)
            for index, section in enumerate(recipe.sections):
                if line.startswith(section.prefix) and (
                    value := line.removeprefix(section.prefix).strip()
                ):
                    entry = f"- {_markdown_text(value)}（{_markdown_text(item['path'])}:{number}）"
                    remaining -= len(entry.encode("utf-8")) + 1
                    if remaining < 0:
                        raise WorkspaceError("匹配内容超过 48 KB，请缩小来源或收紧前缀。")
                    matched[index].append(entry)
                    break
    for section, entries in zip(recipe.sections, matched, strict=True):
        if not entries:
            raise WorkspaceError(f"「{section.heading}」没有匹配到内容，未生成文件。")
        lines.extend([f"## {_markdown_text(section.heading)}", "", *entries, ""])
    content = "\n".join(lines)
    if len(content.encode("utf-8")) > MAX_REPORT_BYTES:
        raise WorkspaceError("简报超过 48 KB，请缩小来源或收紧前缀。")
    preview = files.preview(recipe.target, content)
    return ReportDraft(
        path=preview["path"],
        content=content,
        diff=preview["diff"],
        diff_truncated=preview["diff_truncated"],
        destination_sha256=preview["expected_sha256"],
        sources=tuple((item["path"], item["sha256"]) for item in results),
        unchanged=preview["exists"]
        and preview["expected_sha256"] == hashlib.sha256(content.encode("utf-8")).hexdigest(),
        workspace_root=str(files.root),
    )


def apply_report(resolved: ResolvedConfig, recipe: ReportRecipe, draft: ReportDraft) -> None:
    if draft.unchanged or prepare_report(resolved, recipe) != draft:
        raise WorkspaceError("来源或输出已变化，请重新生成并确认预览。")
    _workspace(resolved).write(draft.path, draft.content, draft.destination_sha256)
