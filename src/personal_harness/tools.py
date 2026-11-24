"""PydanticAI tools selected by the user's explicit allowlist."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Sequence
from typing import Annotated, Any, Literal

from pydantic import Field
from pydantic_ai import ModelRetry, RunContext, Tool
from pydantic_ai.tools import ToolDefinition

from personal_harness.config import ToolName
from personal_harness.context import (
    CURRENT_METRICS,
    RunDependencies,
    build_context,
    render_model_context,
)
from personal_harness.places import search_places
from personal_harness.providers import CONTEXT_TOOL
from personal_harness.workspace import (
    ReadRequest,
    WorkspaceError,
    WorkspaceFiles,
    WriteFingerprintError,
)

TOOL_NAMES: dict[str, ToolName] = {
    CONTEXT_TOOL: "context.describe",
    "workspace_list": "workspace.list",
    "workspace_read": "workspace.read",
    "workspace_read_many": "workspace.read_many",
    "workspace_search": "workspace.search",
    "workspace_write_preview": "workspace.write_preview",
    "workspace_write": "workspace.write",
    "workspace_replace_text": "workspace.replace_text",
    "places_search": "places.search",
}


def logical_tool_name(name: str) -> str:
    """Custom functions use a reserved namespace, without a plugin registry."""
    return TOOL_NAMES.get(
        name,
        "user." + name[5:] if re.fullmatch(r"user_[a-z][a-z0-9_]{0,63}", name) else "unregistered",
    )


def personal_tools(
    deps: RunDependencies, functions: Sequence[Callable[..., Any]]
) -> list[Tool[RunDependencies]]:
    tools = configured_tools(deps)
    names: set[str] = set()
    for function in functions:
        name = getattr(function, "__name__", "")
        if not name or name in names:
            raise ValueError("Custom functions need distinct names.")
        names.add(name)
        logical = "user." + name
        if logical not in deps.resolved.config.tools.allowed:
            continue
        tools.append(Tool(function, name="user_" + name, prepare=prepare_task_tool))
    missing = {
        name
        for name in deps.resolved.config.tools.allowed
        if name.startswith("user.") and name[5:] not in names
    }
    if missing:
        raise ValueError("Supply Python functions for the configured user tools.")
    return tools


def select_task_tools(allowed: Sequence[str], requested: Sequence[str] | None) -> tuple[str, ...]:
    """Take a stable subset of personal permissions without widening them."""
    if requested is not None and (
        isinstance(requested, (str, bytes))
        or not isinstance(requested, Sequence)
        or any(not isinstance(name, str) or name not in allowed for name in requested)
    ):
        raise ValueError("Task tools must be a sequence drawn from tools.allowed.")
    selected = set(allowed if requested is None else requested)
    return tuple(name for name in dict.fromkeys(allowed) if name in selected)


async def prepare_task_tool(
    ctx: RunContext[RunDependencies], definition: ToolDefinition
) -> ToolDefinition | None:
    """Use the framework's prepare hook to remove tools for this run."""
    active = set(ctx.deps.active_tools)
    return definition if logical_tool_name(definition.name) in active else None


def describe_context(
    ctx: RunContext[RunDependencies],
    section: Literal["all", "preferences", "budget", "workspace"] = "all",
) -> str:
    """Inspect personal context without external side effects."""
    if ctx.deps.resolved.config.model.provider != "fake":
        return render_model_context(ctx.deps)
    context = build_context(ctx.deps.resolved)
    return json.dumps(
        context if section == "all" else context[section], ensure_ascii=False, indent=2
    )


def _result(operation: Callable[[], dict[str, Any]]) -> str:
    try:
        return json.dumps(operation(), ensure_ascii=False)
    except WorkspaceError as error:
        raise ModelRetry(str(error)) from error


def _workspace(ctx: RunContext[RunDependencies]) -> WorkspaceFiles:
    return ctx.deps.workspace_files or WorkspaceFiles(ctx.deps.resolved.config.workspace)


def _read_page(
    result: dict[str, Any], maximum: int, *, line_limit: int | None = None
) -> dict[str, Any]:
    """Bound model-facing text by complete lines; keep full sources on disk.

    The limit counts text characters, not tokens or JSON metadata. A line that cannot fit is
    never silently split: its number remains the continuation point and the limit can be raised.
    """
    content = result.get("content")
    if not isinstance(content, str) or len(content) <= maximum:
        return result
    lines = content.splitlines(keepends=True)
    kept: list[str] = []
    used = 0
    for line in lines:
        if used + len(line) > maximum:
            break
        kept.append(line)
        used += len(line)
    page = "".join(kept)
    first = result.get("start_line", 1)
    last = first + len(lines) - 1
    metrics = CURRENT_METRICS.get()
    if metrics is not None:
        metrics.paged_read_results += 1
        metrics.read_result_bytes_omitted += len(content.encode("utf-8")) - len(
            page.encode("utf-8")
        )
    return {
        **result,
        "content": page,
        "bytes": len(page.encode("utf-8")),
        "requested_bytes": result["bytes"],
        "start_line": first,
        "end_line": first + len(kept) - 1 if kept else None,
        "requested_end_line": last,
        "complete": False,
        "truncated": True,
        "next_start_line": first + len(kept),
        "oversized_line": len(lines[len(kept)]) > (maximum if line_limit is None else line_limit),
        "notice": (
            "Only the displayed lines were returned; do not infer facts from omitted text. "
            "Use workspace_search to locate relevant lines, or workspace_read with both "
            "start_line=next_start_line and a small end_line to continue. "
            "If one line alone exceeds the result limit, ask the user to increase "
            "tools.max_read_result_chars. SHA-256 still identifies the full file."
        ),
    }


def _paged_read(ctx: RunContext[RunDependencies], result: dict[str, Any]) -> dict[str, Any]:
    maximum = ctx.deps.resolved.config.tools.max_read_result_chars
    return _read_page(result, maximum) if maximum else result


def _paged_batch(ctx: RunContext[RunDependencies], result: dict[str, Any]) -> dict[str, Any]:
    maximum = ctx.deps.resolved.config.tools.max_read_result_chars
    if not maximum:
        return result
    remaining = maximum
    items = []
    for item in result["results"]:
        page = _read_page(item, remaining, line_limit=maximum)
        items.append(page)
        remaining -= len(page.get("content", ""))
    return {
        **result,
        "results": items,
        "complete": result["complete"] and all(item.get("complete", True) for item in items),
    }


def list_workspace(ctx: RunContext[RunDependencies], relative_path: str = ".") -> str:
    """List immediate workspace entries, honoring exclusions and entry limits."""
    return _result(lambda: _workspace(ctx).list(relative_path))


def read_workspace(
    ctx: RunContext[RunDependencies],
    relative_path: str,
    start_line: int | None = None,
    end_line: int | None = None,
) -> str:
    """Read UTF-8 text and the full file's SHA-256. Pass both 1-based inclusive line numbers
    to read only a relevant excerpt. Large results are paged by complete lines; when
    complete=false, inspect next_start_line or search for relevant lines before continuing.
    """
    return _result(
        lambda: _paged_read(ctx, _workspace(ctx).read(relative_path, start_line, end_line))
    )


def search_workspace(ctx: RunContext[RunDependencies], query: str, relative_path: str = ".") -> str:
    """Search text within a workspace directory or one known file. Results contain paths
    and line numbers for focused reads. Use a known file path to avoid unrelated sources.
    A truncated result is incomplete; narrow the search or read the relevant lines.
    """
    return _result(lambda: _workspace(ctx).search(query, relative_path))


def read_many_workspace(
    ctx: RunContext[RunDependencies],
    reads: Annotated[list[ReadRequest], Field(min_length=1, max_length=16)],
) -> str:
    """Read known files/excerpts together, within configured file and full-file byte limits.
    Results preserve input order and include each file's SHA-256. A result with complete=false
    has per-item errors or paged text: do not claim all sources were fully read. The model-facing
    text limit is shared across the batch; use each next_start_line for focused reads.
    """
    return _result(lambda: _paged_batch(ctx, _workspace(ctx).read_many(reads)))


def write_preview(
    ctx: RunContext[RunDependencies],
    relative_path: str,
    content: str,
    expected_sha256: str | None = None,
) -> str:
    """Validate proposed content and return the expected file fingerprint; changes no files.
    Depending on personal settings, returns full content/diff or summary metadata.
    Human approval always shows the exact proposed contents and diff.
    This preview does not request approval. To submit it for approval, call workspace_write
    with the same content and fingerprint; the harness pauses before any file change.
    """

    def preview() -> dict[str, Any]:
        result = _workspace(ctx).preview(relative_path, content, expected_sha256=expected_sha256)
        if ctx.deps.resolved.config.tools.preview_mode == "summary":
            result = {
                key: result[key]
                for key in (
                    "path",
                    "bytes",
                    "exists",
                    "expected_sha256",
                    "requires_confirmation",
                    "written",
                )
            }
            result["proposed_sha256"] = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return result

    return _result(preview)


def write_workspace(
    ctx: RunContext[RunDependencies],
    relative_path: str,
    content: str,
    expected_sha256: str | None,
) -> str:
    """Submit a proposed write for human approval; the harness pauses before executing it.
    Call this tool to request approval, rather than asking for approval in a final answer.
    Reuse the proposed content and preview fingerprint, or null for a new file.
    Use expected_sha256 from the preview, never proposed_sha256 (the proposed content hash).
    Only after approval does the tool write the file and return a write receipt.
    """
    if not ctx.tool_call_approved:
        raise ModelRetry("This write requires explicit human approval.")
    return _result(lambda: _workspace(ctx).write(relative_path, content, expected_sha256))


def validate_write_arguments(
    ctx: RunContext[RunDependencies],
    relative_path: str,
    content: str,
    expected_sha256: str | None,
) -> None:
    """Let the model correct an invalid proposal before presenting it for approval."""
    if ctx.tool_call_approved:
        return  # Agent.resume and the write itself recheck the approved snapshot.
    try:
        preview = _workspace(ctx).preview(relative_path, content, expected_sha256=expected_sha256)
        if preview["expected_sha256"] != expected_sha256:
            raise WriteFingerprintError(
                "Existing files require their current read/preview SHA-256."
            )
    except WriteFingerprintError as error:
        raise ModelRetry(
            f"Invalid write proposal: {error} "
            "Use expected_sha256 from a read/preview; use null for a new file. "
            "Never use proposed_sha256 as the expected file fingerprint."
        ) from error
    except WorkspaceError:
        # Keep permission/path failures terminal at the existing SDK approval boundary.
        return


def replace_workspace_text(
    ctx: RunContext[RunDependencies],
    relative_path: str,
    old_text: str,
    new_text: str,
    expected_sha256: str,
) -> str:
    """Replace one unique literal excerpt in an existing UTF-8 file after human approval.
    Read the file/excerpt first for its full SHA-256. Old text must match exactly once,
    including whitespace; use more surrounding text if ambiguous. No regex or replace-all.
    Unchanged text and line endings are preserved. New text may be empty to delete the excerpt.
    Human approval shows the full resulting content and diff; returns only a write receipt.
    """
    if not ctx.tool_call_approved:
        raise ModelRetry("This replacement requires explicit human approval.")
    return _result(
        lambda: _workspace(ctx).replace_text(relative_path, old_text, new_text, expected_sha256)
    )


def configured_tools(deps: RunDependencies) -> list[Tool[RunDependencies]]:
    definitions: list[Tool[RunDependencies]] = [
        Tool(describe_context, name=CONTEXT_TOOL, prepare=prepare_task_tool),
        Tool(list_workspace, name="workspace_list", prepare=prepare_task_tool),
        Tool(read_workspace, name="workspace_read", prepare=prepare_task_tool),
        Tool(read_many_workspace, name="workspace_read_many", prepare=prepare_task_tool),
        Tool(search_workspace, name="workspace_search", prepare=prepare_task_tool),
        Tool(write_preview, name="workspace_write_preview", prepare=prepare_task_tool),
        Tool(
            write_workspace,
            name="workspace_write",
            requires_approval=True,
            args_validator=validate_write_arguments,
            sequential=True,
            prepare=prepare_task_tool,
        ),
        Tool(
            replace_workspace_text,
            name="workspace_replace_text",
            requires_approval=True,
            sequential=True,
            prepare=prepare_task_tool,
        ),
        Tool(search_places, name="places_search", prepare=prepare_task_tool),
    ]
    allowed = set(deps.resolved.config.tools.allowed)
    return [tool for tool in definitions if TOOL_NAMES[tool.name] in allowed]
