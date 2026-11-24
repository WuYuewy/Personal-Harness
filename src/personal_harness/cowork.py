"""Explicit offline file workflow to exercise the real SDK/tool/approval path."""

import json
from typing import Any

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from personal_harness.config import ConfigError, ResolvedConfig

REQUIRED_TOOLS = {"workspace.read", "workspace.write_preview", "workspace.write"}


def cowork_model(resolved: ResolvedConfig, source: str, target: str) -> FunctionModel:
    """Extract literal TODO lines; this is a scripted demo, not model reasoning."""
    missing = REQUIRED_TOOLS - set(resolved.config.tools.allowed)
    if missing:
        raise ConfigError(f"This workflow requires configured tools: {', '.join(sorted(missing))}")

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        returns = {
            part.tool_name: part
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        }
        written = returns.get("workspace_write")
        if written is not None:
            outcome = "written" if written.outcome == "success" else "declined or failed"
            return ModelResponse([TextPart(f"[OFFLINE WORKFLOW] {target}: {outcome}.")])
        read = returns.get("workspace_read")
        if read is None:
            return ModelResponse([ToolCallPart("workspace_read", {"relative_path": source})])
        if read.outcome != "success":
            return ModelResponse([TextPart("The source could not be read; no output was written.")])
        source_data: dict[str, Any] = json.loads(str(read.content))
        todos = [
            line.strip() for line in source_data["content"].splitlines() if "TODO" in line.upper()
        ]
        preferences = resolved.config.preferences
        chinese = preferences.language.casefold().startswith("zh")
        title = "待办简报" if chinese else "Action brief"
        source_label = "来源" if chinese else "Source"
        lines = [f"# {title}", "", f"{source_label}: {source}", ""]
        lines.extend(f"- {line}" for line in todos)
        if not todos:
            lines.append("未找到 TODO 行。" if chinese else "No TODO lines found.")
        if preferences.response_style == "detailed":
            label = "提取条数" if chinese else "Extracted items"
            lines.extend(["", f"{label}: {len(todos)}"])
        content = "\n".join(lines) + "\n"
        preview = returns.get("workspace_write_preview")
        if preview is None:
            return ModelResponse(
                [
                    ToolCallPart(
                        "workspace_write_preview",
                        {
                            "relative_path": target,
                            "content": content,
                        },
                    )
                ]
            )
        if preview.outcome != "success":
            return ModelResponse([TextPart("The write preview failed; no output was written.")])
        proposal: dict[str, Any] = json.loads(str(preview.content))
        return ModelResponse(
            [
                ToolCallPart(
                    "workspace_write",
                    {
                        "relative_path": target,
                        "content": content,
                        "expected_sha256": proposal["expected_sha256"],
                    },
                )
            ]
        )

    return FunctionModel(respond, model_name="offline-cowork")
