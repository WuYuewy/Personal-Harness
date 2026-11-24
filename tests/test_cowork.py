import asyncio
from pathlib import Path

import pytest

from personal_harness import Agent, load_config
from personal_harness.cowork import cowork_model


@pytest.mark.parametrize(
    ("language", "style", "title", "has_count"),
    [
        ("zh-CN", "concise", "# 待办简报", False),
        ("en", "detailed", "# Action brief", True),
    ],
)
def test_same_workflow_uses_user_language_and_style(
    config_file: Path, language: str, style: str, title: str, has_count: bool
) -> None:
    (config_file.parent / "notes.md").write_text("TODO: My action\nOther line\n", "utf-8")
    resolved = load_config(
        config_file,
        overrides={
            "tools": {"allowed": ["workspace.read", "workspace.write_preview", "workspace.write"]},
            "preferences": {"language": language, "response_style": style},
        },
    )

    async def exercise() -> None:
        agent = Agent(resolved, model=cowork_model(resolved, "notes.md", "brief.md"))
        result = await agent.run("Create brief")
        assert result.status == "pending_approval"
        assert result.requests == 3
        assert result.tool_calls == ("workspace.read", "workspace.write_preview")
        content = result.approvals[0].preview["content"]
        assert content.startswith(title)
        assert ("Extracted items" in content) == has_count
        assert "Other line" not in content
        final = await agent.resume(result.run_id, {result.approvals[0].call_id: True})
        assert final.requests == 4
        assert (config_file.parent / "brief.md").read_text("utf-8") == content

    asyncio.run(exercise())
