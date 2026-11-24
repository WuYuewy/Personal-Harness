"""Check a wheel's CLI and SDK in a fresh directory, without model requests."""

from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import sys
from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel

import personal_harness
from personal_harness import Agent


def cli(root: Path, *args: str) -> str:
    run = subprocess.run(
        [sys.executable, "-m", "personal_harness", *args],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return run.stdout


async def exercise(config: Path) -> None:
    def respond(messages, info):
        returned = [
            p
            for m in messages
            if isinstance(m, ModelRequest)
            for p in m.parts
            if isinstance(p, ToolReturnPart)
        ]
        if returned:
            return ModelResponse([TextPart("Completed from notes.md")])
        return ModelResponse(
            [
                ToolCallPart(
                    "workspace_write",
                    dict(
                        relative_path="brief.md", content="Synthetic brief\n", expected_sha256=None
                    ),
                )
            ]
        )

    async with Agent.from_config(
        config, overrides={"tools": {"allowed": ["workspace.write"]}}, model=FunctionModel(respond)
    ) as agent:
        result = await agent.run("Create a brief")
        assert result.status == "pending_approval"
        assert not (config.parent / "brief.md").exists()
        approved = await agent.resume(result.run_id, {a.call_id: True for a in result.approvals})
        assert approved.status == "completed"
        assert (config.parent / "brief.md").read_text("utf-8") == "Synthetic brief\n"
        assert approved.cost_usd == 0


def main() -> None:
    assert "site-packages" in Path(personal_harness.__file__).resolve().parts
    assert files("personal_harness").joinpath("ui/ModelConnections.jsx").is_file()
    with TemporaryDirectory(prefix="personal-harness-wheel-") as temporary:
        root = Path(temporary)
        cli(root, "init")
        config = root / "personal-harness.toml"
        before = hashlib.sha256(config.read_bytes()).hexdigest()
        assert "Valid schema" in cli(root, "config", "validate")
        record = json.loads(cli(root, "run", "Inspect settings", "--json"))
        assert record["provider"] == "fake" and record["cost_usd"] == "0"
        assert hashlib.sha256(config.read_bytes()).hexdigest() == before
        assert not (root / ".personal-harness/ledger.sqlite3").exists()
        asyncio.run(exercise(config))
    print("Installed CLI, template, zero-network run and SDK approval smoke passed.")


if __name__ == "__main__":
    main()
