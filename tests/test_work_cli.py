import json
import signal
from types import SimpleNamespace

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel
from typer.testing import CliRunner

from personal_harness import Agent as RealAgent
from personal_harness.cli import app

runner = CliRunner()


@pytest.fixture
def working(config_file, monkeypatch):
    config_file.write_text(
        config_file.read_text("utf-8")
        .replace(
            'allowed = ["context.describe"]', 'allowed = ["workspace.read", "workspace.write"]'
        )
        .replace("read_cache_bytes = 0", "read_cache_bytes = 1000"),
        "utf-8",
    )
    (config_file.parent / "notes.md").write_text("Friday review", "utf-8")
    records = []

    def create(resolved):
        record = SimpleNamespace(resolved=resolved, responses=[], results=[], closed=0, tasks=[])

        def respond(messages, info):
            record.responses.append(list(messages))
            returned = [
                part
                for message in messages
                if isinstance(message, ModelRequest)
                for part in message.parts
                if isinstance(part, ToolReturnPart)
            ]
            if not returned:
                available = {tool.name for tool in info.function_tools}
                if record.tasks[-1].startswith("write") and "workspace_write" in available:
                    return ModelResponse(
                        [
                            ToolCallPart(
                                "workspace_write",
                                {
                                    "relative_path": "brief.md",
                                    "content": "Approved brief",
                                    "expected_sha256": None,
                                },
                            )
                        ]
                    )
                if "workspace_read" in available:
                    return ModelResponse(
                        [ToolCallPart("workspace_read", {"relative_path": "notes.md"})]
                    )
            text = "Task complete"
            if returned and returned[0].tool_name == "workspace_read":
                text = json.loads(returned[0].content)["content"]
            return ModelResponse([TextPart(text)])

        agent = RealAgent(resolved, model=FunctionModel(respond))
        original_run, original_resume, original_close = agent.run, agent.resume, agent.aclose

        async def run(task, **kwargs):
            record.tasks.append(task)
            result = await original_run(task, **kwargs)
            record.results.append(result)
            return result

        async def resume(*args, **kwargs):
            result = await original_resume(*args, **kwargs)
            record.results.append(result)
            return result

        async def close():
            record.closed += 1
            await original_close()

        agent.run, agent.resume, agent.aclose = run, resume, close
        record.agent = agent
        records.append(record)
        return agent

    monkeypatch.setattr("personal_harness.cli.Agent", create)
    return SimpleNamespace(config=config_file, records=records)


def invoke(working, text="", *args):
    return runner.invoke(app, ["work", "-c", str(working.config), *args], input=text)


def test_repeated_tasks_reuse_agent_cache_and_start_fresh_history(working):
    result = invoke(working, "read notes\nread notes again\n:quit\n")
    assert result.exit_code == 0, result.output
    assert len(working.records) == 1
    record = working.records[0]
    assert record.closed == 1 and len(record.results) == 2
    assert record.results[0].read_cache_hits == 0 and record.results[1].read_cache_hits == 1
    assert record.results[1].read_bytes == 0
    assert len(record.responses[0]) == len(record.responses[2])
    assert result.stdout.count("Cost: USD 0") == result.stdout.count("Run: ") == 2


@pytest.mark.parametrize("decision", ["y", "n"])
def test_work_uses_real_approval_and_keeps_source_cache_warm(working, decision):
    result = invoke(working, f"read notes\nwrite brief\n{decision}\nread notes again\n:quit\n")
    assert result.exit_code == 0, result.output
    assert "Apply this exact write?" in result.stdout
    assert (working.config.parent / "brief.md").exists() is (decision == "y")
    assert len(working.records) == 1 and working.records[0].closed == 1
    assert working.records[0].results[-1].read_cache_hits == 1
    assert working.records[0].tasks == ["read notes", "write brief", "read notes again"]


@pytest.mark.parametrize("text", ["", "\n\n:quit\n"])
def test_exit_without_task_does_not_create_agent(working, text):
    result = invoke(working, text)
    assert result.exit_code == 0 and working.records == []


def test_eof_after_task_closes_agent(working):
    result = invoke(working, "read notes\n")
    assert result.exit_code == 0 and working.records[0].closed == 1


def test_abandoned_approval_closes_agent_and_does_not_write(working):
    result = invoke(working, "write brief\n")
    assert result.exit_code == 1
    assert "Apply this exact write?" in result.stdout
    assert working.records[0].closed == 1 and working.records[0].agent._pending == {}
    assert not (working.config.parent / "brief.md").exists()


def feed(monkeypatch, values):
    iterator = iter(values)

    def read(prompt):
        value = next(iterator)
        return value() if callable(value) else value

    monkeypatch.setattr("builtins.input", read)


def test_effective_config_change_replaces_agent_but_comments_preserve_it(working, monkeypatch):
    def comment():
        working.config.write_text(working.config.read_text("utf-8") + "\n# user comment\n", "utf-8")
        return "read unchanged"

    def change():
        working.config.write_text(
            working.config.read_text("utf-8").replace(
                'response_style = "detailed"', 'response_style = "concise"'
            ),
            "utf-8",
        )
        return "read changed"

    feed(monkeypatch, ["read first", comment, change, ":quit"])
    result = invoke(working)
    assert result.exit_code == 0, result.output
    assert len(working.records) == 2 and all(record.closed == 1 for record in working.records)
    assert len(working.records[0].results) == 2
    assert working.records[0].results[1].read_cache_hits == 1
    assert working.records[1].results[0].read_cache_hits == 0
    assert working.records[1].resolved.config.preferences.response_style == "concise"


@pytest.mark.parametrize("tools", [[], ["--tool", "workspace.read"]])
def test_config_added_permission_does_not_expand_initial_tools(working, monkeypatch, tools):
    working.config.write_text(
        working.config.read_text("utf-8").replace(
            'allowed = ["workspace.read", "workspace.write"]', 'allowed = ["workspace.read"]'
        ),
        "utf-8",
    )

    def expand():
        working.config.write_text(
            working.config.read_text("utf-8").replace(
                'allowed = ["workspace.read"]', 'allowed = ["workspace.read", "workspace.write"]'
            ),
            "utf-8",
        )
        return "write must remain read-only"

    feed(monkeypatch, ["read first", expand, ":quit"])
    result = invoke(working, "", *tools)
    assert result.exit_code == 0, result.output
    assert len(working.records) == 2
    assert working.records[1].results[0].available_tools == ("workspace.read",)
    assert not (working.config.parent / "brief.md").exists()


def test_revoked_tool_stops_before_replacing_agent_or_sending_task(working, monkeypatch):
    def revoke():
        working.config.write_text(
            working.config.read_text("utf-8").replace(
                'allowed = ["workspace.read", "workspace.write"]', "allowed = []"
            ),
            "utf-8",
        )
        return "read revoked"

    feed(monkeypatch, ["read first", revoke])
    result = invoke(working, "", "--tool", "workspace.read")
    assert result.exit_code == 2 and "Tool selection error" in result.stderr
    assert len(working.records) == 1 and working.records[0].closed == 1
    assert working.records[0].tasks == ["read first"]


def test_invalid_reloaded_config_closes_agent(working, monkeypatch):
    def corrupt():
        working.config.write_text("invalid = [", "utf-8")
        return "read next"

    feed(monkeypatch, ["read first", corrupt])
    result = invoke(working)
    assert result.exit_code == 2 and "Configuration error" in result.stderr
    assert len(working.records) == 1 and working.records[0].closed == 1


def test_initial_profile_is_preserved_when_default_changes(working, monkeypatch):
    def switch_default():
        working.config.write_text(
            working.config.read_text("utf-8").replace(
                'default_profile = "desktop"', 'default_profile = "mobile"'
            ),
            "utf-8",
        )
        return "read still desktop"

    feed(monkeypatch, ["read first", switch_default, ":quit"])
    result = invoke(working)
    assert result.exit_code == 0 and len(working.records) == 1
    assert all(run.profile == "desktop" for run in working.records[0].results)


def test_config_change_during_approval_applies_to_next_task(working, monkeypatch):
    def approve(*args, **kwargs):
        working.config.write_text(
            working.config.read_text("utf-8").replace(
                'allowed = ["workspace.read", "workspace.write"]', "allowed = []"
            ),
            "utf-8",
        )
        return True

    monkeypatch.setattr("personal_harness.cli.typer.confirm", approve)
    result = invoke(working, "write brief\nwrite next\n")
    assert result.exit_code == 2 and "Tool selection error" in result.stderr
    assert (working.config.parent / "brief.md").read_text("utf-8") == "Approved brief"
    assert len(working.records) == 1 and working.records[0].tasks == ["write brief"]
    assert working.records[0].closed == 1


def test_reloaded_output_directory_is_enforced_before_cli_approval(working, monkeypatch):
    (working.config.parent / "out").mkdir()

    def restrict():
        working.config.write_text(
            working.config.read_text("utf-8").replace(
                'root = "."', 'root = "."\nwrite_directory = "out"'
            ),
            "utf-8",
        )
        return "write outside out"

    feed(monkeypatch, ["read first", restrict])
    result = invoke(working)
    assert result.exit_code == 1 and "write directory" in result.stderr
    assert "Apply this exact write?" not in result.stdout
    assert len(working.records) == 2 and all(record.closed == 1 for record in working.records)
    assert not (working.config.parent / "brief.md").exists()


def test_keyboard_interrupt_between_tasks_closes_agent(working, monkeypatch):
    def interrupt():
        raise KeyboardInterrupt

    feed(monkeypatch, ["read first", interrupt])
    result = invoke(working)
    assert result.exit_code == 130 and "Work interrupted" in result.stderr
    assert working.records[0].closed == 1


def test_sigint_during_task_input_interrupts_immediately_and_restores_handler(working, monkeypatch):
    previous = signal.getsignal(signal.SIGINT)

    def interrupt():
        signal.raise_signal(signal.SIGINT)
        pytest.fail("SIGINT must interrupt terminal input immediately")

    feed(monkeypatch, ["read first", interrupt])
    result = invoke(working)
    assert result.exit_code == 130, result.output
    assert working.records[0].closed == 1
    assert signal.getsignal(signal.SIGINT) == previous


def test_sigint_during_approval_abandons_write_and_restores_handler(working, monkeypatch):
    previous = signal.getsignal(signal.SIGINT)

    def interrupt(prompt):
        if prompt.startswith("Apply this exact write?"):
            signal.raise_signal(signal.SIGINT)
            pytest.fail("SIGINT must interrupt approval input immediately")
        return "write brief"

    monkeypatch.setattr("builtins.input", interrupt)
    result = invoke(working)
    assert result.exit_code == 1, result.output
    assert not (working.config.parent / "brief.md").exists()
    assert working.records[0].closed == 1 and working.records[0].agent._pending == {}
    assert signal.getsignal(signal.SIGINT) == previous


@pytest.mark.parametrize(
    "args", [["--no-tools"], ["--tool", "workspace.read", "--tool", "workspace.read"]]
)
def test_tool_subset_and_empty_choice_are_preserved(working, args):
    result = invoke(working, "read first\nread second\n:quit\n", *args)
    assert result.exit_code == 0, result.output
    expected = () if args == ["--no-tools"] else ("workspace.read",)
    assert all(run.available_tools == expected for run in working.records[0].results)


@pytest.mark.parametrize(
    "args", [["--no-tools", "--tool", "workspace.read"], ["--tool", "workspace.search"]]
)
def test_invalid_tool_options_never_create_agent(working, args):
    result = invoke(working, "", *args)
    assert result.exit_code == 2 and working.records == []
