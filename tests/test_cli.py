import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from personal_harness.cli import app
from tests import mock_provider

runner = CliRunner()


def test_cli_task_tools_and_no_tools_are_explicit(config_file: Path) -> None:
    for options, expected_tools, requests in (
        ([], ["context.describe"], 2),
        (["--tool", "context.describe"], ["context.describe"], 2),
        (["--no-tools"], [], 1),
    ):
        result = runner.invoke(app, ["run", "Inspect", "-c", str(config_file), "--json", *options])
        assert result.exit_code == 0, result.output
        data = json.loads(result.stdout)
        assert data["available_tools"] == expected_tools
        assert data["usage"]["requests"] == requests
    denied = runner.invoke(
        app, ["run", "Inspect", "-c", str(config_file), "--tool", "workspace.write"]
    )
    assert denied.exit_code == 1
    assert denied.stdout == "" and "Task stopped (tool_selection)" in denied.stderr
    structured = runner.invoke(
        app, ["run", "Inspect", "-c", str(config_file), "--tool", "workspace.write", "--json"]
    )
    assert structured.exit_code == 1
    assert json.loads(structured.stdout)["reason"] == "tool_selection"
    invalid = runner.invoke(
        app, ["run", "Inspect", "-c", str(config_file), "--no-tools", "--tool", "context.describe"]
    )
    assert invalid.exit_code == 2


def test_provider_prices_are_available_without_network(config_file):
    result = runner.invoke(app, ["provider", "prices", "-c", str(config_file)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {"provider": "fake", "paid_execution": False}
    config_file.write_text(mock_provider.template(), "utf-8")
    result = runner.invoke(app, ["provider", "prices", "-c", str(config_file)])
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["prices"]["currency"] == "CNY"
    assert data["prices"]["input_per_million"] == "1"
    assert data["context_window"] == 262144


def test_online_template_and_diagnostics_do_not_send_requests(tmp_path: Path) -> None:
    path = tmp_path / "personal.toml"
    assert (
        runner.invoke(app, ["init", "--provider", "openai-compatible", "-c", str(path)]).exit_code
        == 0
    )
    assert runner.invoke(app, ["config", "validate", "-c", str(path)]).exit_code == 0
    doctor = runner.invoke(app, ["doctor", "-c", str(path)])
    assert doctor.exit_code == 0
    assert json.loads(doctor.stdout)["provider"] == {
        "name": "openai-compatible",
        "network_required": True,
    }
    assert not (tmp_path / ".personal-harness").exists()
    assert runner.invoke(app, ["demo", "Hello", "-c", str(path)]).exit_code == 2
    assert runner.invoke(app, ["ledger", "list", "-c", str(path)]).exit_code == 2


def test_init_edit_inspect_and_demo(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "personal.toml"
    assert runner.invoke(app, ["init", "--config", str(path)]).exit_code == 0
    original = path.read_text("utf-8")
    path.write_text(original.replace('display_name = "You"', 'display_name = "我的名字"'), "utf-8")
    show = runner.invoke(app, ["config", "show", "-c", str(path), "-p", "mobile", "--explain"])
    assert show.exit_code == 0, show.output
    data = json.loads(show.stdout)
    assert data["config"]["user"]["display_name"] == "我的名字"
    assert data["sources"]["model.max_output_tokens"] == "profile:mobile"
    result = runner.invoke(
        app, ["demo", "安排一天的工作", "-c", str(path), "-p", "mobile", "--json"]
    )
    assert result.exit_code == 0, result.output
    run = json.loads(result.stdout)
    assert run["context"]["user"]["display_name"] == "我的名字"
    assert run["cost_usd"] == "0"
    assert run["profile"] == "mobile"


def test_init_never_overwrites_existing_file(config_file: Path) -> None:
    original = config_file.read_bytes()
    result = runner.invoke(app, ["init", "-c", str(config_file)])
    assert result.exit_code == 2
    assert "already exists" in result.output
    assert config_file.read_bytes() == original


def test_environment_selection_and_explicit_path_precedence(
    config_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PERSONAL_HARNESS_CONFIG", str(tmp_path / "absent.toml"))
    assert runner.invoke(app, ["config", "validate"]).exit_code == 2
    assert runner.invoke(app, ["config", "validate", "-c", str(config_file)]).exit_code == 0
    monkeypatch.setenv("PERSONAL_HARNESS_CONFIG", str(config_file))
    assert runner.invoke(app, ["config", "validate"]).exit_code == 0


def test_doctor_identifies_missing_workspace(config_file: Path) -> None:
    healthy = runner.invoke(app, ["doctor", "-c", str(config_file)])
    assert healthy.exit_code == 0
    assert json.loads(healthy.stdout)["ok"] is True
    config_file.write_text(
        config_file.read_text("utf-8").replace('root = "."', 'root = "absent"'), "utf-8"
    )
    unhealthy = runner.invoke(app, ["doctor", "-c", str(config_file)])
    assert unhealthy.exit_code == 1
    assert json.loads(unhealthy.stdout)["workspace"]["is_directory"] is False


def test_configuration_error_is_actionable_without_traceback(tmp_path: Path) -> None:
    result = runner.invoke(app, ["demo", "Hello", "-c", str(tmp_path / "absent.toml")])
    assert result.exit_code == 2
    assert "Configuration error" in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize("decision", ["y", "n"])
def test_workspace_workflow_cli_requests_confirmation_before_writing(
    config_file: Path, decision: str
) -> None:
    config_file.write_text(
        config_file.read_text("utf-8")
        + (
            "\n[profiles.cowork.tools]\n"
            'allowed = ["workspace.read", "workspace.write_preview", "workspace.write"]\n'
        ),
        "utf-8",
    )
    (config_file.parent / "notes.md").write_text("TODO: personalize tool limits\n", "utf-8")
    result = runner.invoke(
        app,
        ["workspace-demo", "notes.md", "brief.md", "-c", str(config_file), "-p", "cowork"],
        input=decision + "\n",
    )
    assert result.exit_code == 0, result.output
    assert "Apply this exact write?" in result.output
    assert "personalize tool limits" in result.output
    target = config_file.parent / "brief.md"
    if decision == "y":
        assert "personalize tool limits" in target.read_text("utf-8")
        assert "提取条数" not in target.read_text("utf-8")  # New profile inherits base concise.
    else:
        assert not target.exists()


def test_workspace_workflow_cli_cannot_expand_user_allowlist(config_file: Path) -> None:
    result = runner.invoke(app, ["workspace-demo", "notes.md", "brief.md", "-c", str(config_file)])
    assert result.exit_code == 2
    assert "requires configured tools" in result.output
    assert not (config_file.parent / "brief.md").exists()
