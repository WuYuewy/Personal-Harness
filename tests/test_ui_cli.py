"""Installed UI launch contract without starting a server or importing Chainlit."""

import os
import subprocess
import sys
from importlib.resources import files
from pathlib import Path

import pytest
from click import unstyle
from typer.testing import CliRunner

from personal_harness import cli

runner = CliRunner()


def test_ui_uses_bundled_app_and_config_location(tmp_path, monkeypatch):
    personal = tmp_path / "个人资料 & workspace"
    personal.mkdir()
    config = personal / "settings.toml"
    config.write_text(
        files("personal_harness").joinpath("templates/personal.toml").read_text("utf-8"),
        "utf-8",
    )
    invocation = tmp_path / "elsewhere"
    invocation.mkdir()
    monkeypatch.chdir(invocation)
    monkeypatch.setenv("PERSONAL_HARNESS_CONFIG", "does-not-exist.toml")
    monkeypatch.setenv("CHAINLIT_APP_ROOT", str(invocation))
    monkeypatch.setattr(cli, "find_spec", lambda name: object())
    before = dict(os.environ)
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        assert Path(command[4]).is_file()
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(cli.subprocess, "run", run)
    result = runner.invoke(
        cli.app, ["ui", "-c", str(config), "--host", "127.0.0.2", "--port", "8123"]
    )
    assert result.exit_code == 0, result.output
    command, kwargs = calls[0]
    assert command == [
        sys.executable,
        "-m",
        "chainlit",
        "run",
        str(files("personal_harness").joinpath("chainlit_app.py")),
        "--headless",
        "--host",
        "127.0.0.2",
        "--port",
        "8123",
    ]
    assert kwargs["cwd"] == personal.resolve()
    assert kwargs["env"]["PERSONAL_HARNESS_CONFIG"] == str(config.resolve())
    assert kwargs["env"]["CHAINLIT_APP_ROOT"] == str(personal.resolve())
    assert kwargs["check"] is False
    assert dict(os.environ) == before
    assert not list(invocation.iterdir())
    assert (personal / ".personal-harness/ui.sqlite3").is_file()
    assert (personal / "public/elements/ModelConnections.jsx").is_file()
    assert (personal / "public/elements/UserInsights.jsx").is_file()
    assert (personal / "public/vendor/chart.umd.min.js").is_file()
    assert (personal / "public/vendor/Chart.js.LICENSE.md").is_file()
    assert kwargs["env"]["PERSONAL_HARNESS_UI_PERSIST"] == "1"
    assert kwargs["env"]["CHAINLIT_AUTH_SECRET"]


def test_ui_missing_extra_is_actionable_without_starting_server(config_file, monkeypatch):
    monkeypatch.setattr(cli, "find_spec", lambda name: None)
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **kw: pytest.fail("server started"))
    result = runner.invoke(cli.app, ["ui", "-c", str(config_file)])
    assert result.exit_code == 2
    assert "[ui]" in result.stderr and "uv sync --extra ui" in result.stderr
    assert not (config_file.parent / ".chainlit").exists()


def test_ui_checks_config_before_optional_import_or_subprocess(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "find_spec", lambda name: pytest.fail("checked extra"))
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **kw: pytest.fail("server started"))
    result = runner.invoke(cli.app, ["ui", "-c", str(tmp_path / "absent.toml")])
    assert result.exit_code == 2 and "Configuration error" in result.stderr


@pytest.mark.parametrize("port", ["0", "65536", "invalid"])
def test_ui_rejects_invalid_ports_without_starting_server(config_file, monkeypatch, port):
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **kw: pytest.fail("server started"))
    result = runner.invoke(cli.app, ["ui", "-c", str(config_file), "--port", port])
    assert result.exit_code == 2


@pytest.mark.parametrize("returncode, expected", [(0, 0), (1, 1), (2, 2), (-15, 143)])
def test_ui_default_options_environment_config_and_server_exit(
    config_file, monkeypatch, returncode, expected
):
    monkeypatch.setenv("PERSONAL_HARNESS_CONFIG", str(config_file))
    monkeypatch.setattr(cli, "find_spec", lambda name: object())

    def run(command, **kwargs):
        assert command[-5:] == ["--headless", "--host", "127.0.0.1", "--port", "8000"]
        assert kwargs["env"]["PERSONAL_HARNESS_CONFIG"] == str(config_file)
        return subprocess.CompletedProcess(command, returncode)

    monkeypatch.setattr(cli.subprocess, "run", run)
    result = runner.invoke(cli.app, ["ui"])
    assert result.exit_code == expected, result.output


@pytest.mark.parametrize(
    "error, expected", [(OSError(13, "Denied"), 2), (KeyboardInterrupt(), 130)]
)
def test_ui_launch_error_and_interrupt(config_file, monkeypatch, error, expected):
    monkeypatch.setattr(cli, "find_spec", lambda name: object())

    def run(*args, **kwargs):
        raise error

    monkeypatch.setattr(cli.subprocess, "run", run)
    result = runner.invoke(cli.app, ["ui", "-c", str(config_file)])
    assert result.exit_code == expected
    if expected == 2:
        assert "Cannot start UI: Denied" in result.stderr


def test_ui_help_needs_no_config_or_chainlit_import(monkeypatch):
    monkeypatch.setattr(cli, "find_spec", lambda name: pytest.fail("checked extra"))
    result = runner.invoke(cli.app, ["ui", "--help"])
    output = unstyle(result.stdout)
    assert result.exit_code == 0 and "--config" in output and "--port" in output
