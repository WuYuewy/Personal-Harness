"""CLI for personal tasks, configuration, costs and offline rehearsals."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sqlite3
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stdout
from decimal import Decimal, InvalidOperation
from importlib.resources import as_file, files
from importlib.util import find_spec
from pathlib import Path
from threading import current_thread, main_thread
from typing import Annotated, Any, Literal

import typer

from personal_harness import __version__
from personal_harness.config import ConfigError, ResolvedConfig, load_config
from personal_harness.cowork import cowork_model
from personal_harness.display import budget_receipt, ledger_cost_receipt, run_cost_receipt
from personal_harness.ledger import Ledger
from personal_harness.runtime import Agent, RunError, RunResult
from personal_harness.tools import select_task_tools
from personal_harness.workspace import WorkspaceError, WorkspaceFiles

app = typer.Typer(no_args_is_help=True, help="A personal harness shaped by your configuration.")
config_app = typer.Typer(no_args_is_help=True, help="Validate and inspect personal configuration.")
app.add_typer(config_app, name="config")
provider_app = typer.Typer(
    no_args_is_help=True, help="Check optional providers and recorded prices."
)
app.add_typer(provider_app, name="provider")
ledger_app = typer.Typer(
    no_args_is_help=True, help="Inspect and reconcile local CNY request costs."
)
app.add_typer(ledger_app, name="ledger")
data_app = typer.Typer(no_args_is_help=True, help="Manage local application data.")
app.add_typer(data_app, name="data")

ConfigPath = Annotated[Path | None, typer.Option("--config", "-c", help="Personal TOML file.")]
Profile = Annotated[str | None, typer.Option("--profile", "-p", help="Select a named profile.")]


def _show_version(value: bool) -> None:
    if value:
        typer.echo(f"personal-harness {__version__}")
        raise typer.Exit()


@app.callback()
def configure_output(
    version: Annotated[
        bool, typer.Option("--version", callback=_show_version, is_eager=True, help="Show version.")
    ] = False,
) -> None:
    """Use UTF-8 for pipes and files, including on Windows legacy code pages."""
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if not stream.isatty() and reconfigure is not None:
            reconfigure(encoding="utf-8")


def _json(value: Any) -> None:
    typer.echo(json.dumps(value, ensure_ascii=False, indent=2))


@provider_app.command("prices")
def provider_prices(config: ConfigPath = None, profile: Profile = None) -> None:
    """Show your configured prices; no network requests."""
    from personal_harness.connections import model_prices

    resolved = _load(config, profile)
    if resolved.config.model.provider == "fake":
        _json({"provider": "fake", "paid_execution": False})
    else:
        prices, context_window = model_prices(resolved.config.model)
        _json({"prices": prices.model_dump(mode="json"), "context_window": context_window})


def _config_path(path: Path | None) -> Path:
    if path is not None:
        return path
    return Path(os.environ.get("PERSONAL_HARNESS_CONFIG", "personal-harness.toml"))


def _load(path: Path | None, profile: str | None) -> ResolvedConfig:
    try:
        return load_config(_config_path(path), profile=profile)
    except (ConfigError, OSError, ValueError) as error:
        typer.echo(f"Configuration error: {error}", err=True)
        raise typer.Exit(2) from None


@app.command()
def init(
    config: ConfigPath = None,
    provider: Annotated[Literal["fake", "openai-compatible"], typer.Option("--provider")] = "fake",
) -> None:
    """Create an editable personal config. Existing files are never overwritten."""
    path = _config_path(config).expanduser()
    name = "openai-compatible.toml" if provider == "openai-compatible" else "personal.toml"
    template = files("personal_harness").joinpath(f"templates/{name}").read_text("utf-8")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="\n") as file:
            file.write(template)
    except FileExistsError:
        typer.echo(f"Config already exists: {path}. Edit it or choose another --config.", err=True)
        raise typer.Exit(2) from None
    except OSError as error:
        typer.echo(f"Cannot create config: {error.strerror}", err=True)
        raise typer.Exit(2) from None
    typer.echo(
        f"Created {path.resolve()}. Edit [base.user] and [base.preferences] to make it yours."
    )


@config_app.command("validate")
def validate(config: ConfigPath = None, profile: Profile = None) -> None:
    """Validate the base and every profile, including inactive profiles."""
    resolved = _load(config, profile)
    typer.echo(f"Valid schema v1; selected profile: {resolved.profile}")


@app.command("ui")
def launch_ui(
    config: ConfigPath = None,
    host: Annotated[str, typer.Option("--host", help="Interface for the local UI.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", min=1, max=65535)] = 8000,
) -> None:
    """Start the optional Chainlit UI using your installed package and personal config."""
    resolved = _load(config, None)
    if find_spec("chainlit") is None:
        typer.echo(
            "The UI needs the 'ui' extra. Install your wheel with [ui], "
            "or run uv sync --extra ui in the source checkout.",
            err=True,
        )
        raise typer.Exit(2)
    environment = dict(os.environ)
    environment["PERSONAL_HARNESS_CONFIG"] = str(resolved.source_file)
    # Keep Chainlit's generated config/uploads beside the personal config, not in the package.
    environment["CHAINLIT_APP_ROOT"] = str(resolved.source_file.parent)
    try:
        from personal_harness.ui_bootstrap import prepare_ui

        environment.update(prepare_ui(resolved.source_file.parent, host))
        typer.echo("Personal Harness UI: 用户名 local；首次登录设置至少 8 位密码。")
        with as_file(files("personal_harness").joinpath("chainlit_app.py")) as target:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "chainlit",
                    "run",
                    str(target),
                    "--headless",
                    "--host",
                    host,
                    "--port",
                    str(port),
                ],
                cwd=resolved.source_file.parent,
                env=environment,
                check=False,
            )
    except KeyboardInterrupt:
        raise typer.Exit(130) from None
    except (OSError, ValueError) as error:
        detail = error.strerror if isinstance(error, OSError) else str(error)
        typer.echo(f"Cannot start UI: {detail}", err=True)
        raise typer.Exit(2) from None
    if completed.returncode:
        code = completed.returncode
        raise typer.Exit(code if code > 0 else 128 - code)


@config_app.command("show")
def show(
    config: ConfigPath = None,
    profile: Profile = None,
    explain: Annotated[bool, typer.Option("--explain", help="Show each setting's source.")] = False,
) -> None:
    """Show the effective configuration after applying all layers."""
    resolved = _load(config, profile)
    result = resolved.as_dict()
    if explain:
        result["sources"] = dict(resolved.sources)
    _json(result)


@app.command()
def doctor(config: ConfigPath = None, profile: Profile = None) -> None:
    """Check configuration and workspace without making a model request."""
    resolved = _load(config, profile)
    root = Path(resolved.config.workspace.root)
    workspace_ok = root.is_dir()
    workspace: dict[str, Any] = {"path": str(root), "is_directory": workspace_ok}
    places_ok = True
    if "places.search" in resolved.config.tools.allowed:
        settings = resolved.config.local_life
        data_path = Path(settings.data_file) if settings.data_file is not None else None
        places_available = False
        data_city_matches = False
        places_stale: bool | None = None
        if data_path is not None:
            from datetime import UTC, datetime, timedelta

            from personal_harness.places import _load_places_data

            try:
                if not data_path.is_symlink():
                    data = _load_places_data(str(data_path))
                    places_available = data.fetched_at.utcoffset() is not None
                    data_city_matches = data.city == resolved.config.preferences.city
                    if places_available:
                        age = datetime.now(UTC) - data.fetched_at.astimezone(UTC)
                        places_stale = age < timedelta(0) or (
                            age.total_seconds() > settings.max_data_age_hours * 3600
                        )
            except (OSError, UnicodeError, ValueError):
                pass
        places_configured = (
            resolved.config.preferences.city is not None
            and settings.latitude is not None
            and settings.longitude is not None
        )
        places_ok = places_available and places_configured and data_city_matches
        workspace["local_places"] = {
            "configured": places_configured,
            "available": places_available,
            "city_matches": data_city_matches,
            "stale": places_stale,
            "required": True,
            "source": "local file only",
        }
    directory = resolved.config.workspace.write_directory
    write_ok = True
    if directory is not None:
        required = bool(
            {"workspace.write", "workspace.write_preview", "workspace.replace_text"}.intersection(
                resolved.config.tools.allowed
            )
        )
        try:
            available = WorkspaceFiles(resolved.config.workspace).path(directory).is_dir()
        except WorkspaceError:
            available = False
        workspace["write_directory"] = {
            "path": directory,
            "available": available,
            "required": required,
        }
        write_ok = available or not required
    _json(
        {
            "ok": workspace_ok and write_ok and places_ok,
            "profile": resolved.profile,
            "config": str(resolved.source_file),
            "workspace": workspace,
            "provider": {
                "name": resolved.config.model.provider,
                "network_required": resolved.config.model.provider != "fake",
            },
            "execution": "offline configuration rehearsal only"
            if resolved.config.model.provider == "fake"
            else "metered configured model; no request sent",
        }
    )
    if not workspace_ok or not write_ok or not places_ok:
        raise typer.Exit(1)


@data_app.command("backup")
def data_backup(
    destination: Annotated[Path, typer.Argument(help="New private directory for the copy.")],
    config: ConfigPath = None,
) -> None:
    """After stopping the UI and tasks, copy settings, chats, attachments and the ledger."""
    from personal_harness.backup import backup_data

    resolved = _load(config, None)
    try:
        copied_config = backup_data(resolved.source_file, destination)
    except (ValueError, OSError, sqlite3.Error) as error:
        typer.echo(f"Backup failed: {error}", err=True)
        raise typer.Exit(2) from None
    typer.echo(f"Backup configuration: {copied_config}")


@app.command("places")
def local_places(
    config: ConfigPath = None,
    profile: Profile = None,
    category: Annotated[str | None, typer.Option("--category")] = None,
    limit: Annotated[int | None, typer.Option("--limit", min=1, max=20)] = None,
) -> None:
    """Filter configured local places as JSON, with no model or external API request."""
    from personal_harness.places import query_local_places

    resolved = _load(config, profile)
    try:
        typer.echo(query_local_places(resolved, category=category, limit=limit))
    except ValueError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(2) from None


@app.command("budget")
def inspect_budget(
    config: ConfigPath = None,
    profile: Profile = None,
    all_profiles: Annotated[
        bool, typer.Option("--all-profiles", help="Inspect every configured profile.")
    ] = False,
    as_json: Annotated[
        bool, typer.Option("--json", help="Output a read-only budget snapshot.")
    ] = False,
) -> None:
    """Check whether a new task request fits, without keys or network calls."""
    from personal_harness.budget import budget_status, budget_statuses

    if all_profiles and profile is not None:
        typer.echo("Choose either --profile or --all-profiles.", err=True)
        raise typer.Exit(2)
    resolved = _load(config, None if all_profiles else profile)
    try:
        snapshots = (
            budget_statuses([_load(resolved.source_file, name) for name in resolved.profiles])
            if all_profiles
            else [budget_status(resolved)]
        )
    except ValueError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(2) from None
    if as_json:
        _json(snapshots[0] if not all_profiles else {"schema_version": 1, "profiles": snapshots})
    else:
        typer.echo("\n\n---\n\n".join(budget_receipt(snapshot) for snapshot in snapshots))
    if any(
        snapshot["paid_execution"] and not snapshot["new_task_request_fits"]
        for snapshot in snapshots
    ):
        raise typer.Exit(1)


@app.command("demo")
def demo(
    task: Annotated[str, typer.Argument(help="A sample task; it will not be executed.")],
    config: ConfigPath = None,
    profile: Profile = None,
    as_json: Annotated[bool, typer.Option("--json", help="Output structured run data.")] = False,
) -> None:
    """Rehearse how your preferences shape a run. Zero network/API cost."""
    resolved = _load(config, profile)
    if resolved.config.model.provider != "fake":
        typer.echo("demo requires a fake provider; use run for a metered real task.", err=True)
        raise typer.Exit(2)
    try:
        result = asyncio.run(Agent(resolved).run(task))
    except (ValueError, RunError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(2) from None
    if as_json:
        _json(result.as_dict())
    else:
        typer.echo(result.output)
        typer.echo(f"Cost: USD {result.cost_usd}")


@app.command("workspace-demo")
def workspace_demo(
    source: Annotated[str, typer.Argument(help="UTF-8 source path within your workspace.")],
    target: Annotated[str, typer.Argument(help="Proposed output path within your workspace.")],
    config: ConfigPath = None,
    profile: Profile = None,
) -> None:
    """Extract TODO lines with an offline scripted workflow; ask before writing."""
    resolved = _load(config, profile)

    async def execute() -> RunResult:
        async with Agent(resolved, model=cowork_model(resolved, source, target)) as agent:
            return await _complete_task(
                agent,
                f"Extract TODO lines from {source} into {target}.",
                json_output=False,
            )

    try:
        result = asyncio.run(execute())
    except (ConfigError, RunError, ValueError) as error:
        typer.echo(f"Workflow error: {error}", err=True)
        raise typer.Exit(2) from None
    typer.echo(result.output)
    typer.echo(f"Requests: {result.requests}; cost: USD {result.cost_usd}")


@contextmanager
def _terminal_input() -> Iterator[None]:
    # asyncio's SIGINT handler cancels a task but cannot wake synchronous terminal input.
    if current_thread() is not main_thread():
        yield
        return
    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


async def _complete_task(
    agent: Agent, task: str, *, json_output: bool, tools: list[str] | None = None
) -> RunResult:
    result = await agent.run(task, tools=tools)
    while result.status == "pending_approval":
        decisions: dict[str, bool] = {}
        for approval in result.approvals:
            await asyncio.sleep(0)
            preview = approval.preview
            typer.echo(
                f"Proposed write: {preview['path']}\n"
                f"Expected SHA-256: {preview['expected_sha256']}\n{preview['diff']}",
                err=json_output,
            )
            if preview["diff_truncated"]:
                typer.echo(preview["content"], err=json_output)
            with _terminal_input():
                if json_output:
                    # Click's input echo can use stdout even when its prompt uses stderr.
                    with redirect_stdout(sys.stderr):
                        approved = typer.confirm("Apply this exact write?", default=False, err=True)
                else:
                    approved = typer.confirm("Apply this exact write?", default=False)
            decisions[approval.call_id] = approved
        result = await agent.resume(result.run_id, decisions)
    return result


def _show_run_error(error: RunError, *, as_json: bool = False) -> None:
    if as_json:
        _json(
            dict(
                error=str(error),
                reason=error.reason,
                run_id=error.run_id,
                cost_summary=error.cost_summary,
                execution_trace=error.execution_trace.model_dump(mode="json")
                if error.execution_trace is not None
                else None,
            )
        )
    else:
        typer.echo(f"Task stopped ({error.reason}): {error}", err=True)
        if error.cost_summary is not None:
            typer.echo(ledger_cost_receipt(error.cost_summary), err=True)
        if error.run_id is not None:
            typer.echo(f"Run: {error.run_id}", err=True)


def _show_result(result: RunResult) -> None:
    typer.echo(result.output)
    typer.echo(run_cost_receipt(result))
    typer.echo(f"Run: {result.run_id}")


@app.command("run")
def run_task(
    task: Annotated[
        str, typer.Argument(help="A task executed with your configured model and tools.")
    ],
    config: ConfigPath = None,
    profile: Profile = None,
    as_json: Annotated[bool, typer.Option("--json")] = False,
    tools: Annotated[
        list[str] | None,
        typer.Option(
            "--tool", help="Enable only these configured tools for this task; repeatable."
        ),
    ] = None,
    no_tools: Annotated[
        bool, typer.Option("--no-tools", help="Run this task without any tools.")
    ] = False,
) -> None:
    """Execute a personal task; meter each model request and ask before writes."""
    if no_tools and tools is not None:
        raise typer.BadParameter("--no-tools cannot be combined with --tool.")
    resolved = _load(config, profile)

    async def execute() -> RunResult:
        async with Agent(resolved) as agent:
            return await _complete_task(
                agent, task, json_output=as_json, tools=[] if no_tools else tools
            )

    try:
        result = asyncio.run(execute())
    except (ConfigError, RunError, ValueError) as error:
        if isinstance(error, RunError):
            _show_run_error(error, as_json=as_json)
        else:
            typer.echo(str(error), err=True)
        raise typer.Exit(1) from None
    if as_json:
        _json(result.as_dict())
    else:
        _show_result(result)


@app.command("work")
def work(
    config: ConfigPath = None,
    profile: Profile = None,
    tools: Annotated[
        list[str] | None,
        typer.Option("--tool", help="Enable only these configured tools; repeatable."),
    ] = None,
    no_tools: Annotated[bool, typer.Option("--no-tools", help="Work without any tools.")] = False,
) -> None:
    """Enter independent tasks, reusing your Agent and asking before every write."""
    if no_tools and tools is not None:
        raise typer.BadParameter("--no-tools cannot be combined with --tool.")
    initial = _load(config, profile)
    try:
        selected = select_task_tools(initial.config.tools.allowed, [] if no_tools else tools)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from None
    typer.echo(
        f"Profile: {initial.profile}; model: {initial.config.model.provider}/"
        f"{initial.config.model.model}; tools: {len(selected)}"
    )
    typer.echo("Each line starts an independent task. Use :quit or EOF to exit; Ctrl-C to stop.")

    async def execute() -> None:
        agent: Agent | None = None
        current: ResolvedConfig | None = None
        try:
            while True:
                await asyncio.sleep(0)
                try:
                    with _terminal_input():
                        task = input("Task> ").strip()
                except EOFError:
                    return
                if task == ":quit":
                    return
                if not task:
                    continue
                # Freeze the chosen file/profile and tool ceiling; reload effective settings.
                latest = _load(initial.source_file, initial.profile)
                try:
                    select_task_tools(latest.config.tools.allowed, selected)
                except ValueError as error:
                    typer.echo(f"Tool selection error: {error}", err=True)
                    raise typer.Exit(2) from None
                if current is None or current.config != latest.config:
                    if agent is not None:
                        await agent.aclose()
                    agent = None
                    if current is not None:
                        typer.echo(
                            f"Using updated configuration: {latest.profile}; model: "
                            f"{latest.config.model.provider}/{latest.config.model.model}."
                        )
                    current = latest
                    agent = Agent(latest)
                assert agent is not None
                result = await _complete_task(agent, task, json_output=False, tools=list(selected))
                _show_result(result)
                if not result.total_cost_complete or not result.usage_complete:
                    raise typer.Exit(1)
        finally:
            if agent is not None:
                await agent.aclose()

    try:
        asyncio.run(execute())
    except RunError as error:
        _show_run_error(error)
        raise typer.Exit(1) from None
    except (ConfigError, ValueError) as error:
        typer.echo(f"Work stopped: {error}", err=True)
        raise typer.Exit(2) from None
    except (KeyboardInterrupt, asyncio.CancelledError):
        typer.echo(
            "Work interrupted. Use ledger list to inspect any recorded request costs.", err=True
        )
        raise typer.Exit(130) from None


def _existing_ledger(config: ConfigPath, profile: Profile) -> Ledger:
    resolved = _load(config, profile)
    path = resolved.source_file.parent / ".personal-harness/ledger.sqlite3"
    if not path.is_file():
        raise ValueError("No ledger exists beside this configuration yet.")
    return Ledger(path)


@ledger_app.command("show")
def ledger_show(
    run_id: Annotated[str, typer.Argument(help="Run ID from a result or error.")],
    config: ConfigPath = None,
    profile: Profile = None,
    as_text: Annotated[
        bool, typer.Option("--text", help="Show a readable receipt instead of JSON.")
    ] = False,
) -> None:
    """Show a run's settled costs, pending reservations and every request charge."""
    try:
        summary = _existing_ledger(config, profile).summary(run_id)
        if as_text:
            typer.echo(f"Run: {summary['run_id']}\nStatus: {summary['status']}")
            if summary["stop_reason"] is not None:
                typer.echo(f"Stop reason: {summary['stop_reason']}")
            typer.echo(ledger_cost_receipt(summary))
        else:
            _json(summary)
    except ValueError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(2) from None


@ledger_app.command("list")
def ledger_list(config: ConfigPath = None, profile: Profile = None) -> None:
    """List the latest 20 runs, including interrupted runs and unresolved charges."""
    try:
        _json(_existing_ledger(config, profile).recent())
    except ValueError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(2) from None


@ledger_app.command("reconcile")
def ledger_reconcile(
    request_id: Annotated[str, typer.Argument(help="Unknown/overrun request to reconcile.")],
    amount_cny: Annotated[str, typer.Option("--amount-cny", help="Verified billed CNY amount.")],
    source: Annotated[
        str, typer.Option("--source", help="Billing reference or verification note.")
    ],
    config: ConfigPath = None,
    profile: Profile = None,
) -> None:
    """Record an explicitly verified bill; do not use zero unless verified uncharged."""
    try:
        _existing_ledger(config, profile).reconcile(request_id, Decimal(amount_cny), source=source)
    except (ValueError, InvalidOperation) as error:
        typer.echo(
            f"Cannot reconcile ({type(error).__name__}); check ID, amount and source.", err=True
        )
        raise typer.Exit(2) from None
    typer.echo("Recorded billing reconciliation; the previous charge remains in the audit table.")
