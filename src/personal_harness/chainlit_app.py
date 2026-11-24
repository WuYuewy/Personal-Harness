"""Optional Chainlit entry point with per-chat Agent/client/cache reuse.

Start from an installed package with:
    personal-harness ui --config personal-harness.toml
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shlex
import sqlite3
import time
from contextlib import suppress
from copy import deepcopy
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

import chainlit as cl
from chainlit.types import ChatProfile, ThreadDict
from pydantic import SecretStr, ValidationError
from pydantic_ai.messages import ModelMessage

from personal_harness import Agent, ConfigError, RunError, load_config
from personal_harness import ui_features as features
from personal_harness.conversation import latest_turn_size
from personal_harness.cowork import cowork_model
from personal_harness.display import budget_receipt, ledger_cost_receipt, run_cost_receipt
from personal_harness.reports import (
    ReportDraft,
    ReportRecipe,
    apply_report,
    inspect_report_sources,
    prepare_report,
)
from personal_harness.tools import select_task_tools
from personal_harness.ui_state import Connection, LocalShortcut
from personal_harness.user_insights import MemoryInput, UserInsights
from personal_harness.workspace import WorkspaceError

TOOL_LABELS = {
    "context.describe": "Inspect personal preferences",
    "workspace.list": "List files",
    "workspace.read": "Read a file or excerpt",
    "workspace.read_many": "Read several files",
    "workspace.search": "Search files",
    "workspace.write_preview": "Preview file changes",
    "workspace.write": "Write files after approval",
    "workspace.replace_text": "Replace exact text after approval",
    "places.search": "Search configured local places",
}


class _ChatRuntime:
    # Tasks are serialized through approvals; another message cannot replace its Agent.
    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.agent: Agent | None = None
        self.key: tuple[str, ...] | None = None
        self.ended = False
        self.active_task: asyncio.Task[object] | None = None
        self.generation = 0
        self.history: list[ModelMessage] = []
        self.runs: list[dict[str, Any]] = []
        self.report_drafts: dict[str, tuple[str, ReportRecipe, ReportDraft]] = {}

    async def retire(self) -> None:
        agent, self.agent = self.agent, None
        self.key = None
        if agent is not None:
            await agent.aclose()

    def cancel_active(self) -> None:
        if (
            self.active_task is not None
            and self.active_task is not asyncio.current_task()
            and not self.active_task.done()
            and not self.active_task.cancelling()
        ):
            self.active_task.cancel()


class _ProgressReporter:
    """Show bounded execution state without exposing hidden model reasoning."""

    _STAGES = (
        (0, "正在准备任务上下文…"),
        (8, "模型仍在处理任务，正在等待响应…"),
        (25, "任务耗时较长，网关可能正在生成或执行工具…"),
        (60, "仍在等待模型结果；不会重复发送请求…"),
    )

    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled
        self.message: Any | None = None
        self.step: Any | None = None
        self.started = time.monotonic()
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if not self.enabled:
            return
        content = self._content(self._STAGES[0][1])
        # Native Steps are collapsed by default, like Codex execution details.
        # The message fallback keeps progress harmless in older Chainlit versions
        # and in lightweight test doubles that do not provide a request context.
        try:
            step = cl.Step(
                name="执行进度",
                type="run",
                default_open=False,
                auto_collapse=True,
                show_input=False,
            )
            step.output = content
            if not callable(getattr(step, "update", None)):
                return
            await step.send()
            self.step = step
        except Exception:
            message = cl.Message(content=content)
            if not callable(getattr(message, "update", None)):
                return
            self.message = message
            await message.send()
        self._task = asyncio.create_task(self._heartbeat())

    def _content(self, stage: str) -> str:
        elapsed = int(time.monotonic() - self.started)
        return f"执行进度 · {stage}\n已等待 {elapsed} 秒"

    async def _heartbeat(self) -> None:
        try:
            while not self._stop.is_set():
                await asyncio.sleep(5)
                elapsed = int(time.monotonic() - self.started)
                stage = next(text for seconds, text in reversed(self._STAGES) if elapsed >= seconds)
                await self.update(stage)
        except asyncio.CancelledError:
            raise
        except Exception:
            # A progress indicator must never affect the task itself.
            return

    async def update(self, stage: str) -> None:
        if self.step is None and self.message is None:
            return
        content = self._content(stage)
        if self.step is not None:
            self.step.output = content
            try:
                await self.step.update()
            except Exception:
                return
            return
        assert self.message is not None
        self.message.content = content
        try:
            await self.message.update()
        except Exception:
            return

    async def finish(self, result: Any) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        if self.step is None and self.message is None:
            return
        calls = tuple(getattr(result, "tool_calls", ()) or ())
        if calls:
            detail = "、".join(f"`{call}`" for call in calls)
            await self.update(f"已完成，使用工具：{detail}。")
        else:
            await self.update("已完成，模型已返回回复。")

    async def pause(self, stage: str) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        await self.update(stage)

    async def resume(self) -> None:
        if self.step is None and self.message is None:
            return
        self._stop.clear()
        await self.update("审批已结束，正在继续处理…")
        self._task = asyncio.create_task(self._heartbeat())

    async def fail(self, text: str) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        await self.update(text)


class _StreamingReply:
    """Stream visible answer text into one Chainlit message, when available."""

    def __init__(self) -> None:
        self.message: cl.Message | None = None

    async def on_text_delta(self, delta: str) -> None:
        if self.message is None:
            self.message = cl.Message(content="")
        await self.message.stream_token(delta)

    async def discard(self) -> None:
        message, self.message = self.message, None
        if message is not None:
            with suppress(Exception):
                await message.remove()

    async def finish(
        self,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
        elements: list[Any] | None = None,
    ) -> bool:
        message, self.message = self.message, None
        if message is None:
            return False
        message.content = content
        if metadata is not None:
            message.metadata = metadata
        if elements is not None:
            message.elements = elements
        # Chainlit closes the stream and persists the completed message under the same id.
        await message.send()
        return True


def _runtime() -> _ChatRuntime:
    runtime = cl.user_session.get("harness_runtime")
    if runtime is None:
        runtime = _ChatRuntime()
        cl.user_session.set("harness_runtime", runtime)
    assert isinstance(runtime, _ChatRuntime)
    return runtime


def config_path() -> Path:
    return Path(os.environ.get("PERSONAL_HARNESS_CONFIG", "personal-harness.toml"))


if os.environ.get("PERSONAL_HARNESS_UI_PERSIST") == "1":
    from personal_harness.chainlit_data import LocalDataLayer

    @cl.data_layer
    def data_layer() -> LocalDataLayer:
        store = features.state()
        assert store is not None
        return LocalDataLayer(store)

    @cl.password_auth_callback
    async def authenticate(username: str, password: str) -> cl.User | None:
        store = features.state()
        assert store is not None
        if await asyncio.to_thread(store.authenticate, username, password):
            return cl.User(identifier="local", display_name="我")
        return None

    # Persistent downloads use authenticated URLs, never an unguarded public folder.
    from chainlit.auth import get_current_user
    from chainlit.server import app as web_app
    from fastapi import Depends, HTTPException
    from fastapi.responses import FileResponse

    @web_app.get("/harness/asset/{element_id}")
    async def asset(
        element_id: str,
        user: Annotated[Any, Depends(get_current_user)],
    ) -> FileResponse:
        store = features.state()
        if store is None or user is None:
            raise HTTPException(401)
        with store.db() as db:
            row = db.execute(
                "SELECT thread_id, data FROM elements WHERE id=?", (element_id,)
            ).fetchone()
        if not row:
            raise HTTPException(404)
        thread = store.thread(row[0])
        if not thread or thread.get("userIdentifier") != user.identifier:
            raise HTTPException(404)
        import json

        data = json.loads(row[1])
        name = data.get("objectKey") or ""
        if not name or any(c not in "0123456789abcdef" for c in name):
            raise HTTPException(404)
        path = store.path.parent / "ui-assets" / name
        if not path.is_file():
            raise HTTPException(404)
        return FileResponse(
            path,
            media_type=data.get("mime") or "application/octet-stream",
            filename=data.get("name") or "download",
        )

    # Chainlit includes an SPA catch-all before loading the user app. Register the
    # authenticated asset route ahead of that fallback, or downloads receive HTML.
    web_app.router.routes.insert(0, web_app.router.routes.pop())


@cl.set_chat_profiles
async def chat_profiles(user: cl.User | None = None) -> list[ChatProfile]:
    try:
        resolved = load_config(config_path())
    except ConfigError:
        return [
            ChatProfile(
                name="configuration-error",
                display_name="Configuration error",
                markdown_description="Fix PERSONAL_HARNESS_CONFIG before starting a run.",
                default=True,
            )
        ]
    profiles = [
        ChatProfile(
            name=name,
            display_name=name.replace("_", " ").title(),
            markdown_description=f"Use your {name} personal configuration.",
            default=name == resolved.profile,
        )
        for name in resolved.profiles
    ]
    store = features.state()
    if store and user:
        for scene in await asyncio.to_thread(store.scenes, user.identifier):
            if not scene["archived"] and scene["base_profile"] in resolved.profiles:
                profiles.append(
                    ChatProfile(
                        name=scene["id"],
                        display_name=scene["name"],
                        markdown_description="使用你保存的场景设置。",
                    )
                )
    return profiles


@cl.on_chat_start
async def start() -> None:
    cl.user_session.set("harness_start_error", False)
    try:
        profile = cl.user_session.get("chat_profile")
        cl.user_session.set("profile", profile)
        resolved = features.load_chat_config(profile, new_chat=True)
    except (ConfigError, ValueError) as error:
        cl.user_session.set("harness_start_error", True)
        await cl.Message(content=f"Configuration error: {error}").send()
        return
    profile = resolved.profile
    cl.user_session.set("profile", profile)
    store = features.state()
    defaults = store.defaults(features.owner(), profile) if store else {}
    if store:
        defaults = features.apply_personal_profile(
            defaults, store.personal_profile(features.owner())
        )
    settings = await features.settings_form(resolved, defaults)
    cl.user_session.set("settings", settings)
    await features.install_commands()
    try:
        resolved, _ = features.resolve_settings(settings, profile)
    except ValueError:
        await cl.Message(
            content="已保存的模型连接不可用。请在设置中重新选择模型或在模型连接中配置，未发送模型请求。"
        ).send()
        return
    if store:
        shortcuts = await asyncio.to_thread(store.local_shortcuts, features.owner(), profile)
        await cl.Message(
            content="",
            elements=[
                cl.CustomElement(
                    name="WorkspaceWelcome",
                    props={
                        "offline": resolved.config.runtime.offline,
                        "model": resolved.config.model.model,
                        "initialShortcuts": [
                            {"id": item["id"], "name": item["name"]} for item in shortcuts
                        ],
                    },
                )
            ],
        ).send()
    else:
        await cl.Message(
            content=f"**Personal Harness** · `{profile}` · `{resolved.config.model.model}`"
        ).send()


@cl.on_chat_resume
async def resume_chat(thread: ThreadDict) -> None:
    cl.user_session.set("harness_start_error", False)
    metadata = thread.get("metadata") or {}
    profile = metadata.get("profile") or metadata.get("chat_profile")
    store = features.state()
    runtime = _ChatRuntime()
    if store:
        runtime.history = store.history(thread["id"])
        runtime.runs = store.runs(thread["id"])
    cl.user_session.set("harness_runtime", runtime)
    try:
        resolved = features.load_chat_config(profile)
    except ConfigError as error:
        cl.user_session.set("harness_start_error", True)
        cl.user_session.set("profile", profile)
        await cl.Message(content=f"场景配置不可用：{error}").send()
        return
    cl.user_session.set("profile", resolved.profile)
    settings = metadata.get("settings") or metadata.get("chat_settings") or {}
    if store:
        settings = features.apply_personal_profile(
            settings, store.personal_profile(features.owner())
        )
    settings = await features.settings_form(resolved, settings)
    cl.user_session.set("settings", settings)
    # Chainlit emits resume_thread after this callback. Its metadata must agree
    # with the form, or the browser restores stale values over the fresh widgets.
    thread["metadata"] = {**metadata, "settings": settings, "chat_settings": settings}
    await features.install_commands()


@cl.on_settings_update
async def settings_update(settings: dict[str, object]) -> None:
    previous = dict(cl.user_session.get("settings") or {})
    submitted = {**previous, **deepcopy(settings)}
    store, identity = features.state(), features.thread_id()
    account = store.personal_profile(features.owner()) if store else {}
    previous_preferences = features.personal_profile_from_settings(previous)
    changed = {
        name: value
        for name, value in features.personal_profile_from_settings(settings).items()
        if value != previous_preferences.get(name)
    }
    current = features.apply_personal_profile(submitted, {**account, **changed})
    permissions = settings.get("tool_permissions")
    if isinstance(permissions, list):
        current["tools"] = [name for name in current.get("tools", []) if name in permissions]
    try:
        features.resolve_settings(current, cl.user_session.get("profile"))
    except (ValueError, ConfigError):
        resolved = features.load_chat_config(cl.user_session.get("profile"))
        restored = features.apply_personal_profile(previous, account)
        cl.user_session.set("settings", await features.settings_form(resolved, restored))
        await cl.Message(content="设置未应用：请检查模型连接、预算及数值字段。").send()
        return
    if store and changed:
        store.save_personal_profile(features.owner(), changed)
    cl.user_session.set("settings", current)
    if isinstance(permissions, list) or current != submitted:
        resolved = features.load_chat_config(cl.user_session.get("profile"))
        current = await features.settings_form(resolved, current)
        cl.user_session.set("settings", current)
    if store and identity:
        store.update_thread(identity, metadata={"settings": current})


@cl.action_callback("open_models")
async def open_models(_: cl.Action) -> None:
    await features.show_connections()


@cl.action_callback("show_help")
async def show_help(_: cl.Action) -> None:
    await cl.Message(content=features.HELP).send()


@cl.action_callback("open_insights")
async def open_insights(_: cl.Action) -> None:
    await features.show_insights()


@cl.action_callback("workspace_tool")
async def workspace_tool(action: cl.Action) -> None:
    command = action.payload.get("command")
    if not isinstance(command, str):
        return
    if command in features.WORKSPACE_TOOLS:
        await message_received(cl.Message(content=command))
    elif command in {"/lookup", "/sum", "/actions"}:
        fields: list[str] = []
        keys = (
            ("path", "column", "value")
            if command == "/lookup"
            else (
                ("path", "numeric_column", "filter_column", "filter_value")
                if command == "/sum"
                else ("path", "prefix")
            )
        )
        for key in keys:
            field = action.payload.get(key)
            if (
                not isinstance(field, str)
                or len(field) > 240
                or (field and not field.isprintable())
            ):
                return
            fields.append(field)
        if command == "/lookup" and not all(fields):
            return
        if command == "/sum" and (
            not fields[0] or not fields[1] or bool(fields[2]) != bool(fields[3])
        ):
            return
        if command == "/actions" and (not all(fields) or len(fields[1]) > 80):
            return
        await message_received(cl.Message(content=f"{command} {shlex.join(fields)}"))
    elif command in features.LOCAL_QUERIES:
        query = action.payload.get("query", "")
        if isinstance(query, str) and len(query) <= 200 and (not query or query.isprintable()):
            await message_received(cl.Message(content=f"{command} {query}".strip()))
    store, identity = features.state(), features.thread_id()
    if store and identity:
        thread = store.thread(identity)
        if thread and thread.get("name") in {None, "", "workspace_tool"}:
            store.update_thread(identity, name="新聊天")


@cl.action_callback("user_insights")
async def user_insights(action: cl.Action) -> dict[str, Any]:
    """The authenticated account comes from the session, never from the action payload."""
    payload = dict(action.payload)
    action.payload.clear()
    try:
        store = features.state()
        if store is None:
            raise ValueError("Persistent UI mode is required.")
        service, owner = UserInsights(store), features.owner()
        operation = payload.get("operation", "refresh")
        if not isinstance(operation, str):
            raise ValueError("Invalid operation.")
        if operation == "configure":
            name, enabled = payload.get("name"), payload.get("enabled")
            if not isinstance(name, str) or not isinstance(enabled, bool):
                raise ValueError("Invalid preference option.")
            await asyncio.to_thread(service.set_option, owner, name, enabled)
        elif operation == "add":
            memory = MemoryInput.model_validate(payload.get("memory"))
            await asyncio.to_thread(service.remember, owner, memory)
        elif operation in {"save", "delete", "dismiss"}:
            identity = payload.get("id")
            if not isinstance(identity, str):
                raise ValueError("Choose an existing preference.")
            updated_memory = (
                MemoryInput.model_validate(payload.get("memory")) if operation == "save" else None
            )
            await asyncio.to_thread(service.change, owner, identity, operation, updated_memory)
        elif operation == "clear":
            await asyncio.to_thread(service.clear, owner)
        elif operation != "refresh":
            raise ValueError("Unknown operation.")
        return {"ok": True, "snapshot": await asyncio.to_thread(service.snapshot, owner)}
    except (ValueError, sqlite3.Error):
        return {"ok": False, "message": "未保存。请检查偏好长度、重复条目和账户状态，刷新后重试。"}


@cl.action_callback("personal_scene")
async def personal_scene(action: cl.Action) -> dict[str, Any]:
    payload = dict(action.payload)
    action.payload.clear()
    try:
        store = features.state()
        if store is None:
            raise ValueError("请登录后管理个人场景。")
        owner = features.owner()
        operation = payload.get("operation")
        raw_id, raw_name = payload.get("id"), payload.get("name")
        identity = raw_id if isinstance(raw_id, str) else ""
        name = raw_name if isinstance(raw_name, str) else ""
        needs_identity = operation in ("rename", "copy", "archive", "restore")
        if needs_identity and not features.is_scene(identity):
            raise ValueError("请选择一个个人场景。")
        if operation in ("create", "copy", "rename") and not name.strip():
            raise ValueError("请输入场景名称。")
        if operation in ("create", "save_current"):
            async with _runtime().lock:
                current = cl.user_session.get("profile")
                settings = dict(cl.user_session.get("settings") or {})
                config, _ = features.resolve_settings(settings, current)
                scene = (
                    store.scene(owner, config.profile)
                    if features.is_scene(config.profile)
                    else None
                )
                base = scene["base_profile"] if scene else config.profile
                for field in features.PROFILE_FIELDS.values():
                    settings.pop(field, None)
                if operation == "create":
                    await asyncio.to_thread(
                        store.save_scene,
                        owner,
                        name,
                        base_profile=base,
                        defaults=settings,
                        copy_from=config.profile,
                    )
                elif scene and not scene["archived"]:
                    await asyncio.to_thread(store.save_defaults, owner, config.profile, settings)
                else:
                    raise ValueError("请在启用的个人场景聊天中保存默认设置。")
        elif operation == "copy":
            source = store.scene(owner, identity)
            if source is None:
                raise ValueError("找不到此个人场景。")
            load_config(config_path(), profile=source["base_profile"])
            await asyncio.to_thread(
                store.save_scene,
                owner,
                name,
                base_profile=source["base_profile"],
                defaults=store.defaults(owner, identity),
                copy_from=identity,
            )
        elif operation == "rename":
            await asyncio.to_thread(store.save_scene, owner, name, identity=identity)
        elif operation in ("archive", "restore"):
            await asyncio.to_thread(
                store.archive_scene, owner, identity, archived=operation == "archive"
            )
        elif operation != "list":
            raise ValueError("无法识别场景操作。")
        return {
            "ok": True,
            "items": await asyncio.to_thread(store.scenes, owner),
            "current": cl.user_session.get("profile"),
        }
    except (ConfigError, ValueError) as error:
        return {"ok": False, "message": str(error)}
    except sqlite3.Error:
        return {"ok": False, "message": "暂时无法读取或保存场景，请重试。"}


@cl.action_callback("local_shortcut")
async def local_shortcut(action: cl.Action) -> dict[str, Any]:
    payload = dict(action.payload)
    action.payload.clear()
    try:
        store = features.state()
        profile = cl.user_session.get("profile")
        if store is None or not isinstance(profile, str):
            raise ValueError("请在已登录的聊天中打开快捷查询。")
        owner = features.owner()
        operation, identity = payload.get("operation"), payload.get("id")
        if identity is not None and (
            not isinstance(identity, str) or len(identity) != 32 or not identity.isalnum()
        ):
            raise ValueError("请选择已有快捷查询。")
        if operation == "save":
            try:
                shortcut = LocalShortcut.model_validate(payload.get("shortcut"))
            except ValidationError:
                raise ValueError("请输入有效的名称和查询条件。") from None
            await asyncio.to_thread(store.save_local_shortcut, owner, profile, shortcut, identity)
        elif operation in ("run", "delete"):
            if identity is None:
                raise ValueError("请选择已有快捷查询。")
            if operation == "run":
                shortcut = await asyncio.to_thread(store.local_shortcut, owner, profile, identity)
                await message_received(cl.Message(content=shortcut.content()))
                thread_id = features.thread_id()
                thread = store.thread(thread_id) if thread_id else None
                if thread_id and thread and thread.get("name") == "local_shortcut":
                    store.update_thread(thread_id, name="快捷查询")
                return {"ok": True}
            await asyncio.to_thread(store.delete_local_shortcut, owner, profile, identity)
        elif operation != "list":
            raise ValueError("无法识别快捷查询操作。")
        return {"ok": True, "items": await asyncio.to_thread(store.local_shortcuts, owner, profile)}
    except ValueError as error:
        return {"ok": False, "message": str(error)}
    except sqlite3.Error:
        return {"ok": False, "message": "暂时无法读取或保存快捷查询，请重试。"}


@cl.action_callback("report_recipe")
async def report_recipe(action: cl.Action) -> dict[str, Any]:
    """Manage per-user recipes; prepare, preview and approve deterministic local reports."""
    payload = dict(action.payload)
    action.payload.clear()
    try:
        store = features.state()
        if store is None:
            raise ValueError("请通过 personal-harness ui 启动个人方案管理。")
        account = features.owner()
        runtime = _runtime()
        operation = payload.get("operation")
        if not isinstance(operation, str):
            raise ValueError("未知的任务方案操作。")
        if operation == "refresh":
            return {"ok": True, "recipes": store.report_recipes(account)}

        if operation == "inspect":
            sources = payload.get("sources")
            if (
                not isinstance(sources, list)
                or not 1 <= len(sources) <= 8
                or any(not isinstance(item, str) or len(item) > 240 for item in sources)
            ):
                raise ValueError("请选择 1–8 个工作区来源文件。")
            resolved, _ = features.resolve_settings(
                cl.user_session.get("settings") or {},
                cl.user_session.get("profile"),
                local_only=True,
            )
            discovery = await asyncio.to_thread(inspect_report_sources, resolved, sources)
            return {"ok": True, "discovery": discovery}

        identity = payload.get("id")
        if identity is not None and (
            not isinstance(identity, str) or not identity.isascii() or not identity.isalnum()
        ):
            raise ValueError("个人方案编号无效。")

        if operation in {"create", "save"}:
            if len(json.dumps(payload, ensure_ascii=False)) > 12_000:
                raise ValueError("方案字段总长度不能超过 12 KB。")
            sources, sections = payload.get("sources"), payload.get("sections")
            if (
                not isinstance(sources, list)
                or not 1 <= len(sources) <= 8
                or not isinstance(sections, list)
                or not 1 <= len(sections) <= 6
                or any(not isinstance(item, str) for item in sources)
                or any(not isinstance(item, dict) for item in sections)
            ):
                raise ValueError("每个方案需要 1–8 个来源和 1–6 个内容章节。")
            recipe = ReportRecipe.model_validate(
                {
                    "name": payload.get("name"),
                    "sources": sources,
                    "target": payload.get("target"),
                    "sections": sections,
                }
            )
            recipe_id = store.save_report_recipe(
                account, recipe, identity if operation == "save" else None
            )
            return {
                "ok": True,
                "id": recipe_id,
                "recipes": store.report_recipes(account),
                "message": "方案已保存在本机。它只限定读取路径、精确前缀和 Markdown 输出路径。",
            }

        if operation == "delete":
            if identity is None:
                raise ValueError("请先选择要删除的方案。")
            store.delete_report_recipe(account, identity)
            runtime.report_drafts = {
                token: value
                for token, value in runtime.report_drafts.items()
                if value[0] != identity
            }
            return {"ok": True, "recipes": store.report_recipes(account), "message": "方案已删除。"}

        if operation == "preview":
            if identity is None:
                raise ValueError("请先保存方案再生成预览。")
            recipe = store.report_recipe(account, identity)
            settings = features.apply_personal_profile(
                cl.user_session.get("settings") or {}, store.personal_profile(account)
            )
            resolved, _ = features.resolve_settings(
                settings, cl.user_session.get("profile"), local_only=True
            )
            draft = await asyncio.to_thread(prepare_report, resolved, recipe)
            token = uuid4().hex
            runtime.report_drafts[token] = (identity, recipe, draft)
            while len(runtime.report_drafts) > 12:
                runtime.report_drafts.pop(next(iter(runtime.report_drafts)))
            return {
                "ok": True,
                "token": token,
                "preview": draft.public(),
                "message": "每行均来自当前文件并标有路径和行号。检查完整正文及差异后再提交。",
            }

        if operation == "apply":
            submitted_token = payload.get("token")
            if identity is None or not isinstance(submitted_token, str):
                raise ValueError("预览已过期，请重新生成。")
            pending = runtime.report_drafts.pop(submitted_token, None)
            if pending is None:
                raise ValueError("预览已过期，请重新生成。")
            recipe_id, recipe, draft = pending
            if recipe_id != identity:
                raise ValueError("请使用此方案生成的预览。")
            if draft.unchanged:
                raise ValueError("文件内容未变化，无需写入。")
            async with runtime.lock:
                if runtime.ended or store.report_recipe(account, recipe_id) != recipe:
                    raise ValueError("聊天或个人方案已变化，请重新预览。")
                resolved, _ = features.resolve_settings(
                    cl.user_session.get("settings") or {},
                    cl.user_session.get("profile"),
                    local_only=True,
                )
                await asyncio.to_thread(apply_report, resolved, recipe, draft)
            recorded = features.record_local_operation(
                runtime, "report", source_count=len(draft.sources)
            )
            try:
                store.record_report_run(account, recipe_id, len(draft.sources))
            except (ValueError, sqlite3.Error):
                return {"ok": True, "message": f"已写入 {draft.path}。本地执行次数未能保存。"}
            return {
                "ok": True,
                "recipes": store.report_recipes(account),
                "message": f"已写入 `{draft.path}`；模型请求 0 次，API 费用 ¥0。"
                + ("本聊天的用量记录未能保存。" if not recorded else ""),
            }

        raise ValueError("未知的任务方案操作。")
    except (ValueError, sqlite3.Error) as error:
        return {"ok": False, "message": f"方案未完成（无模型请求）：{error}"}


@cl.action_callback("save_model_connection")
async def save_connection(action: cl.Action) -> dict[str, Any]:
    # Credentials arrive only as an action payload, never as a message or persisted props.
    payload = dict(action.payload)
    raw_key = payload.pop("api_key", "")
    action.payload.clear()
    try:
        store = features.state()
        if store is None:
            raise ValueError("Persistent UI mode is required.")
        identity = payload.pop("connection_id", None)
        name = payload.pop("name", "")
        make_default = payload.pop("make_default", False) is True
        model = {"provider": "openai-compatible", **payload}
        for key in ("context_window", "max_output_tokens"):
            model[key] = int(model[key])
        if not model.get("cache_read_per_million_cny"):
            model["cache_read_per_million_cny"] = None
        connection = Connection(name=name, model=model)
        async with _runtime().lock:
            current = dict(cl.user_session.get("settings") or {})
            # Validate the new primary against the current fallback before persisting.
            validation_settings = dict(current)
            available = {item["id"] for item in store.connections(features.owner())}
            if validation_settings.get("connection", "profile") not in {"profile", *available}:
                validation_settings["connection"] = "profile"
            if validation_settings.get("fallback_connection", "profile") not in {
                "profile",
                "off",
                *available,
            }:
                validation_settings["fallback_connection"] = "off"
            effective, _ = features.resolve_settings(
                validation_settings, cl.user_session.get("profile")
            )
            candidate = effective.config.model_dump()
            candidate["model"] = connection.model.model_dump()
            candidate["runtime"]["offline"] = False
            if identity and current.get("fallback_connection") == identity:
                candidate["fallback_model"] = connection.model.model_dump()
            features.load_chat_config(cl.user_session.get("profile"), overrides=candidate)
            owner = features.owner()
            identity = store.save_connection(
                owner, connection, SecretStr(raw_key) if raw_key else None, identity
            )
            current["connection"] = identity
            current["fallback_connection"] = validation_settings.get(
                "fallback_connection", "profile"
            )
            resolved, _ = features.resolve_settings(current, cl.user_session.get("profile"))
            await _runtime().retire()
            current = await features.settings_form(resolved, current)
            cl.user_session.set("settings", current)
            thread_id = features.thread_id()
            if thread_id:
                store.update_thread(thread_id, metadata={"settings": current})
            if make_default:
                defaults = store.defaults(owner, resolved.profile)
                if defaults.get("fallback_connection") == identity:
                    defaults["fallback_connection"] = "off"
                store.save_defaults(owner, resolved.profile, {**defaults, "connection": identity})
        default_notice = " 已设为此场景的新聊天默认模型。" if make_default else ""
        await cl.Message(
            content=(
                f"已保存并选择模型连接：**{connection.name}** · `{connection.model.model}`。"
                f"本聊天使用此在线模型。{default_notice} 未发送模型请求。"
            )
        ).send()
        return {
            "message": f"已保存并选择连接。{default_notice}",
            "connection_id": identity,
            "connections": store.connections(features.owner()),
        }
    except (ValueError, ValidationError, KeyError, TypeError):
        await cl.Message(
            content="连接未启用：请检查必填字段、API 地址、Token 上限与人民币价格。"
        ).send()
        return {"message": "保存失败，请检查连接参数。"}
    finally:
        raw_key = ""


@cl.action_callback("delete_model_connection")
async def delete_connection(action: cl.Action) -> None:
    store = features.state()
    identity = action.payload.get("connection_id")
    if store and isinstance(identity, str):
        async with _runtime().lock:
            store.delete_connection(features.owner(), identity)
            current = dict(cl.user_session.get("settings") or {})
            await _runtime().retire()
            resolved = features.load_chat_config(cl.user_session.get("profile"))
            current = await features.settings_form(resolved, current)
            cl.user_session.set("settings", current)
            thread_id = features.thread_id()
            if thread_id:
                store.update_thread(thread_id, metadata={"settings": current})
        await cl.Message(
            content="模型连接已删除，历史聊天和费用账本仍保留。使用该连接的聊天请重新选择模型。"
        ).send()


@cl.on_chat_end
async def end() -> None:
    runtime = _runtime()
    runtime.ended = True
    runtime.cancel_active()
    async with runtime.lock:
        await runtime.retire()


@cl.on_stop
async def stop() -> None:
    runtime = _runtime()
    runtime.generation += 1
    runtime.cancel_active()
    # Chainlit cancels the active message; its exception path releases approvals/client.
    # Also handle a stop received after a task has just finished.
    async with runtime.lock:
        await runtime.retire()


@cl.on_message
async def message_received(message: cl.Message) -> None:
    runtime = _runtime()
    if runtime.ended:
        # Chainlit can restore a disconnected websocket session without on_chat_start.
        # New messages get a cold Agent after the old runtime has finished cleanup.
        async with runtime.lock:
            await runtime.retire()
        runtime = _runtime()
        if runtime.ended:
            history, runs = runtime.history, runtime.runs
            runtime = _ChatRuntime()
            runtime.history, runtime.runs = history, runs
            cl.user_session.set("harness_runtime", runtime)
    generation = runtime.generation
    async with runtime.lock:
        if not runtime.ended and generation == runtime.generation:
            runtime.active_task = asyncio.current_task()
            try:
                await _execute_message(runtime, message)
            finally:
                runtime.active_task = None


async def _execute_message(runtime: _ChatRuntime, message: cl.Message) -> None:
    command = getattr(message, "command", None)
    if command == "工具箱" or message.content.strip() == "/工具箱":
        message.content = "/tools"
    elif command in {"models", "files", "usage", "help", "insights"}:
        message.content = f"/{command}"
    profile = cl.user_session.get("profile")
    settings = cl.user_session.get("settings") or {}
    store = features.state()
    if store:
        settings = features.apply_personal_profile(
            settings, store.personal_profile(features.owner())
        )
    parts = message.content.strip().split(maxsplit=1)
    chat_only = command == "仅聊天" or parts[:1] == ["/chat"]
    if parts[:1] == ["/chat"]:
        message.content = parts[1] if len(parts) == 2 else ""
    if chat_only and not message.content.strip():
        await cl.Message(content="请输入要处理的内容，例如 `/chat 总结附件`。").send()
        return
    if not chat_only and parts == ["/usage"]:
        await features.show_usage(runtime)
        return
    if not chat_only and parts == ["/tools"]:
        await features.show_toolbox()
        return
    if not chat_only and parts == ["/reports"]:
        await features.show_report_recipes()
        return
    if not chat_only and parts == ["/scenes"]:
        await features.show_scenes()
        return
    if cl.user_session.get("harness_start_error"):
        await cl.Message(
            content="当前场景不可用。请恢复场景或修复基础配置后新建聊天；历史记录和费用仍保留。"
        ).send()
        return
    if not chat_only and parts and parts[0] in features.LOCAL_QUERIES:
        try:
            if getattr(message, "elements", None):
                raise ValueError("本地查询使用已配置的资料，请先移除本条消息的附件。")
            local, _ = features.resolve_settings(settings, profile, local_only=True)
            await features.show_local_query(
                parts[0], parts[1] if len(parts) == 2 else "", local, runtime
            )
        except (ConfigError, ValueError) as error:
            await cl.Message(content=f"本地查询未完成（未调用模型）：{error}").send()
        return
    progress: _ProgressReporter | None = None
    streamed_reply: _StreamingReply | None = None
    try:
        resolved, api_key = features.resolve_settings(settings, profile)
        fallback_api_key = features.fallback_key(settings)
        if not chat_only and message.content.strip() == "/budget":
            from personal_harness.budget import budget_status

            snapshot = budget_status(resolved)
            await cl.Message(content=budget_receipt(snapshot)).send()
            return
        if not chat_only and await features.local_command(message.content, runtime, resolved):
            return
        background: tuple[str, ...] = ()
        if store and not message.content.lstrip().startswith("/"):
            knowledge = UserInsights(store)
            try:
                await asyncio.to_thread(
                    knowledge.observe,
                    features.owner(),
                    features.thread_id() or "",
                    str(getattr(message, "id", None) or uuid4().hex),
                    message.content,
                )
                background = await asyncio.to_thread(
                    knowledge.recall, features.owner(), message.content
                )
            except (ValueError, sqlite3.Error):
                await cl.Message(
                    content="本轮个人偏好读取或采集失败，任务继续使用显式配置。"
                ).send()
        tools = None
        if "tools" in settings:
            requested = settings["tools"]
            if not isinstance(requested, list):
                raise ValueError("Tool selection must be a list; clear all to disable tools.")
            tools = select_task_tools(resolved.config.tools.allowed, requested)
        workflow: tuple[str, ...] = ()
        if not chat_only and message.content.startswith("/workspace-demo "):
            parts = shlex.split(message.content)
            if len(parts) != 3:
                raise ValueError("Use /workspace-demo <source> <target>; quote paths with spaces.")
            workflow = (parts[1], parts[2])
        test_model = not chat_only and message.content.strip() == "/test-model"
        task = (
            "Reply with exactly OK. Do not call tools."
            if test_model
            else await features.task_content(message, resolved)
        )
        history = (
            runtime.history
            if settings.get("carry_history", True) and not workflow and not test_model
            else []
        )
        latest_bytes = latest_turn_size(history) if resolved.config.runtime.history_turns else 0
        if latest_bytes > resolved.config.runtime.history_max_chars:
            answer = await cl.AskActionMessage(
                content=(
                    f"上一轮内容约 {latest_bytes} 字节，超过当前历史容量 "
                    f"{resolved.config.runtime.history_max_chars} 字节。"
                    "本轮将无法携带之前的对话和附件正文。\n\n"
                    "需要继续分析上一轮资料时，可先取消，在设置 → 费用与上下文调高历史容量，"
                    "或重新附上所需片段。提高容量可能增加费用。"
                ),
                actions=[
                    cl.Action(
                        name="history_cancel", payload={"continue": False}, label="取消，先调整"
                    ),
                    cl.Action(
                        name="history_continue",
                        payload={"continue": True},
                        label="仅用本轮内容继续",
                    ),
                ],
                timeout=120,
            ).send()
            if not answer or answer.get("payload", {}).get("continue") is not True:
                await cl.Message(content="本轮未发送模型请求；原聊天与附件记录仍保留。").send()
                return
        key = (
            str(resolved.source_file),
            resolved.profile,
            resolved.config.model_dump_json(),
            hashlib.sha256(api_key.get_secret_value().encode()).hexdigest() if api_key else "",
            hashlib.sha256(fallback_api_key.get_secret_value().encode()).hexdigest()
            if fallback_api_key
            else "",
            *workflow,
        )
        if runtime.key != key:
            await runtime.retire()
            model = cowork_model(resolved, *workflow) if workflow else None
            options: dict[str, Any] = {}
            if api_key:
                options["api_key"] = api_key
            if fallback_api_key:
                options["fallback_api_key"] = fallback_api_key
            runtime.agent = Agent(resolved, model=model, **options)
            runtime.key = key
        agent = runtime.agent
        assert agent is not None
        progress = _ProgressReporter(
            enabled=bool(features.state()) and not resolved.config.runtime.offline
        )
        await progress.start()
        stream_options = {}
        if (
            progress.enabled
            and resolved.config.runtime.stream_responses
            and resolved.config.fallback_model is None
            and not test_model
        ):
            streamed_reply = _StreamingReply()
            stream_options["on_text_delta"] = streamed_reply.on_text_delta
        result = await agent.run(
            task,
            tools=[] if test_model or chat_only else tools,
            message_history=history,
            user_context=background,
            **stream_options,
        )
        while result.status == "pending_approval":
            if streamed_reply is not None:
                await streamed_reply.discard()
            await progress.pause("工具已生成文件变更预览，等待你检查并审批。")
            decisions: dict[str, bool] = {}
            for approval in result.approvals:
                preview = approval.preview
                await cl.Message(
                    content=(
                        f"Proposed write: `{preview['path']}` ({preview['bytes']} bytes).\n\n"
                        f"Expected SHA-256: `{preview['expected_sha256']}`. "
                        "Review the full proposed contents and diff before confirming below."
                    ),
                    elements=[
                        cl.Text(
                            name="Proposed contents",
                            content=preview["content"] or "(Empty file)",
                            display="inline",
                        ),
                        cl.Text(
                            name="Diff",
                            content=preview["diff"] or "(No text changes)",
                            display="inline",
                        ),
                    ],
                ).send()
                answer = await cl.AskActionMessage(
                    content=(
                        f"Apply the proposed write to `{preview['path']}` "
                        f"({preview['bytes']} bytes)? Review the contents and diff above."
                    ),
                    actions=[
                        cl.Action(name="approve", payload={"approved": True}, label="Apply write"),
                        cl.Action(name="reject", payload={"approved": False}, label="Reject"),
                    ],
                    timeout=120,
                ).send()
                decisions[approval.call_id] = bool(
                    answer and answer.get("payload", {}).get("approved") is True
                )
            await progress.resume()
            result = await agent.resume(result.run_id, decisions, **stream_options)
        await progress.finish(result)
        first_task = not test_model and not any(
            item.get("status") == "completed" and not item.get("diagnostic")
            for item in runtime.runs
        )
        if not test_model and not workflow:
            runtime.history.extend(result.messages)
        run = {
            "run_id": result.run_id,
            "execution": "offline" if result.provider == "fake" else "model",
            "status": result.status,
            "currency": result.currency,
            "cost": str(result.cost) if result.cost is not None else None,
            "external_charges": [c.model_dump(mode="json") for c in result.external_charges],
            "external_cost": str(result.external_cost)
            if result.external_cost is not None
            else None,
            "external_cost_complete": result.external_cost_complete,
            "total_cost": str(result.total_cost) if result.total_cost is not None else None,
            "total_cost_complete": result.total_cost_complete,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "requests": result.requests,
            "duration_ms": result.duration_ms,
            "diagnostic": test_model,
        }
        runtime.runs.append(run)
        store, identity = features.state(), features.thread_id()
        if store and identity:
            store.save_history(identity, runtime.history)
            store.record_run(identity, run)
            store.update_thread(
                identity, metadata={"settings": settings, "profile": resolved.profile}
            )
            if first_task:
                title = (message.content.strip().split("\n", 1)[0] or "新聊天")[:80]
                store.update_thread(identity, name=title)
                await cl.context.emitter.emit(
                    "first_interaction",
                    {
                        "interaction": title,
                        "thread_id": identity,
                    },
                )
    except (ConfigError, RunError, ValueError, WorkspaceError) as error:
        if progress is not None:
            await progress.fail("任务未完成，正在展示错误和费用状态。")
        await runtime.retire()
        detail = f"Configuration or run error: {error}"
        if isinstance(error, RunError) and error.cost_summary is not None:
            detail += (
                "\n\n"
                + ledger_cost_receipt(error.cost_summary).replace("\n", "\n\n")
                + f"\n\nRun: `{error.run_id}`"
            )
            failed = {
                "run_id": error.run_id,
                "status": "failed",
                "currency": "CNY",
                "cost": error.cost_summary.get("cost"),
                "external_charges": error.cost_summary["external_charges"],
                "external_cost": error.cost_summary["external_cost"],
                "external_cost_complete": error.cost_summary["external_cost_complete"],
                "total_cost": error.cost_summary["total_cost"],
                "total_cost_complete": error.cost_summary["total_cost_complete"],
                "input_tokens": error.cost_summary.get("input_tokens", 0),
                "output_tokens": error.cost_summary.get("output_tokens", 0),
            }
            runtime.runs.append(failed)
            store, identity = features.state(), features.thread_id()
            if store and identity:
                store.record_run(identity, failed)
        if streamed_reply is None or not await streamed_reply.finish(detail):
            await cl.Message(content=detail).send()
        return
    except BaseException:
        if progress is not None:
            await progress.fail("任务已停止，正在清理本轮执行状态。")
        agent = runtime.agent
        cancelled_cost = (
            agent.cost_summary(agent.last_run_id) if agent and agent.last_run_id else None
        )
        if cancelled_cost is not None:
            run = {**cancelled_cost, **cancelled_cost["metrics"]}
            runtime.runs.append(run)
            store, identity = features.state(), features.thread_id()
            if store and identity:
                store.record_run(identity, run)
        # Cancellation while waiting for approval must not leave a reusable pending run.
        await runtime.retire()
        if streamed_reply is not None:
            await streamed_reply.discard()
        raise
    if features.state():
        rehearsal = result.provider == "fake" and result.output.startswith(
            "[OFFLINE DEMO: configuration rehearsal]\n"
        )
        receipt = run_cost_receipt(result)
        content = (
            "已完成离线演练。配置与任务输入已检查，实际任务需要在线模型。"
            if rehearsal
            else result.output
        )
        elements = [
            cl.CustomElement(
                name="RunReceipt",
                props={
                    "profile": result.profile,
                    "offline": result.provider == "fake",
                    "currency": result.currency,
                    "cost": format(result.total_cost.normalize(), "f")
                    if result.total_cost is not None
                    else None,
                    "receipt": receipt,
                    "historyReadBytesOmitted": result.history_read_bytes_omitted,
                    "historyReadsReferenced": result.history_reads_referenced,
                    "pagedReadResults": result.paged_read_results,
                    "readResultBytesOmitted": result.read_result_bytes_omitted,
                    "cacheReadTokens": result.cache_read_tokens,
                    "cacheSavings": (
                        format(result.cache_savings_cny.normalize(), "f")
                        if result.cache_savings_cny is not None
                        else None
                    ),
                    "rehearsal": result.output if rehearsal else "",
                    "tokens": result.input_tokens + result.output_tokens,
                    "availableTools": len(result.available_tools),
                    "toolCalls": len(result.tool_calls),
                    "preferences": len(background),
                    "runId": result.run_id,
                },
            )
        ]
        metadata = {"harness_receipt": receipt, "harness_run_id": result.run_id}
        if streamed_reply is None or not await streamed_reply.finish(
            content, metadata=metadata, elements=elements
        ):
            await cl.Message(content=content, metadata=metadata, elements=elements).send()
    else:
        await cl.Message(
            content=(
                f"{result.output}\n\n"
                f"Profile: `{result.profile}` · tools enabled: `{len(result.available_tools)}` · "
                f"tool calls: `{len(result.tool_calls)}`\n\n"
                + run_cost_receipt(result).replace("\n", "\n\n")
                + f"\n\nRun: `{result.run_id}`"
            )
        ).send()
