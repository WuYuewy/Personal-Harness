"""Optional UI boundary tests, run with `uv run --extra ui pytest tests/test_ui.py`."""

import asyncio
import importlib.util
import json
import os
import sys
from decimal import Decimal
from importlib.resources import files
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from personal_harness import Agent, RunError


@pytest.fixture
def ui(config_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    pytest.importorskip("chainlit", reason="Install the ui extra to test Chainlit callbacks")
    monkeypatch.setenv("PERSONAL_HARNESS_CONFIG", str(config_file))
    path = Path(str(files("personal_harness").joinpath("chainlit_app.py")))
    spec = importlib.util.spec_from_file_location("harness_ui_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    session: dict[str, Any] = {}
    sent: list[str] = []
    messages: list[dict[str, Any]] = []
    monkeypatch.setattr(
        module.cl, "user_session", SimpleNamespace(get=session.get, set=session.__setitem__)
    )

    class Message:
        def __init__(self, *, content: str, **kwargs: Any) -> None:
            self.content = content
            self.kwargs = kwargs

        async def send(self) -> None:
            sent.append(self.content)
            messages.append({"content": self.content, **self.kwargs})

    monkeypatch.setattr(module.cl, "Message", Message)
    return SimpleNamespace(
        app=module, session=session, sent=sent, messages=messages, config=config_file
    )


def test_profile_picker_reads_user_defined_names(ui: Any) -> None:
    ui.config.write_text(ui.config.read_text("utf-8") + "\n[profiles.research]\n", "utf-8")
    names = [profile.name for profile in asyncio.run(ui.app.chat_profiles())]
    assert names == ["desktop", "mobile", "research"]


def test_scene_rename_and_archive_preserve_existing_chats_and_isolate_owners(ui, monkeypatch):
    from chainlit.context import init_http_context

    from personal_harness.ui_state import LocalShortcut

    monkeypatch.setenv("PERSONAL_HARNESS_UI_PERSIST", "1")
    account = "alice"
    monkeypatch.setattr(ui.app.features, "owner", lambda: account)
    ui.session["profile"] = "desktop"
    ui.session["settings"] = {"tool_permissions": ["workspace.read"], "experience_tone": "warm"}
    store = ui.app.features.state()
    shortcut = LocalShortcut(name="待办", command="/actions", args=("notes.md",))
    store.save_local_shortcut(account, "desktop", shortcut, None)
    monkeypatch.setattr(ui.app, "Agent", lambda *a, **kw: pytest.fail("Scenes must run locally"))

    async def action(operation, **fields):
        return await ui.app.personal_scene(
            SimpleNamespace(payload={"operation": operation, **fields})
        )

    async def exercise():
        nonlocal account
        init_http_context()
        created = await action("create", name="项目协作")
        (item,) = created["items"]
        identity = item["id"]
        assert store.defaults(account, identity) == {"tool_permissions": ["workspace.read"]}
        assert store.local_shortcuts(account, identity)[0]["args"] == ["notes.md"]
        ui.session["profile"] = identity
        ui.session["settings"]["tool_permissions"] = []
        assert (await action("save_current"))["ok"]
        assert store.defaults(account, identity)["tool_permissions"] == []

        assert (await action("rename", id=identity, name="项目周报"))["ok"]
        assert (await action("copy", id=identity, name="复核周报"))["ok"]
        profiles = await ui.app.chat_profiles(ui.app.cl.User(identifier=account))
        assert next(p.display_name for p in profiles if p.name == identity) == "项目周报"
        assert store.scene(account, identity)["base_profile"] == "desktop"

        assert (await action("archive", id=identity))["ok"]
        profiles = await ui.app.chat_profiles(ui.app.cl.User(identifier=account))
        assert identity not in {p.name for p in profiles}
        with pytest.raises(ui.app.ConfigError, match="已归档"):
            ui.app.features.load_chat_config(identity, new_chat=True)
        resolved = ui.app.features.load_chat_config(identity)
        assert resolved.profile == identity
        assert "project_collaboration" in resolved.config.preferences.priorities
        assert (await action("restore", id=identity))["ok"]

        account = "bob"
        assert (await action("list"))["items"] == []
        assert not (await action("rename", id=identity, name="冒用"))["ok"]
        with pytest.raises(ui.app.ConfigError, match="找不到"):
            ui.app.features.load_chat_config(identity)

    asyncio.run(exercise())


@pytest.mark.parametrize("field", ["connection", "fallback_connection"])
def test_missing_connection_remains_selected_until_user_repairs_it(ui, monkeypatch, field):
    from chainlit.context import init_http_context

    monkeypatch.setenv("PERSONAL_HARNESS_UI_PERSIST", "1")
    monkeypatch.setattr(ui.app.features, "owner", lambda: "alice")
    store = ui.app.features.state()
    store.save_scene("alice", "项目", base_profile="desktop", defaults={field: "deleted"})
    scene = store.scenes("alice")[0]["id"]
    ui.session["chat_profile"] = scene
    widgets = []
    original = ui.app.cl.ChatSettings

    class Settings(original):
        async def send(self):
            widgets.extend(w for tab in self.inputs for w in tab.inputs)
            return self.settings()

    monkeypatch.setattr(ui.app.cl, "ChatSettings", Settings)
    monkeypatch.setattr(
        ui.app, "Agent", lambda *a, **kw: pytest.fail("Missing connection called Agent")
    )

    async def exercise():
        init_http_context()
        await ui.app.start()
        assert "已保存的模型连接不可用" in ui.sent[-1]
        assert ui.session["settings"][field] == "deleted"
        selector = next(w for w in widgets if w.id == field)
        assert selector.initial_value == "deleted" and "deleted" in selector.items.values()
        thread = {"id": "old", "metadata": {"profile": scene, "settings": {field: "deleted"}}}
        await ui.app.resume_chat(thread)
        assert thread["metadata"]["settings"][field] == "deleted"
        await ui.app.message_received(SimpleNamespace(content="处理本轮任务"))
        assert "Model connection no longer exists" in ui.sent[-1]
        await ui.app.settings_update({field: "profile"})
        resolved, _ = ui.app.features.resolve_settings(ui.session["settings"], scene)
        assert resolved.profile == scene
        assert resolved.config.tools.allowed == ("context.describe",)

    asyncio.run(exercise())
    assert not (ui.config.parent / ".personal-harness/ledger.sqlite3").exists()


def test_missing_scene_base_preserves_usage_and_blocks_execution(ui, monkeypatch):
    monkeypatch.setenv("PERSONAL_HARNESS_UI_PERSIST", "1")
    monkeypatch.setattr(ui.app.features, "owner", lambda: "alice")
    store = ui.app.features.state()
    store.save_scene("alice", "项目", base_profile="removed", defaults={})
    scene = store.scenes("alice")[0]["id"]
    store.record_run("old", {"run_id": "example", "requests": 0, "cost": "0"})
    thread = {"id": "old", "metadata": {"profile": scene, "settings": {}}}
    monkeypatch.setattr(ui.app, "Agent", lambda *a, **kw: pytest.fail("Missing scene called Agent"))

    async def exercise():
        await ui.app.resume_chat(thread)
        assert "场景配置不可用" in ui.sent[-1]
        assert ui.session["profile"] == scene
        assert ui.session["harness_runtime"].runs == store.runs("old")
        await ui.app.message_received(SimpleNamespace(content="继续之前的任务"))
        assert "当前场景不可用" in ui.sent[-1]
        assert thread["metadata"]["profile"] == scene

    asyncio.run(exercise())


def test_archived_scene_cached_by_browser_cannot_start_new_tasks(ui, monkeypatch):
    monkeypatch.setenv("PERSONAL_HARNESS_UI_PERSIST", "1")
    monkeypatch.setattr(ui.app.features, "owner", lambda: "alice")
    store = ui.app.features.state()
    store.save_scene("alice", "归档项目", base_profile="desktop", defaults={})
    scene = store.scenes("alice")[0]["id"]
    store.archive_scene("alice", scene, archived=True)
    ui.session["chat_profile"] = scene
    monkeypatch.setattr(
        ui.app, "Agent", lambda *a, **kw: pytest.fail("Archived scene called Agent")
    )

    async def exercise():
        await ui.app.start()
        assert "已归档" in ui.sent[-1]
        await ui.app.message_received(SimpleNamespace(content="执行任务"))
        assert "当前场景不可用" in ui.sent[-1]
        await ui.app.message_received(SimpleNamespace(content="/find private"))
        assert "当前场景不可用" in ui.sent[-1]

    asyncio.run(exercise())
    assert not (ui.config.parent / ".personal-harness/ledger.sqlite3").exists()


def test_new_connection_repairs_deleted_selection_without_sending_request(ui, monkeypatch):
    from chainlit.context import init_http_context

    monkeypatch.setenv("PERSONAL_HARNESS_UI_PERSIST", "1")
    monkeypatch.setattr(ui.app.features, "owner", lambda: "alice")
    monkeypatch.setattr(ui.app.features, "thread_id", lambda: None)
    ui.session["profile"] = "desktop"
    ui.session["settings"] = {
        "connection": "deleted",
        "fallback_connection": "deleted",
        "max_task_cny": "0.5",
        "max_daily_cny": "1",
    }
    original = ui.app.cl.ChatSettings

    class Settings(original):
        async def send(self):
            return self.settings()

    monkeypatch.setattr(ui.app.cl, "ChatSettings", Settings)
    monkeypatch.setattr(ui.app, "Agent", lambda *a, **kw: pytest.fail("Model setup called Agent"))

    async def exercise():
        init_http_context()
        result = await ui.app.save_connection(
            SimpleNamespace(
                payload={
                    "name": "New model",
                    "model": "my-model",
                    "base_url": "https://example.test/v1",
                    "context_window": 8192,
                    "max_output_tokens": 128,
                    "input_per_million_cny": "1",
                    "output_per_million_cny": "1",
                }
            )
        )
        identity = result["connection_id"]
        assert ui.session["settings"]["connection"] == identity
        assert ui.session["settings"]["fallback_connection"] == "off"
        await ui.app.delete_connection(SimpleNamespace(payload={"connection_id": identity}))
        assert ui.session["settings"]["connection"] == identity
        with pytest.raises(ValueError, match="no longer exists"):
            ui.app.features.resolve_settings(ui.session["settings"], "desktop")

    asyncio.run(exercise())
    assert not (ui.config.parent / ".personal-harness/ledger.sqlite3").exists()


def test_continuous_chat_persists_native_context_and_clear_keeps_transcript(ui, monkeypatch):
    from chainlit.context import init_http_context
    from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
    from pydantic_ai.models.function import FunctionModel

    def respond(messages, info):
        prompts = [
            p.content
            for m in messages
            if isinstance(m, ModelRequest)
            for p in m.parts
            if isinstance(p, UserPromptPart)
        ]
        return ModelResponse([TextPart(" | ".join(prompts))])

    class ChatAgent(Agent):
        def __init__(self, resolved, **kwargs):
            super().__init__(resolved, model=FunctionModel(respond))

    monkeypatch.setattr(ui.app, "Agent", ChatAgent)
    monkeypatch.setenv("PERSONAL_HARNESS_UI_PERSIST", "1")
    store = ui.app.features.state()
    store.save_user("alice", {})
    monkeypatch.setattr(ui.app.features, "owner", lambda: "alice")
    monkeypatch.setattr(ui.app.features, "thread_id", lambda: None)

    async def exercise():
        init_http_context()
        await ui.app.message_received(SimpleNamespace(content="项目 Aurora"))
        store.save_personal_profile("alice", {"tone": "professional"})
        await ui.app.message_received(SimpleNamespace(content="接着讨论"))
        assert "项目 Aurora | 接着讨论" in ui.sent[-1]
        runtime = ui.session["harness_runtime"]
        assert runtime.agent.resolved.config.experience.tone == "professional"
        assert len(runtime.history) == 4
        await ui.app.message_received(SimpleNamespace(content="/clear"))
        assert not runtime.history and len(runtime.runs) == 2
        await ui.app.message_received(SimpleNamespace(content="新话题"))
        assert "Aurora" not in ui.sent[-1]
        assert len(runtime.history) == 2
        await ui.app.end()

    asyncio.run(exercise())


def test_insights_dashboard_and_actions_are_local_and_owner_scoped(ui, monkeypatch):
    from chainlit.context import init_http_context

    monkeypatch.setenv("PERSONAL_HARNESS_UI_PERSIST", "1")
    store = ui.app.features.state()
    store.save_user("alice", {})
    store.save_user("bob", {})
    monkeypatch.setattr(ui.app.features, "owner", lambda: "alice")
    monkeypatch.setattr(ui.app.features, "thread_id", lambda: None)
    monkeypatch.setattr(
        ui.app, "Agent", lambda *a, **k: pytest.fail("Local dashboard called a model")
    )

    async def exercise():
        init_http_context()
        await ui.app.message_received(SimpleNamespace(content="/insights"))
        element = ui.messages[-1]["elements"][0]
        assert element.name == "UserInsights" and element.props == {}
        action = SimpleNamespace(
            payload={
                "operation": "add",
                "owner": "bob",
                "memory": {"kind": "tools", "text": "I use Python"},
            }
        )
        added = await ui.app.user_insights(action)
        assert added["ok"] and action.payload == {}
        assert len(added["snapshot"]["memories"]) == 1
        service = ui.app.UserInsights(store)
        assert service.snapshot("bob")["memories"] == []
        bad = await ui.app.user_insights(
            SimpleNamespace(
                payload={
                    "operation": "add",
                    "memory": {"kind": "tools", "text": "sk-" + "test-private-key"},
                }
            )
        )
        assert not bad["ok"]
        await ui.app.user_insights(SimpleNamespace(payload={"operation": "clear"}))
        assert service.snapshot("alice")["memories"] == []

    asyncio.run(exercise())


def test_confirmed_user_memories_reach_next_run_and_delete_removes_guidance(ui, monkeypatch):
    from chainlit.context import init_http_context
    from pydantic_ai.messages import ModelResponse, TextPart
    from pydantic_ai.models.function import FunctionModel

    from personal_harness.user_insights import InsightOptions, MemoryInput, UserInsights

    monkeypatch.setenv("PERSONAL_HARNESS_UI_PERSIST", "1")
    store = ui.app.features.state()
    store.save_user("alice", {})
    monkeypatch.setattr(ui.app.features, "owner", lambda: "alice")
    monkeypatch.setattr(ui.app.features, "thread_id", lambda: None)
    knowledge = UserInsights(store)
    knowledge.configure("alice", InsightOptions(capture=True))
    seen = []

    def respond(messages, info):
        seen.append(info.instructions)
        return ModelResponse([TextPart("Done")])

    class ChatAgent(Agent):
        def __init__(self, resolved, **kwargs):
            super().__init__(resolved, model=FunctionModel(respond))

    monkeypatch.setattr(ui.app, "Agent", ChatAgent)

    async def exercise():
        init_http_context()
        await ui.app.message_received(SimpleNamespace(content="我习惯周五整理周报", id="first"))
        assert "User-confirmed background preferences" not in seen[-1]
        runtime = ui.session["harness_runtime"]
        agent = runtime.agent
        candidate = knowledge.snapshot("alice")["memories"][0]
        knowledge.change(
            "alice",
            candidate["id"],
            "save",
            MemoryInput(kind="workflow", text="我习惯周五整理周报"),
        )
        await ui.app.message_received(SimpleNamespace(content="整理项目周报", id="second"))
        assert '["我习惯周五整理周报"]' in seen[-1]
        assert ui.messages[-1]["elements"][0].props["preferences"] == 1
        knowledge.change("alice", candidate["id"], "delete")
        await ui.app.message_received(SimpleNamespace(content="继续整理周报", id="third"))
        assert "User-confirmed background preferences" not in seen[-1]
        assert runtime.agent is agent and len(runtime.history) == 6
        assert "user_context" not in ui.session.get("settings", {})
        await ui.app.end()

    asyncio.run(exercise())


def test_text_attachments_reach_model_with_size_and_binary_checks(ui, monkeypatch):
    from personal_harness import load_config

    path = ui.config.parent / "source.md"
    path.write_text("预算 3000 元", "utf-8")
    message = SimpleNamespace(
        content="总结资料", elements=[SimpleNamespace(path=str(path), name="source.md")]
    )
    resolved = load_config(ui.config)
    task = asyncio.run(ui.app.features.task_content(message, resolved))
    assert "预算 3000 元" in task and "source.md" in task
    assert "不要用工作区工具查找这些附件" in task
    message.content = " "
    with pytest.raises(ValueError, match="说明要如何处理附件"):
        asyncio.run(ui.app.features.task_content(message, resolved))
    message.content = "总结资料"
    path.write_bytes(b"\x00binary")
    with pytest.raises(ValueError, match="二进制"):
        asyncio.run(ui.app.features.task_content(message, resolved))
    path.write_bytes(b"x" * (resolved.config.workspace.max_read_bytes + 1))
    with pytest.raises(ValueError, match="读取上限"):
        asyncio.run(ui.app.features.task_content(message, resolved))


@pytest.mark.parametrize("decision", [True, False, None])
def test_oversized_previous_attachment_requires_confirmation_before_creating_agent(
    ui, monkeypatch, decision
):
    from chainlit.context import init_http_context
    from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
    from pydantic_ai.models.function import FunctionModel

    calls, prompts = [], []

    class ChatAgent(Agent):
        def __init__(self, resolved, **kwargs):
            calls.append("agent")
            super().__init__(resolved, model=FunctionModel(respond))

    def respond(messages, info):
        prompts.extend(
            part.content
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, UserPromptPart)
        )
        return ModelResponse([TextPart("仅本轮")])

    class Ask:
        def __init__(self, **kwargs):
            assert "上一轮内容" in kwargs["content"]
            assert "附件正文" in kwargs["content"]
            assert kwargs["actions"][0].payload["continue"] is False

        async def send(self):
            assert not calls
            return {"payload": {"continue": decision}} if decision is not None else None

    monkeypatch.setattr(ui.app, "Agent", ChatAgent)
    monkeypatch.setattr(ui.app.cl, "AskActionMessage", Ask)
    ui.session["settings"] = {"history_max_chars": "1000"}

    async def exercise():
        init_http_context()
        runtime = ui.app._ChatRuntime()
        runtime.history = [
            ModelRequest(
                [
                    UserPromptPart(
                        '<attachment name="old.pdf">' + "预算1200" * 1000 + "</attachment>"
                    )
                ]
            ),
            ModelResponse([TextPart("摘要")]),
        ]
        ui.session["harness_runtime"] = runtime
        await ui.app.message_received(SimpleNamespace(content="继续分析"))
        if decision is True:
            assert calls == ["agent"] and prompts == ["继续分析"]
            assert len(runtime.history) == 4 and len(runtime.runs) == 1
        else:
            assert not calls and not prompts and not runtime.runs
            assert len(runtime.history) == 2 and "未发送模型请求" in ui.sent[-1]
        await ui.app.end()

    asyncio.run(exercise())


def test_increasing_ui_history_capacity_restores_attachment_for_followup(ui, monkeypatch):
    from chainlit.context import init_http_context
    from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
    from pydantic_ai.models.function import FunctionModel

    def respond(messages, info):
        prompts = [
            p.content
            for m in messages
            if isinstance(m, ModelRequest)
            for p in m.parts
            if isinstance(p, UserPromptPart)
        ]
        assert len(prompts) == 2 and "<attachment" in prompts[0]
        return ModelResponse([TextPart("预算1200")])

    class ChatAgent(Agent):
        def __init__(self, resolved, **kwargs):
            assert resolved.config.runtime.history_max_chars == 100000
            super().__init__(resolved, model=FunctionModel(respond))

    monkeypatch.setattr(ui.app, "Agent", ChatAgent)
    ui.session["settings"] = {"history_max_chars": "100000"}

    async def exercise():
        init_http_context()
        runtime = ui.app._ChatRuntime()
        runtime.history = [
            ModelRequest(
                [
                    UserPromptPart(
                        '<attachment name="old.pdf">' + "预算1200" * 1000 + "</attachment>"
                    )
                ]
            ),
            ModelResponse([TextPart("摘要")]),
        ]
        ui.session["harness_runtime"] = runtime
        await ui.app.message_received(SimpleNamespace(content="继续分析"))
        assert ui.sent[-1].startswith("预算1200")
        assert len(runtime.history) == 4
        await ui.app.end()

    asyncio.run(exercise())


def test_pdf_attachment_is_extracted_before_model_context(ui):
    from io import BytesIO

    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    from personal_harness import load_config

    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = writer._add_object(
        DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
    )
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 72 720 Td (Trip budget 1200) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    source = BytesIO()
    writer.write(source)
    path = ui.config.parent / "trip.pdf"
    path.write_bytes(source.getvalue())
    message = SimpleNamespace(
        content="请读取 PDF 的预算", elements=[SimpleNamespace(path=str(path), name="trip.pdf")]
    )
    task = asyncio.run(ui.app.features.task_content(message, load_config(ui.config)))
    assert "[PDF 第 1 页]" in task
    assert "Trip budget 1200" in task


def test_chat_only_omits_tools_for_one_message_and_preserves_history(ui, monkeypatch):
    from pydantic_ai.messages import ModelResponse, TextPart
    from pydantic_ai.models.function import FunctionModel

    sent = []

    def respond(messages, info):
        sent.append((str(messages), [tool.name for tool in info.function_tools]))
        return ModelResponse([TextPart("Done")])

    class ChatAgent(Agent):
        def __init__(self, resolved, **kwargs):
            super().__init__(resolved, model=FunctionModel(respond))

    monkeypatch.setattr(ui.app, "Agent", ChatAgent)

    async def exercise():
        await ui.app.message_received(SimpleNamespace(content="first topic", command="仅聊天"))
        await ui.app.message_received(SimpleNamespace(content="/chat second topic"))
        await ui.app.message_received(SimpleNamespace(content="normal turn"))
        await ui.app.end()

    asyncio.run(exercise())
    assert [tools for _, tools in sent] == [[], [], ["context_describe"]]
    assert "first topic" in sent[1][0] and "second topic" in sent[2][0]


def test_online_progress_reports_waiting_and_tool_stage(ui, monkeypatch):
    class ProgressMessage:
        instances = []

        def __init__(self, content, **kwargs):
            self.content = content
            self.updates = []
            self.__class__.instances.append(self)

        async def send(self):
            return None

        async def update(self):
            self.updates.append(self.content)

    monkeypatch.setattr(ui.app.cl, "Message", ProgressMessage)

    async def exercise():
        reporter = ui.app._ProgressReporter(enabled=True)
        await reporter.start()
        await reporter.finish(SimpleNamespace(tool_calls=("workspace_read_many",)))
        return ProgressMessage.instances[0]

    message = asyncio.run(exercise())
    assert "执行进度" in message.updates[0]
    assert "不是模型隐藏思考内容" not in message.updates[0]
    assert "workspace_read_many" in message.updates[-1]


def test_online_progress_uses_collapsed_native_step(ui, monkeypatch):
    class ProgressStep:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.output = ""
            self.updates = []
            self.__class__.instances.append(self)

        async def send(self):
            return None

        async def update(self):
            self.updates.append(self.output)

    monkeypatch.setattr(ui.app.cl, "Step", ProgressStep)

    async def exercise():
        reporter = ui.app._ProgressReporter(enabled=True)
        await reporter.start()
        await reporter.finish(SimpleNamespace(tool_calls=()))
        return ProgressStep.instances[0]

    step = asyncio.run(exercise())
    assert step.kwargs["default_open"] is False
    assert step.kwargs["auto_collapse"] is True
    assert "已完成，模型已返回回复" in step.updates[-1]


def test_empty_optional_budget_remains_empty_after_reopening_settings(ui, monkeypatch):
    from chainlit.context import init_http_context

    from personal_harness import load_config

    original = ui.app.cl.ChatSettings

    class Settings(original):
        async def send(self):
            return self.settings()

    monkeypatch.setattr(ui.app.cl, "ChatSettings", Settings)

    async def exercise():
        init_http_context()
        values = await ui.app.features.settings_form(
            load_config(ui.config), {"max_total_cny": None, "personal_instructions": None}
        )
        assert values["max_total_cny"] == ""
        assert values["experience_personal_instructions"] == ""
        resolved, _ = ui.app.features.resolve_settings(values, None)
        assert resolved.config.budget.max_total_cny is None
        values["max_total_cny"] = None
        again = await ui.app.features.settings_form(resolved, values)
        assert again["max_total_cny"] == ""

    asyncio.run(exercise())


def test_history_read_mode_is_editable_and_survives_form_reopen_without_model_calls(
    ui, monkeypatch
):
    from chainlit.context import init_http_context

    from personal_harness import load_config

    original = ui.app.cl.ChatSettings

    class Settings(original):
        async def send(self):
            return self.settings()

    monkeypatch.setattr(ui.app.cl, "ChatSettings", Settings)
    monkeypatch.setattr(ui.app, "Agent", lambda *a, **k: pytest.fail("Settings called a model"))

    async def exercise():
        init_http_context()
        values = await ui.app.features.settings_form(
            load_config(ui.config), {"history_read_mode": "references"}
        )
        assert values["history_read_mode"] == "references"
        resolved, _ = ui.app.features.resolve_settings(values, None)
        assert resolved.config.runtime.history_read_mode == "references"
        await ui.app.settings_update({"history_read_mode": "full"})
        assert ui.session["settings"]["history_read_mode"] == "full"
        reopened = await ui.app.features.settings_form(resolved, ui.session["settings"])
        assert reopened["history_read_mode"] == "full"
        with pytest.raises(ValueError):
            ui.app.features.resolve_settings({"history_read_mode": "unknown"}, None)

    asyncio.run(exercise())


def test_read_page_limit_survives_tool_permissions_and_settings_reopen(ui, monkeypatch):
    from chainlit.context import init_http_context

    from personal_harness import load_config

    original = ui.app.cl.ChatSettings

    class Settings(original):
        async def send(self):
            return self.settings()

    monkeypatch.setattr(ui.app.cl, "ChatSettings", Settings)
    monkeypatch.setattr(ui.app, "Agent", lambda *a, **k: pytest.fail("Settings called a model"))

    async def exercise():
        init_http_context()
        settings = {"max_read_result_chars": "8000", "tool_permissions": ["workspace.read"]}
        values = await ui.app.features.settings_form(load_config(ui.config), settings)
        resolved, _ = ui.app.features.resolve_settings(values, None)
        assert resolved.config.tools.max_read_result_chars == 8000
        assert resolved.config.tools.allowed == ("workspace.read",)
        await ui.app.settings_update(settings)
        reopened = await ui.app.features.settings_form(resolved, ui.session["settings"])
        assert reopened["max_read_result_chars"] == "8000"
        for invalid in ("-1", "2000001", "abc"):
            with pytest.raises(ValueError):
                ui.app.features.resolve_settings({"max_read_result_chars": invalid}, None)

    asyncio.run(exercise())


def test_account_profile_settings_persist_and_override_each_chat(ui, monkeypatch):
    monkeypatch.setenv("PERSONAL_HARNESS_UI_PERSIST", "1")
    store = ui.app.features.state()
    store.save_user("alice", {})
    monkeypatch.setattr(ui.app.features, "owner", lambda: "alice")
    monkeypatch.setattr(ui.app.features, "thread_id", lambda: None)
    ui.session.update(
        profile="desktop",
        settings={
            "experience_preferred_name": "",
            "language": "en",
            "response_style": "detailed",
            "experience_tone": "warm",
            "experience_verbosity": "brief",
            "experience_answer_format": "steps",
            "experience_clarification": "minimize",
            "experience_initiative": False,
            "experience_cite_sources": False,
            "experience_cost_strategy": "balanced",
            "experience_personal_instructions": "Start with the result",
            "max_task_cny": "0.2",
            "tools": [],
        },
    )
    asyncio.run(
        ui.app.settings_update(
            {
                "experience_preferred_name": "Lin",
                "experience_tone": "professional",
                "experience_initiative": True,
            }
        )
    )
    profile = store.personal_profile("alice")
    assert profile["tone"] == "professional" and profile["initiative"] is True
    assert "max_task_cny" not in profile and "tools" not in profile
    assert ui.app.features.apply_personal_profile({}, profile)["experience_tone"] == "professional"
    resolved, _ = ui.app.features.resolve_settings(ui.session["settings"], "desktop")
    assert resolved.config.experience.tone == "professional"
    assert resolved.config.experience.preferred_name == "Lin"
    assert resolved.config.experience.personal_instructions == "Start with the result"
    assert resolved.config.budget.max_task_cny == Decimal("0.2")

    # A second tab sends the entire stale form when changing a chat-only setting.
    async def settings_form(_, values):
        return values

    monkeypatch.setattr(ui.app.features, "settings_form", settings_form)
    stale = {**ui.session["settings"], "experience_tone": "warm"}
    ui.session["settings"] = stale
    asyncio.run(ui.app.settings_update({**stale, "max_task_cny": "0.3"}))
    assert store.personal_profile("alice") == profile
    assert ui.session["settings"]["experience_tone"] == "professional"
    asyncio.run(ui.app.settings_update({"experience_personal_instructions": None}))
    assert store.personal_profile("alice")["personal_instructions"] == ""
    cleared, _ = ui.app.features.resolve_settings(ui.session["settings"], "desktop")
    assert cleared.config.experience.personal_instructions == ""


def test_resume_form_and_chainlit_metadata_share_latest_account_preferences(ui, monkeypatch):
    monkeypatch.setenv("PERSONAL_HARNESS_UI_PERSIST", "1")
    store = ui.app.features.state()
    store.save_user("alice", {})
    store.save_personal_profile("alice", {"preferred_name": "Lin", "tone": "professional"})
    monkeypatch.setattr(ui.app.features, "owner", lambda: "alice")

    async def settings_form(_, values):
        return values

    async def install_commands():
        pass

    monkeypatch.setattr(ui.app.features, "settings_form", settings_form)
    monkeypatch.setattr(ui.app.features, "install_commands", install_commands)
    thread = {
        "id": "old-chat",
        "metadata": {
            "profile": "desktop",
            "chat_settings": {"experience_preferred_name": "Old", "experience_tone": "warm"},
            "settings": {"experience_tone": "warm", "max_task_cny": "0.2"},
        },
    }
    asyncio.run(ui.app.resume_chat(thread))
    current = ui.session["settings"]
    assert current["experience_preferred_name"] == "Lin"
    assert current["experience_tone"] == "professional"
    assert current["max_task_cny"] == "0.2"
    assert thread["metadata"]["settings"] == thread["metadata"]["chat_settings"] == current


def test_local_life_settings_parse_and_reject_invalid_values(ui):
    values = {
        "places_latitude": "31.8",
        "places_longitude": "117.2",
        "places_max_distance_m": "2000",
        "places_max_price_cny": "40",
        "places_query_cache_seconds": "90",
        "read_concurrency": "2",
    }
    resolved, key = ui.app.features.resolve_settings(values, "mobile")
    assert key is None
    assert resolved.config.local_life.latitude == 31.8
    assert resolved.config.local_life.query_cache_seconds == 90
    assert resolved.config.local_life.max_price_cny == Decimal("40")
    assert resolved.config.runtime.read_concurrency == 2
    for patch in (
        {"places_latitude": "100"},
        {"places_query_cache_seconds": "3601"},
        {"places_max_distance_m": "1.5"},
        {"read_concurrency": "0"},
    ):
        with pytest.raises(ValueError):
            ui.app.features.resolve_settings({**values, **patch}, "mobile")
    cleared, _ = ui.app.features.resolve_settings({**values, "places_max_price_cny": ""}, "mobile")
    assert cleared.config.local_life.max_price_cny is None


def test_settings_apply_to_run_and_do_not_modify_config(ui: Any) -> None:
    original = ui.config.read_bytes()
    ui.session.update(profile="mobile", settings={"city": "Hangzhou", "response_style": "detailed"})
    asyncio.run(ui.app.message_received(SimpleNamespace(content="Inspect preferences")))
    assert "Hangzhou" in ui.sent[-1]
    assert "Response style: detailed" in ui.sent[-1]
    assert ui.config.read_bytes() == original


def test_ui_budget_command_reports_offline_profile_without_creating_agent(ui: Any) -> None:
    asyncio.run(ui.app.message_received(SimpleNamespace(content="/budget")))
    assert "Offline provider; no paid request budget applies." in ui.sent[-1]
    assert ui.session["harness_runtime"].agent is None
    assert not (ui.config.parent / ".personal-harness").exists()


def test_local_queries_use_settings_and_skip_model_budget_and_history(ui, monkeypatch):
    from chainlit.context import init_http_context

    data = {
        "schema_version": 1,
        "source": "UI test fixture",
        "fetched_at": "2000-01-01T00:00:00Z",
        "city": "合肥",
        "places": [
            {
                "id": str(price),
                "name": name,
                "price_cny": price,
                "category": "餐厅",
                "latitude": 31.82,
                "longitude": 117.23,
            }
            for price, name in [(12, "示例饭馆"), (25, "示例面馆")]
        ],
    }
    (ui.config.parent / "places.json").write_text(json.dumps(data), "utf-8")
    (ui.config.parent / "notes.md").write_text("周会\n项目期限：星期五\n", "utf-8")
    ui.session["settings"] = {
        "connection": "deleted-model-connection",
        "fallback_connection": "deleted-fallback-connection",
        "city": "合肥",
        "places_data_file": "places.json",
        "places_latitude": "31.82",
        "places_longitude": "117.23",
        "places_max_price_cny": "15",
        "tool_permissions": ["workspace.search", "places.search"],
        "tools": [],
    }
    monkeypatch.setattr(ui.app, "Agent", lambda *a, **k: pytest.fail("Local query invoked Agent"))

    class Store:
        def __init__(self):
            self.data = {"name": "workspace_tool"}
            self.updated = None

        def thread(self, identity):
            return self.data

        def personal_profile(self, owner):
            return {}

        def update_thread(self, identity, name=None, **kwargs):
            self.updated = (identity, name)

        def record_run(self, identity, run):
            assert identity == "chat-1" and run["execution"] == "local"

    store = Store()
    monkeypatch.setattr(ui.app.features, "state", lambda: store)
    monkeypatch.setattr(ui.app.features, "thread_id", lambda: "chat-1")
    monkeypatch.setattr(ui.app.features, "owner", lambda: "alice")

    async def exercise():
        init_http_context()
        await ui.app.workspace_tool(
            SimpleNamespace(payload={"command": "/find", "query": "项目期限"})
        )
        result = ui.messages[-1]["elements"][0]
        assert result.name == "LocalResults"
        assert result.props["matches"] == [
            {"path": "notes.md", "line": 2, "text": "项目期限：星期五"}
        ]
        assert store.updated == ("chat-1", "新聊天")
        await ui.app.message_received(SimpleNamespace(content="/places 餐厅"))
        result = ui.messages[-1]["elements"][0].props
        assert [item["name"] for item in result["places"]] == ["示例饭馆"]
        assert result["stale"] and not result["current_status_verified"]
        assert "0 次模型调用" in ui.messages[-1]["metadata"]["harness_receipt"]
        await ui.app.message_received(SimpleNamespace(content="/places 咖啡"))
        assert ui.messages[-1]["elements"][0].props["places"] == []

        ui.session["settings"]["tool_permissions"] = []
        await ui.app.message_received(SimpleNamespace(content="/find 项目期限"))
        assert "启用 workspace.search" in ui.sent[-1]
        await ui.app.message_received(SimpleNamespace(content="/places 餐厅"))
        assert "启用 places.search" in ui.sent[-1]

    asyncio.run(exercise())
    runtime = ui.session["harness_runtime"]
    assert runtime.agent is None and len(runtime.runs) == 3 and not runtime.history
    assert all(run["execution"] == "local" and run["requests"] == 0 for run in runtime.runs)
    assert not (ui.config.parent / ".personal-harness").exists()


def test_personal_report_recipe_writes_only_after_preview_and_explicit_approval(ui, monkeypatch):
    from chainlit.context import init_http_context

    monkeypatch.setenv("PERSONAL_HARNESS_UI_PERSIST", "1")
    store = ui.app.features.state()
    store.save_user("alice", {})
    monkeypatch.setattr(ui.app.features, "owner", lambda: "alice")
    workspace = ui.config.parent
    (workspace / "notes.md").write_text("TODO: 核对报告\n负责人：林\n", "utf-8")
    monkeypatch.setattr(
        ui.app,
        "Agent",
        lambda *args, **kwargs: pytest.fail("A local report must not create a model agent"),
    )
    ui.session["settings"] = {
        "tool_permissions": ["workspace.read", "workspace.write"],
        "tools": [],
    }

    async def exercise():
        init_http_context()
        created = await ui.app.report_recipe(
            SimpleNamespace(
                payload={
                    "operation": "create",
                    "name": "Weekly brief",
                    "sources": ["notes.md"],
                    "target": "weekly-brief.md",
                    "sections": [
                        {"heading": "Tasks", "prefix": "TODO:"},
                        {"heading": "Owners", "prefix": "负责人："},
                    ],
                }
            )
        )
        assert created["ok"] and len(created["recipes"]) == 1
        identity = created["id"]
        preview = await ui.app.report_recipe(
            SimpleNamespace(payload={"operation": "preview", "id": identity})
        )
        assert preview["ok"] and preview["preview"]["model_requests"] == 0
        assert "notes.md:1" in preview["preview"]["content"]
        assert not (workspace / "weekly-brief.md").exists()
        result = await ui.app.report_recipe(
            SimpleNamespace(
                payload={"operation": "apply", "id": identity, "token": preview["token"]}
            )
        )
        assert result["ok"] and "模型请求 0 次" in result["message"]
        assert (workspace / "weekly-brief.md").read_text("utf-8") == preview["preview"]["content"]
        assert result["recipes"][0]["runs"] == 1

        same = await ui.app.report_recipe(
            SimpleNamespace(payload={"operation": "preview", "id": identity})
        )
        assert same["ok"] and same["preview"]["unchanged"]
        assert same["preview"]["cost_cny"] == "0"

        (workspace / "notes.md").write_text("TODO: 预览阶段的计划\n负责人：林\n", "utf-8")
        stale = await ui.app.report_recipe(
            SimpleNamespace(payload={"operation": "preview", "id": identity})
        )
        assert stale["ok"] and not stale["preview"]["unchanged"]
        (workspace / "notes.md").write_text("TODO: 改过的计划\n负责人：林\n", "utf-8")
        result = await ui.app.report_recipe(
            SimpleNamespace(payload={"operation": "apply", "id": identity, "token": stale["token"]})
        )
        assert not result["ok"] and "请重新生成并确认预览" in result["message"]
        updated_content = (workspace / "weekly-brief.md").read_text("utf-8")
        assert "核对报告" in updated_content
        assert "改过的计划" not in updated_content
        assert store.report_recipes("alice")[0]["runs"] == 1

        rejected = await ui.app.report_recipe(
            SimpleNamespace(payload={"operation": "apply", "id": identity, "token": stale["token"]})
        )
        assert not rejected["ok"] and "预览已过期" in rejected["message"]

        fresh = await ui.app.report_recipe(
            SimpleNamespace(payload={"operation": "preview", "id": identity})
        )
        ui.session["settings"]["tool_permissions"] = ["workspace.read"]
        blocked = await ui.app.report_recipe(
            SimpleNamespace(payload={"operation": "apply", "id": identity, "token": fresh["token"]})
        )
        assert not blocked["ok"] and "授权" in blocked["message"]
        assert (workspace / "weekly-brief.md").read_text("utf-8") == updated_content
        local_runs = store.runs(ui.app.features.thread_id())
        assert len(local_runs) == 1
        assert local_runs[0]["operation"] == "report" and local_runs[0]["status"] == "written"

    asyncio.run(exercise())
    assert ui.session["harness_runtime"].agent is None
    assert not (workspace / ".personal-harness/ledger.sqlite3").exists()


def test_csv_lookup_with_toolbox_and_command_has_no_model_or_ledger(ui, monkeypatch):
    from chainlit.context import init_http_context

    (ui.config.parent / "my expenses.csv").write_text(
        "发票号,金额\nINV-1,12.50\nINV-11,60.00\n", encoding="utf-8"
    )
    ui.session["settings"] = {"tool_permissions": ["workspace.read"], "connection": "bad-id"}
    monkeypatch.setattr(ui.app, "Agent", lambda *a, **k: pytest.fail("CSV lookup invoked a model"))

    async def exercise():
        init_http_context()
        await ui.app.workspace_tool(
            SimpleNamespace(
                payload={
                    "command": "/lookup",
                    "path": "my expenses.csv",
                    "column": "发票号",
                    "value": "INV-1",
                }
            )
        )
        props = ui.messages[-1]["elements"][0].props
        assert props["csv"] and props["match_count"] == 1
        assert props["rows"][0]["values"] == ["INV-1", "12.50"]
        await ui.app.message_received(
            SimpleNamespace(content='/lookup "my expenses.csv" 发票号 missing')
        )
        assert ui.messages[-1]["elements"][0].props["match_count"] == 0
        ui.session["settings"]["tool_permissions"] = []
        await ui.app.message_received(
            SimpleNamespace(content='/lookup "my expenses.csv" 发票号 INV-1')
        )
        assert "workspace.read" in ui.sent[-1]
        await ui.app.message_received(SimpleNamespace(content="/usage"))
        assert "CSV 查找 2 次" in ui.sent[-1] and "返回 0 项" in ui.sent[-1]
        assert "INV-1" not in ui.sent[-1]

    asyncio.run(exercise())
    assert not (ui.config.parent / ".personal-harness/ledger.sqlite3").exists()


def test_csv_summary_with_toolbox_and_command_has_no_model_or_ledger(ui, monkeypatch):
    from chainlit.context import init_http_context

    (ui.config.parent / "expenses.csv").write_text(
        "id,status,amount\nA,待报销,847.50\nB,已报销,12.50\nC,待报销,2\n", encoding="utf-8"
    )
    ui.session["settings"] = {"tool_permissions": ["workspace.read"], "connection": "bad-id"}
    monkeypatch.setattr(ui.app, "Agent", lambda *a, **k: pytest.fail("CSV summary invoked a model"))

    async def exercise():
        init_http_context()
        await ui.app.workspace_tool(
            SimpleNamespace(
                payload={
                    "command": "/sum",
                    "path": "expenses.csv",
                    "numeric_column": "amount",
                    "filter_column": "status",
                    "filter_value": "待报销",
                }
            )
        )
        props = ui.messages[-1]["elements"][0].props
        assert props["summary"] and props["matched_count"] == 2
        assert props["total"] == "849.50" and props["average"] == "424.75"
        await ui.app.message_received(SimpleNamespace(content='/sum "expenses.csv" amount'))
        assert ui.messages[-1]["elements"][0].props["matched_count"] == 3
        ui.session["settings"]["tool_permissions"] = []
        await ui.app.message_received(SimpleNamespace(content='/sum "expenses.csv" amount'))
        assert "workspace.read" in ui.sent[-1]
        assert not (ui.config.parent / ".personal-harness/ledger.sqlite3").exists()

    asyncio.run(exercise())


def test_marked_action_extraction_from_chat_and_toolbox_is_local(ui, monkeypatch):
    from chainlit.context import init_http_context

    workspace = ui.config.parent / "action-workspace"
    workspace.mkdir()
    source = workspace / "activity-log.md"
    source.write_text(
        "TODO: 行动000，负责人王青，截止12月1日。\n"
        "归档001：已完成，无需跟进。\n"
        + "历史归档：已完成。\n" * 10
        + "TODO: 行动649，负责人王青，截止12月10日。",
        encoding="utf-8",
    )
    ui.session["settings"] = {
        "workspace": str(workspace),
        "tool_permissions": ["workspace.read"],
        "connection": "missing-online-model",
    }
    monkeypatch.setattr(
        ui.app, "Agent", lambda *a, **k: pytest.fail("Action extraction called a model")
    )

    async def exercise():
        init_http_context()
        await ui.app.message_received(SimpleNamespace(content='/actions "activity-log.md" "TODO:"'))
        props = ui.messages[-1]["elements"][0].props
        assert props["match_count"] == 2
        assert [(item["line"], item["text"]) for item in props["actions"]] == [
            (1, "TODO: 行动000，负责人王青，截止12月1日。"),
            (13, "TODO: 行动649，负责人王青，截止12月10日。"),
        ]
        assert ui.session["harness_runtime"].runs[-1]["operation"] == "actions"
        assert ui.session["harness_runtime"].runs[-1]["cost"] == "0"

        await ui.app.workspace_tool(
            SimpleNamespace(
                payload={
                    "command": "/actions",
                    "path": "activity-log.md",
                    "prefix": "TODO:",
                }
            )
        )
        assert ui.messages[-1]["elements"][0].props["match_count"] == 2
        ui.session["settings"]["tool_permissions"] = []
        await ui.app.message_received(SimpleNamespace(content="/actions activity-log.md TODO:"))
        assert "workspace.read" in ui.sent[-1]
        assert not (ui.config.parent / ".personal-harness/ledger.sqlite3").exists()

    asyncio.run(exercise())


def test_saved_shortcuts_recheck_permissions_and_current_data_without_model(ui, monkeypatch):
    from chainlit.context import init_http_context

    monkeypatch.setenv("PERSONAL_HARNESS_UI_PERSIST", "1")
    monkeypatch.setattr(ui.app.features, "owner", lambda: "alice")
    monkeypatch.setattr(ui.app.features, "thread_id", lambda: None)
    monkeypatch.setattr(ui.app, "Agent", lambda *a, **k: pytest.fail("Shortcut called model"))
    ui.session["profile"] = "desktop"
    ui.session["settings"] = {"tool_permissions": ["workspace.read"], "connection": "deleted"}
    path = ui.config.parent / "expenses.csv"
    path.write_text("id,amount\nINV-1,10\n", encoding="utf-8")

    async def action(**payload):
        return await ui.app.local_shortcut(SimpleNamespace(payload=payload))

    async def exercise():
        init_http_context()
        saved = await action(
            operation="save",
            shortcut={"name": "报销", "command": "/lookup", "args": [path.name, "id", "INV-1"]},
        )
        identity = saved["items"][0]["id"]
        await action(operation="run", id=identity)
        assert ui.messages[-1]["elements"][0].props["rows"][0]["values"] == ["INV-1", "10"]
        path.write_text("id,amount\nINV-1,25\n", encoding="utf-8")
        await action(operation="run", id=identity)
        assert ui.messages[-1]["elements"][0].props["rows"][0]["values"] == ["INV-1", "25"]
        ui.session["settings"]["tool_permissions"] = []
        await action(operation="run", id=identity)
        assert "workspace.read" in ui.sent[-1]
        await ui.app.message_received(SimpleNamespace(content="/tools"))
        assert ui.messages[-1]["elements"][0].name == "WorkspaceWelcome"
        ui.session["profile"] = "mobile"
        assert (await action(operation="list"))["items"] == []
        assert not (await action(operation="run", id=identity))["ok"]
        ui.session["profile"] = "desktop"
        assert (await action(operation="delete", id=identity))["items"] == []
        assert not (await action(operation="run", id=identity))["ok"]
        assert len(ui.session["harness_runtime"].runs) == 2

    asyncio.run(exercise())
    assert not (ui.config.parent / ".personal-harness/ledger.sqlite3").exists()


def test_local_usage_survives_reconnect_without_model_connection_or_queries_in_records(
    ui, monkeypatch
):
    from chainlit.context import init_http_context

    monkeypatch.setenv("PERSONAL_HARNESS_UI_PERSIST", "1")
    store = ui.app.features.state()
    user = store.save_user("alice", {})
    store.update_thread("mine", "本地任务", user["id"])
    monkeypatch.setattr(ui.app.features, "owner", lambda: "alice")
    monkeypatch.setattr(ui.app.features, "thread_id", lambda: "mine")
    monkeypatch.setattr(ui.app, "Agent", lambda *a, **k: pytest.fail("Usage called a model"))
    ui.config.write_text(
        ui.config.read_text("utf-8").replace("max_search_results = 50", "max_search_results = 1"),
        "utf-8",
    )
    (ui.config.parent / "notes.md").write_text(
        "private-query-marker\nprivate-query-marker\n", "utf-8"
    )
    ui.session["settings"] = {
        "tool_permissions": ["workspace.search"],
        "connection": "deleted-connection",
        "fallback_connection": "deleted-fallback",
    }

    async def exercise():
        init_http_context()
        await ui.app.message_received(SimpleNamespace(content="/find private-query-marker"))
        await ui.app.message_received(SimpleNamespace(content="/find no-such-marker"))
        recorded = store.runs("mine")
        assert len(recorded) == 2 and all(r["requests"] == 0 and r["cost"] == "0" for r in recorded)
        assert "private-query-marker" not in json.dumps(recorded)
        assert "notes.md" not in json.dumps(recorded)
        assert recorded[0]["truncated"] and recorded[1]["result_count"] == 0
        ui.session["harness_runtime"] = ui.app._ChatRuntime()
        # Viewing usage should work even with malformed current settings.
        ui.session["settings"]["max_read_bytes"] = "invalid"
        await ui.app.message_received(SimpleNamespace(content="/usage"))
        assert "文件搜索 2 次" in ui.sent[-1] and "返回 0 项" in ui.sent[-1]
        assert "扫描或结果不完整" in ui.sent[-1] and "API 费用 ¥0" in ui.sent[-1]
        assert "费用未完整确认的轮次：0" in ui.sent[-1]
        assert store.runs("another") == [] and len(store.runs("mine")) == 2
        store.delete_thread("mine")
        assert store.runs("mine") == []

    asyncio.run(exercise())
    assert not (ui.config.parent / ".personal-harness/ledger.sqlite3").exists()


def test_local_results_remain_visible_when_usage_persistence_fails(ui, monkeypatch):
    from chainlit.context import init_http_context

    monkeypatch.setenv("PERSONAL_HARNESS_UI_PERSIST", "1")
    store = ui.app.features.state()
    store.save_user("alice", {})
    monkeypatch.setattr(ui.app.features, "owner", lambda: "alice")
    (ui.config.parent / "notes.md").write_text("可查的内容\n", "utf-8")
    ui.session["settings"] = {"tool_permissions": ["workspace.search"]}

    def fail(*args):
        import sqlite3

        raise sqlite3.OperationalError("disk full")

    monkeypatch.setattr(store, "record_run", fail)

    async def exercise():
        init_http_context()
        await ui.app.message_received(SimpleNamespace(content="/find 可查"))
        assert ui.messages[-2]["elements"][0].props["matches"][0]["text"] == "可查的内容"
        assert "结果已展示" in ui.sent[-1] and "未调用模型" in ui.sent[-1]
        assert len(ui.session["harness_runtime"].runs) == 1
        await ui.app.message_received(SimpleNamespace(content="/usage"))
        assert "文件搜索 1 次" in ui.sent[-1] and "关闭后可能丢失" in ui.sent[-1]

    asyncio.run(exercise())


@pytest.mark.parametrize("ledger_state", ["missing", "corrupt", "unsupported"])
def test_usage_keeps_snapshots_and_local_counts_when_ledger_cannot_be_read(
    ui, monkeypatch, ledger_state
):
    import sqlite3

    from chainlit.context import init_http_context

    ledger_path = ui.config.parent / ".personal-harness/ledger.sqlite3"
    if ledger_state != "missing":
        ledger_path.parent.mkdir()
        if ledger_state == "corrupt":
            ledger_path.write_bytes(b"not a database")
        else:
            with sqlite3.connect(ledger_path) as db:
                db.execute("PRAGMA user_version=99")
    before = ledger_path.read_bytes() if ledger_path.exists() else None
    monkeypatch.setattr(ui.app, "Agent", lambda *a, **k: pytest.fail("Usage invoked Agent"))
    ui.session["settings"] = {"connection": "deleted"}

    async def exercise():
        init_http_context()
        runtime = ui.app._ChatRuntime()
        runtime.runs = [
            {
                "run_id": "old-model-run",
                "execution": "model",
                "currency": "CNY",
                "status": "completed",
                "cost": "0.1",
                "cache_read_tokens": 3,
                "cache_savings_cny": "0.0000027",
            }
        ]
        ui.session["harness_runtime"] = runtime
        assert ui.app.features.record_local_operation(runtime, "search", result_count=0)
        await ui.app.message_received(SimpleNamespace(content="/usage"))
        assert "上次保存的快照" in ui.sent[-1]
        assert "CNY 已知小计：0.1" in ui.sent[-1] and "文件搜索 1 次" in ui.sent[-1]
        assert "供应商缓存价差：约 ¥0.0000027" in ui.sent[-1]
        assert "不据此推算省下的金额" in ui.sent[-1]

    asyncio.run(exercise())
    if before is None:
        assert not ledger_path.parent.exists()
    else:
        assert ledger_path.read_bytes() == before


def test_personal_recipe_marker_discovery_is_read_only_current_and_never_connects(ui, monkeypatch):
    from chainlit.context import init_http_context

    monkeypatch.setenv("PERSONAL_HARNESS_UI_PERSIST", "1")
    store = ui.app.features.state()
    store.save_user("alice", {})
    monkeypatch.setattr(ui.app.features, "owner", lambda: "alice")
    monkeypatch.setattr(ui.app, "Agent", lambda *a, **k: pytest.fail("Discovery called a model"))
    source = ui.config.parent / "notes.md"
    source.write_text("TODO: 核对预算\n风险：资料缺失\n", "utf-8")
    ui.session["settings"] = {"tool_permissions": ["workspace.read"], "tools": []}

    async def exercise():
        init_http_context()
        action = SimpleNamespace(payload={"operation": "inspect", "sources": ["notes.md"]})
        result = await ui.app.report_recipe(action)
        assert result["ok"] and action.payload == {}
        assert result["discovery"]["candidates"][0]["example"] == "核对预算"
        assert result["discovery"]["model_requests"] == 0
        assert store.report_recipes("alice") == []
        source.write_text("TODO: 更新后的材料\n", "utf-8")
        result = await ui.app.report_recipe(
            SimpleNamespace(payload={"operation": "inspect", "sources": ["notes.md"]})
        )
        assert result["discovery"]["candidates"][0]["example"] == "更新后的材料"
        for sources in ([], ["notes.md"] * 9, ["../notes.md"], ["absent.md"]):
            result = await ui.app.report_recipe(
                SimpleNamespace(payload={"operation": "inspect", "sources": sources})
            )
            assert not result["ok"]
        ui.session["settings"]["tool_permissions"] = []
        result = await ui.app.report_recipe(
            SimpleNamespace(payload={"operation": "inspect", "sources": ["notes.md"]})
        )
        assert not result["ok"] and "授权" in result["message"]

    asyncio.run(exercise())
    assert ui.session["harness_runtime"].agent is None
    assert not (ui.config.parent / ".personal-harness/ledger.sqlite3").exists()


@pytest.mark.parametrize("approved", [True, False, None])
def test_ui_approval_or_timeout_controls_actual_file_write(
    ui: Any, approved: bool | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    ui.config.write_text(
        ui.config.read_text("utf-8")
        + (
            "\n[profiles.cowork.tools]\n"
            'allowed = ["workspace.read", "workspace.write_preview", "workspace.write"]\n'
        ),
        "utf-8",
    )
    (ui.config.parent / "meeting notes.md").write_text("TODO: configuration first\n", "utf-8")
    ui.session["profile"] = "cowork"
    asked: list[dict[str, Any]] = []

    class Ask:
        def __init__(self, **kwargs: Any) -> None:
            asked.append(kwargs)

        async def send(self) -> dict[str, Any] | None:
            return {"payload": {"approved": approved}} if approved is not None else None

    monkeypatch.setattr(ui.app.cl, "AskActionMessage", Ask)

    async def exercise() -> None:
        from chainlit.context import init_http_context

        init_http_context()
        await ui.app.message_received(
            SimpleNamespace(content='/workspace-demo "meeting notes.md" brief.md')
        )

    asyncio.run(exercise())
    assert len(asked) == 1
    assert "brief.md" in asked[0]["content"]
    assert "elements" not in asked[0]
    preview = next(message for message in ui.messages if "elements" in message)
    assert len(preview["elements"]) == 2
    assert (ui.config.parent / "brief.md").exists() is (approved is True)


@pytest.mark.parametrize("approved", [True, False, None])
def test_ui_constructs_real_chainlit_preview_and_approval(ui, monkeypatch, approved):
    from chainlit.context import init_http_context
    from chainlit.message import AskActionMessage, Message

    ui.config.write_text(
        ui.config.read_text("utf-8")
        + (
            "\n[profiles.cowork.tools]\n"
            'allowed = ["workspace.read", "workspace.write_preview", "workspace.write"]\n'
        ),
        "utf-8",
    )
    (ui.config.parent / "notes.md").write_text("TODO: Review the actual preview\n", "utf-8")
    ui.session["profile"] = "cowork"
    messages = []
    questions = []

    async def send_message(self):
        messages.append(self)

    async def send_question(self):
        questions.append(self)
        # The complete proposal must be sent successfully before requesting a decision.
        assert len([message for message in messages if message.elements]) == 1
        preview = next(message for message in messages if message.elements)
        assert len(preview.elements) == 2
        assert "Review the actual preview" in preview.elements[0].content
        assert "+" in preview.elements[1].content
        assert not (ui.config.parent / "brief.md").exists()
        return {"payload": {"approved": approved}} if approved is not None else None

    monkeypatch.setattr(ui.app.cl, "Message", Message)
    monkeypatch.setattr(ui.app.cl, "AskActionMessage", AskActionMessage)
    monkeypatch.setattr(Message, "send", send_message)
    monkeypatch.setattr(AskActionMessage, "send", send_question)

    async def exercise():
        init_http_context()
        await ui.app.message_received(SimpleNamespace(content="/workspace-demo notes.md brief.md"))
        await ui.app.end()

    asyncio.run(exercise())
    assert len(questions) == 1 and "brief.md" in questions[0].content
    assert [action.payload for action in questions[0].actions] == [
        {"approved": True},
        {"approved": False},
    ]
    assert (ui.config.parent / "brief.md").exists() is (approved is True)


@pytest.mark.parametrize("inside", [False, True])
def test_ui_cowork_obeys_configured_write_directory(ui, monkeypatch, inside):
    ui.config.write_text(
        ui.config.read_text("utf-8")
        + (
            '\n[profiles.cowork.workspace]\nwrite_directory = "out"\n'
            "\n[profiles.cowork.tools]\n"
            'allowed = ["workspace.read", "workspace.write_preview", "workspace.write"]\n'
        ),
        "utf-8",
    )
    (ui.config.parent / "out").mkdir()
    (ui.config.parent / "notes.md").write_text("TODO: Keep sources unchanged\n", "utf-8")
    ui.session["profile"] = "cowork"
    asked = []

    class Ask:
        def __init__(self, **kwargs):
            asked.append(kwargs)

        async def send(self):
            return {"payload": {"approved": True}}

    monkeypatch.setattr(ui.app.cl, "AskActionMessage", Ask)
    target = "out/brief.md" if inside else "brief.md"

    async def exercise():
        from chainlit.context import init_http_context

        init_http_context()
        await ui.app.message_received(SimpleNamespace(content=f"/workspace-demo notes.md {target}"))
        await ui.app.end()

    asyncio.run(exercise())
    assert len(asked) == (1 if inside else 0)
    assert (ui.config.parent / target).exists() is inside
    assert (ui.config.parent / "notes.md").read_text("utf-8") == "TODO: Keep sources unchanged\n"


@pytest.fixture
def recording_agents(ui: Any, monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        TextPart,
        ToolCallPart,
        ToolReturnPart,
    )
    from pydantic_ai.models.function import FunctionModel

    def respond(messages, info):
        if not any(tool.name == "workspace_read" for tool in info.function_tools):
            return ModelResponse([TextPart("No file tools selected")])
        returns = [
            part
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        if returns:
            # No message history should accumulate across tasks.
            assert len(returns) == 1
            return ModelResponse([TextPart(json.loads(returns[0].content)["content"])])
        return ModelResponse([ToolCallPart("workspace_read", {"relative_path": "notes.md"})])

    instances = []

    class RecordingAgent(Agent):
        def __init__(self, resolved, *, model=None):
            super().__init__(resolved, model=model or FunctionModel(respond))
            self.results = []
            self.close_count = 0
            instances.append(self)

        async def run(self, *args, **kwargs):
            result = await super().run(*args, **kwargs)
            self.results.append(result)
            return result

        async def resume(self, *args, **kwargs):
            result = await super().resume(*args, **kwargs)
            self.results.append(result)
            return result

        async def aclose(self):
            self.close_count += 1
            await super().aclose()

    monkeypatch.setattr(ui.app, "Agent", RecordingAgent)
    ui.config.write_text(
        ui.config.read_text("utf-8")
        + (
            "\n[profiles.cowork.workspace]\nread_cache_bytes = 1000\n"
            "\n[profiles.cowork.tools]\n"
            'allowed = ["workspace.read", "workspace.write_preview", "workspace.write"]\n'
        ),
        "utf-8",
    )
    (ui.config.parent / "notes.md").write_text("TODO: Friday review\n", "utf-8")
    ui.session["profile"] = "cowork"
    # These I/O/cache tests intentionally exercise independent tasks; continuous
    # conversation behavior has its own history and persistence tests.
    ui.session["settings"] = {"carry_history": False}
    return instances


def test_same_chat_reuses_source_cache_and_starts_fresh_tasks(ui, recording_agents):
    async def exercise():
        await ui.app.message_received(SimpleNamespace(content="Read notes.md"))
        await ui.app.message_received(SimpleNamespace(content="Read notes.md again"))
        assert len(recording_agents) == 1
        agent = recording_agents[0]
        cold, warm = agent.results
        assert cold.read_bytes > 0 and cold.read_cache_hits == 0
        assert warm.read_bytes == 0 and warm.read_cache_hits == 1
        assert warm.cached_read_bytes == cold.read_bytes
        assert cold.run_id != warm.run_id and cold.requests == warm.requests == 2
        assert cold.output == warm.output
        assert not agent._closed
        await ui.app.end()
        assert agent.close_count == 1 and not agent._files._read_cache
        await ui.app.end()
        assert len(agent.results) == 2 and agent.close_count == 1
        await ui.app.message_received(SimpleNamespace(content="New message after reconnect"))
        assert len(recording_agents) == 2
        assert recording_agents[1].results[0].read_cache_hits == 0
        await ui.app.end()

    asyncio.run(exercise())


@pytest.mark.parametrize("change", ["settings", "profile", "config", "source-file"])
def test_preferences_or_config_changes_replace_agent_before_next_task(
    ui, recording_agents, change, monkeypatch
):
    original = ui.config.read_bytes()

    async def exercise():
        await ui.app.message_received(SimpleNamespace(content="Read notes.md"))
        first = recording_agents[0]
        if change == "settings":
            await ui.app.settings_update({"city": "Hangzhou", "response_style": "detailed"})
        elif change == "profile":
            ui.config.write_text(
                ui.config.read_text("utf-8")
                + ('\n[profiles.other.tools]\nallowed = ["workspace.read"]\n'),
                "utf-8",
            )
            ui.session["profile"] = "other"
        elif change == "config":
            ui.config.write_text(
                ui.config.read_text("utf-8").replace(
                    "read_cache_bytes = 1000", "read_cache_bytes = 2000"
                ),
                "utf-8",
            )
        else:
            # Same resolved settings in a different file still need a new Agent/ledger location.
            alternate = ui.config.parent / "alternate.toml"
            alternate.write_bytes(ui.config.read_bytes())
            ui.app.config_path = lambda: alternate
            monkeypatch.setattr(ui.app.features, "config_path", lambda: alternate)
        await ui.app.message_received(SimpleNamespace(content="Read again"))
        assert len(recording_agents) == 2
        second = recording_agents[1]
        assert first.close_count == 1 and not first._files._read_cache
        assert second.results[0].read_cache_hits == 0 and second.results[0].read_bytes > 0
        if change == "settings":
            assert second.resolved.config.preferences.city == "Hangzhou"
            assert second.resolved.config.preferences.response_style == "detailed"
            assert ui.config.read_bytes() == original
        await ui.app.end()

    asyncio.run(exercise())


def test_different_chats_do_not_share_agent_or_cache(ui, recording_agents):
    async def exercise():
        await ui.app.message_received(SimpleNamespace(content="Read notes.md"))
        first_runtime = ui.session.pop("harness_runtime")
        await ui.app.message_received(SimpleNamespace(content="Read notes.md"))
        second_runtime = ui.session["harness_runtime"]
        assert len(recording_agents) == 2 and first_runtime is not second_runtime
        assert all(agent.results[0].read_cache_hits == 0 for agent in recording_agents)
        await ui.app.end()
        ui.session["harness_runtime"] = first_runtime
        await ui.app.end()
        assert all(agent.close_count == 1 for agent in recording_agents)

    asyncio.run(exercise())


def test_approved_writes_keep_sources_warm_and_source_changes_are_read(
    ui, recording_agents, monkeypatch
):
    class Ask:
        def __init__(self, **kwargs):
            pass

        async def send(self):
            return {"payload": {"approved": True}}

    monkeypatch.setattr(ui.app.cl, "AskActionMessage", Ask)

    async def exercise():
        from chainlit.context import init_http_context

        init_http_context()
        command = SimpleNamespace(content="/workspace-demo notes.md brief.md")
        await ui.app.message_received(command)
        await ui.app.message_received(command)
        assert len(recording_agents) == 1
        agent = recording_agents[0]
        assert len(agent.results) == 4, ([(r.status, r.output) for r in agent.results], ui.sent)
        cold, warm = agent.results[1], agent.results[3]
        assert cold.read_cache_hits == 0 and warm.read_cache_hits == 1
        assert warm.cached_read_bytes == len((ui.config.parent / "notes.md").read_bytes())
        assert cold.requests == warm.requests == 4
        source = ui.config.parent / "notes.md"
        previous = source.stat()
        source.write_text("TODO: Monday review\n", "utf-8")
        os.utime(source, ns=(previous.st_atime_ns, previous.st_mtime_ns + 2_000_000_000))
        await ui.app.message_received(command)
        final = agent.results[-1]
        assert final.read_cache_hits == 0
        text = (ui.config.parent / "brief.md").read_text("utf-8")
        assert "Monday" in text and "Friday" not in text
        # Different demo arguments and normal task models cannot reuse a scripted demo Agent.
        await ui.app.message_received(SimpleNamespace(content="/workspace-demo notes.md other.md"))
        await ui.app.message_received(SimpleNamespace(content="Read notes.md"))
        assert len(recording_agents) == 3
        assert [a.close_count for a in recording_agents] == [1, 1, 0]
        await ui.app.end()

    asyncio.run(exercise())


@pytest.mark.parametrize("action", ["stop", "end"])
def test_stopping_during_approval_cancels_task_discards_queue_and_releases_agent(
    ui, recording_agents, monkeypatch, action
):
    async def exercise():
        from chainlit.context import init_http_context

        init_http_context()
        waiting = asyncio.Event()

        class Ask:
            def __init__(self, **kwargs):
                pass

            async def send(self):
                waiting.set()
                await asyncio.Event().wait()

        monkeypatch.setattr(ui.app.cl, "AskActionMessage", Ask)
        active = asyncio.create_task(
            ui.app.message_received(SimpleNamespace(content="/workspace-demo notes.md brief.md"))
        )
        await asyncio.wait_for(waiting.wait(), 5)
        agent = recording_agents[0]
        assert agent._pending
        queued = asyncio.create_task(
            ui.app.message_received(SimpleNamespace(content="Queued task"))
        )
        await asyncio.sleep(0)
        await asyncio.wait_for(getattr(ui.app, action)(), 5)
        with pytest.raises(asyncio.CancelledError):
            await active
        await queued
        assert len(recording_agents) == 1 and agent.close_count == 1
        assert not agent._pending and not agent._files._read_cache
        assert not (ui.config.parent / "brief.md").exists()
        if action == "stop":
            await ui.app.message_received(SimpleNamespace(content="Read notes.md"))
            assert len(recording_agents) == 2 and not recording_agents[1]._closed
        await ui.app.end()

    asyncio.run(exercise())


def test_queued_task_uses_new_preferences_after_existing_approval(
    ui, recording_agents, monkeypatch
):
    async def exercise():
        from chainlit.context import init_http_context

        init_http_context()
        waiting, release = asyncio.Event(), asyncio.Event()

        class Ask:
            def __init__(self, **kwargs):
                pass

            async def send(self):
                waiting.set()
                await release.wait()
                return {"payload": {"approved": True}}

        monkeypatch.setattr(ui.app.cl, "AskActionMessage", Ask)
        first = asyncio.create_task(
            ui.app.message_received(SimpleNamespace(content="/workspace-demo notes.md brief.md"))
        )
        await asyncio.wait_for(waiting.wait(), 5)
        await ui.app.settings_update({"response_style": "detailed"})
        queued = asyncio.create_task(
            ui.app.message_received(SimpleNamespace(content="/workspace-demo notes.md brief.md"))
        )
        await asyncio.sleep(0)
        assert len(recording_agents) == 1 and not recording_agents[0]._closed
        release.set()
        await asyncio.gather(first, queued)
        assert len(recording_agents) == 2 and recording_agents[0].close_count == 1
        assert recording_agents[0].resolved.config.preferences.response_style == "concise"
        assert recording_agents[1].resolved.config.preferences.response_style == "detailed"
        assert "提取条数" in (ui.config.parent / "brief.md").read_text("utf-8")
        await ui.app.end()

    asyncio.run(exercise())


@pytest.mark.parametrize("failure", ["invalid-config", "run-error"])
def test_error_retires_cached_agent_and_next_valid_task_can_recover(ui, recording_agents, failure):
    async def exercise():
        await ui.app.message_received(SimpleNamespace(content="Read notes.md"))
        first = recording_agents[0]
        original = ui.config.read_bytes()
        if failure == "invalid-config":
            ui.config.write_text("invalid TOML = [", "utf-8")
        else:

            async def fail(*a, **kw):
                raise RunError("test_failure", "Synthetic failure")

            first.run = fail
        await ui.app.message_received(SimpleNamespace(content="Read notes.md"))
        assert "Configuration or run error" in ui.sent[-1]
        assert first.close_count == 1 and ui.session["harness_runtime"].agent is None
        ui.config.write_bytes(original)
        await ui.app.message_received(SimpleNamespace(content="Read notes.md"))
        assert len(recording_agents) == 2 and recording_agents[1].results[0].read_cache_hits == 0
        await ui.app.end()

    asyncio.run(exercise())


@pytest.fixture
def online_ui(ui, monkeypatch):
    httpx = pytest.importorskip("httpx2")
    openai = pytest.importorskip("openai")
    from tests import mock_provider

    ui.config.write_text(
        mock_provider.template()
        .replace("your-model-id", "test-chat")
        .replace("context_window = 32768", "context_window = 262144")
        .replace("read_cache_bytes = 0", "read_cache_bytes = 1000")
        .replace("provider_retries = 2", "provider_retries = 0")
        .replace('max_daily_cny = "2.00"', 'max_daily_cny = "0.2630"'),
        "utf-8",
    )
    (ui.config.parent / "notes.md").write_text("Friday review", "utf-8")
    ui.session["profile"] = "cowork"
    clients, payloads = [], []
    ui.session["settings"] = {"carry_history": False}
    pending = {"event": None, "response_hook": None, "include_usage": True}

    async def respond(request):
        payload = json.loads(request.content)
        payloads.append(payload)
        if pending["event"] is not None:
            pending["event"].set()
            await asyncio.Event().wait()
        if pending["response_hook"] is not None:
            overridden = pending["response_hook"](len(payloads))
            if overridden is not None:
                return overridden
        returned = [m for m in payload["messages"] if m["role"] == "tool"]
        message = {"role": "assistant", "content": "Friday review (notes.md)"}
        if not returned and any(
            tool["function"]["name"] == "workspace_read" for tool in payload.get("tools", [])
        ):
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "read-1",
                        "type": "function",
                        "function": {
                            "name": "workspace_read",
                            "arguments": '{"relative_path":"notes.md"}',
                        },
                    }
                ],
            }
        elif returned:
            assert len(returned) == 1
        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "total_tokens": 110,
            "prompt_tokens_details": {"cached_tokens": 0},
        }
        if payload.get("stream"):
            common = {
                "id": "mock-ui",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "test-chat",
            }
            tool_calls = message.get("tool_calls")
            deltas = (
                [{"tool_calls": [{"index": 0, **tool_calls[0]}]}]
                if tool_calls
                else [{"content": "Friday "}, {"content": "review (notes.md)"}]
            )
            events = [
                {**common, "choices": [{"index": 0, "delta": {"role": "assistant", **delta}}]}
                for delta in deltas
            ]
            events.append(
                {
                    **common,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "tool_calls" if tool_calls else "stop",
                        }
                    ],
                }
            )
            if pending["include_usage"]:
                events.append({**common, "choices": [], "usage": usage})
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                text="".join(f"data: {json.dumps(event)}\n\n" for event in events)
                + "data: [DONE]\n\n",
            )
        return httpx.Response(
            200,
            json={
                "id": "mock-ui",
                "object": "chat.completion",
                "created": 0,
                "model": "test-chat",
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "stop"
                        if returned or not payload.get("tools")
                        else "tool_calls",
                    }
                ],
                "usage": usage,
            },
        )

    def create(model, path, timeout_seconds=45, api_key=None):
        client = openai.AsyncOpenAI(
            api_key="local-test",
            base_url=mock_provider.BASE_URL,
            timeout=timeout_seconds,
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        )
        clients.append(client)
        return client

    monkeypatch.setattr("personal_harness.connections.compatible_client", create)
    return SimpleNamespace(clients=clients, payloads=payloads, pending=pending)


@pytest.mark.parametrize("scenario", ["success", "missing_tool_usage", "missing_text_usage"])
def test_opt_in_streaming_ui_meters_sse_and_finalizes_one_message(
    ui, online_ui, monkeypatch, scenario
):
    from chainlit.context import init_http_context
    from chainlit.message import Message

    from personal_harness.ledger import Ledger

    monkeypatch.setenv("PERSONAL_HARNESS_UI_PERSIST", "1")
    ui.app.features.state().save_user("alice", {})
    monkeypatch.setattr(ui.app.features, "owner", lambda: "alice")
    monkeypatch.setattr(ui.app.features, "thread_id", lambda: None)
    ui.session["settings"]["stream_responses"] = True
    if scenario == "missing_text_usage":
        ui.session["settings"]["tools"] = []
    online_ui.pending["include_usage"] = scenario == "success"
    chunks, sent = [], []

    class Reply(Message):
        async def send(self):
            sent.append((self.id, self.content, self.metadata, self.elements, tuple(chunks)))

    async def exercise():
        init_http_context()
        emitter = ui.app.cl.context.emitter

        async def start(step):
            chunks.append((step["id"], step["output"]))

        async def token(id, token, is_sequence):
            chunks.append((id, token))

        monkeypatch.setattr(emitter, "stream_start", start)
        monkeypatch.setattr(emitter, "send_token", token)
        monkeypatch.setattr(ui.app.cl, "Message", Reply)
        await ui.app.message_received(SimpleNamespace(content="Read notes.md"))
        ledger = Ledger(ui.config.parent / ".personal-harness/ledger.sqlite3", read_only=True)
        [run] = ledger.recent()
        assert all(p["stream"] and p["stream_options"]["include_usage"] for p in online_ui.payloads)
        assert len(sent) == 1, sent
        message_id, output, metadata, elements, prior_chunks = sent[0]
        if scenario == "success":
            assert run["status"] == "completed" and run["requests"] == 2
            assert ledger.total_exposure() == Decimal("0.00022")
            assert output == "Friday review (notes.md)"
            assert prior_chunks == (
                (message_id, "Friday "),
                (message_id, "review (notes.md)"),
            )
            assert metadata["harness_run_id"] == run["run_id"]
            assert elements[0].props["cost"] == "0.00022"
            ui.session["settings"]["max_task_cny"] = "0"
            await ui.app.message_received(SimpleNamespace(content="Read it again"))
            assert "task_budget" in sent[-1][1]
            assert len(online_ui.payloads) == 2
            assert ledger.recent()[0]["requests"] == 0
        else:
            # Missing usage on the first tool request stops the next call.
            assert run["status"] == "budget_stopped" and run["requests"] == 1
            assert run["stop_reason"] == "unknown_usage"
            if scenario == "missing_text_usage":
                assert prior_chunks[0] == (message_id, "Friday ")
                assert "Friday review (notes.md)" not in output
            else:
                assert not prior_chunks
            assert "unknown_usage" in output
            assert ledger.total_exposure() == Decimal("0.262656")
        await ui.app.end()

    asyncio.run(exercise())


def test_attachment_followup_guard_avoids_unintended_paid_request(ui, online_ui, monkeypatch):
    from chainlit.context import init_http_context

    from personal_harness.ledger import Ledger

    source = ui.config.parent / "minutes.txt"
    source.write_text("预算上限1200元。\n" * 3000, "utf-8")
    ui.session["settings"] = {"carry_history": True, "tools": []}

    class Ask:
        def __init__(self, **kwargs):
            assert "超过当前历史容量" in kwargs["content"]

        async def send(self):
            assert len(online_ui.payloads) == 1
            return {"payload": {"continue": False}}

    monkeypatch.setattr(ui.app.cl, "AskActionMessage", Ask)

    async def exercise():
        init_http_context()
        await ui.app.message_received(
            SimpleNamespace(
                content="摘要",
                elements=[SimpleNamespace(path=str(source), name=source.name)],
            )
        )
        ledger = Ledger(ui.config.parent / ".personal-harness/ledger.sqlite3", read_only=True)
        assert len(ledger.recent()) == 1
        await ui.app.message_received(SimpleNamespace(content="上一轮资料的预算上限？"))
        assert "未发送模型请求" in ui.sent[-1]
        assert len(online_ui.payloads) == len(ledger.recent()) == 1
        assert ledger.total_exposure() == Decimal("0.00011")
        ui.session["settings"]["history_max_chars"] = "150000"
        await ui.app.message_received(SimpleNamespace(content="上一轮资料的预算上限？"))
        assert len(online_ui.payloads) == len(ledger.recent()) == 2
        sent = [m["content"] for m in online_ui.payloads[-1]["messages"] if m["role"] == "user"]
        assert sent[-1] == "上一轮资料的预算上限？"
        assert "预算上限1200元" in sent[0] and 'name="minutes.txt"' in sent[0]
        assert ledger.total_exposure() == Decimal("0.00022")
        await ui.app.end()

    asyncio.run(exercise())


def test_online_ui_reuses_client_and_keeps_daily_budget_across_tasks(ui, online_ui):
    from personal_harness.ledger import Ledger

    async def exercise():
        await ui.app.message_received(SimpleNamespace(content="Read notes.md"))
        agent = ui.session["harness_runtime"].agent
        assert agent is not None, ui.sent
        await ui.app.message_received(SimpleNamespace(content="Read again"))
        assert ui.session["harness_runtime"].agent is agent
        assert len(online_ui.clients) == 1 and len(online_ui.payloads) == 4
        ledger = Ledger(ui.config.parent / ".personal-harness/ledger.sqlite3")
        records = ledger.recent()
        assert len(records) == 2 and all(r["requests"] == 2 for r in records)
        assert records[0]["metrics"]["read_cache_hits"] == 1
        assert records[1]["metrics"]["read_cache_hits"] == 0
        assert ledger.total_exposure() == Decimal("0.00044")
        await ui.app.message_received(SimpleNamespace(content="Budget must stop this"))
        assert online_ui.clients[0].is_closed and len(online_ui.payloads) == 4
        stopped = ledger.recent()[0]
        assert stopped["status"] == "budget_stopped" and stopped["requests"] == 0
        assert stopped["stop_reason"] == "daily_budget"
        await ui.app.message_received(SimpleNamespace(content="A new client cannot reset spending"))
        assert len(online_ui.clients) == 2 and online_ui.clients[1].is_closed
        assert len(online_ui.payloads) == 4 and ledger.total_exposure() == Decimal("0.00044")
        await ui.app.end()

    asyncio.run(exercise())


def test_online_ui_reuses_client_and_withdraws_deleted_memory_from_history(
    ui, online_ui, monkeypatch
):
    from chainlit.context import init_http_context

    from personal_harness.user_insights import MemoryInput, UserInsights

    monkeypatch.setenv("PERSONAL_HARNESS_UI_PERSIST", "1")
    store = ui.app.features.state()
    store.save_user("alice", {})
    monkeypatch.setattr(ui.app.features, "owner", lambda: "alice")
    monkeypatch.setattr(ui.app.features, "thread_id", lambda: "insight-thread")
    knowledge = UserInsights(store)
    identity = knowledge.remember(
        "alice",
        MemoryInput(kind="workflow", text="I prefer Python data summaries"),
    )
    ui.config.write_text(
        ui.config.read_text("utf-8").replace('max_daily_cny = "0.2630"', 'max_daily_cny = "2.00"'),
        "utf-8",
    )
    ui.session["settings"]["carry_history"] = True

    async def exercise():
        init_http_context()
        await ui.app.message_received(SimpleNamespace(content="Read notes.md using Python"))
        runtime = ui.session["harness_runtime"]
        agent = runtime.agent
        assert agent is not None, ui.sent[-1]
        assert len(online_ui.payloads) == 2
        assert "I prefer Python data summaries" in json.dumps(
            online_ui.payloads[:], ensure_ascii=False
        )
        await ui.app.message_received(SimpleNamespace(content="Read notes.md again with Python"))
        assert len(online_ui.payloads) == 3
        knowledge.change("alice", identity, "delete")
        await ui.app.message_received(
            SimpleNamespace(content="Read notes.md once more with Python")
        )
        assert len(online_ui.payloads) == 4, ui.sent
        later_requests = json.dumps(online_ui.payloads[3:], ensure_ascii=False)
        assert "I prefer Python data summaries" not in later_requests
        assert ui.session["harness_runtime"].agent is agent
        assert len(online_ui.clients) == 1 and not online_ui.clients[0].is_closed(), ui.sent[-1]
        await ui.app.end()

    asyncio.run(exercise())


def test_online_ui_budget_command_is_read_only_and_sends_no_requests(ui, online_ui):
    from personal_harness.ledger import Ledger

    async def exercise():
        await ui.app.message_received(SimpleNamespace(content="/budget"))
        assert "One request hold: CNY 0.262656" in ui.sent[-1]
        assert "First request of a new task fits" in ui.sent[-1]
        assert not online_ui.payloads
        assert not online_ui.clients
        assert ui.session["harness_runtime"].agent is None
        assert not (ui.config.parent / ".personal-harness" / "ledger.sqlite3").exists()

        await ui.app.message_received(SimpleNamespace(content="Read notes.md"))
        ledger = Ledger(ui.config.parent / ".personal-harness/ledger.sqlite3")
        before = ledger.recent()
        request_count = len(online_ui.payloads)
        await ui.app.message_received(SimpleNamespace(content="/budget"))
        assert "First request of a new task fits" in ui.sent[-1]
        assert len(online_ui.payloads) == request_count
        assert ledger.recent() == before
        await ui.app.end()

    asyncio.run(exercise())


def test_online_ui_total_budget_persists_across_tasks_and_new_agent(ui, online_ui):
    from personal_harness.ledger import Ledger

    ui.config.write_text(
        ui.config.read_text("utf-8").replace(
            'max_daily_cny = "0.2630"',
            'max_daily_cny = "2.00"\nmax_total_cny = "0.2629"',
        ),
        "utf-8",
    )

    async def exercise():
        await ui.app.message_received(SimpleNamespace(content="Read notes.md"))
        assert len(online_ui.payloads) == 2
        await ui.app.message_received(SimpleNamespace(content="Read again within total budget"))
        # The next read is admitted, but the model's final response cannot reserve enough.
        assert len(online_ui.payloads) == 3 and online_ui.clients[0].is_closed
        ledger = Ledger(ui.config.parent / ".personal-harness/ledger.sqlite3")
        stopped = ledger.recent()[0]
        assert stopped["status"] == "budget_stopped" and stopped["stop_reason"] == "total_budget"
        assert stopped["requests"] == 1 and stopped["total_limit_cny"] == "0.2629"
        assert "Ledger total limit for this task: CNY 0.2629" in ui.sent[-1]
        assert "Cumulative spending and budget holds" in ui.sent[-1]
        await ui.app.message_received(SimpleNamespace(content="Do not reset on new client"))
        assert len(online_ui.payloads) == 3 and len(online_ui.clients) == 2
        assert ledger.recent()[0]["requests"] == 0
        assert ledger.total_exposure() == Decimal("0.00033")
        await ui.app.end()

    asyncio.run(exercise())


def test_stop_during_online_request_closes_client_and_preserves_unknown_charge(ui, online_ui):
    from personal_harness.ledger import Ledger

    async def exercise():
        waiting = asyncio.Event()
        online_ui.pending["event"] = waiting
        task = asyncio.create_task(
            ui.app.message_received(SimpleNamespace(content="Read notes.md"))
        )
        await asyncio.wait_for(waiting.wait(), 5)
        await asyncio.wait_for(ui.app.stop(), 5)
        with pytest.raises(asyncio.CancelledError):
            await task
        assert online_ui.clients[0].is_closed
        ledger = Ledger(ui.config.parent / ".personal-harness/ledger.sqlite3")
        stopped = ledger.recent()[0]
        assert stopped["status"] == "cancelled" and stopped["requests"] == 1
        assert stopped["cost"] is None and stopped["reserved_cost"] == "0.262656"
        assert stopped["charges"][0]["outcome"] == "cancelled"
        online_ui.pending["event"] = None
        await ui.app.message_received(SimpleNamespace(content="Do not bypass unknown cost"))
        assert len(online_ui.payloads) == 1 and online_ui.clients[1].is_closed
        assert ledger.recent()[0]["stop_reason"] == "unreconciled_cost"
        assert "Cost: CNY 0" in ui.sent[-1] and "No model request sent." in ui.sent[-1]
        assert "Earlier unresolved charges require billing reconciliation" in ui.sent[-1]
        await ui.app.end()

    asyncio.run(exercise())


def test_chainlit_loader_imports_app_and_registers_lifecycle_hooks(ui):
    from chainlit.config import config, load_module

    path = str(files("personal_harness").joinpath("chainlit_app.py"))
    load_module(path)
    assert config.code.on_message is not None
    assert config.code.on_chat_end is not None and config.code.on_stop is not None


@pytest.mark.parametrize("scenario", ["retry", "failure"])
def test_online_ui_shows_known_and_unknown_costs_without_losing_failed_attempts(
    ui, online_ui, scenario
):
    import httpx2

    from personal_harness.ledger import Ledger

    contents = ui.config.read_text("utf-8").replace(
        'max_daily_cny = "0.2630"', 'max_daily_cny = "2"'
    )
    if scenario == "retry":
        contents = contents.replace("provider_retries = 0", "provider_retries = 1")
    ui.config.write_text(contents, "utf-8")
    original = ui.config.read_bytes()

    def override(attempt):
        if scenario == "retry" and attempt == 1:
            return httpx2.Response(
                429,
                headers={"Retry-After": "0.001"},
                json={"error": {"message": "private-error-body"}},
            )
        if scenario == "failure" and attempt == 2:
            return httpx2.Response(500, json={"error": {"message": "private-error-body"}})
        return None

    online_ui.pending["response_hook"] = override

    async def exercise():
        await ui.app.settings_update({"tools": ["workspace.read"]})
        await ui.app.message_received(SimpleNamespace(content="Read notes.md"))
        summary = Ledger(ui.config.parent / ".personal-harness/ledger.sqlite3").recent()[0]
        text = ui.sent[-1]
        assert "Cost: CNY unknown" in text
        assert f"Known subtotal: CNY {'0.00022' if scenario == 'retry' else '0.00011'}" in text
        assert "Unknown charges: 1" in text
        assert "Budget hold: CNY 0.262656 (not a billed amount)" in text
        assert "usage complete: no" in text and summary["run_id"] in text
        assert "None" not in text and "private-error-body" not in text
        assert summary["cost"] is None and not summary["cost_complete"]
        assert summary["requests"] == len(online_ui.payloads) == (3 if scenario == "retry" else 2)
        assert summary["status"] == ("completed" if scenario == "retry" else "failed")
        assert ui.config.read_bytes() == original
        request_count = len(online_ui.payloads)
        ui.session["settings"]["connection"] = "deleted-connection"
        await ui.app.message_received(SimpleNamespace(content="/usage"))
        assert "失败/取消/结果未明的请求 1 次" in ui.sent[-1]
        assert "费用未知 1 次" in ui.sent[-1] and "未知" in ui.sent[-1]
        assert len(online_ui.payloads) == request_count
        await ui.app.end()
        assert all(client.is_closed for client in online_ui.clients)

    asyncio.run(exercise())


@pytest.mark.parametrize("allowed", [[], ["workspace.read", "workspace.search", "workspace.read"]])
def test_start_uses_native_selector_with_only_profile_permissions(ui, monkeypatch, allowed):
    ui.config.write_text(
        ui.config.read_text("utf-8")
        + ("\n[profiles.scoped.tools]\nallowed = " + json.dumps(allowed) + "\n"),
        "utf-8",
    )
    ui.session["chat_profile"] = "scoped"
    widgets = []
    original_settings = ui.app.cl.ChatSettings

    class Settings(original_settings):
        async def send(self):
            widgets.extend(w for tab in self.inputs for w in tab.inputs)
            return self.settings()

    monkeypatch.setattr(ui.app.cl, "ChatSettings", Settings)
    asyncio.run(ui.app.start())
    assert "Personal Harness" in ui.sent[-1] and "`scoped`" in ui.sent[-1]
    selected = list(dict.fromkeys(allowed))
    assert ui.session["settings"]["tools"] == selected
    selectors = [widget for widget in widgets if widget.id == "tools"]
    assert len(selectors) == (1 if selected else 0)
    if selectors:
        serialized = selectors[0].to_dict()
        assert serialized["type"] == "multiselect" and serialized["initial"] == selected
        assert [item["value"] for item in serialized["items"]] == selected
        assert "workspace.write" not in serialized["initial"]


def test_tools_change_without_replacing_agent_or_losing_source_cache(ui, recording_agents):
    original = ui.config.read_bytes()

    async def exercise():
        await ui.app.settings_update({"tools": ["workspace.read", "workspace.read"]})
        await ui.app.message_received(SimpleNamespace(content="Read notes.md"))
        await ui.app.settings_update({"tools": []})
        await ui.app.message_received(SimpleNamespace(content="Only answer text"))
        await ui.app.settings_update({"tools": ["workspace.read"]})
        await ui.app.message_received(SimpleNamespace(content="Read again"))
        assert len(recording_agents) == 1
        cold, text, warm = recording_agents[0].results
        assert cold.available_tools == warm.available_tools == ("workspace.read",)
        assert text.available_tools == text.tool_calls == () and text.read_bytes == 0
        assert text.requests == 1 and cold.requests == warm.requests == 2
        assert warm.read_cache_hits == 1 and warm.read_bytes == 0
        assert cold.run_id != text.run_id != warm.run_id
        assert "tools enabled: `1`" in ui.sent[-1] and warm.run_id in ui.sent[-1]
        await ui.app.end()

    asyncio.run(exercise())
    assert ui.config.read_bytes() == original


@pytest.mark.parametrize(
    "invalid", [None, "workspace.read", [1], ["workspace.replace_text"], ["unknown"]]
)
def test_invalid_selection_stops_before_creating_agent(ui, recording_agents, invalid):
    ui.session["settings"] = {"tools": invalid}
    asyncio.run(ui.app.message_received(SimpleNamespace(content="Do not invoke model")))
    assert not recording_agents
    assert "Configuration or run error" in ui.sent[-1]
    assert ui.session["harness_runtime"].agent is None


def test_partial_settings_update_preserves_empty_selection_and_copies_input(ui):
    async def exercise():
        settings = {"tools": [], "response_style": "concise"}
        await ui.app.settings_update(settings)
        settings["tools"].append("workspace.write")
        await ui.app.settings_update({"city": "Hangzhou"})
        assert ui.session["settings"] == {
            "tools": [],
            "response_style": "concise",
            "city": "Hangzhou",
        }
        await ui.app.message_received(SimpleNamespace(content="Inspect preferences"))
        assert "Hangzhou" in ui.sent[-1] and "tools enabled: `0`" in ui.sent[-1]
        await ui.app.end()

    asyncio.run(exercise())


@pytest.mark.parametrize("change", ["grant", "revoke"])
def test_reloaded_permissions_do_not_widen_or_override_saved_selection(
    ui, recording_agents, change
):
    async def exercise():
        await ui.app.settings_update({"tools": ["workspace.read"]})
        await ui.app.message_received(SimpleNamespace(content="Read notes.md"))
        before = recording_agents[0]
        config = ui.config.read_text("utf-8")
        replacement = (
            '["workspace.search"]'
            if change == "revoke"
            else (
                '["workspace.read", "workspace.write_preview", '
                '"workspace.write", "workspace.read_many"]'
            )
        )
        ui.config.write_text(
            config.replace(
                '["workspace.read", "workspace.write_preview", "workspace.write"]', replacement
            ),
            "utf-8",
        )
        await ui.app.message_received(SimpleNamespace(content="Read again"))
        assert before.close_count == 1
        assert ui.session["settings"]["tools"] == ["workspace.read"]
        if change == "revoke":
            assert len(recording_agents) == 1 and "Configuration or run error" in ui.sent[-1]
        else:
            assert len(recording_agents) == 2
            assert recording_agents[1].results[0].available_tools == ("workspace.read",)
        await ui.app.end()

    asyncio.run(exercise())


def test_tool_selection_change_during_approval_only_affects_queued_task(
    ui, recording_agents, monkeypatch
):
    async def exercise():
        from chainlit.context import init_http_context

        init_http_context()
        waiting, release = asyncio.Event(), asyncio.Event()

        class Ask:
            def __init__(self, **kwargs):
                pass

            async def send(self):
                waiting.set()
                await release.wait()
                return {"payload": {"approved": True}}

        monkeypatch.setattr(ui.app.cl, "AskActionMessage", Ask)
        await ui.app.settings_update(
            {"tools": ["workspace.read", "workspace.write_preview", "workspace.write"]}
        )
        command = SimpleNamespace(content="/workspace-demo notes.md brief.md")
        active = asyncio.create_task(ui.app.message_received(command))
        await asyncio.wait_for(waiting.wait(), 5)
        await ui.app.settings_update({"tools": []})
        queued = asyncio.create_task(ui.app.message_received(command))
        await asyncio.sleep(0)
        assert len(recording_agents) == 1 and recording_agents[0]._pending
        release.set()
        await asyncio.gather(active, queued)
        assert len(recording_agents) == 1
        pending, approved, text = recording_agents[0].results
        assert (
            pending.available_tools
            == approved.available_tools
            == ("workspace.read", "workspace.write_preview", "workspace.write")
        )
        assert approved.status == "completed" and "workspace.write" in approved.tool_calls
        assert text.available_tools == () and text.tool_calls == ()
        assert (ui.config.parent / "brief.md").exists()
        await ui.app.end()

    asyncio.run(exercise())


def test_online_ui_tool_choice_reaches_actual_request_and_keeps_one_budget(ui, online_ui):
    from personal_harness.ledger import Ledger

    async def exercise():
        await ui.app.settings_update({"tools": ["workspace.read"]})
        await ui.app.message_received(SimpleNamespace(content="Read notes.md"))
        agent = ui.session["harness_runtime"].agent
        assert agent is not None
        assert len(online_ui.payloads) == 2
        assert all(
            [t["function"]["name"] for t in p["tools"]] == ["workspace_read"]
            for p in online_ui.payloads
        )
        await ui.app.settings_update({"tools": []})
        await ui.app.message_received(SimpleNamespace(content="Text only"))
        assert ui.session["harness_runtime"].agent is agent
        assert len(online_ui.clients) == 1 and len(online_ui.payloads) == 3
        assert not online_ui.payloads[-1].get("tools")
        assert "tools enabled: `0`" in ui.sent[-1]
        ledger = Ledger(ui.config.parent / ".personal-harness/ledger.sqlite3")
        assert ledger.total_exposure() == Decimal("0.00033")
        await ui.app.settings_update({"tools": ["workspace.read"]})
        await ui.app.message_received(SimpleNamespace(content="Budget must still stop this"))
        # The remaining budget admits the read request, then stops the final response.
        assert len(online_ui.payloads) == 4 and online_ui.clients[0].is_closed
        assert ledger.recent()[0]["stop_reason"] == "daily_budget"
        assert ledger.recent()[0]["requests"] == 1
        assert ledger.total_exposure() == Decimal("0.00044")
        await ui.app.end()

    asyncio.run(exercise())
