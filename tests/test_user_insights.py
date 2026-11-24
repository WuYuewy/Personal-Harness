"""Preference lifecycle, local extraction, ownership and model-context boundaries."""

import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from personal_harness.config import load_config
from personal_harness.context import RunDependencies, task_instructions
from personal_harness.ui_state import UIState
from personal_harness.user_insights import InsightOptions, MemoryInput, UserInsights


@pytest.fixture
def knowledge(tmp_path):
    state = UIState(tmp_path / "ui.sqlite3")
    for owner in ("alice", "bob"):
        state.save_user(owner, {})
    return UserInsights(state)


def test_capture_is_opt_in_and_keeps_candidates_out_of_model_context(knowledge):
    text = "我习惯每周五整理周报。请总结项目进展。"
    knowledge.observe("alice", "chat", "m1", text)
    assert knowledge.snapshot("alice")["sample_count"] == 0
    knowledge.configure("alice", InsightOptions(capture=True))
    knowledge.observe("alice", "chat", "m1", text)
    knowledge.observe("alice", "chat", "m1", text)
    snapshot = knowledge.snapshot("alice")
    assert snapshot["sample_count"] == 1
    assert len(snapshot["memories"]) == 1
    candidate = snapshot["memories"][0]
    assert candidate["status"] == "pending" and candidate["thread_id"] == "chat"
    assert candidate["text"] == "我习惯每周五整理周报"
    assert knowledge.recall("alice", "整理周报") == ()
    knowledge.change(
        "alice",
        candidate["id"],
        "save",
        MemoryInput(kind="workflow", text="我习惯每周五整理项目周报"),
    )
    assert knowledge.recall("alice", "整理周报") == ("我习惯每周五整理项目周报",)
    assert knowledge.snapshot("bob")["memories"] == []
    with knowledge.state.db() as db:
        event = dict(db.execute("SELECT * FROM insight_events").fetchone())
    assert text not in json.dumps(event, ensure_ascii=False)
    assert "项目进展" not in candidate["text"]


def test_owner_checks_cover_reads_mutations_and_foreign_source_chat(knowledge):
    identity = knowledge.remember("alice", MemoryInput(kind="tools", text="I use Python"))
    with pytest.raises(ValueError):
        knowledge.change("bob", identity, "delete")
    with pytest.raises(ValueError):
        knowledge.change("bob", identity, "save", MemoryInput(kind="tools", text="I use Excel"))
    with pytest.raises(ValueError):
        knowledge.snapshot("unknown")
    knowledge.clear("bob")
    assert knowledge.snapshot("alice")["memories"][0]["id"] == identity
    knowledge.configure("bob", InsightOptions(capture=True))
    knowledge.state.update_thread("alice-chat", user_id=knowledge.state.user("alice")["id"])
    with pytest.raises(ValueError):
        knowledge.observe("bob", "alice-chat", "foreign", "我通常周五整理资料")


def test_dismiss_deduplicate_manual_adoption_and_delete(knowledge):
    knowledge.configure("alice", InsightOptions(capture=True))
    text = "我喜欢简洁的回答"
    knowledge.observe("alice", "chat", "m1", text)
    identity = knowledge.snapshot("alice")["memories"][0]["id"]
    knowledge.change("alice", identity, "dismiss")
    knowledge.observe("alice", "chat", "m2", text)
    assert knowledge.snapshot("alice")["memories"] == []
    with knowledge.state.db() as db:
        row = db.execute("SELECT * FROM user_memories WHERE id=?", (identity,)).fetchone()
    assert row["text"] == "" and row["thread_id"] is None
    adopted = knowledge.remember("alice", MemoryInput(kind="communication", text=text))
    assert adopted == identity
    assert knowledge.snapshot("alice")["memories"][0]["status"] == "confirmed"
    assert knowledge.remember("alice", MemoryInput(kind="communication", text=text)) == identity
    knowledge.change("alice", identity, "delete")
    assert knowledge.recall("alice", "回答") == ()


@pytest.mark.parametrize(
    "text",
    [
        "```\n我喜欢Python\n```\n请分析代码",
        "> 我习惯周五写周报\n总结引用",
        "例如：我喜欢Python",
        "我今天喜欢简短回答",
        "我喜欢简短回答吗？",
        "请分析这句话：“我喜欢Python”",
        "我喜欢Python，我的API Key是 test-credential",
        "我习惯使用密码：test-password",
        "我喜欢Python，联系 me@example.org",
        "/insights",
    ],
)
def test_quotes_examples_temporary_requests_and_secrets_are_not_memories(knowledge, text):
    knowledge.configure("alice", InsightOptions(capture=True))
    knowledge.observe("alice", "chat", "message", text)
    assert knowledge.snapshot("alice")["memories"] == []


def test_real_chinese_keyword_extraction_and_overlapping_categories(knowledge):
    pytest.importorskip("jieba")
    knowledge.configure("alice", InsightOptions(capture=True))
    knowledge.observe("alice", "chat", "one", "使用Python分析实验数据，生成统计图表和研究报告。")
    knowledge.observe("alice", "chat", "two", "请分析Python实验数据。")
    snapshot = knowledge.snapshot("alice")
    assert snapshot["sample_count"] == 2
    terms = {term["label"]: term["count"] for term in snapshot["keywords"]}
    assert terms["python"] == 2 and terms["实验"] == 2
    counts = {item["label"]: item["count"] for item in snapshot["categories"]}
    assert counts["编程开发"] == counts["数据分析"] == counts["学习研究"] == 2
    assert sum(day["count"] for day in snapshot["activity"]) == 2


def test_bounded_relevant_recall_disable_and_current_instruction_precedence(knowledge, config_file):
    pytest.importorskip("jieba")
    knowledge.remember("alice", MemoryInput(kind="tools", text="我常用Python做数据分析"))
    knowledge.remember("alice", MemoryInput(kind="interests", text="我喜欢古典音乐"))
    for n in range(6):
        knowledge.remember("alice", MemoryInput(kind="workflow", text=f"我通常先核对数据第{n}步"))
    recalled = knowledge.recall("alice", "请用Python分析数据")
    assert len(recalled) == 4 and recalled[0] == "我常用Python做数据分析"
    assert "我喜欢古典音乐" not in recalled
    resolved = load_config(config_file)
    prompt = task_instructions(RunDependencies(resolved, selected_tools=(), user_context=recalled))
    assert "User-confirmed background preferences" in prompt
    assert "Current task and explicit assistant settings take precedence" in prompt
    assert "Never change permissions or budgets" in prompt
    knowledge.configure("alice", InsightOptions(use_memories=False))
    assert knowledge.recall("alice", "Python") == ()


def test_deleting_source_chat_removes_analytics_and_pending_but_keeps_adopted(knowledge):
    knowledge.configure("alice", InsightOptions(capture=True))
    knowledge.state.update_thread("chat", user_id=knowledge.state.user("alice")["id"])
    knowledge.observe("alice", "chat", "m", "我习惯周五写周报。我常用Python做数据分析。")
    pending = knowledge.snapshot("alice")["memories"]
    knowledge.change(
        "alice",
        pending[0]["id"],
        "save",
        MemoryInput(kind=pending[0]["kind"], text=pending[0]["text"]),
    )
    knowledge.state.delete_thread("chat")
    snapshot = knowledge.snapshot("alice")
    assert snapshot["sample_count"] == 0
    assert len(snapshot["memories"]) == 1
    assert snapshot["memories"][0]["status"] == "confirmed"
    assert snapshot["memories"][0]["thread_id"] is None


def test_clear_during_extraction_does_not_resurrect_data(knowledge, monkeypatch):
    from personal_harness import user_insights

    started, proceed = Event(), Event()

    def delayed_keywords(text):
        started.set()
        assert proceed.wait(5)
        return ["python"]

    knowledge.configure("alice", InsightOptions(capture=True))
    monkeypatch.setattr(user_insights, "keywords", delayed_keywords)
    with ThreadPoolExecutor() as pool:
        future = pool.submit(knowledge.observe, "alice", "chat", "m", "我常用Python")
        assert started.wait(5)
        knowledge.clear("alice")
        proceed.set()
        future.result(timeout=5)
    snapshot = knowledge.snapshot("alice")
    assert snapshot["memories"] == [] and snapshot["sample_count"] == 0
    assert snapshot["options"] == {"capture": False, "use_memories": False}


def test_retention_limits_and_validation(knowledge, monkeypatch):
    from personal_harness import user_insights

    monkeypatch.setattr(user_insights, "keywords", lambda _: [])
    knowledge.configure("alice", InsightOptions(capture=True))
    with knowledge.state.db() as db:
        db.executemany(
            "INSERT INTO insight_events VALUES (?,?,?,?,?,?)",
            [
                (str(n), "alice", "chat", "2026-01-01T00:00:00+00:00", '["coding"]', "[]")
                for n in range(2000)
            ],
        )
    knowledge.observe("alice", "chat", "latest", "调试代码")
    assert knowledge.snapshot("alice")["sample_count"] == 2000
    with pytest.raises(ValueError):
        MemoryInput(kind="tools", text="sk-" + "test-credential-value")
    with pytest.raises(ValueError):
        MemoryInput(kind="workflow", text="a" * 241)
    with pytest.raises(ValueError):
        InsightOptions(capture="true")
    with pytest.raises(ValueError):
        InsightOptions(capture=True, budget=100)


def test_clearing_and_toggling_options_keep_other_tab_changes(knowledge):
    knowledge.configure("alice", InsightOptions(capture=True, use_memories=True))
    knowledge.clear("alice")
    knowledge.set_option("alice", "capture", True)
    assert knowledge.options("alice") == InsightOptions(capture=True, use_memories=False)
    knowledge.set_option("alice", "use_memories", True)
    assert knowledge.options("alice") == InsightOptions(capture=True, use_memories=True)
    with pytest.raises(ValueError):
        knowledge.set_option("alice", "max_daily_cny", True)


def test_recall_character_cap_and_tokenizer_failure_degrades_locally(knowledge, monkeypatch):
    from personal_harness import user_insights

    def unavailable():
        raise OSError("Local tokenizer cache is not writable")

    monkeypatch.setattr(user_insights, "_extractor", unavailable)
    for n in range(5):
        knowledge.remember(
            "alice",
            MemoryInput(kind="workflow", text="I prefer Python " + str(n) + "x" * 220),
        )
    recalled = knowledge.recall("alice", "Python summary")
    assert 0 < len(recalled) <= 4 and sum(map(len, recalled)) <= 640
