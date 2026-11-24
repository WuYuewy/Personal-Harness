"""Local, inspectable user knowledge and usage signals. No model or embedding calls."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
from collections import Counter
from datetime import UTC, datetime
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from personal_harness.ui_state import UIState

KINDS = {
    "communication": "沟通偏好",
    "workflow": "工作习惯",
    "tools": "常用工具",
    "interests": "关注主题",
    "constraints": "个人约束",
}
CATEGORIES = {
    "coding": ("编程开发", ("代码", "开发", "编程", "调试", "bug", "python", "typescript", "api")),
    "writing": ("写作整理", ("写作", "文案", "报告", "总结", "周报", "简报", "润色", "文档")),
    "research": ("学习研究", ("研究", "论文", "学习", "解释", "文献", "实验", "分析原理")),
    "data": ("数据分析", ("数据", "统计", "图表", "表格", "csv", "excel", "可视化")),
    "planning": ("计划协作", ("计划", "安排", "会议", "项目", "截止", "待办", "日程")),
    "life": ("日常生活", ("餐厅", "吃饭", "出行", "旅行", "附近", "购物", "菜谱", "通勤")),
    "other": ("其他", ()),
}
SECRET = re.compile(
    r"sk-[\w-]{8,}|(?<![a-z])(?:api[ _-]?key|password|authorization|bearer|secret)(?![a-z])|"
    r"密码|密钥|身份证|银行卡|病史|"
    r"[\w.+-]+@[\w.-]+\.[a-z]{2,}|(?<!\d)1[3-9]\d{9}(?!\d)",
    re.I,
)
DECLARATION = re.compile(
    r"^(?:请记住[，,:： ]*)?(?:我(?:个人)?(?:通常|一般|习惯|喜欢|偏好|更喜欢|"
    r"不喜欢|不希望|希望|常用|常常|每周)|"
    r"i (?:usually|prefer|like|dislike|always|use)\b)",
    re.I,
)
STOP_WORDS = {
    "请",
    "帮我",
    "请帮",
    "请帮我",
    "一个",
    "一下",
    "我们",
    "你们",
    "他们",
    "自己",
    "这个",
    "那个",
    "这些",
    "一些",
    "进行",
    "可以",
    "需要",
    "目前",
    "继续",
    "使用",
    "工作",
    "任务",
    "完成",
    "用户",
    "agent",
    "please",
    "help",
    "with",
    "this",
    "that",
    "want",
    "would",
    "could",
    "prefer",
    "usually",
    "like",
    "don't",
    "希望",
    "喜欢",
    "习惯",
    "通常",
    "每周",
    "助手",
    "记住",
}


class MemoryInput(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    kind: Literal["communication", "workflow", "tools", "interests", "constraints"]
    text: str = Field(strict=True, min_length=2, max_length=240)

    @field_validator("text")
    @classmethod
    def clean_text(cls, value: str) -> str:
        value = " ".join(value.split())
        if len(value) < 2 or SECRET.search(value):
            raise ValueError(
                "Keep preferences short and exclude credentials or private identifiers."
            )
        return value


class InsightOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    capture: StrictBool = False
    use_memories: StrictBool = True


def categories(text: str) -> list[str]:
    lowered = text.casefold()
    found = [
        key
        for key, (_, words) in CATEGORIES.items()
        if words and any(word in lowered for word in words)
    ]
    return found or ["other"]


@lru_cache(maxsize=1)
def _extractor() -> Any:
    # An independent tokenizer keeps the application's vocabulary out of jieba globals.
    import jieba
    from jieba.analyse import TFIDF

    jieba.setLogLevel(logging.WARNING)
    extractor = TFIDF()
    extractor.tokenizer = jieba.Tokenizer()
    return extractor


def keywords(text: str) -> list[str]:
    text = SECRET.sub(" ", text[:12000])
    # URLs, quoted text and code are excluded from the source in observe().
    try:
        terms = _extractor().extract_tags(text, topK=24, withWeight=False)
    except Exception:
        # Base SDK installs do not need the optional Chinese tokenizer.
        logger.debug(
            "Local Chinese keyword extraction unavailable; using Latin terms.", exc_info=True
        )
        terms = re.findall(r"[a-zA-Z][a-zA-Z0-9_+.-]{1,29}", text)
    result: list[str] = []
    for term in terms:
        term = term.casefold()
        if 2 <= len(term) <= 30 and term not in STOP_WORDS and term not in result:
            result.append(term)
        if len(result) == 8:
            break
    return result


def direct_text(text: str) -> str:
    """Analyze user-authored prose only, never tool output, attachments or quoted examples."""
    text = re.sub(r"```[\s\S]*?```|~~~[\s\S]*?~~~", " ", text[:12000])
    text = re.sub(r"https?://\S+|`[^`]*`|[“\"][^”\"]*[”\"]|「[^」]*」|‘[^’]*’", " ", text)
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith(">"))


def suggestions(text: str) -> list[MemoryInput]:
    found: list[MemoryInput] = []
    for sentence in re.findall(r"[^。！？!?\n;；]+[！？!?]?", text):
        sentence = sentence.strip()
        if (
            sentence.endswith(("?", "？"))
            or not DECLARATION.match(sentence)
            or SECRET.search(sentence)
            or len(sentence) > 240
        ):
            continue
        # Questions, reported speech and temporary instructions are not stable preferences.
        if any(
            word in sentence
            for word in ("例如", "比如", "他说", "她说", "假设", "今天", "本次", "这次", "如果")
        ):
            continue
        kind = "interests"
        if any(
            word in sentence.lower()
            for word in ("回复", "回答", "结论", "解释", "语气", "要点", "表格", "简洁")
        ):
            kind = "communication"
        elif any(
            word in sentence.lower()
            for word in ("python", "excel", "工具", "编辑器", "常用", "typescript")
        ):
            kind = "tools"
        elif any(word in sentence for word in ("不能", "不希望", "上限", "预算", "不要")):
            kind = "constraints"
        elif any(word in sentence for word in ("习惯", "通常", "每周", "一般", "整理")):
            kind = "workflow"
        found.append(MemoryInput(kind=kind, text=sentence))
    return found[:3]


class UserInsights:
    """Small UI-local store; each read and mutation scopes data to the authenticated user."""

    def __init__(self, state: UIState) -> None:
        self.state = state

    def _require_user(self, owner: str) -> None:
        if self.state.user(owner) is None:
            raise ValueError("Sign in to manage personal knowledge.")

    def options(self, owner: str) -> InsightOptions:
        self._require_user(owner)
        with self.state.db() as db:
            row = db.execute("SELECT data FROM insight_options WHERE owner=?", (owner,)).fetchone()
        return InsightOptions.model_validate_json(row[0]) if row else InsightOptions()

    def configure(self, owner: str, options: InsightOptions) -> None:
        self._require_user(owner)
        with self.state.db() as db:
            db.execute(
                "INSERT OR REPLACE INTO insight_options VALUES (?,?)",
                (owner, options.model_dump_json()),
            )

    def set_option(self, owner: str, name: str, value: bool) -> None:
        self._require_user(owner)
        if name not in InsightOptions.model_fields or not isinstance(value, bool):
            raise ValueError("Invalid preference option.")
        with self.state.db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT data FROM insight_options WHERE owner=?", (owner,)).fetchone()
            options = InsightOptions.model_validate_json(row[0]) if row else InsightOptions()
            options = InsightOptions.model_validate({**options.model_dump(), name: value})
            db.execute(
                "INSERT OR REPLACE INTO insight_options VALUES (?,?)",
                (owner, options.model_dump_json()),
            )

    def remember(
        self,
        owner: str,
        memory: MemoryInput,
        *,
        status: str = "confirmed",
        thread_id: str | None = None,
        message_id: str | None = None,
    ) -> str:
        self._require_user(owner)
        with self.state.db() as db:
            db.execute("BEGIN IMMEDIATE")
            return self._remember(db, owner, memory, status, thread_id, message_id)

    @staticmethod
    def _remember(
        db: sqlite3.Connection,
        owner: str,
        memory: MemoryInput,
        status: str,
        thread_id: str | None,
        message_id: str | None,
    ) -> str:
        identity = uuid4().hex
        fingerprint = hashlib.sha256(memory.text.casefold().encode()).hexdigest()
        timestamp = datetime.now(UTC).isoformat()
        row = db.execute(
            "SELECT id,status FROM user_memories WHERE owner=? AND fingerprint=?",
            (owner, fingerprint),
        ).fetchone()
        if row and (status == "pending" or row[1] == "confirmed"):
            return str(row[0])
        count = db.execute(
            "SELECT count(*) FROM user_memories WHERE owner=? AND status=?", (owner, status)
        ).fetchone()[0]
        if count >= (100 if status == "confirmed" else 50):
            if status == "pending":
                return ""
            raise ValueError("Keep at most 100 confirmed preferences; remove an old entry first.")
        if row:
            db.execute(
                "UPDATE user_memories SET kind=?,text=?,status='confirmed',thread_id=?,"
                "message_id=?,updated=? WHERE id=?",
                (memory.kind, memory.text, thread_id, message_id, timestamp, row[0]),
            )
            return str(row[0])
        db.execute(
            "INSERT INTO user_memories VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                identity,
                owner,
                memory.kind,
                memory.text,
                status,
                fingerprint,
                thread_id,
                message_id,
                timestamp,
                timestamp,
            ),
        )
        return identity

    def change(
        self, owner: str, identity: str, operation: str, memory: MemoryInput | None = None
    ) -> None:
        self._require_user(owner)
        with self.state.db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT status FROM user_memories WHERE owner=? AND id=?", (owner, identity)
            ).fetchone()
            if not row:
                raise ValueError("Preference no longer exists for this account.")
            timestamp = datetime.now(UTC).isoformat()
            if operation == "delete":
                db.execute("DELETE FROM user_memories WHERE owner=? AND id=?", (owner, identity))
            elif operation == "dismiss":
                if row[0] != "pending":
                    raise ValueError("Only unconfirmed suggestions can be dismissed.")
                # Keep only a fingerprint so this suggestion is not proposed again.
                db.execute(
                    "UPDATE user_memories SET status='dismissed', text='', thread_id=NULL, "
                    "message_id=NULL, updated=? WHERE owner=? AND id=?",
                    (timestamp, owner, identity),
                )
            elif operation == "save" and memory:
                count = db.execute(
                    "SELECT count(*) FROM user_memories WHERE owner=? AND status='confirmed'",
                    (owner,),
                ).fetchone()[0]
                if row[0] != "confirmed" and count >= 100:
                    raise ValueError("Keep at most 100 confirmed preferences.")
                fingerprint = hashlib.sha256(memory.text.casefold().encode()).hexdigest()
                other = db.execute(
                    "SELECT id FROM user_memories WHERE owner=? AND fingerprint=? AND id<>?",
                    (owner, fingerprint, identity),
                ).fetchone()
                if other:
                    raise ValueError("An identical preference already exists.")
                db.execute(
                    "UPDATE user_memories SET kind=?,text=?,status='confirmed',fingerprint=?,"
                    "updated=? WHERE owner=? AND id=?",
                    (memory.kind, memory.text, fingerprint, timestamp, owner, identity),
                )
            else:
                raise ValueError("Invalid preference operation.")

    def observe(self, owner: str, thread_id: str, message_id: str, text: str) -> None:
        if not self.options(owner).capture or text.lstrip().startswith("/"):
            return
        text = direct_text(text)
        if not text.strip() or SECRET.search(text):
            return  # Never derive a preference or keyword from a credential-bearing message.
        tags = categories(text)
        terms = keywords(text)
        timestamp = datetime.now(UTC).isoformat()
        with self.state.db() as db:
            # Check options again after extraction, including a concurrent clear/disable.
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT data FROM insight_options WHERE owner=?", (owner,)).fetchone()
            if not row or not InsightOptions.model_validate_json(row[0]).capture:
                return
            source_owner = db.execute(
                "SELECT user_id FROM threads WHERE id=?", (thread_id,)
            ).fetchone()
            if source_owner and source_owner[0] is not None:
                user = db.execute("SELECT id FROM users WHERE identifier=?", (owner,)).fetchone()
                if user is None or source_owner[0] != user[0]:
                    raise ValueError("Chat belongs to another account.")
            inserted = db.execute(
                "INSERT OR IGNORE INTO insight_events VALUES (?,?,?,?,?,?)",
                (message_id, owner, thread_id, timestamp, json.dumps(tags), json.dumps(terms)),
            ).rowcount
            db.execute(
                "DELETE FROM insight_events WHERE owner=? AND rowid NOT IN "
                "(SELECT rowid FROM insight_events WHERE owner=? "
                "ORDER BY created DESC,rowid DESC LIMIT 2000)",
                (owner, owner),
            )
            if inserted:
                for memory in suggestions(text):
                    self._remember(db, owner, memory, "pending", thread_id, message_id)

    def recall(self, owner: str, task: str) -> tuple[str, ...]:
        if not self.options(owner).use_memories:
            return ()
        with self.state.db() as db:
            rows = db.execute(
                "SELECT kind,text FROM user_memories WHERE owner=? AND status='confirmed' "
                "ORDER BY updated DESC,id",
                (owner,),
            ).fetchall()
        if not rows:
            return ()
        task_words = set(keywords(direct_text(task))) if not SECRET.search(task[:12000]) else set()

        def rank(row: Any) -> int:
            overlap = task_words.intersection(keywords(row[1]))
            return sum(6 if re.fullmatch(r"[a-z][\w.+-]*", word) else 3 for word in overlap) + (
                row[0] in {"workflow", "communication", "constraints"}
            )

        # Relevant terms first, then global working preferences. Always bounded.
        selected = sorted(((rank(row), row[1]) for row in rows), key=lambda r: r[0], reverse=True)
        recalled: list[str] = []
        remaining = 640
        for score, text in selected:
            if score <= 0 or len(recalled) == 4:
                break
            if len(text) <= remaining:
                recalled.append(text)
                remaining -= len(text)
        return tuple(recalled)

    def clear(self, owner: str) -> None:
        self._require_user(owner)
        with self.state.db() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("DELETE FROM user_memories WHERE owner=?", (owner,))
            db.execute("DELETE FROM insight_events WHERE owner=?", (owner,))
            options = InsightOptions(capture=False, use_memories=False)
            db.execute(
                "INSERT OR REPLACE INTO insight_options VALUES (?,?)",
                (owner, options.model_dump_json()),
            )

    def snapshot(self, owner: str) -> dict[str, Any]:
        options = self.options(owner)
        with self.state.db() as db:
            memories = [
                dict(row)
                for row in db.execute(
                    "SELECT id,kind,text,status,thread_id,message_id,created,updated "
                    "FROM user_memories WHERE owner=? AND status<>'dismissed' "
                    "ORDER BY updated DESC,rowid DESC",
                    (owner,),
                )
            ]
            events = db.execute(
                "SELECT categories,keywords,created FROM insight_events WHERE owner=? "
                "ORDER BY created DESC,rowid DESC",
                (owner,),
            ).fetchall()
        topics: Counter[str] = Counter()
        terms: Counter[str] = Counter()
        activity: Counter[str] = Counter()
        for row in events:
            topics.update(json.loads(row[0]))
            terms.update(json.loads(row[1]))
            activity.update([row[2][:10]])
        confirmed = Counter(
            memory["kind"] for memory in memories if memory["status"] == "confirmed"
        )
        return {
            "options": options.model_dump(),
            "memories": memories,
            "kinds": KINDS,
            "sample_count": len(events),
            "dimensions": [
                {"label": label, "count": confirmed[key]} for key, label in KINDS.items()
            ],
            "categories": [
                {"label": CATEGORIES[key][0], "count": count} for key, count in topics.most_common()
            ],
            "keywords": [{"label": term, "count": count} for term, count in terms.most_common(16)],
            "activity": [
                {"label": date, "count": count} for date, count in sorted(activity.items())[-14:]
            ],
        }
