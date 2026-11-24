"""Local UI persistence. Conversation data and model credentials stay off the prompt."""

from __future__ import annotations

import base64
import ctypes
import hashlib
import hmac
import json
import os
import secrets
import shlex
import sqlite3
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter

from personal_harness.config import ModelSettings
from personal_harness.reports import ReportRecipe


def now() -> str:
    return datetime.now(UTC).isoformat()


def public_metadata(value: Any) -> Any:
    """Defense in depth: never persist credential inputs in a chat or an export."""
    if isinstance(value, dict):
        return {
            key: public_metadata(item)
            for key, item in value.items()
            if key.lower()
            not in {"env", "api_key", "apikey", "password", "authorization", "harness_runtime"}
        }
    if isinstance(value, list):
        return [public_metadata(item) for item in value]
    return value


def _protect(value: bytes, *, decrypt: bool = False) -> bytes:
    # Windows keys are tied to the current OS user. Unix relies on private local files.
    if sys.platform != "win32":
        return value

    class Blob(ctypes.Structure):
        _fields_ = [("size", ctypes.c_ulong), ("data", ctypes.POINTER(ctypes.c_char))]

    buffer = ctypes.create_string_buffer(value)
    source = Blob(len(value), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_char)))
    target = Blob()
    operation = (
        ctypes.windll.crypt32.CryptUnprotectData
        if decrypt
        else ctypes.windll.crypt32.CryptProtectData
    )
    if not operation(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(target)):
        raise OSError("Cannot access the local credential store.")
    try:
        return ctypes.string_at(target.data, target.size)
    finally:
        ctypes.windll.kernel32.LocalFree(target.data)


class Connection(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    name: str = Field(min_length=1, max_length=80)
    model: ModelSettings


SCENE_PREFIX = "local_scene_"


class LocalShortcut(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True, frozen=True)
    name: str = Field(min_length=1, max_length=48)
    command: Literal["/find", "/lookup", "/places", "/sum", "/actions"]
    args: tuple[str, ...] = Field(max_length=4)

    @model_validator(mode="after")
    def validate_fields(self) -> Self:
        if not self.name.strip() or not self.name.isprintable():
            raise ValueError("请输入单行快捷查询名称。")
        if self.command == "/actions":
            valid = (
                len(self.args) in {1, 2}
                and bool(self.args[0].strip())
                and len(self.args[0]) <= 240
                and self.args[0].isprintable()
                and (
                    len(self.args) == 1
                    or bool(self.args[1].strip())
                    and len(self.args[1]) <= 80
                    and self.args[1].isprintable()
                )
            )
        elif self.command == "/lookup":
            valid = len(self.args) == 3 and all(
                1 <= len(arg) <= 240 and arg.isprintable() for arg in self.args
            )
        elif self.command == "/sum":
            valid = (
                len(self.args) == 4
                and all(len(arg) <= 240 and (not arg or arg.isprintable()) for arg in self.args)
                and bool(self.args[0])
                and bool(self.args[1])
                and bool(self.args[2]) == bool(self.args[3])
            )
        else:
            valid = len(self.args) == 1 and (
                (not self.args[0] and self.command == "/places")
                or (0 < len(self.args[0].strip()) <= 200 and self.args[0].isprintable())
            )
        if not valid:
            raise ValueError("请检查查询字段。")
        return self

    def content(self) -> str:
        args = (
            shlex.join(self.args)
            if self.command in {"/lookup", "/sum", "/actions"}
            else self.args[0].strip()
        )
        return f"{self.command} {args}".rstrip()


class UIState:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        if os.name != "nt":
            path.parent.chmod(0o700)
        with self.db() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS ui_values (id TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS users (id TEXT PRIMARY KEY, identifier TEXT UNIQUE,
                    data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS threads (id TEXT PRIMARY KEY, user_id TEXT,
                    created TEXT NOT NULL, updated TEXT NOT NULL, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS steps (id TEXT PRIMARY KEY, thread_id TEXT NOT NULL,
                    created TEXT NOT NULL, data TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS ui_steps_thread ON steps(thread_id, created);
                CREATE TABLE IF NOT EXISTS elements (id TEXT PRIMARY KEY, thread_id TEXT,
                    data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS feedback (id TEXT PRIMARY KEY, step_id TEXT,
                    data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS conversations (id TEXT PRIMARY KEY, messages BLOB);
                CREATE TABLE IF NOT EXISTS connections (id TEXT PRIMARY KEY, owner TEXT NOT NULL,
                    data TEXT NOT NULL, credential BLOB);
                CREATE TABLE IF NOT EXISTS ui_runs (id TEXT PRIMARY KEY, thread_id TEXT NOT NULL,
                    data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS local_shortcuts (id TEXT PRIMARY KEY,
                    owner TEXT NOT NULL, profile TEXT NOT NULL, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS scenes (id TEXT PRIMARY KEY, owner TEXT NOT NULL,
                    name TEXT NOT NULL, base_profile TEXT NOT NULL,
                    archived INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS report_recipes (id TEXT PRIMARY KEY,
                    owner TEXT NOT NULL, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS report_recipe_runs (id TEXT PRIMARY KEY,
                    owner TEXT NOT NULL, recipe_id TEXT NOT NULL, created TEXT NOT NULL,
                    source_count INTEGER NOT NULL, model_requests INTEGER NOT NULL,
                    cost_cny TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS report_recipe_runs_owner
                    ON report_recipe_runs(owner, recipe_id, created);
                CREATE TABLE IF NOT EXISTS insight_options (owner TEXT PRIMARY KEY,
                    data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS user_memories (id TEXT PRIMARY KEY, owner TEXT NOT NULL,
                    kind TEXT NOT NULL, text TEXT NOT NULL, status TEXT NOT NULL,
                    fingerprint TEXT NOT NULL, thread_id TEXT, message_id TEXT,
                    created TEXT NOT NULL, updated TEXT NOT NULL, UNIQUE(owner, fingerprint));
                CREATE INDEX IF NOT EXISTS ui_memories_owner ON user_memories(owner, status);
                CREATE TABLE IF NOT EXISTS insight_events (id TEXT NOT NULL, owner TEXT NOT NULL,
                    thread_id TEXT NOT NULL, created TEXT NOT NULL, categories TEXT NOT NULL,
                    keywords TEXT NOT NULL, PRIMARY KEY(owner,thread_id,id));
                CREATE INDEX IF NOT EXISTS ui_insights_owner ON insight_events(owner,created);
            """)
        if os.name != "nt":
            path.chmod(0o600)

    @contextmanager
    def db(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def auth_secret(self) -> str:
        with self.db() as db:
            db.execute(
                "INSERT OR IGNORE INTO ui_values VALUES ('jwt', ?)", (secrets.token_urlsafe(48),)
            )
            return str(db.execute("SELECT value FROM ui_values WHERE id='jwt'").fetchone()[0])

    def authenticate(self, username: str, password: str) -> bool:
        if username != "local" or len(password) < 8 or len(password) > 1024:
            return False
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT value FROM ui_values WHERE id='password'").fetchone()
            if row is None:
                salt = secrets.token_bytes(16)
                digest = hashlib.scrypt(password.encode(), salt=salt, n=16384, r=8, p=1)
                db.execute(
                    "INSERT INTO ui_values VALUES ('password', ?)",
                    (base64.b64encode(salt + digest).decode(),),
                )
                return True
            saved = base64.b64decode(row[0])
            digest = hashlib.scrypt(password.encode(), salt=saved[:16], n=16384, r=8, p=1)
            return hmac.compare_digest(saved[16:], digest)

    def has_password(self) -> bool:
        with self.db() as db:
            return db.execute("SELECT 1 FROM ui_values WHERE id='password'").fetchone() is not None

    def user(self, identifier: str) -> dict[str, Any] | None:
        with self.db() as db:
            row = db.execute("SELECT data FROM users WHERE identifier=?", (identifier,)).fetchone()
            return json.loads(row[0]) if row else None

    def save_user(self, identifier: str, metadata: dict[str, Any]) -> dict[str, Any]:
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT data FROM users WHERE identifier=?", (identifier,)).fetchone()
            data = (
                json.loads(row[0])
                if row
                else {
                    "id": uuid4().hex,
                    "identifier": identifier,
                    "createdAt": now(),
                }
            )
            existing_metadata = data.get("metadata", {})
            if not isinstance(existing_metadata, dict):
                existing_metadata = {}
            data["metadata"] = {**existing_metadata, **public_metadata(metadata)}
            db.execute(
                "INSERT OR REPLACE INTO users VALUES (?, ?, ?)",
                (data["id"], identifier, json.dumps(data)),
            )
            return data

    def personal_profile(self, identifier: str) -> dict[str, Any]:
        data = self.user(identifier)
        if data is None:
            return {}
        metadata = data.get("metadata", {})
        profile = metadata.get("personal_harness_profile", {})
        return dict(profile) if isinstance(profile, dict) else {}

    def save_personal_profile(self, identifier: str, profile: dict[str, Any]) -> None:
        """Store only validated, non-secret assistant experience preferences per account."""
        allowed = {
            "preferred_name",
            "language",
            "response_style",
            "tone",
            "verbosity",
            "answer_format",
            "clarification",
            "initiative",
            "cite_sources",
            "cost_strategy",
            "personal_instructions",
        }
        clean = {key: value for key, value in profile.items() if key in allowed}
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT data FROM users WHERE identifier=?", (identifier,)).fetchone()
            if row is None:
                raise ValueError("User profile requires an authenticated account.")
            data = json.loads(row[0])
            metadata = data.get("metadata", {})
            existing = metadata.get("personal_harness_profile", {})
            if isinstance(existing, dict):
                clean = {
                    **{key: value for key, value in existing.items() if key in allowed},
                    **clean,
                }
            metadata["personal_harness_profile"] = public_metadata(clean)
            data["metadata"] = metadata
            db.execute("UPDATE users SET data=? WHERE identifier=?", (json.dumps(data), identifier))

    def update_thread(
        self,
        thread_id: str,
        name: str | None = None,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT data FROM threads WHERE id=?", (thread_id,)).fetchone()
            data = (
                json.loads(row[0])
                if row
                else {
                    "id": thread_id,
                    "createdAt": now(),
                    "name": None,
                    "userId": None,
                    "userIdentifier": None,
                    "tags": [],
                    "metadata": {},
                }
            )
            if name is not None:
                data["name"] = name[:200]
            if user_id is not None:
                data["userId"] = user_id
                author = db.execute(
                    "SELECT identifier FROM users WHERE id=?", (user_id,)
                ).fetchone()
                data["userIdentifier"] = author[0] if author else None
            if metadata is not None:
                data["metadata"].update(public_metadata(metadata))
            if tags is not None:
                data["tags"] = tags
            db.execute(
                "INSERT OR REPLACE INTO threads VALUES (?, ?, ?, ?, ?)",
                (thread_id, data["userId"], data["createdAt"], now(), json.dumps(data)),
            )

    def put_step(self, data: dict[str, Any]) -> None:
        thread_id = data["threadId"]
        self.update_thread(thread_id)
        with self.db() as db:
            db.execute(
                "INSERT OR REPLACE INTO steps VALUES (?, ?, ?, ?)",
                (
                    data["id"],
                    thread_id,
                    data.get("createdAt") or now(),
                    json.dumps(public_metadata(data)),
                ),
            )

    def thread(self, thread_id: str) -> dict[str, Any] | None:
        with self.db() as db:
            row = db.execute("SELECT data FROM threads WHERE id=?", (thread_id,)).fetchone()
            if not row:
                return None
            data: dict[str, Any] = json.loads(row[0])
            data["steps"] = [
                json.loads(r[0])
                for r in db.execute(
                    "SELECT data FROM steps WHERE thread_id=? ORDER BY created, rowid", (thread_id,)
                )
            ]
            for step in data["steps"]:
                # A historical approval is never an active approval after reconnect/restart.
                step["waitForAnswer"] = False
            data["elements"] = [
                json.loads(r[0])
                for r in db.execute("SELECT data FROM elements WHERE thread_id=?", (thread_id,))
            ]
            return data

    def list_threads(self, owner: str, search: str = "") -> list[dict[str, Any]]:
        with self.db() as db:
            rows = db.execute(
                "SELECT data FROM threads WHERE user_id=? ORDER BY updated DESC, id", (owner,)
            ).fetchall()
            threads = [json.loads(row[0]) for row in rows]
            if search:
                matched = {
                    row[0]
                    for row in db.execute(
                        "SELECT DISTINCT thread_id FROM steps "
                        "WHERE instr(lower(json_extract(data, '$.output')), lower(?)) > 0",
                        (search,),
                    )
                }
                threads = [
                    t
                    for t in threads
                    if search.casefold() in (t["name"] or "").casefold() or t["id"] in matched
                ]
            return [{**t, "steps": [], "elements": []} for t in threads]

    def delete_thread(self, thread_id: str) -> None:
        with self.db() as db:
            db.execute("DELETE FROM insight_events WHERE thread_id=?", (thread_id,))
            db.execute(
                "DELETE FROM user_memories WHERE thread_id=? AND status='pending'", (thread_id,)
            )
            db.execute(
                "UPDATE user_memories SET thread_id=NULL,message_id=NULL WHERE thread_id=?",
                (thread_id,),
            )
            assets = [
                json.loads(row[0]).get("objectKey")
                for row in db.execute("SELECT data FROM elements WHERE thread_id=?", (thread_id,))
            ]
            db.execute(
                "DELETE FROM feedback WHERE step_id IN (SELECT id FROM steps WHERE thread_id=?)",
                (thread_id,),
            )
            for table, column in (
                ("threads", "id"),
                ("steps", "thread_id"),
                ("elements", "thread_id"),
                ("conversations", "id"),
                ("ui_runs", "thread_id"),
            ):
                db.execute(f"DELETE FROM {table} WHERE {column}=?", (thread_id,))
        base = (self.path.parent / "ui-assets").resolve()
        for name in assets:
            if isinstance(name, str) and name and all(c in "0123456789abcdef" for c in name):
                path = (base / name).resolve()
                if path.is_relative_to(base):
                    path.unlink(missing_ok=True)

    def save_history(self, thread_id: str, messages: list[ModelMessage]) -> None:
        encoded = ModelMessagesTypeAdapter.dump_json(messages)
        with self.db() as db:
            db.execute("INSERT OR REPLACE INTO conversations VALUES (?, ?)", (thread_id, encoded))

    def history(self, thread_id: str) -> list[ModelMessage]:
        with self.db() as db:
            row = db.execute(
                "SELECT messages FROM conversations WHERE id=?", (thread_id,)
            ).fetchone()
            return ModelMessagesTypeAdapter.validate_json(row[0]) if row else []

    def connections(self, owner: str) -> list[dict[str, Any]]:
        with self.db() as db:
            return [
                {"id": r[0], **json.loads(r[1]), "has_key": r[2] is not None}
                for r in db.execute(
                    "SELECT id, data, credential FROM connections WHERE owner=? ORDER BY rowid",
                    (owner,),
                )
            ]

    def save_connection(
        self,
        owner: str,
        data: Connection,
        key: SecretStr | None,
        connection_id: str | None = None,
    ) -> str:
        if data.model.provider != "openai-compatible":
            raise ValueError("Saved connections must use an OpenAI-compatible endpoint.")
        identity = connection_id or uuid4().hex
        with self.db() as db:
            existing = db.execute(
                "SELECT owner, credential FROM connections WHERE id=?", (identity,)
            ).fetchone()
            if existing and existing[0] != owner:
                raise ValueError("Connection does not belong to this user.")
            credential = (
                _protect(key.get_secret_value().encode())
                if key
                else (existing[1] if existing else None)
            )
            db.execute(
                "INSERT OR REPLACE INTO connections VALUES (?, ?, ?, ?)",
                (identity, owner, data.model_dump_json(), credential),
            )
        return identity

    def connection(self, owner: str, connection_id: str) -> tuple[Connection, SecretStr | None]:
        with self.db() as db:
            row = db.execute(
                "SELECT data, credential FROM connections WHERE id=? AND owner=?",
                (connection_id, owner),
            ).fetchone()
            if row is None:
                raise ValueError("Model connection no longer exists. Select another connection.")
            return Connection.model_validate_json(row[0]), (
                SecretStr(_protect(row[1], decrypt=True).decode()) if row[1] else None
            )

    def delete_connection(self, owner: str, connection_id: str) -> None:
        with self.db() as db:
            db.execute("DELETE FROM connections WHERE id=? AND owner=?", (connection_id, owner))

    def report_recipes(self, owner: str) -> list[dict[str, Any]]:
        with self.db() as db:
            recipes = db.execute(
                "SELECT id, data FROM report_recipes WHERE owner=? ORDER BY rowid", (owner,)
            ).fetchall()
            result = []
            for row in recipes:
                recipe = ReportRecipe.model_validate_json(row[1])
                usage = db.execute(
                    "SELECT count(*), max(created) FROM report_recipe_runs "
                    "WHERE owner=? AND recipe_id=?",
                    (owner, row[0]),
                ).fetchone()
                result.append(
                    {
                        **recipe.model_dump(mode="json"),
                        "id": row[0],
                        "runs": usage[0],
                        "last_run": usage[1],
                        "model_requests": 0,
                        "cost_cny": "0",
                    }
                )
            return result

    def report_recipe(self, owner: str, identity: str) -> ReportRecipe:
        with self.db() as db:
            row = db.execute(
                "SELECT data FROM report_recipes WHERE owner=? AND id=?", (owner, identity)
            ).fetchone()
        if row is None:
            raise ValueError("该任务方案已不存在，请刷新列表。")
        return ReportRecipe.model_validate_json(row[0])

    def save_report_recipe(self, owner: str, recipe: ReportRecipe, identity: str | None) -> str:
        with self.db() as db:
            if identity is None:
                db.execute("BEGIN IMMEDIATE")
                count = db.execute(
                    "SELECT count(*) FROM report_recipes WHERE owner=?", (owner,)
                ).fetchone()[0]
                if count >= 50:
                    raise ValueError("已达到 50 个个人任务方案的上限。")
                identity = uuid4().hex
                db.execute(
                    "INSERT INTO report_recipes VALUES (?, ?, ?)",
                    (identity, owner, recipe.model_dump_json()),
                )
            elif not db.execute(
                "UPDATE report_recipes SET data=? WHERE owner=? AND id=?",
                (recipe.model_dump_json(), owner, identity),
            ).rowcount:
                raise ValueError("该任务方案已不存在，请刷新列表。")
        return identity

    def delete_report_recipe(self, owner: str, identity: str) -> None:
        with self.db() as db:
            db.execute(
                "DELETE FROM report_recipe_runs WHERE owner=? AND recipe_id=?", (owner, identity)
            )
            if not db.execute(
                "DELETE FROM report_recipes WHERE owner=? AND id=?", (owner, identity)
            ).rowcount:
                raise ValueError("该任务方案已不存在，请刷新列表。")

    def record_report_run(self, owner: str, identity: str, source_count: int) -> None:
        with self.db() as db:
            if not db.execute(
                "SELECT 1 FROM report_recipes WHERE owner=? AND id=?", (owner, identity)
            ).fetchone():
                raise ValueError("任务方案已删除。")
            db.execute(
                "INSERT INTO report_recipe_runs VALUES (?, ?, ?, ?, ?, ?, ?)",
                (uuid4().hex, owner, identity, now(), source_count, 0, "0"),
            )

    def record_run(self, thread_id: str, run: dict[str, Any]) -> None:
        with self.db() as db:
            db.execute(
                "INSERT OR REPLACE INTO ui_runs VALUES (?, ?, ?)",
                (run["run_id"], thread_id, json.dumps(run)),
            )

    def local_shortcuts(self, owner: str, profile: str) -> list[dict[str, Any]]:
        with self.db() as db:
            rows = db.execute(
                "SELECT id, data FROM local_shortcuts WHERE owner=? AND profile=? ORDER BY rowid",
                (owner, profile),
            ).fetchall()
        return [{"id": row["id"], **json.loads(row["data"])} for row in rows]

    def local_shortcut(self, owner: str, profile: str, identity: str) -> LocalShortcut:
        with self.db() as db:
            row = db.execute(
                "SELECT data FROM local_shortcuts WHERE id=? AND owner=? AND profile=?",
                (identity, owner, profile),
            ).fetchone()
        if row is None:
            raise ValueError("此快捷查询已删除或不属于当前场景。")
        return LocalShortcut.model_validate_json(row[0])

    def save_local_shortcut(
        self, owner: str, profile: str, shortcut: LocalShortcut, identity: str | None
    ) -> None:
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            if identity is None:
                count = db.execute(
                    "SELECT count(*) FROM local_shortcuts WHERE owner=? AND profile=?",
                    (owner, profile),
                ).fetchone()[0]
                if count >= 8:
                    raise ValueError("此场景最多保存 8 个快捷查询。")
                db.execute(
                    "INSERT INTO local_shortcuts VALUES (?, ?, ?, ?)",
                    (uuid4().hex, owner, profile, shortcut.model_dump_json()),
                )
            elif not db.execute(
                "UPDATE local_shortcuts SET data=? WHERE id=? AND owner=? AND profile=?",
                (shortcut.model_dump_json(), identity, owner, profile),
            ).rowcount:
                raise ValueError("此快捷查询已删除或不属于当前场景。")

    def delete_local_shortcut(self, owner: str, profile: str, identity: str) -> None:
        with self.db() as db:
            if not db.execute(
                "DELETE FROM local_shortcuts WHERE id=? AND owner=? AND profile=?",
                (identity, owner, profile),
            ).rowcount:
                raise ValueError("此快捷查询已删除或不属于当前场景。")

    def defaults(self, owner: str, profile: str) -> dict[str, Any]:
        with self.db() as db:
            row = db.execute(
                "SELECT value FROM ui_values WHERE id=?", (f"defaults:{owner}:{profile}",)
            ).fetchone()
            return json.loads(row[0]) if row else {}

    def scenes(self, owner: str) -> list[dict[str, Any]]:
        with self.db() as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT id,name,base_profile,archived FROM scenes WHERE owner=? ORDER BY rowid",
                    (owner,),
                )
            ]

    def scene(self, owner: str, identity: str) -> dict[str, Any] | None:
        with self.db() as db:
            row = db.execute(
                "SELECT id,name,base_profile,archived FROM scenes WHERE owner=? AND id=?",
                (owner, identity),
            ).fetchone()
        return dict(row) if row else None

    def save_scene(
        self,
        owner: str,
        name: str,
        *,
        identity: str | None = None,
        base_profile: str | None = None,
        defaults: dict[str, Any] | None = None,
        copy_from: str | None = None,
    ) -> None:
        name = name.strip()
        if not name or len(name) > 48 or not name.isprintable():
            raise ValueError("请输入 1–48 个字符的场景名称。")
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute("SELECT id,name FROM scenes WHERE owner=?", (owner,)).fetchall()
            if any(
                row["id"] != identity and row["name"].casefold() == name.casefold() for row in rows
            ):
                raise ValueError("已有同名个人场景。")
            if identity is not None:
                changed = db.execute(
                    "UPDATE scenes SET name=? WHERE owner=? AND id=?", (name, owner, identity)
                )
                if not changed.rowcount:
                    raise ValueError("找不到此个人场景。")
                return
            if base_profile is None or len(rows) >= 32:
                raise ValueError("请选择基础场景。每个账号最多保存 32 个个人场景。")
            identity = SCENE_PREFIX + uuid4().hex
            db.execute(
                "INSERT INTO scenes VALUES (?, ?, ?, ?, 0)", (identity, owner, name, base_profile)
            )
            db.execute(
                "INSERT INTO ui_values VALUES (?, ?)",
                (f"defaults:{owner}:{identity}", json.dumps(public_metadata(defaults or {}))),
            )
            if copy_from is not None:
                shortcuts = db.execute(
                    "SELECT data FROM local_shortcuts WHERE owner=? AND profile=?",
                    (owner, copy_from),
                ).fetchall()
                db.executemany(
                    "INSERT INTO local_shortcuts VALUES (?, ?, ?, ?)",
                    [(uuid4().hex, owner, identity, row["data"]) for row in shortcuts],
                )

    def archive_scene(self, owner: str, identity: str, *, archived: bool) -> None:
        with self.db() as db:
            result = db.execute(
                "UPDATE scenes SET archived=? WHERE owner=? AND id=?", (archived, owner, identity)
            )
            if not result.rowcount:
                raise ValueError("找不到此个人场景。")

    def save_defaults(self, owner: str, profile: str, settings: dict[str, Any]) -> None:
        with self.db() as db:
            db.execute(
                "INSERT OR REPLACE INTO ui_values VALUES (?, ?)",
                (f"defaults:{owner}:{profile}", json.dumps(public_metadata(settings))),
            )

    def runs(self, thread_id: str) -> list[dict[str, Any]]:
        with self.db() as db:
            return [
                json.loads(row[0])
                for row in db.execute(
                    "SELECT data FROM ui_runs WHERE thread_id=? ORDER BY rowid", (thread_id,)
                )
            ]
