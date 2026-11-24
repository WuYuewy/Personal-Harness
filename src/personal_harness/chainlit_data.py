"""Small local SQLite adapter for Chainlit's native history sidebar."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from chainlit.data.base import BaseDataLayer
from chainlit.data.utils import queue_until_user_message
from chainlit.element import Element, ElementDict
from chainlit.step import StepDict
from chainlit.types import (
    Feedback,
    PageInfo,
    PaginatedResponse,
    Pagination,
    ThreadDict,
    ThreadFilter,
)
from chainlit.user import PersistedUser, User

from personal_harness.ui_state import UIState, public_metadata


class LocalDataLayer(BaseDataLayer):
    def __init__(self, state: UIState) -> None:
        self.state = state

    async def get_user(self, identifier: str) -> PersistedUser | None:
        data = await asyncio.to_thread(self.state.user, identifier)
        return PersistedUser(**data) if data else None

    async def create_user(self, user: User) -> PersistedUser:
        data = await asyncio.to_thread(self.state.save_user, user.identifier, user.metadata)
        return PersistedUser(**data)

    async def delete_feedback(self, feedback_id: str) -> bool:
        with self.state.db() as db:
            row = db.execute("SELECT step_id FROM feedback WHERE id=?", (feedback_id,)).fetchone()
            if row is None:
                return False
            step = db.execute("SELECT data FROM steps WHERE id=?", (row[0],)).fetchone()
            if step:
                data = json.loads(step[0])
                if (data.get("feedback") or {}).get("id") == feedback_id:
                    data.pop("feedback", None)
                    db.execute("UPDATE steps SET data=? WHERE id=?", (json.dumps(data), row[0]))
            db.execute("DELETE FROM feedback WHERE id=?", (feedback_id,))
            return True

    async def upsert_feedback(self, feedback: Feedback) -> str:
        data = asdict(feedback)
        identity = feedback.id or uuid4().hex
        data["id"] = identity
        with self.state.db() as db:
            db.execute(
                "INSERT OR REPLACE INTO feedback VALUES (?, ?, ?)",
                (identity, feedback.forId, json.dumps(data)),
            )
            row = db.execute("SELECT data FROM steps WHERE id=?", (feedback.forId,)).fetchone()
            if row:
                step = json.loads(row[0])
                step["feedback"] = data
                db.execute("UPDATE steps SET data=? WHERE id=?", (json.dumps(step), feedback.forId))
        return identity

    @queue_until_user_message()
    async def create_element(self, element: Element) -> None:
        data = dict(element.to_dict())
        props = data.get("props")
        if element.name == "WorkspaceWelcome" and isinstance(props, dict):
            data["props"] = {
                key: value for key, value in props.items() if key != "initialShortcuts"
            }
        # Snapshot local attachments and approval diffs so they survive server restart.
        if element.type != "custom" and (element.content is not None or element.path):
            asset = self.state.path.parent / "ui-assets" / uuid4().hex
            asset.parent.mkdir(parents=True, exist_ok=True)
            content = element.content
            if content is None and element.path:
                content = await asyncio.to_thread(Path(element.path).read_bytes)
            await asyncio.to_thread(
                asset.write_bytes, content.encode() if isinstance(content, str) else content or b""
            )
            data["objectKey"] = asset.name
            data["url"] = f"/harness/asset/{element.id}"
            data["chainlitKey"] = None
        with self.state.db() as db:
            db.execute(
                "INSERT OR REPLACE INTO elements VALUES (?, ?, ?)",
                (element.id, element.thread_id, json.dumps(public_metadata(data))),
            )

    async def get_element(self, thread_id: str, element_id: str) -> ElementDict | None:
        with self.state.db() as db:
            row = db.execute(
                "SELECT data FROM elements WHERE id=? AND thread_id=?", (element_id, thread_id)
            ).fetchone()
            return json.loads(row[0]) if row else None

    @queue_until_user_message()
    async def delete_element(self, element_id: str, thread_id: str | None = None) -> None:
        with self.state.db() as db:
            db.execute("DELETE FROM elements WHERE id=?", (element_id,))

    @queue_until_user_message()
    async def create_step(self, step_dict: StepDict) -> None:
        await asyncio.to_thread(self.state.put_step, dict(step_dict))

    @queue_until_user_message()
    async def update_step(self, step_dict: StepDict) -> None:
        await asyncio.to_thread(self.state.put_step, dict(step_dict))

    @queue_until_user_message()
    async def delete_step(self, step_id: str) -> None:
        with self.state.db() as db:
            db.execute(
                "DELETE FROM steps WHERE id=? OR json_extract(data, '$.parentId')=?",
                (step_id, step_id),
            )

    async def get_thread_author(self, thread_id: str) -> str:
        thread = await asyncio.to_thread(self.state.thread, thread_id)
        return thread.get("userIdentifier") or "" if thread else ""

    async def delete_thread(self, thread_id: str) -> None:
        await asyncio.to_thread(self.state.delete_thread, thread_id)

    async def list_threads(
        self,
        pagination: Pagination,
        filters: ThreadFilter,
    ) -> PaginatedResponse[ThreadDict]:
        threads = await asyncio.to_thread(
            self.state.list_threads, filters.userId or "", filters.search or ""
        )
        if filters.feedback is not None:
            with self.state.db() as db:
                matches = {
                    r[0]
                    for r in db.execute(
                        "SELECT DISTINCT s.thread_id FROM steps s "
                        "JOIN feedback f ON s.id=f.step_id "
                        "WHERE json_extract(f.data, '$.value')=?",
                        (filters.feedback,),
                    )
                }
            threads = [t for t in threads if t["id"] in matches]
        start = 0
        if pagination.cursor:
            index = next((i for i, t in enumerate(threads) if t["id"] == pagination.cursor), None)
            if index is None:
                return PaginatedResponse(PageInfo(False, None, None), [])
            start = index + 1
        count = min(max(pagination.first, 1), 100)
        data = threads[start : start + count]
        return PaginatedResponse(
            PageInfo(
                start + count < len(threads),
                data[0]["id"] if data else None,
                data[-1]["id"] if data else None,
            ),
            data,  # type: ignore[arg-type]
        )

    async def get_thread(self, thread_id: str) -> ThreadDict | None:
        data = await asyncio.to_thread(self.state.thread, thread_id)
        return cast(ThreadDict, data) if data is not None else None

    async def update_thread(
        self,
        thread_id: str,
        name: str | None = None,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        await asyncio.to_thread(self.state.update_thread, thread_id, name, user_id, metadata, tags)

    async def build_debug_url(self) -> str:
        return ""

    async def close(self) -> None:
        pass  # Connections are short-lived and closed after each transaction.

    async def get_favorite_steps(self, user_id: str) -> list[StepDict]:
        with self.state.db() as db:
            return [
                json.loads(row[0])
                for row in db.execute(
                    "SELECT s.data FROM steps s JOIN threads t ON s.thread_id=t.id "
                    "WHERE t.user_id=? AND json_extract(s.data, '$.metadata.favorite')=1 "
                    "ORDER BY s.created DESC",
                    (user_id,),
                )
            ]
