"""Exercise the native sidebar pagination, feedback and favorites contracts."""

import asyncio

import pytest

pytest.importorskip("chainlit")

from chainlit.types import Feedback, Pagination, ThreadFilter  # noqa: E402

from personal_harness.chainlit_data import LocalDataLayer  # noqa: E402
from personal_harness.ui_state import UIState  # noqa: E402


def test_sidebar_pagination_search_feedback_and_delete(tmp_path):
    state = UIState(tmp_path / "ui.sqlite3")
    alice = state.save_user("alice", {})["id"]
    bob = state.save_user("bob", {})["id"]
    for identity in ("a", "b", "c"):
        state.update_thread(identity, f"Chat {identity}", alice)
    state.update_thread("private", "Private", bob)
    state.put_step({"id": "answer", "threadId": "b", "output": "Aurora"})
    layer = LocalDataLayer(state)

    async def exercise():
        filters = ThreadFilter(userId=alice)
        first = await layer.list_threads(Pagination(first=2), filters)
        assert len(first.data) == 2 and first.pageInfo.hasNextPage
        second = await layer.list_threads(
            Pagination(first=2, cursor=first.pageInfo.endCursor), filters
        )
        assert len(second.data) == 1 and not second.pageInfo.hasNextPage
        assert {t["id"] for t in first.data + second.data} == {"a", "b", "c"}
        result = await layer.list_threads(
            Pagination(first=10), ThreadFilter(userId=alice, search="Aurora")
        )
        assert [t["id"] for t in result.data] == ["b"]
        identity = await layer.upsert_feedback(Feedback(forId="answer", value=1))
        result = await layer.list_threads(
            Pagination(first=10), ThreadFilter(userId=alice, feedback=1)
        )
        assert [t["id"] for t in result.data] == ["b"]
        assert state.thread("b")["steps"][0]["feedback"]["value"] == 1
        assert await layer.delete_feedback(identity)
        assert "feedback" not in state.thread("b")["steps"][0]
        assert not await layer.delete_feedback(identity)
        result = await layer.list_threads(
            Pagination(first=10), ThreadFilter(userId=alice, feedback=1)
        )
        assert result.data == []
        await layer.update_thread("a", name="Renamed")
        assert (await layer.get_thread("a"))["name"] == "Renamed"
        await layer.delete_thread("b")
        assert await layer.get_thread("b") is None

    asyncio.run(exercise())


def test_favorite_answers_are_scoped_to_thread_owner(tmp_path):
    state = UIState(tmp_path / "ui.sqlite3")
    alice = state.save_user("alice", {})["id"]
    bob = state.save_user("bob", {})["id"]
    state.update_thread("a", "Alice", alice)
    state.update_thread("b", "Bob", bob)
    for identity in ("a", "b"):
        state.put_step({"id": identity, "threadId": identity, "metadata": {"favorite": True}})
    layer = LocalDataLayer(state)
    assert [s["id"] for s in asyncio.run(layer.get_favorite_steps(alice))] == ["a"]
