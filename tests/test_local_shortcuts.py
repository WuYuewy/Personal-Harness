import pytest
from pydantic import ValidationError

from personal_harness.ui_state import LocalShortcut, UIState


def test_shortcuts_persist_and_are_scoped_to_user_and_profile(tmp_path):
    path = tmp_path / "ui.sqlite3"
    state = UIState(path)
    shortcut = LocalShortcut(name="项目风险", command="/find", args=("风险",))
    state.save_local_shortcut("alice", "desktop", shortcut, None)
    identity = state.local_shortcuts("alice", "desktop")[0]["id"]
    assert UIState(path).local_shortcut("alice", "desktop", identity) == shortcut
    for user, profile in (("bob", "desktop"), ("alice", "mobile")):
        assert state.local_shortcuts(user, profile) == []
        with pytest.raises(ValueError):
            state.local_shortcut(user, profile, identity)
        with pytest.raises(ValueError):
            state.save_local_shortcut(user, profile, shortcut, identity)
        with pytest.raises(ValueError):
            state.delete_local_shortcut(user, profile, identity)
    updated = LocalShortcut(name="附近图书馆", command="/places", args=("library",))
    state.save_local_shortcut("alice", "desktop", updated, identity)
    assert state.local_shortcut("alice", "desktop", identity) == updated
    for _ in range(7):
        state.save_local_shortcut("alice", "desktop", shortcut, None)
    with pytest.raises(ValueError, match="8"):
        state.save_local_shortcut("alice", "desktop", shortcut, None)
    state.delete_local_shortcut("alice", "desktop", identity)
    assert len(state.local_shortcuts("alice", "desktop")) == 7
    with pytest.raises(ValueError):
        state.local_shortcut("alice", "desktop", identity)


@pytest.mark.parametrize(
    ("command", "args"),
    [
        ("/chat", ["hello"]),
        ("/find", [""]),
        ("/lookup", ["file.csv", "id"]),
        ("/lookup", ["file.csv", "id", ""]),
        ("/places", ["library\n/chat do something"]),
        ("/places", ["x" * 201]),
        ("/sum", ["expenses.csv", "amount", "status"]),
        ("/sum", ["expenses.csv", "amount", "status", "待报销", "extra"]),
        ("/actions", []),
        ("/actions", ["actions.md", "TODO:", "extra"]),
        ("/actions", ["actions.md", "x" * 81]),
    ],
)
def test_saved_operations_cannot_inject_model_requests(command, args):
    with pytest.raises(ValidationError):
        LocalShortcut(name="query", command=command, args=args)


def test_csv_fields_are_quoted_and_blank_category_remains_valid():
    import shlex

    args = ("my expense's.csv", "发票 编号", "INV ' 01")
    shortcut = LocalShortcut(name="票据", command="/lookup", args=args)
    assert tuple(shlex.split(shortcut.content())[1:]) == args
    assert LocalShortcut(name="附近", command="/places", args=("",)).content() == "/places"
    summary = LocalShortcut(name="报销合计", command="/sum", args=("expenses.csv", "金额", "", ""))
    assert tuple(shlex.split(summary.content())[1:]) == summary.args
    actions = LocalShortcut(name="项目待办", command="/actions", args=("my notes.md", "TODO:"))
    assert tuple(shlex.split(actions.content())[1:]) == actions.args
