"""Persisted conversations, owner isolation, secrets and single-user login."""

import json

import pytest
from pydantic import SecretStr
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from personal_harness.reports import ReportRecipe
from personal_harness.ui_bootstrap import prepare_ui
from personal_harness.ui_state import Connection, UIState


def connection():
    return Connection.model_validate(
        {
            "name": "Custom Qwen",
            "model": {
                "provider": "openai-compatible",
                "model": "custom-model-v1",
                "base_url": "https://example.test/v1",
                "context_window": 8192,
                "input_per_million_cny": "1",
                "output_per_million_cny": "2",
            },
        }
    )


def test_restart_restores_threads_history_search_and_metadata(tmp_path):
    path = tmp_path / "state/ui.sqlite3"
    state = UIState(path)
    alice = state.save_user("alice", {})
    bob = state.save_user("bob", {})
    state.update_thread(
        "a", "Aurora", alice["id"], {"settings": {"city": "合肥", "api_key": "hidden"}}
    )
    state.update_thread("b", "Private", bob["id"])
    state.put_step(
        {
            "id": "s",
            "threadId": "a",
            "type": "user_message",
            "output": "预算三千元",
            "waitForAnswer": True,
        }
    )
    history = [ModelRequest([UserPromptPart("Aurora")]), ModelResponse([TextPart("OK")])]
    state.save_history("a", history)
    reopened = UIState(path)
    assert len(reopened.history("a")) == 2 and reopened.history("b") == []
    assert [t["id"] for t in reopened.list_threads(alice["id"], "三千")] == ["a"]
    assert reopened.list_threads(alice["id"], "Private") == []
    thread = reopened.thread("a")
    assert thread["metadata"]["settings"] == {"city": "合肥"}
    assert thread["steps"][0]["waitForAnswer"] is False
    reopened.delete_thread("a")
    assert reopened.thread("a") is None and reopened.history("a") == []
    assert reopened.thread("b") is not None


def test_credentials_never_appear_in_public_connection_or_metadata(tmp_path):
    state = UIState(tmp_path / "ui.sqlite3")
    secret = "private-test-credential"
    identity = state.save_connection("alice", connection(), SecretStr(secret))
    assert secret not in json.dumps(state.connections("alice"))
    with pytest.raises(ValueError):
        state.connection("bob", identity)
    _, key = state.connection("alice", identity)
    assert key.get_secret_value() == secret and secret not in repr(key)
    state.save_connection("alice", connection(), None, identity)
    assert state.connection("alice", identity)[1].get_secret_value() == secret
    state.delete_connection("bob", identity)
    assert state.connections("alice")
    state.delete_connection("alice", identity)
    assert not state.connections("alice")


def test_personal_profile_is_per_user_and_only_persists_allowed_preferences(tmp_path):
    state = UIState(tmp_path / "ui.sqlite3")
    state.save_user("alice", {"theme": "dark"})
    state.save_user("bob", {})
    state.save_personal_profile(
        "alice",
        {
            "preferred_name": "Lin",
            "tone": "warm",
            "initiative": True,
            "personal_instructions": "对我简洁说明",
            "api_key": "must not persist",
            "tool_permissions": ["workspace.read"],
        },
    )
    assert state.personal_profile("alice") == {
        "preferred_name": "Lin",
        "tone": "warm",
        "initiative": True,
        "personal_instructions": "对我简洁说明",
    }
    assert state.personal_profile("bob") == {}
    assert state.user("alice")["metadata"]["theme"] == "dark"


def test_report_recipes_and_run_counts_are_isolated_and_delete_cleans_only_owner(tmp_path):
    state = UIState(tmp_path / "ui.sqlite3")
    state.save_user("alice", {})
    state.save_user("bob", {})
    recipe = ReportRecipe.model_validate(
        {
            "name": "Weekly brief",
            "sources": ["notes.md"],
            "target": "brief.md",
            "sections": [{"heading": "Tasks", "prefix": "TODO:"}],
        }
    )
    identity = state.save_report_recipe("alice", recipe, None)
    other = state.save_report_recipe("bob", recipe, None)
    with pytest.raises(ValueError, match="不存在"):
        state.report_recipe("bob", identity)
    with pytest.raises(ValueError, match="不存在"):
        state.delete_report_recipe("bob", identity)
    assert state.report_recipe("alice", identity) == recipe

    state.record_report_run("alice", identity, 2)
    reported = state.report_recipes("alice")[0]
    assert reported["runs"] == 1
    assert reported["model_requests"] == 0 and reported["cost_cny"] == "0"
    assert reported["last_run"]

    state.delete_report_recipe("alice", identity)
    assert state.report_recipes("alice") == []
    assert state.report_recipes("bob")[0]["id"] == other


def test_first_login_sets_password_persistently_and_never_replaces_it(tmp_path):
    state = UIState(tmp_path / "ui.sqlite3")
    assert not state.authenticate("local", "short")
    assert not state.authenticate("other", "test-password")
    assert not state.has_password()
    assert state.authenticate("local", "test-password")
    assert not state.authenticate("local", "wrong-password")
    again = UIState(state.path)
    assert again.authenticate("local", "test-password")
    assert state.auth_secret() == again.auth_secret()


def test_ui_bootstrap_preserves_unrelated_options_and_installs_assets(tmp_path):
    directory = tmp_path / "personal"
    directory.mkdir()
    (directory / ".chainlit").mkdir()
    (directory / ".chainlit/config.toml").write_text(
        '[UI]\nname = "Assistant"\ndefault_theme = "dark"\n'
        '[features.spontaneous_file_upload]\naccept = ["text/*"]\n',
        "utf-8",
    )
    (directory / "chainlit.md").write_text("My welcome", "utf-8")
    environment = prepare_ui(directory, "127.0.0.1")
    assert environment["CHAINLIT_AUTH_SECRET"]
    assert environment["PERSONAL_HARNESS_UI_PERSIST"] == "1"
    config = (directory / ".chainlit/config.toml").read_text("utf-8")
    assert 'name = "Personal Harness"' in config and 'default_theme = "light"' in config
    assert (directory / "chainlit.md").read_text("utf-8") == "My welcome"
    assert (directory / "public/elements/ModelConnections.jsx").exists()
    assert (directory / "public/harness.js").exists()
    from importlib.util import find_spec

    if find_spec("chainlit"):
        entrypoint = (directory / ".chainlit/frontend/index.html").read_text("utf-8")
        assert "https://" not in entrypoint
        assert "/public/vendor/katex/katex.min.css" in entrypoint
        assert (directory / "public/vendor/katex/fonts/KaTeX_Main-Regular.woff2").exists()
    assert 'custom_js = "/public/harness.js"' in config
    assert (directory / "public/elements/UserInsights.jsx").exists()
    assert (directory / "public/vendor/chart.umd.min.js").exists()
    assert (directory / "public/vendor/Chart.js.LICENSE.md").exists()
    assert environment == prepare_ui(directory, "127.0.0.1")
    import tomllib

    upload = tomllib.loads(config)["features"]["spontaneous_file_upload"]
    assert ".md" in upload["accept"]["application/octet-stream"]
    assert ".exe" not in upload["accept"]["application/octet-stream"]
    assert "application/pdf" in upload["accept"]
    assert tomllib.loads((directory / ".chainlit/config.toml").read_text("utf-8")) == tomllib.loads(
        config
    )


def test_remote_first_login_cannot_claim_local_account(tmp_path):
    with pytest.raises(ValueError, match="before allowing remote"):
        prepare_ui(tmp_path, "0.0.0.0")


def test_local_frontend_guards_unmounted_multiselect_list_without_editing_dependency(tmp_path):
    from personal_harness.ui_bootstrap import _prepare_frontend

    package = tmp_path / "installed"
    source = package / "frontend/dist"
    assets = source / "assets"
    assets.mkdir(parents=True)
    (source / "index.html").write_text("<html></html>", "utf-8")
    script = assets / "bundle.js"
    original = (
        "function V(){var W;return Array.from((W=A.current)==null?void 0:W.querySelectorAll(Z5e))}"
    )
    script.write_text(original, "utf-8")
    directory = tmp_path / "personal"
    _prepare_frontend(directory, package)
    local = directory / ".chainlit/frontend/assets/bundle.js"
    assert "??[]" in local.read_text("utf-8")
    assert script.read_text("utf-8") == original
    assert not local.samefile(script)
    _prepare_frontend(directory, package)
    assert script.read_text("utf-8") == original
    assert "??[]" in local.read_text("utf-8")


def test_local_frontend_replaces_assets_without_modifying_installed_files(tmp_path):
    from personal_harness.ui_bootstrap import _prepare_frontend

    package = tmp_path / "installed"
    source = package / "frontend/dist"
    (source / "assets").mkdir(parents=True)
    (source / "index.html").write_text(
        '<head><!-- FONT START --><link href="https://fonts.example/test">'
        '<!-- FONT END --><link rel="stylesheet" href="https://cdn.example/math.css">'
        '<!-- CSS INJECTION PLACEHOLDER --></head><script src="/assets/app.js"></script>',
        "utf-8",
    )
    original = source / "assets/app.js"
    original.write_text("original", "utf-8")
    root = tmp_path / "user"
    _prepare_frontend(root, package)
    local = root / ".chainlit/frontend"
    assert "https://" not in (local / "index.html").read_text("utf-8")
    assert 'src="/assets/app.js"' in (local / "index.html").read_text("utf-8")
    upgraded = source / "assets/new.js"
    upgraded.write_text("upgraded version", "utf-8")
    upgraded.replace(original)
    _prepare_frontend(root, package)
    assert (local / "assets/app.js").read_text("utf-8") == "upgraded version"
    assert original.read_text("utf-8") == "upgraded version"
