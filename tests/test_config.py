from pathlib import Path

import pytest
from pydantic import ValidationError

from personal_harness import ConfigError, load_config
from tests import mock_provider


def test_resolution_layers_keep_base_and_replace_arrays(config_file: Path) -> None:
    text = config_file.read_text("utf-8").replace('display_name = "You"', 'display_name = "Alex"')
    config_file.write_text(text, encoding="utf-8")
    mobile = load_config(
        config_file,
        profile="mobile",
        overrides={"preferences": {"city": "Hangzhou"}, "tools": {"allowed": []}},
    )
    desktop = load_config(config_file, profile="desktop")

    assert mobile.config.user.display_name == "Alex"
    assert mobile.config.preferences.language == "zh-CN"
    assert mobile.config.preferences.city == "Hangzhou"
    assert mobile.config.preferences.priorities == ("local_life", "control_cost")
    assert mobile.config.tools.allowed == ()
    assert mobile.config.model.max_output_tokens == 512
    assert str(mobile.config.budget.max_task_usd) == "0.03"
    assert desktop.config.preferences.response_style == "detailed"
    assert desktop.config.preferences.city is None
    assert desktop.config.tools.allowed == ("context.describe",)
    assert mobile.sources["user.display_name"] == "base"
    assert mobile.sources["model.max_output_tokens"] == "profile:mobile"
    assert mobile.sources["preferences.city"] == "override"
    assert mobile.sources["tools.allowed"] == "override"


def test_minimal_config_uses_defaults_and_tracks_their_origin(tmp_path: Path) -> None:
    path = tmp_path / "minimal.toml"
    path.write_text('schema_version = 1\ndefault_profile = "mine"\n[profiles.mine]\n', "utf-8")
    resolved = load_config(path)
    assert resolved.profile == "mine"
    assert resolved.config.preferences.response_style == "concise"
    assert resolved.sources["preferences.response_style"] == "default"
    assert resolved.config.tools.preview_mode == "full"
    assert resolved.sources["tools.preview_mode"] == "default"


def test_workspace_is_relative_to_config_not_process_directory(
    config_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = config_file.read_text("utf-8").replace('root = "."', 'root = "project"')
    config_file.write_text(text, "utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    resolved = load_config(config_file)
    assert Path(resolved.config.workspace.root) == (config_file.parent / "project").resolve()
    overridden = load_config(config_file, overrides={"workspace": {"root": "other"}})
    assert Path(overridden.config.workspace.root) == (config_file.parent / "other").resolve()


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("schema_version = 1", "schema_version = 2", "schema_version"),
        ("schema_version = 1", "schema_version = true", "schema_version"),
        ("schema_version = 1", "schema_version = 1.0", "schema_version"),
        ('default_profile = "desktop"', 'default_profile = "absent"', "Unknown default"),
        ('response_style = "detailed"', 'respnse_style = "detailed"', "respnse_style"),
        ('max_task_usd = "0.03"', 'max_task_usd = "-0.01"', "max_task_usd"),
        ('max_task_usd = "0.03"', 'max_task_usd = "NaN"', "finite"),
        ('allowed = ["context.describe"]', 'allowed = ["shell"]', "allowed"),
        ("max_steps = 8", "max_steps = 0", "max_steps"),
        ("max_steps = 8", "max_steps = true", "max_steps"),
        ("offline = true", 'offline = "true"', "offline"),
        ('provider = "fake"', 'provider = "openai"', "provider"),
    ],
)
def test_invalid_config_fails_before_execution(
    config_file: Path, old: str, new: str, message: str
) -> None:
    config_file.write_text(config_file.read_text("utf-8").replace(old, new), "utf-8")
    with pytest.raises(ConfigError, match=message):
        load_config(config_file)


def test_unselected_profile_is_validated(config_file: Path) -> None:
    config_file.write_text(
        config_file.read_text("utf-8") + '\n[profiles.mobile.workspace]\nrooot = "."\n', "utf-8"
    )
    with pytest.raises(ConfigError, match="Invalid profile mobile.*rooot"):
        load_config(config_file, profile="desktop")


def test_base_errors_cannot_be_hidden_by_valid_profile_override(config_file: Path) -> None:
    config_file.write_text(
        config_file.read_text("utf-8").replace(
            'response_style = "concise"', 'response_style = "bad"'
        ),
        "utf-8",
    )
    with pytest.raises(ConfigError, match="Invalid base"):
        load_config(config_file, profile="desktop")


def test_unsupported_override_is_not_silently_ignored(config_file: Path) -> None:
    with pytest.raises(ConfigError, match="preferences.langauge"):
        load_config(config_file, overrides={"preferences": {"langauge": "en"}})


def test_unknown_profile_is_an_error(config_file: Path) -> None:
    with pytest.raises(ConfigError, match="Unknown profile"):
        load_config(config_file, profile="unknown")


def test_resolved_settings_are_immutable(config_file: Path) -> None:
    config = load_config(config_file)
    with pytest.raises(ValidationError, match="frozen"):
        config.config.preferences.language = "en"
    with pytest.raises(TypeError):
        config.sources["preferences.language"] = "changed"  # type: ignore[index]


def test_errors_do_not_echo_private_input_values(config_file: Path) -> None:
    secret = "private-value-not-for-error-output"
    with pytest.raises(ConfigError) as error:
        load_config(config_file, overrides={"preferences": {"response_style": secret}})
    assert secret not in str(error.value)
    assert "response_style" in str(error.value)


def test_malformed_toml_and_missing_files_have_domain_errors(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text('schema_version = "unterminated', "utf-8")
    with pytest.raises(ConfigError, match="Invalid TOML"):
        load_config(path)
    with pytest.raises(ConfigError, match="Cannot read config"):
        load_config(tmp_path / "missing.toml")


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"runtime": {"offline": True}}, "offline=false"),
        ({"budget": {"currency": "USD"}}, "CNY"),
        ({"budget": {"currency": "CNY", "max_task_cny": None}}, "explicit CNY"),
        ({"model": {"provider": "openai-compatible", "model": "offline-demo"}}, "Model must match"),
    ],
)
def test_paid_execution_requires_consistent_model_and_explicit_limits(
    config_file: Path,
    changes: dict,
    message: str,
) -> None:
    overrides = dict(
        model=dict(**mock_provider.settings("test-chat")),
        runtime=dict(offline=False),
        budget=dict(currency="CNY", max_task_cny="0.5", max_daily_cny="2"),
    )
    overrides.update(changes)
    with pytest.raises(ConfigError, match=message):
        load_config(config_file, overrides=overrides)


def test_online_request_timeout_must_fit_overall_task_timeout(config_file: Path) -> None:
    with pytest.raises(ConfigError, match="request timeout cannot exceed"):
        load_config(
            config_file,
            overrides={
                "model": {**mock_provider.settings("test-chat")},
                "runtime": {
                    "offline": False,
                    "request_timeout_seconds": 90,
                    "timeout_seconds": 60,
                },
                "budget": {
                    "currency": "CNY",
                    "max_task_cny": "1",
                    "max_daily_cny": "2",
                },
            },
        )


@pytest.mark.parametrize("limit", ["-0.01", "NaN", "Infinity"])
def test_total_budget_rejects_invalid_amounts(config_file: Path, limit: str) -> None:
    with pytest.raises(ConfigError, match="max_total_cny"):
        load_config(config_file, overrides={"budget": {"max_total_cny": limit}})


def test_total_budget_is_optional_and_has_config_provenance(config_file: Path) -> None:
    assert load_config(config_file).config.budget.max_total_cny is None
    config_file.write_text(
        config_file.read_text("utf-8").replace(
            'max_daily_usd = "1.00"', 'max_daily_usd = "1.00"\nmax_total_cny = "1.00"'
        ),
        "utf-8",
    )
    base = load_config(config_file)
    assert str(base.config.budget.max_total_cny) == "1.00"
    assert base.sources["budget.max_total_cny"] == "base"
    zero = load_config(config_file, overrides={"budget": {"max_total_cny": "0"}})
    assert zero.config.budget.max_total_cny == 0
    assert zero.sources["budget.max_total_cny"] == "override"


@pytest.mark.parametrize(
    "retries,message",
    [(4, "provider_retries"), (-1, "provider_retries")],
)
def test_provider_retry_count_is_bounded(config_file: Path, retries: int, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_config(
            config_file,
            overrides={
                "model": {**mock_provider.settings("test-chat")},
                "runtime": {"offline": False, "provider_retries": retries},
                "budget": {
                    "currency": "CNY",
                    "max_task_cny": "1",
                    "max_daily_cny": "2",
                },
            },
        )


def test_preview_mode_is_validated_and_resolved_per_profile(config_file: Path) -> None:
    config_file.write_text(
        config_file.read_text("utf-8") + '\n[profiles.mobile.tools]\npreview_mode = "summary"\n',
        "utf-8",
    )
    mobile = load_config(config_file, profile="mobile")
    assert mobile.config.tools.preview_mode == "summary"
    assert mobile.sources["tools.preview_mode"] == "profile:mobile"
    assert load_config(config_file, profile="desktop").config.tools.preview_mode == "full"
    assert (
        load_config(
            config_file, profile="mobile", overrides={"tools": {"preview_mode": "full"}}
        ).sources["tools.preview_mode"]
        == "override"
    )
    with pytest.raises(ConfigError, match="preview_mode"):
        load_config(config_file, overrides={"tools": {"preview_mode": "silent"}})
