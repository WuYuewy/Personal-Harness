import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from personal_harness import Agent, ConfigError, load_config
from personal_harness.context import RunDependencies, model_context, task_instructions
from personal_harness.places import (
    PlacesCache,
    _load_places_data,
    query_local_places,
    search_places,
)


def test_query_cache_is_exact_bounded_invalidates_updates_and_never_hides_expiry(
    config_file, tmp_path, monkeypatch
):
    from datetime import timedelta

    import personal_harness.places as places
    from personal_harness.context import CURRENT_METRICS, RunMetrics

    source = _dataset(tmp_path)
    resolved = _resolved(config_file, source, query_cache_seconds=60)
    cache = PlacesCache()
    original = places._load_places_data
    loads = []

    def load(path):
        loads.append(path)
        return original(path)

    monkeypatch.setattr(places, "_load_places_data", load)
    metrics = RunMetrics()
    token = CURRENT_METRICS.set(metrics)
    try:
        first = query_local_places(resolved, category="餐厅", cache=cache)
        assert query_local_places(resolved, category="餐厅", cache=cache) == first
        assert len(loads) == 1 and metrics.places_cache_hits == 1
        query_local_places(resolved, category="咖啡", cache=cache)
        assert len(loads) == 2
        # A separate Agent's cache cannot reuse another Agent's query.
        query_local_places(resolved, category="餐厅", cache=PlacesCache())
        assert len(loads) == 3
        data = json.loads(source.read_text("utf-8"))
        data["places"][0]["price_cny"] = 999
        source.write_text(json.dumps(data), "utf-8")
        updated = query_local_places(resolved, category="餐厅", cache=cache)
        assert first != updated and len(loads) == 4
        # An aged source must stay stale on repeated queries, without a cached fresh result.
        data["fetched_at"] = (datetime.now(UTC) - timedelta(days=3)).isoformat()
        source.write_text(json.dumps(data), "utf-8")
        for _ in range(2):
            assert json.loads(query_local_places(resolved, cache=cache))["stale"]
        assert len(loads) == 6
        cache.clear()
        assert not cache._entries
    finally:
        CURRENT_METRICS.reset(token)


def test_query_cache_expiry_and_removal_are_checked_on_hit(config_file, tmp_path, monkeypatch):
    import personal_harness.places as places

    source = _dataset(tmp_path)
    resolved = _resolved(config_file, source, query_cache_seconds=10)
    cache = PlacesCache()
    monkeypatch.setattr(places, "monotonic", lambda: 0)
    query_local_places(resolved, cache=cache)
    monkeypatch.setattr(places, "monotonic", lambda: 11)
    # Reload after expiration; the entry gets a new monotonic deadline.
    query_local_places(resolved, cache=cache)
    assert next(iter(cache._entries.values()))[1] == 21
    source.unlink()
    with pytest.raises(ValueError, match="Cannot read"):
        query_local_places(resolved, cache=cache)


def _dataset(tmp_path: Path, *, city="合肥", fetched_at=None, places=None) -> Path:
    path = tmp_path / "places.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source": "user-curated-demo",
                "fetched_at": fetched_at or datetime.now(UTC).isoformat(),
                "city": city,
                "places": places
                or [
                    {
                        "id": "near",
                        "name": "近处素食",
                        "category": "餐厅",
                        "price_cny": 25,
                        "latitude": 31.001,
                        "longitude": 121.0,
                        "open_status": "open",
                    },
                    {
                        "id": "far",
                        "name": "远处餐厅",
                        "category": "餐厅",
                        "price_cny": 12,
                        "latitude": 31.02,
                        "longitude": 121.0,
                        "open_status": "unknown",
                    },
                    {
                        "id": "costly",
                        "name": "高价餐厅",
                        "category": "餐厅",
                        "price_cny": 101,
                        "latitude": 31.0001,
                        "longitude": 121.0,
                        "open_status": "open",
                    },
                    {
                        "id": "cafe",
                        "name": "近处咖啡",
                        "category": "咖啡",
                        "price_cny": 18,
                        "latitude": 31.0002,
                        "longitude": 121.0,
                        "open_status": "closed",
                    },
                ],
            },
            ensure_ascii=False,
        ),
        "utf-8",
    )
    return path


def _resolved(config_file: Path, data: Path, **settings):
    return load_config(
        config_file,
        overrides={
            "preferences": {"city": "合肥"},
            "local_life": {
                "data_file": str(data),
                "latitude": 31.0,
                "longitude": 121.0,
                "max_distance_m": 1_000,
                "max_price_cny": "40",
                "max_results": 3,
                **settings,
            },
            "tools": {"allowed": ["places.search"]},
        },
    )


def _place_result(config_file, data, **settings):
    captured = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert [tool.name for tool in info.function_tools] == ["places_search"]
        returns = [
            part.content
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        if returns:
            for item in returns:
                try:
                    captured.append(json.loads(item))
                except (TypeError, ValueError):
                    captured.append({"error": item})
            return ModelResponse([TextPart("Options found")])
        return ModelResponse([ToolCallPart("places_search", {"category": "餐厅"})])

    async def exercise():
        async with Agent(
            _resolved(config_file, data, **settings), model=FunctionModel(respond)
        ) as agent:
            return await agent.run("Find a local place")

    result = asyncio.run(exercise())
    assert result.available_tools == ("places.search",)
    return result, captured[0]


def test_local_places_filter_in_sdk_and_calculate_distance_without_network(config_file, tmp_path):
    data = _dataset(tmp_path)
    result, payload = _place_result(config_file, data)
    assert result.requests == 2 and result.cost == 0
    assert [place["name"] for place in payload["places"]] == ["近处素食"]
    assert 100 <= payload["places"][0]["distance_m"] <= 120
    assert payload["city_matches"] and not payload["stale"]
    assert payload["source"] == "user-curated-demo"
    assert payload["current_status_verified"] is False
    assert payload["places"][0]["current_status_verified"] is False


def test_local_places_constraints_change_per_personal_profile(config_file, tmp_path):
    data = _dataset(tmp_path)
    _, payload = _place_result(config_file, data, max_price_cny="15", max_distance_m=3000)
    assert [place["name"] for place in payload["places"]] == ["远处餐厅"]
    assert payload["places"][0]["open_status"] == "unknown"


def test_unknown_prices_are_visible_without_a_ceiling_and_never_count_as_free(
    config_file, tmp_path
):
    base = {"category": "图书馆", "latitude": 31.001, "longitude": 121.0}
    path = _dataset(
        tmp_path,
        places=[
            {**base, "id": "a", "name": "未知价格馆", "address": "45 Pearl St"},
            {**base, "id": "b", "name": "已知免费馆", "price_cny": 0},
            {**base, "id": "c", "name": "未知价格咖啡", "category": "咖啡"},
            {**base, "id": "d", "name": "超出范围馆", "latitude": 31.05},
        ],
    )
    uncapped = _resolved(config_file, path, max_price_cny=None)
    result = json.loads(query_local_places(uncapped, category="图书馆"))
    assert [p["id"] for p in result["places"]] == ["b", "a"]
    assert result["places"][1]["price_cny"] is None
    assert result["places"][1]["address"] == "45 Pearl St"
    assert not result["unknown_price_excluded"]

    free_only = _resolved(config_file, path, max_price_cny="0")
    result = json.loads(query_local_places(free_only, category="图书馆"))
    assert [p["id"] for p in result["places"]] == ["b"]
    assert result["unknown_price_excluded"] == 1


def test_category_aliases_match_exactly_and_cannot_shadow_a_category(config_file, tmp_path):
    path = _dataset(tmp_path)
    data = json.loads(path.read_text("utf-8"))
    data["category_aliases"] = {"restaurant": "餐厅"}
    path.write_text(json.dumps(data), "utf-8")
    resolved = _resolved(config_file, path)
    matching = json.loads(query_local_places(resolved, category=" Restaurant "))
    assert [p["id"] for p in matching["places"]] == ["near"]
    assert json.loads(query_local_places(resolved, category="rest"))["places"] == []

    data["category_aliases"]["咖啡"] = "餐厅"
    path.write_text(json.dumps(data), "utf-8")
    with pytest.raises(ValueError, match="distinct"):
        _load_places_data(str(path))


def test_local_places_report_city_mismatch_and_stale_source(config_file, tmp_path):
    data = _dataset(
        tmp_path,
        city="杭州",
        fetched_at="2020-01-01T00:00:00+00:00",
    )
    _, payload = _place_result(config_file, data)
    assert payload["city_matches"] is False and payload["places"] == []
    assert payload["stale"] is True and payload["current_status_verified"] is False


def test_local_places_configuration_is_opt_in_and_tool_subset_is_enforced(config_file, tmp_path):
    data = _dataset(tmp_path)
    resolved = _resolved(config_file, data)
    assert load_config(config_file).config.tools.allowed == ("context.describe",)
    scoped = load_config(
        config_file,
        overrides={
            "runtime": {"context_mode": "task"},
            "preferences": {"city": "合肥"},
            "local_life": {"data_file": str(data), "latitude": 31, "longitude": 121},
            "tools": {"allowed": ["places.search"]},
        },
    )
    assert model_context(scoped, ()).get("local_life") is None
    instructions = task_instructions(RunDependencies(resolved, selected_tools=("places.search",)))
    assert "places_search" in instructions and "current" in instructions

    calls = 0

    def respond(messages, info):
        nonlocal calls
        assert info.function_tools == []
        calls += 1
        return ModelResponse([TextPart("No local tool selected")])

    async def exercise():
        async with Agent(resolved, model=FunctionModel(respond)) as agent:
            return await agent.run("Do not access local places", tools=[])

    result = asyncio.run(exercise())
    assert calls == 1 and result.tool_calls == ()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"latitude": 91, "longitude": 10}, "latitude"),
        ({"longitude": None}, "configured together"),
        ({"longitude": 200}, "longitude"),
        ({"max_distance_m": 0}, "max_distance_m"),
        ({"max_results": 21}, "max_results"),
        ({"max_data_age_hours": 0}, "max_data_age_hours"),
        ({"max_price_cny": "NaN"}, "finite"),
    ],
)
def test_local_places_settings_are_validated(config_file, tmp_path, overrides, message):
    data = _dataset(tmp_path)
    with pytest.raises(ConfigError, match=message):
        _resolved(config_file, data, **overrides)


def test_local_places_data_is_size_and_schema_limited(config_file, tmp_path):
    data = _dataset(tmp_path)
    data.write_text("{" + " " * 1_000_001, "utf-8")
    with pytest.raises(ValueError, match="1 MB"):
        _load_places_data(str(data))


def test_relative_data_path_profile_sources_and_real_cli(config_file, tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from personal_harness.cli import app

    _dataset(tmp_path)
    config_file.write_text(
        config_file.read_text("utf-8")
        + """
[profiles.places.preferences]
city = "合肥"
[profiles.places.local_life]
data_file = "places.json"
latitude = 31.0
longitude = 121.0
max_price_cny = "40"
max_results = 1
[profiles.places.tools]
allowed = ["places.search"]
""",
        "utf-8",
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    resolved = load_config(config_file, profile="places")
    assert resolved.config.local_life.data_file == str(tmp_path / "places.json")
    assert resolved.sources["local_life.max_price_cny"] == "profile:places"
    runner = CliRunner()
    result = runner.invoke(
        app, ["places", "-c", str(config_file), "-p", "places", "--category", "餐厅"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert len(payload["places"]) == 1
    assert payload["places"][0]["name"] == "近处素食"
    assert "latitude" not in result.stdout and "longitude" not in result.stdout
    assert not (tmp_path / ".personal-harness").exists()
    checked = runner.invoke(app, ["doctor", "-c", str(config_file), "-p", "places"])
    assert checked.exit_code == 0, checked.output
    assert json.loads(checked.stdout)["workspace"]["local_places"]["available"]
    (tmp_path / "places.json").write_text("{}", "utf-8")
    invalid = runner.invoke(app, ["doctor", "-c", str(config_file), "-p", "places"])
    assert invalid.exit_code == 1
    assert not json.loads(invalid.stdout)["workspace"]["local_places"]["available"]


@pytest.mark.parametrize("problem", ["empty", "duplicate", "naive", "boolean", "bad_version"])
def test_malformed_source_is_rejected_without_exposing_data(tmp_path, problem):
    path = _dataset(tmp_path)
    raw = json.loads(path.read_text("utf-8"))
    if problem == "empty":
        raw["source"] = " "
    elif problem == "duplicate":
        raw["places"][1]["id"] = raw["places"][0]["id"]
    elif problem == "naive":
        raw["fetched_at"] = "2026-11-19T00:00:00"
    elif problem == "boolean":
        raw["places"][0]["latitude"] = True
    else:
        raw["schema_version"] = 2
    path.write_text(json.dumps(raw), "utf-8")
    with pytest.raises(ValueError):
        _load_places_data(str(path))


def test_limits_categories_future_and_no_match(config_file, tmp_path):
    path = _dataset(tmp_path, fetched_at="2999-01-01T00:00:00Z")
    resolved = _resolved(config_file, path, max_results=1)
    result = json.loads(query_local_places(resolved, limit=20))
    assert len(result["places"]) == 1 and result["truncated"]
    assert result["stale"] and result["timestamp_in_future"]
    assert result["distance_basis"] == "straight_line_wgs84"
    assert json.loads(query_local_places(resolved, category="missing"))["places"] == []
    assert (
        json.loads(query_local_places(_resolved(config_file, path, max_price_cny="0")))["places"]
        == []
    )


def test_query_errors_and_model_retry_do_not_expose_source_path(config_file, tmp_path):
    from pydantic_ai import ModelRetry

    path = _dataset(tmp_path)
    resolved = _resolved(config_file, path)
    path.write_text("private invalid material", "utf-8")
    with pytest.raises(ModelRetry) as failure:
        search_places(SimpleNamespace(deps=RunDependencies(resolved)))
    assert str(path) not in str(failure.value) and "private" not in str(failure.value)
    with pytest.raises(ValueError, match="Enable places.search"):
        query_local_places(load_config(config_file))


@pytest.mark.parametrize("setting", ["data_file", "latitude", "city"])
def test_missing_settings_stop_without_query(config_file, tmp_path, setting):
    path = _dataset(tmp_path)
    changes = {setting: None}
    overrides = {"preferences": changes} if setting == "city" else {"local_life": changes}
    if setting == "latitude":
        overrides["local_life"]["longitude"] = None
    resolved = _resolved(config_file, path)
    config = resolved.config
    from dataclasses import replace

    if "preferences" in overrides:
        config = config.model_copy(
            update={"preferences": config.preferences.model_copy(update=changes)}
        )
    else:
        config = config.model_copy(
            update={"local_life": config.local_life.model_copy(update=changes)}
        )
    with pytest.raises(ValueError, match="Configure"):
        query_local_places(replace(resolved, config=config))
