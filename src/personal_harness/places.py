"""Local places data filtering; no external API or location upload."""

from __future__ import annotations

import json
import math
import os
import stat
from collections import OrderedDict
from datetime import UTC, datetime
from decimal import Decimal
from threading import RLock
from time import monotonic
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator
from pydantic_ai import ModelRetry, RunContext

from personal_harness.config import ResolvedConfig
from personal_harness.context import CURRENT_METRICS, RunDependencies


class PlacesCache:
    """At most 32 exact local queries, isolated to one Agent and its permissions."""

    def __init__(self) -> None:
        self._entries: OrderedDict[str, tuple[tuple[int, ...], float, datetime, str]] = (
            OrderedDict()
        )
        self._lock = RLock()

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def get(self, key: str, identity: tuple[int, ...], now: datetime) -> str | None:
        with self._lock:
            entry = self._entries.pop(key, None)
            if entry is None:
                return None
            if entry[0] != identity or monotonic() >= entry[1] or now >= entry[2]:
                return None
            self._entries[key] = entry
            metrics = CURRENT_METRICS.get()
            if metrics is not None:
                metrics.places_cache_hits += 1
            return entry[3]

    def put(
        self, key: str, identity: tuple[int, ...], seconds: int, fresh_until: datetime, value: str
    ) -> None:
        with self._lock:
            self._entries[key] = (identity, monotonic() + seconds, fresh_until, value)
            while len(self._entries) > 32:
                self._entries.popitem(last=False)


def _source_identity(path: str) -> tuple[int, ...]:
    # Reopen on every hit so changed access and missing/replaced sources are not cached away.
    before = os.stat(path, follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("Places data must be a regular file.")
    with open(path, "rb") as file:
        info = os.fstat(file.fileno())
    if (before.st_dev, before.st_ino) != (info.st_dev, info.st_ino):
        raise ValueError("Places source changed during inspection.")
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
        info.st_mode,
    )


Label = Annotated[
    str, StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=200)
]


class _Place(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: Label
    name: Label
    category: Label
    price_cny: Annotated[Decimal, Field(ge=0, allow_inf_nan=False)] | None = None
    address: Label | None = None
    latitude: Annotated[float, Field(ge=-90, le=90, allow_inf_nan=False)]
    longitude: Annotated[float, Field(ge=-180, le=180, allow_inf_nan=False)]
    open_status: Literal["open", "closed", "unknown"] = "unknown"


class _PlaceData(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[1]
    source: Label
    fetched_at: datetime
    city: Label
    places: Annotated[list[_Place], Field(max_length=5_000)]
    category_aliases: Annotated[dict[Label, Label], Field(max_length=100)] = Field(
        default_factory=dict
    )

    @field_validator("schema_version", mode="before")
    @classmethod
    def integer_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 1.")
        return value


def _load_places_data(path: str) -> _PlaceData:
    file_stat = os.stat(path, follow_symlinks=False)
    if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size > 1_000_000:
        raise ValueError("Places data must be a regular file no larger than 1 MB.")
    with open(path, "rb") as file:
        content = file.read(1_000_001)
    if len(content) > 1_000_000:
        raise ValueError("Places data must be no larger than 1 MB.")
    return _parse_places_data(content)


def _parse_places_data(content: str | bytes) -> _PlaceData:
    data = _PlaceData.model_validate_json(content)
    if data.fetched_at.utcoffset() is None:
        raise ValueError("Places source fetched_at must include a timezone.")
    if len({place.id for place in data.places}) != len(data.places):
        raise ValueError("Place identifiers must be unique.")
    categories = {place.category.strip().casefold() for place in data.places}
    aliases: dict[str, str] = {}
    for alias, target in data.category_aliases.items():
        alias_key, target_key = alias.strip().casefold(), target.strip().casefold()
        if not alias_key or alias_key in aliases or alias_key in categories:
            raise ValueError(
                "Category aliases must be distinct from each other and category names."
            )
        if target_key not in categories:
            raise ValueError("Category alias must point to an existing category.")
        aliases[alias_key] = target_key
    data.category_aliases = aliases
    return data


def search_places(
    ctx: RunContext[RunDependencies],
    category: str | None = None,
    limit: Annotated[int, Field(strict=True, ge=1, le=20)] | None = None,
) -> str:
    """Filter a user-configured local places JSON file by city, distance, budget and category.

    Distance is calculated locally from configured coordinates. No network lookup occurs.
    Null prices are unknown; a configured price ceiling excludes unknown prices.
    The source timestamp and open status are reported as supplied; neither proves current truth.
    """
    try:
        return query_local_places(
            ctx.deps.resolved, category=category, limit=limit, cache=ctx.deps.places_cache
        )
    except ValueError as error:
        raise ModelRetry(str(error)) from error


def query_local_places(
    resolved: ResolvedConfig,
    *,
    category: str | None = None,
    limit: int | None = None,
    cache: PlacesCache | None = None,
) -> str:
    """Apply local constraints without a model call; also used for CLI source inspection."""
    if "places.search" not in resolved.config.tools.allowed:
        raise ValueError("Enable places.search in tools.allowed before querying local places.")
    if limit is not None and (type(limit) is not int or not 1 <= limit <= 20):
        raise ValueError("Place limit must be an integer from 1 to 20.")

    settings = resolved.config.local_life
    city = resolved.config.preferences.city
    if settings.data_file is None:
        raise ValueError("Configure local_life.data_file before using places.search.")
    if city is None:
        raise ValueError("Configure preferences.city before searching local places.")
    if settings.latitude is None or settings.longitude is None:
        raise ValueError("Configure both local_life.latitude and longitude for distance filtering.")
    path = settings.data_file
    now = datetime.now(UTC)
    cache_key = json.dumps(
        [settings.model_dump(mode="json"), city, category, limit], sort_keys=True
    )
    identity: tuple[int, ...] | None = None
    try:
        if cache is not None and settings.query_cache_seconds:
            identity = _source_identity(path)
            cached = cache.get(cache_key, identity, now)
            if cached is not None:
                return cached
        data = _load_places_data(path)
    except (OSError, UnicodeError, ValueError) as error:
        raise ValueError(
            f"Cannot read configured local places data ({type(error).__name__})."
        ) from error
    if data.fetched_at.utcoffset() is None:
        raise ValueError("Places source fetched_at must include a timezone.")
    timestamp_in_future = data.fetched_at.astimezone(UTC) > now
    stale = (
        timestamp_in_future
        or (now - data.fetched_at.astimezone(UTC)).total_seconds()
        > settings.max_data_age_hours * 3600
    )
    origin_lat, origin_lon = settings.latitude, settings.longitude

    def distance_m(place: _Place) -> float:
        lat1, lat2 = math.radians(origin_lat), math.radians(place.latitude)
        delta_lat = lat2 - lat1
        delta_lon = math.radians(place.longitude - origin_lon)
        haversine = (
            math.sin(delta_lat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
        )
        return 6_371_000 * 2 * math.asin(math.sqrt(min(1, haversine)))

    max_distance = settings.max_distance_m
    max_price = settings.max_price_cny
    effective_limit = min(limit or settings.max_results, settings.max_results)
    category_key = category.strip().casefold() if category is not None else None
    if category_key is not None:
        category_key = data.category_aliases.get(category_key, category_key)
    matched = []
    unknown_price_excluded = 0
    for place in data.places:
        distance = distance_m(place)
        if data.city != city or distance > max_distance:
            continue
        if category_key is not None and place.category.strip().casefold() != category_key:
            continue
        if max_price is not None:
            if place.price_cny is None:
                unknown_price_excluded += 1
                continue
            if place.price_cny > max_price:
                continue
        matched.append((distance, place))
    matched.sort(
        key=lambda item: (item[0], item[1].price_cny is None, item[1].price_cny, item[1].id)
    )
    output = json.dumps(
        {
            "city_matches": data.city == city,
            "source": data.source,
            "fetched_at": data.fetched_at.isoformat(),
            "stale": stale,
            "timestamp_in_future": timestamp_in_future,
            "current_status_verified": False,
            "limits": {
                "max_distance_m": max_distance,
                "max_price_cny": str(max_price) if max_price is not None else None,
                "max_results": effective_limit,
            },
            "places": [
                {
                    **place.model_dump(
                        mode="json", exclude={"latitude", "longitude"}, exclude_none=True
                    ),
                    **({"price_cny": None} if place.price_cny is None else {}),
                    "distance_m": round(distance),
                    "city": data.city,
                    "source": data.source,
                    "fetched_at": data.fetched_at.isoformat(),
                    "current_status_verified": False,
                }
                for distance, place in matched[:effective_limit]
            ],
            "truncated": len(matched) > effective_limit,
            "unknown_price_excluded": unknown_price_excluded,
            "distance_basis": "straight_line_wgs84",
        },
        ensure_ascii=False,
    )
    if cache is not None and identity is not None and not stale:
        from datetime import timedelta

        if identity == _source_identity(path):
            cache.put(
                cache_key,
                identity,
                settings.query_cache_seconds,
                data.fetched_at.astimezone(UTC) + timedelta(hours=settings.max_data_age_hours),
                output,
            )
    return output
