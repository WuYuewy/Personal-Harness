"""Versioned personal configuration and deterministic profile resolution."""

from __future__ import annotations

import re
import tomllib
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Annotated, Any, Literal, Self
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

Text = Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1)]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]
Money = Annotated[Decimal, Field(ge=0, allow_inf_nan=False)]
BuiltinToolName = Literal[
    "context.describe",
    "workspace.list",
    "workspace.read",
    "workspace.read_many",
    "workspace.search",
    "workspace.write_preview",
    "workspace.write",
    "workspace.replace_text",
    "places.search",
]
ToolName = BuiltinToolName | Annotated[str, Field(pattern=r"^user\.[a-z][a-z0-9_]{0,63}$")]


class ConfigError(ValueError):
    """A configuration cannot be safely interpreted."""


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)


class UserSettings(Settings):
    display_name: Text = "You"


class Preferences(Settings):
    language: Text = "zh-CN"
    response_style: Literal["concise", "detailed"] = "concise"
    priorities: tuple[Text, ...] = ("control_cost",)
    city: Text | None = None


class ExperiencePreferences(Settings):
    preferred_name: Annotated[str, Field(strict=True, max_length=120)] = ""
    tone: Literal["neutral", "warm", "professional"] = "neutral"
    verbosity: Literal["brief", "balanced", "thorough"] = "balanced"
    answer_format: Literal["adaptive", "bullets", "steps", "prose"] = "adaptive"
    clarification: Literal["when_needed", "minimize"] = "when_needed"
    initiative: StrictBool = False
    cite_sources: StrictBool = True
    cost_strategy: Literal["prioritize_cost", "balanced", "prioritize_quality"] = "prioritize_cost"
    personal_instructions: Annotated[str, Field(strict=True, max_length=4000)] = ""


class ModelSettings(Settings):
    provider: Literal["fake", "openai-compatible"] = "fake"
    model: Text = "offline-demo"
    max_output_tokens: PositiveInt = 1024
    base_url: Text | None = None
    api_key_env: Text = "OPENAI_API_KEY"
    context_window: PositiveInt | None = None
    input_per_million_cny: Money | None = None
    output_per_million_cny: Money | None = None
    cache_read_per_million_cny: Money | None = None

    @model_validator(mode="after")
    def model_matches_provider(self) -> Self:
        if (self.provider == "fake") != (self.model == "offline-demo"):
            raise ValueError("Model must match the selected provider.")
        if self.provider == "openai-compatible":
            if self.base_url is None:
                raise ValueError("Custom provider requires a base_url.")
            url = urlsplit(self.base_url)
            if (
                url.scheme not in {"http", "https"}
                or not url.hostname
                or url.username
                or url.password
                or url.query
                or url.fragment
            ):
                raise ValueError("base_url must be an HTTP(S) URL without credentials or query.")
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.api_key_env):
                raise ValueError("api_key_env must be an environment variable name.")
            if any(
                value is None
                for value in (
                    self.context_window,
                    self.input_per_million_cny,
                    self.output_per_million_cny,
                )
            ):
                raise ValueError("Custom provider requires context_window and explicit CNY prices.")
            if self.context_window is not None and self.max_output_tokens > self.context_window:
                raise ValueError("Output token limit cannot exceed context_window.")
        return self


class BudgetSettings(Settings):
    currency: Literal["USD", "CNY"] = "USD"
    max_task_usd: Money = Decimal("0.10")
    max_daily_usd: Money = Decimal("1.00")
    max_task_cny: Money | None = None
    max_daily_cny: Money | None = None
    max_total_cny: Money | None = None


class WorkspaceSettings(Settings):
    root: Text = "."
    write_directory: Annotated[str, Field(strict=True, min_length=1)] | None = None
    read_cache_bytes: Annotated[int, Field(strict=True, ge=0, le=16_000_000)] = 0
    max_read_bytes: Annotated[int, Field(strict=True, gt=0, le=1_073_741_824)] = 104_857_600
    max_batch_files: Annotated[int, Field(strict=True, ge=1, le=16)] = 8
    max_batch_read_bytes: Annotated[int, Field(strict=True, gt=0, le=1_073_741_824)] = 104_857_600
    max_search_results: PositiveInt = 50
    max_search_result_chars: PositiveInt = 12_000
    max_write_bytes: PositiveInt = 256_000
    max_entries: PositiveInt = 200
    max_scan_entries: PositiveInt = 2_000
    max_search_files: PositiveInt = 200
    max_search_bytes: PositiveInt = 2_000_000
    max_preview_chars: PositiveInt = 12_000
    exclude: tuple[Text, ...] = (
        ".git",
        ".env",
        ".env.*",
        ".venv",
        ".python",
        ".uv-cache",
        ".personal-harness",
        ".chainlit",
        ".files",
        ".aws",
        ".ssh",
        "node_modules",
        "__pycache__",
        "personal-harness.toml",
    )

    @field_validator("write_directory")
    @classmethod
    def relative_write_directory(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.replace("\\", "/")
        path = PurePosixPath(normalized)
        windows = PureWindowsPath(normalized)
        if (
            value != value.strip()
            or "\x00" in value
            or path.is_absolute()
            or windows.drive
            or windows.root
            or ".." in path.parts
            or any(
                part.endswith((".", " "))
                or any(char in part for char in ':*?"<>|')
                or PureWindowsPath(part).is_reserved()
                for part in path.parts
            )
        ):
            raise ValueError("Use a relative write directory without traversal or device names.")
        return path.as_posix()


class LocalLifeSettings(Settings):
    data_file: Text | None = None
    latitude: Annotated[float, Field(strict=True, ge=-90, le=90, allow_inf_nan=False)] | None = None
    longitude: Annotated[float, Field(strict=True, ge=-180, le=180, allow_inf_nan=False)] | None = (
        None
    )
    max_distance_m: PositiveInt = 2_000
    max_price_cny: Money | None = None
    max_data_age_hours: PositiveInt = 24
    max_results: Annotated[int, Field(strict=True, ge=1, le=20)] = 5
    query_cache_seconds: Annotated[int, Field(strict=True, ge=0, le=3600)] = 0

    @model_validator(mode="after")
    def coordinates_are_a_pair(self) -> Self:
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError("Local-life latitude and longitude must be configured together.")
        return self


class ToolSettings(Settings):
    allowed: tuple[ToolName, ...] = ("context.describe",)
    preview_mode: Literal["full", "summary"] = "full"
    max_read_result_chars: Annotated[int, Field(strict=True, ge=0, le=2_000_000)] = 0


class RuntimeSettings(Settings):
    context_mode: Literal["configured", "task"] = "configured"
    compact_read_history: StrictBool = False
    history_read_mode: Literal["auto", "full", "references"] = "auto"
    max_steps: PositiveInt = 8
    max_tool_calls: PositiveInt = 8
    tool_retries: Annotated[int, Field(strict=True, ge=0)] = 1
    provider_retries: Annotated[int, Field(strict=True, ge=0, le=3)] = 2
    stream_responses: StrictBool = False
    request_timeout_seconds: PositiveInt = 45
    timeout_seconds: PositiveInt = 60
    offline: StrictBool = True
    history_turns: Annotated[int, Field(strict=True, ge=0, le=100)] = 12
    history_max_chars: Annotated[int, Field(strict=True, ge=1000, le=2_000_000)] = 48_000
    read_concurrency: Annotated[int, Field(strict=True, ge=1, le=16)] = 4
    max_model_upgrades: Annotated[int, Field(strict=True, ge=0, le=1)] = 0


class HarnessConfig(Settings):
    user: UserSettings = Field(default_factory=UserSettings)
    preferences: Preferences = Field(default_factory=Preferences)
    experience: ExperiencePreferences = Field(default_factory=ExperiencePreferences)
    model: ModelSettings = Field(default_factory=ModelSettings)
    fallback_model: ModelSettings | None = None
    budget: BudgetSettings = Field(default_factory=BudgetSettings)
    workspace: WorkspaceSettings = Field(default_factory=WorkspaceSettings)
    local_life: LocalLifeSettings = Field(default_factory=LocalLifeSettings)
    tools: ToolSettings = Field(default_factory=ToolSettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    instructions: tuple[Text, ...] = ()

    @property
    def effective_history_read_mode(self) -> Literal["full", "references"]:
        """Cost preference selects a real local policy; explicit chat choice wins."""
        mode = self.runtime.history_read_mode
        if mode == "auto":
            return "references" if self.experience.cost_strategy == "prioritize_cost" else "full"
        return mode

    @model_validator(mode="after")
    def paid_execution_is_explicit(self) -> Self:
        if self.fallback_model is not None:
            if self.model.provider == "fake" or self.fallback_model.provider == "fake":
                raise ValueError("Fallback models require explicit online primary and fallback.")
            if self.runtime.max_model_upgrades != 1:
                raise ValueError("Set max_model_upgrades=1 to authorize the fallback model.")
            if self.fallback_model.max_output_tokens != self.model.max_output_tokens:
                raise ValueError("Fallback output limit must match primary max_output_tokens.")
        elif self.runtime.max_model_upgrades:
            raise ValueError("Configure fallback_model before enabling model upgrades.")
        if self.model.provider != "fake":
            if self.runtime.offline:
                raise ValueError("Online provider execution requires runtime.offline=false.")
            if (
                self.budget.currency != "CNY"
                or self.budget.max_task_cny is None
                or self.budget.max_daily_cny is None
            ):
                raise ValueError("Online execution requires explicit CNY task and daily budgets.")
            if self.runtime.request_timeout_seconds > self.runtime.timeout_seconds:
                raise ValueError(
                    "Online execution request timeout cannot exceed the overall task timeout."
                )
        return self


class ConfigDocument(Settings):
    schema_version: Literal[1]
    default_profile: Text
    base: dict[str, Any] = Field(default_factory=dict)
    profiles: dict[str, dict[str, Any]] = Field(min_length=1)


@dataclass(frozen=True)
class ResolvedConfig:
    """A validated immutable run configuration, with per-value provenance."""

    config: HarnessConfig
    profile: str
    source_file: Path
    sources: Mapping[str, str]
    profiles: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "profile": self.profile,
            "available_profiles": list(self.profiles),
            "source_file": str(self.source_file),
            "config": self.config.model_dump(mode="json"),
        }


def _leaves(values: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(_leaves(value, path))
        else:
            result[path] = value
    return result


def _merge(base: dict[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            # Lists replace rather than append: an empty tool allowlist really disables tools.
            result[key] = deepcopy(value)
    return result


def _validation_error(label: str, error: ValidationError) -> ConfigError:
    # Never render raw input values: personal config may contain private information.
    fields = "; ".join(
        f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
        for item in error.errors(include_input=False, include_url=False)
    )
    return ConfigError(f"{label}: {fields}")


def _validate(values: dict[str, Any], label: str) -> HarnessConfig:
    try:
        return HarnessConfig.model_validate(values)
    except ValidationError as error:
        raise _validation_error(label, error) from None


def load_config(
    path: str | Path,
    *,
    profile: str | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> ResolvedConfig:
    """Resolve defaults < base < profile < explicit overrides.

    Nested tables merge recursively; arrays replace. Paths are anchored to the
    config file's directory, including paths supplied by explicit overrides.
    Every profile is validated so errors cannot hide in an inactive profile.
    """
    config_path = Path(path).expanduser().resolve()
    try:
        with config_path.open("rb") as file:
            raw = tomllib.load(file)
    except tomllib.TOMLDecodeError:
        raise ConfigError(
            f"Invalid TOML in {config_path}; check syntax and duplicate keys."
        ) from None
    except OSError as error:
        raise ConfigError(f"Cannot read config {config_path}: {error.strerror}") from None

    # bool and float compare equal to 1 in Python; schema versions must be actual integers.
    if type(raw.get("schema_version")) is not int or raw["schema_version"] != 1:
        raise ConfigError("schema_version must be the integer 1.")
    try:
        document = ConfigDocument.model_validate(raw)
    except ValidationError as error:
        raise _validation_error("Invalid document", error) from None

    defaults = HarnessConfig().model_dump(mode="json")
    base = _merge(defaults, document.base)
    _validate(base, "Invalid base")
    for name, patch in document.profiles.items():
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", name):
            raise ConfigError("Profile names must match [a-z][a-z0-9_-]*.")
        _validate(_merge(base, patch), f"Invalid profile {name}")
    if document.default_profile not in document.profiles:
        raise ConfigError(f"Unknown default profile: {document.default_profile}")
    selected = profile if profile is not None else document.default_profile
    if selected not in document.profiles:
        raise ConfigError(f"Unknown profile: {selected}")

    values = base
    sources = dict.fromkeys(_leaves(defaults), "default")
    sources.update(dict.fromkeys(_leaves(document.base), "base"))
    for label, layer in (
        (f"profile:{selected}", document.profiles[selected]),
        ("override", overrides if overrides is not None else {}),
    ):
        values = _merge(values, layer)
        sources.update(dict.fromkeys(_leaves(layer), label))

    config = _validate(values, f"Invalid resolved profile {selected}")
    workspace = Path(config.workspace.root).expanduser()
    if not workspace.is_absolute():
        workspace = config_path.parent / workspace
    local_life = config.local_life
    if local_life.data_file is not None:
        data_file = Path(local_life.data_file).expanduser()
        if not data_file.is_absolute():
            data_file = config_path.parent / data_file
        local_life = local_life.model_copy(update={"data_file": str(data_file.resolve())})
    config = config.model_copy(
        update={
            "workspace": config.workspace.model_copy(update={"root": str(workspace.resolve())}),
            "local_life": local_life,
        }
    )
    return ResolvedConfig(
        config, selected, config_path, MappingProxyType(sources), tuple(document.profiles)
    )
