from importlib.resources import files
from pathlib import Path

import pytest


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    path = tmp_path / "personal.toml"
    path.write_text(
        files("personal_harness").joinpath("templates/personal.toml").read_text("utf-8"),
        encoding="utf-8",
    )
    return path
