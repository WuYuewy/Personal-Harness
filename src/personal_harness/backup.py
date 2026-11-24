"""Make a portable copy of local settings, chat history, attachments and accounting."""

from __future__ import annotations

import re
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory


def _copy_database(source: Path, target: Path) -> None:
    with (
        closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as original,
        closing(sqlite3.connect(target)) as copy,
    ):
        original.backup(copy)


def backup_data(config: Path, destination: Path) -> Path:
    """Create a separate state directory. Stop UI and tasks before copying attachments."""
    destination = destination.absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("The backup destination already exists.")
    source = config.resolve().parent / ".personal-harness"
    if destination.resolve().is_relative_to(source.resolve()):
        raise ValueError("Choose a destination outside the active state directory.")

    with TemporaryDirectory(prefix=".harness-backup-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "backup"
        staging.mkdir(mode=0o700)
        state = staging / ".personal-harness"
        state.mkdir(mode=0o700)
        shutil.copyfile(config, staging / "personal-harness.toml")

        ledger = source / "ledger.sqlite3"
        if ledger.is_file():
            _copy_database(ledger, state / ledger.name)

        ui = source / "ui.sqlite3"
        if ui.is_file():
            scratch = Path(temporary) / ui.name
            _copy_database(ui, scratch)
            with closing(sqlite3.connect(scratch)) as copy:
                copy.execute("UPDATE connections SET credential=NULL")
                copy.execute("DELETE FROM ui_values WHERE id IN ('password', 'jwt')")
                assets = {
                    key
                    for (key,) in copy.execute(
                        "SELECT json_extract(data, '$.objectKey') FROM elements"
                    )
                    if isinstance(key, str) and re.fullmatch(r"[a-f0-9]{32}", key)
                }
                copy.commit()
                # A clean database image also omits old values from freed SQLite pages.
                copy.execute("VACUUM INTO ?", (str(state / ui.name),))
            if assets:
                asset_directory = source / "ui-assets"
                if asset_directory.is_symlink():
                    raise ValueError("The attachment directory must be a local directory.")
                (state / "ui-assets").mkdir(mode=0o700)
                for key in assets:
                    attachment = asset_directory / key
                    if attachment.is_symlink() or not attachment.is_file():
                        raise ValueError("A saved chat attachment is missing or invalid.")
                    shutil.copyfile(attachment, state / "ui-assets" / key)

        if destination.exists() or destination.is_symlink():
            raise FileExistsError("The backup destination already exists.")
        staging.rename(destination)
    return destination / "personal-harness.toml"
