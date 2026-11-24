import sqlite3
from contextlib import closing
from uuid import uuid4

from typer.testing import CliRunner

from personal_harness.cli import app
from personal_harness.ui_state import UIState


def test_backup_restores_wal_state_and_assets_without_credentials(config_file):
    state_dir = config_file.parent / ".personal-harness"
    source = UIState(state_dir / "ui.sqlite3")
    source.save_scene("local", "Cowork", base_profile="desktop", defaults={"workspace": "notes"})
    assert source.authenticate("local", "original-local-password")
    original_jwt = source.auth_secret()
    asset_key = uuid4().hex
    assets = state_dir / "ui-assets"
    assets.mkdir()
    (assets / asset_key).write_bytes(b"attachment")
    (assets / "orphan").write_bytes(b"unused")
    (config_file.parent / ".env").write_text("OPENAI_API_KEY=environment-secret", "utf-8")
    destination = config_file.parent / "new-install"
    runner = CliRunner()

    # Keep connections open so the backup must see committed rows in the WAL.
    with (
        closing(sqlite3.connect(source.path)) as ui,
        closing(sqlite3.connect(state_dir / "ledger.sqlite3")) as ledger,
    ):
        ledger.execute("PRAGMA journal_mode=WAL")
        ledger.execute("CREATE TABLE example (amount TEXT)")
        ledger.execute("INSERT INTO example VALUES ('0.013')")
        ledger.commit()
        ui.execute(
            "INSERT INTO connections VALUES ('key', 'local', '{}', ?)",
            (b"stored-credential",),
        )
        ui.execute(
            "INSERT INTO elements VALUES ('asset', 'thread', json_object('objectKey', ?))",
            (asset_key,),
        )
        ui.commit()

        result = runner.invoke(app, ["data", "backup", str(destination), "-c", str(config_file)])
        assert result.exit_code == 0, result.output
        restored_dir = destination / ".personal-harness"
        assert (restored_dir / "ui-assets" / asset_key).read_bytes() == b"attachment"
        assert not (restored_dir / "ui-assets" / "orphan").exists()
        assert not (destination / ".env").exists()
        assert b"stored-credential" not in (restored_dir / "ui.sqlite3").read_bytes()
        assert original_jwt.encode() not in (restored_dir / "ui.sqlite3").read_bytes()
        copied = UIState(restored_dir / "ui.sqlite3")
        (scene,) = copied.scenes("local")
        assert copied.defaults("local", scene["id"]) == {"workspace": "notes"}
        assert not copied.has_password() and source.has_password()
        with closing(sqlite3.connect(restored_dir / "ledger.sqlite3")) as restored:
            assert restored.execute("SELECT amount FROM example").fetchone() == ("0.013",)
        assert (destination / "personal-harness.toml").read_bytes() == config_file.read_bytes()
        assert (
            runner.invoke(
                app, ["data", "backup", str(destination), "-c", str(config_file)]
            ).exit_code
            == 2
        )
