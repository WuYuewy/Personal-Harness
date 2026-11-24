"""Start the installed UI from outside the checkout; check HTTP and personal profiles.

Run with a wheel installed with [ui]. Uses only loopback HTTP, no browser or model requests.
"""

import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from decimal import Decimal
from http.cookiejar import CookieJar
from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import HTTPCookieProcessor, ProxyHandler, Request, build_opener

import personal_harness
from personal_harness.backup import backup_data
from personal_harness.ledger import Ledger
from personal_harness.ui_state import LocalShortcut, UIState


def stop_server(server: subprocess.Popen) -> None:
    if os.name == "nt":
        # The launcher has a child Chainlit process; stop only this smoke's process tree.
        stopped = subprocess.run(
            ["taskkill", "/PID", str(server.pid), "/T", "/F"],
            capture_output=True,
            check=False,
        )
        if stopped.returncode and server.poll() is None:
            raise RuntimeError("Cannot stop UI smoke process tree: " + repr(stopped.stderr))
    else:
        try:
            os.killpg(server.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        server.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            os.killpg(server.pid, signal.SIGKILL)
        else:
            server.kill()
        server.wait(timeout=5)


def main(*, restore: bool = False) -> None:
    module = Path(personal_harness.__file__).resolve()
    assert "site-packages" in module.parts, f"Expected an installed package, got {module}"
    with TemporaryDirectory(prefix="personal-harness-ui-") as temporary:
        root = Path(temporary)
        personal = root / "personal settings"
        invocation = root / "invocation"
        personal.mkdir()
        invocation.mkdir()
        config = personal / "my-settings.toml"
        config.write_text(
            files("personal_harness").joinpath("templates/personal.toml").read_text("utf-8")
            + '\n[profiles.installed_cowork]\ninstructions = ["Use my project sources."]\n',
            "utf-8",
        )
        scene = None
        if restore:
            source = UIState(personal / ".personal-harness/ui.sqlite3")
            user = source.save_user("local", {})
            assert source.authenticate("local", "previous-smoke-password")
            source.auth_secret()
            source.save_scene("local", "恢复的项目", base_profile="desktop", defaults={"tools": []})
            scene = source.scenes("local")[0]["id"]
            source.save_local_shortcut(
                "local",
                scene,
                LocalShortcut(name="待办", command="/actions", args=("notes.md",)),
                None,
            )
            source.update_thread("restored-chat", "恢复的聊天", user["id"], {"profile": scene})
            asset_key = "a" * 32
            asset_dir = source.path.parent / "ui-assets"
            asset_dir.mkdir()
            (asset_dir / asset_key).write_bytes(b"restored attachment")
            with source.db() as db:
                db.execute(
                    "INSERT INTO elements VALUES (?, ?, ?)",
                    (
                        "restored-asset",
                        "restored-chat",
                        json.dumps(
                            {
                                "id": "restored-asset",
                                "objectKey": asset_key,
                                "name": "notes.txt",
                                "mime": "text/plain",
                                "url": "/harness/asset/restored-asset",
                            }
                        ),
                    ),
                )
            original_ledger = source.path.parent / "ledger.sqlite3"
            ledger = Ledger(original_ledger)
            ledger.start_run(
                "historical-run",
                provider="test",
                model="test",
                task_limit=Decimal("0.1"),
                daily_limit=Decimal("1"),
            )
            original_files = (config, source.path, original_ledger, asset_dir / asset_key)
            fingerprints = {p: hashlib.sha256(p.read_bytes()).digest() for p in original_files}
            config = backup_data(config, root / "restored personal settings")
            personal = config.parent
            restored = UIState(personal / ".personal-harness/ui.sqlite3")
            assert not restored.has_password() and source.has_password()
            assert restored.defaults("local", scene) == {"tools": []}
            assert len(restored.local_shortcuts("local", scene)) == 1
            ledger_fingerprint = hashlib.sha256(
                (restored.path.parent / "ledger.sqlite3").read_bytes()
            ).digest()
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        environment.pop("OPENAI_API_KEY", None)
        environment["PERSONAL_HARNESS_CONFIG"] = str(root / "absent.toml")
        environment["CHAINLIT_APP_ROOT"] = str(invocation)
        environment["PYTHONIOENCODING"] = "utf-8"
        # A private local server; do not open a browser or inherit a proxy for loopback.
        opener = build_opener(ProxyHandler({}), HTTPCookieProcessor(CookieJar()))
        with (root / "server.log").open("w+", encoding="utf-8") as log:
            server = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "personal_harness",
                    "ui",
                    "--config",
                    str(config),
                    "--port",
                    str(port),
                ],
                cwd=invocation,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                start_new_session=os.name != "nt",
            )
            try:
                deadline = time.monotonic() + 45
                while True:
                    if server.poll() is not None:
                        log.seek(0)
                        raise AssertionError("UI exited before HTTP readiness:\n" + log.read())
                    try:
                        with opener.open(f"http://127.0.0.1:{port}/", timeout=1) as response:
                            assert response.status == 200
                        break
                    except (URLError, TimeoutError):
                        if time.monotonic() >= deadline:
                            log.seek(0)
                            raise AssertionError(
                                "UI did not become ready:\n" + log.read()
                            ) from None
                        time.sleep(0.1)
                login = Request(
                    f"http://127.0.0.1:{port}/login",
                    data=urlencode(
                        {"username": "local", "password": "installed-smoke-password"}
                    ).encode(),
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
                with opener.open(login, timeout=10) as response:
                    assert response.status == 200
                with opener.open(
                    f"http://127.0.0.1:{port}/project/settings", timeout=10
                ) as response:
                    settings = json.load(response)
                assert [p["name"] for p in settings["chatProfiles"]] == [
                    "desktop",
                    "mobile",
                    "installed_cowork",
                ] + ([scene] if scene else [])
                if restore:
                    with opener.open(
                        f"http://127.0.0.1:{port}/project/thread/restored-chat", timeout=5
                    ) as response:
                        assert json.load(response)["name"] == "恢复的聊天"
                    with opener.open(
                        f"http://127.0.0.1:{port}/harness/asset/restored-asset", timeout=5
                    ) as response:
                        assert response.read() == b"restored attachment"
                with opener.open(f"http://127.0.0.1:{port}/", timeout=5) as response:
                    assert response.status == 200
                    assert "text/html" in response.headers["Content-Type"]
                assert (personal / ".chainlit/config.toml").is_file()
                assert not list(invocation.iterdir())
                assert (personal / ".personal-harness/ui.sqlite3").is_file()
                assert (personal / "public/elements/ModelConnections.jsx").is_file()
                assert (personal / "public/elements/LocalResults.jsx").is_file()
                assert (personal / "public/elements/PersonalScenes.jsx").is_file()
                assert (personal / "public/elements/ReportRecipes.jsx").is_file()
                assert (personal / "public/elements/UserInsights.jsx").is_file()
                assert (personal / "public/vendor/chart.umd.min.js").is_file()
                assert (personal / "public/vendor/Chart.js.LICENSE.md").is_file()
                with opener.open(
                    f"http://127.0.0.1:{port}/public/vendor/chart.umd.min.js", timeout=5
                ) as response:
                    assert response.status == 200
                    assert b"Chart.js v4.5.1" in response.read()
                assert (personal / "public/harness.css").is_file()
                if restore:
                    assert (
                        hashlib.sha256(
                            (personal / ".personal-harness/ledger.sqlite3").read_bytes()
                        ).digest()
                        == ledger_fingerprint
                    )
                else:
                    assert not (personal / ".personal-harness/ledger.sqlite3").exists()
            finally:
                stop_server(server)
        print("OK: installed UI login, HTML, assets and user profiles outside the checkout")
        if restore:
            assert all(
                hashlib.sha256(p.read_bytes()).digest() == value
                for p, value in fingerprints.items()
            )
            print("OK: backup boots with new login, scenes, chat and authenticated attachment")
            print("OK: original files and restored ledger remain unchanged; no model requests")
        else:
            print("OK: Chainlit state stays beside personal config; no model request or ledger")
    print(f"Imported installed package: {module}")


if __name__ == "__main__":
    main()
    main(restore=True)
