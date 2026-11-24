"""Check UI workflows through a real browser, using only synthetic local files.

Install Playwright and its Chromium browser before running. No API key is needed.
"""

import argparse
import os
import socket
import sqlite3
import subprocess
import sys
import time
import urllib.request
from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory

from playwright.sync_api import expect, sync_playwright

from personal_harness.ui_bootstrap import prepare_ui

parser = argparse.ArgumentParser(
    description="Browser smoke for first login and zero-request shortcuts."
)
parser.add_argument("--browser", help="Optional Chromium/Edge executable path.")
args = parser.parse_args()
private = TemporaryDirectory(prefix="harness-browser-")
root = Path(private.name)
config = root / "personal.toml"
template = files("personal_harness").joinpath("templates/personal.toml").read_text("utf-8")
config.write_text(
    template.replace('allowed = ["context.describe"]', 'allowed = ["workspace.read"]'), "utf-8"
)
source = root / "expenses.csv"
source.write_text("id,amount\nINV-1,10.00\nINV-2,20.00\n", "utf-8")
env = {
    **os.environ,
    **prepare_ui(root, "127.0.0.1"),
    "CHAINLIT_APP_ROOT": str(root),
    "PERSONAL_HARNESS_CONFIG": str(config),
    "PYTHONIOENCODING": "utf-8",
}
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
url = f"http://127.0.0.1:{port}"


def thread_count():
    with sqlite3.connect(root / ".personal-harness/ui.sqlite3") as db:
        return db.execute("SELECT count(*) FROM threads").fetchone()[0]


with (root / "server.log").open("w", encoding="utf-8") as log:
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "chainlit",
            "run",
            str(files("personal_harness").joinpath("chainlit_app.py")),
            "--headless",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        env=env,
        cwd=root,
        stdout=log,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    try:
        for _ in range(180):
            if server.poll() is not None:
                raise RuntimeError(f"UI stopped; see {root / 'server.log'}")
            try:
                urllib.request.urlopen(url, timeout=0.5).close()
                break
            except OSError:
                time.sleep(0.15)
        with sync_playwright() as p:
            browser = p.chromium.launch(
                executable_path=args.browser,
                args=["--no-proxy-server"],
            )
            page = browser.new_page(viewport={"width": 1440, "height": 1000}, color_scheme="dark")
            page.set_default_timeout(10000)
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            external_requests = []
            page.on(
                "request",
                lambda request: (
                    external_requests.append(request.url)
                    if not request.url.startswith(url)
                    else None
                ),
            )
            page.goto(url, wait_until="domcontentloaded")
            illustration = page.locator('img[src="/public/login-panel.svg"]')
            expect(illustration).to_be_visible()
            illustration.evaluate("img => img.decode()")
            assert (
                page.request.get(url + "/logo?theme=light").body()
                == files("personal_harness").joinpath("ui/brand.svg").read_bytes()
            )
            page.set_viewport_size({"width": 390, "height": 844})
            expect(illustration).not_to_be_visible()
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            page.set_viewport_size({"width": 1440, "height": 1000})
            page.locator("input").first.fill("local")
            page.locator("input[type=password]").fill("local-ui-test-only")
            page.locator("button[type=submit]").click()
            welcome = page.get_by_role("region", name="新聊天", exact=True)
            expect(welcome).to_be_visible(timeout=15000)
            assert thread_count() == 0
            assert "light" in page.locator("html").get_attribute("class").split()
            page.locator("#chat-settings-header-button").click()
            page.get_by_role("tab", name="工具", exact=True).click()
            permission = page.get_by_label("本聊天允许的工具（修改即授权）", exact=True)
            for tool in ("workspace.read_many", "workspace.write_preview", "workspace.write"):
                permission.click()
                permission.fill(tool)
                option = page.get_by_role("option", name=tool, exact=True)
                expect(option).to_be_visible()
                option.click()
                expect(option).to_have_count(0)
                permission.press("Escape")
            page.get_by_role("button", name="确认", exact=True).click()
            page.wait_for_timeout(800)
            page.locator("#chat-settings-header-button").click()
            page.get_by_role("tab", name="工具", exact=True).click()
            selected = page.get_by_label("Tools for next task", exact=True)
            for tool in ("workspace.read_many", "workspace.write_preview", "workspace.write"):
                selected.click()
                selected.fill(tool)
                option = page.get_by_role("option", name=tool, exact=True)
                expect(option).to_be_visible()
                option.click()
                expect(option).to_have_count(0)
                selected.press("Escape")
            page.get_by_role("button", name="确认", exact=True).click()
            page.wait_for_timeout(800)
            page.get_by_role("button", name="打开工具箱", exact=True).click()
            panel = page.get_by_role("region", name="工具箱", exact=True)
            panel.get_by_role("button", name="CSV 查找").click()
            panel.get_by_label("工作区 CSV 路径", exact=True).fill(source.name)
            panel.get_by_label("列名", exact=True).fill("id")
            panel.get_by_label("精确值", exact=True).fill("INV-1")
            panel.get_by_label("快捷查询名称", exact=True).fill("本周报销")
            panel.get_by_role("button", name="保存快捷查询", exact=True).click()
            expect(panel.get_by_text("快捷查询已保存。", exact=True)).to_be_visible()
            before = thread_count()
            assert before == 1
            page.locator("#new-chat-button").click()
            page.get_by_role("dialog").get_by_role("button", name="确认", exact=True).click()
            pinned = welcome.get_by_role("button", name="本周报销", exact=True)
            expect(pinned).to_be_visible()
            assert thread_count() == before
            page.screenshot(path=str(root / "desktop-shortcut.png"))
            pinned.click()
            expect(page.get_by_text("10.00", exact=True)).to_be_visible()
            source.write_text("id,amount\nINV-1,25.00\nINV-2,40.00\n", "utf-8")
            page.set_viewport_size({"width": 390, "height": 844})
            chat = page.locator("#chat-input")
            chat.fill("/tools")
            page.locator("#chat-submit").click()
            panel = page.get_by_role("region", name="工具箱", exact=True).last
            panel.get_by_role("button", name="本周报销", exact=True).click()
            expect(page.get_by_text("25.00", exact=True)).to_be_visible()
            panel.get_by_role("button", name="编辑 本周报销", exact=True).click()
            panel.get_by_label("快捷查询名称", exact=True).fill("复核另一张发票")
            panel.get_by_label("精确值", exact=True).fill("INV-2")
            panel.get_by_role("button", name="更新快捷查询", exact=True).click()
            panel.get_by_role("button", name="复核另一张发票", exact=True).click()
            expect(page.get_by_text("40.00", exact=True)).to_be_visible()
            panel.get_by_role("button", name="删除 复核另一张发票", exact=True).click()
            expect(panel.get_by_text("快捷查询已删除。", exact=True)).to_be_visible()
            chat.fill("/usage")
            page.locator("#chat-submit").click()
            expect(page.get_by_text("CSV 查找 3 次", exact=False)).to_be_visible()
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            page.screenshot(path=str(root / "mobile-shortcut-usage.png"))
            assert not errors, errors
            assert not external_requests, external_requests
            assert not list(root.rglob("ledger.sqlite3"))
            browser.close()
    finally:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(server.pid), "/T", "/F"], capture_output=True)
        else:
            server.terminate()
        server.wait(timeout=15)
private.cleanup()
print(
    "First login, shortcut save, new chat, current-data reread, "
    "edit/delete, mobile and zero-request usage passed."
)
