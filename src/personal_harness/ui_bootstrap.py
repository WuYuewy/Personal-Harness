"""Prepare the optional local app without importing the UI into the SDK."""

from __future__ import annotations

import json
import os
import re
import shutil
from importlib.resources import files
from importlib.util import find_spec
from pathlib import Path

from personal_harness.ui_state import UIState


def _prepare_frontend(directory: Path, package_root: Path) -> None:
    """Use the installed UI assets with a local-only HTML entry point.

    Hard links avoid duplicating the installed build when supported. Assets are
    immutable; replacements unlink first so an upgrade never edits installed files.
    """
    source = package_root / "frontend/dist"
    if not (source / "index.html").is_file():
        raise ValueError("Installed Chainlit frontend is missing; reinstall the ui extra.")
    target = directory / ".chainlit/frontend"
    for item in source.rglob("*"):
        if not item.is_file() or item.name == "index.html":
            continue
        destination = target / item.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Chainlit 2.12's cmdk MultiSelect can filter while its suggestion list is
        # unmounted (e.g. selecting its last match). Array.from(undefined) then
        # crashes the app. Guard only this compiled optional-ref expression in
        # our local copy; never modify or hard-link a patched dependency asset.
        if item.suffix == ".js":
            original = item.read_text("utf-8")
            patched = re.sub(
                r"Array\.from\((\((\w+)=(\w+)\.current\)==null\?void 0:"
                r"\2\.querySelectorAll\((\w+)\))\)",
                r"Array.from((\1)??[])",
                original,
            )
            if patched != original:
                content_bytes = patched.encode("utf-8")
                if destination.exists() and destination.read_bytes() == content_bytes:
                    continue
                if destination.exists():
                    destination.unlink()
                destination.write_bytes(content_bytes)
                continue
        if destination.exists():
            if destination.stat().st_size == item.stat().st_size and (
                destination.stat().st_mtime_ns == item.stat().st_mtime_ns
            ):
                continue
            destination.unlink()
        try:
            os.link(item, destination)
        except OSError:
            shutil.copy2(item, destination)
    content = (source / "index.html").read_text("utf-8")
    content = re.sub(r"<!-- FONT START -->.*?<!-- FONT END -->", "", content, flags=re.S)
    content = re.sub(r'<link\b[^>]*href="https://[^"]+"[^>]*>', "", content, flags=re.S)
    content = content.replace(
        "<!-- CSS INJECTION PLACEHOLDER -->",
        '<link rel="stylesheet" href="/public/vendor/katex/katex.min.css">'
        "<!-- CSS INJECTION PLACEHOLDER -->",
    )
    target.mkdir(parents=True, exist_ok=True)
    (target / "index.html").write_text(content, "utf-8")


def prepare_ui(directory: Path, host: str) -> dict[str, str]:
    state = UIState(directory / ".personal-harness/ui.sqlite3")
    if host not in {"127.0.0.1", "localhost", "::1", "127.0.0.2"} and not state.has_password():
        raise ValueError(
            "Set your local login password on 127.0.0.1 before allowing remote access."
        )
    template = files("personal_harness").joinpath("ui/config.toml").read_text("utf-8")
    config_path = directory / ".chainlit/config.toml"
    config_path.parent.mkdir(exist_ok=True)
    if config_path.exists():
        # Preserve unrelated Chainlit settings while updating supported app features.
        import tomllib

        desired = tomllib.loads(template)
        template_lines = template.splitlines()
        inline_tables: set[tuple[str, str]] = set()
        section_name = ""
        for line in template_lines:
            if line.startswith("["):
                section_name = line.removeprefix("[").removesuffix("]")
            elif " = {" in line:
                inline_tables.add((section_name, line.split(" = ", 1)[0]))
        sections: dict[str, dict[str, object]] = {}

        def collect(data: dict[str, object], prefix: str = "") -> None:
            for name, value in data.items():
                if isinstance(value, dict) and (prefix, name) not in inline_tables:
                    section = f"{prefix}.{name}" if prefix else name
                    collect(value, section)
                else:
                    sections.setdefault(prefix, {})[name] = value

        collect(desired)
        text = config_path.read_text("utf-8")
        for section, values in sections.items():
            for key in values:
                line = next(line for line in template_lines if line.startswith(key + " ="))
                pattern = rf"(?ms)(^\[{re.escape(section)}\]\s*\n)(.*?)(?=^\[|\Z)"
                match = re.search(pattern, text)
                if not match:
                    text += f"\n[{section}]\n{line}\n"
                else:
                    body = match[2]
                    setting = rf"(?m)^\s*{re.escape(key)}\s*=.*$"
                    body = (
                        re.sub(setting, lambda _, replacement=line: replacement, body)
                        if re.search(setting, body)
                        else (body.rstrip() + "\n" + line + "\n\n")
                    )
                    text = text[: match.start(2)] + body + text[match.end(2) :]
    else:
        text = template
    config_path.write_text(text, "utf-8")
    spec = find_spec("chainlit")
    if spec and spec.origin:
        _prepare_frontend(directory, Path(spec.origin).parent)
        vendor = files("personal_harness").joinpath("ui/vendor/katex")
        katex = directory / "public/vendor/katex"
        katex.mkdir(parents=True, exist_ok=True)
        for item in vendor.iterdir():
            if item.is_dir():
                folder = katex / item.name
                folder.mkdir(exist_ok=True)
                for asset in item.iterdir():
                    (folder / asset.name).write_bytes(asset.read_bytes())
            else:
                (katex / item.name).write_bytes(item.read_bytes())
        for language in ("zh-CN", "en-US"):
            translation_target = config_path.parent / "translations" / f"{language}.json"
            original = Path(spec.origin).parent / "translations" / f"{language}.json"
            translation = json.loads(
                (translation_target if translation_target.exists() else original).read_text("utf-8")
            )
            login = translation["auth"]["login"]
            login["title"] = (
                "Personal Harness · 首次登录设置密码（至少 8 位）"
                if language == "zh-CN"
                else "Personal Harness · Set a password on first login (8+ characters)"
            )
            login["form"]["email"]["label"] = "用户名" if language == "zh-CN" else "Username"
            login["form"]["email"]["placeholder"] = "local"
            translation["navigation"]["newChat"]["dialog"]["description"] = (
                "当前会话可从侧栏重新打开。确定开始新会话吗？"
                if language == "zh-CN"
                else "You can reopen this chat from the sidebar. Start a new chat?"
            )
            translation_target.parent.mkdir(exist_ok=True)
            translation_target.write_text(
                json.dumps(translation, ensure_ascii=False, indent=2), "utf-8"
            )
    for source, target in (
        ("ModelConnections.jsx", "public/elements/ModelConnections.jsx"),
        ("UserInsights.jsx", "public/elements/UserInsights.jsx"),
        ("WorkspaceWelcome.jsx", "public/elements/WorkspaceWelcome.jsx"),
        ("PersonalScenes.jsx", "public/elements/PersonalScenes.jsx"),
        ("LocalResults.jsx", "public/elements/LocalResults.jsx"),
        ("ReportRecipes.jsx", "public/elements/ReportRecipes.jsx"),
        ("RunReceipt.jsx", "public/elements/RunReceipt.jsx"),
        ("mark.png", "public/favicon.png"),
        ("brand.svg", "public/brand.svg"),
        ("brand.svg", "public/logo_light.svg"),
        ("brand.svg", "public/logo_dark.svg"),
        ("login-panel.svg", "public/login-panel.svg"),
        ("vendor/chart.umd.min.js", "public/vendor/chart.umd.min.js"),
        ("vendor/Chart.js.LICENSE.md", "public/vendor/Chart.js.LICENSE.md"),
        ("vendor/kurkle-color.LICENSE.md", "public/vendor/kurkle-color.LICENSE.md"),
        ("harness.css", "public/harness.css"),
        ("harness.js", "public/harness.js"),
        ("welcome.md", "chainlit.md"),
    ):
        path = directory / target
        path.parent.mkdir(parents=True, exist_ok=True)
        if source == "welcome.md" and path.exists():
            continue
        path.write_bytes(files("personal_harness").joinpath("ui", source).read_bytes())
    return {"PERSONAL_HARNESS_UI_PERSIST": "1", "CHAINLIT_AUTH_SECRET": state.auth_secret()}
