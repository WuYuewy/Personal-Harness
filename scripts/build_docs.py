"""Build the GitHub Pages documentation; committed HTML can be served without dependencies."""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]
PAGES = [
    ("getting-started", "快速开始", "开始使用"),
    ("task-demo", "综合任务演示", "开始使用"),
    ("chat", "聊天与附件", "开始使用"),
    ("workspace", "工作区与本地任务", "日常工作"),
    ("personalization", "你的个性化助手", "日常工作"),
    ("local-life", "手机与本地生活", "日常工作"),
    ("savings", "省钱机制与账本", "理解 Harness"),
    ("architecture", "精简架构与 SDK 效率", "理解 Harness"),
    ("sdk", "CLI 与 SDK", "开发与维护"),
    ("limits", "常见问题与边界", "开发与维护"),
    ("maintenance", "数据与发布", "开发与维护"),
]


def render() -> str:
    md = MarkdownIt("commonmark", {"html": False}).enable("table")
    articles, nav, index = [], [], []
    group = ""
    for slug, label, category in PAGES:
        source = (ROOT / "docs" / f"{slug}.md").read_text("utf-8")
        content = md.render(source)
        # Markdown uses paths relative to docs/ so screenshots also work on GitHub.
        content = content.replace('src="site/', 'src="').replace('href="site/', 'href="')
        for page, _, _ in PAGES:
            content = content.replace(f'href="{page}.md"', f'href="#{page}"')
        headings = []

        def anchor(match: re.Match[str], slug: str = slug, headings: list = headings) -> str:
            identity = f"{slug}-section-{len(headings) + 1}"
            heading = re.sub("<[^>]+>", "", match[1])
            headings.append((identity, heading))
            return f'<h2 id="{identity}">{match[1]}</h2>'

        content = re.sub(r"<h2>(.*?)</h2>", anchor, content)
        toc = "".join(
            f'<a href="#{identity}">{html.escape(heading)}</a>' for identity, heading in headings
        )
        articles.append(
            f'<article data-page="{slug}" aria-label="{label}" hidden>'
            f'<div class="eyebrow">{category} / {label}</div>{content}'
            "<footer>Personal Harness · 0.1.0 · Apache-2.0</footer></article>"
            f'<aside class="outline" data-outline="{slug}" hidden>'
            f"<span>本页内容</span>{toc}</aside>"
        )
        if category != group:
            nav.append(f'<div class="nav-group">{category}</div>')
            group = category
        nav.append(f'<a data-nav="{slug}" href="#{slug}">{label}<span>↗</span></a>')
        index.append({"slug": slug, "title": label, "text": re.sub(r"<[^>]+>", " ", content)})
    template = (ROOT / "docs/site/template.html").read_text("utf-8")
    return (
        template.replace("<!-- NAV -->", "".join(nav))
        .replace("<!-- ARTICLES -->", "".join(articles))
        .replace("/* SEARCH_DATA */[]", json.dumps(index, ensure_ascii=False).replace("</", "<\\/"))
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    target = ROOT / "docs/site/index.html"
    expected = render()
    example = ROOT / "examples/community-workshop"
    demo_assets = {path.name: path.read_bytes() for path in example.iterdir() if path.is_file()}
    demo_target = ROOT / "docs/site/demo"
    if args.check:
        if not target.exists() or target.read_text("utf-8") != expected:
            print("Docs are stale. Run python scripts/build_docs.py.")
            return 1
        if any(
            not (demo_target / name).is_file() or (demo_target / name).read_bytes() != data
            for name, data in demo_assets.items()
        ):
            print("Demo downloads are stale. Run python scripts/build_docs.py.")
            return 1
        print("Documentation is up to date.")
        return 0
    target.write_text(expected, "utf-8", newline="\n")
    demo_target.mkdir(exist_ok=True)
    for name, data in demo_assets.items():
        (demo_target / name).write_bytes(data)
    print("Built docs/site/index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
