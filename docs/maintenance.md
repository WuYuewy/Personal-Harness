# 数据、维护与发布

## 数据在哪里

个人 TOML 旁的 `.personal-harness/` 保存 UI 数据库、费用账本和附件；Chainlit 还会生成 `.chainlit/`、`public/`、`.files/` 等运行文件。它们不是开源源码。模型 Key 在本机数据库独立存储，不进入模型提示或连接列表导出；这不等于操作系统级密钥保险库，应保护本机账号和磁盘。

## 备份与恢复

停止 UI 后执行：

```sh
uv run personal-harness data backup my-backup --config personal-harness.toml
uv run personal-harness ui --config my-backup/personal-harness.toml
```

备份目录自身可作为恢复后的运行目录，也可先复制到新的私人目录。具体参数以 `data backup --help` 为准。备份包含历史与附件等私人数据，不得公开。备份会剔除 API 凭据、密码和 JWT；恢复后重新设置登录密码与模型 Key。费用账本随数据恢复，不自动清零。

## 从源码检查

```sh
uv sync --locked --extra ui --dev
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run --extra ui pytest
uv run python scripts/build_docs.py --check
uv build
uv run python scripts/smoke_wheel.py
uv run python scripts/smoke_wheel.py --ui
```

自动测试使用模拟传输，不调用真实模型。真实首次用户验收记录另见 [0.1.0 验收记录](https://github.com/WuYuewy/Personal-Harness/blob/main/docs/RELEASE_REVIEW.md)，不在 CI 存放任何真实 Key。

## 在线文档与发布

[在线使用文档](https://wuyuewy.github.io/Personal-Harness/) 通过 GitHub Pages 托管，不需要读者在本机启动文档服务。[项目仓库](https://github.com/WuYuewy/Personal-Harness) 提供源码、Issue 和贡献入口，[Releases](https://github.com/WuYuewy/Personal-Harness/releases) 提供版本包与 SHA256 校验文件。源码使用 Apache-2.0 许可证；尚未发布到 PyPI，请使用源码或 Release 中的 wheel。

文档源文件在 `docs/*.md`，截图和演示下载位于 `docs/site/`。修改后运行 `uv run python scripts/build_docs.py`，将 Markdown 与生成的静态站一同提交。`.github/workflows/pages.yml` 在 `main` 更新时校验并部署 `docs/site/`；仓库 Settings → Pages 的 Source 选择 **GitHub Actions**。站点使用相对资源与 hash 路由，支持项目站点的子路径。

维护时可用 `uv run python scripts/serve_docs.py` 预览尚未提交的修改；对外入口始终是上述在线文档。部署方式见 [GitHub Pages 官方说明](https://pages.github.com/)。

发布新版本前运行 `uv run python scripts/audit_release.py` 检查 Git 索引中的私人路径与疑似凭据；未初始化 Git 的干净副本使用 `--tree .`。检查 CI、LICENSE、README、构建包与 SHA256 后，再创建版本标签和 Release。不要提交个人配置、运行数据库、凭据或备份。

贡献请保持最小改动、类型校验、权限边界和可重复测试。供应商差异通过通用连接参数处理，避免引入特定部署默认配置。

## 可重复的浏览器冒烟

```sh
uv run --with playwright playwright install chromium
uv run --extra ui --with playwright python scripts/browser_smoke.py
```

也可用 `--browser` 指定本机 Chromium 或 Edge。脚本在临时目录首次登录，实际点击快捷查询、切换聊天、修改资料、编辑/删除快捷项，并验证手机布局和零模型调用。不会使用真实密钥或任何机构的部署配置。
