# 精简的 SDK 与工程结构

## 只有三个主要入口

配置定义「你需要什么」，Agent 执行「这次做什么」，工具定义「允许做什么」。复用 PydanticAI 的 Agent、工具协议、消息与 OpenAI 兼容适配；复用 Chainlit 的认证、侧栏、设置和附件，不再建立一套通用多 Agent 编排框架。

```text
TOML / UI 设置
       ↓ 校验、合并与权限选择
ResolvedConfig → Agent → PydanticAI → 用户配置的模型
                   ↓           ↕
              文件 / 本地工具  预算与 usage 账本
                   ↓
              变更预览 → 用户审批 → 文件
```

## 配置如何生效

TOML 先合并 base 与选定 profile，再应用显式 overrides；UI 在此基础上应用账号助手偏好、场景默认值和当前聊天设置。账号体验偏好对该账号保持一致，聊天工具不超过明确授权范围。

`personal-harness config show --explain` 给出有效配置和来源。密钥通过独立参数或本地环境读取，不进入配置导出、提示词或费用收据。

## 模块定位

| 源码 | 职责 |
| --- | --- |
| `config.py`, `context.py` | 类型化配置、来源、任务上下文 |
| `runtime.py` | 执行、恢复审批、资源生命周期 |
| `tools.py`, `workspace.py` | 工具与文件边界 |
| `connections.py`, `openai_adapter.py` | 通用连接与 usage 校验 |
| `metering.py`, `ledger.py`, `costs.py` | 请求预留、费用结算与核账 |
| `history.py`, `conversation.py` | 历史范围与文件正文精简 |
| `reports.py`, `csv_lookup.py`, `csv_summary.py` | 本地确定性任务 |
| `chainlit_app.py`, `ui_features.py`, `ui_state.py` | 可选 UI、状态和账号设置 |
| `user_insights.py`, `places.py` | 本地偏好分析与地点过滤 |

UI 依赖放在可选 extra；不装 UI 仍可用 CLI 和 SDK。首版是单人、本地优先的完整小工具，不是分布式服务平台。
