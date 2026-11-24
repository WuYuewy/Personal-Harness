# CLI 与 SDK

## 通用在线配置

UI 和 CLI 共用执行引擎，但 UI 的模型连接不会偷偷回写 TOML。CLI 用户可生成通用模板，然后填写自己的端点、模型、上下文和价格：

```sh
uv run personal-harness init --provider openai-compatible --config my-agent.toml
uv run personal-harness config validate -c my-agent.toml
uv run personal-harness doctor -c my-agent.toml
uv run personal-harness provider prices -c my-agent.toml
uv run personal-harness budget -c my-agent.toml
uv run personal-harness run "总结 notes.md" -c my-agent.toml
```

模板里的 `api.example.com` 和 `your-model-id` 是无效示例，必须替换。Key 放在 `OPENAI_API_KEY` 或配置文件旁被忽略的 `.env`，不写进 TOML。实际联网还需安装 `api` 或 `ui` extra。

## 执行并审批

```python
import asyncio
from personal_harness import Agent


async def main():
    async with Agent.from_config("my-agent.toml") as agent:
        result = await agent.run("读取 notes.md，生成 brief.md")
        while result.status == "pending_approval":
            decisions = {}
            for approval in result.approvals:
                print(approval.preview["diff"])
                decisions[approval.call_id] = input("批准这次写入? [y/N] ") == "y"
            result = await agent.resume(result.run_id, decisions)
        print(result.output, result.total_cost)


asyncio.run(main())
```

审批绑定本次预览和文件身份。SDK 调用者负责把明确决定交给 resume；不要自动同意所有修改。

## 扩展自己的工具

```python
from pydantic_ai import RunContext
from personal_harness.context import RunDependencies


async def user_greeting(ctx: RunContext[RunDependencies], name: str) -> str:
    """Generate a local greeting without network calls."""
    return f"Hello, {name}"
```

把 `user.greeting` 加入工具授权，构造 `Agent(resolved, tools=[user_greeting])`。使用 PydanticAI 原生工具类型，不需要继承自建 Tool 基类。涉及外部付费接口的工具应先通过 `ctx.deps.external_costs.reserve` 预留，再 settle 实际费用；未经计量的服务不应宣称包含在账本内。

## 连续会话与错误

UI 自动管理上下文；SDK 调用者可将 `result.messages` 追加到自己的历史，再传 `message_history`。用上下文管理器关闭连接。`RunError` 携带原因、运行 ID 和可用费用信息，超时不应导致应用把已经发送的请求记为免费。
