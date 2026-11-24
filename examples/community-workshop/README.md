# 社区读书会综合任务（虚构示例）

公开输入为 planning.md、expenses.csv、updates.md 和 4 页文字 PDF venue-guide.pdf。
workshop-brief.md 是真实在线 UI 任务经审批写入的参考产物，不是模型固定答案。

在项目根目录运行：

```sh
uv run python examples/community-workshop/prepare_demo.py --dest ../my-personal-workshop
```

脚本仅复制四份输入，不复制参考产物，不覆盖已有目录。随后在 UI 设置工作区与工具授权。
完整提示词、预期金额、审批及后续个性化流程见 [在线文档「综合任务演示」](https://wuyuewy.github.io/Personal-Harness/#task-demo)（文档源码：`docs/task-demo.md`）。
全部人名和资料为虚构示例，不包含模型部署参数和凭据。
