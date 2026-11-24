# 首版实施与维护计划

0.1.0 已收敛为可本地安装、独立配置在线模型的个人 Agent。产品目标与范围以 docs/PRODUCT_SCOPE.md 为准，验收状态见 docs/RELEASE_REVIEW.md。

后续按以下顺序维护：

1. 修复可复现的首次使用、数据恢复和模型兼容性问题，保留回归测试。
2. 在用户提供的真实任务上比较同等质量下的总费用，记录失败/重试与样本范围。
3. 依据重复任务扩充本地方案能力，优先减少模型调用，避免无目的地接入服务。
4. 有实际需求再设计实时数据连接器、系统凭据存储和部署方案，不在首版承诺这些能力。

本版经维护者授权公开发布；版本与自动检查见 [GitHub Releases](https://github.com/WuYuewy/Personal-Harness/releases) 和 [Actions](https://github.com/WuYuewy/Personal-Harness/actions)，使用文档见 [GitHub Pages](https://wuyuewy.github.io/Personal-Harness/)。
