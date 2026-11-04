# Agent 算法与系统：从数学原理到工程实践

面向大厂 Agent 算法工程师的中文自学教材（v0.1，2026-11）。

**教学原则**：从实际问题出发，先定义随机变量、条件概率分布、数学期望和优化目标，再引入 Agent/强化学习领域术语。章节包括数学基础、马尔可夫链、Transformer、模型训练、工具调用、规划、RAG、记忆、强化学习、Agent 评测、Harness、Multi-Agent、安全与综合项目。

- 教材章节：[`chapters/`](chapters/)
- PDF 构建源：[`book/source.md`](book/source.md)
- 离线实验：[`examples/toy_agent.py`](examples/toy_agent.py)
- 学习和修订计划：[`ROADMAP.md`](ROADMAP.md)

## 构建 PDF

使用 XeLaTeX、Pandoc 和 Noto CJK 字体：`bash scripts/build_pdf.sh`。GitHub Actions 会进行示例测试并构建 PDF。

**初版状态**：19 章 + 附录，仍需要进一步完善数学证明、实验、习题和引用；不应视作已经经过同行评审的正式教材。

本仓库暂未选择开源许可证，转载和贡献政策待确定。