# Agent 算法与系统：从数学原理到工程实践

面向 Agent 算法工程师的中文教材。**参考 CSAPP 的问题驱动写法**：先讲清楚为什么需要某项技术，再严格定义数学对象、推导结论、编写实验代码并验证结果；不复刻 CSAPP 原文或插图。

## 教材文件

- **[阅读/下载最新 PDF](book/Agent_Algorithm_Textbook_2026.pdf)**（首次构建完成后自动出现）
- [完整 Markdown + LaTeX 原始章节](book/source.md)
- [分章编辑文件](chapters/)
- [PDF 排版配置](book/header.tex)
- [源码编译脚本](scripts/build_pdf.sh)
- [可离线运行的 Agent 实验](examples/toy_agent.py)

## 内容

数学与模型基础（概率、数学期望、马尔可夫链、Transformer、PyTorch）；Agent核心（工具调用、规划、RAG、记忆）；训练和系统（SFT、DPO、强化学习、Harness、Multi-Agent、安全、评测）；实战及面试（代码审计、代码修复、推理部署、LoRA、岗位要求），共19章与附录。

## 数学优先原则

将“策略”解释为**给定状态的行动条件概率分布**，将“价值函数”解释为**累计随机收益的条件数学期望**。每个公式要说明随机变量、假设和适用范围。区分数学定理、经验观察和工程启发。

## 使用

运行离线实验：

```bash
python3 examples/toy_agent.py
```

使用 XeLaTeX、Pandoc 与 Noto CJK 构建 PDF：

```bash
bash scripts/build_pdf.sh
```

Push 到 main 后 GitHub Actions 也会自动运行实验、构建PDF，并尝试把最新PDF提交回仓库。若自动提交受到GitHub仓库权限限制，可从Actions的构建产物中下载PDF。

## 持续迭代

- [版本计划](ROADMAP.md)
- [质量门槛](QUALITY.md)
- [更新日志](CHANGELOG.md)

**版本说明：** 初始排版快照的PDF为67页，Markdown重新构建的版式可能不同。仓库里的PDF以CI实际构建结果为准。本项目不是厂商官方教材，内容需要随论文、岗位和工程实践更新。

**许可：** 尚未选择开源许可证。公开仓库不表示作者授权随意再分发。
