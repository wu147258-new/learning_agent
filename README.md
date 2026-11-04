# Agent 算法与系统：从数学原理到工程实践

面向 Agent 算法与系统工程方向的中文教材，包含数学推导、代码实验、系统设计与评测方法。

## 教材文件

- **[阅读/下载最新 PDF](book/Agent_Algorithm_Textbook_2026.pdf)**（首次构建完成后自动出现）
- [自动合并的 Markdown + LaTeX 章节快照](book/source.md)
- [分章编辑文件](chapters/)
- [PDF 排版配置](book/header.tex)
- [源码编译脚本](scripts/build_pdf.sh)
- [可离线运行的 Agent 实验](examples/toy_agent.py)

## 内容

数学与模型基础（概率、数学期望、马尔可夫链、Transformer、PyTorch）；Agent核心（工具调用、规划、RAG、记忆）；训练和系统（SFT、DPO、强化学习、Harness、Multi-Agent、安全、评测）；实战及面试（代码审计、代码修复、推理部署、LoRA、岗位要求），以初稿19章为基础，逐步增加严谨的数学、训练与评测专题；实际章节清单以 chapters/ 为准。

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

## 质量验收

章节文件是唯一的可编辑正文源，book/source.md 由构建脚本自动合并。PDF 使用 Pandoc 与 XeLaTeX 生成；通过如下命令进行结构、实验与PDF文本检查：

~~~bash
python3 scripts/check_textbook_structure.py
bash scripts/build_pdf.sh
python3 scripts/check_textbook_structure.py --pdf dist/Agent_Algorithm_Textbook_2026.pdf
~~~

新增内容通过PR检查后合并至main，且版本号的更新不等于质量验收完成。

## 持续迭代

- [版本计划](ROADMAP.md)
- [质量门槛](QUALITY.md)
- [更新日志](CHANGELOG.md)

**版本说明：** 初始排版快照的PDF为67页，Markdown重新构建的版式可能不同。仓库里的PDF以CI实际构建结果为准。本项目不是厂商官方教材，内容需要随论文、岗位和工程实践更新。

**许可：** 尚未选择开源许可证。公开仓库不表示作者授权随意再分发。
