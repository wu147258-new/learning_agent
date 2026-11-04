# Agent 算法与系统：从数学原理到工程实践

面向从零学习计算机、Python、数学、大模型与Agent算法的中文教材项目。包含可编辑章节、公式、代码实验与习题，支持从源码生成PDF。

## 直接阅读

- [最新正式PDF（main分支）](book/Agent_Algorithm_Textbook_2026.pdf)
- [独立入门章节](chapters/beginner/)
- [现有算法与系统章节](chapters/)
- [长篇教材完整知识结构与进度](BOOK_PLAN.md)
- [后续开发计划](ROADMAP.md)

新增的零基础章节覆盖终端、Python、算法与数据结构、线性代数、概率统计、Git、HTTP/SQL与受控工具执行器。

## 章节结构与目录

- 一级标题（#）：章
- 二级标题（##）：节
- 三级标题（###）：小节
- **PDF目录只展示章、节；小节只在正文显示和编号。**

章节源文件保存在 chapters/，编译时依序组合为 book/source.md。合订文件是构建产物，修改章节应编辑 chapters/ 中的源文件。

## 运行实验

Python标准库实验：

~~~bash
python3 examples/beginner_cli.py self-test
python3 examples/foundation_algorithms.py
python3 examples/foundation_math.py
python3 examples/foundation_http_sql.py
python3 examples/beginner_tool_agent.py self-test
python3 examples/toy_agent.py
python3 examples/markov_control.py
python3 examples/evaluate_agents.py
~~~

PyTorch实验需要额外安装兼容的PyTorch版本：

~~~bash
python3 examples/tiny_transformer.py
~~~

## 构建教材PDF

需要 Pandoc、XeLaTeX、中文字体及常用LaTeX宏包：

~~~bash
bash scripts/build_pdf.sh
python3 scripts/check_heading_hierarchy.py --pdf dist/Agent_Algorithm_Textbook_2026.pdf
~~~

生成路径：dist/Agent_Algorithm_Textbook_2026.pdf。

GitHub Actions会运行离线实验并验证PDF编译，main分支通过后发布新的PDF。若PR尚未合并，请到相应PR的Actions产物下载预览，而非将主分支PDF视作已经更新。

## 版本与规模

目前仍是持续编写的初稿，不应将已有几十或上百页视作已完成千页教材。最终目标为**至少1000页有效教学内容**，计划由易到难扩充、整合和验证，实际完成情况以编译PDF页数与验收记录为准。

[验收规范](QUALITY.md) · [版本记录](CHANGELOG.md)

本仓库尚未指定开放许可。仓库公开可访问不等同于内容可以自由再许可、复制或重新出版。
