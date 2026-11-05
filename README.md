# Agent 算法与系统：从数学原理到工程实践

面向从零学习计算机、Python、数学、大模型与Agent算法的中文教材项目。包含可编辑章节、公式、代码实验与习题，支持从源码生成PDF。

## 直接阅读

- [最新正式PDF（main分支）](book/Agent_Algorithm_Textbook_2026.pdf)
- [独立入门章节](chapters/beginner/)
- [现有算法与系统章节](chapters/)
- [长篇教材完整知识结构与进度](BOOK_PLAN.md)
- [后续开发计划](ROADMAP.md)

新增的零基础章节覆盖终端、Python、算法与数据结构、线性代数、概率统计、Git、HTTP/SQL、受控工具执行器、测试与调试、操作系统并发，以及网络协议基础。

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
python3 examples/foundation_testing.py
python3 examples/foundation_concurrency.py
python3 examples/foundation_network.py
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

## 深入Agent运行时实验

~~~bash
python3 examples/agent_runtime_recovery.py
python3 examples/multiagent_coordinator.py
python3 examples/agent_eval_security_lab.py
~~~

新增深度章节：[Agent运行时与故障恢复](chapters/28_agent_runtime_recovery.md)、[多Agent协作与一致性](chapters/29_multiagent_coordination.md)、[Agent评测与安全验收](chapters/30_agent_eval_security.md)。章节配套的可编辑矢量图位于 figures/，随构建编入PDF。

原有章节穿插的模型结构与Agent架构实验：

~~~bash
python examples/model_architecture_patterns.py
python examples/model_training_architectures.py
python examples/agent_architecture_patterns.py
~~~

对应正文为现有Transformer、SFT/DPO/RL、RAG、推理部署、Agent规划、运行时和多Agent章节；**没有新增独立的模型名录或架构单元**。模型案例以公开技术报告和模型卡为依据，代码模拟用于验证算子，不代表重现商用模型权重。


本书的Agent系统章节新增与2026年协议生态相关的实践内容，均穿插在原有章节，不新增独立单元：

- [RAG检索](chapters/23_rag_retrieval_deep_dive.md)：Token预算、0-1背包与多来源证据覆盖。
- [证据验证](chapters/24_rag_evidence_verification.md)：长期记忆、来源版本、过期与撤销。
- [Agent运行时](chapters/28_agent_runtime_recovery.md)：MCP工具与资源契约、JSON-RPC、权限与幂等。
- [多Agent协作](chapters/29_multiagent_coordination.md)：A2A能力发现、Task生命周期与跨Agent委派。
- [评测安全](chapters/30_agent_eval_security.md)：MCP/A2A信任边界与负向验收。

新增全部离线代码（Python标准库即可执行）：

~~~bash
python examples/agent_protocol_contracts.py
python examples/context_budget_lab.py
python examples/agent_memory_provenance_lab.py
~~~

协议案例采用官方公开规范；脚本仅模拟安全边界与状态机，并不声称实现了完整MCP/A2A服务端。

## 推理与端到端Agent实验

继续在原有推理基础设施、强化学习和Agent规划章节增加可运行实验：

~~~bash
python examples/inference_quantization_lab.py
python examples/speculative_decoding_lab.py
python examples/verified_code_repair_agent.py
python examples/multistep_rl_rollout_lab.py
~~~

最后一项需要PyTorch CPU。算子与执行流程均为离线教学实现，不是主流模型权重或vLLM内核的完整复现。
