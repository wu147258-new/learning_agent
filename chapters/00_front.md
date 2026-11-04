\frontmatter
\begin{titlepage}
\vspace*{25mm}
{\color{accent}\rule{\linewidth}{2.3pt}}\\[16mm]
{\sffamily\fontsize{36}{45}\selectfont\bfseries Agent 算法与系统}\\[5mm]
{\sffamily\fontsize{24}{34}\selectfont 从数学原理到工程实践}\\[8mm]
{\sffamily\Large 面向大厂 Agent 算法工程师的系统教材}\\[22mm]
{\normalsize\color{deepblue}问题驱动 · 数学推导 · 可执行代码 · 系统实验 · 面试与研究}\\[25mm]
\begin{tcolorbox}[colback=softgray,colframe=softgray,boxrule=0pt]
\small\noindent\textbf{写作原则}：先讲为什么需要，再引入解决问题的数学工具；区分可证明的数学结论、实验观察与工程启发式方法；不用“策略”“价值”“记忆”等术语替代定义。每章设置练习，每个主要工程模块提供评价方法。
\end{tcolorbox}
\vfill
{\sffamily\large 2026 年 11 月 · 第一版（完整学习版）}\\[2mm]
{\small 参考行业岗位要求与公开论文编写；非企业官方培训资料。}\\[12mm]
{\color{accent}\rule{\linewidth}{2.3pt}}
\end{titlepage}
\cleardoublepage
\chapter*{前言：为什么要这样学习 Agent？}
\addcontentsline{toc}{chapter}{前言：为什么要这样学习 Agent？}

如果从一个框架的 `create_agent()` 函数开始学起，很快就能做出能够调用工具的演示。但只要任务变长，模型可能反复调用同一工具、忽略失败日志、将检索内容当作指令，或者输出一份看似完整却不能通过测试的结果。此时，继续增加提示词往往不能解决根本问题。

学习应当从真实的失败出发：**为何任务失败？哪些不确定性来自模型，哪些来自工具，哪些来自数据与评测？如何通过实验判定修改确实有效？** 这正是本书的起点。

本书借鉴《深入理解计算机系统》（CSAPP）的教学思路，而不复刻其文字或版式：采用问题驱动的章首、从底层机制到上层系统的组织方式、可复现的代码与实验，以及穿插的练习和章末习题。特别强调先定义随机变量、概率分布和数学期望，再介绍强化学习领域的行业术语。

**读者基础。** 默认已掌握 Python 基本语法、大学线性代数、微积分和概率论基础。不要求事先熟悉 Agent 框架。基础薄弱者可以先读第 2、3 章与附录 A，再回到模型部分。

**学习目标。** 读完并完成实验后，你应能够：（1）从数学上描述一个有状态、可调用工具的系统；（2）阅读 Transformer、SFT、偏好优化与强化学习论文中的主要公式；（3）设计可恢复、可审计的 Agent 执行器；（4）用基准、消融实验和统计方法评价改进；（5）构建具有安全边界的真实项目。

**关于“算法岗”。** 企业的“Agent 算法工程师”可能侧重模型训练，也可能偏向工具规划、检索、评估或运行时系统。全书以“算法 + 系统 + 实验”三者交叉为主线；单纯调用框架并不等同于算法研发。招聘要求的依据见附录 D。

**代码边界。** 本书优先使用 Python 标准库，模型实验使用 PyTorch；案例可以本地运行，不依赖 Docker。示例旨在讲清机制，不代表已经具备生产安全性。生产系统必须增加身份鉴权、配额、审计、权限边界与故障恢复。

\tableofcontents
\cleardoublepage
\mainmatter
