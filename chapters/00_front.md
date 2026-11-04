\frontmatter
\begin{titlepage}
\vspace*{25mm}
{\color{accent}\rule{\linewidth}{2.3pt}}\\[16mm]
{\sffamily\fontsize{36}{45}\selectfont\bfseries Agent 算法与系统}\\[5mm]
{\sffamily\fontsize{24}{34}\selectfont 从数学原理到工程实践}\\[8mm]
{\sffamily\Large 面向大厂 Agent 算法工程师的系统教材}\\[22mm]


\vfill
{\sffamily\large 2026 年 11 月 · 修订稿}\\[12mm]
{\color{accent}\rule{\linewidth}{2.3pt}}
\end{titlepage}
\cleardoublepage
\chapter*{导论：Agent 的任务执行问题}
\addcontentsline{toc}{chapter}{导论：Agent 的任务执行问题}

一个代码修复系统可能能够准确解释登录失败的原因，却无法正确修改仓库；也可能在修复主问题后引入新的缺陷。文本回答的正确性与任务执行的成功率并不等价。对外部环境产生影响的系统，必须同时处理模型输出、工具调用、执行状态、权限限制和结果验证。

考虑一次自动化缺陷修复：系统读取仓库文件、执行测试、修改代码并重新验证。若测试失败，需要区分是模型生成的修改有误、工具执行失败、依赖环境变化，还是验证条件设置不正确。要量化这些因素，应明确任务状态、可选行动、执行后的条件分布、时间与资源成本，以及最终成功的判定规则。

Agent 算法相关工作包括模型训练与推理、工具使用、任务规划、信息检索、状态维护、统计评测和运行时系统。各方向涉及的核心数学对象和工程限制不同。例如，工具执行结果可视为随机变量，长期任务成本可以用累计随机成本的数学期望描述，检索效果则需要在固定任务集合及外部标注上测量。

运行实验需要 Python 环境；神经网络训练示例需要 PyTorch。高风险工具操作必须有独立的授权与隔离机制，教学示例不能直接作为生产级执行器使用。

\tableofcontents
\cleardoublepage
\mainmatter
