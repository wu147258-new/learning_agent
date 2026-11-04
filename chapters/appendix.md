\appendix
# 数学速查与关键公式推导

## 随机变量与概率分布

一个离散随机变量 $X$ 的取值空间为 $\mathcal X$，概率质量函数满足 $p(x)\ge0$ 和 $\sum_{x\in\mathcal X}p(x)=1$。对于离散变量 $X,Y$，联合分布 $p(x,y)$ 可推得边缘分布 $p(x)=\sum_y p(x,y)$。当 $p(x)>0$ 时，条件概率为 $p(y\mid x)=p(x,y)/p(x)$。

两个事件独立的定义是 $P(A\cap B)=P(A)P(B)$，并非“看起来没有关系”。条件独立是另一种关系：$P(A,B\mid C)=P(A\mid C)P(B\mid C)$。判断多个Agent错误是否独立时，应该关注它们是否共享模型、训练数据、提示词与观察来源。

数学期望具有线性性：$\mathbb E[aX+bY]=a\mathbb E[X]+b\mathbb E[Y]$，不要求 $X$ 和 $Y$ 独立。方差相加通常需要协方差项：

$$\operatorname{Var}(X+Y)=\operatorname{Var}(X)+\operatorname{Var}(Y)+2\operatorname{Cov}(X,Y).$$

为什么强调这一点？因为把多次Agent执行结果当成独立样本，往往会低估估计误差。

## 条件期望的递推来源

设 $X_{t+1}$ 由当前状态 $X_t$ 和行动 $A_t$ 的条件分布决定，当前收益是 $R_t$。将总收益拆成当前项与未来项：

$$G_t=R_t+\gamma G_{t+1}.$$

在给定当前状态和行动的条件下取期望：

$$\mathbb E[G_t\mid X_t=s,A_t=a]=\mathbb E[R_t\mid s,a]+\gamma\mathbb E[G_{t+1}\mid s,a].$$

如果未来只依赖下一状态，利用全期望公式：

$$\mathbb E[G_{t+1}\mid s,a]=\sum_{s'}P(X_{t+1}=s'\mid s,a)\,\mathbb E[G_{t+1}\mid X_{t+1}=s'].$$

代回便得到贝尔曼型递推。因此它不是脱离概率论的“人工智能专有原理”，而是随机变量分解和条件期望的应用。若要在所有合法行动中寻找最小期望成本，再对 $a$ 取最小值。

## 有限时域和无限时域的区别

对有限时域 $T$，定义 $V_t(s)$ 为从时刻$t$开始到终止时刻的最小预期成本，可写为

$$V_t(s)=\min_a\left[c_t(s,a)+\sum_{s'}p_t(s'\mid s,a)V_{t+1}(s')\right],$$

给定终止条件 $V_T(s)=c_T(s)$，从后往前求解即可。对于折扣无限时域，时间齐次、$0\le\gamma<1$ 等条件下，可构造固定点递推。对于无界成本或复杂连续状态，解存在性与收敛性需要额外假设，不能直接照搬有限状态结论。

## 从交叉熵到最大似然

设正确类别为 $y_i$，模型估计 $p_\theta(y_i\mid x_i)$。假定样本条件独立，联合似然为 $\mathcal L(\theta)=\prod_i p_\theta(y_i\mid x_i)$。最大化联合似然等价于最大化对数似然，也等价于最小化负对数似然。这就是交叉熵作为训练目标的直接来源。

多类别softmax分类中，设 $z_j$ 为类别$j$的logit，

$$p_j=\frac{e^{z_j}}{\sum_k e^{z_k}}.$$

如果正确类别为 $y$，单样本损失 $L=-\log p_y$，则对各logit的导数为

$$\frac{\partial L}{\partial z_j}=p_j-\mathbf 1(j=y).$$

这个简洁结果非常重要：模型预测概率与正确标签之间的差，直接影响参数梯度。

## 为什么数值计算需要稳定实现？

计算softmax时，若logit非常大，直接计算 $e^{z_j}$ 可能溢出。因为给所有logit同时减去同一个常数不改变结果，可以计算

$$p_j=\frac{e^{z_j-m}}{\sum_k e^{z_k-m}},\qquad m=\max_k z_k.$$

同理，在实现对数概率时，使用稳定的log-sum-exp算子通常优于直接先求概率再取对数。深度学习框架的交叉熵接口往往已经处理了这些问题，因此要理解接口期望的是logits还是概率。

## 轨迹概率与得分函数梯度

设 $\tau$ 是环境交互轨迹，目标为 $J(\theta)=\sum_\tau p_\theta(\tau)G(\tau)$。对参数求导：

$$\begin{aligned}
\nabla J(\theta)&=\sum_\tau \nabla p_\theta(\tau)G(\tau)\\
&=\sum_\tau p_\theta(\tau)\nabla\log p_\theta(\tau)G(\tau)\\
&=\mathbb E_{\tau\sim p_\theta}\left[G(\tau)\nabla\log p_\theta(\tau)\right].
\end{aligned}$$

这里没有对离散的动作编号求导，而是利用了概率函数对参数的可微性。多步情形下，动作选择的对数概率求和产生各时刻梯度；后续具体收益如何归给每一步，构成估计方差与信用分配问题。

## 一个值得牢记的条件

对数学模型中的结论，要明确假设。例如：

- $\mu_{n+1}=\mu_nP$ 要求给定齐次转移矩阵。
- 稳态分布收敛结论需要状态空间、可达性与周期性条件。
- 用 $p^k$ 计算连续成功概率需要独立同分布等假设。
- 低秩分解减少可训练参数不意味着自动保证同等预测性能。
- 配对试验得到的结论只对其任务抽样机制和评测环境具有直接解释力。

工程师需要学会说出“公式成立的条件是什么”，而不是只会记公式。

# 实验手册：环境、任务与验收

## 不使用Docker的基本环境

本书的核心实验可在Windows、Linux或macOS上运行。建议Python 3.11或以上版本；不同依赖可能有各自版本要求。纯标准库实验无需第三方包，神经网络实验需要安装与你的硬件和驱动匹配的PyTorch。

```bash
python --version
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -U pip
python -m pip install pytest numpy
# Install PyTorch according to the official selector
```

PyTorch涉及CUDA或CPU平台差异，建议使用官方安装页给出的命令，不在教材中写死可能过时的CUDA轮子地址。运行实验之前先执行 `python -c "import torch; print(torch.__version__)"` 检查环境。

## 实验1：马尔可夫链的分布演化

目标：验证 $\mu_n=\mu_0P^n$ 与长期分布。本实验只用Python标准库。

```python
def mul(vec, mat):
    return [sum(vec[i] * mat[i][j]
                for i in range(len(vec)))
            for j in range(len(vec))]

P = [[.8, .2], [.3, .7]]
mu = [1., 0.]
for step in range(12):
    print(step, [round(x, 5) for x in mu])
    mu = mul(mu, P)
```

验收要求：说明为什么每一步和为1；计算第2步分布 $(0.7,0.3)$；解释长期分布趋近 $(0.6,0.4)$ 的原因；把 $P$ 改成周期链，观察是否仍然逐步收敛。

## 实验2：自己实现软最大化与交叉熵

目标：理解为什么模型的输出概率是归一化指数形式，并认识数值溢出风险。

```python
import math

def softmax(logits):
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    z = sum(exps)
    return [x / z for x in exps]

p = softmax([1., 2., 3.])
loss = -math.log(p[2])
print(p, loss, sum(p))
```

验收要求：改变正确类别，观察交叉熵变化；把所有logit都加10000，验证稳定softmax给出相同分布；手算 $\partial L/\partial z_j$ 并用数值微分检查。

## 实验3：PyTorch张量与自动微分

使用第5章二分类代码。逐步检查张量形状、参数量、梯度值和训练前后准确率。将学习率设置为0.001、0.1、1.0，比较训练曲线，分析过大学习率可能造成的问题。

实验报告不能只给最终准确率，应展示至少三种超参数下的训练过程，并说明哪些结论依赖当前小数据集。

## 实验4：工具接口的拒绝测试

使用第6章只读工具。尝试合法路径、越界路径、禁止后缀、超大文件、不存在文件。测试要求：所有非法输入返回预期异常，而不是读取或写入任何范围外文件；保留测试记录。不要在包含真实机密的生产项目目录中执行未经隔离的Agent试验。

参考单元测试：

```python
import pytest

# Import your tested implementation before running.

def test_escape_denied(read_project_file):
    with pytest.raises(PermissionError):
        read_project_file("../../secret.txt")
```

上例展示测试意图，实际编写时需将 `read_project_file` 通过pytest fixture或普通导入传入；不能直接把它当作独立可运行测试文件。

## 实验5：RAG检索基线

建立至少50条文档片段、20条查询与人工标注的相关文档集合。先做简单关键词检索，再接入BM25，最后引入向量检索和混合排序。每个版本都记录Recall@3、Recall@5、首条正确证据排名、索引时间和查询时间。

验收要求：版本隔离、权限过滤、检索器与生成器分开测量。至少分析五条召回失败样例；任何答案的事实陈述都应可追溯到来源材料。

## 实验6：状态与失败恢复

使用第14章玩具Agent，在第2步人为抛出异常，并把当前状态写入JSON文件。重新启动程序后，应该从检查点继续，而不是重新执行之前可能已产生副作用的动作。对只读工具可以采用可重复读取；对写入工具必须配合幂等标识或事务状态查询。

验收要求：三种测试场景——正常运行、执行前失败、执行后返回结果丢失；每种都检查执行日志和最终状态。对无法确定的写入结果必须进入人工或查询确认流程。

## 实验7：训练和评测的对照

设计100条工具调用指令，至少包含正常参数、缺失参数、权限不足、工具超时、无可用工具等类别。比较未微调模型与微调模型在工具参数准确率和端到端成功率方面的表现。硬件不足时可以先比较不同的提示词或受控规则基线，但必须注明不是参数训练实验。

验证方法包括：冻结测试集、固定工具协议、设置相同的调用预算、记录每个任务的配对结果，以及报告失败类别。

## 实验8：综合Agent项目

二选一：安全工具型代码Agent或证据驱动的代码审计Agent。要求至少包括状态机、工具白名单、异常处理、任务集、自动验证器、日志、成本统计和一项实证改进。不能仅提交一张聊天截图。

推荐验收表如下：

| 项目维度 | 通过条件 |
|:--|:--|
| 可靠性 | 对明确失败类型有可重复的处理路径 |
| 安全性 | 违规提议无法绕过工具权限检查 |
| 可测性 | 能在固定任务集上报告完整分母与成功率 |
| 可追溯性 | 关键结论具备证据与版本信息 |
| 可恢复性 | 中断后具有可检查的状态恢复行为 |
| 可维护性 | 模型、工具、状态、评价器模块分离 |

## 建议的实验报告格式

一份适合面试展示的报告，不需要华丽包装，但至少应包含：研究问题；系统边界；数据来源；实验假设；对照方案；实现细节；测试环境；指标定义；结果表；置信区间或不确定性说明；失败样例；消融实验；无法排除的偏差；结论与下一步。

最忌讳的做法是先写“效果显著提升”，再选择支持该结论的样例。实验的价值恰恰在于允许假设被事实否定。

# 习题选解：从计算到工程判断

## 第1、2章选解

**1.2。** 78/100与83/100分别只是样本成功率。若两个系统在同一批任务上测试，应观察哪些任务只被一个系统完成，并采用配对比较。只有五个任务的表面差异不能解释任务抽样误差、随机采样和是否存在过拟合。

**2.1。** A的整体成功概率是 $0.5(0.95)+0.5(0.35)=0.65$；B为 $0.5(0.75)+0.5(0.65)=0.70$。A在容易任务上更好，但困难任务存在严重短板。若真实业务中困难任务占比提高，A的劣势可能进一步扩大。

**2.2。** $D_{\mathrm{KL}}(p\|q)=0.8\ln(1.6)+0.2\ln(0.4)\approx0.1927$，数值单位为nat。反方向的KL值不同。

## 第3、4章选解

**3.1。**

$$P^2=\begin{bmatrix}0.70&0.30\\0.45&0.55\end{bmatrix}.$$

从故障状态出发初始分布是 $(0,1)$，两步以后得到 $(0.45,0.55)$。注意这与从正常初态得到 $(0.70,0.30)$ 不同。

**3.2。** 检查的期望成本为 $k+2$，等待的期望成本为7，因此 $k<5$ 时检查严格更有利；$k=5$ 时两种选择期望成本相同；$k>5$ 时等待更有利。

**3.3。** 若还剩 $n$ 步，记最小预期成本为 $V_n(s)$，则

$$V_0(s)=0,\quad V_n(s)=\min_a\left[c(s,a)+\sum_{s'}p(s'\mid s,a)V_{n-1}(s')\right].$$

对固定行动，括号内部是“当前成本加下一状态条件期望”；对合法行动取最小值。

**4.2。** $M_{\mathrm{KV}}\approx2\cdot32\cdot8\cdot128\cdot4096\cdot2=536{,}870{,}912$ 字节，等于512 MiB（不含其他显存开销）。

## 第5--9章选解

**5.2。** 随机初始化、数据顺序和部分底层计算过程会影响结果。重复实验可估计不同随机因素造成的波动。结果最好报告平均值、离散程度和原始任务级数据，而不是只展示最好的随机种子。

**6.2。** 读文件一般可重试但需要权限和版本控制；发送邮件需要幂等处理或发送状态查询；删除文件属于高风险写操作，应限制路径、要求明确授权并考虑不可逆风险。

**7.3。** 把失败分为网络不可达、依赖版本冲突、权限不足、包不存在、资源不足等。每一类定义不同的恢复动作，并设置最大重试次数和人工介入条件。

**8.1。** Recall@5为 $2/3$，第一条相关结果排第3位时，该查询的倒数排名是 $1/3$。MRR是多条查询倒数排名的平均，而不是单条查询的某个新指标。

**9.2。** 原样塞入全部历史会增加成本，且长上下文中重要约束仍可能被遗漏。结构化摘要虽然节省上下文，但必须保证关键约束、证据和版本不被压缩掉。

## 第10、11章选解

**10.2。** 当 $\theta=\ln3$ 时，$\sigma(\theta)=3/4$。$J'(\theta)=2\cdot(3/4)\cdot(1/4)=3/8=0.375$。

**10.3。** SFT优化参考Token的条件对数概率；端到端Agent成功还取决于工具是否执行、状态是否保存、重试是否正确以及验证器是否通过。Token级损失和环境级成功率并不是同一个目标函数。

**11.2。** 若每次独立成功概率为0.85，连续三次全部成功的概率为 $0.85^3\approx0.6141$。如果执行结果相关，则不能直接套乘法。

**11.4。** 假设测试集包含公开GitHub issue的原始描述和修复代码，而训练集包含同一issue的答案。模型可能记住修复内容，评测分数高却不能说明其能处理未知仓库问题。

## 第12--16章选解

**12.1。** A和B同时读取版本10，A先写入版本11，B随后根据旧版本10写入并覆盖A。解决方案之一是条件更新：B只有在当前版本仍等于10时才能提交，否则重新读取并合并。

**13.2。** 模型可能先执行违规工具调用，再在最终回复中表示拒绝。真正需要检查的是执行器日志、权限判定和环境副作用，不能仅检验最终自然语言。

**14.1。** 对 `syntax`，顺序是inspect、patch_syntax、test；对 `logic`，顺序是inspect、patch_logic、test。两类任务在当前确定性规则下都成功。它只是验证流程正确，不代表真实模型成功率。

**15.4。** 如果行号对应旧版本，不能直接把它写为已确认的证据。应读取当前提交与文件内容，重新定位，必要时标注“已过时／待核验”，再重新验证。

**16.4。** 最有说服力的求职项目不是展示很多Agent角色，而是展示：固定任务集、准确的环境版本、具备安全边界的系统、与基线相比的量化改进、失败分类和可复现实验过程。

## 第17--19章选解

**17.1。** 模型耗时 $8\times4=32$ 秒，加工具累计12秒，估计总耗时44秒。可以优化无效模型循环，或降低单次调用成本；并行执行无依赖的工具操作也可能降低时延，但需处理资源与一致性约束。

**18.1。** 当矩阵为 $4096\times4096$ 时，LoRA附加参数量为 $2\cdot4096\cdot r=8192r$。秩8为65536，秩16为131072，秩32为262144。

**19.1。** 构造四组：旧模型+旧检索、旧模型+新检索、新模型+旧检索、新模型+新检索。比较各组结果，并考察改动是否存在交互效应。必须在同一任务集、相同资源约束下测量。

# 术语对照与进一步阅读

## 重要术语的数学对应

| 常见术语 | 本书优先使用的解释 |
|:--|:--|
| Policy | 给定状态时行动的条件概率分布或确定性选择规则 |
| Value Function | 给定初始条件下累计收益的条件数学期望 |
| Action Space | 合法行动随机变量的取值集合 |
| State Transition | 状态转移的条件概率分布 |
| Reward | 单步收益随机变量或收益函数 |
| Return | 累计（可能折扣）收益随机变量 |
| Advantage | 某行动的条件期望收益与参考期望的差 |
| Trajectory | 状态、行动、观察及收益组成的样本路径 |
| Tool Use | 结构化动作提议、权限验证与外部执行 |
| Memory | 上下文、外部持久状态或模型参数，不应混用 |
| RAG | 检索证据并以其约束生成的系统方法 |
| Agent Harness | 任务状态、工具、循环、权限、观察与验证框架 |

## 岗位要求：官方信息

**[J1]** 阿里巴巴，算法工程师--AI Agent（2027届实习），2026年3月更新。公开岗位要求包括规划、推理、工具使用、记忆、RAG、SFT、DPO/PPO、强化学习、运行时和评估。链接：\url{https://campus-talent.alibaba.com/campus/position/199903260012}

**[J2]** 百度，智能体算法工程师（J102787），2026年7月21日。要求任务拆解、工具调用、Agent评估、数据飞轮、PyTorch、Transformer及训练经验。链接：\url{https://talent.baidu.com/jobs/detail/SOCIAL/94ae343d-0673-40ab-98a3-99c723690c37}

**[J3]** 阿里巴巴，Agent Infra工程师，2026年9月更新。涉及沙箱、调度、状态管理、高并发、认证与检查点。链接：\url{https://campus-talent.alibaba.com/campus/position/199907640058}

> 注：岗位描述是当时可检索到的公开文本，不同招聘批次和部门有所差异；职位要求会发生变化。岗位文字不能替代个人能力与实际项目验收。

## 建议精读的论文与技术资料

**[R1]** Vaswani et al. (2017), *Attention Is All You Need*. Transformer原始论文。\url{https://arxiv.org/abs/1706.03762}

**[R2]** Yao et al. (2022/2023), *ReAct: Synergizing Reasoning and Acting in Language Models*. 将模型推理与外部动作交替执行的经典路线。\url{https://arxiv.org/abs/2210.03629}

**[R3]** Rafailov et al. (2023), *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*. 理解DPO损失。\url{https://arxiv.org/abs/2305.18290}

**[R4]** DeepSeek-AI et al. (2025), *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*. 观察强化学习与推理能力。\url{https://arxiv.org/abs/2501.12948}

**[R5]** Hu et al. (2021), *LoRA: Low-Rank Adaptation of Large Language Models*. 低秩参数高效微调。\url{https://arxiv.org/abs/2106.09685}

**[R6]** Lewis et al. (2020), *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. RAG基础研究。\url{https://arxiv.org/abs/2005.11401}

**[R7]** Jimenez et al. (2023/2024), *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* 代码任务评测研究。\url{https://arxiv.org/abs/2310.06770}

**[R8]** Anthropic (2026), *Demystifying Evals for AI Agents*. Agent评测方法的工程说明。\url{https://www.anthropic.com/research/demystifying-evals-for-ai-agents}

**[R9]** OWASP, *LLM Prompt Injection Prevention Cheat Sheet*. 防范提示词注入的安全参考。\url{https://cheatsheetseries.owasp.org/cheatsheets/LLM_Prompt_Injection_Prevention_Cheat_Sheet.html}

**[R10]** PyTorch Tutorials, *Scaled Dot Product Attention and Transformer Building Blocks*. 阅读张量形状与高效注意力实现。\url{https://pytorch.org/tutorials/}

**[R11]** Sutton and Barto, *Reinforcement Learning: An Introduction*, Second Edition (2018). 以随机过程、期望和动态规划为主线补充理论。\url{http://incompleteideas.net/book/the-book-2nd.html}

**[R12]** Russell and Norvig, *Artificial Intelligence: A Modern Approach*, Fourth Edition. 搜索、逻辑、概率推断和决策的经典体系。\url{https://aima.cs.berkeley.edu/}

## 推荐阅读顺序

阅读论文时，建议先读其**问题定义与实验设计**，再读算法，再看实现；遇到新术语，先翻译回随机变量、概率分布、数学期望、优化目标和系统操作。不要因为模型名字或框架名字变化，就误以为基础数学也发生了变化。

对Agent岗位而言，最有价值的研究习惯是逐步建立证据链：问题是什么、旧方法为什么不够、提出什么可证伪假设、怎样控制实验变量、结果能否复现，以及哪些限制还没有解决。

\cleardoublepage
\begin{center}
\vspace*{35mm}
{\sffamily\Large\bfseries 结束语}\\[12mm]
\end{center}

成为优秀的 Agent 算法工程师，不取决于能背出多少缩写，也不取决于使用了多少框架，而取决于能否把真实问题还原为明确的数学对象和工程约束，用算法提出可检验的改进，并在结果不符合预期时承认问题、修正方法。

**数学让你知道自己在优化什么，系统工程让你知道程序实际做了什么，实验让你知道改动到底有没有用。**

\vspace{16mm}
\begin{center}\color{accent}\rule{0.62\linewidth}{1.3pt}\end{center}
