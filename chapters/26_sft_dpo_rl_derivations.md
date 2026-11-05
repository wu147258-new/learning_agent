# SFT、DPO与强化学习：从概率目标到参数更新

## 为什么同一个模型需要不同的训练数据

考虑一个能够选择“读取日志”或“修改配置”的Agent。可以收集三种资料：

1. 专家在某个任务状态下采取了什么动作；
2. 对于同一个问题，专家更喜欢两个候选答案中的哪一个；
3. 某个执行轨迹最终是否真正完成任务，以及产生多少成本。

这三类资料表达的是不同的信息。示例动作构成监督学习，成对偏好构成比较数据，环境反馈才直接提供可用于计算任务收益的随机变量。训练方法的差别首先应从**数据给出的数学约束**解释。

### 统一使用条件概率分布

设输入为 $x$、输出序列为 $y$，模型定义：

$$
\pi_\theta(y\mid x)
=P_\theta(Y=y\mid X=x),\qquad \sum_y\pi_\theta(y\mid x)=1.
$$

在离散序列模型里，$y=(y_1,\ldots,y_T)$ 的概率由逐Token条件概率相乘：

$$
\log\pi_\theta(y\mid x)
=\sum_{t=1}^T\log p_\theta(y_t\mid x,y_{<t}).
$$

实践中还需约定结束Token、截断和Padding。不同长度序列如果把Token平均对数概率与总对数概率混为一谈，将得到不同的优化目标。

## SFT：从最大似然估计开始

### 指令—答案样本是什么

设监督数据集：

$$D_{\mathrm{SFT}}=\{(x_i,y_i)\}_{i=1}^N.$$

在样本按训练协议抽取的条件下，最大化答案条件似然：

$$
\max_\theta \prod_i\pi_\theta(y_i\mid x_i).
$$

对数将乘积转成求和。训练时通常最小化负对数似然：

$$
\boxed{
L_{\mathrm{SFT}}(\theta)=
-\frac1N\sum_{i=1}^{N}\log\pi_\theta(y_i\mid x_i).
}
$$

如果是批量逐Token交叉熵，应明确分母是样本数、有效Token数，还是逐样本归一化后的均值。不同方式改变长答案与短答案的相对权重。

### 为什么不能把提示词也当作答案训练

指令微调常把提示词 $x$ 与答案 $y$ 拼成同一个Token序列输入模型。训练希望模型**根据提示词预测答案**，而不是让模型损失大量权重去拟合已有提示词。

设答案部分从第 $s$ 个位置开始，可把训练标签中不属于答案的Token置为忽略标记，只对答案位置求交叉熵。这称为响应掩码（response masking）。必须特别注意自回归标签右移之后的索引，避免把真实答案提前暴露给位置 $t$ 的预测。

### 手算二行动最大似然

在一个固定状态下，专家做了10次选择，其中行动0出现8次、行动1出现2次。令 $\pi(0)=p$、$\pi(1)=1-p$：

$$
L(p)=-\frac1{10}[8\log p+2\log(1-p)].
$$

求导：

$$L'(p)=-\frac8{10p}+\frac2{10(1-p)}.$$

令导数为0，得到 $p=0.8$。这说明理想的无限制最大似然解拟合示例频率。

**它并没有证明行动0的客观收益最高。** 如果专家数据包含错误或只能覆盖简单任务，SFT会继承这些限制。

### 标签平滑与数据质量

如果训练数据存在多个合理答案，单一参考答案的最大似然目标未必充分反映任务完成质量。训练者有时采用标签平滑、多个参考答案、样本重加权等方法，但这些方法会改变目标函数，不能仅凭“更稳定”就假定效果更好。

数据去重、权限过滤、证据溯源、训练/测试分割和标签质量核验通常比增加训练轮数更重要。错误答案重复出现时，模型很可能更努力地拟合错误。

## DPO：从带KL约束的奖励目标推导

### 为什么仅有偏好比较仍能训练

假设标注者不知道给每个回答打多少分，但可以判断同一提示词下两个回答 $y^+$、$y^-$ 哪个更好。

这种观测并没有给出绝对奖励 $r(x,y)$，却给出了比较事件：

$$Y^+\succ Y^-.$$

DPO将偏好数据连接到一个特定的带KL约束的奖励优化问题。

### 先写出KL正则化目标

对于固定输入 $x$，设参考条件分布为 $\pi_{\rm ref}(y\mid x)$，奖励为 $r(x,y)$，约束系数为 $\beta>0$。目标：

$$
\max_{\pi}
\left[
\sum_y\pi(y\mid x)r(x,y)
-\beta D_{\rm KL}(\pi(\cdot\mid x)\|\pi_{\rm ref}(\cdot\mid x))
\right].
$$

其中：

$$
D_{\rm KL}(\pi\|\pi_{\rm ref})
=\sum_y\pi(y\mid x)\log\frac{\pi(y\mid x)}
{\pi_{\rm ref}(y\mid x)}.
$$

这里要求参考分布对所考虑响应具有正概率，且归一化常数有限。KL项限制候选分布偏离参考分布过大。它不是硬性地禁止任何分布变化，而是将变化成本加入优化目标。

### 利用拉格朗日乘子求最优分布

引入归一化约束 $\sum_y\pi(y\mid x)=1$ 的乘子 $\lambda$。对每个 $\pi(y\mid x)$ 求导：

$$
r(x,y)-\beta\left[\log\frac{\pi(y\mid x)}
{\pi_{\rm ref}(y\mid x)}+1\right]+\lambda=0.
$$

整理可得：

$$
\boxed{
\pi^*(y\mid x)
=\frac{\pi_{\rm ref}(y\mid x)\exp(r(x,y)/\beta)}{Z(x)},
}
$$

其中：

$$
Z(x)=\sum_y\pi_{\rm ref}(y\mid x)\exp(r(x,y)/\beta).
$$

反解奖励：

$$
r(x,y)=\beta\log\frac{\pi^*(y\mid x)}
{\pi_{\rm ref}(y\mid x)}
+\beta\log Z(x).
$$

对于同一个输入 $x$ 的两个答案，最后一项相同，做差时会抵消。

### 从成对比较得到DPO损失

采用Bradley–Terry型成对偏好概率模型：

$$
P(y^+\succ y^-\mid x)
=\sigma(r(x,y^+)-r(x,y^-)),
$$

其中 $\sigma(z)=1/(1+e^{-z})$。令：

$$
d_\theta=
\left[\log\pi_\theta(y^+\mid x)
-\log\pi_\theta(y^-\mid x)\right]
-
\left[\log\pi_{\rm ref}(y^+\mid x)
-\log\pi_{\rm ref}(y^-\mid x)\right].
$$

代入可得：

$$
P_\theta(y^+\succ y^-\mid x)=\sigma(\beta d_\theta).
$$

因此最大化偏好比较的条件似然，等价于最小化：

$$
\boxed{
L_{\rm DPO}(\theta)
=-\mathbb E_{(x,y^+,y^-)\sim D}
\log\sigma(\beta d_\theta).
}
$$

\begin{figure}[htbp]
\centering
\resizebox{0.94\linewidth}{!}{\input{figures/training_objectives.tex}}
\caption{SFT、DPO和强化学习所需的观测数据及对应优化目标}
\end{figure}

### DPO梯度体现了什么

对单个比较样本：

$$
\frac{\partial L_{\rm DPO}}{\partial d_\theta}
=-\beta\,\sigma(-\beta d_\theta).
$$

若偏好的回答相对于参考模型不够占优，$d_\theta$ 较小，推动其相对概率增加的梯度较大；若已经十分占优，梯度逐步变小。

这是一种**参考分布相对对数比**的学习，不是简单把偏好答案的绝对概率升高。实际训练还必须冻结参考模型，明确响应Token的对数概率聚合方法及偏好标签质量。

### DPO不是“所有RL的替代品”

DPO能够在离线偏好数据上更新模型，而不必在这一阶段在线采样、训练显式奖励网络并运行PPO。但它不能自动探索没有出现在偏好数据中的交互路径，也不能保证偏好标注等价于真实环境的长期任务收益。

如果用户偏好短答案，而实际任务要求精确执行复杂工具链，偏好模型可能与环境成功率不一致。必须分别评估。

## 强化学习：直接优化累计随机收益

### 从单步有限行动开始

固定状态 $s$ 下，有两个行动0和1，奖励分别为 $r(0)=1,r(1)=3$。令模型产生的行动条件分布为 $\pi_\theta(a\mid s)$。

目标定义为：

$$
J(\theta)=\mathbb E_{A\sim\pi_\theta}[r(A)]
=\sum_a\pi_\theta(a\mid s)r(a).
$$

若两行动概率都为0.5，当前期望收益为2。提高高收益行动1的概率，能提高理论期望，但要考虑行动被允许、奖励定义正确等前提。

### 推导对数导数恒等式

只要 $\pi_\theta(a)>0$ 且可微：

$$
\nabla_\theta\pi_\theta(a)
=\pi_\theta(a)\nabla_\theta\log\pi_\theta(a).
$$

代入期望的梯度：

$$
\begin{aligned}
\nabla_\theta J
&=\sum_a\nabla_\theta\pi_\theta(a)r(a)\\
&=\sum_a\pi_\theta(a)r(a)\nabla_\theta\log\pi_\theta(a)\\
&=\mathbb E_{A\sim\pi_\theta}
[r(A)\nabla_\theta\log\pi_\theta(A)].
\end{aligned}
$$

这个恒等式是REINFORCE类方法的核心，但在采样数量有限时，梯度估计可能具有很高的方差。

### 为什么可以减去基线

若基线 $b(s)$ 对给定状态不依赖采样行动：

$$
\begin{aligned}
\mathbb E[b(s)\nabla_\theta\log\pi_\theta(A\mid s)]
&=b(s)\sum_a\nabla_\theta\pi_\theta(a\mid s)\\
&=b(s)\nabla_\theta1=0.
\end{aligned}
$$

因此：

$$
\nabla_\theta J
=\mathbb E[(r(A)-b(s))\nabla_\theta\log\pi_\theta(A\mid s)].
$$

合适的基线可能降低采样方差，却不能任意取一个与行动相关的数仍套用该结论。若基线由同一网络产生，还需要对相应的梯度路径作正确处理。

## 多步环境中的Bellman关系

### 状态与行动的随机转移

设 $S_t$ 为状态随机变量、$A_t$ 为行动随机变量。环境转移由：

$$P(S_{t+1}=s'\mid S_t=s,A_t=a)$$

定义。条件行动分布通常记为 $\pi_\theta(a\mid s)$，它在强化学习文献中称为策略，严格地说是给定状态后的行动条件概率分布。

若每步奖励为 $R_t$，折扣因子 $0\le\gamma<1$，累计收益：

$$
G_t=\sum_{k=0}^{\infty}\gamma^kR_{t+k}.
$$

有限任务则将求和限制在终止时刻，终止后的奖励视具体定义处理。

### 条件期望与价值函数

定义：

$$V^\pi(s)=\mathbb E_\pi[G_t\mid S_t=s].$$

在马尔可夫假设和适当可积条件下：

$$
V^\pi(s)=\sum_a\pi(a\mid s)
\sum_{s'}P(s'\mid s,a)
\left[r(s,a,s')+\gamma V^\pi(s')\right].
$$

这是Bellman期望方程。它来自全期望公式以及回报的递推定义，而不是只需记忆的行业术语。

### 轨迹概率与梯度

若环境转移分布不依赖待优化参数 $\theta$，轨迹 $\tau=(S_0,A_0,\ldots,S_T)$ 的联合概率可分解：

$$
P_\theta(\tau)
=P(S_0)\prod_{t=0}^{T-1}
\pi_\theta(A_t\mid S_t)\,
P(S_{t+1}\mid S_t,A_t).
$$

于是：

$$
\nabla_\theta\log P_\theta(\tau)
=\sum_{t=0}^{T-1}\nabla_\theta\log\pi_\theta(A_t\mid S_t).
$$

可将累计收益乘上轨迹对数梯度，构造对 $\nabla_\theta\mathbb E[G]$ 的采样估计。若环境动力学也依赖 $\theta$，就不能不加说明地忽略它的梯度项。

## 优势函数、GAE与PPO

### 为什么需要估计行动相对于基线的好坏

定义行动条件价值：

$$
Q^\pi(s,a)=\mathbb E_\pi[G_t\mid S_t=s,A_t=a].
$$

优势：

$$A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s).$$

它表示在状态 $s$ 下采取行动 $a$ 的期望回报与该状态的平均回报之差。注意它是条件期望的差，不是对一个动作好坏的主观评分。

### GAE的时间差分递推

对于价值估计 $\widehat V$，定义单步时间差分：

$$\delta_t=R_t+\gamma\widehat V(S_{t+1})-\widehat V(S_t).$$

广义优势估计（GAE）的一种形式为：

$$
\widehat A_t=\sum_{\ell=0}^{T-t-1}
(\gamma\lambda)^\ell\delta_{t+\ell},\qquad0\le\lambda\le1.
$$

在有限终止轨迹上，应正确处理终止状态的价值和截断（truncation）状态的自举项；把时间限制截断都误当作真实终止，会造成偏差。

### PPO的剪切代理目标

设采样时的旧分布为 $\pi_{\rm old}$，需要优化的新分布为 $\pi_\theta$。对采样行动定义：

$$
\rho_t(\theta)=
\frac{\pi_\theta(A_t\mid S_t)}
{\pi_{\rm old}(A_t\mid S_t)}.
$$

PPO的一种常见剪切代理目标：

$$
\boxed{
L^{\rm CLIP}(\theta)=
\mathbb E\left[
\min\left(
\rho_t\widehat A_t,\,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)
\widehat A_t
\right)
\right].
}
$$

优化时最大化此目标。对正优势，过高的比率不会持续得到额外奖励；对负优势，过低的比率也受到剪切限制。剪切机制降低单次更新过大的激励，但**不严格保证KL距离始终不超过固定阈值**，实际仍需要监控KL与奖励、失败率等指标。

### 为什么不应重复使用不断变化的旧概率

PPO比率的分母应对应**实际采样该数据时的旧分布**。若每次训练迭代都随意把分母改成当前新分布，比例会失去应有意义。多轮小批量更新应明确数据收集时点与旧模型快照。

## 从公式实现SFT和DPO损失

### 对答案Token求对数概率

对于相同长度的合成训练序列，可以在模型输入中保留提示词，但只让答案Token参与监督：

~~~python
import torch
from torch.nn import functional as F

def answer_logprob(model, seq, prompt_len):
    logits = model(seq[:, :-1])
    labels = seq[:, 1:]
    token_logp = F.log_softmax(
        logits, dim=-1
    ).gather(
        -1, labels.unsqueeze(-1)
    ).squeeze(-1)
    return token_logp[:, prompt_len-1:].sum(-1)
~~~

若完整序列是BOS、提示词、答案、EOS，第一个答案Token出现在原序列的索引prompt_len处。模型输入中索引prompt_len-1的位置负责预测它，所以切片必须从prompt_len-1开始。

注意返回的是**每个完整答案的对数概率之和**，不是逐Token平均值。DPO通常对响应序列的对数概率作比较；改变聚合方式会改变偏好目标对长度的处理。

### 从答案概率组装DPO损失

~~~python
def dpo_loss(
    model, chosen, rejected,
    ref_chosen_logp, ref_rejected_logp,
    prompt_len, beta
):
    chosen_logp = answer_logprob(
        model, chosen, prompt_len
    )
    rejected_logp = answer_logprob(
        model, rejected, prompt_len
    )
    margin = (
        chosen_logp - rejected_logp
        - ref_chosen_logp.detach()
        + ref_rejected_logp.detach()
    )
    return -F.logsigmoid(beta * margin).mean()
~~~

这里要求首选和拒绝答案对应相同提示词，并且参考模型已冻结。训练模型只需要产生新的条件概率，参考模型不能随着当前优化步骤一起更新。对不同长度的响应要额外处理Padding、EOS及截断，不能直接套用固定长度示例。

### 从失败输入检查实现

若错误地将prompt_len改小1，则会把一个提示词Token纳入答案对数概率；虽然代码依然能反向传播，但优化目标已经变化。完整实验以显式的逐Token求和断言检查这一索引关系，保证“程序能运行”与“公式实现正确”是两项分别检查的要求。


## 工程中的训练数据与安全验证

### 完整的训练记录

训练样本至少应区分：

- 输入提示词及其授权范围；
- 监督答案或成对偏好；
- 工具调用轨迹和外部环境版本；
- 原始奖励与奖励计算依据；
- 训练/验证/测试分割；
- 模型权重、随机种子、优化器和学习率。

对真实Agent，文本偏好和环境成功率可能冲突。仅靠偏好模型得分无法证明代码修复正确、数据查询没有越权或付款操作没有重复执行。

### 奖励错配的反例

假设奖励定义为“程序退出码为0时得分1”。某个Agent可以提交只包含退出码0、但从未运行目标测试的脚本。模型得分提高，真实任务仍未完成。这是奖励设计被利用的例子。

改进方式不是简单提高奖励权重，而是建立不可由同一Agent任意修改的独立检查器，校验测试覆盖、文件变化和目标条件。

## PyTorch离线对照实验

### 运行命令

~~~bash
python examples/sft_dpo_rl_lab.py
~~~

实验包括：有限行动的SFT最大似然、偏好对DPO、两行动期望收益梯度与基线恒等式、PPO正负优势剪切数值检查，以及有限回报的GAE递推。

使用合成分布时能验证公式和梯度，不应把结果冒充真实大模型微调或RLHF的性能。

### 在真实Decoder上运行SFT与DPO

两行动实验说明目标函数的数学性质，但真实语言模型的输出是完整Token序列。为连接两者，运行：

~~~bash
python examples/tiny_lm_sft_dpo.py
~~~

程序导入仓库现有的TinyDecoder，构造两条合成任务提示词，各有一个首选答案和一个拒绝答案。它执行三个步骤：

1. **SFT**：将提示词与答案拼成序列，使用右移标签，但仅把答案Token计入交叉熵；提示词Token是条件信息，不被当作监督答案。
2. **冻结参考模型**：复制SFT完成的参数快照，关闭参考模型梯度，并分别计算首选和拒绝答案在参考模型下的完整响应对数概率。
3. **DPO**：在可训练Decoder下重新计算这两个响应的对数概率，减去参考模型的对数比，优化负对数Sigmoid损失。

特别注意对数概率的单位：一个响应的 $\log \pi_\theta(y\mid x)$ 是**答案Token对数概率之和**，不能直接把不同Token位置的Softmax最大概率相加，也不能将提示词条件输入错误计入被优化的答案。

实验检查：响应掩码索引与显式求和一致；SFT损失下降；DPO初始损失等于 $\log2$；优化后首选响应相对拒绝响应的对数优势大于固定参考模型；参考模型所有参数均未获得梯度。

只使用两条合成提示词，故此实验能说明公式与实现自洽，**不能**证明模型获得了真实指令遵循、泛化或人类偏好对齐能力。

### 如何迁移到序列语言模型

1. 把两行动Softmax换为真实模型的逐Token对数概率。
2. SFT只对答案Token计算损失，明确每样本和每Token归一化。
3. DPO对每个偏好答案分别求**完整响应的对数概率**，使用固定的参考模型计算对数比。
4. RL需要从当前模型采样轨迹，记录每次动作概率、奖励、终止状态与执行成本，再计算更新目标。
5. 保留冻结的验证任务集和独立执行检查器，不把训练损失当作最终业务成功率。

## 练习与详解

### 练习：SFT最大似然

十次专家动作里行动0出现7次、行动1出现3次。求无限制二分类分布的最大似然估计。

**解：** 令 $\pi(0)=p$，对数似然为 $7\log p+3\log(1-p)$。导数 $7/p-3/(1-p)=0$，得 $p=0.7$。

### 练习：DPO对数比

参考分布为 $(0.2,0.8)$，当前分布为 $(0.6,0.4)$，偏好行动0胜过1。计算 $d_\theta$。

**解：**

$$
d_\theta=
\log(0.6/0.4)-\log(0.2/0.8)
=\log1.5-\log0.25=\log6.
$$

若 $\beta=1$，单样本DPO损失是 $-\log\sigma(\log6)=-\log(6/7)=\log(7/6)$。

### 练习：期望收益梯度

两行动收益为1和3，初始Softmax得分都为0。求对两个得分的梯度。

**解：** 两行动概率都为0.5，$J=2$。使用Softmax导数可得 $\partial J/\partial z_0=-0.5$、$\partial J/\partial z_1=+0.5$。梯度上升倾向于提高行动1的概率。

### 练习：PPO剪切

设 $\epsilon=0.2$，样本优势为 $A=2$，新旧概率比为 $\rho=1.5$。剪切代理样本目标是多少？

**解：** 第一项为 $1.5\times2=3$，第二项为 $1.2\times2=2.4$，取最小值为2.4。

### 进阶编程

1. 修改偏好标签使30%比较被翻转，分析DPO损失与泛化验证的变化。
2. 把SFT样本频率改成0.2/0.8，检查收敛方向。
3. 使用5个随机种子做蒙特卡洛梯度估计，报告均值与方差。
4. 构造一个奖励高但未完成真实任务的执行轨迹，验证外部检查器能拒绝。
5. 对比不同 $\epsilon$ 下PPO剪切样本项，画出正优势和负优势的形状。

## 参考文献

- Ouyang et al. (2022), *Training language models to follow instructions with human feedback*, arXiv:2203.02155.
- Rafailov et al. (2023), *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*, arXiv:2305.18290.
- Williams (1992), *Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning*.
- Schulman et al. (2016), *High-Dimensional Continuous Control Using Generalized Advantage Estimation*, arXiv:1506.02438.
- Schulman et al. (2017), *Proximal Policy Optimization Algorithms*, arXiv:1707.06347.
