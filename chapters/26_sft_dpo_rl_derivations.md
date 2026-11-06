# SFT、DPO与强化学习：从概率目标到参数更新

## 为什么同一个模型需要不同的训练数据

考虑一个能够选择“读取日志”或“修改配置”的Agent。可以收集三种资料：

1. 专家在某个任务状态下采取了什么动作；
2. 对于同一个问题，专家更喜欢两个候选答案中的哪一个；
3. 某个执行轨迹最终是否真正完成任务，以及产生多少成本。

这三类资料表达的是不同的信息。示例动作构成监督学习，成对偏好构成比较数据，环境反馈才直接提供可用于计算任务收益的随机变量。训练方法的差别首先应从**数据给出的数学约束**解释。

### 统一使用条件概率分布

设输入为 $x$、输出序列为 $y$，模型定义：

\begin{equation*}
\pi_\theta(y\mid x)
=P_\theta(Y=y\mid X=x),\qquad \sum_y\pi_\theta(y\mid x)=1.
\end{equation*}

在离散序列模型里，$y=(y_1,\ldots,y_T)$ 的概率由逐Token条件概率相乘：

\begin{equation*}
\log\pi_\theta(y\mid x)
=\sum_{t=1}^T\log p_\theta(y_t\mid x,y_{<t}).
\end{equation*}

实践中还需约定结束Token、截断和Padding。不同长度序列如果把Token平均对数概率与总对数概率混为一谈，将得到不同的优化目标。

## SFT：从最大似然估计开始

### 指令—答案样本是什么

设监督数据集：

$$D_{\mathrm{SFT}}=\{(x_i,y_i)\}_{i=1}^N.$$

在样本按训练协议抽取的条件下，最大化答案条件似然：

\begin{equation*}
\max_\theta \prod_i\pi_\theta(y_i\mid x_i).
\end{equation*}

对数将乘积转成求和。训练时通常最小化负对数似然：

\begin{equation*}
\boxed{
L_{\mathrm{SFT}}(\theta)=
-\frac1N\sum_{i=1}^{N}\log\pi_\theta(y_i\mid x_i).
}
\end{equation*}

如果是批量逐Token交叉熵，应明确分母是样本数、有效Token数，还是逐样本归一化后的均值。不同方式改变长答案与短答案的相对权重。

### 为什么不能把提示词也当作答案训练

指令微调常把提示词 $x$ 与答案 $y$ 拼成同一个Token序列输入模型。训练希望模型**根据提示词预测答案**，而不是让模型损失大量权重去拟合已有提示词。

设答案部分从第 $s$ 个位置开始，可把训练标签中不属于答案的Token置为忽略标记，只对答案位置求交叉熵。这称为响应掩码（response masking）。必须特别注意自回归标签右移之后的索引，避免把真实答案提前暴露给位置 $t$ 的预测。

### 手算二行动最大似然

在一个固定状态下，专家做了10次选择，其中行动0出现8次、行动1出现2次。令 $\pi(0)=p$、$\pi(1)=1-p$：

\begin{equation*}
L(p)=-\frac1{10}[8\log p+2\log(1-p)].
\end{equation*}

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

\begin{equation*}
\max_{\pi}
\left[
\sum_y\pi(y\mid x)r(x,y)
-\beta D_{\rm KL}(\pi(\cdot\mid x)\|\pi_{\rm ref}(\cdot\mid x))
\right].
\end{equation*}

其中：

\begin{equation*}
D_{\rm KL}(\pi\|\pi_{\rm ref})
=\sum_y\pi(y\mid x)\log\frac{\pi(y\mid x)}
{\pi_{\rm ref}(y\mid x)}.
\end{equation*}

这里要求参考分布对所考虑响应具有正概率，且归一化常数有限。KL项限制候选分布偏离参考分布过大。它不是硬性地禁止任何分布变化，而是将变化成本加入优化目标。

### 利用拉格朗日乘子求最优分布

引入归一化约束 $\sum_y\pi(y\mid x)=1$ 的乘子 $\lambda$。对每个 $\pi(y\mid x)$ 求导：

\begin{equation*}
r(x,y)-\beta\left[\log\frac{\pi(y\mid x)}
{\pi_{\rm ref}(y\mid x)}+1\right]+\lambda=0.
\end{equation*}

整理可得：

\begin{equation*}
\boxed{
\pi^*(y\mid x)
=\frac{\pi_{\rm ref}(y\mid x)\exp(r(x,y)/\beta)}{Z(x)},
}
\end{equation*}

其中：

\begin{equation*}
Z(x)=\sum_y\pi_{\rm ref}(y\mid x)\exp(r(x,y)/\beta).
\end{equation*}

反解奖励：

\begin{equation*}
r(x,y)=\beta\log\frac{\pi^*(y\mid x)}
{\pi_{\rm ref}(y\mid x)}
+\beta\log Z(x).
\end{equation*}

对于同一个输入 $x$ 的两个答案，最后一项相同，做差时会抵消。

### 从成对比较得到DPO损失

采用Bradley–Terry型成对偏好概率模型：

\begin{equation*}
P(y^+\succ y^-\mid x)
=\sigma(r(x,y^+)-r(x,y^-)),
\end{equation*}

其中 $\sigma(z)=1/(1+e^{-z})$。令：

\begin{equation*}
d_\theta=
\left[\log\pi_\theta(y^+\mid x)
-\log\pi_\theta(y^-\mid x)\right]
-
\left[\log\pi_{\rm ref}(y^+\mid x)
-\log\pi_{\rm ref}(y^-\mid x)\right].
\end{equation*}

代入可得：

\begin{equation*}
P_\theta(y^+\succ y^-\mid x)=\sigma(\beta d_\theta).
\end{equation*}

因此最大化偏好比较的条件似然，等价于最小化：

\begin{equation*}
\boxed{
L_{\rm DPO}(\theta)
=-\mathbb E_{(x,y^+,y^-)\sim D}
\log\sigma(\beta d_\theta).
}
\end{equation*}

\begin{figure}[htbp]
\centering
\resizebox{0.94\linewidth}{!}{\input{figures/training_objectives.tex}}
\caption{SFT、DPO和强化学习所需的观测数据及对应优化目标}
\end{figure}

### DPO梯度体现了什么

对单个比较样本：

\begin{equation*}
\frac{\partial L_{\rm DPO}}{\partial d_\theta}
=-\beta\,\sigma(-\beta d_\theta).
\end{equation*}

若偏好的回答相对于参考模型不够占优，$d_\theta$ 较小，推动其相对概率增加的梯度较大；若已经十分占优，梯度逐步变小。

这是一种**参考分布相对对数比**的学习，不是简单把偏好答案的绝对概率升高。实际训练还必须冻结参考模型，明确响应Token的对数概率聚合方法及偏好标签质量。

### DPO不是“所有RL的替代品”

DPO能够在离线偏好数据上更新模型，而不必在这一阶段在线采样、训练显式奖励网络并运行PPO。但它不能自动探索没有出现在偏好数据中的交互路径，也不能保证偏好标注等价于真实环境的长期任务收益。

如果用户偏好短答案，而实际任务要求精确执行复杂工具链，偏好模型可能与环境成功率不一致。必须分别评估。

## 强化学习：直接优化累计随机收益

### 从单步有限行动开始

固定状态 $s$ 下，有两个行动0和1，奖励分别为 $r(0)=1,r(1)=3$。令模型产生的行动条件分布为 $\pi_\theta(a\mid s)$。

目标定义为：

\begin{equation*}
J(\theta)=\mathbb E_{A\sim\pi_\theta}[r(A)]
=\sum_a\pi_\theta(a\mid s)r(a).
\end{equation*}

若两行动概率都为0.5，当前期望收益为2。提高高收益行动1的概率，能提高理论期望，但要考虑行动被允许、奖励定义正确等前提。

### 推导对数导数恒等式

只要 $\pi_\theta(a)>0$ 且可微：

\begin{equation*}
\nabla_\theta\pi_\theta(a)
=\pi_\theta(a)\nabla_\theta\log\pi_\theta(a).
\end{equation*}

代入期望的梯度：

\begin{equation*}
\begin{aligned}
\nabla_\theta J
&=\sum_a\nabla_\theta\pi_\theta(a)r(a)\\
&=\sum_a\pi_\theta(a)r(a)\nabla_\theta\log\pi_\theta(a)\\
&=\mathbb E_{A\sim\pi_\theta}
[r(A)\nabla_\theta\log\pi_\theta(A)].
\end{aligned}
\end{equation*}

这个恒等式是REINFORCE类方法的核心，但在采样数量有限时，梯度估计可能具有很高的方差。

### 为什么可以减去基线

若基线 $b(s)$ 对给定状态不依赖采样行动：

\begin{equation*}
\begin{aligned}
\mathbb E[b(s)\nabla_\theta\log\pi_\theta(A\mid s)]
&=b(s)\sum_a\nabla_\theta\pi_\theta(a\mid s)\\
&=b(s)\nabla_\theta1=0.
\end{aligned}
\end{equation*}

因此：

\begin{equation*}
\nabla_\theta J
=\mathbb E[(r(A)-b(s))\nabla_\theta\log\pi_\theta(A\mid s)].
\end{equation*}

合适的基线可能降低采样方差，却不能任意取一个与行动相关的数仍套用该结论。若基线由同一网络产生，还需要对相应的梯度路径作正确处理。

## 多步环境中的Bellman关系

### 状态与行动的随机转移

设 $S_t$ 为状态随机变量、$A_t$ 为行动随机变量。环境转移由：

$$P(S_{t+1}=s'\mid S_t=s,A_t=a)$$

定义。条件行动分布通常记为 $\pi_\theta(a\mid s)$，它在强化学习文献中称为策略，严格地说是给定状态后的行动条件概率分布。

若每步奖励为 $R_t$，折扣因子 $0\le\gamma<1$，累计收益：

\begin{equation*}
G_t=\sum_{k=0}^{\infty}\gamma^kR_{t+k}.
\end{equation*}

有限任务则将求和限制在终止时刻，终止后的奖励视具体定义处理。

### 条件期望与价值函数

定义：

$$V^\pi(s)=\mathbb E_\pi[G_t\mid S_t=s].$$

在马尔可夫假设和适当可积条件下：

\begin{equation*}
V^\pi(s)=\sum_a\pi(a\mid s)
\sum_{s'}P(s'\mid s,a)
\left[r(s,a,s')+\gamma V^\pi(s')\right].
\end{equation*}

这是Bellman期望方程。它来自全期望公式以及回报的递推定义，而不是只需记忆的行业术语。

### 轨迹概率与梯度

若环境转移分布不依赖待优化参数 $\theta$，轨迹 $\tau=(S_0,A_0,\ldots,S_T)$ 的联合概率可分解：

\begin{equation*}
P_\theta(\tau)
=P(S_0)\prod_{t=0}^{T-1}
\pi_\theta(A_t\mid S_t)\,
P(S_{t+1}\mid S_t,A_t).
\end{equation*}

于是：

\begin{equation*}
\nabla_\theta\log P_\theta(\tau)
=\sum_{t=0}^{T-1}\nabla_\theta\log\pi_\theta(A_t\mid S_t).
\end{equation*}

可将累计收益乘上轨迹对数梯度，构造对 $\nabla_\theta\mathbb E[G]$ 的采样估计。若环境动力学也依赖 $\theta$，就不能不加说明地忽略它的梯度项。

## 优势函数、GAE与PPO

### 为什么需要估计行动相对于基线的好坏

定义行动条件价值：

\begin{equation*}
Q^\pi(s,a)=\mathbb E_\pi[G_t\mid S_t=s,A_t=a].
\end{equation*}

优势：

$$A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s).$$

它表示在状态 $s$ 下采取行动 $a$ 的期望回报与该状态的平均回报之差。注意它是条件期望的差，不是对一个动作好坏的主观评分。

### GAE的时间差分递推

对于价值估计 $\widehat V$，定义单步时间差分：

$$\delta_t=R_t+\gamma\widehat V(S_{t+1})-\widehat V(S_t).$$

广义优势估计（GAE）的一种形式为：

\begin{equation*}
\widehat A_t=\sum_{\ell=0}^{T-t-1}
(\gamma\lambda)^\ell\delta_{t+\ell},\qquad0\le\lambda\le1.
\end{equation*}

在有限终止轨迹上，应正确处理终止状态的价值和截断（truncation）状态的自举项；把时间限制截断都误当作真实终止，会造成偏差。

### PPO的剪切代理目标

设采样时的旧分布为 $\pi_{\rm old}$，需要优化的新分布为 $\pi_\theta$。对采样行动定义：

\begin{equation*}
\rho_t(\theta)=
\frac{\pi_\theta(A_t\mid S_t)}
{\pi_{\rm old}(A_t\mid S_t)}.
\end{equation*}

PPO的一种常见剪切代理目标：

\begin{equation*}
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
\end{equation*}

优化时最大化此目标。对正优势，过高的比率不会持续得到额外奖励；对负优势，过低的比率也受到剪切限制。剪切机制降低单次更新过大的激励，但**不严格保证KL距离始终不超过固定阈值**，实际仍需要监控KL与奖励、失败率等指标。

### 为什么不应重复使用不断变化的旧概率

PPO比率的分母应对应**实际采样该数据时的旧分布**。若每次训练迭代都随意把分母改成当前新分布，比例会失去应有意义。多轮小批量更新应明确数据收集时点与旧模型快照。

## 结合DeepSeek-R1理解GRPO：为什么不一定需要价值网络

### 从PPO的价值基线转向组内比较

前文的PPO使用优势估计 $\widehat A_t$，实践中可借助价值网络预测 $V(s)$。但在可自动判定答案正确性的任务里，可以针对**同一个提示词**采样多个候选回答，由这些回答的相对奖励构造优势。

DeepSeek-R1论文公开了使用Group Relative Policy Optimization（GRPO）的强化学习路线：对同一问题采样一组输出，使用组内奖励归一化作为优势信号，并结合策略更新约束。论文还区分了以纯强化学习启动的R1-Zero与包含冷启动监督数据和后续训练阶段的R1；两者不应混为同一训练流程。

设某个问题 $q$ 有 $G$ 个候选答案，其奖励为：

\begin{equation*}
r_1,r_2,\ldots,r_G.
\end{equation*}

令组内均值和标准差为：

\begin{equation*}
\bar r=\frac1G\sum_{i=1}^G r_i,\qquad
s_r=\sqrt{\frac1G\sum_{i=1}^G(r_i-\bar r)^2}.
\end{equation*}

当 $s_r>0$ 时，可以构造：

\begin{equation*}
\boxed{A_i=\frac{r_i-\bar r}{s_r}.}
\end{equation*}

该式解释了同一道题中“比其他答案好”与“奖励绝对值高”的区别。组内中心化并不保证奖励本身无偏，也不能修复错误的自动检查器。

### 逐步手算四个候选的优势

假设同一问题采样四条回答，奖励是 $(1,0,1,0)$。则：

\begin{equation*}
\bar r=(1+0+1+0)/4=0.5.
\end{equation*}

标准差：

\begin{equation*}
s_r=\sqrt{[(0.5)^2+(-0.5)^2+(0.5)^2+(-0.5)^2]/4}=0.5.
\end{equation*}

因此：

\begin{equation*}
(A_1,A_2,A_3,A_4)=(1,-1,1,-1).
\end{equation*}

优势为正的样本应提高相对发生概率，为负的样本应降低其相对发生概率，但还需考虑采样时旧模型分布、剪切限制和参考分布正则项。

### 与PPO剪切目标的联系

一个简化的按响应记录的代理目标可写为：

\begin{equation*}
J(\theta)=\frac1G\sum_{i=1}^G
\min\left(
\rho_i(\theta)A_i,\,
\operatorname{clip}(\rho_i(\theta),1-\epsilon,1+\epsilon)A_i
\right),
\end{equation*}

其中 $\rho_i(\theta)=\pi_\theta(o_i\mid q)/\pi_{\rm old}(o_i\mid q)$。这用于理解“组内比较 + 受控策略更新”的组合。

**需要特别区分：** 论文中的完整GRPO在具体实现里还涉及序列Token级比率、参考策略KL正则项、奖励定义和采样数据处理；上式是教学简化目标，不能声称与DeepSeek-R1训练代码完全一致。

### 奖励全相同会怎样

若所有候选奖励相同，$s_r=0$，公式不能直接计算。工程实现应显式定义退化组的处理方法，例如给这一组零优势并报告退化比例，而不是让标准差除零产生NaN。

这也提醒我们：如果所有候选都拿满分，组内比较不提供有用的相对信息；如果所有候选都得零分，同样不能凭空产生哪个候选更好的监督。

### 在真实任务中如何防止奖励投机

如果数学题只有最终答案可验证，模型可能以多种路径获得相同答案。若用“输出符合固定格式”作为过高权重的奖励，模型可能为了格式忽略实际问题。

对代码修复Agent，奖励应来自隔离环境中的**真实测试与修改范围检查**，并固定验收器，防止模型通过跳过测试或修改检查脚本刷高代理奖励。

运行：

~~~bash
python examples/model_training_architectures.py
~~~

该离线实验用四候选手算验证组内优势、退化组处理和简化剪切目标。它没有训练DeepSeek-R1，更没有复刻其真实训练规模。

### 参考资料

DeepSeek-AI等，*DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*，Nature（2025）：https://www.nature.com/articles/s41586-025-09422-z


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


## 大规模Agent强化学习为何采用异步采样：以GLM-5技术报告为例

### 为什么生成轨迹与训练更新会争抢资源

多步Agent训练不仅要计算梯度，还要调用工具、启动代码测试、等待环境反馈并保存轨迹。如果要求“采样一条完整轨迹后，训练一次，再采样下一条”，模型训练GPU可能大量时间都在等待外部工具。

GLM-5团队在2026年的公开技术报告中描述了**将轨迹生成与策略训练解耦的异步强化学习基础设施**，以及针对长时序Agent交互的训练设计。这是理解大规模Agent RL工程化的实际案例；但不能由此直接推断任何闭源模型的具体训练集、基础设施配置或未公开奖励函数。

### 异步采样造成旧策略问题

设工作者在采样时使用行为分布 $\mu(a\mid s)$，训练服务器当前要优化的分布为 $\pi_\theta(a\mid s)$。两者可能不同，因为训练器在采样期间已经更新参数。

对于满足覆盖条件的离散行动分布：

\begin{equation*}
\mathbb E_{A\sim\pi_\theta}[f(A)]
=\mathbb E_{A\sim\mu}
\left[\frac{\pi_\theta(A)}{\mu(A)}f(A)\right].
\end{equation*}

其中必须要求：若 $\pi_\theta(a)>0$，则 $\mu(a)>0$。这个恒等式解释了离策略重要性权重的数学来源，不代表在任意长轨迹上使用它都数值稳定。

例如旧策略下行动1概率为0.1，新策略为0.5，则该行动的比率为5。若旧轨迹非常少地包含行动1，少量带大权重样本可能带来较高方差。

对于多步轨迹，若简单将多个时间步的比率相乘，权重变化更可能剧烈。因此异步系统需要控制策略新旧差异、保留采样版本、监控比率分布，并决定过时轨迹何时不再用于更新。

### 一个可审计的数据记录

~~~text
task_id
environment_snapshot
rollout_policy_version
response_tokens / tool_actions
old_action_log_probabilities
observations / rewards
terminal_vs_timeout
reference_policy_version
verification_results
~~~

从多个工作者汇聚训练数据时，不应只收集最终回答和一个奖励分数。若缺少旧策略概率和实际环境结果，就很难正确实现需要重要性比率的训练目标。

### 工程题

假设有两条轨迹，一条在旧模型版本A生成，另一条在版本B生成，当前训练版本是C。讨论：如何记录每条轨迹的旧动作概率、如何限制高比率导致的方差、哪些失败轨迹不应当悄悄删除？此外，说明为什么异步训练吞吐提高不能单独证明最终任务成功率提高。

参考：GLM-5 Team，*GLM-5: from Vibe Coding to Agentic Engineering*（2026），https://arxiv.org/abs/2602.15763 。


## 多步强化学习实战：从一条Agent轨迹到可微训练目标

### 为什么只学习最终答案不够

假设Agent正在处理系统故障，可以选择直接猜测原因，也可以先执行一项检查。检查本身消耗成本，却可能提供能改变后续动作的新信息。

如果仅训练“最终回答正确”，模型未必能学会判断何时值得花成本调用工具。如果简单把调用工具次数当作奖励，模型又可能故意进行多次无用检查。

最自然的表示是一个带观测的多步决策过程：行动不仅影响眼前奖励，也影响后续可获得的信息。

### 建立一个可以枚举全部轨迹的两步环境

设故障原因是一个隐藏比特：

$H\in\{0,1\},\qquad P(H=0)=P(H=1)=1/2.$

第一步Agent有三种行动：

- Inspect：支付0.1单位成本，并观测到真实的隐藏比特；
- Guess0：不检查，直接猜测0并结束；
- Guess1：不检查，直接猜测1并结束。

如果先检查，第二步根据观测到的比特选择Guess0或Guess1；猜对得到1，猜错得到0。检查成本在轨迹总奖励中减去。

这不是文本游戏，而是一个简化的部分可观察决策问题：检查行动改变了可获得的信息，因此其收益需要由后续行动的条件期望衡量。

### 从全概率公式算出最优决策倾向

设首步策略为：

\begin{equation*}
\pi_\theta(I)=p_I,\quad
\pi_\theta(G_0)=p_0,\quad
\pi_\theta(G_1)=p_1,\quad
p_I+p_0+p_1=1.
\end{equation*}

在检查后，如果隐藏原因是 $h$，令第二步选择正确答案的概率为：

\begin{equation*}
c_h=P_\theta(A_1=h\mid H=h,\ A_0=I).
\end{equation*}

那么期望总收益：

\begin{equation*}
\boxed{
J(\theta)=
p_I\left[-0.1+\tfrac12c_0+\tfrac12c_1\right]
+\tfrac12p_0+\tfrac12p_1.
}
\end{equation*}

这是从三种首步行动分别取条件期望，再按首步行动概率求和得到的。

若首步三种行动等概率，检查后的两个选择也等概率，则：

\begin{equation*}
J=\tfrac13(0.4)+\tfrac13(0.5)+\tfrac13(0.5)
=\tfrac7{15}\approx0.4667.
\end{equation*}

若Agent学会在首步总是检查，并在第二步总是猜对，则期望收益趋近于 $1-0.1=0.9$。

对照之下，始终不检查并直接猜测任意固定值的期望收益只有0.5。检查是否值得，取决于信息价值与检查成本的比较；若检查成本超过0.5，最佳策略可能改变。

### 轨迹联合概率为何能分解

将环境隐藏比特、首步行动和可选的第二步行动组合为轨迹：

$\tau=(H,A_0,A_1).$

其中未执行Inspect时不包含 $A_1$。其联合概率为：

\begin{equation*}
P_\theta(\tau)
=P(H)\pi_\theta(A_0\mid s_0)
\begin{cases}
\pi_\theta(A_1\mid H),& A_0=I,\\
1,& A_0\ne I.
\end{cases}
\end{equation*}

因此，对数概率的策略相关部分：

\begin{equation*}
\log\pi_\theta(\tau)
=\log\pi_\theta(A_0\mid s_0)
+\mathbf1[A_0=I]\log\pi_\theta(A_1\mid H).
\end{equation*}

这说明多步Agent并不是只有一次Softmax：每一步实际选择的行动都贡献策略对数概率。为了训练，必须记录实际行动、当时的策略版本和外部观测，而不能仅保留最终回答。

### 由轨迹直接推导策略梯度

设轨迹总奖励为 $R(\tau)$，环境随机变量的采样分布不依赖模型参数。则：

\begin{equation*}
J(\theta)=\sum_\tau P_\theta(\tau)R(\tau).
\end{equation*}

使用对数导数恒等式：

\begin{equation*}
\nabla_\theta J
=\sum_\tau
P_\theta(\tau)R(\tau)
\nabla_\theta\log\pi_\theta(\tau).
\end{equation*}

可以加上不依赖于当次行动的基线 $b$：

\begin{equation*}
\boxed{
\nabla_\theta J
=
\mathbb E_{\tau\sim P_\theta}
[(R(\tau)-b)\nabla_\theta\log\pi_\theta(\tau)].
}
\end{equation*}

该表达式解释REINFORCE。若基线依赖当前状态而不依赖当前行动，可逐时间步构造状态基线；但不能随意让基线依赖采样行动而仍假定梯度无偏。

### 枚举全部轨迹验证梯度，而不只看损失下降

本环境有8种带隐藏状态的结果：两个隐藏值分别对应两种直接猜测与两种检查后的猜测。

程序逐一构造每条轨迹的概率和奖励，计算两种梯度：

1. 直接对 $J(\theta)=\sum_\tau P_\theta(\tau)R(\tau)$ 求自动微分；
2. 构造对数导数形式的期望并求导。

正确实现时，两种梯度应在数值精度内一致。然后从策略实际采样6000条轨迹，使用样本得分函数估计梯度，比较它与精确梯度之间的误差。

这比“训练损失下降”更严格地检查了：采样轨迹是否与策略对应、哪一步产生奖励、隐藏变量是否被正确边缘化，以及基线是否错误地进入了梯度。

\begin{figure}[htbp]
\centering
\resizebox{0.94\linewidth}{!}{\input{figures/rl_two_step_rollout.tex}}
\caption{两步Agent：检查带来信息收益，也产生即时成本；最终奖励取决于完整执行轨迹}
\end{figure}

## 将PPO从单个动作扩展到多步轨迹

### 为什么需要记录旧概率

假设轨迹是由旧策略 $\mu$ 生成，而当前训练策略是 $\pi_\theta$。只要旧策略对目标策略支持的结果有正概率：

\begin{equation*}
\mathbb E_{\tau\sim P_\theta}[R(\tau)]
=
\mathbb E_{\tau\sim P_\mu}
\left[
\frac{P_\theta(\tau)}{P_\mu(\tau)}R(\tau)
\right].
\end{equation*}

在环境转移分布不依赖 $\theta$ 且行为与目标使用相同环境机制时，轨迹比率可以化简为各实际行动概率比率的乘积：

\begin{equation*}
\frac{P_\theta(\tau)}{P_\mu(\tau)}
=
\prod_{t\in\tau}
\frac{\pi_\theta(A_t\mid S_t)}
{\mu(A_t\mid S_t)}.
\end{equation*}

这是多步离策略校正的基础。但轨迹越长，多个比率相乘可能造成很大的数值波动，导致采样方差剧增。因此真实长轨迹训练往往需要比率截断、逐步优势、数据新鲜度限制和其他稳定性措施。

### 一个教学用的轨迹级PPO剪切项

设旧轨迹概率为 $P_{\rm old}(\tau)$，新旧比率：

$\rho_\tau(\theta)=P_\theta(\tau)/P_{\rm old}(\tau).$

给定一个固定优势值 $A_\tau$，可定义教学用剪切代理目标：

\begin{equation*}
L_{\rm toy}(\theta)
=
\sum_\tau P_{\rm old}(\tau)
\min\left(
\rho_\tau A_\tau,
\operatorname{clip}(\rho_\tau,1-\epsilon,1+\epsilon)A_\tau
\right).
\end{equation*}

这里的计算是**轨迹级**而不是标准PPO常见的逐时间步剪切目标。它用来解释比率与剪切的关系，不能误写成完整的RLHF/PPO算法复现。

真实Agent训练需明确旧策略概率记录、奖励延迟、多步归因、GAE、参考策略KL、行动遮罩和工具调用的时间尺度。

### 强化学习与规划算法为什么需要共享状态契约

规划章节用状态空间和合法行动集合表示执行；强化学习章节又用状态与行动分布表示选择。如果二者使用的状态定义不一致，训练得到的策略可能提出执行器不能识别或不允许的行动。

例如训练时将“提交补丁”视为一个总能成功的动作，生产环境却要求审批、代码版本检查和测试，那么训练回报中就遗漏了实际失败概率及审批成本。

解决方法是让训练环境与生产工具网关共享明确的动作类型与约束，并在训练回放中记录被拒绝、超时和实际结果。不能把Agent自己生成的“测试通过”视为真实环境奖励。

### 跟着代码复现整条训练链

~~~bash
python examples/multistep_rl_rollout_lab.py
~~~

实验包括：

1. 构造隐藏原因与三种首步行动；
2. 枚举八种轨迹，计算期望总收益；
3. 用自动微分和Score Function独立求导并验证一致；
4. 从环境真实采样6000条轨迹，估计策略梯度；
5. 进行一次PPO式剪切更新，比较真实期望收益；
6. 检验离策略重要性加权恒等式；
7. 用精确期望目标训练策略，让Agent倾向于先检查再正确处理。

这是有限玩具MDP，不训练语言模型权重。实验的价值在于每个概率、成本、梯度和观测都可以逐项核算。

### 练习与详解

**题一：** 若检查后总能猜对，检查成本从0.1涨到0.6，是否还应总是先检查？

**解：** 检查后期望收益为 $1-0.6=0.4$，直接固定猜测的期望收益为0.5，因此纯收益目标下不应总检查。

**题二：** 为什么轨迹级比率不是简单的最后一个动作概率比率？

**解：** 整条轨迹的概率是各阶段条件行动概率与环境转移概率的乘积。若首步Inspect发生概率改变，即使第二步条件概率完全不变，轨迹概率也会改变。

**题三：** 旧策略对某条目标策略可能产生的轨迹赋予概率零，会发生什么？

**解：** 重要性比率分母为零，不能使用该恒等式从旧样本无偏估计目标策略在这条轨迹上的贡献。这是支持集覆盖条件。

**题四：** 模型是否只要学会“先检查”就一定获得0.9期望收益？

**解：** 不一定。0.9的上界还要求检查后的行动能正确利用观测；如果第二步仍随机猜测，Inspect路径的期望收益只有0.4。


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

\begin{equation*}
d_\theta=
\log(0.6/0.4)-\log(0.2/0.8)
=\log1.5-\log0.25=\log6.
\end{equation*}

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
