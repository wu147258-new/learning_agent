# Transformer深度解析：注意力、因果建模与高效推理

## 从序列概率问题出发

给定句子“系统正在读取数据库”，当模型读到“系统正在读取”之后，它需要为词表中可能出现的下一个词分配条件概率。一般写为：

$$p_\theta(X_{t+1}=v\mid X_1=x_1,\ldots,X_t=x_t).$$

概率乘法公式说明：

$$p_\theta(x_1,\ldots,x_T)=\prod_{t=1}^Tp_\theta(x_t\mid x_{<t}).$$

这是任何联合概率分布都满足的基本分解，而不属于Transformer独有的定理。Transformer Decoder是一种计算这些条件概率的参数化神经网络。训练时已经知道完整序列，可以同时计算多个位置的预测损失；生成时仍必须逐Token提供已产生的前缀。

### 训练与生成的差别

对Token序列 $(1,4,7,9,2)$，自回归训练可以使用：

~~~text
输入: 1, 4, 7, 9
标签: 4, 7, 9, 2
~~~

输入位置 $t$ 预测真实位置 $t+1$ 的Token。若输入和标签错误地对齐为同一位置，训练任务可能退化成复制输入。

### 为什么需要跨位置交互

一个Token的解释可能依赖许多位置之前的变量定义、否定词、函数声明和工具调用结果。逐位置的线性层不能直接读取其他位置的表示，因此需要跨位置的信息交互机制。注意力将当前查询向量与其他位置的键向量比较，得到一组权重，再对相应的值向量求加权和。

## 从一个二维例子推导缩放点积注意力

### 先计算相似度

设查询 $q=(1,0)$，两个键为 $k_1=(1,0)$ 与 $k_2=(0,1)$；值向量为 $v_1=(2,0)$ 与 $v_2=(0,4)$。

键维度 $d_k=2$，得到缩放得分：

$$s_1=q^\mathsf Tk_1/\sqrt2=1/\sqrt2,\qquad
s_2=q^\mathsf Tk_2/\sqrt2=0.$$

### 再构造概率权重

对两个得分进行Softmax：

$$
\alpha_i=\frac{e^{s_i}}{e^{s_1}+e^{s_2}},
\qquad \alpha_1+\alpha_2=1.
$$

数值上 $\alpha_1\approx0.6698$，$\alpha_2\approx0.3302$。输出：

$$o=\alpha_1v_1+\alpha_2v_2\approx(1.3395,1.3209).$$

注意力没有从两个值中“选一个”；它求的是以归一化权重加权的向量和。计算中的归一化权重也不能直接解释成某个自然语言结论的因果证据。

### 推广到矩阵

设输入表示 $X\in\mathbb R^{T\times d}$。三个线性映射：

$$Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V,$$

其中 $Q,K\in\mathbb R^{T\times d_k}$，$V\in\mathbb R^{T\times d_v}$。

查询矩阵与键矩阵相乘得到：

$$QK^\mathsf T\in\mathbb R^{T\times T}.$$

其第 $(i,j)$ 个元素是第 $i$ 个查询与第 $j$ 个键的内积。设掩码矩阵为 $M$，注意力结果为：

$$
\boxed{
O=\operatorname{softmax}_{\mathrm{row}}
\left(QK^\mathsf T/\sqrt{d_k}+M\right)V
\in\mathbb R^{T\times d_v}.
}
$$

\begin{figure}[htbp]
\centering
\resizebox{\linewidth}{!}{\input{figures/transformer_attention.tex}}
\caption{单头注意力的张量流向：查询、键、值、因果掩码和加权输出}
\end{figure}

## 为什么缩放因子是根号维度

### 从方差推导动机

假设查询与键的各维坐标独立、均值为0、方差为1。令：

$$Z=q^\mathsf Tk=\sum_{i=1}^{d_k}q_ik_i.$$

在这些简化假设下：

$$\mathbb E[Z]=0,\qquad \operatorname{Var}(Z)=d_k.$$

所以内积的标准差随着维度增长，约为 $\sqrt{d_k}$。当维度很大时，如果直接把未经缩放的得分送入Softmax，输出可能过于尖锐，影响梯度分布。

通过除以 $\sqrt{d_k}$，上述简化假设下的得分方差回到1附近。这是一项统计动机，不意味着真实训练后的所有查询和键都满足独立、单位方差条件。

### 不要把缩放理解为概率除法

归一化是在Softmax步骤完成的。除以 $\sqrt{d_k}$ 只改变得分数值范围，并不会使得分总和变成1。给定两个相同的得分，Softmax仍返回 $(0.5,0.5)$，无论缩放因子是多少。

## 因果掩码究竟屏蔽了什么

### 未来Token不能泄漏到历史位置

如果模型在位置 $t$ 预测位置 $t+1$ 的Token，就不能在计算位置 $t$ 的表示时读取真实的未来Token。

令：

$$
M_{ij}=\begin{cases}
0,&j\le i,\\
-\infty,&j>i.
\end{cases}
$$

这样对于未来位置 $j>i$：

$$e^{s_{ij}+M_{ij}}=0,\qquad A_{ij}=0.$$

**把未来位置的得分设置为0，不等于掩码。** 若两个得分都是0，Softmax会给出 $(0.5,0.5)$，未来仍有一半注意力权重。

### 如何用反事实输入测试因果性

构造两个序列，它们的前3个Token完全相同，后缀不同。对处于评估模式的因果Decoder，应得到：

$$x_{1:3}=x'_{1:3}
\Longrightarrow f_\theta(x)_{1:3}=f_\theta(x')_{1:3}.$$

这项测试不仅要检查掩码矩阵，还应检验整个前向传播。错误的序列移位或跨时间维度的归一化也可能引入信息泄漏。

### Padding掩码与因果掩码不同

批次中多个序列长度不同时，填充位置通常不能参与有效监督，在注意力中也不应被误认为真实内容。因果掩码限制未来位置，Padding掩码限制填充位置，两者要分别处理。

如果一个查询位置的全部键都被屏蔽，直接对全部为负无穷的分数执行Softmax可能产生NaN。实现必须明确定义此类无效位置的行为。

## 多头注意力的数学结构

### 多个头如何计算

对第 $h$ 个头：

$$Q_h=XW_Q^{(h)},\ K_h=XW_K^{(h)},\ V_h=XW_V^{(h)}.$$

计算：

$$O_h=\operatorname{Attention}(Q_h,K_h,V_h).$$

将所有头在特征维拼接：

$$
H=\operatorname{Concat}(O_1,\ldots,O_{n_h}),
\qquad \operatorname{MHA}(X)=HW_O.
$$

多头结构允许不同投影关注不同关系，但不能保证每个头必定学到一种人类能命名的语言规则。

### 逐项检查张量维度

若批量大小 $B=2$、序列长度 $T=5$、隐藏维度 $d=32$、头数 $H=4$：

| 对象 | 张量形状 |
|---|---|
| 输入 | $(2,5,32)$ |
| 合并QKV投影 | $(2,5,96)$ |
| 拆分Q/K/V | 各为$(2,5,32)$ |
| 分头后Q/K/V | 各为$(2,4,5,8)$ |
| 得分矩阵 | $(2,4,5,5)$ |
| 分头结果 | $(2,4,5,8)$ |
| 拼接结果 | $(2,5,32)$ |

代码中的维度重排需要谨慎处理。改变维度顺序和改变内存布局是不同操作；错误转置可能仍产生合法张量，却破坏头与位置的对应关系。

## Decoder Block中的残差与归一化

### Pre-Norm Decoder

一种常见的Pre-Norm结构为：

$$U=X+\operatorname{MHA}(\operatorname{LN}(X)),$$

$$Y=U+\operatorname{FFN}(\operatorname{LN}(U)).$$

残差连接使输入可以直接传递到后续层，并为梯度提供额外传播路径。它有助于优化，却不能保证任意深度和超参数都收敛。

### LayerNorm的计算

对Token特征向量 $x\in\mathbb R^d$：

$$
\mu=\frac1d\sum_i x_i,\quad
\sigma^2=\frac1d\sum_i(x_i-\mu)^2,
$$

$$
\operatorname{LN}(x)_i=
\gamma_i\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta_i.
$$

LN通常对单个Token的特征维度计算统计量；它与依赖批次统计量的BatchNorm不同。

### 前馈网络

典型前馈模块：

$$\operatorname{FFN}(x)=W_2\phi(W_1x+b_1)+b_2.$$

它先映射到更宽的隐藏空间、施加非线性，再映射回原维度。中间宽度可以取 $4d$，但这不是数学上的硬性要求；许多模型采用门控前馈网络。

## 从Dense FFN到MoE：Qwen3、DeepSeek-V3与gpt-oss的专家路由

### Dense模型与稀疏激活模型的区别

传统Dense Transformer的前馈网络对每个Token使用同一组前馈参数：

$y=\operatorname{FFN}(x).$

MoE（Mixture of Experts）把前馈部分替换为 $E$ 个专家网络 $f_1,\ldots,f_E$ 和一个路由器。对Token表示 $x$，路由器计算专家得分 $z=Wx$，选择得分最高的 $k$ 个专家：

$
\mathcal K(x)=\operatorname{TopK}(z,k).
$

将被选择专家的得分重新做Softmax归一化：

$
\alpha_i(x)=
\frac{e^{z_i}}{\sum_{j\in\mathcal K(x)}e^{z_j}},
\quad i\in\mathcal K(x).
$

输出：

$
\boxed{y=\sum_{i\in\mathcal K(x)}\alpha_i(x)f_i(x).}
$

并非每个模型都使用完全相同的路由归一化，也可能包含共享专家、路由偏置、专家容量限制及负载均衡机制；以上是最基础的Top-k MoE教学定义。

### “总参数”与“单Token激活参数”不同

假设专家数 $E=8$，每个专家有 $P_e=10^6$ 个参数，路由时选择 $k=2$ 个专家。仅考虑专家部分：

$P_{\rm experts,total}=8\times10^6,$

$P_{\rm experts,active/token}\approx2\times10^6.$

这样模型可以保存更多不同的专家参数，而单个Token的前馈计算只调用部分专家。

**不能将单Token激活参数简单当成推理显存总需求。** 推理服务通常还需存储或加载模型中的全部专家权重，除非另有换入换出或分布式专家部署机制。

### 公开模型实例与结构差异

| 模型实例 | 公开结构事实 | 教学重点 |
|---|---|---|
| Meta Llama 3 405B | Dense Transformer | 所有Token运行同一组Dense层参数 |
| Qwen3 MoE系列（技术报告） | 披露细粒度专家与稀疏路由；所述MoE设计采用128个专家、每Token激活8个 | Top-k路由与专家负载 |
| DeepSeek-V3 | 671B总参数、每Token约37B激活；MLA与DeepSeekMoE | 大规模总参数与推理计算之间的差异 |
| gpt-oss-120b | 128个专家、每Token激活4个 | 选择路由与活跃参数数量 |
| gpt-oss-20b | 32个专家、每Token激活4个 | 同系列不同规模下的专家组织 |

这里的配置来自相应公开技术报告或模型卡，并不表示这些模型彼此采用同一套具体工程实现。上述条目也不构成对未公开闭源模型结构的猜测。

### 负载不均会造成什么问题

假设8个专家中的大部分Token都选择同一专家，而其余专家很少被调用。在专家并行部署中，热门专家可能成为计算与通信瓶颈，其他GPU空闲。

可以用一批Token分配到每个专家的数量 $n_e$ 观察负载：

$
p_e=\frac{n_e}{\sum_{j=1}^{E}n_j},
\quad
H=-\sum_e p_e\log p_e.
$

熵 $H$ 可以描述分配是否集中，但并不自动等于训练中实际使用的负载均衡损失。不同模型可能采用辅助负载损失、无辅助损失的动态偏置等具体策略。DeepSeek-V3报告披露了不额外引入辅助损失的负载均衡方法；这些细节需要和一般MoE理论明确区分。

### 为什么MoE不是无条件更好

MoE带来参数容量与活跃计算的解耦，但训练和推理可能需要专家间路由通信、负载均衡、专家参数分布与动态批处理。小规模任务未必能从稀疏路由获益。

训练时Top-k选择本身还是离散的；常见实现通过对已选择专家的连续权重进行反向传播，配合特定路由训练机制。不能因为普通前馈网络可微，就认为离散专家选择在所有边界上都可微。


## 从负对数似然推导梯度

### 交叉熵从哪里来

位置 $t$ 的词表得分为 $z\in\mathbb R^V$，真实Token编号为 $y$：

$$
p_j=\frac{e^{z_j}}{\sum_ke^{z_k}},
\qquad \ell(z,y)=-\log p_y.
$$

展开：

$$\ell(z,y)=-z_y+\log\sum_ke^{z_k}.$$

求导：

$$\boxed{\frac{\partial\ell}{\partial z_j}=p_j-\mathbf1[j=y].}$$

这个梯度经链式法则传到注意力、前馈网络和嵌入参数。自动微分只负责求导，不负责检查训练目标和数据是否定义正确。

### 数值稳定的log-sum-exp

直接计算 $e^{1000}$ 容易浮点溢出。令 $m=\max_j z_j$：

$$
\log\sum_je^{z_j}=m+\log\sum_je^{z_j-m}.
$$

这两项严格等价，但右式的指数输入均不大于0，数值更稳定。训练程序应使用实现了数值稳定性的交叉熵算子，而不是自行串联未经保护的exp、sum与log。

## 从多头注意力走向GQA：Llama、Qwen与gpt-oss的公开设计

### 同样的查询头数，为什么不必配置同样多的KV头

标准多头注意力（MHA）对每个查询头分别计算Key与Value。若查询头数为 $H_q$，每头维度为 $d_h$，通常有 $H_{\rm kv}=H_q$。对长文本解码，历史K/V都需缓存，成为显存瓶颈。

多查询注意力（MQA）让所有查询头共享一组K/V；分组查询注意力（GQA）取折中方案，让 $g=H_q/H_{\rm kv}$ 个查询头共享一组K/V，通常要求 $H_q$ 能被 $H_{\rm kv}$ 整除。

对于查询头 $h$，可令它使用KV组：

$
\operatorname{group}(h)=\left\lfloor h/g\right\rfloor.
$

单头输出仍为缩放点积注意力，只是多个查询头复用同一组Key和Value：

$
O_h=\operatorname{softmax}
\left(\frac{Q_hK_{\operatorname{group}(h)}^{\mathsf T}}{\sqrt{d_h}}+M\right)
V_{\operatorname{group}(h)}.
$

**减少KV头不等于减少查询头。** 它主要降低KV投影与缓存规模，同时可能影响模型表达与质量，需要在特定模型上评测。

### 用缓存公式手算GQA的收益

忽略额外元数据时，KV缓存：

$
M_{\mathrm{KV}}=2BLTH_{\rm kv}d_hb.
$

假设批量 $B=1$，层数 $L=32$，缓存长度 $T=8192$，查询头数 $H_q=32$，每头维度 $d_h=128$，KV元素占2字节。

MHA的KV头数为32，缓存约为 $2\times1\times32\times8192\times32\times128\times2=4$ GiB。

若GQA使用8个KV头，则缓存约为1 GiB；在这个简化模型里节约75%。这不代表整个模型显存下降75%，因为权重、临时激活和运行时分配并未计入。

### 公开模型实例：Llama 3、Qwen3与gpt-oss

Meta公开的Llama 3技术报告将大规模模型归为Dense Transformer设计。Qwen3技术报告说明其Dense模型沿用GQA、RoPE、SwiGLU和Pre-Norm RMSNorm，并加入QK-Norm；Qwen3同时提供MoE变体。

OpenAI公开权重的gpt-oss模型卡给出具体的可复核配置：注意力层有64个Query头、8个KV头，采用GQA；同时使用局部窗口与全局稠密注意力交替出现的模式。这些都是**特定已公开权重模型**的设计，不能据此外推任何未披露的闭源GPT模型内部结构。

| 结构 | Query头 | KV头 | 缓存及权衡 |
|---|---:|---:|---|
| MHA示意 | 32 | 32 | KV缓存最多，表达最灵活 |
| GQA示意 | 32 | 8 | KV缓存约为MHA的四分之一 |
| MQA示意 | 32 | 1 | KV缓存更小，但共享程度更高 |
| gpt-oss公开配置 | 64 | 8 | 以官方公开模型卡为准 |

\begin{figure}[htbp]
\centering
\resizebox{0.93\linewidth}{!}{\input{figures/model_attention_patterns.tex}}
\caption{MHA、GQA与低秩KV表示的结构比较；MLA示意仅用于理解压缩思想，不等同于完整DeepSeek-V3实现}
\end{figure}

### QK-Norm为什么不能与LayerNorm混为一谈

QK-Norm在注意力打分前对查询和键施加适当归一化；Pre-Norm通常指Transformer块在注意力/前馈模块前对残差流进行归一化。二者处于不同位置，不应将模型结构中的所有“Norm”直接视为同一操作。

对于加入QK-Norm的版本，推导注意力分数时需将经过归一化的 $\widetilde Q,\widetilde K$ 代入：

$
S=\widetilde Q\widetilde K^{\mathsf T}/\sqrt{d_h}+M.
$

实际归一化的具体定义、是否带可学习权重和作用维度须以模型实现为准。


## KV缓存为什么能减少重复计算

### 不使用缓存时会重复求什么

自回归解码生成第 $t+1$ 个Token时，直接重新运行整个前缀，会重新计算许多历史位置的Key和Value。而因果模型的历史位置不会因新的未来Token发生变化。

可以保存历史K/V：

$$K_{1:t}=\begin{bmatrix}k_1\\\vdots\\k_t\end{bmatrix},
\qquad V_{1:t}=\begin{bmatrix}v_1\\\vdots\\v_t\end{bmatrix}.$$

加入新Token以后，仅需追加新位置的 $k_{t+1}$ 和 $v_{t+1}$，并计算新查询对全部已缓存键的注意力。

### 缓存占用的理论估算

若层数为 $L$、批量为 $B$、序列长度为 $T$、KV头数为 $H_{\rm kv}$、每头维度为 $d_h$、每元素占 $b$ 字节，则：

$$\boxed{M_{\rm KV}\approx 2BLTH_{\rm kv}d_hb.}$$

因子2来自Key和Value。实际峰值显存还包括权重、激活、分块存储、内存对齐和框架开销，不应把上述公式当作实际峰值的精确测量。

例如 $B=1,L=12,T=2048,H_{\rm kv}=8,d_h=64,b=2$ 时，缓存约为48MiB。Grouped-Query Attention通过减少KV头数，可以改变这一项内存需求。

### 缓存必须匹配哪些条件

KV缓存必须关联模型权重版本、对应层、前缀内容、位置编码和会话状态。不能把其他任务或模型权重的缓存无条件复用。

## 从GQA到DeepSeek-V3的MLA：压缩KV表示而不是只减少头数

### 为什么低秩投影可能节省缓存

GQA通过共享KV头降低缓存维度。另一条路线是先把历史Token的表示压缩成较小的潜在向量，再为注意力计算恢复需要的投影。

假设原始维度为 $d$，潜在缓存维度为 $r$，且 $r\ll d$：

$
c_t=W_{\mathrm{down}}x_t\in\mathbb R^r,
\qquad
\widetilde k_t=W_{\mathrm{up},k}c_t,\quad
\widetilde v_t=W_{\mathrm{up},v}c_t.
$

如果只缓存 $c_t$，每个历史位置所需存储可以从若干完整KV向量降到 $r$ 个元素；但是计算过程中是否需要额外的解码、矩阵吸收和位置编码处理，要取决于真实实现。

DeepSeek-V3技术报告披露使用**Multi-head Latent Attention（MLA）**以及DeepSeekMoE。其MLA包含对位置编码与内容表示的专门处理，不能把上述三行通用低秩分解误称为DeepSeek-V3源码复现。本节实验只验证低秩缓存概念和内存计数。

### 低秩不代表无损

若 $W_{\mathrm{down}}\in\mathbb R^{r\times d}$，且 $r<d$，线性映射存在非零零空间，通常无法从潜在向量精确恢复任意 $d$ 维输入。

因此“缓存更小”和“信息完全不丢失”不能同时由低秩维度约束直接推出。模型会在训练中学习有利于任务的投影，但是否保持质量必须评测，不能只看缓存大小。

### 可运行的数值实验

执行：

~~~bash
python examples/model_architecture_patterns.py
~~~

程序会验证GQA共享KV头的张量形状与结果等价性、稀疏专家路由，以及低秩KV表示相对于完整KV的理论元素数量。它不用下载Qwen、Llama或DeepSeek的权重，所以验证的是**架构算子原理**，并不宣称复现大型模型。

### 公开技术资料

- Meta, *The Llama 3 Herd of Models*：https://arxiv.org/abs/2407.21783
- Qwen Team, *Qwen3 Technical Report*：https://arxiv.org/abs/2505.09388
- DeepSeek, *DeepSeek-V3 Technical Report*：https://arxiv.org/abs/2412.19437
- OpenAI, *gpt-oss Model Card*：https://deploymentsafety.openai.com/gpt-oss/architecture


## RoPE如何将位置差放入向量内积

### 二维旋转

定义：

$$R(m\omega)=
\begin{pmatrix}
\cos(m\omega)&-\sin(m\omega)\\
\sin(m\omega)&\cos(m\omega)
\end{pmatrix}.$$

若位置 $m,n$ 对查询与键分别旋转：

$$q'_m=R(m\omega)q,\qquad k'_n=R(n\omega)k,$$

则：

$$
(q'_m)^\mathsf Tk'_n
=q^\mathsf TR(m\omega)^\mathsf TR(n\omega)k
=q^\mathsf TR((n-m)\omega)k.
$$

因此内积包含相对位置差 $n-m$。多维RoPE将特征分成多个二维坐标对并使用不同频率；频率定义、坐标布局和长度扩展方案必须与模型实现一致。

## 用十几行PyTorch实现因果注意力

### 从数学式到程序

对输入矩阵Q、K和V，实现单头注意力：

~~~python
import math
import torch

def causal_attention(q, k, v):
    # q,k: (T,d_k), v: (T,d_v)
    scores = q @ k.T / math.sqrt(q.shape[-1])
    T = q.shape[0]
    visible = torch.ones(
        T, T, device=q.device, dtype=torch.bool
    ).tril()
    scores = scores.masked_fill(
        ~visible, float("-inf")
    )
    weights = torch.softmax(scores, dim=-1)
    return weights @ v, weights
~~~

逐行解释：第一行矩阵乘法计算所有位置对之间的查询—键相似度；tril构造下三角可见性；masked_fill将未来位置设为负无穷；Softmax在最后一个维度归一化；最终矩阵乘法实现加权求和。

这个版本仅用于同长度、自回归、无Padding的学习实验。将它直接用于批处理生产推理还需要处理Batch维、多个注意力头、Padding、设备精度和缓存。

### 用最小输入验证形状与掩码

~~~python
q = torch.tensor([[1., 0.], [0., 1.]])
k = q.clone()
v = torch.tensor([[2., 0.], [0., 4.]])
output, weights = causal_attention(q, k, v)

assert output.shape == (2, 2)
assert weights.shape == (2, 2)
assert weights[0, 1].item() == 0
~~~

第一位置不能读取第二个位置，所以第一行的第二个权重必须严格为零。注意这并不是说第二个位置不能读取第一个位置；因果可见性是有方向的。

### 下一Token标签怎样移位

~~~python
from torch.nn import functional as F

# token_ids 形状为(B,T)
model_input = token_ids[:, :-1]
target = token_ids[:, 1:]
logits = model(model_input)  # (B,T-1,V)
loss = F.cross_entropy(
    logits.reshape(-1, logits.shape[-1]),
    target.reshape(-1)
)
~~~

如果输入包含Padding或指令部分，必须进一步构造损失掩码；否则会把本应忽略的位置也纳入监督。仅会调用交叉熵函数，不等于已经正确设置训练任务。


## PyTorch实验与验收

### 可执行的验证

运行：

~~~bash
python examples/transformer_attention_lab.py
python examples/tiny_transformer.py
~~~

第一个实验验证手算注意力、掩码、Softmax稳定性、RoPE恒等式及缓存解码与整段因果注意力的一致性。第二个实验训练小型Decoder并验证梯度和前缀不变性。

这些实验分别用于证明数学公式计算正确、实现行为正确，以及固定合成数据上的训练目标可下降。**训练损失下降不代表对未见任务的泛化能力，更不能代替真实Agent任务成功率。**

## 习题与解答

### 练习：注意力的形状

设 $B=3,T=128,d=512,H=8$，每头维度64。求分头Q/K/V与分数矩阵的形状。

**解：** Q/K/V各为 $(3,8,128,64)$，分数矩阵为 $(3,8,128,128)$，拼接后的输出为 $(3,128,512)$。

### 练习：掩码的数值意义

位置1对位置1和位置2的未掩码分数都是0。把位置2的分数设置成0能否防止未来泄漏？

**解：** 不能。Softmax$(0,0)=(0.5,0.5)$。要把位置2排除，应将其分数设为 $-\infty$，得到 $(1,0)$。

### 练习：缓存显存

使用上述 $B=1,L=12,T=2048,H_{\rm kv}=8,d_h=64,b=2$ 的参数。

**解：**

$$M=2\times1\times12\times2048\times8\times64\times2
=50\,331\,648\text{字节}=48\text{MiB}.$$

实际显存还包括模型权重等其他开销。

### 编程练习

1. 删除因果掩码，检查后缀变化是否影响早期位置的输出。
2. 故意把KV缓存中的位置下标错移1，构造缓存解码与完整计算不一致的反例。
3. 用有限差分检验一个注意力得分对输出的梯度。
4. 将序列长度从512增加到2048，比较注意力得分矩阵大小，并记录真实运行时间。
5. 为一组未见合成序列计算验证集交叉熵，区分训练拟合与数据外推。

## 参考文献

- Vaswani et al. (2017), *Attention Is All You Need*, arXiv:1706.03762.
- Ba, Kiros and Hinton (2016), *Layer Normalization*, arXiv:1607.06450.
- Su et al. (2021), *RoFormer: Enhanced Transformer with Rotary Position Embedding*, arXiv:2104.09864.
- Shazeer (2019), *Fast Transformer Decoding: One Write-Head is All You Need*, arXiv:1911.02150.
