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
