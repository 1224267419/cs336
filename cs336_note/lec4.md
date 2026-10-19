# Lecture 4 课程报告：从长上下文注意力替代到 MoE 稀疏专家架构

## 0. 资料来源与改写说明

本报告是对你上传的 Lecture 4 原始课程报告 `lec4.md` 的系统性改写，目标是把原报告从“课程内容整理”进一步优化为“适合学习、复习和长期保存的中文讲义”。本次改写保留原报告的课程主线：先讨论长上下文下 attention 的成本，再进入 linear attention、Mamba-2、Gated Delta Net 和 DSA，随后转入 Mixture of Experts，并继续讨论 MoE 的 routing、训练、系统、稳定性、微调、upcycling 与 DeepSeek MoE 演化。

本次对话中实际提供的是原始报告文件，而不是字幕文件和 PDF 文件本体。因此，报告中出现的 `【字幕†source】` 与 `【PDF†source】` 是沿用原报告已经给出的课程资料引用标记；我不会把新增解释伪装成课程原文。凡是为了帮助理解而加入的内容，会用 **补充说明**、**延伸理解**、**工程直觉** 或 **学习提示** 标出。

同时，我核对了官方课程页和 GitHub 仓库。官方 CS336 课程页显示该课程是 Stanford Spring 2026 的 “Language Modeling from Scratch”，课程目标是带学生走完整个语言模型构建流程，包括数据收集与清洗、Transformer 构建、模型训练和部署前评估；课程表中 Lecture 4 是 “Attention alternatives and mixture of experts [Tatsu]”。 GitHub 仓库目录中列出了 `lecture_04.pdf`，同时 README 说明可执行讲义命名为 `lecture_XX.py`，非可执行讲义命名为 `lecture_XX.pdf`；在目录列表中 Lecture 4 对应的是 PDF，而不是 Python 文件。

因此，本报告的“代码解析”部分不会假装存在 `lecture_04.py`。它会改为解释 PDF 中的公式、算法流程和伪代码等价实现，帮助你把数学表达转化成可以写成程序的思路。

------

## 1. 学习导读

### 1.1 本讲要解决的核心问题

本讲围绕一个非常现实的大模型问题展开：**标准 Transformer 在规模继续扩大时，哪些模块会成为效率瓶颈，我们能否用更稀疏、更状态化的方式替代它们？**

具体来说，Lecture 4 处理两条主线：

第一条主线是 **attention alternatives**。当上下文长度从几千 token 增长到几十万、百万甚至更长时，标准 softmax attention 的 all-to-all 交互会带来二次复杂度。课程因此讨论 local/global attention、FlashAttention、linear attention、Mamba-2、Gated Delta Net 和 DSA 等替代方案。【字幕†source】

第二条主线是 **Mixture of Experts**。当模型需要更大容量时，传统 dense FFN 会让参数量和计算量一起上升。MoE 的想法是：保留大量专家参数，但每个 token 只激活少数专家，从而让“总参数量”增加，而“每次实际计算量”不等比例增加。【字幕†source】

### 1.2 本讲在 CS336 课程中的位置

Lecture 4 位于 Lecture 3 “Architectures, hyperparameters” 之后、Lecture 5 “GPUs, TPUs” 之前。这个位置很重要：它把前一讲的 Transformer 架构知识推进到“现代架构改造”，也为后一讲的硬件与系统效率问题做铺垫。官方课程表也显示，Lecture 4 的主题正是 “Attention alternatives and mixture of experts [Tatsu]”。

换句话说，本讲不是单纯介绍某一个模型结构，而是在回答：

> 当标准 Transformer 的 attention block 和 FFN block 开始变贵时，现代大模型架构怎样重新设计这两个模块？

### 1.3 本讲知识结构

| 模块              | 解决的问题                                          | 核心概念                                          | 与后续内容的关系                                         |
| ----------------- | --------------------------------------------------- | ------------------------------------------------- | -------------------------------------------------------- |
| 长上下文成本      | 为什么 context window 变长后 attention 会成为瓶颈   | $QK^\top$、all-to-all、$O(n^2)$                   | 引出 local attention、FlashAttention 和 linear attention |
| 线性注意力        | 如何把 attention 的序列长度依赖从二次降低到近似线性 | 结合律、去掉 softmax、$Q(K^\top V)$               | 引出 recurrent form 和状态压缩                           |
| recurrent duality | 如何做到训练并行、推理递归                          | $S_t$ 状态、dense/serial duality                  | 连接 RNN、state space model 和 Mamba 系列                |
| Mamba-2 / GDN     | 如何增强线性注意力的表达能力                        | gate、遗忘、写入、擦除                            | 解释现代 hybrid architecture 为什么有效                  |
| DSA               | 不线性化 attention，而是只看少数重要 token          | indexer、Top-K、稀疏 full attention               | 为 MoE 的 Top-K routing 做概念铺垫                       |
| MoE 基本思想      | 如何增加参数量但控制 FLOPs                          | expert、router、active parameters                 | 引出现代稀疏激活大模型                                   |
| MoE routing       | token 如何选择 expert                               | token-choice Top-K、expert-choice、global routing | 影响质量、训练稳定性和系统负载                           |
| MoE 训练          | routing 不可微且容易 expert collapse                | load balancing loss、noisy routing、bias          | 是 MoE 能否训练成功的关键                                |
| MoE 系统          | 如何把专家分布到多设备并高效计算                    | expert parallelism、MegaBlocks、通信压缩          | 连接 Lecture 5 之后的系统优化主题                        |
| DeepSeek MoE 演化 | 现代 MoE 如何从 v1 到 v3 复杂化                     | shared expert、fine-grained expert、MLA、MTP      | 展示架构、训练目标和系统设计共同演进                     |

### 1.4 学习本讲前需要知道什么

本讲需要的前置知识主要来自四类：

| 前置知识           | 为什么需要                                                   |
| ------------------ | ------------------------------------------------------------ |
| 线性代数与矩阵乘法 | attention 的复杂度分析依赖 $QK^\top$、$K^\top V$ 和矩阵结合律 |
| 概率与 softmax     | 标准 attention 的权重归一化、router softmax、z-loss 都和概率分布有关 |
| Transformer 基础   | 需要知道 attention block、MLP/FFN block、residual connection 的位置 |
| 深度学习系统基础   | FlashAttention、KV cache、expert parallelism、通信开销都属于系统效率问题 |

官方课程页也把 Python、深度学习与系统优化、线性代数、概率统计和机器学习列为 CS336 的重要前置能力。

------

## 2. 第一部分：长上下文与 attention 成本瓶颈

### 2.1 本节要解决什么问题

这一节要回答：**为什么上下文长度一变长，标准 attention 就会变成大模型训练和推理中的核心瓶颈？**

课程从一个现实趋势开始：现代语言模型越来越追求更长的 context window。更长上下文可以容纳更多文档、更多工具调用记录、更多对话历史，也更适合 agent 场景。但 context 变长后，attention 的 all-to-all 连接成本迅速上升。【字幕†source】

### 2.2 标准 attention 的直觉与正式定义

直觉上，self-attention 做的是：每个 token 都可以“看”前面或全部位置上的 token，然后决定哪些历史信息对当前 token 最有用。

在矩阵形式中，设：
$$
Q\in \mathbb{R}^{n\times d_k},\quad
K\in \mathbb{R}^{n\times d_k},\quad
V\in \mathbb{R}^{n\times d_v}
$$
其中：

| 符号   | 含义                                               |
| ------ | -------------------------------------------------- |
| $n$    | 序列长度，也就是 token 数                          |
| $d_k$  | query/key 的向量维度                               |
| $d_v$  | value 的向量维度                                   |
| $Q$    | 每个位置的 query 向量组成的矩阵                    |
| $K$    | 每个位置的 key 向量组成的矩阵                      |
| $V$    | 每个位置的 value 向量组成的矩阵                    |
| $\rho$ | 课程中抽象表示 softmax 或 row normalization 的操作 |

标准 attention 可以抽象写成：
$$
\mathrm{Attn}(Q,K,V)=\rho(QK^\top)V
$$
这里最贵的一步是 $QK^\top$。因为 $Q$ 有 $n$ 行，$K$ 也有 $n$ 行，所以 $QK^\top$ 会产生一个 $n\times n$ 的注意力分数矩阵。也就是说，每个 token 都要和每个 token 建立关系。

计算复杂度大致为：
$$
QK^\top: O(n^2d_k)
$$
再乘以 $V$：
$$
(QK^\top)V: O(n^2d_v)
$$
所以整体成本与 $n^2$ 相关。PDF 第 4 页正是用这个形式引出 attention 的 quadratic cost。【PDF†source】

### 2.3 为什么 FFN 不是同样的瓶颈

FFN/MLP 对每个 token 独立做变换。也就是说，如果序列长度从 $n$ 变成 $2n$，FFN 大致只是处理两倍 token，成本线性上升。

而 attention 中每个 token 要和其他 token 交互。如果序列长度从 $n$ 变成 $2n$，$n\times n$ 的注意力矩阵会变成 $2n\times 2n$，成本约变成四倍。

这就是课程中强调的核心区别：**FFN 随序列长度主要线性增长，attention 随序列长度二次增长**。短上下文下，FFN 可能是主要计算成本；长上下文下，attention 会逐渐成为主要问题。【字幕†source】

### 2.4 第一类缓解方法：local/global attention 混合

一种自然想法是：不是每层都做 full global attention。

例如，模型可以让大多数层只做 local attention，只在少数层做 global attention。原报告中提到，课程举例说明可以每隔若干层才做一次 global attention，其余层使用 local attention，从而降低昂贵的全局 all-to-all 成本。【字幕†source】

直觉上，这类似于阅读长文档：

- 不是每句话都需要和全文每句话直接比较；
- 大多数时候只需要看附近上下文；
- 偶尔再通过全局层把远距离信息汇总起来。

**学习提示：** local attention 不是完全解决长程依赖，而是在“成本”和“长程信息流”之间做折中。

### 2.5 第二类缓解方法：FlashAttention

FlashAttention 的核心不是改变 attention 的数学定义，而是改变计算组织方式。

标准 attention 的大问题不仅是 FLOPs，还有内存读写。FlashAttention 通过更硬件友好的分块计算、减少中间矩阵写回显存等方式，显著降低 memory transfer overhead。课程强调它属于 systems engineering：它不改变 $O(n^2)$ 的大 O 复杂度，但会带来很大的常数因子收益。【字幕†source】

**工程直觉：** 在 GPU 上，速度常常不只取决于算了多少乘加，还取决于数据在 HBM、SRAM、cache 和寄存器之间搬了多少次。FlashAttention 的价值就在于减少不必要的数据搬运。

### 2.6 为什么还需要更激进的替代方案

local/global attention 和 FlashAttention 都很有用，但它们没有根本消除 $n^2$ 依赖。如果目标是百万级甚至更长上下文，仅靠局部化和系统优化可能仍不够。因此课程转向下一类方法：**linear-time attention alternatives**。【字幕†source】

### 2.7 本节小结

本节最重要的结论是：

> 长上下文让标准 attention 的 $O(n^2)$ all-to-all 交互成为瓶颈；local/global attention 和 FlashAttention 可以缓解成本，但如果上下文继续扩大，就需要更激进的线性化、状态化或稀疏选择方法。

复习时要抓住三点：

1. $QK^\top$ 为什么产生 $n\times n$ 矩阵；
2. FlashAttention 为什么是常数因子优化，不是复杂度阶数改变；
3. local attention 为什么是牺牲部分全局交互换取效率。

------

## 3. 第二部分：线性注意力、RNN duality、Mamba-2 与 Gated Delta Net

### 3.1 本节要解决什么问题

这一节要回答：**如果标准 attention 的 $n^2$ 成本太高，能不能通过数学重排把它变成近似线性时间？**

课程的关键转折是：先暂时把 softmax 去掉，把 attention 看成普通矩阵乘法。这样就可以利用矩阵乘法结合律：
$$
(QK^\top)V = Q(K^\top V)
$$
这一步是 linear attention 的核心直觉。【PDF†source】

### 3.2 线性注意力的核心直觉

标准 attention 通常先算：
$$
QK^\top
$$
这会得到 $n\times n$ 的矩阵，然后再乘 $V$。问题是这个 $n\times n$ 中间对象太大。

如果没有 softmax 或 row normalization，我们可以先算：
$$
K^\top V
$$
由于 $K^\top$ 的形状是 $d_k\times n$，$V$ 的形状是 $n\times d_v$，所以 $K^\top V$ 的结果是：
$$
d_k\times d_v
$$
这个结果不再随 $n^2$ 增长，而是一个与 hidden dimension 相关的状态矩阵。再用 $Q$ 去读这个状态：
$$
Q(K^\top V)
$$
复杂度从：
$$
O(n^2d_k+n^2d_v)
$$
变成接近：
$$
O(nd_kd_v)
$$
PDF 第 4 页用这个对比说明 linear attention 的数学来源。【PDF†source】

### 3.3 重要限制：去掉 softmax 是有损的

这里最容易误解的是：**linear attention 不等价于 full softmax attention**。

标准 attention 的表达力很大一部分来自 softmax，它把每个 query 对所有 key 的分数转化为概率分布。去掉 softmax 后，模型不再具有完全相同的注意力权重机制。

课程问答中特别澄清：从 full softmax attention 到 linear attention 的第一步是 lossy；只有在线性 attention 内部，从 dense form 到 recurrent form 的变换才是精确等价。【字幕†source】

**学习提示：** 不要说“linear attention 等价于 attention”。更准确的说法是：

> linear attention 是在放弃 softmax 归一化后，利用结合律得到的一类高效替代结构；它和 full softmax attention 之间存在表达力折中。

### 3.4 从 dense form 到 recurrent form

linear attention 的另一个关键优势是，它可以写成 RNN 风格的状态递推。

PDF 第 5 页将其写为：
$$
S_t=S_{t-1}+k_tv_t^\top
$$
输出可以写成：
$$
y_t=q_t^\top S_t
$$
符号解释如下：

| 符号  | 含义                                                         |
| ----- | ------------------------------------------------------------ |
| $S_t$ | 到第 $t$ 个 token 为止累计得到的状态矩阵，形状约为 $d_k\times d_v$ |
| $k_t$ | 第 $t$ 个 token 的 key 向量                                  |
| $v_t$ | 第 $t$ 个 token 的 value 向量                                |
| $q_t$ | 第 $t$ 个 token 的 query 向量                                |
| $y_t$ | 第 $t$ 个 token 的输出                                       |

直觉上，$k_tv_t^\top$ 是把当前 token 的 key-value 信息写入状态。随后，当前 query $q_t$ 从这个状态中读出信息。

这就像把完整历史压缩进一个固定大小的“记忆矩阵”：

- 标准 attention：保留所有历史 token 的 key/value；
- recurrent linear attention：把历史不断累积进 $S_t$；
- 推理时只需要携带 $S_t$，不需要保存完整的历史矩阵。【字幕†source】

### 3.5 dense/serial duality：训练并行，推理递归

linear attention 的一个重要卖点是 duality：

| 阶段 | 使用形式                                            | 优点                         |
| ---- | --------------------------------------------------- | ---------------------------- |
| 训练 | dense/parallel form，即 $Q(K^\top V)$               | 可以利用矩阵乘法并行训练     |
| 推理 | serial/recurrent form，即 $S_t=S_{t-1}+k_tv_t^\top$ | 自回归生成时携带固定大小状态 |

PDF 第 5 页明确强调，这种 duality 允许训练时使用 parallel/dense form，推理时使用 serial/recurrent form。【PDF†source】

这解释了为什么 state space model、linear attention 和 Mamba 系列重新受到关注：它们不像传统 RNN 那样完全牺牲训练并行性，又能在推理时像 RNN 一样用固定状态前进。

### 3.6 Mamba-2：给线性状态加遗忘门

线性 attention 的状态更新是：
$$
S_t=S_{t-1}+k_tv_t^\top
$$
这个更新有一个明显问题：它总是把历史状态全部保留下来，再加入当前信息。现实中，模型应该能决定哪些信息保留、哪些信息遗忘。

Mamba-2 在这个基础上加入 gate：
$$
S_t=\gamma_tS_{t-1}+k_tv_t^\top
$$
其中：

| 符号          | 含义                                    |
| ------------- | --------------------------------------- |
| $\gamma_t$    | 当前输入相关的 gate，控制旧状态保留多少 |
| $k_tv_t^\top$ | 当前 token 写入状态的信息               |
| $S_t$         | 更新后的状态                            |

当 $\gamma_t$ 接近 1 时，模型保留大部分旧状态；当 $\gamma_t$ 较小时，模型更强地遗忘过去状态。课程用 LSTM 的直觉解释这个设计：好的序列模型需要知道什么时候保留、什么时候遗忘。【字幕†source】

Mamba-2 还包含类似 residual pass-through 的项：
$$
y_t=q_t^\top S_t+v_t^\top D
$$
其中 $v_t^\top D$ 让当前 token 的 value 可以直接影响输出。字幕问答中解释，这类似让当前 value 通过一条直通路径进入输出，而不完全依赖历史状态读取。【字幕†source】

### 3.7 Mamba-2 的课堂例子：Nemotron 3

PDF 第 8 页给出 Nemotron 3 作为 Mamba attention hybrid 的案例，原报告总结其大约采用 3:1 的 Mamba/attention 混合结构，并说明它在类似模型中取得可比或更好的性能。【PDF†source】

这个例子的意义不在于“完全用 Mamba 替代 attention”，而在于展示一种更现实的工程折中：

- 多数层使用轻量的 Mamba-2 / 状态空间层；
- 少数层保留 full softmax attention；
- 既降低长上下文推理成本，又保留一定全局表达力。【字幕†source】

### 3.8 Gated Delta Net：写入之前先擦除

Gated Delta Net 进一步改造状态更新。PDF 第 9 页给出公式：
$$
S_t=\gamma_t(I-\beta_tk_tk_t^\top)S_{t-1}+\beta_tk_tv_t^\top
$$
这个公式可以分成三部分理解：

| 结构                   | 作用                            | 直觉                   |
| ---------------------- | ------------------------------- | ---------------------- |
| $\gamma_t$             | 控制旧状态整体保留多少          | 遗忘门                 |
| $I-\beta_tk_tk_t^\top$ | 擦除与当前 key 方向相关的旧信息 | 写新内容之前清理旧槽位 |
| $\beta_tk_tv_t^\top$   | 把当前 key-value 写入状态       | 输入写入门             |

其中 $\beta_t$ 的作用是控制当前输入是否写入状态。当 $\beta_t=0$ 时，当前 token 不写入；当 $\beta_t$ 较大时，当前 token 对状态影响更强。【字幕†source】

擦除项 $I-\beta_tk_tk_t^\top$ 很关键。它的直觉是：如果当前 key 方向上已有旧信息，而现在要写入新 value，就应该先把旧信息擦掉，否则新旧内容会混在一起。课程也指出，这个结构和 fast weight programming、test-time training、某些 meta-learning least-squares 更新有联系。【字幕†source】【PDF†source】

**补充说明：** 这个擦除项可以类比“键值数据库里同一个 key 被更新”。如果你要给同一个 key 写入新 value，理想情况下不是简单叠加，而是先覆盖旧 value，再写入新 value。

### 3.9 GDN 的课堂例子：Qwen Next 与 hybrid performance

PDF 第 10 页介绍 Qwen 3.5 / Qwen Next 使用 3:1 的 GDN / Attention hybrid，并展示长上下文下 decoding throughput 的优势。【PDF†source】

PDF 第 11 页展示 hybrid performance 的受控实验：随着非 full attention 层比例增加，性能开始下降；低比例替换时损失较小，完全变成 RNN/状态空间层时退化更明显。【PDF†source】

这支持课程的重要结论：

> 当前更稳妥的路线通常不是完全抛弃 softmax attention，而是在大量轻量层之间插入少量 full attention 层，形成 hybrid architecture。【字幕†source】

### 3.10 本节小结

本节最重要的结论是：

> linear attention 的核心是去掉 softmax 后用结合律把 $QK^\top V$ 改写为 $Q(K^\top V)$，从而获得线性化成本；它还能写成 recurrent state update，在训练和推理之间切换执行形式。Mamba-2 和 Gated Delta Net 则通过 gate、遗忘、写入和擦除增强状态模型的表达能力。

复习时要重点掌握：

1. 为什么结合律只有在去掉 softmax 后才直接成立；
2. 为什么 linear attention 与 full softmax attention 不等价；
3. $S_t$ 的状态递推如何连接训练并行和推理递归；
4. Mamba-2 的 $\gamma_t$ 和 GDN 的 $\beta_t$ 分别解决什么问题；
5. 为什么现代模型多采用 hybrid，而不是完全替代 attention。

------

## 4. 第三部分：DSA 稀疏注意力——不压缩全部历史，而是选择重要历史

### 4.1 本节要解决什么问题

这一节要回答：**除了把 attention 线性化，还有没有别的方式降低长上下文成本？**

DSA，也就是 DeepSeek Sparse Attention，走的是另一条路线。它不把历史压缩进固定大小状态，而是先用一个轻量 indexer 从历史 token 中选出 Top-K 个重要位置，然后只在这些位置上做 full attention。【字幕†source】

### 4.2 DSA 的核心直觉

linear attention 的想法是：

> 把所有历史信息压缩进一个状态矩阵。

DSA 的想法则是：

> 不压缩所有历史，而是先判断哪些历史 token 值得看，只对少数 token 做精确 attention。

这很像读长文档时的检索过程。你不一定要把整本书都压缩成一个摘要；也可以先用目录或索引找到最相关的几页，再精读这些页。

### 4.3 DSA 的 indexer 公式

PDF 第 12 页给出 DSA 的原型。indexer 为当前 query token $h_t$ 和前序 token $h_s$ 计算 index score：
$$
I_{t,s}=\sum_{j=1}^{H_I} w_{t,j}^I\cdot \mathrm{ReLU}\big((q_{t,j}^I)^\top k_s^I\big)
$$
然后选择 Top-K 分数对应的 token 子集 $c_s$，再做 attention：
$$
u_t=\mathrm{Attn}(h_t,\{c_s\mid I_{t,s}\in \mathrm{Top\text{-}K}(I_{t,:})\})
$$
符号解释：

| 符号      | 含义                                     |
| --------- | ---------------------------------------- |
| $h_t$     | 当前 query token 的 hidden state         |
| $h_s$     | 前序 token 的 hidden state               |
| $I_{t,s}$ | 当前 token $t$ 对历史位置 $s$ 的索引分数 |
| $H_I$     | indexer 使用的头数或索引维度             |
| $q^I,k^I$ | 用于索引的轻量 query/key                 |
| Top-K     | 只保留得分最高的 K 个历史 token          |

PDF 说明 indexer 可以设计得很轻量，因此虽然它仍要扫描历史位置，但整体可以节省大量后续 full attention 的成本。【PDF†source】

### 4.4 DSA 不是严格线性时间

这里也有一个重要误区：**DSA 不是严格的 linear attention。**

课程问答中明确指出，indexer 为了知道选哪些 token，仍然要看所有位置，仍涉及 q-k inner products；它不是一个固定状态递推，也不是严格线性时间。它的收益来自：

- indexer 可以低维；
- indexer 可以低精度；
- full attention 只在 Top-K 子集上做；
- 有效参与 full attention 的 token 数大幅减少。【字幕†source】

因此，DSA 更像是“轻量检索 + 小集合精确 attention”，而不是“把 attention 改写成 RNN 状态”。

### 4.5 DSA 与 MoE 的桥梁：Top-K 选择

DSA 在课程中的另一个作用，是为下半讲 MoE 做铺垫。

DSA 中的 Top-K 是：

> 当前 token 选择 Top-K 历史 token 来 attention。

MoE 中的 Top-K 是：

> 当前 token 选择 Top-K expert 来计算。

两者共享一个重要思想：**不是所有候选对象都参与计算，而是先做轻量打分，再选择少数对象执行昂贵操作。** 字幕也明确指出，DSA 中的 Top-K selection 会成为下一部分 MoE 的核心思想。【字幕†source】

### 4.6 本节小结

本节最重要的结论是：

> DSA 不通过状态压缩替代 attention，而是通过轻量 indexer 选择少量重要 token，再在这些 token 上做 full attention。它不是严格线性时间，但通过 Top-K 稀疏选择显著降低了有效 attention 成本。

复习时要特别区分：

| 方法             | 核心思路                                           | 是否保留 full attention           |
| ---------------- | -------------------------------------------------- | --------------------------------- |
| linear attention | 去掉 softmax，用状态矩阵压缩历史                   | 不保留标准 full softmax attention |
| DSA              | 先选 Top-K 历史 token，再对小集合做 full attention | 局部保留 full attention           |
| local attention  | 固定看附近窗口                                     | 不根据内容动态选择                |
| FlashAttention   | 优化计算组织和内存访问                             | 保留原始 attention 数学形式       |

------

## 5. 第四部分：Mixture of Experts——用稀疏专家增加模型容量

### 5.1 本节要解决什么问题

这一节要回答：**如果想让模型拥有更多参数，但不想每个 token 都经过所有参数，应该怎样设计模型？**

这就是 Mixture of Experts。MoE 的核心目标是：增加 total parameters，但控制 active parameters。

### 5.2 MoE 的基本定义

标准 Transformer block 中通常有一个 FFN/MLP。MoE 将这个 FFN 替换成多个 expert FFN，再加入一个 router 或 selector 来决定每个 token 送到哪些 expert。

原报告中引用 PDF 第 15 页对 MoE 的定义：用许多大的 feed-forward networks 和一个 selector layer 替代原来的大 FFN，并指出可以增加 expert 数量而不等比例增加 FLOPs。【PDF†source】

直觉例子：

假设原来一个 block 中只有一个 FFN。现在放 4 个 FFN expert，但每个 token 只选择其中 1 个。这样：

- 总 FFN 参数量约变成 4 倍；
- 每个 token 实际只经过 1 个 FFN；
- 每次 forward/backward 的主要计算量没有变成 4 倍。【字幕†source】

这就是 MoE 的核心心理模型：

> 参数很多，但每次只用一小部分。

### 5.3 total parameters、active parameters 与 sparse parameters

理解 MoE 必须区分三个概念：

| 概念              | 含义                         | 为什么重要                             |
| ----------------- | ---------------------------- | -------------------------------------- |
| total parameters  | 所有 expert 加起来的参数量   | 决定模型总容量、存储成本和加载成本     |
| active parameters | 某个 token 实际经过的参数量  | 决定每次 forward/backward 的主要 FLOPs |
| sparse parameters | 存在但每次只激活一部分的参数 | 是 MoE 性价比的来源                    |

课程反复强调，同等 FLOPs 下，更多 sparse parameters 往往可以改善模型质量。PDF 第 16 页展示 Fedus et al. 的结果：在 same FLOP 下增加 sparse model parameters，test loss 下降；训练步数相近时，更多 experts 的模型 perplexity 更好。【PDF†source】

**学习提示：** dense model 的参数大多每次都参与计算；MoE 的大部分专家参数只在部分 token 上被激活。因此比较 MoE 和 dense model 时，不能只看 total parameters，也不能只看 active parameters，要同时看两者。

### 5.4 为什么 MoE 流行，又为什么不容易训练

MoE 流行的原因主要有三个。

第一，质量/成本比好。PDF 第 17 页显示 MoE 相比 dense model 训练更快，在 training loss、validation loss 和 downstream benchmark 上更优。【PDF†source】

第二，在相同 activated parameters 下，MoE 往往能利用更多总参数获得更强能力。PDF 第 18 页用 active parameters 与 MMLU performance 的图展示 MoE 与 dense equivalents 的竞争力。【PDF†source】

第三，MoE 提供了新的系统并行轴。每个 expert 可以放在不同设备上，token activation 被 route 到对应设备，这形成 expert parallelism。【字幕†source】

但 MoE 不容易早早普及，原因也很清楚：

- 参数多，不容易放在单设备；
- expert parallelism 带来跨设备通信；
- router 的稀疏选择不可微；
- expert 可能 collapse；
- 训练目标包含很多 heuristic；
- router softmax 还可能带来数值稳定性问题。【字幕†source】【PDF†source】

### 5.5 MoE routing：token 如何选择 expert

PDF 第 26 页把 MoE 设计空间分成三条轴：routing function、expert sizes、training objectives。【PDF†source】

routing function 决定 token 去哪个 expert。PDF 第 27 页列出三类：

| routing 类型         | 含义                               | 课程评价             |
| -------------------- | ---------------------------------- | -------------------- |
| token chooses expert | 每个 token 选择 Top-K expert       | 现代 MoE 最常见      |
| expert chooses token | 每个 expert 选择自己要处理的 token | 有案例，但不是主流   |
| global routing       | 解全局优化或匹配问题               | 理论上漂亮，但成本高 |

课程强调，现代 MoE 几乎都采用 token-choice Top-K routing：每个 token 给所有 expert 打分，选出前 $K$ 个 expert。【字幕†source】

### 5.6 Top-K routing 的公式

PDF 第 31 页给出经典 Top-K routing 形式：
$$
h_t^l=\sum_{i=1}^{N}\left(g_{i,t}\mathrm{FFN}_i(u_t^l)\right)+u_t^l
$$
符号解释：

| 符号             | 含义                                     |
| ---------------- | ---------------------------------------- |
| $u_t^l$          | 第 $l$ 层第 $t$ 个 token 的输入表示      |
| $\mathrm{FFN}_i$ | 第 $i$ 个 expert                         |
| $N$              | expert 总数                              |
| $s_{i,t}$        | token $t$ 对 expert $i$ 的 routing score |
| $g_{i,t}$        | Top-K 截断后的 gate 权重                 |
| $h_t^l$          | MoE 层输出                               |
| $+u_t^l$         | residual connection                      |

直觉上，这个公式做了三件事：

1. router 给每个 expert 打分；
2. 只保留 Top-K expert；
3. 把这些 expert 的输出按 gate 权重加权求和，再加上 residual。

课程中特别提醒，router 通常并不是一个复杂的“语义分类器”。它常常只是 token 表示和 expert routing 向量之间的内积。也就是说，不要简单把 expert 想象成“医疗专家”“法律专家”这类人类可解释的语义专家。【字幕†source】

### 5.7 routing 变体：Switch、GShard、Mixtral、DeepSeek 等

PDF 第 29 页总结了不同模型的 routing 选择：

| 模型或方法              | routing 特点         |
| ----------------------- | -------------------- |
| Switch Transformer      | $k=1$                |
| GShard / Grok / Mixtral | $k=2$                |
| Qwen / DBRX             | $k=4$                |
| DeepSeek                | 更大的 $k$，如 $k=7$ |
| Hashing                 | 常见 baseline        |

课程还讨论了 RL routing 和 linear assignment。RL 从 bandit 角度看很自然，因为每次只观察被选中 expert 的反馈；但 RL 带来高方差、复杂度和额外随机性，因此并不常见。global assignment 理论上更优雅，但规模化成本太高。【字幕†source】

### 5.8 Shared experts 与 fine-grained experts

DeepSeekMoE 引入两个重要设计：shared experts 和 fine-grained experts。

**shared experts** 是不经过 router、始终开启的专家。它们处理通用模式，让 routed experts 更专注于条件化处理。换句话说，不是所有知识都适合通过稀疏路由完成；有些基础转换可能每个 token 都需要。【字幕†source】

**fine-grained experts** 是把原来较大的 expert 切成更多更小的 expert。这样 router 可以更细粒度地组合专家，提高稀疏选择的灵活性。

PDF 第 32 页展示了从 conventional top-2 routing 到 fine-grained expert segmentation，再到 shared expert isolation 的结构变化。【PDF†source】

PDF 第 35 页列出现代 MoE 的 routed / active / shared / fine-grained ratio 设置。例如 DeepSeek v1、Qwen 1.5、DeepSeek v3 和 OlMoE 都采用不同组合，说明现代 MoE 没有唯一固定配方。【PDF†source】

### 5.9 MoE 训练难点：稀疏 routing 不可微

MoE 训练的根本困难是：为了效率，训练时也必须稀疏激活 expert；但 Top-K gating 是离散选择，不可微。

PDF 第 36 页直接提出：训练需要 sparsity 才有效率，但 sparse gating decisions 不可微。【PDF†source】

这带来一种 bandit flavor：token 只试了少数 expert，不知道没选的 expert 会表现怎样。但现代 MoE 通常不直接用 RL 解决，而是依赖 noisy routing、load balancing loss 和训练动态等 heuristic。【字幕†source】

### 5.10 expert collapse：MoE 最核心的失败模式

expert collapse 指少数 expert 获得越来越多 token，而其他 expert 几乎不用。

这个过程可以理解为 rich-get-richer：

1. 某个 expert 初始稍微好一点；
2. router 把更多 token 分给它；
3. 它获得更多梯度信号；
4. 它变得更好；
5. router 进一步偏向它；
6. 其他 expert 得不到训练，逐渐“饿死”。【字幕†source】

这不仅浪费参数，也会造成系统负载不均。一个设备上的 expert 过热，其他设备空闲，整体吞吐下降。

### 5.11 Switch Transformer 的 load balancing loss

PDF 第 40 页给出辅助负载均衡损失：
$$
\mathrm{loss}=\alpha\cdot N\cdot \sum_{i=1}^{N} f_iP_i
$$
符号解释：

| 符号     | 含义                                      |
| -------- | ----------------------------------------- |
| $N$      | expert 数量                               |
| $T$      | batch 中 token 数量                       |
| $f_i$    | 实际 dispatch 到 expert $i$ 的 token 比例 |
| $P_i$    | router 分配给 expert $i$ 的概率质量       |
| $\alpha$ | balancing loss 权重                       |
| $p_i(x)$ | token $x$ 被分配给 expert $i$ 的概率      |

这个 loss 的直觉不是来自一个特别优雅的第一性原理推导，而是来自梯度效果：热门 expert 的 $f_i$ 高，因此它对应的概率质量 $P_i$ 会受到更强惩罚，从而推动 router 把 token 分给较少使用的 expert。【字幕†source】【PDF†source】

**学习提示：** load balancing loss 不是无关紧要的小技巧。原报告引用 OlMoE ablation 表明，移除 load balancing 后，训练/验证损失明显变差，expert assignment 也高度不均。【PDF†source】

### 5.12 DeepSeek 的 balancing：从 auxiliary loss 到 per-expert bias

DeepSeek v1-v2 使用 per-expert balancing 和 per-device balancing。前者让 expert 使用更均匀，后者按设备聚合平衡通信和计算负载。【PDF†source】

DeepSeek v3 则引入 per-expert bias：给每个 expert 一个 bias $b_i$，让它更容易或更不容易被选中，并通过 online learning 调节。PDF 称其为 auxiliary-loss-free balancing，但原报告也提醒，它仍然有 sequence-wise auxiliary loss 来防止单个 sequence 内极端不平衡。【PDF†source】

这个演化说明，MoE 的训练目标不仅是机器学习问题，也是系统问题：不仅要让模型质量好，还要让设备负载均衡、通信可控。

### 5.13 MoE 系统问题：expert parallelism 与通信

MoE 的系统价值在于 expert parallelism。不同 expert 可以放到不同设备上，token 根据 routing 被发送到对应设备。【PDF†source】

但这个过程会带来 all-to-all 通信。课程问答中解释，这是一个 trade-off：expert parallelism 提供更多 aggregate FLOPs，也可能缓解单设备内存压力，但必须支付跨设备传输 activation 的成本，是否划算取决于硬件拓扑。【字幕†source】

### 5.14 MegaBlocks 与 block-sparse 计算

如果一张 GPU 上有多个 expert，逐个执行很多小矩阵乘法会非常低效。PDF 第 45 页展示了 MoE routing 的批量矩阵乘法、block diagonal matrix multiplication 和 block sparse matrix multiplication。【PDF†source】

MegaBlocks 等库的作用，就是把多个专家计算组织成更硬件友好的块稀疏大操作，从而提高 GPU 利用率。【字幕†source】

**工程直觉：** GPU 喜欢大而规则的矩阵乘法，不喜欢很多零碎的小 kernel。MoE 的工程优化很大一部分就是把“稀疏逻辑”重排成“硬件友好的块状计算”。

### 5.15 降低通信：down-project activations

PDF 第 46 页介绍 Nemotron 3 的思路：先把 residual stream down-project 到较低维 latent activation，再把这个较低维表示发送到 routed expert，从而减少跨设备通信量。【PDF†source】

直觉上，shared expert 可以保留大维度，因为它不需要跨设备 routing；routed expert 的 activation 需要跨设备发送，因此可以先降维，换取通信效率。【字幕†source】

### 5.16 token dropping 与 drop-less MoE

PDF 第 47 页讨论早期 MoE 的 stochasticity：当某个 expert 队列太长时，系统可能进行 token dropping。字幕中老师解释，这意味着其他用户请求可能影响你的 token，因为他们的 token 可能占用了同一个 expert 队列。【字幕†source】

现代 drop-less architecture 和 MegaBlocks 等框架大多缓解了这个问题，但它提醒我们：MoE 不仅改变模型结构，也改变服务时的系统行为。

### 5.17 router stability 与 z-loss

MoE 引入额外的 router softmax，而 softmax 涉及 exponentials 和 divisions，容易出现数值稳定性问题。

PDF 第 48-49 页介绍 router 使用 FP32 和 z-loss 来提高稳定性。【PDF†source】

z-loss 形式为：
$$
L_z(x)=\frac{1}{B}\sum_{i=1}^{B}\left(\log\sum_{j=1}^{N}e^{x_j^i}\right)^2
$$
其中：

| 符号    | 含义                                            |
| ------- | ----------------------------------------------- |
| $B$     | batch size                                      |
| $N$     | expert 数量                                     |
| $x_j^i$ | 第 $i$ 个样本对第 $j$ 个 expert 的 router logit |

它惩罚过大的 log-sum-exp，防止 router logits 无界变大，从而降低 softmax 溢出或训练尖峰风险。PDF 展示移除 z-loss 后训练曲线出现尖峰，说明它对稳定性有帮助。【PDF†source】

### 5.18 MoE fine-tuning 的过拟合问题

PDF 第 50 页指出 sparse MoE 在小型 fine-tuning data 上容易过拟合。【PDF†source】

原因很直观：MoE 总参数量很大，许多 expert 在小数据微调中可能被少量样本强烈影响，从而产生 train/val gap。课程提到的干预包括：

- 只 fine-tune non-MoE MLP；
- 只调 attention 或部分模块；
- 使用足够大的 SFT 数据；
- 谨慎控制专家参数更新。【字幕†source】

### 5.19 Upcycling：从 dense model 初始化 MoE

PDF 第 51 页提出 upcycling：能否用已有 dense LM 初始化 MoE？【PDF†source】

做法通常是复制 dense model 的 MLP 到多个 expert 中，router 随机初始化，然后继续训练。由于 routing 随机性和不同 token 的梯度差异，expert 会逐渐分化。【字幕†source】

PDF 第 52 页给出 MiniCPM 例子；PDF 第 53 页给出 Qwen MoE 例子，说明从已有 dense base model upcycle 成 MoE 是可行路线。【PDF†source】

### 5.20 DeepSeek MoE v1-v2-v3 的演化

原报告最后用 DeepSeek MoE 演化串起现代 MoE 设计。

| 版本            | 核心设置                                                     | 设计重点                                         |
| --------------- | ------------------------------------------------------------ | ------------------------------------------------ |
| DeepSeek MoE v1 | 16B total、2.8B active；标准 Top-K routing；2 shared experts；fine-grained experts；auxiliary balancing | shared + fine-grained + Top-K + balancing 的原型 |
| DeepSeek MoE v2 | 236B total、21B active；2 shared experts；160/10 fine-grained experts；新增 Top-M device routing 和 communication balancing | 从模型结构推进到设备通信优化                     |
| DeepSeek MoE v3 | 671B total、37B active；1 shared expert；更多 fine-grained experts；sigmoid + softmax Top-K + Top-M；per-expert bias / aux-loss-free-ish balancing | 更大规模、更复杂的 router 与 balancing 机制      |

PDF 第 54-56 页分别总结 v1、v2、v3 的这些设计。【PDF†source】

### 5.21 MLA 与 MTP

PDF 第 57-58 页介绍 MLA：Multihead Latent Attention。其核心想法是把 $Q,K,V$ 表示成较低维 latent activation 的函数，KV caching 时只需缓存较小的 $c_t^{KV}$，从而节省缓存。【PDF†source】

字幕解释，MLA 需要小心 RoPE 与 latent cache 的冲突，因此保留一些 non-latent key dimensions 来承载旋转位置编码。【字幕†source】

PDF 第 59 页介绍 MTP：让轻量模型预测多个未来 token。原报告指出 DeepSeek v3 实际只做 one token ahead；字幕补充，MTP 也可与 speculative decoding 联系起来，用于推理加速。【PDF†source】【字幕†source】

### 5.22 本节小结

本节最重要的结论是：

> MoE 用稀疏 expert 激活增加模型总容量，但它带来 routing 不可微、expert collapse、load balancing、通信、系统调度和 router 稳定性等一整套新问题。现代 MoE 成功不是单一架构技巧，而是模型设计、训练目标和系统工程共同作用的结果。

复习时要抓住：

1. total parameters 与 active parameters 的区别；
2. token-choice Top-K routing 的公式；
3. 为什么 expert collapse 会发生；
4. load balancing loss 如何从梯度上抑制热门 expert；
5. expert parallelism 为什么既有价值又有通信成本；
6. DeepSeek v1-v2-v3 的演化主线。

------

## 6. Python 代码示例详解：本讲没有 `lecture_04.py`，改用公式 / 伪代码等价解析

### 6.1 文件整体结构核对

GitHub 仓库目录显示 Lecture 4 对应的是 `lecture_04.pdf`，并且 README 说明可执行讲义命名为 `lecture_XX.py`，非可执行讲义命名为 `lecture_XX.pdf`。 因此，本讲没有可逐段解析的 `lecture_04.py` 或 notebook 文件。

本节不提供伪造代码，而是把 PDF 中的公式转换为“如果写成代码，会是什么结构”的伪代码解释。

### 6.2 关键公式的伪代码等价

| 课程知识点                 | 伪代码                                                       | 输入                      | 输出                | 理论对应                  |
| -------------------------- | ------------------------------------------------------------ | ------------------------- | ------------------- | ------------------------- |
| 标准 attention             | `scores = Q @ K.T; weights = rho(scores); out = weights @ V` | $Q,K,V$                   | attention 输出      | $\rho(QK^\top)V$          |
| 线性 attention             | `state = K.T @ V; out = Q @ state`                           | $Q,K,V$                   | 线性 attention 输出 | $Q(K^\top V)$             |
| recurrent linear attention | `S += outer(k_t, v_t); y_t = q_t @ S`                        | $q_t,k_t,v_t,S_{t-1}$     | $y_t,S_t$           | $S_t=S_{t-1}+k_tv_t^\top$ |
| Mamba-2                    | `S = gamma_t * S + outer(k_t, v_t)`                          | gate、key、value、旧状态  | 新状态              | 遗忘门状态更新            |
| Gated Delta Net            | `S = gamma * (I - beta * outer(k,k)) @ S + beta * outer(k,v)` | gate、key、value、旧状态  | 新状态              | 擦除后写入                |
| DSA                        | `idx = topk(index_scores); out = attention(q, K[idx], V[idx])` | 当前 token、历史 token    | 稀疏 attention 输出 | Top-K token selection     |
| MoE routing                | `active = topk(router_scores); out = sum(g_i * FFN_i(x)) + x` | token 表示、expert        | MoE 层输出          | Top-K expert selection    |
| load balancing             | `loss += alpha * N * dot(f, P)`                              | expert 使用比例和概率质量 | 辅助损失            | 防止 expert collapse      |
| z-loss                     | `loss += mean(logsumexp(router_logits)**2)`                  | router logits             | 稳定性正则项        | 控制 softmax logits       |

### 6.3 伪代码片段解析

#### 6.3.1 标准 attention

```
scores = Q @ K.T
weights = rho(scores)
out = weights @ V
```

这段伪代码对应标准 attention。`scores` 是每个 query token 和每个 key token 的相似度矩阵，形状为 $n\times n$。`rho` 表示 softmax 或行归一化。最后 `weights @ V` 表示用注意力权重对 value 加权求和。

学习重点是：`scores` 的形状就是成本来源。只要构造 $n\times n$ score matrix，就会遇到 $O(n^2)$ 复杂度。【PDF†source】

#### 6.3.2 线性 attention

```
state = K.T @ V
out = Q @ state
```

这段伪代码对应线性 attention 的结合律重排。`state` 的形状是 $d_k\times d_v$，不再是 $n\times n$。因此，当 $n$ 很大而 $d_k,d_v$ 相对固定时，这种写法在长上下文下更有优势。

但这段代码成立的前提是去掉或替代 softmax。不能把它当成标准 softmax attention 的无损实现。【字幕†source】

#### 6.3.3 recurrent linear attention

```
S = zeros(d_k, d_v)
for t in range(n):
    S = S + outer(k[t], v[t])
    y[t] = q[t] @ S
```

这段伪代码说明了 dense/recurrent duality。训练时可以用矩阵乘法并行计算；推理时可以逐 token 更新状态 $S$。这就是本讲中“best of both worlds”的直觉：训练像 Transformer 一样并行，推理像 RNN 一样携带固定状态。【PDF†source】【字幕†source】

#### 6.3.4 MoE Top-K routing

```
scores = router(x)           # token 对每个 expert 的打分
active = topk(scores, k)     # 只选 K 个 expert
out = 0
for i in active:
    out += gate[i] * expert[i](x)
out = out + x                # residual
```

这段伪代码对应 MoE forward。关键不是 expert 的内部 MLP，而是 `topk` 这一步。它让模型拥有大量 expert，但每个 token 只激活少数 expert。

学习重点是：`topk` 带来稀疏计算，也带来不可微和负载不均问题。

### 6.4 代码与课程知识点对应表

| 课程知识点           | 对应函数 / 变量 / 片段            | 代码功能                      | 对应理论原理          | 在课程中的作用                     |
| -------------------- | --------------------------------- | ----------------------------- | --------------------- | ---------------------------------- |
| attention 二次复杂度 | `scores = Q @ K.T`                | 生成 $n\times n$ score matrix | all-to-all attention  | 引出长上下文瓶颈                   |
| linear attention     | `state = K.T @ V`                 | 先聚合 key-value 状态         | 矩阵结合律            | 降低序列长度依赖                   |
| recurrent state      | `S += outer(k, v)`                | 累积历史状态                  | RNN-style update      | 解释推理高效性                     |
| Mamba-2 gate         | `S = gamma * S + update`          | 控制旧状态保留                | 输入相关遗忘门        | 增强表达力                         |
| GDN erase/write      | `(I - beta kk^T) @ S + beta kv^T` | 擦除旧 key 方向并写入新信息   | projector-like update | 解释状态更新更精细                 |
| DSA Top-K            | `idx = topk(index_scores)`        | 选择重要历史 token            | 稀疏 attention        | 连接 attention alternatives 与 MoE |
| MoE router           | `active = topk(router_scores)`    | 选择 expert                   | 稀疏激活              | 增加参数但控制 FLOPs               |
| load balancing       | `dot(f, P)`                       | 惩罚热门 expert               | 负载均衡正则          | 防止 expert collapse               |
| z-loss               | `logsumexp(router_logits)**2`     | 抑制 logits 过大              | softmax 稳定性        | 减少训练尖峰                       |

------

## 7. PDF / 幻灯片 / 图表解析

### 7.1 主要图表总览

| PDF 页面 / 图表组 | 展示内容                                                     | 支持的课程结论                                   | 对应课程内容          |
| ----------------- | ------------------------------------------------------------ | ------------------------------------------------ | --------------------- |
| 第 2 页           | context window 增长趋势；attention 与 FFN 成本对比           | 长上下文下 attention 成本会成为瓶颈              | 课程开头动机          |
| 第 3 页           | local/global attention 与 FlashAttention                     | 架构局部化和系统工程都能缓解成本                 | 长上下文基础工具箱    |
| 第 4 页           | $\rho(QK^\top)V$ 与 $Q(K^\top V)$                            | 去掉 softmax 后可用结合律降低复杂度              | linear attention      |
| 第 5 页           | $S_t=S_{t-1}+k_tv_t^\top$                                    | dense/recurrent duality                          | 训练并行、推理递归    |
| 第 7-11 页        | Mamba-2、Nemotron、GDN、Qwen Next、hybrid performance        | gate 和 hybrid architecture 提升效率与表达力折中 | 状态空间 / 线性注意力 |
| 第 12-13 页       | DSA indexer 与 sparse attention 结果                         | 轻量 Top-K 选择可减少 full attention 成本        | DSA                   |
| 第 15-18 页       | dense vs sparse model、same FLOP 下 sparse parameters        | MoE 可提高质量/成本比                            | MoE 基本思想          |
| 第 27-35 页       | routing 类型、Top-K routing、shared/fine-grained experts     | routing 和 expert 粒度是 MoE 设计核心            | MoE 架构设计          |
| 第 40-43 页       | balancing loss 和 ablation                                   | load balancing 对防止 collapse 很关键            | MoE 训练              |
| 第 44-49 页       | expert parallel、MegaBlocks、down-projection、token dropping、z-loss | MoE 是系统与稳定性问题                           | MoE 工程              |
| 第 51-60 页       | upcycling、MiniCPM、Qwen、DeepSeek v1-v3、MLA、MTP           | 现代 MoE 设计持续演化                            | 课程总结案例          |

### 7.2 如何读第 2 页：长上下文成本图

第 2 页左图展示 context window size 随时间增长。它要传达的不是某个模型的单点性能，而是一个趋势：模型越来越需要处理更长输入。【PDF†source】

第 2 页右图比较 attention 与 feed-forward 随 sequence length 增长的成本。读图时应关注：

- 横轴：sequence length；
- 纵轴：计算成本或占比；
- attention 曲线：随长度增长更快；
- FFN 曲线：随长度主要线性增长。

这张图支持课程开头的判断：当上下文越来越长，attention 会从“可接受成本”变成“主要瓶颈”。【PDF†source】

### 7.3 如何读第 4-5 页：linear attention 公式图

第 4 页的重点是矩阵乘法括号变化：
$$
(QK^\top)V \quad \rightarrow \quad Q(K^\top V)
$$
读图时要注意，这不是随便改变计算顺序，而是建立在 $\rho$ 为 identity 或被替代的前提上。也就是说，标准 softmax attention 不能直接这样重排。【PDF†source】

第 5 页把 dense form 转成 recurrent form。它要说明：

- 训练时可以使用并行矩阵乘法；
- 推理时可以使用固定状态递推；
- 这是 linear attention 与传统 RNN 的关键不同。【PDF†source】

### 7.4 如何读 Mamba-2 / GDN / hybrid performance 图

第 7-11 页的图表共同回答：如果 linear attention 表达力不够，能不能通过 gate 和 hybrid 设计弥补？

读图时应注意：

- Mamba-2 加的是遗忘 gate；
- GDN 加的是写入 gate 和擦除项；
- hybrid performance 图显示，少量 full attention 层仍然重要；
- 完全去掉 full attention 可能造成明显退化。【PDF†source】【字幕†source】

### 7.5 如何读 DSA 图

DSA 图表的核心是两阶段结构：

1. indexer 给历史 token 打分；
2. 只对 Top-K 历史 token 做 full attention。

读图时不要误以为 DSA 是 RNN-style linear attention。它仍然要扫描历史位置，只是扫描过程轻量化，后续 full attention 的集合变小。【字幕†source】

### 7.6 如何读 MoE 图表

MoE 图表可分成四组：

第一组，第 15-18 页，说明为什么 MoE 有吸引力：same FLOP 下 sparse parameters 增加可以带来更好 test loss 或 benchmark 表现。【PDF†source】

第二组，第 27-35 页，说明 routing 和 expert 设计空间：token-choice Top-K 是主流，shared/fine-grained experts 是现代 MoE 的重要设计。【PDF†source】

第三组，第 40-43 页，说明训练稳定性：没有 load balancing，expert 使用会高度不均，损失变差。【PDF†source】

第四组，第 44-49 页，说明系统问题：expert parallelism、块稀疏矩阵乘法、通信压缩、token dropping 和 router z-loss 都属于 MoE 真实落地必须解决的问题。【PDF†source】

------

## 8. 关键公式与推导汇总

### 8.1 公式总表

| 公式                                                         | 符号解释                                             | 适用场景                   | 直觉解释                      | 代码 / 例子对应          |
| ------------------------------------------------------------ | ---------------------------------------------------- | -------------------------- | ----------------------------- | ------------------------ |
| $\mathrm{Attn}(Q,K,V)=\rho(QK^\top)V$                        | $\rho$ 是 softmax/归一化；$Q,K,V$ 是 query/key/value | 标准 attention             | 每个 token 与所有 token 交互  | `scores = Q @ K.T`       |
| $(QK^\top)V=Q(K^\top V)$                                     | 仅在无 softmax 或适当替代时成立                      | linear attention           | 先聚合 $K,V$，再由 $Q$ 读取   | `state = K.T @ V`        |
| $S_t=S_{t-1}+k_tv_t^\top$                                    | $S_t$ 是累计状态                                     | recurrent linear attention | 把历史 KV 压进固定状态        | `S += outer(k, v)`       |
| $y_t=q_t^\top S_t$                                           | $q_t$ 从状态中读信息                                 | recurrent inference        | 当前 query 读取历史状态       | `y = q @ S`              |
| $S_t=\gamma_tS_{t-1}+k_tv_t^\top$                            | $\gamma_t$ 是输入相关 gate                           | Mamba-2                    | 控制旧状态保留多少            | `S = gamma * S + update` |
| $y_t=q_t^\top S_t+v_t^\top D$                                | $D$ 控制当前 value 直通                              | Mamba-2 输出               | 让当前 value 直接影响输出     | pass-through / residual  |
| $S_t=\gamma_t(I-\beta_tk_tk_t^\top)S_{t-1}+\beta_tk_tv_t^\top$ | $\beta_t$ 控制写入；擦除项清理旧 key 方向            | Gated Delta Net            | 先擦除再写入                  | GDN 状态更新             |
| $I_{t,s}=\sum_j w_{t,j}^I\mathrm{ReLU}((q_{t,j}^I)^\top k_s^I)$ | indexer score                                        | DSA                        | 轻量判断哪些历史 token 值得看 | `topk(index_scores)`     |
| $h_t^l=\sum_i g_{i,t}\mathrm{FFN}_i(u_t^l)+u_t^l$            | $g_{i,t}$ 是 Top-K gate                              | MoE forward                | 只激活少数 expert             | `topk(router_scores)`    |
| $\mathrm{loss}=\alpha N\sum_i f_iP_i$                        | $f_i$ 是实际 token 比例；$P_i$ 是概率质量            | load balancing             | 惩罚热门 expert               | `dot(f, P)`              |
| $L_z=\frac1B\sum_i(\log\sum_j e^{x_j^i})^2$                  | $x_j^i$ 是 router logits                             | router stability           | 抑制 logits 过大              | `logsumexp(logits)**2`   |

### 8.2 关键推导：linear attention 为什么能降复杂度

标准形式：
$$
(QK^\top)V
$$
如果先算 $QK^\top$：
$$
QK^\top \in \mathbb{R}^{n\times n}
$$
复杂度约为：
$$
O(n^2d_k)
$$
再乘 $V$：
$$
O(n^2d_v)
$$
如果改为先算 $K^\top V$：
$$
K^\top V\in \mathbb{R}^{d_k\times d_v}
$$
复杂度约为：
$$
O(nd_kd_v)
$$
再乘 $Q$：
$$
Q(K^\top V)\in \mathbb{R}^{n\times d_v}
$$
复杂度约为：
$$
O(nd_kd_v)
$$
所以总成本接近：
$$
O(2nd_kd_v)
$$
这就是 PDF 第 4 页给出的复杂度改善直觉。【PDF†source】

但复习时必须写清楚限制：

> 这个推导依赖于没有 softmax 归一化；从 full softmax attention 到 linear attention 是有损近似或替代，而不是无损重写。【字幕†source】

### 8.3 关键推导：load balancing loss 的梯度直觉

load balancing loss：
$$
\alpha N\sum_i f_iP_i
$$
如果某个 expert $i$ 已经获得很多 token，那么 $f_i$ 较大。此时该 expert 的概率质量 $P_i$ 会带来更大损失贡献。梯度下降会倾向于降低热门 expert 的 router probability，从而把 token 推向较少使用的 expert。

所以它的目标不是让每个 token 选择“最语义正确”的 expert，而是防止系统进入少数 expert 过热、其他 expert 饥饿的状态。【字幕†source】

------

## 9. 概念对比与易混淆点

| 易混淆概念                                              | 区别                                                         | 联系                              | 常见误解                           | 正确理解                                                     |
| ------------------------------------------------------- | ------------------------------------------------------------ | --------------------------------- | ---------------------------------- | ------------------------------------------------------------ |
| full softmax attention vs linear attention              | 前者有 softmax 全局归一化，后者通常去掉或替代 softmax        | 都用 Q/K/V 表示 token 交互        | 以为二者完全等价                   | 去掉 softmax 是有损步骤；linear attention 内部 dense/recurrent 才等价 |
| linear attention vs recurrent form                      | 一个是并行矩阵形式，一个是序列递推形式                       | 在线性 attention 内部可以等价转换 | 以为 recurrent form 训练一定慢     | duality 允许训练并行、推理递归                               |
| Mamba-2 vs GDN                                          | Mamba-2 主要加入遗忘 gate；GDN 还加入写入和擦除              | 都增强状态模型表达力              | 以为都是简单 RNN                   | 它们保留某种并行/状态 duality，并服务长上下文效率            |
| DSA vs linear attention                                 | DSA 选择 Top-K token；linear attention 压缩所有历史          | 都降低 full attention 成本        | 以为 DSA 是严格线性时间            | DSA indexer 仍扫描历史，收益来自轻量化和 Top-K               |
| total parameters vs active parameters                   | total 是所有专家总量；active 是每个 token 实际经过的参数     | 都是描述 MoE 规模的重要指标       | 只看 total 判断计算成本            | MoE 要同时看 total、active、routing 和通信                   |
| expert as semantic specialist vs learned routing expert | 人类语义专家有明确领域；MoE expert 未必可解释                | expert 可能学到某些模式           | 以为 expert 一定对应医疗/法律/数学 | router 通常是轻量矩阵打分，不保证语义可解释                  |
| load balancing loss vs main LM loss                     | 前者约束专家使用，后者优化语言建模                           | 共同影响训练                      | 以为 balancing 是可有可无小项      | 没有 balancing 可能 expert collapse                          |
| expert parallelism vs data/model parallelism            | expert parallel 按 expert 切分；data parallel 按 batch 切分；model parallel 按模型层/张量切分 | 可组合使用                        | 以为 MoE 只影响模型质量            | MoE 同时是系统并行设计                                       |
| z-loss vs load balancing loss                           | z-loss 控制 router logits 稳定性；load balancing 控制 expert 使用均衡 | 都作用于 router                   | 混为同一个正则                     | 一个防数值不稳，一个防负载不均                               |
| shared experts vs routed experts                        | shared 始终开启；routed 由 token 选择                        | 都是 FFN expert                   | 以为 shared 越多越好               | 是否有收益依赖规模、数据和训练配方                           |

------

## 10. 学习拓展与发散理解

### 10.1 与大模型训练实践的关系

**与数据准备的关系。** 长上下文模型需要更长文档、更长对话和更复杂的数据拼接策略。attention alternatives 决定模型能否真正利用这些长上下文，而不是只在短窗口内有效。

**与模型训练的关系。** linear attention、Mamba-2、GDN 和 MoE 都改变了训练动态。前者改变 token 间信息流，后者改变 token 到参数的映射方式。MoE 还需要额外的 balancing loss、router stability 正则和系统调度。

**与优化策略的关系。** MoE 的训练目标不只是 next-token prediction，还包含负载均衡、router 稳定性、可能的通信平衡目标。这说明现代大模型训练常常是多目标优化。

**与训练效率的关系。** attention alternatives 降低长上下文成本；MoE 降低“每 token 激活计算量相对于总参数量”的成本。两者共同服务于更高效地扩大模型能力。

**与推理部署的关系。** linear/recurrent attention 可以降低 KV cache 或长上下文 decode 成本；MoE 推理则要处理专家分布、通信、batch 中 token 路由和潜在负载不均。

**与工程成本的关系。** MoE 模型不能只从论文公式理解。真实部署需要考虑 expert placement、all-to-all 通信、块稀疏 kernel、router 精度、token dropping 和服务稳定性。

### 10.2 与其他 lecture 的联系

Lecture 4 接在 Lecture 3 架构与超参数之后，因为本讲讨论的 attention alternatives 和 MoE 都是对 Transformer 架构模块的进一步改造。官方课程表中 Lecture 5 进入 GPU/TPU，Lecture 6 进入 Kernels/Triton，Lecture 7-8 进入 Parallelism，这些后续主题会继续支撑本讲中 FlashAttention、MegaBlocks、expert parallelism 和通信优化等系统问题。

可以把 CS336 的主线理解为：

- Lecture 1-3：先建立语言模型与 Transformer 的基础；
- Lecture 4：讨论当标准结构不够高效时如何改造 attention 和 FFN；
- Lecture 5-8：进入硬件、kernel 和并行系统；
- 后续 scaling、inference、evaluation、data、post-training 继续把架构选择放到完整大模型训练流程中考察。

### 10.3 面试 / 考试 / 作业可能考察角度

| 问题                                       | 考察点                                                       | 回答思路                                                     |
| ------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 为什么标准 attention 是 $O(n^2)$？         | $QK^\top$ 的形状和复杂度                                     | 写出 $QK^\top\in\mathbb{R}^{n\times n}$，解释每个 token 与所有 token 交互 |
| linear attention 如何利用结合律？          | 数学重排与限制条件                                           | 写出 $(QK^\top)V=Q(K^\top V)$，强调去掉 softmax 是有损       |
| dense/recurrent duality 是什么？           | 训练和推理执行方式                                           | 训练用矩阵并行，推理用 $S_t$ 递推                            |
| Mamba-2 的 gate 解决什么问题？             | 状态遗忘                                                     | $\gamma_t$ 控制旧状态保留多少                                |
| GDN 中 $\beta_t$ 和擦除项做什么？          | 写入门和状态更新                                             | $\beta_t$ 控制写入，$I-\beta kk^\top$ 擦除当前 key 方向旧信息 |
| DSA 和 linear attention 有何不同？         | 稀疏选择 vs 状态压缩                                         | DSA 用轻量 indexer 选 Top-K token，再做 full attention       |
| MoE 为什么能增加参数但不等比例增加 FLOPs？ | total vs active parameters                                   | expert 总数大，但每个 token 只激活 Top-K expert              |
| Top-K routing 为什么难训练？               | 离散选择不可微                                               | 只观察被选 expert 的结果，没选 expert 没有 counterfactual feedback |
| 为什么需要 load balancing loss？           | expert collapse                                              | 热门 expert 获得更多 token 和梯度，需正则打散                |
| MoE 的主要系统瓶颈是什么？                 | 通信与负载不均                                               | expert parallelism 带来 all-to-all activation transfer       |
| z-loss 的作用是什么？                      | router softmax 稳定性                                        | 惩罚 log-sum-exp 过大，防止 logits 爆炸                      |
| DeepSeek MoE v1-v2-v3 的演化主线是什么？   | shared/fine-grained、device routing、bias balancing、MLA/MTP | 从架构稀疏化走向训练目标和系统协同设计                       |

------

## 11. 课程复习总结

### 11.1 本讲核心思想

Lecture 4 的主线是：**标准 Transformer 在长上下文和大参数规模下会遇到 attention 与 FFN 两类效率瓶颈。attention 侧可以通过局部化、系统优化、线性化、状态化或稀疏 token 选择来降低成本；FFN 侧可以通过 MoE 让模型拥有更多参数，但每个 token 只激活少数 expert。**

这两条线看似不同，但共享一个思想：**不要让所有 token 与所有对象都发生昂贵交互。**

在 attention alternatives 中，“对象”是历史 token；在 MoE 中，“对象”是 expert 参数。

### 11.2 必须掌握的概念

必须掌握：

- standard attention 的 $O(n^2)$ 成本；
- local/global attention 与 FlashAttention 的作用边界；
- linear attention 的结合律重排；
- 去掉 softmax 是有损替代；
- recurrent linear attention 的 $S_t$ 状态；
- dense/recurrent duality；
- Mamba-2 的 $\gamma_t$ 遗忘门；
- Gated Delta Net 的 $\beta_t$ 写入门和擦除项；
- DSA 的轻量 indexer 与 Top-K token selection；
- MoE 的 total parameters 与 active parameters；
- token-choice Top-K routing；
- shared experts 与 fine-grained experts；
- expert collapse 与 load balancing loss；
- expert parallelism 与通信开销；
- MegaBlocks / block-sparse computation；
- router z-loss；
- MoE fine-tuning 过拟合；
- upcycling；
- DeepSeek MoE v1-v2-v3、MLA、MTP。

### 11.3 必须理解的公式

| 公式                                                         | 解决的问题                            |
| ------------------------------------------------------------ | ------------------------------------- |
| $\rho(QK^\top)V$                                             | 标准 attention 如何计算               |
| $(QK^\top)V=Q(K^\top V)$                                     | linear attention 如何通过结合律降成本 |
| $S_t=S_{t-1}+k_tv_t^\top$                                    | 如何把 linear attention 写成状态递推  |
| $S_t=\gamma_tS_{t-1}+k_tv_t^\top$                            | Mamba-2 如何加入遗忘                  |
| $S_t=\gamma_t(I-\beta_tk_tk_t^\top)S_{t-1}+\beta_tk_tv_t^\top$ | GDN 如何先擦除再写入                  |
| $I_{t,s}=\sum_j w_{t,j}^I\mathrm{ReLU}((q_{t,j}^I)^\top k_s^I)$ | DSA 如何给历史 token 打索引分         |
| $h_t^l=\sum_i g_{i,t}\mathrm{FFN}_i(u_t^l)+u_t^l$            | MoE 如何只激活少数 expert             |
| $\alpha N\sum_i f_iP_i$                                      | 如何防止 expert 使用不均              |
| $\frac1B\sum_i(\log\sum_j e^{x_j^i})^2$                      | 如何稳定 router softmax               |

### 11.4 必须看懂的伪代码

本讲没有 `lecture_04.py`，但必须能看懂这些伪代码模式：

```
scores = Q @ K.T
weights = softmax(scores)
out = weights @ V
```

它对应标准 attention，也是 $O(n^2)$ 成本来源。

```
state = K.T @ V
out = Q @ state
```

它对应 linear attention 的结合律重排。

```
S += outer(k_t, v_t)
y_t = q_t @ S
```

它对应 recurrent form，解释推理时的固定状态。

```
active = topk(router_scores)
out = sum(g_i * expert_i(x) for i in active) + x
```

它对应 MoE 的 Top-K expert routing。

```
loss += alpha * N * dot(f, P)
loss += mean(logsumexp(router_logits) ** 2)
```

它对应 MoE 的负载均衡和 router 稳定性。

### 11.5 一句话总结

**Lecture 4 讲的是现代大模型如何用“状态压缩”和“稀疏选择”突破标准 Transformer 的成本瓶颈：attention 侧通过线性化、门控状态和 Top-K token selection 降低长上下文代价，FFN 侧通过 MoE 用少量 active parameters 激活大量 total parameters，从而提升模型容量与效率。**