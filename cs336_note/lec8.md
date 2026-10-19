# Lecture 8 课程报告：并行化基础——从单 GPU 瓶颈到 3D/4D 大模型训练系统

## 0. 资料来源与改写说明

本报告基于你上传的原始课程报告 `lec8.md` 和对应讲义 `lecture_08.pdf` 改写。原报告的主题是 “大语言模型训练中的并行化策略与系统权衡”，内容顺序为：多 GPU / 多节点训练动机、通信原语与硬件拓扑、数据并行与 ZeRO/FSDP、Pipeline Parallel、Tensor Parallel、Sequence/Context Parallel、Expert Parallel、3D/4D 并行以及真实模型训练案例。

本次改写保留原报告的课程主线、核心公式、图表解释、案例和引用格式。PDF 讲义标题为 **Lecture 8 — Parallelism Basics**，课程目标是理解巨型模型训练的系统复杂性、不同并行范式以及为什么实际训练中经常同时使用多种并行方法。

当前会话中可直接读取的资料包括 `lec8.md` 和 `lecture_08.pdf`。原报告中提到其主线来源包括字幕文件，但本次会话没有单独上传字幕文件。因此，本报告会**保留原报告已有的 `【字幕†source】` 标注**，但不会新增未核验的字幕引用。新增的解释、类比和学习化整理会明确标注为“延伸理解”“补充说明”“工程直觉”或“学习提示”。

我也核对了 Stanford CS336 的 `stanford-cs336/lectures` 仓库：仓库说明可执行讲义命名为 `lecture_XX.py`，非可执行讲义命名为 `lecture_XX.pdf`；文件列表中 Lecture 8 对应 `lecture_08.pdf`，相邻位置有 `lecture_06.py`、`lecture_07.py`、`lecture_09.pdf`，但没有显示 `lecture_08.py`。因此，本讲按当前仓库状态属于 PDF 讲义型 lecture，而不是官方 Python 可执行讲义。

本报告中的引用含义如下：

| 引用形式                         | 含义                                           |
| -------------------------------- | ---------------------------------------------- |
| `【PDF†source】`                 | 原报告或本次改写中对应 `lecture_08.pdf` 的内容 |
| `【字幕†source】`                | 原报告中已有的字幕来源标注，本次保留但不新增   |
| `【GitHub†source】`              | 原报告中已有的 GitHub 来源标注                 |
| `/`                              | 本次回答中对上传文件或网页核对结果的系统引用   |
| “延伸理解 / 工程直觉 / 学习提示” | 为帮助学习而加入的解释，不伪装成课程原文       |

------

## 1. 学习导读

### 1.1 本讲要解决的核心问题

本讲最核心的问题是：

**当单个 GPU 既放不下模型、又算不完训练任务时，如何把大语言模型训练拆到多个 GPU、多个节点甚至整个数据中心上，并且在显存、计算、通信和 batch size 之间取得平衡。**

换句话说，Lecture 8 不只是介绍几个并行名词，而是在回答一个更工程化的问题：**如何把一个“理论上能训练”的模型，真正变成一个“在硬件集群上能高效训练”的系统。**

PDF 讲义开头明确把本讲目标概括为：理解训练巨型模型的系统复杂性，理解不同并行范式以及为什么要同时使用多种并行方法，并观察大规模训练运行通常长什么样。

### 1.2 本讲在 CS336 课程中的位置

CS336 的主线是从零理解语言模型训练。前面的内容更多关注模型、优化、数据或基础训练过程；本讲进入系统层面，讨论当模型规模继续扩大时，训练不再只是“写一个 loss 然后跑 optimizer”，而是必须处理：

- 模型参数和优化器状态是否放得下；
- activation 在 forward/backward 中是否爆显存；
- GPU 之间如何同步梯度、参数或激活；
- 哪些通信可以重叠在计算之下；
- 哪些并行方式适合节点内高速互联，哪些适合节点间慢速互联；
- 真实大模型训练为什么常见 3D/4D 并行组合。

**延伸理解：**
 这也是现代大模型训练和普通深度学习训练的重要分水岭。普通训练常常默认“模型能放进一张卡或一台机器”；大模型训练则首先要解决“怎么把训练任务拆开，同时不被通信拖死”。

### 1.3 本讲知识结构

| 模块                                     | 解决的问题                                                   | 核心概念                                                     | 与后续内容的关系                                   |
| ---------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | -------------------------------------------------- |
| 网络与通信基础                           | 为什么单 GPU 不够？多 GPU 怎么通信？                         | datacenter as compute unit、intra-node / inter-node、collective primitives | 后续所有并行策略都要用这些通信原语分析成本         |
| 数据并行与 ZeRO/FSDP                     | 如何切 batch？如何减少复制的 optimizer state / gradient / parameter？ | DDP、ZeRO-1/2/3、FSDP、all-gather、reduce-scatter            | 解决参数相关显存，但不能完全解决 activation memory |
| Pipeline Parallel                        | 如何按层切模型？                                             | pipeline stage、microbatch、bubble、zero-bubble              | 适合跨节点慢链路，但依赖足够 microbatch            |
| Tensor Parallel                          | 如何按矩阵宽度切模型？                                       | column-wise / row-wise partition、activation all-reduce      | 适合节点内高速互联，常用于 Transformer 大矩阵      |
| Activation / Sequence / Context Parallel | activation 太大怎么办？长上下文怎么切？                      | activation memory、sequence parallel、context parallel、ring attention | 处理长序列和 activation/KV 显存压力                |
| Expert Parallel                          | MoE 模型如何并行？                                           | expert sharding、router、all-to-all dispatch                 | MoE 训练的核心并行方式                             |
| 3D/4D 并行与真实案例                     | 多种并行如何组合？                                           | DP/TP/PP/EP/CP/FSDP 组合、roofline 直觉                      | 连接真实大模型训练配置                             |

PDF 讲义同样把当天课程组织为三部分：LLM 网络基础、不同形式的并行 LLM 训练、以及使用并行化扩展和训练大模型。

### 1.4 学习本讲前需要知道什么

学习本讲最好具备以下前置知识：

| 前置知识                  | 为什么需要                                                   |
| ------------------------- | ------------------------------------------------------------ |
| 梯度下降与 batch training | 数据并行本质上是在多个设备上计算 batch shard 的梯度再同步    |
| 反向传播                  | ZeRO-2/FSDP、pipeline backward、activation memory 都依赖 backward 生命周期 |
| Transformer block 结构    | Tensor Parallel 需要理解 QKV、attention output、MLP up/down projection |
| GPU 显存组成              | 参数、梯度、optimizer state、activation、KV cache 分别占用不同内存 |
| 分布式通信基本概念        | all-reduce、all-gather、reduce-scatter 是本讲的共同语言      |
| 大模型训练工程直觉        | 需要同时考虑 FLOPs、带宽、延迟、batch size、并行粒度         |

------

## 2. 第一部分：并行化为什么是大语言模型训练的基本前提

本节要解决的问题是：**为什么训练大语言模型不能只靠单 GPU？为什么硬件网络拓扑会直接决定并行算法的选择？**

### 2.1 问题引入：单 GPU 的两个硬瓶颈

课程从两个限制开始：**算力限制**和**显存限制**。

第一个限制是算力。单个 GPU 的 FLOPs 有上限，而大模型训练需要的总计算量会随着参数量、数据量和训练 token 数迅速增长。PDF 第 4 页用 “Limits to GPU-based scaling – compute” 说明单 GPU scaling 存在计算上限。

第二个限制是显存。模型参数越来越大，单张 GPU 往往放不下完整参数，更不用说训练时还需要存梯度、optimizer state 和 activation。PDF 第 5 页明确指出：模型变得很大，单 GPU 放不下许多大型模型。

因此，并行化不是额外优化，而是必要条件。课程希望多机多卡训练同时获得两种扩展：
$$
\text{更多 GPU} \Rightarrow \text{更多可用显存} + \text{更多可用计算}
$$
也就是：

- **线性显存扩展**：可训练最大模型参数量随 GPU 数接近线性增长；
- **线性计算扩展**：可用训练 FLOPs 随 GPU 数接近线性增长。

PDF 的 Part 1 recap 把新的计算单元称为 **datacenter**，并明确提出多机扩展希望获得 linear memory scaling 和 linear compute scaling【PDF†source】。

**学习提示：**
 理解本讲时不要把 GPU 看成单独设备，而要把整个数据中心看成一个“巨型虚拟加速器”。并行化的任务，就是让这个虚拟加速器尽可能像一台大机器一样工作。

### 2.2 概念定义：intra-node 与 inter-node

并行训练首先要区分两类通信环境：

| 概念                   | 直觉解释                              | 典型硬件                         | 对并行策略的影响                         |
| ---------------------- | ------------------------------------- | -------------------------------- | ---------------------------------------- |
| Intra-node parallelism | 同一台服务器或同一 GPU box 内部的并行 | NVLink、NVSwitch、PCIe           | 带宽高、延迟低，适合通信频繁的 TP/EP     |
| Inter-node parallelism | 多台服务器之间的并行                  | InfiniBand、高速以太网、交换网络 | 相对更慢，适合通信较少或点对点的 PP/FSDP |

PDF 第 6 页展示了多 GPU / 多机器并行：节点内依靠高速互联，节点间也需要高速互联，但两者成本和性能不同【PDF†source】。

**工程直觉：**
 并行算法不是抽象数学游戏。一个算法在 8 张 NVLink 互联 GPU 内可能很快，跨 64 台机器后可能变慢。原因不是算法“错了”，而是通信模式和硬件拓扑不匹配。

### 2.3 方法与原理：集体通信原语

课程接下来介绍 collective communication primitives。它们是分布式训练的基本“动词”。

| 通信原语       | 输入                   | 输出                         | 训练中的典型用途           |
| -------------- | ---------------------- | ---------------------------- | -------------------------- |
| Broadcast      | 一个 rank 有张量       | 所有 rank 获得同一张量       | 分发参数、配置或初始状态   |
| Reduce         | 所有 rank 有张量       | 一个 rank 获得归约结果       | 汇总梯度或统计量           |
| All-reduce     | 所有 rank 有张量       | 所有 rank 获得归约结果       | DDP 中同步梯度             |
| All-gather     | 每个 rank 有一个 shard | 每个 rank 获得完整拼接张量   | FSDP 中临时拼出完整参数    |
| Reduce-scatter | 每个 rank 有完整张量   | 每个 rank 获得归约后的一部分 | ZeRO/FSDP 中分发梯度 shard |

PDF 第 7 页列出了这些基本 collective；第 8 页进一步强调 all-reduce 可以用 reduce-scatter 和 all-gather 两步实现。

本讲非常关键的恒等式是：
$$
\text{AllReduce}
\equiv
\text{ReduceScatter}
+
\text{AllGather}
$$
直觉上看，all-reduce 要让每个设备都拿到所有设备张量求和后的完整结果。它可以分两步做：

1. **Reduce-scatter**：先把不同 shard 的归约结果分给不同 rank；
2. **All-gather**：再把每个 rank 的归约 shard 收集回所有 rank。

这个等价关系是 ZeRO-1 看起来“免费”节省 optimizer state 显存的基础：原本 DDP 做一次 all-reduce；ZeRO-1 改成 reduce-scatter + all-gather，在 bandwidth-limited regime 中通信成本相近，但内存布局发生了变化【PDF†source】。

### 2.4 硬件拓扑：TPU mesh、GPU all-to-all/tree 与 MoE 需求

课程随后比较 TPU 和 GPU 的通信设计。

TPU 经典设计偏向 **toroidal mesh**。每个芯片只连接邻近芯片，边界可以 wrap around。优点是拓扑规则、扩展成本低、适合结构化通信。GPU 集群更常见的是节点内高速互联加跨节点交换网络，整体更接近 all-to-all 或 tree/fat-tree 风格，适合更不规则的通信。PDF 第 9–10 页对比了 TPU networking 的 toroidal mesh 和 GPU networking 的 all-to-all，并指出 mesh 更便宜、适合结构化通信，而 tree/A2A 更适合 expert parallel 这类不规则通信【PDF†source】。

这会影响策略选择：

| 模型/通信类型                            | 更适合的拓扑直觉                 |
| ---------------------------------------- | -------------------------------- |
| Dense Transformer 中规则矩阵切分         | mesh 类拓扑较友好                |
| MoE 中 token-to-expert 路由              | all-to-all / tree 类拓扑更灵活   |
| 频繁 all-reduce 的 Tensor Parallel       | 需要低延迟、高带宽互联           |
| 点对点传 activation 的 Pipeline Parallel | 可以放在较慢的 inter-node 链路上 |

课程还提到 TPU8i/t 的拓扑变化：TPU8i 更接近 tree-style topology，TPU8t 的 scale-out networking 使用 Virgo switched networks【PDF†source】。原报告将其解释为工作负载推动硬件演化，尤其是现代 MoE 模型需要更多 all-to-all 能力【字幕†source】。

最后，课程用 Huawei Ascend 910 / CloudMatrix 384 作为硬件权衡例子：通过大规模互联弥补单芯片性能不足，但代价是功耗和系统成本上升【字幕†source】。这个例子说明，通信瓶颈可以用更多硬件“硬顶”，但并不免费。

### 2.5 本节学习提示

本节最重要的结论是：**并行策略必须和通信拓扑一起理解。**

容易误解的地方有三个：

1. 不要以为 GPU 数越多一定越快。通信可能抵消计算收益。
2. 不要把 all-reduce / all-gather / reduce-scatter 当作实现细节。它们决定了算法的带宽成本和显存布局。
3. 不要脱离硬件谈 TP、PP、EP。TP/EP 通信密集，PP 通信更稀疏，因此适合的网络层级不同。

一句话总结：**大模型训练的基本单位不再是单 GPU，而是由计算、显存和通信网络共同构成的数据中心级训练系统。**

------

## 3. 第二部分：数据并行、ZeRO 与 FSDP

本节要解决的问题是：**如果最自然的做法是把 batch 切给多张 GPU，那么为什么还不够？ZeRO/FSDP 如何减少数据并行中的显存冗余？**

### 3.1 问题引入：朴素数据并行为什么最自然

数据并行是最直观的并行方式：每个 GPU 都有一份完整模型，但处理不同训练样本。每个 GPU 算出自己 batch shard 的梯度，然后所有 GPU 同步梯度，得到等价于大 batch 的更新。

PDF 第 15 页从 naive SGD 开始，给出更新公式，并说明 naive data parallelism 把大小为 $B$ 的 batch 切给 $M$ 台机器，再交换梯度进行同步。

### 3.2 公式与推导：朴素数据并行的 SGD 更新

课程中的 SGD 更新可以写成：
$$
\theta_{t+1}
=
\theta_t
-
\eta
\sum_{i=1}^{B}
\nabla f(x_i)
$$
符号解释：

| 符号            | 含义                        |
| --------------- | --------------------------- |
| $\theta_t$      | 第 $t$ 步的模型参数         |
| $\theta_{t+1}$  | 更新后的参数                |
| $\eta$          | 学习率                      |
| $B$             | global batch size           |
| $x_i$           | batch 中第 $i$ 个训练样本   |
| $\nabla f(x_i)$ | 样本 $x_i$ 对损失函数的梯度 |

数据并行时，$M$ 个 worker 各自计算约 $B/M$ 个样本的梯度。然后通过 all-reduce 把梯度加起来，使每张 GPU 都得到相同的总梯度。

直觉上，这相当于：
$$
\sum_{i=1}^{B}\nabla f(x_i)
=
\sum_{m=1}^{M}
\sum_{x_i \in \mathcal{B}_m}
\nabla f(x_i)
$$
每个 worker 只负责右侧的一小部分，最后通过通信把它们合并。

PDF 对朴素数据并行的总结是：

- compute scaling：每个 GPU 得到 $B/M$ 个样本；
- communication overhead：每个 batch 传输 $2 \times \#params$；
- memory scaling：没有，每个 GPU 至少需要完整参数。

### 3.3 概念定义：训练显存账本

朴素数据并行最大的问题不是计算，而是显存。每张 GPU 不只复制参数，还复制梯度和 optimizer state。

PDF 第 17 页指出，根据精度设置，训练可能需要约 5 份 weight-like storage，约 16 bytes per parameter。

| 内容                   | 典型字节数 / 参数 | 为什么需要                      |
| ---------------------- | ----------------- | ------------------------------- |
| BF16 / FP16 parameters | 2                 | forward/backward 使用的模型权重 |
| BF16 / FP16 gradients  | 2                 | 保存反向传播梯度                |
| FP32 master weights    | 4                 | 用更高精度累积更新              |
| Adam first moment      | 4 或 2            | Adam 的一阶动量 $m_t$           |
| Adam second moment     | 4 或 2            | Adam 的二阶动量 $v_t$           |

这里的 Adam moment 和 master weights 通常被称为 **optimizer state**。它们在大模型训练中经常比“参数本身”更让人头疼，因为参数 2 bytes/param，但训练状态可能让总内存膨胀到 12–16 bytes/param。

**延伸理解：**
 很多初学者会问：“一个 7B 参数模型 BF16 不是只要 14GB 吗？”这是推理时的直觉。训练时还要存梯度、优化器状态、activation，显存需求会成倍增长。

### 3.4 ZeRO 的核心思想：切分昂贵状态

ZeRO 的目标是减少数据并行中的状态冗余。PDF 第 18 页把 ZeRO 的核心思想概括为：**split up the expensive parts (state) and use the reduce-scatter equivalence**【PDF†source】。

ZeRO 分三个阶段：

| 阶段          | 切分对象                      | 仍然复制的对象 | 主要收益                 | 主要复杂度                   |
| ------------- | ----------------------------- | -------------- | ------------------------ | ---------------------------- |
| ZeRO-1        | optimizer state               | 参数、梯度     | 大幅减少 Adam state 显存 | 较低                         |
| ZeRO-2        | optimizer state + gradients   | 参数           | 进一步减少梯度显存       | backward 中增量 reduce/free  |
| ZeRO-3 / FSDP | 参数 + 梯度 + optimizer state | 无完整状态常驻 | 参数相关显存近似线性下降 | 按层 all-gather/free/overlap |

### 3.5 ZeRO-1：只切 optimizer state

ZeRO-1 的直觉是：每张 GPU 不必保存所有参数的 optimizer state，只要保存自己负责更新的那一部分即可。

流程如下：

1. 每个 GPU 持有完整参数；
2. 每个 GPU 对自己的 batch shard 计算完整梯度；
3. 用 reduce-scatter 把梯度归约后分发到负责对应参数 shard 的 GPU；
4. 每个 GPU 用本地 optimizer state 更新自己的参数 shard；
5. 用 all-gather 把更新后的参数 shard 拼回所有 GPU。

PDF 第 19–21 页说明 ZeRO-1 把 first/second moments 分片，每个 worker 负责更新一个参数 slice；并对比了 Naive DDP 与 ZeRO-1：二者通信成本都是 $2 \times \#params$，但 ZeRO-1 的内存从 $(4+K)\#params$ 降到 $(4+K/N_{gpu})\#params$。

因此，ZeRO-1 的学习重点是：
$$
\text{同样的通信量}
\quad+\quad
\text{不同的状态布局}
\quad=\quad
\text{显存收益}
$$
这就是原报告中说的 “free memory wins”【PDF†source】。

### 3.6 ZeRO-2：再切 gradients

ZeRO-2 进一步切分梯度。困难在于：每个 worker 仍然需要通过完整模型做 backward，因此从数学上每个 worker 都会“产生”完整梯度；但系统上不能把完整梯度长期 materialize 在显存里。

解决方法是生命周期管理：

1. backward 按计算图逐层向后执行；
2. 某层梯度一算出来，立即 reduce / scatter 到负责该参数 shard 的 GPU；
3. 一旦该梯度不再被 backward graph 使用，立即释放；
4. 最后每个 worker 只保留自己负责的 gradient shard 和 optimizer state。

PDF 第 22–23 页强调 ZeRO-2 的复杂点：不能实例化完整 gradient vector，但每个 worker 又必须计算完整梯度，因为它仍然是 data parallel【PDF†source】。

**工程直觉：**
 ZeRO-2 的难点不是公式，而是系统实现。它需要在 autograd 中插入 hook，及时通信、及时释放，避免梯度峰值撑爆显存。

### 3.7 ZeRO-3 / FSDP：参数也切分

ZeRO-3，也就是 FSDP，把 parameters、gradients、optimizer states 全部分片。PDF 第 24 页把它概括为 **shard everything**，包括 parameters，并在计算图中按需请求参数【PDF†source】。

一个简化的 FSDP 流程是：

1. 某层 forward 前，对该层参数做 all-gather；
2. 用完整参数计算 forward；
3. 计算完成后释放完整参数，只保留本地 shard；
4. backward 到该层时，再 all-gather 参数；
5. 计算 backward；
6. 对梯度 reduce-scatter，使每张 GPU 只保留自己的 gradient shard；
7. 释放临时完整参数；
8. 用本地 optimizer state 更新本地参数 shard。

FSDP 的通信成本近似为：
$$
C_{\text{FSDP}}
\approx
2 \times \text{AllGather}(\#params)
+
1 \times \text{ReduceScatter}(\#params)
\approx
3 \times \#params
$$
PDF 第 25 页明确写出 ZeRO-3 / FSDP 的通信成本为 2 次 all-gather 和 1 次 reduce-scatter。

与 DDP 的 $2P$ 通信量相比，FSDP 约为 $3P$，即大约 1.5 倍。但课程强调，通过参数预取、及时释放和通信计算重叠，实际 wall-clock overhead 可以被部分隐藏。PDF 第 26 页展示 FSDP timeline：参数和梯度被请求、发送后立即释放，all-gather 可以和 forward 同时发生【PDF†source】。

**容易误解的点：**
 FSDP 不是 Pipeline Parallel。FSDP 中每张 GPU 仍然执行完整模型，只是每层参数在需要时临时拼出来；Pipeline Parallel 则是真的把不同层放在不同 GPU 上，设备之间传 activation。

### 3.8 ZeRO 的实际容量收益：8×A100 例子

PDF 第 28 页给出一个非常实用的容量估算：假设 pure BF16 training，除 master weights 外都用 BF16，并用 Kahan summation，约 12 bytes per parameter；在 8×A100 80GB 上，不同 ZeRO 阶段可容纳模型规模如下【PDF†source】：

| 配置     | 每参数字节估算 | 最大模型参数量 |
| -------- | -------------- | -------------- |
| Baseline | 12             | 约 6.66B       |
| ZeRO-1   | 5              | 约 16B         |
| ZeRO-2   | $2 + 10/8$     | 约 24.62B      |
| ZeRO-3   | $12/8$         | 约 53.33B      |

这张表最重要的学习价值是建立显存估算直觉：**ZeRO-3/FSDP 能让参数相关显存随 GPU 数近似线性下降，但它不直接解决 activation memory。**

### 3.9 数据并行仍然剩下的问题

数据并行和 ZeRO 之后，课程马上指出两个剩余限制。

第一，数据并行受 batch size 限制。PDF 第 29 页写明，data parallel 下机器数必须小于 batch size，并且 batch size 存在 diminishing returns。

第二，ZeRO-3 不减少 activation memory。PDF 第 30 页明确说，ZeRO stages 1 and 2 don’t let you scale memory；ZeRO stage 3 虽然原则上很好，但 does not reduce activation memory。

这就引出下一部分：**模型并行**。如果参数状态切完还不够，就要开始切模型计算本身。

### 3.10 本节学习提示

本节复习时抓住三句话：

1. **DDP 简单，但每张 GPU 复制完整训练状态。**
2. **ZeRO-1/2/3 依次切 optimizer state、gradients、parameters。**
3. **FSDP 解决参数相关显存，但不直接解决 activation memory，也不解决 batch size 扩展的边际收益问题。**

------

## 4. 第三部分：模型并行——Pipeline Parallel 与 Tensor Parallel

本节要解决的问题是：**如果数据并行仍然放不下模型或扩展不够好，如何把模型本身切开？**

### 4.1 概念定义：模型并行与 FSDP 的区别

PDF 第 31 页定义 model parallelism：它也会把参数跨 GPU 切分，但与 ZeRO-3 不同的是，**model parallelism 通信的是 activations，而 ZeRO-3/FSDP 通信的是 parameters**【PDF†source】。

可以这样区分：

| 方法              | 参数在哪里                        | 每个 GPU 是否执行完整模型 | 主要通信对象                            |
| ----------------- | --------------------------------- | ------------------------- | --------------------------------------- |
| FSDP / ZeRO-3     | 参数分片存储，按层临时 all-gather | 是                        | 参数、梯度 shard                        |
| Pipeline Parallel | 不同层放在不同 GPU                | 否                        | 层间 activation / activation gradients  |
| Tensor Parallel   | 同一层矩阵切到不同 GPU            | 否                        | activation partial results / all-reduce |
| Expert Parallel   | 不同 expert 放在不同 GPU          | 否                        | routed token activations                |

**直觉类比：**

- FSDP 像“每个人都能临时借到全套工具，但平时只保管一部分工具”；
- Pipeline Parallel 像“流水线工厂，每个人只负责几个工序”；
- Tensor Parallel 像“同一道工序太大，多个人一起切开做”；
- Expert Parallel 像“不同专家坐在不同办公室，token 被派到对应专家那里”。

------

### 4.2 Pipeline Parallel：沿深度切模型

#### 4.2.1 基本思想

Pipeline Parallel 又叫 layer-wise parallel。它把模型按层切开，不同 GPU 负责不同层。前向传播时，上游 GPU 把 activation 传给下游 GPU；反向传播时，下游 GPU 把 partial gradients 传回上游 GPU。

PDF 第 32 页写明：layer-wise parallel cuts up layers, assigns some subset to GPUs，activations and partial gradients are passed back and forth【PDF†source】。

例如 4 张 GPU：

| GPU  | 负责部分                  |
| ---- | ------------------------- |
| GPU0 | 前几层 Transformer blocks |
| GPU1 | 中间一段 blocks           |
| GPU2 | 后面一段 blocks           |
| GPU3 | 最后几层 + 输出头         |

这种方法直接节省参数显存，因为每张 GPU 只存一部分层。但朴素实现会带来严重空闲。

#### 4.2.2 Pipeline bubble：为什么朴素 layer-wise parallel 很差

如果每次只送一个 batch：

1. GPU0 计算第一段；
2. GPU0 把 activation 传给 GPU1；
3. GPU0 空闲，GPU1 开始算；
4. GPU1 算完传给 GPU2；
5. GPU1 空闲；
6. backward 也类似。

结果是大部分时间只有一个 GPU 在工作。PDF 第 33 页写道：with $n$ GPUs, each GPU is active $1/n$ of the time【PDF†source】。

这段空闲时间叫 **pipeline bubble**。

**直觉类比：**
 如果工厂流水线一次只生产一件产品，那么第一道工序完成后就没事做，直到整件产品完成再开始下一件。这样的流水线利用率很低。

#### 4.2.3 Micro-batching：用 batch 填满流水线

解决 bubble 的方法是把 global batch 切成多个 micro-batches。第一个 microbatch 离开 GPU0 后，GPU0 立刻开始第二个 microbatch；这样多个 stage 可以同时工作。

PDF 第 34 页给出 bubble time 与 useful compute 的比例：
$$
\text{bubble ratio}
=
\frac{n_{\text{stages}} - 1}{n_{\text{micro}}}
$$
其中：

| 符号                | 含义              |
| ------------------- | ----------------- |
| $n_{\text{stages}}$ | pipeline stage 数 |
| $n_{\text{micro}}$  | microbatch 数     |

这个公式说明：

- stage 越多，pipeline 越深，bubble 越大；
- microbatch 越多，流水线越容易填满，bubble 越小；
- 因此 Pipeline Parallel 需要足够大的 batch 或 gradient accumulation。

PDF 第 36 页也展示 pipeline performance 对 batch size 高度敏感：小 batch 下 pipeline-parallel size 增大时性能下降更明显，而大 batch 能更好隐藏 bubble【PDF†source】。

#### 4.2.4 为什么仍然使用 Pipeline Parallel

Pipeline 看起来麻烦，但课程强调它仍然常用，原因有两个。

第一，它节省显存。每个 stage 只存一部分层的参数和局部 activation buffer。

第二，它有良好的通信性质。Pipeline 每次主要在相邻 stage 之间传 activation，通信量通常是：
$$
C_{\text{pipeline}}
\sim
bsh
$$
其中：

| 符号 | 含义             |
| ---- | ---------------- |
| $b$  | microbatch size  |
| $s$  | sequence length  |
| $h$  | hidden dimension |

PDF 第 35 页强调，pipeline communication depends only on activations $(b \times s \times h)$，并且是 point-to-point，因此通常适合放在较慢的 inter-node 链路上。

#### 4.2.5 Zero-bubble pipelining：把 backward 拆成 B 和 W

课程还介绍了更高级的 zero-bubble pipelining。核心观察是 backward 可以拆成两类工作：

1. **B：backpropagating activations**
    也就是把梯度信号继续传给前一个 stage。这会影响上游 stage 能否继续 backward，因此优先级高。
2. **W：computing weight gradients**
    也就是计算当前层参数梯度。这部分只影响本地参数更新，可以延后到空隙中做。

PDF 第 38 页明确把 backward 拆成 backpropagating activations $(z, x)$ 和 computing weight gradients $(W)$，并指出第二部分可以 “done whenever”【PDF†source】。

**学习重点：**
 Zero-bubble 的关键不是减少总 FLOPs，而是重新调度 backward 中不同任务的时间，使原本的空泡被 weight-gradient computation 填满。

------

### 4.3 Tensor Parallel：沿宽度切矩阵

#### 4.3.1 基本思想

Pipeline Parallel 沿模型深度切层；Tensor Parallel 沿模型宽度切矩阵。

课程从矩阵乘法观察开始：一个大矩阵乘法可以拆成多个小矩阵乘法，然后把 partial sums 加回来。PDF 第 39 页概括为：decompose into submatrices, add partial sums【PDF†source】。

以 Transformer MLP 为例，串行计算为：
$$
Y = \text{GeLU}(XA),
\quad
Z = YB
$$
Tensor Parallel 可以把第一个矩阵 $A$ 按列切，把第二个矩阵 $B$ 按行切：
$$
A = [A_1, A_2],
\quad
B =
\begin{bmatrix}
B_1 \\
B_2
\end{bmatrix}
$$
于是：
$$
Y_1 = \text{GeLU}(XA_1),
\quad
Y_2 = \text{GeLU}(XA_2)
$$
再分别计算：
$$
Z_1 = Y_1B_1,
\quad
Z_2 = Y_2B_2
$$
最后聚合：
$$
Z = Z_1 + Z_2
$$
这个求和通常需要 all-reduce。PDF 第 40 页展示了 $A=[A_1,A_2]$、$B=[B_1;B_2]$ 的结构，并说明 forward 中 $f$ 是 identity，$g$ 是 all-reduce；backward 中二者反过来【PDF†source】。

#### 4.3.2 Transformer 中的 row-wise 与 column-wise 切分

PDF 第 41 页总结了 Transformer block 中常见切分方式【PDF†source】：

| 模块                        | 切分方式    | 为什么这样切                               |
| --------------------------- | ----------- | ------------------------------------------ |
| QKV projection              | Column-wise | attention head / hidden channel 可分开计算 |
| MLP up-projection           | Column-wise | expansion 维度可切分                       |
| Attention output projection | Row-wise    | 各分片输出需要加回 hidden 维               |
| MLP down-projection         | Row-wise    | expansion 分片投回 hidden                  |
| LayerNorm / Router / 小算子 | Replicated  | 参数少，切分收益小，通信成本不值得         |

**延伸理解：**
 Column-wise 往往让每张 GPU 得到输出的一部分；row-wise 往往让每张 GPU 产生 partial sum，最后需要聚合。这种 column/row 配对是 Megatron-style tensor parallel 的重要设计。

#### 4.3.3 Tensor Parallel 的优缺点

Tensor Parallel 的优点包括：

- 没有 pipeline bubble；
- 不依赖很大的 batch size；
- 可以局部包装大矩阵层；
- 能减少 TP-sharded weights 和相关 matmul activation；
- 对 Transformer 的大矩阵乘法很自然。

但缺点也很明显：通信非常频繁。PDF 第 43 页对比 Pipeline 和 Tensor 的通信量：
$$
C_{\text{pipeline}}
\sim
bsh
$$
Pipeline 是每个 microbatch 的 point-to-point activation traffic；Tensor 则是每层 activation-sized all-reduce。PDF 因此总结：Use tensor parallel whenever we have low-latency, high-bandwidth interconnects。

这解释了为什么 GPU 上 TP 往往限制在节点内，例如 8 张 NVLink GPU；一旦跨节点，通信延迟和带宽压力会显著增加【字幕†source】。

### 4.4 本节学习提示

Pipeline 和 Tensor 的对比可以这样记：

| 维度       | Pipeline Parallel    | Tensor Parallel                      |
| ---------- | -------------------- | ------------------------------------ |
| 切分方向   | 深度：按层切         | 宽度：按矩阵/hidden 维切             |
| 通信对象   | 相邻层 activation    | 每层 partial activation / all-reduce |
| 通信模式   | point-to-point       | collective，通常 all-reduce          |
| 主要问题   | bubble               | 通信频繁                             |
| batch 需求 | 需要足够 microbatch  | 不强依赖大 batch                     |
| 适合链路   | 较慢 inter-node 也可 | 低延迟高带宽 intra-node              |

一句话总结：**Pipeline 用时间调度换显存，Tensor 用高带宽通信换更细粒度的层内并行。**

------

## 5. 第四部分：Activation Memory、Sequence/Context Parallel、Expert Parallel 与 3D/4D 组合

本节要解决的问题是：**参数切分之后，activation、长上下文、MoE 专家和多种并行策略如何继续影响训练系统？**

### 5.1 Activation memory：显存是动态的

课程提醒我们：显存不只是参数、梯度和 optimizer state。Forward 过程中保存的 activations 也可能非常大。

PDF 第 44 页展示了一张 PyTorch profiler 风格的显存时间曲线：参数和 optimizer state 像底座，forward 过程中 activation 形成峰值，backward 过程中 gradient 也形成峰值【PDF†source】。PDF 第 45 页进一步指出，到目前为止主要讨论的是 parameter memory，但 activation memory 也可能很大。

这对 ZeRO-3/FSDP 很关键：即使参数、梯度、optimizer state 都切了，如果 activation 不切，仍可能放不下。

### 5.2 每层 activation memory 公式

PDF 第 46 页给出存储所有 activation 时的每层激活内存估算：
$$
M_{\text{act}}
=
sbh
\left(
34 + 5\frac{as}{h}
\right)
$$
符号解释：

| 符号            | 含义                                             |
| --------------- | ------------------------------------------------ |
| $s$             | sequence length                                  |
| $b$             | microbatch size                                  |
| $h$             | hidden dimension                                 |
| $a$             | attention heads 数                               |
| $5\frac{as}{h}$ | quadratic attention 相关项，包括 dropout         |
| $sbh$           | batch × sequence × hidden 的基础 activation 规模 |

PDF 同页说明，$5as/h$ 项来自 quadratic attention terms including dropout，并且类似 FlashAttention，可以通过 recomputation 去掉这部分存储。

**直觉解释：**
 $sbh$ 是 unavoidable 的基本体积：每个 batch 元素、每个 token、每个 hidden channel 都可能产生需要保存的激活。
 $5as/h$ 则反映 attention 中随序列长度二次增长的部分，长上下文时尤其危险。

### 5.3 Tensor Parallel 对 activation memory 的部分作用

Tensor Parallel 可以切 attention 和 MLP 中的大矩阵乘法，所以 activation memory 变为：
$$
M_{\text{act, TP}}
=
sbh
\left(
10 + \frac{24}{t} + 5\frac{as}{ht}
\right)
$$
其中 $t$ 是 tensor parallel size。

PDF 第 47 页解释：剩下的 10 项来自 LayerNorm $4sbh$、Dropout $2sbh$、attention 和 MLP 输入 $4sbh$，这些项不会随 tensor parallel size 降低【PDF†source】。

这说明 TP 只能切掉与大矩阵相关的 activation，不能自动切掉所有 activation。

### 5.4 Sequence Parallel：切掉剩余的 $10sbh$

Sequence Parallel 的核心观察是：剩下的 $10sbh$ 大多来自对 sequence 维度逐点的操作，比如 LayerNorm、Dropout 和 residual input 保存。这些操作不需要跨 token 交互，因此可以沿 sequence axis 切分。

PDF 第 48 页写明：所有 $10sbh$ 项都是 sequence 上的 pointwise ops，因此把 layer norm / dropout 沿 sequence axis 分片；forward 中 $g$ 是 all-gather，$\bar g$ 是 reduce-scatter，backward 中二者反转【PDF†source】。

组合后，PDF 第 49 页给出不同配置下的 activation memory：

| 配置                                        | 每层 activation memory      |
| ------------------------------------------- | --------------------------- |
| No parallelism                              | $sbh(34 + 5as/h)$           |
| Tensor parallel                             | $sbh(10 + 24/t + 5as/(ht))$ |
| Tensor + Sequence parallel                  | $sbh(34/t + 5as/(ht))$      |
| Tensor + selective activation recomputation | $sbh(10 + 24/t)$            |
| Tensor + Sequence + selective recomputation | $sbh(34/t)$                 |

PDF 将最后一类称为让 activation memory fully scale with more machines 的组合。

**学习重点：**
 TP 解决大矩阵相关 activation；SP 解决 TP 留下的 pointwise activation；selective recomputation 解决 attention 中可重算的二次项。三者组合后，activation memory 才更接近线性缩放。

### 5.5 Context Parallel / Ring Attention：长上下文中的序列并行

Context Parallel 和 Sequence Parallel 名字相似，但目标不同。

| 方法                              | 主要目标                                                     | 典型场景                               |
| --------------------------------- | ------------------------------------------------------------ | -------------------------------------- |
| Sequence Parallel                 | 配合 TP，切 LayerNorm/Dropout 等 pointwise activation        | 常规训练中的 activation memory scaling |
| Context Parallel / Ring Attention | 把长上下文的 activation 或 KV 沿 context length 分布到多设备 | 长上下文扩展、serving、KV cache 压力   |

PDF 第 54 页把 Context Parallel / Ring Attention 描述为在长序列中跨 GPU 切分 activations【PDF†source】。原报告也保留了字幕中的解释：它常用于 long context extension 和 serving【字幕†source】。

**容易误解：**
 SP 不是“所有序列维并行”的统称。课程中 SP 更偏向 TP 配套技术；CP/Ring Attention 更偏向长上下文问题。

------

### 5.6 Expert Parallel：MoE 模型中的专家切分

#### 5.6.1 基本思想

Expert Parallel 主要用于 Mixture of Experts。它不是切一个 dense matmul，而是把不同 expert 放在不同设备上，然后通过 router 把 token activation 发往对应 expert。

PDF 第 50 页定义 Expert Parallel：Instead of splitting up the matmul, split up the experts and route activations【PDF†source】。

在 MoE layer 中：

1. token 进入 router；
2. router 选择一个或多个 expert；
3. token activation 被发送到对应 expert 所在 GPU；
4. expert 完成 MLP 计算；
5. 结果再被收回并合并。

#### 5.6.2 为什么 MoE 中通常优先 EP 而不是 TP

PDF 第 51 页引用 Megatron guideline：对 expert layers，优先 EP over TP。理由包括更好的 GEMM efficiency、更低通信、更简单计算图和 token permutation 优化【PDF†source】。

可以这样理解：
$$
\text{Dense MLP: often TP}
\quad
\text{but}
\quad
\text{MoE MLP: prefer EP}
$$
原因是 MoE 的 expert 本来就是天然分开的计算单元。如果继续用 TP 把每个 expert 的矩阵切得太小，局部 GEMM 可能变小，GPU utilization 下降。相比之下，把 token 路由到完整 expert 往往更自然。

#### 5.6.3 Expert Parallel 的系统难点

EP 的主要难点是 **all-to-all token dispatch**。每个 MoE layer 中，不同 token 可能去不同 expert，通信模式不如 dense TP 那样规则。

系统挑战包括：

- router 输出不均衡时可能导致某些 expert 负载过重；
- token dispatch 对延迟敏感；
- all-to-all 通信难以隐藏；
- 需要优化 token permutation / unpermutation；
- 与 DP/TP/PP 组合时 parallel group 关系复杂。

原报告还提到 DeepSeek 的 DPP 和 NVIDIA Hybrid EP 这类底层优化，以及 DeepSeek 使用低层 GPU/network primitive 甚至 undocumented PTX 指令来提升效率【字幕†source】。这些内容说明，frontier MoE 训练往往需要非常底层的系统优化。

### 5.7 EP 与 Attention 并行的冲突

MoE 主要作用在 MLP，不直接作用在 attention。因此，attention 和 MoE MLP 的并行需求可能冲突：

| 模块      | 偏好的并行                                  |
| --------- | ------------------------------------------- |
| Attention | 可能需要较高 TP 或 CP                       |
| MoE MLP   | 更偏好 EP，且不希望 TP 把 expert 矩阵切太碎 |

PDF 第 53 页指出，Megatron 中可以把 attention layers 的 TP/CP/DP 与 MoE layers 的 ETP/EP/EDP 分开【PDF†source】。这说明现代训练系统不再只是设置一个全局 TP/PP/DP，而是可能对 attention 和 MoE 层使用不同并行折叠方式。

------

### 5.8 3D / 4D 并行：没有单一最优策略

课程后半部分进入组合策略。PDF 第 55 页给出 LLM parallelism table，对 DDP/ZeRO-1、FSDP/ZeRO-3、Pipeline、Tensor、Sequence/Context、Expert Parallel 的通信、显存、带宽成本、是否扩展 global batch 和易用性做了对比。

可以把各方法的角色总结为：

| 方法                        | 最擅长解决                           | 主要代价                      |
| --------------------------- | ------------------------------------ | ----------------------------- |
| DDP / ZeRO-1                | 简单扩展 compute，切 optimizer state | 参数和 activation 不降        |
| FSDP / ZeRO-3               | 参数、梯度、optimizer state 显存     | 额外参数通信，不切 activation |
| Pipeline Parallel           | 跨慢链路切深度                       | bubble，需要 microbatch       |
| Tensor Parallel             | 节点内切宽度                         | 每层 activation collective    |
| Sequence / Context Parallel | activation / KV / 长上下文           | per-layer sequence exchange   |
| Expert Parallel             | MoE expert 参数和 token routing      | all-to-all dispatch，系统复杂 |

课程的核心结论是：**没有单一并行策略严格占优。真实训练往往需要组合多种策略。**

### 5.9 选择策略的 roofline 直觉

一个重要判断标准是：
$$
T_{\text{compute}}
\ge
T_{\text{comm}}
\quad
\Rightarrow
\quad
\text{通信可以被计算隐藏}
$$
如果某层计算时间足够长，那么 all-gather、reduce-scatter 或 all-reduce 可以重叠在计算下面；如果通信时间更长，GPU 就会等待通信，进入 communication-bound 区域。

原报告中保留了字幕对这一点的解释：只要 compute time longer than communication time，原则上就能 hide communication underneath computation【字幕†source】。

**工程直觉：**
 并行策略的目标不是让通信为零，而是让通信尽可能发生在 GPU 仍有计算可做的时候。

### 5.10 经验规则：先让模型放下，再最大化数据并行

PDF 第 57 页给出简单规则【PDF†source】：

1. 在模型还放不下时：
   - 节点内使用 Tensor / Expert Parallel；
   - 跨机器使用 Pipeline Parallel；
   - 或根据带宽情况使用 ZeRO-3/FSDP。
2. 模型已经能放下后：
   - 剩余 GPU 尽量用于 Data Parallel；
   - 如果 batch size 太小，可以用 gradient accumulation 换取更高通信效率。

原报告保留的字幕总结也很直接：快链路上放 TP/EP，慢链路上放 PP 或 FSDP，模型放下后用 DP，batch 太小时用 gradient accumulation【字幕†source】。

### 5.11 Narayanan 2021 scaling 策略与 TP=8 经验

PDF 第 59 页展示 Narayanan 2021 的 scaling strategies，并总结：

- Tensor Parallel 先增加到 8，然后 capped at 8；
- Pipeline Parallel 继续增加以让模型放下；
- Data Parallel 随模型规模增长逐渐下降，最大模型 DP=6【PDF†source】。

PDF 第 61 页还展示 Tensor Parallel = 8 在某些配置下经常较优：跨 64 machines 并行时，8×8 配置效果最好【PDF†source】。

**学习提示：**
 “TP=8” 不是数学定律，而是 GPU 节点内高速互联和常见模型规模下的经验。TPU mesh 或特殊网络下，最优 TP 可能不同。

### 5.12 Activation recomputation 的反直觉收益

PDF 第 62 页指出：activation recomputation can pay for itself，因为它节省显存后允许更大 batch，从而提升 throughput。

这很反直觉：recomputation 增加 FLOPs，却可能更快。原因是：
$$
\text{更多计算}
\quad\not\Rightarrow\quad
\text{更慢}
$$
如果多出来的计算换来了更大 microbatch、更少 bubble、更好的通信隐藏，整体吞吐反而可能提升。

**工程直觉：**
 大模型训练的目标不是最小化理论 FLOPs，而是最大化有效吞吐。显存、batch、通信 overlap 和 GPU utilization 共同决定速度。

------

### 5.13 真实大模型训练案例

课程最后用近期公开训练案例说明并行组合。PDF 第 72 页汇总了多个模型的 DP/TP-SP/EP/PP/CP 配置，并总结：TP 通常 $\le 8$，EP 可以很大但困难，长上下文阶段会使用较大 CP。

| 模型 / 系统                  | 并行配置或特征                                          | 学习重点                               |
| ---------------------------- | ------------------------------------------------------- | -------------------------------------- |
| OLMo / Dolma 7B              | 使用 FSDP                                               | 7B 级 dense 模型可较好依赖 FSDP        |
| DeepSeek                     | ZeRO-1 + Tensor + Sequence + Pipeline                   | dense 阶段常见 TP/SP/PP 组合           |
| DeepSeek V3                  | PP=16，EP=64-way，8 nodes，ZeRO-1                       | MoE 中 EP 变成核心，并需要复杂 overlap |
| Yi                           | ZeRO-1 + Tensor + Pipeline；MoE 后 EP 替代 TP           | MoE 后 dense TP 不一定是最佳           |
| Llama 3 405B                 | 预训练中 TP=8、CP=1、PP=16、DP=128；长上下文阶段提高 CP | dense 巨型模型需要 TP/PP/DP/CP 组合    |
| Gemma 2                      | ZeRO-3、MP=TP+SP、DP                                    | TPU mesh 风格下可较多依赖 TP/SP        |
| Mixtral 8×22B                | TP/PP/CP/EP = 4/4/1/8                                   | MoE 中中等 EP + TP/PP 组合             |
| Nemotron 3 120B long-context | TP/PP/CP/EP = 2/0/64/64                                 | 长上下文阶段 CP 很大                   |
| Qwen 3                       | 大模型采用 2/8/32 parallelism                           | 大 EP + PP + 小 TP 的 MoE 配方         |

PDF 对 Qwen 3 的页面也说明，225B-A22B 和 30B-A3B 主要使用 expert parallel，较大模型采用 2/8/32 parallelism。

### 5.14 本节学习提示

本节最重要的结论是：

**参数显存只是第一层问题；activation、长上下文、MoE 路由和通信 overlap 才决定真实训练系统的复杂度。**

复习时抓住四个组合规律：

1. Dense 模型：常见 TP + PP + DP + FSDP/ZeRO。
2. MoE 模型：MLP 部分偏向 EP，attention 仍可能需要 TP/CP。
3. 长上下文：CP/Ring Attention 变重要。
4. 大规模集群：没有单一最优策略，通常要 3D/4D 组合。

------

## 6. Python 代码示例详解

### 6.1 文件整体结构：本讲没有官方 Python 文件

本讲对应的 GitHub 仓库中当前列出的是 `lecture_08.pdf`，不是 `lecture_08.py`。仓库 README 说明可执行讲义命名为 `lecture_XX.py`，非可执行讲义命名为 `lecture_XX.pdf`；Lecture 8 文件列表对应 `lecture_08.pdf`。

因此，本讲没有可逐段解析的官方 Python 文件、函数、变量或 `main()` 流程。为避免混淆，本报告不把其他年份或其他仓库中的代码伪装成本讲官方代码，也不新增 `【Python代码†source】` 引用。

### 6.2 与“代码实现”最接近的内容：算法流程图

虽然没有 Python 文件，但 PDF 中有许多可视化流程图，实际上起到了“伪代码”的作用：

| 课程知识点                               | PDF / 伪流程位置 | 功能                              | 对应理论原理                                     | 在课程中的作用            |
| ---------------------------------------- | ---------------- | --------------------------------- | ------------------------------------------------ | ------------------------- |
| All-reduce = Reduce-scatter + All-gather | 第 7–8 页        | 展示通信原语和等价关系            | collective decomposition                         | 支撑 ZeRO-1 显存收益      |
| ZeRO-1                                   | 第 19–21 页      | optimizer state sharding          | reduce-scatter gradients + all-gather params     | 说明通信不变但显存下降    |
| ZeRO-2                                   | 第 22–23 页      | gradient sharding                 | backward 中逐层 reduce/free                      | 避免完整梯度常驻          |
| ZeRO-3 / FSDP                            | 第 24–27 页      | shard everything                  | all-gather params、reduce-scatter grads、overlap | 参数显存线性缩放          |
| Pipeline Parallel                        | 第 32–38 页      | layers across GPUs                | microbatch pipeline + bubble scheduling          | 用慢链路切深度            |
| Tensor Parallel                          | 第 39–43 页      | submatrices across GPUs           | row/column partition + all-reduce                | 用快链路切宽度            |
| Sequence Parallel                        | 第 46–49 页      | sequence-axis activation sharding | all-gather / reduce-scatter duality              | 让 activation memory 缩放 |
| Expert Parallel                          | 第 50–53 页      | split experts and route tokens    | sparse token dispatch + all-to-all               | MoE 训练核心并行方式      |

### 6.3 关键流程的伪代码化理解

以下不是课程官方 Python 代码，而是学习用伪代码，帮助把课程概念和实现流程对应起来。

#### 6.3.1 ZeRO-1 伪流程

```
for each training step:
    each GPU computes full gradients on local batch shard
    reduce_scatter(gradients)        # each GPU gets its gradient shard
    each GPU updates its parameter shard using local optimizer state
    all_gather(parameter_shards)     # all GPUs recover full parameters
```

对应课程概念：

| 伪代码步骤             | 课程概念                 | 学习重点                                   |
| ---------------------- | ------------------------ | ------------------------------------------ |
| compute full gradients | data parallel            | 每个 worker 仍走完整模型                   |
| reduce_scatter         | all-reduce 等价式前半段  | 梯度归约并分片                             |
| update parameter shard | optimizer state sharding | 每个 worker 只存自己负责的 optimizer state |
| all_gather             | all-reduce 等价式后半段  | 恢复完整参数供下一步 forward               |

#### 6.3.2 FSDP / ZeRO-3 伪流程

```
for each layer in forward:
    all_gather(layer_parameter_shards)
    forward(layer)
    free_full_layer_parameters()

for each layer in backward:
    all_gather(layer_parameter_shards)
    backward(layer)
    reduce_scatter(layer_gradients)
    free_full_layer_parameters()

optimizer_update(local_parameter_shards, local_optimizer_state)
```

对应课程概念：

| 伪代码步骤                         | 课程概念                      | 学习重点                  |
| ---------------------------------- | ----------------------------- | ------------------------- |
| all_gather(layer_parameter_shards) | 参数按需 materialize          | FSDP 不是永久持有完整参数 |
| free_full_layer_parameters         | incremental memory management | 用完立即释放，降低峰值    |
| reduce_scatter(layer_gradients)    | gradient sharding             | 每张 GPU 只保留梯度 shard |
| optimizer_update                   | local optimizer state         | optimizer state 也分片    |

#### 6.3.3 Pipeline Parallel 伪流程

```
split global batch into microbatches

for microbatch in microbatches:
    stage0.forward(microbatch)
    stage1.forward(activation_from_stage0)
    ...
    
for microbatch in reversed(microbatches):
    last_stage.backward(loss_grad)
    previous_stage.backward(activation_grad)
    ...
```

学习重点：microbatch 数越多，pipeline 越容易被填满，bubble ratio 越小。

#### 6.3.4 Tensor Parallel 伪流程

```
# Column-wise first projection
Y_i = GeLU(X @ A_i)     # each GPU owns columns A_i

# Row-wise second projection
Z_i = Y_i @ B_i         # each GPU computes partial output

Z = all_reduce_sum(Z_i)
```

学习重点：TP 的数学并不复杂，复杂的是每层都要通信，因此需要低延迟高带宽互联。

### 6.4 代码与课程知识点对应表

| 课程知识点        | 对应伪代码变量 / 操作                             | 功能                          | 理论原理                      | 在课程中的作用      |
| ----------------- | ------------------------------------------------- | ----------------------------- | ----------------------------- | ------------------- |
| Data Parallel     | `local_batch_shard`, `all_reduce`                 | 多 GPU 计算不同样本梯度       | 大 batch 梯度分解             | 最基础的计算扩展    |
| ZeRO-1            | `reduce_scatter(gradients)`, `all_gather(params)` | 切 optimizer state            | all-reduce 等价式             | 显存收益几乎免费    |
| ZeRO-2            | backward hook + `reduce_scatter(layer_grad)`      | 梯度及时分片释放              | gradient lifecycle management | 避免完整梯度常驻    |
| FSDP              | `all_gather(layer_params)`, `free()`              | 参数按需拼接和释放            | parameter sharding            | 参数显存线性下降    |
| Pipeline          | `microbatches`, `stage.forward/backward`          | 多 stage 流水线               | bubble scheduling             | 跨节点切深度        |
| Tensor Parallel   | `A_i`, `B_i`, `all_reduce_sum`                    | 切矩阵宽度                    | partial sums                  | 节点内切大矩阵      |
| Sequence Parallel | sequence shard + all-gather/reduce-scatter        | 切 pointwise activation       | sequence-axis sharding        | 降低剩余 activation |
| Expert Parallel   | `route(tokens, experts)`                          | token 到 expert 的 all-to-all | sparse dispatch               | MoE 训练核心        |

------

## 7. PDF / 幻灯片 / 图表解析

### 7.1 主要图表总览

| PDF 页面    | 展示内容                                      | 支持的课程结论                   | 对应课程部分      |
| ----------- | --------------------------------------------- | -------------------------------- | ----------------- |
| 第 4–6 页   | 单 GPU compute/memory 限制，多 GPU / 多机互联 | 并行化是必要条件                 | 并行化动机        |
| 第 7–8 页   | collective primitives 与 all-reduce 分解      | 通信原语是并行算法共同语言       | 通信基础          |
| 第 15–17 页 | naive DP 与训练显存账本                       | DDP compute 扩展但 memory 不扩展 | 数据并行          |
| 第 18–28 页 | ZeRO stages 与 FSDP timeline                  | ZeRO 通过分片状态降低显存        | ZeRO/FSDP         |
| 第 29–30 页 | data parallel 剩余问题                        | batch 和 activation 限制仍存在   | 引出模型并行      |
| 第 32–38 页 | layer-wise parallel、bubble、zero-bubble      | Pipeline 需要 microbatch 调度    | Pipeline Parallel |
| 第 39–43 页 | tensor parallel 的矩阵切分和通信量            | TP 无 bubble 但通信密集          | Tensor Parallel   |
| 第 44–49 页 | activation memory 与 sequence parallel        | activation memory 也需要并行     | SP / activation   |
| 第 50–53 页 | expert parallel 与 MoE 并行折叠               | MoE 偏好 EP，但组合复杂          | Expert Parallel   |
| 第 55–57 页 | LLM parallelism table 和 3D/4D 规则           | 没有单一最优策略                 | 组合并行          |
| 第 59–62 页 | Narayanan scaling、TP=8、recomputation        | 实践中常见经验规则               | scaling           |
| 第 63–72 页 | 真实模型并行配置                              | 真实训练是多策略组合             | 训练案例          |

### 7.2 关键图表详细解释

#### 7.2.1 第 4–6 页：compute 与 memory 限制

这些页面回答“为什么要并行”。第 4 页强调单 GPU compute 上限，第 5 页强调模型参数增长导致单 GPU 放不下，第 6 页展示多 GPU / 多机器通过节点内和节点间互联组合成训练系统【PDF†source】。

读图重点：

- 这些图不是在比较具体某张 GPU，而是在建立趋势：模型和训练计算增长速度超过单 GPU scaling；
- 结论是训练系统需要把 compute 和 memory 都拆到多设备上。

#### 7.2.2 第 7–8 页：Collective communication

第 7 页图示 all-reduce、broadcast、reduce、all-gather、reduce-scatter。第 8 页展示 all-reduce 可拆成 reduce-scatter 和 all-gather【PDF†source】。

读图重点：

- All-reduce 的输出是所有 rank 都有完整归约结果；
- Reduce-scatter 的输出是每个 rank 只拿归约结果的一部分；
- All-gather 的输出是所有 rank 都拿到完整拼接结果；
- ZeRO-1 就是把 DDP 的 all-reduce 改写成 reduce-scatter + all-gather，并改变状态存储位置。

#### 7.2.3 第 18–28 页：ZeRO 分阶段显存图

这些图用不同颜色表示 parameters、gradients、optimizer states，展示从 baseline 到 ZeRO-1/2/3 的显存变化。

读图重点：

- ZeRO-1 主要切绿色 optimizer states；
- ZeRO-2 再切粉色 gradients；
- ZeRO-3 连蓝色 parameters 也切；
- 第 25–26 页的 FSDP timeline 说明 all-gather / compute / free / reduce-scatter 如何沿计算图发生；
- 第 28 页的 A100 容量表把抽象公式变成实际参数量估算。

#### 7.2.4 第 32–38 页：Pipeline bubble 与 zero-bubble

第 33 页展示朴素 layer-wise parallel 的大量空闲，第 34 页展示 microbatch pipeline，第 38 页展示 zero-bubble 中 backward 被拆成 B 和 W【PDF†source】。

读图重点：

- 横轴是时间；
- 不同颜色/块表示不同 stage 在做 forward/backward；
- 空白区域就是 bubble；
- zero-bubble 的本质是把可延迟的 W 任务塞进 bubble。

#### 7.2.5 第 40–43 页：Tensor Parallel 的 MLP / Attention 切分

第 40 页展示 MLP 中 $A$ 按列切、$B$ 按行切；第 41 页展示 attention 和 MLP 中 column-wise / row-wise 的位置；第 43 页对比 Pipeline 和 Tensor 的通信量【PDF†source】。

读图重点：

- Column-wise 通常产生输出分片；
- Row-wise 通常产生 partial sum，需要 all-reduce；
- TP 的优势是无 bubble；
- TP 的代价是每层都通信，必须放在快链路上。

#### 7.2.6 第 44–49 页：Activation memory 与 Sequence Parallel

第 44 页动态图显示显存随 forward/backward 变化；第 46 页给出 activation memory 公式；第 47 页说明 TP 后仍剩 $10sbh$；第 48–49 页展示 SP 如何把这些 pointwise activation 沿 sequence 维切分【PDF†source】。

读图重点：

- 显存峰值不是静态参数决定的；
- activation 往往在 forward 末尾和 backward 开始处形成高峰；
- TP 不能切掉所有 activation；
- SP 让剩余 pointwise activation 也随 $t$ 缩小。

#### 7.2.7 第 55–72 页：并行策略总表与真实案例

第 55 页总表说明各种并行方式的优缺点；第 57 页给出 3D/4D 经验规则；第 59–62 页展示实践 scaling 规律；第 72 页汇总真实模型配置【PDF†source】。

读图重点：

- 表格中每种方法都有代价；
- TP 通常不无限增大；
- EP 可以很大，但系统困难；
- 长上下文阶段 CP 显著增加；
- 真实训练通常是 DP/TP/PP/EP/CP/FSDP 的组合。

------

## 8. 关键公式与推导汇总

### 8.1 公式总表

| 公式                                                         | 符号解释                                   | 适用场景                  | 直觉解释                                              | 代码 / 流程对应                              |
| ------------------------------------------------------------ | ------------------------------------------ | ------------------------- | ----------------------------------------------------- | -------------------------------------------- |
| $\theta_{t+1}=\theta_t-\eta\sum_{i=1}^B\nabla f(x_i)$        | $\theta$ 参数，$\eta$ 学习率，$B$ batch    | 数据并行                  | 多 GPU 分别算 batch shard 梯度再同步                  | DDP 梯度 all-reduce                          |
| $\text{AllReduce}=\text{ReduceScatter}+\text{AllGather}$     | collective 原语等价                        | ZeRO-1                    | 先归约分片，再收集完整结果                            | reduce-scatter gradients + all-gather params |
| $C_{\text{DDP}}\approx 2P$                                   | $P=\#params$                               | DDP                       | 每步同步梯度需要参数量级通信                          | gradient all-reduce                          |
| $C_{\text{FSDP}}\approx 3P$                                  | 两次 all-gather + 一次 reduce-scatter      | ZeRO-3/FSDP               | 多一些参数通信，换显存线性下降                        | all-gather / compute / free                  |
| $\text{bubble ratio}=\frac{n_{\text{stages}}-1}{n_{\text{micro}}}$ | stage 数与 microbatch 数                   | Pipeline                  | microbatch 越多，流水线越满                           | pipeline scheduler                           |
| $C_{\text{pipeline}}\sim bsh$                                | $b$ microbatch，$s$ 序列，$h$ hidden       | Pipeline                  | 只传相邻 stage activation                             | point-to-point send/recv                     |
| $C_{\text{tensor}}\sim 8bsh\frac{n-1}{n}$                    | $n$ 为 TP devices                          | Tensor Parallel           | 每层 activation-sized all-reduce                      | tensor all-reduce                            |
| $M_{\text{act}}=sbh(34+5as/h)$                               | $s,b,h,a$ 分别为序列、batch、hidden、heads | activation memory         | activation 随 $s,b,h$ 线性，attention 项含 $s^2$ 影响 | activation checkpoint/recompute              |
| $M_{\text{act,TP}}=sbh(10+24/t+5as/(ht))$                    | $t$ 为 TP size                             | TP 后 activation          | TP 切掉部分大矩阵 activation，但剩 $10sbh$            | TP + residual pointwise ops                  |
| $M_{\text{act,TP+SP+recomp}}=sbh(34/t)$                      | TP+SP+重算                                 | activation memory scaling | SP 和重算让 activation 更接近线性缩放                 | sequence sharding + recomputation            |

### 8.2 All-reduce 等价式的直觉推导

All-reduce 的目标是：
$$
\forall r,\quad y_r = \sum_{k=1}^{N} x_k
$$
也就是说，每个 rank $r$ 最终都要拿到所有 rank 输入张量的归约结果。

可以拆成两步：

1. **Reduce-scatter**
    把归约后的结果按 shard 分给不同 rank：

$$
r_j \leftarrow \left(\sum_{k=1}^{N} x_k\right)_j
$$

1. **All-gather**
    每个 rank 再收集所有 shard：

$$
y_r = [r_1, r_2, \dots, r_N]
$$

这样所有 rank 最终仍获得完整归约结果。

这就是 ZeRO-1 的关键：DDP 原本 all-reduce 梯度，ZeRO-1 改成 reduce-scatter 梯度 + all-gather 参数，在带宽受限情况下通信成本相当，却让 optimizer state 可以分片存储。

### 8.3 Pipeline bubble ratio 的解释

公式：
$$
\text{bubble ratio}
=
\frac{n_{\text{stages}} - 1}{n_{\text{micro}}}
$$
含义：

- pipeline 启动时，需要 $n_{\text{stages}}-1$ 个时间单位填满流水线；
- microbatch 数越多，有效计算越多，启动/排空的固定开销越容易被摊薄；
- 因此 pipeline parallel 需要足够 microbatch。

常见误解：
 “Pipeline 切层后每张 GPU 参数少，所以一定快。”
 正确理解是：Pipeline 主要解决显存和跨节点通信问题；如果 microbatch 不够，bubble 会让 GPU 利用率很差。

### 8.4 Activation memory 公式的分层理解

原始公式：
$$
M_{\text{act}}
=
sbh
\left(
34 + 5\frac{as}{h}
\right)
$$
第一层：$sbh$ 是基本体积。序列更长、batch 更大、hidden 更宽，activation 都会增加。

第二层：常数 34 来自 Transformer 每层中需要保存的多类 activation。

第三层：$5as/h$ 对应 attention 中随序列长度增长更快的部分。

TP 后：
$$
M_{\text{act, TP}}
=
sbh
\left(
10 + \frac{24}{t} + 5\frac{as}{ht}
\right)
$$
TP 把 $24$ 和 attention 二次项的一部分除以 $t$，但 $10sbh$ 不变。

SP 后：
$$
M_{\text{act, TP+SP}}
=
sbh
\left(
\frac{34}{t}
+
5\frac{as}{ht}
\right)
$$
SP 把剩余 pointwise activation 也沿 sequence 维切分。

再加 selective recomputation：
$$
M_{\text{act, TP+SP+recomp}}
=
sbh
\left(
\frac{34}{t}
\right)
$$
重算去掉 attention softmax/dropout 等可重算项的存储，使 activation memory 更接近线性随 $t$ 缩放。

------

## 9. 概念对比与易混淆点

| 易混淆概念                                | 区别                                              | 联系                 | 常见误解                | 正确理解                                      |
| ----------------------------------------- | ------------------------------------------------- | -------------------- | ----------------------- | --------------------------------------------- |
| DDP vs FSDP                               | DDP 每卡完整模型；FSDP 每卡只存 shard             | 都属于数据并行范畴   | FSDP 是另一种 Pipeline  | FSDP 仍让每卡执行完整模型，只是参数按需拼接   |
| ZeRO-1 vs ZeRO-2 vs ZeRO-3                | 分别切 optimizer state、gradients、parameters     | 都减少 DP 冗余       | ZeRO-1/2 能解决所有显存 | ZeRO-1/2 不切参数，ZeRO-3 不直接切 activation |
| FSDP vs Pipeline Parallel                 | FSDP 通信参数；PP 通信 activation                 | 都可减少单卡参数常驻 | 两者都是“把模型切开”    | FSDP 是状态分片，PP 是计算图分层              |
| Pipeline vs Tensor Parallel               | PP 按深度切，TP 按宽度切                          | 都是模型并行         | TP 总是比 PP 好         | TP 通信频繁，PP 有 bubble，取决于硬件和 batch |
| Tensor Parallel vs Sequence Parallel      | TP 切矩阵，SP 切 sequence 上 pointwise activation | SP 常作为 TP 配套    | SP 就是长上下文 CP      | SP 主要解决 TP 后剩余 activation 项           |
| Sequence Parallel vs Context Parallel     | SP 切 pointwise activation；CP 切长上下文/KV      | 都涉及 sequence 维   | 名字相近所以一样        | CP 更偏长上下文和 KV 分布                     |
| Tensor Parallel vs Expert Parallel        | TP 切 dense matrix；EP 切 expert 集合             | 都可能服务 MLP       | MoE 中继续用高 TP 最好  | MoE expert 天然可分，EP 往往更适合            |
| DP size vs batch size                     | DP 消耗 global batch                              | DP 扩展 compute      | DP 越大越好             | batch 太大后优化边际收益下降                  |
| Activation recomputation vs checkpointing | 保存少量 activation，反向时重算                   | 都用计算换显存       | 增加计算一定更慢        | 省下显存可能换来更大 batch 和更好 overlap     |
| TP=8                                      | 常见 GPU 节点内经验                               | 来自硬件拓扑         | 永远最优                | 取决于互联、模型、batch、实现                 |

------

## 10. 学习拓展与发散理解

### 10.1 与大模型训练实践的关系

**延伸理解：数据准备与 batch size**

数据并行直接消耗 global batch size。如果训练 token 流和 batch 构造不支持足够大的 effective batch，DP 扩展会受限。实际训练中常用 gradient accumulation，把多个 microbatch 的梯度累积后再同步，以提升通信效率。

**工程直觉：模型训练**

训练前通常先做显存预算：
$$
\text{参数显存}
+
\text{梯度显存}
+
\text{optimizer state}
+
\text{activation}
+
\text{通信 buffer}
\le
\text{GPU memory}
$$
然后决定是否需要 ZeRO/FSDP、TP、PP、SP/CP 或 recomputation。

**工程直觉：优化策略**

并行策略会改变有效 batch、梯度同步频率和通信 overlap，因此会影响 optimizer step 的节奏。大规模训练中，优化超参数和并行策略不是完全独立的。

**工程直觉：训练效率**

GPU 利用率不是只由 FLOPs 决定，还受通信、bubble、kernel size、GEMM efficiency 和 load balance 影响。MoE 中即使理论 FLOPs 较低，也可能被 all-to-all 或 expert imbalance 拖慢。

**工程直觉：推理部署**

Context Parallel 和 KV cache 分布不仅和训练有关，也和长上下文推理有关。长上下文服务时，KV cache 可能成为主要内存压力。

### 10.2 与其他 lecture 的联系

本讲可能与 CS336 中以下主线相连：

| 相关主线                | 连接方式                                          |
| ----------------------- | ------------------------------------------------- |
| 语言模型训练目标        | 本讲不改变 loss，但决定 loss 计算如何分布到多卡   |
| Transformer 架构        | TP/SP/EP 都依赖 Transformer block 内部结构        |
| 优化器与训练稳定性      | ZeRO/FSDP 主要围绕 optimizer state 和梯度生命周期 |
| Scaling laws / 训练预算 | 并行策略决定给定预算下能训练多大模型、多少 token  |
| 数据与 batch            | DP、PP、gradient accumulation 都依赖 batch 组织   |
| 评估与部署              | CP/Ring Attention 和长上下文推理部署相关          |

谨慎说明：这里是学习上的联系，不表示这些内容都在 Lecture 8 中详细展开。

### 10.3 面试 / 考试 / 作业可能考察角度

| 问题                                       | 考察点                           | 回答思路                                                     |
| ------------------------------------------ | -------------------------------- | ------------------------------------------------------------ |
| 为什么 ZeRO-1 通信成本和 DDP 相同？        | all-reduce 等价式                | DDP all-reduce 可拆成 reduce-scatter + all-gather；ZeRO-1 改变状态布局 |
| ZeRO-1/2/3 分别切什么？                    | 状态分片层级                     | optimizer state → gradients → parameters                     |
| FSDP 为什么不是 Pipeline Parallel？        | 通信对象与计算图                 | FSDP 每卡仍走完整模型，通信参数；PP 不同层在不同 GPU，通信 activation |
| Pipeline Parallel 为什么需要 microbatch？  | bubble                           | microbatch 增多可摊薄 $(n_{stages}-1)$ 的启动/排空开销       |
| Tensor Parallel 为什么适合节点内？         | 通信频率                         | 每层都有 activation-sized collective，需要低延迟高带宽       |
| Sequence Parallel 解决 TP 后什么问题？     | activation memory                | TP 后仍剩 $10sbh$ pointwise activation，SP 沿 sequence 维切它 |
| MoE 为什么偏好 EP？                        | expert 天然可分                  | sparse token routing 往往比切每个 expert 的 dense matmul 更高效 |
| CP 与 SP 有什么区别？                      | 长上下文 vs pointwise activation | CP 主要切长序列/KV，SP 多作为 TP 配套                        |
| 为什么 recomputation 可能提高 throughput？ | 显存-计算权衡                    | 增加 FLOPs 换更大 batch、更少 bubble、更好通信隐藏           |
| 如何设计 3D/4D 并行？                      | 综合策略                         | 先让模型放下：TP/EP 节点内、PP 跨节点、FSDP 视带宽；再最大化 DP |

------

## 11. 课程复习总结

### 11.1 本讲核心思想

Lecture 8 的核心思想是：**大模型训练并行化不是单一技术，而是在显存、计算、通信、batch size、activation memory、模型结构和硬件拓扑之间做组合优化。**

单 GPU 不够用后，最自然的起点是数据并行；数据并行遇到显存冗余后，引入 ZeRO/FSDP；FSDP 仍解决不了 activation 和 batch 限制，于是引入 Pipeline、Tensor、Sequence、Context 和 Expert Parallel；最后在真实训练中把这些方法组合成 3D/4D 并行系统。

### 11.2 必须掌握的概念

| 概念                  | 必须掌握的点                                       |
| --------------------- | -------------------------------------------------- |
| Collective primitives | all-reduce、all-gather、reduce-scatter 的输入输出  |
| All-reduce 等价式     | all-reduce 可拆成 reduce-scatter + all-gather      |
| Data Parallel         | 切 batch，复制模型，同步梯度                       |
| ZeRO-1/2/3            | 依次切 optimizer state、gradients、parameters      |
| FSDP                  | 参数按层 all-gather，用完释放，梯度 reduce-scatter |
| Pipeline Parallel     | 按层切模型，有 bubble，需要 microbatch             |
| Tensor Parallel       | 按矩阵宽度切，通信密集，适合快链路                 |
| Activation memory     | 不只是参数显存，activation 也会成为瓶颈            |
| Sequence Parallel     | 切 TP 后剩余 pointwise activation                  |
| Context Parallel      | 长上下文/KV 的序列维分布                           |
| Expert Parallel       | MoE 中切专家并路由 token                           |
| 3D/4D parallelism     | 多种并行组合，没有单一最优                         |

### 11.3 必须理解的公式

| 公式                                                     | 解决的问题                                            |
| -------------------------------------------------------- | ----------------------------------------------------- |
| $\theta_{t+1}=\theta_t-\eta\sum_{i=1}^{B}\nabla f(x_i)$  | 数据并行为什么可以把 batch shard 的梯度同步后等价更新 |
| $\text{AllReduce}=\text{ReduceScatter}+\text{AllGather}$ | ZeRO-1 为什么能通信近似不变但节省显存                 |
| $C_{\text{FSDP}}\approx 3P$                              | FSDP 为什么比 DDP 通信更多但显存收益大                |
| $\frac{n_{\text{stages}}-1}{n_{\text{micro}}}$           | Pipeline bubble 如何随 microbatch 数下降              |
| $C_{\text{pipeline}}\sim bsh$                            | Pipeline 为什么适合点对点慢链路                       |
| $C_{\text{tensor}}\sim 8bsh\frac{n-1}{n}$                | TP 为什么通信密集、通常放节点内                       |
| $M_{\text{act}}=sbh(34+5as/h)$                           | activation memory 为什么随序列和 hidden 增长          |
| $M_{\text{act,TP+SP+recomp}}=sbh(34/t)$                  | TP+SP+重算如何让 activation memory 更接近线性缩放     |

### 11.4 必须看懂的“代码流程”

本讲没有官方 Python 文件，但必须能把以下流程写成伪代码：

1. DDP：local forward/backward → gradient all-reduce → optimizer step；
2. ZeRO-1：reduce-scatter gradients → local optimizer update → all-gather params；
3. FSDP：all-gather layer params → compute → free → backward → reduce-scatter grads；
4. Pipeline：microbatch 在多个 stage 中 forward/backward 流动；
5. Tensor Parallel：矩阵 column/row partition → partial matmul → all-reduce；
6. Sequence Parallel：pointwise activation 沿 sequence shard 存储，需要时 all-gather；
7. Expert Parallel：router dispatch token → expert compute → all-to-all collect。

### 11.5 一句话总结

**Lecture 8 讲的是如何把一个放不下、算不动的大语言模型训练任务，沿数据、参数、层、矩阵、序列和专家多个维度拆开，并根据硬件通信拓扑组合成可扩展的 3D/4D 训练系统。**