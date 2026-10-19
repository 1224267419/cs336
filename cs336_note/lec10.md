后，改写成完整中文学习讲义。重点保留授课顺序、公式与系统主线，并区分课程内容和延伸理解。

已确认原报告对应 Lecture 10 “Transformer Inference”，并与 `lecture_10.py` 的模块顺序一致：workload/AI、吞吐与延迟、KV cache、量化剪枝、speculative sampling、continuous batching 与 PagedAttention。

已思考 14m 34s

# Lecture 10 课程报告：Transformer 推理——从 KV Cache 瓶颈到系统级加速

## 0. 资料来源与改写说明

本报告是在用户上传的原始课程报告 `lec10.md` 基础上进行系统性改写、可读性优化和学习化增强而成。原始报告的主题是 **CS336 Lecture 10: Transformer Inference**，主线包括：推理为什么重要、推理为什么不同于训练、为什么 generation 阶段是 memory-bound、KV cache 如何影响 latency/throughput，以及如何通过 GQA、MLA、量化、剪枝、speculative decoding、continuous batching 和 PagedAttention 等方法优化推理。

本次改写还核对了 Stanford CS336 课程主页和 GitHub 仓库中的 `lecture_10.py`。CS336 课程定位是从头构建语言模型，覆盖数据、Transformer 架构、训练、评估和部署前相关内容；课程也强调 Python、深度学习与系统优化经验的重要性。 在课程安排中，Lecture 10 对应主题为 **Inference**，资料文件为 `lecture_10.py`。

需要说明的是：本次对话中没有单独提供 lecture 字幕文件，也没有提供独立 PDF 文件。因此，本报告不新增逐句字幕引用；原报告中已经转述的课堂讲解会保留为课程内容。对于 GitHub 中 `lecture_10.py` 明确出现的代码结构、公式和图表调用，本报告使用可点击网页引用。对于为帮助理解而加入的内容，会明确标注为 **延伸理解**、**补充说明**、**工程直觉** 或 **学习提示**，不把这些拓展伪装成课程原文。

------

## 1. 学习导读

### 1.1 本讲要解决的核心问题

本讲要解决的问题可以概括为：

> **训练好的 Transformer 语言模型，如何在真实服务中更快、更省、更稳定地生成 token？**

训练阶段关心的是如何用大量数据更新参数；推理阶段关心的是在参数固定后，如何面对真实用户请求、评估任务、强化学习 rollouts 或 agent 工作流，高效地产生输出。Lecture 10 的核心判断是：**Transformer 推理的 generation 阶段，尤其是 attention 读取 KV cache 的过程，本质上容易受内存带宽限制，而不是受计算峰值限制。**

这意味着推理优化不能只看 FLOPs。即使 GPU 理论算力很高，如果每一步只是搬运大量 KV cache，而实际计算很少，硬件也无法被充分利用。

### 1.2 本讲在 CS336 课程中的位置

CS336 的整体目标是让学生从零理解并实现语言模型，包括预训练数据处理、Transformer 构建、训练、系统优化、评估和部署前准备。课程主页明确强调，学生需要理解模型构建全过程，并且要具备深度学习和系统优化经验，因为课程会涉及如何让神经语言模型在 GPU 和多机环境中快速高效运行。

Lecture 10 位于 scaling laws 之后、evaluation 之前。这个位置很有意义：

- 前面的 lecture 讨论模型训练、并行、算力、scaling laws，关注“如何把模型训练出来”；
- 本讲转向 inference，关注“训练出来以后，如何高效使用模型”；
- 后续 evaluation、data、post-training/RL 等内容都会继续依赖推理，因为评估、采样、强化学习 rollout 和 agent 行为都需要大量生成。

**延伸理解：**
 大模型系统的成本并不只发生在训练期间。训练是一次或少数几次大规模作业，而推理发生在每一次用户请求、每一次评估、每一次 rollout 中。因此，推理效率会直接影响产品成本、用户体验、评估速度和强化学习训练效率。

### 1.3 本讲知识结构

| 模块                    | 解决的问题                             | 核心概念                                                | 与后续内容的关系                                |
| ----------------------- | -------------------------------------- | ------------------------------------------------------- | ----------------------------------------------- |
| 推理 workload           | 为什么推理值得单独研究                 | TTFT、latency、throughput、prefill、generation          | 为后续 latency/throughput trade-off 打基础      |
| 训练 vs 推理            | 为什么推理比训练更难并行               | 自回归生成、sequence 不能完全并行                       | 引出 generation memory-bound                    |
| Arithmetic intensity    | 如何判断 compute-bound 或 memory-bound | FLOPs/byte、roofline、H100 阈值                         | 解释为什么 token-by-token generation 效率低     |
| KV cache                | 如何避免重复计算历史 token             | prefill、generation、KV cache 大小公式                  | 推理优化的中心对象                              |
| MLP 与 Attention 分析   | 哪部分是瓶颈                           | MLP AI 约为 `BT`，attention AI 为 `ST/(S+T)`            | 得出 generation attention 几乎必然 memory-bound |
| Throughput/latency 模型 | batch size 如何影响服务指标            | 参数 bytes、KV cache bytes、memory bandwidth            | 解释小 batch 与大 batch 的取舍                  |
| KV cache 缩减方法       | 如何减少内存读写                       | GQA、MLA、CLA、sliding window、linear/SSM               | 架构层面的推理加速                              |
| 数值与结构压缩          | 如何降低参数/KV 存储成本               | quantization、pruning、distillation                     | 减少内存与计算，但需验证 accuracy               |
| Speculative decoding    | 如何在不改变目标分布的情况下加速       | draft model、target model、modified rejection sampling  | 利用“检查快于生成”的不对称性                    |
| 动态服务系统            | 真实在线流量如何调度                   | continuous batching、selective batching、PagedAttention | 连接操作系统思想与 LLM serving                  |

### 1.4 学习本讲前需要知道什么

本讲需要的前置知识不是抽象罗列，而是直接服务于推理分析：

| 前置知识                 | 在本讲中的用途                                      |
| ------------------------ | --------------------------------------------------- |
| 矩阵乘法 FLOPs 计算      | 推导 matmul、MLP、attention 的计算量                |
| 张量形状                 | 理解 `B, S, T, D, F, N, K, H, L, V` 等符号          |
| Transformer block        | 区分 attention 与 MLP 的计算/内存模式               |
| causal language modeling | 理解为什么 generation 必须逐 token 自回归           |
| GPU 内存层次             | 理解 HBM bandwidth、memory-bound、compute-bound     |
| softmax attention        | 理解 `QK^T`、attention weights 和 `V` 的乘法        |
| 概率采样                 | 理解 speculative sampling 的接受/拒绝机制           |
| 基础系统概念             | 理解 batching、fragmentation、paging、copy-on-write |

**学习提示：**
 本讲不是单纯讲“怎么让模型跑快”，而是训练学生形成一种系统分析方式：先看 workload，再看 FLOPs/bytes，再判断 bottleneck，最后选择架构、数值、算法或系统层面的优化方案。

------

## 2. 第一部分：推理为什么重要，以及“快”到底是什么意思

这一节要解决的问题是：**为什么 inference 是现代 LLM 系统中的核心成本？当我们说推理要“快”时，到底在优化什么指标？**

### 2.1 问题引入

语言模型训练完成后，推理才是真正面向用户和应用的阶段。Lecture 10 的 `main()` 函数也把本讲安排为：先理解 inference workload，再做 arithmetic intensity 分析，再计算 throughput/latency，然后讨论 KV cache 缩减、量化、剪枝、speculative sampling、dynamic batching 和 PagedAttention。

推理出现的典型场景包括：

- 聊天机器人和 AI assistant；
- 代码补全；
- agent 工具调用与多轮思考；
- 批量数据处理；
- instruction following 或代码生成评估；
- 强化学习中的 rollout 生成。

课程强调的关键动机是：**训练通常是一次性成本，而推理是反复发生的成本。** `lecture_10.py` 中也直接列出 inference 的多种用途，并指出训练是 one-time cost，而 inference 会被 repeated many times。

### 2.2 三个“快”：TTFT、latency、throughput

推理系统里，“快”不是一个单一概念。课程区分了三个指标。

#### 2.2.1 TTFT：Time to First Token

**直觉解释：**
 TTFT 是用户提交 prompt 后，等到第一个 token 出现的时间。它影响的是“模型有没有开始响应”的体感。

**正式定义：**
$$
\text{TTFT}
=
\text{time from request arrival to first generated token}
$$
在 Transformer 推理中，TTFT 主要由 **prefill** 决定。Prefill 要把用户 prompt 中已有 token 一次性送入模型，计算各层 hidden states，并建立 KV cache。因为 prompt 已经完整给出，prefill 可以沿 sequence 维度并行，所以它更接近训练阶段的计算模式。`lecture_10.py` 明确写到 TTFT essentially 是 prefill time 的函数，并建议 prefill 使用较小 batch 以获得更快 TTFT。

**学习提示：**
 TTFT 优化关注的是交互体验：第一个 token 尽快出来，用户会觉得系统更流畅。

#### 2.2.2 Latency：单个请求的生成速度

**直觉解释：**
 Latency 关注单个用户请求生成 token 的速度。例如，一个聊天窗口中 token 是每 20 ms 出一个，还是每 200 ms 出一个。

**常见单位：**
$$
\text{latency}
=
\text{seconds/token}
\quad \text{or} \quad
\text{ms/token}
$$
如果 batch size 太大，一个请求需要和很多请求一起等待同一步 decoding 完成。这样系统总吞吐可能变高，但单个用户看到 token 的间隔会变长。

#### 2.2.3 Throughput：整个系统的 token 产出速度

**直觉解释：**
 Throughput 衡量系统总共能在单位时间内生成多少 token。它更适合离线评估、批量数据处理、大规模 agent rollout 等场景。

**常见单位：**
$$
\text{throughput}
=
\text{tokens/second}
$$
如果任务是批量处理 100 万条样本，单条样本是否立刻返回不重要，系统整体 tokens/s 才是关键。

### 2.3 Latency 与 throughput 的基本取舍

课程中的核心 trade-off 是：
$$
\text{small batch}
\Rightarrow
\text{better latency, worse throughput}
$$
`lecture_10.py` 在 throughput/latency 部分明确总结：小 batch 带来更好的 latency 但更差的 throughput，大 batch 带来更好的 throughput 但更差的 latency。

**工程直觉：**
 这类似公交车和出租车的区别。出租车不用等别人，延迟低；公交车装很多人，单位成本低但个人等待时间可能更长。LLM serving 的 batch size 选择，本质上也在平衡用户体验和系统利用率。

### 2.4 学习提示

本节最重要的是区分三个指标：

| 指标       | 关注对象              | 优化目标                 | 常见场景                 |
| ---------- | --------------------- | ------------------------ | ------------------------ |
| TTFT       | 第一个 token 多快出来 | 降低用户等待感           | 聊天、代码补全           |
| latency    | 单个请求 token 流多快 | 让交互更流畅             | 在线服务                 |
| throughput | 系统总 token/s        | 降低总成本、提高处理能力 | 批处理、评估、RL rollout |

**一句话总结：**
 推理优化不是简单追求“越快越好”，而是要明确是在优化 TTFT、单请求 latency，还是系统 throughput。

------

## 3. 第二部分：为什么推理和训练的 workload 完全不同

这一节要解决的问题是：**同样是 Transformer forward pass，为什么训练可以高效并行，而推理 generation 却很容易低效？**

### 3.1 问题引入

训练时，模型可以一次看到完整训练序列。对于长度为 `T` 的训练样本，模型可以并行计算所有 token 位置的 hidden states、attention、MLP 和 logits。

但推理时，语言模型是自回归生成的。生成第 `t+1` 个 token 前，必须先知道第 `t` 个 token 是什么。也就是说，输出序列本身是动态产生的，不能提前并行展开。

### 3.2 自回归推理的正式表达

给定已有上下文：
$$
x_1, x_2, \dots, x_t
$$
模型下一步计算：
$$
p_\theta(x_{t+1}\mid x_{\le t})
$$
然后从这个分布中选择或采样出：
$$
x_{t+1}
$$
下一步才能继续：
$$
p_\theta(x_{t+2}\mid x_{\le t+1})
$$
这就是 inference 和 training 的根本差异：

| 阶段       | 是否知道完整序列  | sequence 维度能否并行  | 核心瓶颈                        |
| ---------- | ----------------- | ---------------------- | ------------------------------- |
| Training   | 知道完整训练序列  | 可以并行               | 大矩阵乘法、通信、显存          |
| Prefill    | 知道完整 prompt   | 可以并行               | 通常较接近 training             |
| Generation | 新 token 逐步产生 | 不能沿生成维度完全并行 | KV cache 读取、memory bandwidth |

`lecture_10.py` 也明确对比：training 可以看到所有 tokens 并 parallelize over sequence，而 inference 必须 sequentially generate，所以更难充分利用 compute。

### 3.3 直觉例子：训练像批量改卷，推理像接龙

**训练** 像批量改卷：所有题目答案都已经在纸上，可以一次扫描、并行评分。

**推理 generation** 像文字接龙：下一个词取决于上一个词是什么；上一个词没定下来，下一个词就不能真正开始生成。

这就是为什么模型训练时常常能把大矩阵乘法做得很“厚”，而 generation 阶段经常退化成每次只处理一个 token 的“薄”矩阵运算。

### 3.4 学习提示

最容易混淆的是：**prefill 也是推理的一部分，但它不像 generation 那样强自回归。**

- Prefill：prompt 已知，可以并行；
- Generation：输出未知，必须逐步生成；
- 因此本讲说“推理 memory-bound”时，最核心指的是 **generation 尤其是 attention 的 KV cache 读取**。

------

## 4. 第三部分：符号体系与 Transformer 结构回顾

这一节要解决的问题是：**为了推导 FLOPs、bytes 和 KV cache 大小，需要统一哪些符号？**

### 4.1 课程中的符号约定

`lecture_10.py` 使用类似 einops 的形状记号，并定义了 `B, S, T, D, F, N, K, H, L, V` 等符号。

| 符号 | 含义                          | 学习重点                              |
| ---- | ----------------------------- | ------------------------------------- |
| `B`  | batch size，同时处理的序列数  | 影响 MLP 的权重复用和系统 throughput  |
| `S`  | 已有上下文 token 数           | 影响 KV cache 大小和 attention 读取量 |
| `T`  | 当前要计算 logits 的 token 数 | prefill 中可大，generation 中通常为 1 |
| `D`  | model dimension / hidden size | Transformer 主隐藏维度                |
| `F`  | MLP feed-forward dimension    | 常近似为 `4D`，但实际模型可不同       |
| `N`  | query heads 数                | attention query 头数量                |
| `K`  | key/value heads 数            | GQA 中可小于 `N`                      |
| `H`  | 每个 head 的维度              | 通常 `D = N H`                        |
| `L`  | Transformer 层数              | KV cache 与层数线性相关               |
| `V`  | vocabulary size               | embedding/unembedding 参数量相关      |

课程约定包括：
$$
F \approx 4D
$$
对于 GQA：
$$
N = K G
$$
其中 `G` 表示每个 KV head 服务多少个 query heads。`lecture_10.py` 中也写出 `F = 4D`、`D = NH`、`N = KG` 等约定。

### 4.2 Transformer block 的两类主要计算

本讲关心的 Transformer block 主要分成两大部分。

#### 4.2.1 Attention

Attention 部分从 hidden state 投影出：
$$
Q = X W_Q,\quad K = X W_K,\quad V = X W_V
$$
然后做：
$$
\mathrm{Attention}(Q,K,V)
=
\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{H}}\right)V
$$
本讲为了做 FLOPs/bytes 分析，经常忽略 softmax、norm、激活函数等较小开销，主要关注大矩阵乘法和 HBM 读写。

#### 4.2.2 MLP

课程按 gated MLP 计算，包含三组主要权重：
$$
W_{\mathrm{up}}\in\mathbb{R}^{D\times F}
$$
前两组把隐藏维度从 `D` 投到 `F`，最后一组再从 `F` 投回 `D`。原报告也强调课程按 gated MLP 计算三个大矩阵，这一点与 `lecture_10.py` 中 MLP FLOPs/bytes 统计一致。

### 4.3 学习提示

这一节的重点不是背符号，而是理解每个符号如何进入内存和计算公式：

- `B` 增大：更多请求并发；
- `S` 增大：上下文更长，KV cache 更大；
- `T` 在 generation 中通常等于 1，这是许多低 arithmetic intensity 的根源；
- `K` 减小：KV cache 直接变小，这是 GQA/MLA/CLA 等方法的切入点；
- `L` 增大：每层都要存 KV，因此 KV cache 与层数线性增长。

------

## 5. 第四部分：Roofline 与 Arithmetic Intensity

这一节要解决的问题是：**为什么只数 FLOPs 不够？如何判断一个操作是 compute-bound 还是 memory-bound？**

### 5.1 直觉解释

GPU 有两个关键能力：

1. **算得多快**：每秒能做多少 FLOPs；
2. **搬得多快**：每秒能从 HBM 读写多少 bytes。

如果一个操作每搬 1 byte 数据就能做很多计算，GPU 的计算单元容易被喂饱，这类操作更可能 **compute-bound**。

如果一个操作每搬很多数据只做一点点计算，计算单元经常等数据，这类操作更可能 **memory-bound**。

### 5.2 Arithmetic intensity 的定义

$$
\mathrm{AI}
=
\frac{\mathrm{FLOPs}}{\mathrm{Bytes\ transferred}}
$$

它的单位是：
$$
\mathrm{FLOPs/byte}
$$
硬件也有一个对应阈值：
$$
\mathrm{AI}_{\mathrm{hardware}}
=
\frac{\mathrm{Peak\ FLOPs/s}}{\mathrm{Memory\ bandwidth\ bytes/s}}
$$
判断逻辑是：
$$
\mathrm{AI}_{\mathrm{computation}}
>
\mathrm{AI}_{\mathrm{hardware}}
\Rightarrow
\text{compute-bound}
$$
`lecture_10.py` 在 arithmetic intensity 回顾中使用 H100 例子：peak FLOPs/s 为 `989e12`，memory bandwidth 为 `3.35e12` bytes/s，因此 accelerator intensity 约为 295，并据此判断低于该强度的操作更容易 memory-bound。

### 5.3 H100 例子

$$
\mathrm{AI}_{H100}
=
\frac{989\times 10^{12}}
{3.35\times 10^{12}}
\approx 295
$$

这表示：如果某个 kernel 的 arithmetic intensity 明显低于 295 FLOPs/byte，它很可能不是“算不动”，而是“数据搬不过来”。

**工程直觉：**
 大模型推理中的 generation attention 经常要从 HBM 中读大量历史 KV cache，但每个 token 只做一次较薄的 attention 计算。因此即使 GPU 算力极强，也可能因为 HBM 带宽成为瓶颈。

### 5.4 Warm-up：普通矩阵乘法

考虑：
$$
X\in\mathbb{R}^{B\times D}
$$
假设使用 bf16，每个数占 2 bytes。

#### FLOPs

`Y` 中有 `BF` 个元素，每个元素是长度 `D` 的 dot product，需要约 `2D` FLOPs，因此：
$$
\mathrm{FLOPs}
=
2BDF
$$

#### Bytes

需要读 `X`：
$$
2BD
$$
需要读 `W`：
$$
2DF
$$
需要写 `Y`：
$$
2BF
$$
所以：
$$
\mathrm{Bytes}
=
2BD + 2DF + 2BF
$$

#### Arithmetic intensity

$$
\mathrm{AI}
=
\frac{2BDF}
{2BD + 2DF + 2BF}
$$

当 `D,F \gg B` 时，分母主要由 `2DF` 主导：
$$
\mathrm{AI}
\approx
\frac{2BDF}{2DF}
=
B
$$
`lecture_10.py` 中对应代码也统计了 matmul 的 FLOPs 与 bytes，并在 `D,F` 远大于 `B` 时得到 intensity 等于 `B`。

### 5.5 这个 warm-up 想说明什么？

如果 `B=1`，矩阵乘法退化成 matrix-vector product：
$$
\mathrm{AI}\approx 1
$$
这远低于 H100 的约 295，所以会严重 memory-bound。

**学习提示：**
 这正是 generation 阶段的危险信号：每次只生成一个 token，很多大矩阵乘法无法形成足够“厚”的 batch，权重和 KV cache 读了很多，但计算复用不够。

------

## 6. 第五部分：Naive 自回归推理与 KV cache

这一节要解决的问题是：**为什么 naive inference 会重复计算？KV cache 如何把复杂度从不可接受变成可服务？**

### 6.1 不使用 KV cache 的 naive inference

假设 prompt 为：
$$
x_1,\dots,x_S
$$
要继续生成 `T` 个 token。如果每生成一个 token 都把完整历史重新送进 Transformer，那么第 `t` 步需要处理大约 `S+t` 个 token。

Self-attention 对序列长度是二次复杂度，因此单步约为：
$$
O((S+t)^2)
$$
如果忽略初始 `S`，生成 `T` 个 token 的总复杂度为：
$$
\sum_{t=1}^{T} O(t^2)
=
O(T^3)
$$
`lecture_10.py` 中也明确写出：naive inference 每生成一个 token 都把历史送入 Transformer；如果一次 feedforward 是 `O(T^2)`，生成 `T` 个 token 需要 `O(T^3)` FLOPs。

### 6.2 为什么可以缓存？

Transformer 语言模型使用 causal mask。对前面 token 来说，它们不能看未来 token，因此前面 token 的 key/value 不会因为未来 token 出现而改变。

也就是说，对于已经处理过的 token：
$$
K_i = W_K h_i,\quad V_i = W_V h_i
$$
一旦计算出来，就可以保存。生成新 token 时，只需要：

1. 对新 token 计算新的 query；
2. 读取历史 token 的 key/value；
3. 让新 query attend 到历史 KV；
4. 把新 token 的 key/value 追加进 cache。

这就是 **KV cache**。

### 6.3 Prefill 与 generation 两阶段

推理通常拆成两个阶段。

#### Prefill

输入是完整 prompt：
$$
x_1,\dots,x_S
$$
模型并行处理所有 prompt token，计算各层 hidden states，并写入 KV cache。

特点：

- sequence 维度可并行；
- 和训练阶段更像；
- arithmetic intensity 较高；
- 通常更容易 compute-bound；
- 决定 TTFT 的主要部分。

#### Generation

每一步只生成一个新 token：
$$
T=1
$$
特点：

- 自回归依赖强；
- 需要反复读取历史 KV cache；
- attention 读取量随上下文长度增长；
- 通常 memory-bound；
- 决定后续 token 流式输出速度。

`lecture_10.py` 也把 inference 分为 prefill 和 generation 两阶段，并说明 KV cache 是为每个 sequence、token、layer、head 存储 H 维向量。

### 6.4 KV cache 大小公式

对 batch 中每个 sequence，KV cache 需要存：

- `S`：上下文 token 数；
- `L`：层数；
- `K`：KV heads 数；
- `H`：每个 head 的维度；
- `2`：key 和 value 两份；
- `r`：每个数的 bytes，bf16 下 `r=2`。

单个 sequence 的 KV cache bytes：
$$
C_{\mathrm{KV,seq}}
=
S \cdot L \cdot K \cdot H \cdot 2 \cdot r
$$
bf16 下：
$$
C_{\mathrm{KV,seq}}
=
4SLKH
$$
batch size 为 `B` 时：
$$
C_{\mathrm{KV,total}}
=
4BSLKH
$$
`lecture_10.py` 的性能统计函数也使用：

```
kv_cache_size_per_seq = S * (K*H) * L * 2 * 2
```

其中两个 `2` 分别对应 key/value 两份和 bf16 两字节。

### 6.5 学习提示

KV cache 是本讲的中心对象：

- 它避免了重复计算历史 token；
- 它把 naive inference 的巨大重复计算降下来；
- 但它也带来了 generation 阶段最主要的内存读写负担；
- 后续几乎所有推理优化都可以理解为：**减少、压缩、共享、分页或绕开 KV cache**。

------

## 7. 第六部分：MLP 层的 arithmetic intensity

这一节要解决的问题是：**为什么 MLP 在 generation 阶段也可能 memory-bound，但可以通过 batching 得到改善？**

### 7.1 问题引入

Transformer block 中 MLP 部分包含大矩阵乘法。训练或 prefill 时，输入 token 多，矩阵乘法比较“厚”；generation 时，每步通常只有一个 token，因此矩阵乘法变“薄”。

课程先分析 MLP，是为了和后面的 attention 形成对比：**MLP 的 generation 可以靠 batch size 改善 arithmetic intensity，但 attention 的 generation 不行。**

### 7.2 MLP 计算形式

假设输入：
$$
X\in\mathbb{R}^{B\times T\times D}
$$
gated MLP 中有三次主要矩阵乘法：
$$
U = XW_{\mathrm{up}}
$$
其中：
$$
W_{\mathrm{up}}, W_{\mathrm{gate}}
\in
\mathbb{R}^{D\times F}
$$

### 7.3 FLOPs

每次 matmul 的 FLOPs 约为：
$$
2BTDF
$$
三次 matmul：
$$
\mathrm{FLOPs}_{\mathrm{MLP}}
=
6BTDF
$$
`lecture_10.py` 中也通过 assert 检查：

```
assert flops == 6*B*T*D*F
```



### 7.4 Bytes

读输入：
$$
2BTD
$$
读三个权重矩阵：
$$
3\cdot 2DF = 6DF
$$
写中间结果 `U` 和 `G`：
$$
2BTF + 2BTF = 4BTF
$$
写最终输出 `Y`：
$$
2BTD
$$
所以：
$$
\mathrm{Bytes}_{\mathrm{MLP}}
=
4BTD + 4BTF + 6DF
$$
`lecture_10.py` 中对应代码也 assert 了这个表达式。

### 7.5 Arithmetic intensity

$$
\mathrm{AI}_{\mathrm{MLP}}
=
\frac{6BTDF}
{4BTD + 4BTF + 6DF}
$$

当 `D,F \gg BT` 时，分母主要由权重读取 `6DF` 主导：
$$
\mathrm{AI}_{\mathrm{MLP}}
\approx
\frac{6BTDF}{6DF}
=
BT
$$
于是：

#### Prefill 阶段

prefill 中 `T=S`：
$$
\mathrm{AI}_{\mathrm{MLP,prefill}}
\approx
BS
$$
只要 prompt 足够长或 batch 足够大，MLP prefill 容易达到较高 arithmetic intensity。

#### Generation 阶段

generation 中 `T=1`：
$$
\mathrm{AI}_{\mathrm{MLP,generation}}
\approx
B
$$
这意味着 MLP generation 的效率取决于并发请求数 `B`。如果只有一个请求，MLP 也可能 memory-bound；如果有 continuous batching，让很多请求共享同一批权重读取，MLP 的硬件利用率可以提高。

### 7.6 学习提示

MLP 的关键结论是：

> **MLP 的权重对 batch 中所有 sequence 共享，因此增大 batch 可以摊薄读权重的成本。**

这为后面 attention 的反差做准备：attention 的 KV cache 是每个 sequence 独有的，batching 不能像 MLP 那样复用同一份 KV。

------

## 8. 第七部分：Attention 层的 arithmetic intensity——本讲最关键瓶颈

这一节要解决的问题是：**为什么 generation attention 几乎必然 memory-bound？为什么 batch size 对它帮助不大？**

### 8.1 问题引入

Attention 是 Transformer 能利用上下文的核心机制，也是推理阶段最麻烦的部分。尤其在 generation 中，每个新 token 都需要 attend 到已有历史 token。历史越长，KV cache 越大，需要从 HBM 读取的数据越多。

课程在 `lecture_10.py` 中专门分析 attention 的 FLOPs、bytes 和 arithmetic intensity，并得到最重要的公式：
$$
\mathrm{AI}_{\mathrm{attn}}
=
\frac{ST}{S+T}
$$


### 8.2 Attention FLOPs

假设：

- `S`：已有上下文 token 数；
- `T`：当前要计算 logits 的 token 数；
- `D`：隐藏维度，先把多头展平理解。

第一步计算：
$$
QK^\top
$$
FLOPs 约为：
$$
2BSTD
$$
第二步 attention weights 乘 `V`：
$$
\mathrm{softmax}(QK^\top)V
$$
FLOPs 约为：
$$
2BSTD
$$
总 FLOPs：
$$
\mathrm{FLOPs}_{\mathrm{attn}}
=
4BSTD
$$
`lecture_10.py` 中也检查：

```
assert flops == 4*B*S*T*D
```



### 8.3 Attention Bytes

使用 FlashAttention 类方法时，不把完整 attention matrix 写回 HBM，主要读写包括：

读 `Q`：
$$
2BTD
$$
读 `K`：
$$
2BSD
$$
读 `V`：
$$
2BSD
$$
写输出 `Y`：
$$
2BTD
$$
总 bytes：
$$
\mathrm{Bytes}_{\mathrm{attn}}
=
4BSD + 4BTD
$$
`lecture_10.py` 中也检查：

```
assert bytes_transferred == 4*B*S*D + 4*B*T*D
```



### 8.4 Attention arithmetic intensity

$$
\mathrm{AI}_{\mathrm{attn}}
=
\frac{4BSTD}{4BSD+4BTD}
$$

约去 `4BD`：
$$
\mathrm{AI}_{\mathrm{attn}}
=
\frac{ST}{S+T}
$$
这是本讲最重要的公式之一。

### 8.5 Prefill 阶段

prefill 中：
$$
T=S
$$
所以：
$$
\mathrm{AI}_{\mathrm{attn,prefill}}
=
\frac{S^2}{S+S}
=
\frac{S}{2}
$$
如果 prompt 有几百或几千 token，`S/2` 可以比较大，因此 prefill attention 更可能 compute-bound。

### 8.6 Generation 阶段

generation 中每一步只生成一个 token：
$$
T=1
$$
代入：
$$
\mathrm{AI}_{\mathrm{attn,generation}}
=
\frac{S}{S+1}
<
1
$$
当 `S` 很大时：
$$
\frac{S}{S+1}
\approx
1
$$
这仍然远低于 H100 约 295 的硬件阈值。因此 generation attention 几乎必然 memory-bound。`lecture_10.py` 中也明确写出 generation attention intensity `<1`，并总结 prefill is compute-bound, generation is memory-bound。

### 8.7 为什么 batching 对 generation attention 帮助不大？

注意：
$$
\mathrm{AI}_{\mathrm{attn}}
=
\frac{ST}{S+T}
$$
这个公式中没有 `B`。原因是 `B` 同时出现在 FLOPs 和 bytes 中，被抵消了。

对比 MLP 和 attention：

| 部分      | batch 增大后是否复用同一份大数据 | 为什么                                    |
| --------- | -------------------------------- | ----------------------------------------- |
| MLP       | 是                               | batch 中所有 sequence 使用同一组 MLP 权重 |
| Attention | 否                               | 每个 sequence 有自己的 KV cache           |

**工程直觉：**
 对 MLP 来说，读一次 `W_up/W_gate/W_down`，可以给很多请求一起用。对 attention 来说，每个请求都有自己的上下文历史，因此 batch 增大意味着读更多不同的 KV cache，而不是复用同一份 KV cache。

### 8.8 学习提示

本节必须记住的结论：
$$
\mathrm{AI}_{\mathrm{MLP,generation}}\approx B
$$
但：
$$
\mathrm{AI}_{\mathrm{attn,generation}}
=
\frac{S}{S+1}
<
1
$$
因此：

> **generation 阶段的核心瓶颈不是 MLP，而是 attention 读取 KV cache。**

------

## 9. 第八部分：Latency 与 Throughput 的理论模型

这一节要解决的问题是：**当 generation attention 是 memory-bound 时，如何用简单公式估计 latency 和 throughput？**

### 9.1 简化假设

课程采用一个简化模型：

- 每个 generation step 都要读取参数和 KV cache；
- 计算与通信可以理想 overlap；
- 忽略 kernel launch、调度、网络通信等额外开销；
- step time 主要由内存传输量除以 HBM bandwidth 决定。

`lecture_10.py` 中定义了 `TransformerPerformanceStats`，字段包括参数量、总内存、生成一个 token 的 latency 和 tokens/s throughput；性能函数把 memory 写成参数 bytes 加 batch 中所有序列的 KV cache，并用 memory / memory_bandwidth 估计 latency，用 `B / latency` 估计 throughput。

### 9.2 公式定义

设：
$$
P = \text{parameter bytes}
$$
则 batch size 为 `B` 时，每步需要读取的近似内存量为：
$$
\mathrm{Memory}(B)
=
P + BC
$$
Latency：
$$
\mathrm{Latency}(B)
=
\frac{P+BC}{W_{\mathrm{mem}}}
$$
因为一步为 `B` 个 sequence 各生成一个 token，所以 throughput 是：
$$
\mathrm{Throughput}(B)
=
\frac{B}{\mathrm{Latency}(B)}
=
\frac{B W_{\mathrm{mem}}}{P+BC}
$$

### 9.3 Batch size 的影响

对 latency：
$$
\frac{d}{dB}
\mathrm{Latency}(B)
=
\frac{C}{W_{\mathrm{mem}}}
>
0
$$
所以 batch 越大，单步 latency 越差。

对 throughput：
$$
\frac{d}{dB}
\mathrm{Throughput}(B)
=
\frac{W_{\mathrm{mem}}P}{(P+BC)^2}
>
0
$$
所以 batch 越大，总 throughput 越好。

但吞吐增益会递减：
$$
\lim_{B\to\infty}
\mathrm{Throughput}(B)
=
\frac{W_{\mathrm{mem}}}{C}
$$
这说明最终吞吐上限由每个 sequence 的 KV cache 大小 `C` 决定。

### 9.4 LLaMA 2 13B on H100 例子

`lecture_10.py` 中的 LLaMA 2 13B 配置为：
$$
S=1024,\quad D=5120,\quad F=13824
$$
memory bandwidth 为：
$$
3.35\times10^{12}\ \mathrm{bytes/s}
$$
这些参数在 `llama2_13b_config()` 中给出。

#### 参数量公式

`lecture_10.py` 中参数量公式为：
$$
\#\mathrm{params}
=
2VD
+
3DFL
+
(2DNH+2DKH)L
$$
逐项解释：

| 项             | 含义                                                         |
| -------------- | ------------------------------------------------------------ |
| `2VD`          | embedding 与 unembedding                                     |
| `3DFL`         | 每层 MLP 的 up、gate、down 三组矩阵                          |
| `(2DNH+2DKH)L` | attention 中 query/output projection 与 key/value projection |

bf16 参数 bytes：
$$
P
=
2\#\mathrm{params}
$$

#### KV cache per sequence

$$
C
=
S(KH)L\cdot 2\cdot 2
$$

对原始 MHA 设置 `K=40`：
$$
C
=
1024\cdot(40\cdot128)\cdot40\cdot4
=
838{,}860{,}800\ \mathrm{bytes}
$$
约为：
$$
0.839\ \mathrm{GB}
$$

### 9.5 不同 batch size 的结果

| 设置         | 总内存    | latency        | throughput  | 学习解释                         |
| ------------ | --------- | -------------- | ----------- | -------------------------------- |
| `B=1,K=40`   | 26.87 GB  | 8.02 ms/token  | 124.7 tok/s | 单用户响应快，但系统吞吐低       |
| `B=64,K=40`  | 79.72 GB  | 23.80 ms/token | 2689 tok/s  | 吞吐显著提高，但接近 80GB 显存   |
| `B=256,K=40` | 240.78 GB | 71.87 ms/token | 3562 tok/s  | 理论吞吐更高，但单卡 H100 放不下 |

`lecture_10.py` 中也明确检查 `B=256` 时内存超过 80GB H100 memory，并总结 batch size 增大导致 latency 变差、throughput 变好且收益递减。

### 9.6 学习提示

本节的核心不是记某个 LLaMA 2 数字，而是理解：
$$
\mathrm{Latency}
\propto
P+BC
$$
所以推理优化有两条最直接路线：

1. 减小参数 bytes `P`；
2. 减小每个 sequence 的 KV cache bytes `C`。

在 generation attention memory-bound 的前提下，减小 `C` 往往特别关键。

------

## 10. 第九部分：减少 KV cache——架构层面的推理加速

这一节要解决的问题是：**如果 KV cache 是 generation 的核心内存瓶颈，哪些架构方法可以减少它？**

`lecture_10.py` 在 `reduce_kv_cache_size()` 中明确说：memory 是 inference 的 bottleneck，因此要减少 KV cache，但同时要确保 accuracy 不要下降太多。

### 10.1 Grouped Query Attention, GQA

#### 问题引入

标准 multi-head attention 中，每个 query head 都有自己的 key/value head。这给模型表达能力带来好处，但 KV cache 也会很大。

GQA 的问题意识是：

> 是否每个 query head 都必须拥有独立的 key/value head？

#### MHA、MQA、GQA 对比

| 方法 | KV heads 数 | 特点                                                         |
| ---- | ----------- | ------------------------------------------------------------ |
| MHA  | `K=N`       | 每个 query head 有自己的 KV head，KV cache 最大              |
| MQA  | `K=1`       | 所有 query heads 共享一个 KV head，KV cache 最小但可能损失质量 |
| GQA  | `1<K<N`     | query heads 分组，每组共享 KV head，折中质量与效率           |

`lecture_10.py` 中也写明：GQA 有 `N` 个 query heads，但只有 `K` 个 key/value heads，每个 KV head 与 `N/K` 个 query heads 交互；MHA 是 `K=N`，MQA 是 `K=1`，GQA 介于两者之间。

#### KV cache 缩小比例

KV cache 大小正比于：
$$
KH
$$
从 MHA 的 `K=N` 改为 GQA 的 `K` 后，缩小比例为：
$$
\frac{NH}{KH}
=
\frac{N}{K}
$$
即：
$$
\text{reduction factor}
=
\frac{N}{K}
$$
`lecture_10.py` 也明确写出 GQA reduces the KV cache by a factor of `N/K`。

#### 对 latency 和 throughput 的影响

因为：
$$
\mathrm{Latency}(B)
=
\frac{P+BC}{W_{\mathrm{mem}}}
$$
当 `C` 下降时：

- latency 降低；
- throughput 提高；
- 同样显存下可容纳更大 batch；
- 长上下文服务压力下降。

课程中的例子把 `K=40` 改成 `K=8`，相当于 1:5 的 KV head ratio。原报告中的结果为：

| 设置        | 总内存   | latency        | throughput  |
| ----------- | -------- | -------------- | ----------- |
| `B=64,K=8`  | 33.41 GB | 9.97 ms/token  | 6417 tok/s  |
| `B=256,K=8` | 65.63 GB | 19.59 ms/token | 13068 tok/s |

这说明 KV cache 一旦缩小，原本放不下的大 batch 可能变得可行。

**注意：**
 课程代码把 `K` 改小后，也会减少 key/value projection 参数，因此总参数 bytes 也略有变化；但本讲主要强调的是 KV cache 缩小对 generation memory traffic 的改善。

### 10.2 Multi-head Latent Attention, MLA

#### 直觉解释

GQA 是在 heads 之间共享 KV。MLA 更进一步：不直接保存完整 key/value，而是保存一个低维 latent vector，需要时再恢复成 key/value。

普通 attention：
$$
K_t = W_K h_t,\quad V_t = W_V h_t
$$
MLA：
$$
c_t = W_c h_t,\quad c_t\in\mathbb{R}^{C}
$$
缓存的是：
$$
c_t
$$
而不是完整的：
$$
K_t,V_t
$$
需要 attention 时再投影：
$$
K_t = W'_K c_t,\quad V_t = W'_V c_t
$$
如果：
$$
C\ll NH
$$
则 KV cache 大幅缩小。

#### DeepSeek v2 例子

`lecture_10.py` 中写到，DeepSeek v2 将 `N*H=16384` 压缩到 `C=512`；由于 MLA 与 RoPE 不直接兼容，还需要额外 64 维处理 RoPE，因此总缓存维度约为 576。

#### 为什么 RoPE 是 wrinkle？

RoPE 通常作用在 query/key 的位置相关维度上。如果不直接保存完整 key，而只保存 compressed latent，那么位置编码信息如何与重构后的 key 对齐就变得复杂。因此 MLA 需要额外处理位置相关维度。

**学习提示：**
 MLA 的核心系统收益仍然是：
$$
C_{\mathrm{KV}}\downarrow
\Rightarrow
\text{memory traffic}\downarrow
\Rightarrow
\text{latency}\downarrow,\quad
\text{throughput}\uparrow
$$
但它比 GQA 更深地改变 attention 结构，因此必须通过训练和评估确认质量。

### 10.3 Cross-Layer Attention, CLA

#### 直觉解释

GQA 在 heads 之间共享 KV；CLA 在 layers 之间共享 KV。

标准 Transformer 每一层都有独立的：
$$
K^{(\ell)},V^{(\ell)}
$$
所以 KV cache 大小正比于层数 `L`。

CLA 的想法是让某些层复用其他层的 KV。如果只有 `L_{\mathrm{KV}}` 个层真正存 KV，则 KV cache 可从：
$$
O(L)
$$
降低到：
$$
O(L_{\mathrm{KV}})
$$
理想缩小比例约为：
$$
\frac{L}{L_{\mathrm{KV}}}
$$
`lecture_10.py` 中将 CLA 概括为 share KVs across layers，并指出它在经验上改善 accuracy 与 KV cache size 的 Pareto frontier。

### 10.4 Local / Sliding Window Attention

#### 问题引入

前面的方法压缩 KV 的 head/layer/latent 维度。另一条路线是减少可见历史长度。

标准 full attention 中，新 token 可以看所有 `S` 个历史 token。Sliding window attention 设置窗口大小 `W`，每个 token 只看最近 `W` 个 token：
$$
\mathrm{Attend}(x_t)
=
\{x_{t-W},\dots,x_{t-1}\}
$$
这样每层最多保留最近 `W` 个 token 的 KV：
$$
C_{\mathrm{KV,local}}
=
O(WLKH)
$$
而不是：
$$
O(SLKH)
$$
当 `W\ll S` 时，节省巨大。

`lecture_10.py` 中也把 local/sliding window attention 的想法写成：只看 local context，effective context 随层数线性增长，KV cache independent of sequence length，但也提醒这可能 hurt accuracy，需要和 global attention 混合。

#### 为什么 effective context 可能超过窗口大小？

第 1 层只能看局部窗口；第 2 层看到的局部 token 已经混合了各自更早的局部信息。信息可以逐层向远处传播，因此粗略直觉是：
$$
\text{effective receptive field}
\sim
L_{\mathrm{local}}\cdot W
$$
但这不是 full attention。对于需要精确检索长距离信息的任务，local attention 可能不够。

#### Hybrid attention

为了降低 KV cache 又不完全丢失长程能力，实际模型可以：

- 部分层使用 sliding window attention；
- 部分层保留 global/full attention。

这样在系统效率和长上下文能力之间折中。

### 10.5 Linear Attention、SSM、Diffusion 等方向

课程最后也把 linear attention、state-space models 和 diffusion models 列为可能减少或绕开 KV cache 的方向。

它们共同的思路是：不要显式保存所有历史 token 的 KV cache，而是把历史压缩成某种固定大小或更低复杂度的 state：
$$
s_t = f(s_{t-1}, x_t)
$$
生成时读：
$$
s_t
$$
而不是读全部历史 KV。

**补充说明：**
 这类方法并不是免费午餐。对于需要精确长距离检索的任务，例如 needle-in-a-haystack，把历史压缩成小 state 可能会丢失细节。Full attention 的优势仍然是可以对任意历史位置做精细访问。

### 10.6 DeepSeek v4 attention：压缩、稀疏、再压缩

`lecture_10.py` 简要列出 DeepSeek v4 attention，包括：

- Compressed Sparse Attention, CSA：每 `m` 个 token 压缩成 1 个；
- DeepSeek Sparse Attention, DSA：选择 top `k`；
- Heavily Compressed Attention, HCA：进一步压缩；
- 支持 1M context length。

这些方法的系统目标仍然一致：
$$
\text{少读 KV cache}
\Rightarrow
\text{少搬内存}
\Rightarrow
\text{generation 更快}
$$
但它们改变了模型访问历史的方式，因此 accuracy、长上下文能力和任务适用性都必须通过实验验证。

### 10.7 学习提示

这一节可以用一个总表复习：

| 方法                        | 缩减对象        | 核心公式/比例        | 主要风险                  |
| --------------------------- | --------------- | -------------------- | ------------------------- |
| GQA                         | KV heads        | `N/K`                | 共享过度可能损失质量      |
| MLA                         | KV 表示维度     | `NH → C`             | 架构更复杂，RoPE 处理困难 |
| CLA                         | layers          | `L → L_KV`           | 跨层共享可能影响表达能力  |
| Sliding window              | 上下文长度      | `S → W`              | 长距离精确检索变弱        |
| Linear/SSM                  | 历史表示        | `O(S) → O(1)` 或低阶 | 压缩历史可能丢信息        |
| Sparse/compressed attention | 可访问 token 数 | 压缩与选择           | 需要验证长上下文质量      |

------

## 11. 第十部分：Quantization——减少每个数的 bytes

这一节要解决的问题是：**如果推理瓶颈来自内存读写，是否可以让每个数占更少字节？**

### 11.1 基本思想

Quantization 的核心是降低数值精度。原来每个数可能是 bf16，占 2 bytes；量化后可能是 int8，占 1 byte，甚至 int4，占 0.5 byte。

如果参数和 KV cache 都需要从 HBM 读取，那么每个数占用更少 bytes，就可以减少内存传输量，从而改善 memory-bound 的 generation。

`lecture_10.py` 的 `quantization()` 函数也明确写出关键思想：reduce the precision of numbers；less memory means better latency/throughput，但需要担心 accuracy。

### 11.2 基本量化公式

给定实数 `x`、scale `s`、zero point `z`，以及整数范围 `[q_min,q_max]`，量化为：
$$
q
=
\mathrm{clip}
\left(
\mathrm{round}\left(\frac{x}{s}\right)+z,\,
q_{\min},q_{\max}
\right)
$$
反量化为：
$$
\hat{x}
=
s(q-z)
$$
课程代码示例中使用：
$$
x=5.2342,\quad s=0.1,\quad z=4
$$
先得到：
$$
q=\mathrm{round}(5.2342/0.1)+4
$$
再反量化：
$$
\hat{x}=(q-4)\cdot0.1
$$
这会产生小的数值误差。`lecture_10.py` 中对应代码为 `x_quant = round(x / scale) + zero_point` 和 `x_approx = (x_quant - zero_point) * scale`。

### 11.3 常见精度

`lecture_10.py` 列出常见数值格式：

| 格式 | bytes | 典型用途                        |
| ---- | ----- | ------------------------------- |
| fp32 | 4     | 训练中的参数和 optimizer states |
| bf16 | 2     | 常见推理默认精度                |
| fp8  | 1     | 更低内存，部分硬件支持          |
| int8 | 1     | 推理常见低精度整数              |
| int4 | 0.5   | 更激进压缩，精度风险更大        |



### 11.4 QAT 与 PTQ

#### Quantization-Aware Training, QAT

QAT 在训练时就模拟量化误差：
$$
x \rightarrow q(x) \rightarrow \hat{x}
$$
优点：

- 模型在训练中适应量化误差；
- 精度更容易保持。

缺点：

- 需要昂贵的大规模训练；
- 对已经训练好的大模型不一定方便。

#### Post-Training Quantization, PTQ

PTQ 在模型训练完成后再做量化。

优点：

- 成本低；
- 可用于已有模型。

缺点：

- 更容易损失精度；
- 需要 calibration data 来估计 scale 和 zero point。

`lecture_10.py` 中也按 QAT 与 PTQ 介绍：QAT 在 forward pass 中模拟量化误差；PTQ 在训练后进行，并用样本数据确定每层或每个 tensor 的 scale 和 zero point。

### 11.5 GPTQ 的思想

GPTQ 不是简单逐元素 round，而是希望量化后层输出尽量不变。

给定 calibration activations `X`、原始权重 `W`、量化权重 `\hat W`，目标近似为：
$$
\min_{\hat W}
\|XW-X\hat W\|_2^2
$$
展开后会出现近似 Hessian：
$$
H\approx X^\top X
$$
GPTQ 利用 Hessian 信息，在量化一部分权重后，对尚未量化的权重做补偿，使输出误差更小。`lecture_10.py` 对 GPTQ 的概括是：use Hessian information to update non-quantized weights to account for quantization error。

### 11.6 AWQ：Activation-Aware Quantization

AWQ 的观察是：有些 activation channels 很大，与这些 channels 相乘的权重对输出影响更大。

因此 AWQ 的思路是：

- 大多数权重量化到低精度；
- 少量重要权重保留高精度；
- 重要性根据 activation 大小判断。

`lecture_10.py` 写到 AWQ 会根据 activations 选择约 0.1%–1% 权重保持高精度，并给出 fp16 到 int3 带来 4x lower memory 和 3.2x speedup 的例子。

### 11.7 学习提示

Quantization 的复习重点：

| 问题           | 回答                                    |
| -------------- | --------------------------------------- |
| 它是什么？     | 用更低精度表示参数或激活                |
| 为什么需要？   | 减少 memory traffic 和显存占用          |
| 解决什么问题？ | memory-bound inference 中的数据搬运瓶颈 |
| 风险是什么？   | 数值误差导致 accuracy 下降              |
| 如何缓解？     | QAT、PTQ calibration、GPTQ、AWQ         |

------

## 12. 第十一部分：Model Pruning——删除结构，再用蒸馏修复

这一节要解决的问题是：**除了降低数值精度，能否直接让模型结构变小？**

### 12.1 基本思想

Quantization 是“每个数更便宜”；pruning 是“直接减少需要计算和存储的结构”。

课程把 pruning 说得很直观：把昂贵模型的一部分直接拿掉，让它更便宜，然后再修复。`lecture_10.py` 的 `model_pruning()` 也写到：rip out parts of an expensive model to make it cheaper, and then fix it up。

### 12.2 Pruning 的基本流程

`lecture_10.py` 给出的算法是：

1. 在小 calibration dataset 上识别重要的 layer、head、hidden dimension；
2. 移除不重要部分，得到更小模型；
3. 把原模型 distill 到剪枝后的模型中。

可以写成：

| 步骤         | 作用                                     |
| ------------ | ---------------------------------------- |
| calibration  | 判断哪些结构重要                         |
| pruning      | 删除低重要性层、head 或 hidden dimension |
| distillation | 用 teacher model 修复 student model      |

### 12.3 如何判断重要性？

**补充说明：**
 原报告提到课堂问答中的高层思路：可以看 calibration set 上的 activation magnitude。如果某个 hidden unit 或 channel 在很多样本上几乎总是接近 0，可能比较不重要；如果 activation 很大，删除它可能影响更大。

但 activation magnitude 不是万能指标。比如一个单元总是常数 100，它虽然没有变化，但可能相当于 bias-like effect，不能简单删除。更细致的 pruning 还需要考虑均值、方差、输出影响和后续层依赖。

### 12.4 Distillation 修复

剪枝后模型通常会退化，需要用 teacher model 修复。常见 knowledge distillation loss 是：
$$
\mathcal{L}_{\mathrm{KD}}
=
\tau^2
\mathrm{KL}
\left(
\mathrm{softmax}\left(\frac{z_{\mathrm{teacher}}}{\tau}\right)
\;\|\;
\mathrm{softmax}\left(\frac{z_{\mathrm{student}}}{\tau}\right)
\right)
$$
其中：

- `z_teacher` 是原模型 logits；
- `z_student` 是剪枝后模型 logits；
- `\tau` 是 temperature；
- KL divergence 衡量两个输出分布的差异。

**延伸理解：**
 剪枝像是把一个复杂机器拆掉一部分；distillation 像是让拆完后的机器重新模仿原机器工作。没有 distillation，剪枝后的模型往往会有明显性能损失。

### 12.5 学习提示

Pruning 的核心不是“删得越多越好”，而是：
$$
\text{删掉低价值结构}
+
\text{用蒸馏恢复行为}
$$
它和 quantization 的区别是：

| 方法         | 改变对象                   | 是否改变结构 |
| ------------ | -------------------------- | ------------ |
| Quantization | 数值精度                   | 通常不改变   |
| Pruning      | 层、head、hidden dimension | 改变模型结构 |

------

## 13. 第十二部分：Speculative Sampling / Speculative Decoding

这一节要解决的问题是：**能否不改变目标模型分布，却让 generation 更快？**

### 13.1 基本动机

前面的方法如 GQA、MLA、量化、剪枝，很多都会改变模型结构或数值表示，可能损失 accuracy。

Speculative decoding 的特点是：它可以在数学上保持从目标模型 `q` 精确采样，同时加速生成。

课程强调一个不对称性：

- **Prefill/checking**：给定一串 token，让大模型并行计算这些 token 的概率，相对快；
- **Generation**：让大模型一个 token 一个 token 地自回归采样，慢。

`lecture_10.py` 中也明确说：prefill 能并行编码，generation 一次生成一个 token；换句话说，checking is faster than generation。

### 13.2 算法直觉

Speculative decoding 使用两个模型：

- draft model `p`：小、便宜、快，用来猜 token；
- target model `q`：大、昂贵、准确，用来验证 token。

流程：

1. draft model 先猜若干 token；
2. target model 并行检查这些 token；
3. 如果 draft 猜得好，一次接受多个 token；
4. 如果猜错，用 residual distribution 修正，保证最终分布仍然等于 `q`。

`lecture_10.py` 中也写到：用更便宜的 draft model `p` 猜几个 token，用 target model `q` 并行评估并接受；这是 modified rejection sampling，关键性质是 guaranteed exact sample from target model。

### 13.3 单 token 版本

给定目标分布 `q(y)` 和 draft 分布 `p(y)`。

先从 draft 采样：
$$
y\sim p
$$
接受概率：
$$
a(y)
=
\min\left(1,\frac{q(y)}{p(y)}\right)
$$
如果拒绝，则从 residual distribution 采样：
$$
r(y)
=
\frac{[q(y)-p(y)]_+}
{\sum_v [q(v)-p(v)]_+}
$$
其中：
$$
[x]_+ = \max(x,0)
$$

### 13.4 为什么输出分布精确等于 `q`？

输出为 `y` 的概率分成两部分。

第一部分：draft 采到 `y`，并被接受：
$$
p(y)\min\left(1,\frac{q(y)}{p(y)}\right)
=
\min(p(y),q(y))
$$
第二部分：draft 采到某个 token 后被拒绝，再从 residual 采到 `y`。

总拒绝概率是：
$$
P(\mathrm{reject})
=
1-\sum_v \min(p(v),q(v))
$$
由于 `p` 和 `q` 都是概率分布：
$$
\sum_v [p(v)-q(v)]_+
=
\sum_v [q(v)-p(v)]_+
=
1-\sum_v \min(p(v),q(v))
$$
所以第二部分为：
$$
P(\mathrm{reject})r(y)
=
[q(y)-p(y)]_+
$$
总输出概率：
$$
P(\mathrm{output}=y)
=
\min(p(y),q(y))
+
[q(y)-p(y)]_+
=
q(y)
$$
因此 speculative sampling 是 **lossless** 的：它不会改变目标模型 `q` 的采样分布。

### 13.5 二元 vocabulary 例子

`lecture_10.py` 用 `{A,B}` 做例子：假设 draft model 对 `A` 过采样，即 `p(A)>q(A)`，那么 `p(B)<q(B)`。Residual distribution 会补偿 `B`。代码课件中写出：
$$
P[\mathrm{sampling\ A}]
=
p(A)\cdot \frac{q(A)}{p(A)}
=
q(A)
$$
并给出 `B` 的对应推导，最终保证采样分布等于 target model。

### 13.6 多 token speculative decoding

实际中，draft model 一次生成 `K` 个候选 token：
$$
y_1,\dots,y_K
$$
target model 用一个并行 pass 计算这些 token 的 logits，然后从左到右检查每个 token：
$$
a_i
=
\min
\left(
1,
\frac{
q(y_i\mid x,y_{<i})
}{
p(y_i\mid x,y_{<i})
}
\right)
$$
如果连续接受多个 token，就相当于一次 target model pass 产出多个 token，从而提高 tokens/s。

接受率取决于 `p` 和 `q` 的接近程度。越接近，越容易接受更多 token。`lecture_10.py` 也提到实践中可使用 70B target + 8B draft，或 8B target + 1B draft，并通过 distillation 让 draft model 更接近 target。

### 13.7 Medusa 与 EAGLE

`lecture_10.py` 还列出两个扩展方向：

- Medusa：draft model 并行生成多个 token；
- EAGLE：draft model 利用 target model 的高层特征。

这些方法的共同目标是提升 draft 质量或并行预测能力，从而提高接受率。

### 13.8 学习提示

Speculative decoding 的关键是：

| 问题                     | 回答                                           |
| ------------------------ | ---------------------------------------------- |
| 为什么能加速？           | 小模型先猜，大模型并行验证                     |
| 为什么不损失分布正确性？ | modified rejection sampling 保证输出仍来自 `q` |
| 接受率取决于什么？       | draft model 与 target model 的接近程度         |
| 和 GQA/量化不同在哪里？  | 它可以保持 target model 的精确采样分布         |

一句话总结：

> **Speculative decoding 利用“验证一串 token 比逐个生成更快”的不对称性，用数学修正确保最终仍从目标模型采样。**

------

## 14. 第十三部分：动态在线服务、Continuous Batching 与 Selective Batching

这一节要解决的问题是：**真实在线流量为什么比离线 batch 更难？推理服务如何动态调度请求？**

### 14.1 动态 workload 的困难

训练时，batch 往往是规则张量：
$$
[B,S,H]
$$
每个样本长度相同，或者 padding 到相同长度。

在线推理不同：

- 请求在不同时间到达；
- prompt 长度不同；
- 输出长度不同；
- 有些请求先结束，有些请求继续生成；
- 多个请求可能共享 system prompt；
- 同一个 prompt 可能采样多个 response。

`lecture_10.py` 在 live traffic batching 部分列出三个困难：requests arrive at different times、sequences have shared prefixes、sequences have different lengths。

### 14.2 Continuous batching

传统 static batching 会等凑够一批请求，再一起跑到整批结束。这会导致：

- 早到请求等待；
- 短请求被长请求拖住；
- GPU 利用率不稳定。

Continuous batching 的思想是 **iteration-level scheduling**：

1. 当前活跃 batch 中每个请求生成一个 token；
2. 已结束请求移出 batch；
3. 新到请求加入 batch；
4. 下一轮 decoding 继续。

`lecture_10.py` 中对应描述是 decode step by step，并在请求到达时加入 batch，不需要等旧 batch 完全生成结束。

**工程直觉：**
 Continuous batching 像一个不断上下客的电梯，而不是一辆必须等所有乘客到达并且所有乘客同时下车的大巴。

### 14.3 Selective batching

问题是：不同请求长度不同，attention 依赖 sequence length，难以直接拼成规则张量。

例如三个请求长度为：
$$
3,\quad 9,\quad 5
$$
Attention 部分需要分别处理每个 sequence 的 KV cache，因为每个请求历史不同。

但 non-attention 部分，如 MLP、layernorm、projection，只作用在每个 token 的 hidden dimension 上，不需要 token 之间两两交互。因此可以把不同长度的 token 拼接起来：
$$
[3,H], [9,H], [5,H]
\Rightarrow
[3+9+5,H]
$$
`lecture_10.py` 中也写到：attention computation 分开处理；non-attention computation 把所有 sequences 拼接成 `[3+9+5,H]`。

### 14.4 学习提示

Continuous batching 与 selective batching 的区别：

| 方法                | 解决的问题            | 核心思想                                   |
| ------------------- | --------------------- | ------------------------------------------ |
| Continuous batching | 请求到达/结束时间不同 | 每个 decoding iteration 动态加入/移出请求  |
| Selective batching  | 请求长度不同          | attention 单独处理，non-attention 拼接处理 |

这部分体现出推理服务不仅是模型数学问题，也是系统调度问题。

------

## 15. 第十四部分：PagedAttention——用操作系统分页思想管理 KV cache

这一节要解决的问题是：**KV cache 不仅大，而且动态增长；如何减少内存碎片并共享前缀？**

### 15.1 为什么 KV cache 管理困难？

每个请求都需要 KV cache，但请求长度和输出长度不确定。如果系统为每个请求提前预留最大长度的连续内存，就会浪费大量空间。

`lecture_10.py` 在 `paged_attention()` 中描述旧做法：请求到来后，为 prompt 和 response 分配一段 KV cache，直到最大长度。随后指出这会产生 fragmentation。

### 15.2 内部碎片 internal fragmentation

假设系统为某请求预留最大长度 `M`，实际只生成 `\ell` 个 token。每 token KV bytes 为：
$$
b_{\mathrm{token}}
=
2LKHr
$$
浪费为：
$$
(M-\ell)b_{\mathrm{token}}
$$
这叫 internal fragmentation：分配给某个请求的内存块内部没有用完。

### 15.3 外部碎片 external fragmentation

如果不同请求占用连续内存块，释放后可能留下很多小洞。这些小洞总量不小，但单个洞太小，无法容纳新请求，这叫 external fragmentation。

`lecture_10.py` 也明确列出：可能生成更少 token 导致 internal fragmentation，section 之间可能出现 unused space 导致 external fragmentation。

### 15.4 PagedAttention 的解决办法

PagedAttention 借鉴操作系统分页思想，把每个 sequence 的 KV cache 切成固定大小 blocks/pages。

逻辑上，一个 sequence 的 token 是连续的；物理上，这些 blocks 可以分散存储。

如果 block size 为 `P`，每个 sequence 最多只在最后一个 block 浪费：
$$
(P-1)b_{\mathrm{token}}
$$
而不需要按最大长度 `M` 提前预留全部连续空间。

`lecture_10.py` 中写到：PagedAttention 把一个 sequence 的 KV cache 切成 non-contiguous blocks。

### 15.5 共享 prefix 与 copy-on-write

PagedAttention 还能支持多个请求共享相同前缀。例如：

- 多个请求共享同一个 system prompt；
- 同一个 prompt 采样多个 response；
- program synthesis 或 agent 搜索中产生多个分支。

如果两个请求前缀相同，它们可以指向同一批 KV blocks。只有当某个请求开始生成不同 token 时，才在分叉点复制新 block，这就是 **copy-on-write**。

`lecture_10.py` 中也总结：share prefixes, copy-on-write at the block level，并列出 sharing system prompt 和 sampling multiple responses per prompt。

### 15.6 Kernel 与工程优化

在 PagedAttention 之后，`lecture_10.py` 还列出 vLLM 相关工程优化：

- 融合 block read 和 attention，减少 kernel launch overhead；
- 使用 FlashAttention、FlashDecoding 等新 kernel；
- 使用 CUDA graphs 避免 kernel launch overhead。

**工程直觉：**
 LLM 推理系统的优化是多层次的：架构减少 KV cache，量化减少 bytes，speculative decoding 减少昂贵 target step，batching 提高利用率，PagedAttention 管理动态内存，kernel 优化减少底层调度开销。

### 15.7 学习提示

PagedAttention 的复习重点：

| 概念                   | 含义                  | 为什么重要                             |
| ---------------------- | --------------------- | -------------------------------------- |
| internal fragmentation | 预留空间内部没用完    | 浪费显存                               |
| external fragmentation | 空洞分散，难以复用    | 降低可用内存                           |
| blocks/pages           | KV cache 分块存储     | 避免大块连续分配                       |
| prefix sharing         | 多请求共享相同前缀 KV | 节省系统 prompt、multi-sample 场景内存 |
| copy-on-write          | 分叉时再复制          | 共享与独立生成兼顾                     |

------

## 16. Python 代码示例详解

这一节集中解释 `lecture_10.py` 的代码结构。它不是普通训练代码，而是一份 **可执行课件**：通过符号计算、assert、图表调用和示例配置，帮助学生理解 Transformer inference 的系统瓶颈。

### 16.1 文件整体结构

`lecture_10.py` 的 `main()` 函数组织了本讲顺序：

1. `landscape()`：说明 inference workload 与指标；
2. `review_transformer()`：回顾 Transformer 符号和结构；
3. `review_of_arithmetic_intensity()`：回顾 arithmetic intensity；
4. `arithmetic_intensity_of_inference()`：推导 MLP 与 attention 的推理强度；
5. `throughput_and_latency()`：用 LLaMA 2 13B/H100 估算 latency 和 throughput；
6. `reduce_kv_cache_size()`：讨论 GQA、MLA、CLA、local attention 等；
7. `quantization()`：量化；
8. `model_pruning()`：剪枝与蒸馏；
9. `speculative_sampling()`：投机采样；
10. `continuous_batching()`：动态 batching；
11. `paged_attention()`：PagedAttention 与 KV cache 内存管理。

这与原报告的章节顺序基本一致，因此本次改写保持了课程主线。

### 16.2 关键函数逐段解析

| 函数                                            | 输入/依赖                       | 输出/作用                                      | 对应课程概念                 |
| ----------------------------------------------- | ------------------------------- | ---------------------------------------------- | ---------------------------- |
| `landscape()`                                   | 无显式输入                      | 介绍 inference 场景、TTFT、latency、throughput | 推理 workload                |
| `review_transformer()`                          | 符号 `B,T,D,H` 等               | 统一形状记号和 Transformer 约定                | 张量形状、Transformer block  |
| `review_of_arithmetic_intensity()`              | 矩阵乘法形状 `B,D,F`            | 推导 matmul AI 约为 `B`                        | roofline、memory-bound       |
| `arithmetic_intensity_of_inference()`           | `B,S,T,D,F` 等符号              | 推导 MLP 与 attention AI                       | prefill vs generation        |
| `compute_transformer_performance_stats(config)` | 模型配置                        | 计算参数量、总内存、latency、throughput        | 推理性能模型                 |
| `llama2_13b_config(args={})`                    | 可覆盖配置                      | 返回 LLaMA 2 13B 形状参数                      | 数值例子                     |
| `throughput_and_latency()`                      | 调用性能函数                    | 比较 `B=1,64,256`                              | latency/throughput trade-off |
| `reduce_kv_cache_size()`                        | 模型结构引用与配置              | 展示 GQA、MLA、CLA、local attention 等         | KV cache 优化                |
| `quantization()`                                | 示例数值 `x, scale, zero_point` | 演示量化/反量化                                | 低精度推理                   |
| `model_pruning()`                               | calibration dataset 概念        | 介绍剪枝流程                                   | 结构压缩                     |
| `speculative_sampling()`                        | draft `p` 与 target `q`         | 说明精确采样机制                               | lossless 加速                |
| `continuous_batching()`                         | 动态请求                        | iteration-level scheduling                     | 在线 serving                 |
| `paged_attention()`                             | KV cache blocks                 | 分页、共享前缀、copy-on-write                  | 动态内存管理                 |

### 16.3 `compute_transformer_performance_stats` 解析

这个函数是本讲最重要的代码之一。它接收一个 Transformer 配置，计算：

```
num_params = 2*V*D + D*F*3*L + (2*D*N*H + 2*D*K*H)*L
parameter_size = 2*num_params
kv_cache_size_per_seq = S * (K*H) * L * 2 * 2
memory = B * kv_cache_size_per_seq + parameter_size
latency = memory / memory_bandwidth
throughput = B / latency
```

这些代码分别对应：

| 代码变量                | 数学含义              | 学习重点                           |
| ----------------------- | --------------------- | ---------------------------------- |
| `num_params`            | 参数量                | embedding、MLP、attention 参数求和 |
| `parameter_size`        | bf16 参数 bytes       | 每个参数 2 bytes                   |
| `kv_cache_size_per_seq` | 单序列 KV cache bytes | `S(KH)L*2*2`                       |
| `memory`                | 单步读取总内存        | 参数 + batch 内 KV cache           |
| `latency`               | 生成一个 token 时间   | memory / bandwidth                 |
| `throughput`            | tokens/s              | batch 中一次生成 `B` 个 token      |

对应 GitHub 代码明确写出了这些表达式。

### 16.4 代码与课程知识点对应表

| 课程知识点              | 对应代码函数 / 变量 / 片段                | 代码功能                          | 对应理论原理                | 在课程中的作用                 |
| ----------------------- | ----------------------------------------- | --------------------------------- | --------------------------- | ------------------------------ |
| 推理 workload           | `landscape()`                             | 列出 inference 场景和指标         | 推理是重复成本              | 引出为什么 inference 重要      |
| TTFT/latency/throughput | `landscape()`, `throughput_and_latency()` | 定义快的不同含义                  | 服务指标分解                | 指导优化目标                   |
| matmul AI               | `review_of_arithmetic_intensity()`        | 统计 `2BDF` FLOPs 与 bytes        | AI 约为 `B`                 | 说明小 batch memory-bound      |
| MLP AI                  | `arithmetic_intensity_of_inference()`     | assert `6BTDF` 与 `4BTD+4BTF+6DF` | AI 约为 `BT`                | 说明 MLP 可被 batching 改善    |
| Attention AI            | `arithmetic_intensity_of_inference()`     | assert `4BSTD` 与 `4BSD+4BTD`     | AI 为 `ST/(S+T)`            | 得出 generation attention `<1` |
| KV cache 大小           | `kv_cache_size_per_seq`                   | 计算单序列 KV cache               | `S(KH)L*2*2`                | 后续优化核心                   |
| LLaMA 2 13B             | `llama2_13b_config()`                     | 给出 `S,D,F,N,K,H,L,V`            | 数值实例                    | 展示显存与吞吐取舍             |
| GQA                     | `reduce_kv_cache_size()`                  | 改 `K=40` 为 `K=8`                | KV cache 缩小 `N/K`         | 展示架构优化收益               |
| Quantization            | `quantization()`                          | `x_quant`, `x_approx`             | scale/zero point            | 说明低精度误差                 |
| Pruning                 | `model_pruning()`                         | calibration、remove、distill      | 结构压缩 + 修复             | 减少模型成本                   |
| Speculative sampling    | `speculative_sampling()`                  | draft `p` + target `q`            | modified rejection sampling | lossless 加速                  |
| Continuous batching     | `continuous_batching()`                   | step-by-step scheduling           | 动态 batch                  | 处理在线流量                   |
| PagedAttention          | `paged_attention()`                       | blocks、prefix sharing            | paging、copy-on-write       | 管理 KV cache 碎片             |

### 16.5 学习提示

阅读 `lecture_10.py` 时，不要把它当成普通 Python 工程代码。它更像“可执行推导笔记”：

- `assert` 用来确认公式；
- `symbols()` 用来做符号推导；
- `config` 用来代入真实模型形状；
- `image()` 用来串联图表；
- 函数顺序就是课堂讲解顺序。

------

## 17. PDF / 幻灯片 / 图表解析

本次对话中没有提供独立 PDF 文件，因此本节不声称分析了单独 PDF。原始报告提到图表和课件说明；GitHub `lecture_10.py` 中通过 `image(...)` 调用了多张图。本节按这些图表调用整理读图方式。

### 17.1 主要图表总览

| 图表 / 代码位置                | 展示内容                       | 支持的课程结论                   | 对应代码或讲解                        |
| ------------------------------ | ------------------------------ | -------------------------------- | ------------------------------------- |
| `inference-schema.png`         | 推理整体流程                   | inference 是独立 workload        | `main()` 开头图表                     |
| transformer diagram            | Transformer 结构               | 回顾 shape 与 block              | `review_transformer()`                |
| naive inference 图             | 每步重复处理历史               | naive inference 复杂度高         | `arithmetic_intensity_of_inference()` |
| cached inference 图            | KV cache                       | 历史 KV 可复用                   | `arithmetic_intensity_of_inference()` |
| GQA 图                         | query heads 与 KV heads 分组   | KV cache 缩小 `N/K`              | `reduce_kv_cache_size()`              |
| MLA schema                     | latent KV 压缩                 | `NH → C`                         | `reduce_kv_cache_size()`              |
| CLA diagram/results            | 跨层共享 KV                    | 改善 KV/accuracy Pareto          | `reduce_kv_cache_size()`              |
| Longformer attention           | sliding window                 | KV cache 与 sequence length 解耦 | `reduce_kv_cache_size()`              |
| DeepSeek v4 attention          | 压缩稀疏 attention             | 长上下文中减少 KV 访问           | `reduce_kv_cache_size()`              |
| AWQ schema                     | activation-aware quantization  | 重要权重保高精度                 | `quantization()`                      |
| pruning-kd-loop                | 剪枝与蒸馏流程                 | 删除结构后修复                   | `model_pruning()`                     |
| speculative sampling algorithm | draft/target 验证流程          | 精确采样加速                     | `speculative_sampling()`              |
| continuous batching 图         | ragged 请求                    | 动态调度                         | `continuous_batching()`               |
| PagedAttention 图              | fragmentation、blocks、sharing | 分页和 copy-on-write             | `paged_attention()`                   |

### 17.2 关键图表解释

#### 17.2.1 Naive inference 与 cached inference 图

这组图想回答：为什么每次都重算历史 token 很浪费？

- naive inference 图展示每生成一个新 token，都重新处理完整 prefix；
- cached inference 图展示历史 key/value 被保存，下一步只追加新 token 的 KV；
- 这支撑了从 `O(T^3)` naive complexity 转向 KV cache 复用的结论。

`lecture_10.py` 中在说明 naive inference 之后，紧接着写出 solution: store KV cache in HBM，并调用 cached inference 图。

#### 17.2.2 GQA 图

GQA 图想回答：能否减少 KV heads，而不减少 query heads？

读图时要注意：

- query heads 仍然有 `N` 个；
- key/value heads 只有 `K` 个；
- 每个 KV head 被一组 query heads 共享；
- KV cache 缩小比例为 `N/K`。

该图支持的不是“GQA 一定无损”，而是“GQA 可以显著减少 KV cache，并需要检查 accuracy”。课程代码也在 GQA 后写出 check that accuracy doesn’t drop。

#### 17.2.3 MLA 图

MLA 图想说明：缓存对象从完整 KV 变成 compressed latent vector。

读图重点：

- 普通 attention 存 `K,V`；
- MLA 存 `c=W_c h`；
- 使用时再把 `c` 投影为 `K,V`；
- RoPE 需要额外处理。

这张图支持“降低 KV cache 维度”的结论，但它不是简单工程技巧，而是模型结构改变。

#### 17.2.4 PagedAttention 图

PagedAttention 图想回答：KV cache 动态增长时，如何避免碎片？

读图时要区分：

- 逻辑 sequence：token 顺序连续；
- 物理 blocks：内存中可不连续；
- prefix sharing：多个请求指向同一批 blocks；
- copy-on-write：分叉时才复制。

`lecture_10.py` 中依次展示 fragmentation、blocks、logical sharing 和 parallel/copy-on-write 图，并总结使用操作系统 paging 思想管理动态 workload 的内存。

### 17.3 学习提示

读本讲图表时要避免两个误解：

1. 图表展示的是系统或结构直觉，不一定包含所有实现细节；
2. 图表支持的是“为什么这个方向可能加速”，不是“这个方法一定不损失精度”。

尤其对 GQA、MLA、CLA、sliding window、quantization、pruning 来说，都需要评估 accuracy。

------

## 18. 关键公式与推导汇总

### 18.1 公式总表

| 公式                           | 符号解释                            | 适用场景                  | 直觉解释                              | 代码 / 例子对应                    |
| ------------------------------ | ----------------------------------- | ------------------------- | ------------------------------------- | ---------------------------------- |
| `AI = FLOPs / Bytes`           | FLOPs 为计算量，Bytes 为 HBM 读写量 | roofline 分析             | 每搬 1 byte 做多少计算                | `review_of_arithmetic_intensity()` |
| `AI_H100≈295`                  | H100 peak FLOPs / bandwidth         | 判断 compute/memory bound | 低于阈值多半 memory-bound             | H100 warm-up                       |
| `FLOPs_matmul=2BDF`            | `B×D` 乘 `D×F`                      | 普通矩阵乘法              | 每个输出元素做 `D` 长 dot product     | matmul warm-up                     |
| `AI_matmul≈B`                  | `D,F≫B`                             | 小 batch matmul           | batch 越小，权重复用越差              | warm-up                            |
| `C_KV,seq=4SLKH`               | bf16 单序列 KV cache                | KV cache 大小             | token、层、KV heads 线性增长          | `kv_cache_size_per_seq`            |
| `FLOPs_MLP=6BTDF`              | 三次 MLP matmul                     | gated MLP                 | up、gate、down 三组矩阵               | MLP 推导                           |
| `Bytes_MLP=4BTD+4BTF+6DF`      | 读输入/权重，写中间/输出            | MLP memory IO             | 权重读取常主导                        | MLP 推导                           |
| `AI_MLP≈BT`                    | `D,F≫BT`                            | MLP 推理                  | prefill 中 `T=S`，generation 中 `T=1` | MLP 推导                           |
| `FLOPs_attn=4BSTD`             | `QK^T` + `AV`                       | attention                 | 两次主要乘法                          | attention 推导                     |
| `Bytes_attn=4BSD+4BTD`         | 读 Q/K/V，写 Y                      | FlashAttention 简化       | 主要搬 KV                             | attention 推导                     |
| `AI_attn=ST/(S+T)`             | `B,D` 抵消                          | attention                 | batch 对 AI 无帮助                    | attention 推导                     |
| `AI_attn,prefill=S/2`          | `T=S`                               | prefill                   | 长 prompt 提高 AI                     | prefill                            |
| `AI_attn,generation=S/(S+1)<1` | `T=1`                               | generation                | 几乎必然 memory-bound                 | generation                         |
| `Latency=(P+BC)/W_mem`         | 参数 + KV cache / 带宽              | 推理性能模型              | 内存越多越慢                          | performance stats                  |
| `Throughput=B W_mem/(P+BC)`    | batch token 数 / latency            | 服务吞吐                  | 大 batch 摊参数成本                   | performance stats                  |
| `GQA reduction=N/K`            | query heads / KV heads              | GQA                       | KV heads 越少，cache 越小             | GQA                                |
| `q=round(x/s)+z`               | scale + zero point                  | quantization              | 实数映射到整数                        | quantization                       |
| `r(y)∝[q(y)-p(y)]_+`           | residual distribution               | speculative sampling      | 修正 draft 过采样                     | speculative decoding               |

### 18.2 最重要公式展开：Attention generation 为什么无法靠 batch 根治？

从 attention AI 开始：
$$
\mathrm{AI}_{\mathrm{attn}}
=
\frac{4BSTD}{4BSD+4BTD}
$$
约去 `4BD`：
$$
\mathrm{AI}_{\mathrm{attn}}
=
\frac{ST}{S+T}
$$
generation 中：
$$
T=1
$$
所以：
$$
\mathrm{AI}_{\mathrm{attn,generation}}
=
\frac{S}{S+1}
<
1
$$
这里没有 `B`，说明即使 batch size 增大，attention generation 的 arithmetic intensity 也不会像 MLP 那样线性提升。

**直觉总结：**
 batching 能让多个请求共享同一份 MLP 权重，但不能让多个请求共享各自不同的 KV cache。每个请求都带着自己的历史，因此 batch 越大，只是读更多 KV cache。

------

## 19. 概念对比与易混淆点

| 易混淆概念                                   | 区别                                                  | 联系                         | 常见误解                           | 正确理解                                               |
| -------------------------------------------- | ----------------------------------------------------- | ---------------------------- | ---------------------------------- | ------------------------------------------------------ |
| Prefill vs Generation                        | Prefill 处理已知 prompt；generation 逐 token 产生输出 | 都属于 inference             | 以为推理都不能并行                 | Prefill 可并行，generation 强自回归                    |
| Latency vs Throughput                        | latency 是单请求速度；throughput 是系统总 tokens/s    | batch size 同时影响二者      | 只优化 throughput 就等于用户体验好 | 在线交互要平衡 TTFT、latency、throughput               |
| Compute-bound vs Memory-bound                | 前者受算力限制，后者受内存带宽限制                    | 用 arithmetic intensity 判断 | FLOPs 高就一定慢                   | 低 AI 操作可能主要在等 HBM                             |
| MLP vs Attention                             | MLP 读共享权重；attention 读独立 KV cache             | 都在 Transformer block 中    | batching 对所有模块都同样有用      | batching 对 MLP 更有效，对 generation attention 不根治 |
| MHA vs MQA vs GQA                            | KV heads 数不同                                       | 都是 attention 变体          | MQA/GQA 一定无损                   | 减 cache 可能影响 accuracy                             |
| MLA vs GQA                                   | MLA 缓存 latent；GQA 共享 KV heads                    | 都减少 KV cache              | MLA 只是 GQA 的小改动              | MLA 改变表示和 RoPE 处理，更结构化                     |
| Sliding window vs Full attention             | 前者只看局部，后者看全部历史                          | 可混合使用                   | sliding window 等价 full attention | 长距离精确检索能力不同                                 |
| Quantization vs Pruning                      | 量化降精度；剪枝删结构                                | 都降低成本                   | 都只是压缩文件大小                 | 它们会影响推理内存、速度和精度                         |
| Speculative decoding vs Approximate decoding | speculative 可精确保持 target 分布                    | 都用小模型辅助               | 小模型猜错会改变分布               | residual 修正保证 exact sample                         |
| Continuous batching vs PagedAttention        | 前者调度请求；后者管理 KV cache 内存                  | 都服务动态 workload          | 都是同一种 batching 技巧           | 一个是 scheduling，一个是 memory management            |

------

## 20. 学习拓展与发散理解

### 20.1 与大模型训练实践的关系

本讲虽然主题是 inference，但它会反过来影响模型训练和架构设计。

#### 数据准备

如果模型部署场景需要长上下文，训练数据和评估任务就要覆盖长上下文能力。否则使用 sliding window、sparse attention 或 compressed attention 后，长距离任务可能出现质量问题。

#### 模型训练

GQA、MLA、CLA 等结构往往需要从训练阶段就确定。不能简单把一个普通 MHA 模型部署时硬改成 MLA，然后期待质量不变。

#### 优化策略

推理瓶颈提示我们：训练时追求更低 loss 之外，也要考虑部署时的 KV cache 成本、batching 方式、量化兼容性和 speculative decoding 的 draft model 训练。

#### 训练效率与评估效率

评估、RL rollout、agent trace 都需要大量生成。推理越快，实验迭代越快，训练后评估成本越低。

#### 工程成本

在真实服务中，每 token 成本会被请求量放大。KV cache 变小不仅降低 latency，还能让同一 GPU 同时服务更多请求。

### 20.2 与其他 lecture 的联系

- 与前面 systems/parallelism lecture 的联系：本讲继续使用 FLOPs、memory、arithmetic intensity、GPU bandwidth 等系统分析方法；
- 与 scaling laws lecture 的联系：模型越大、上下文越长，推理成本越需要预测和预算；
- 与 evaluation lecture 的联系：许多评估需要生成答案，因此 inference throughput 会影响评估速度；
- 与 data/post-training/RL lecture 的联系：SFT、RLHF/RLVR、agent rollout 都会产生大量推理 token。

**谨慎说明：**
 这里是按 CS336 整体课程主线做的学习联系总结；没有把未提供材料中的具体结论伪装成 Lecture 10 原文。

### 20.3 面试 / 考试 / 作业可能考察角度

| 问题                                                         | 考察点                      | 回答思路                                          |
| ------------------------------------------------------------ | --------------------------- | ------------------------------------------------- |
| 为什么 generation attention 是 memory-bound？                | attention AI 推导           | 从 `AI=ST/(S+T)` 出发，代入 `T=1` 得 `<1`         |
| 为什么 batch size 能改善 MLP generation，却不能根治 attention generation？ | 权重复用 vs KV cache 独立   | MLP 权重共享，attention KV 每请求独有             |
| KV cache 大小如何计算？                                      | shape 与 bytes              | `S·L·K·H·2·r`，bf16 为 `4SLKH`                    |
| GQA 为什么能加速推理？                                       | KV heads 缩减               | cache 正比 `K`，从 `N` 到 `K` 缩小 `N/K`          |
| MLA 和 GQA 有什么区别？                                      | cache 表示方式              | GQA 减 heads，MLA 存 latent                       |
| Speculative decoding 为什么不改变目标分布？                  | modified rejection sampling | 接受部分给 `min(p,q)`，拒绝 residual 补 `[q-p]_+` |
| Continuous batching 解决什么问题？                           | 动态请求调度                | iteration-level scheduling，新请求可中途加入      |
| PagedAttention 解决什么问题？                                | KV cache 内存碎片           | blocks/pages、prefix sharing、copy-on-write       |
| Quantization 的主要风险是什么？                              | 数值误差                    | 低精度减少 bytes，但可能损失 accuracy             |
| Pruning 后为什么要 distillation？                            | 结构删除后的修复            | student 模仿 teacher 分布，恢复性能               |

------

## 21. 课程复习总结

### 21.1 本讲核心思想

Lecture 10 的核心主线是：Transformer 推理并不是训练的简单 forward pass 重复。Prefill 阶段可以并行处理 prompt，通常更接近 compute-bound；generation 阶段必须逐 token 自回归，尤其 attention 需要不断读取每个请求独有的 KV cache，因此 arithmetic intensity 极低，几乎必然 memory-bound。于是，现代 LLM 推理优化的核心，就是减少、压缩、共享、分页或绕开 KV cache，同时通过量化、剪枝、speculative decoding 和动态 serving 系统提升效率。

### 21.2 必须掌握的概念

- **Inference workload**：真实使用、评估、RL rollout、agent 都需要大量生成；
- **TTFT**：第一个 token 出现前的等待时间，主要由 prefill 决定；
- **Latency**：单个请求生成 token 的速度；
- **Throughput**：系统总 tokens/s；
- **Prefill**：prompt 已知，可并行，较容易 compute-bound；
- **Generation**：输出逐步产生，强自回归，容易 memory-bound；
- **Arithmetic intensity**：FLOPs/byte，用于判断 compute-bound 或 memory-bound；
- **KV cache**：缓存历史 token 的 key/value，避免重复计算，但带来巨大内存读写；
- **GQA/MLA/CLA/sliding window**：从不同维度减少 KV cache；
- **Quantization**：降低每个数的 bytes；
- **Pruning**：删除模型结构并蒸馏修复；
- **Speculative decoding**：用小模型猜，大模型验，保持 target distribution；
- **Continuous batching**：动态请求按 decoding step 调度；
- **PagedAttention**：用分页思想管理 KV cache，支持 prefix sharing 和 copy-on-write。

### 21.3 必须理解的公式

$$
\mathrm{AI}
=
\frac{\mathrm{FLOPs}}{\mathrm{Bytes}}
$$

用于判断计算是否受内存带宽限制。
$$
C_{\mathrm{KV,seq}}
=
S\cdot L\cdot K\cdot H\cdot 2\cdot r
$$
用于估计单序列 KV cache 大小；bf16 下为：
$$
C_{\mathrm{KV,seq}}=4SLKH
$$
说明 MLP prefill 可通过长 prompt 或大 batch 提高强度；generation 中约为 `B`。
$$
\mathrm{AI}_{\mathrm{attn}}
=
\frac{ST}{S+T}
$$
说明 attention 的 batch size 被抵消。
$$
\mathrm{AI}_{\mathrm{attn,generation}}
=
\frac{S}{S+1}
<
1
$$
这是本讲最关键结论：generation attention 几乎必然 memory-bound。
$$
\mathrm{Latency}(B)
=
\frac{P+BC}{W_{\mathrm{mem}}}
$$
说明 batch size 增大带来 latency/throughput trade-off。
$$
\text{GQA KV reduction}
=
\frac{N}{K}
$$
说明减少 KV heads 可以直接缩小 KV cache。
$$
r(y)
=
\frac{[q(y)-p(y)]_+}
{\sum_v [q(v)-p(v)]_+}
$$
是 speculative sampling 拒绝后使用的 residual distribution，保证最终输出分布仍为 target model `q`。

### 21.4 必须看懂的代码

| 代码                                      | 学习重点                                                 |
| ----------------------------------------- | -------------------------------------------------------- |
| `review_of_arithmetic_intensity()`        | 如何从 FLOPs 和 bytes 推出 matmul AI 约为 `B`            |
| `arithmetic_intensity_of_inference()`     | 如何推导 MLP 与 attention 的 AI                          |
| `compute_transformer_performance_stats()` | 参数 bytes、KV cache bytes、latency、throughput 如何对应 |
| `llama2_13b_config()`                     | 如何把真实模型形状代入推理性能模型                       |
| `reduce_kv_cache_size()`                  | GQA、MLA、CLA、local attention 如何减少 KV cache         |
| `quantization()`                          | scale/zero point 量化与反量化                            |
| `model_pruning()`                         | calibration、remove、distill 的剪枝流程                  |
| `speculative_sampling()`                  | draft/target 分布如何保持 exact sampling                 |
| `continuous_batching()`                   | iteration-level scheduling                               |
| `paged_attention()`                       | blocks、fragmentation、prefix sharing、copy-on-write     |

### 21.5 一句话总结

**Transformer inference 的核心瓶颈在 generation 阶段读取 KV cache 导致的 memory-bound；推理优化的主线，就是在尽量不损失模型质量的前提下减少内存搬运，并用系统调度让硬件持续高效工作。**