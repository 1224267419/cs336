# Lecture 5 课程报告：从 GPU 硬件模型到 FlashAttention 的 IO-aware 性能优化

## 0. 资料来源与改写说明

本报告是在你上传的原始课程报告 `lec5.md` 基础上改写而成。原报告标题为“Lecture 5 课程报告：GPU / TPU 硬件模型、性能优化与 FlashAttention”，其主线已经覆盖了 GPU/TPU 硬件模型、roofline model、低精度、fusion、recomputation、coalescing、tiling、矩阵乘法性能异常以及 FlashAttention。

本次对话中可直接访问的课程资料包括：

| 资料                                  | 本次使用方式                                                 |
| ------------------------------------- | ------------------------------------------------------------ |
| 原始报告 `lec5.md`                    | 作为主要改写对象，保留原有课程顺序、核心解释、公式、代码说明和复习点 |
| 上传的 `lecture_05.pdf`               | 用于核对讲义页面、图表、术语和课程结构                       |
| GitHub 仓库 `stanford-cs336/lectures` | 用于核对 Lecture 5 对应文件、通用代码 `gpu_util.py` 与 `facts.py` |

需要说明的是，本次对话中没有单独上传可直接检索的字幕文件。原报告中已经包含了若干“字幕中教师指出 / 教师强调”的内容，本报告会保留这些内容，但不会伪造新的逐字字幕引用。对于可核对的幻灯片内容，使用上传 PDF 的文件引用；对于 GitHub 代码和仓库结构，使用网页引用。

GitHub 仓库当前文件列表中包含 `lecture_05.pdf`，同时 README 说明可执行讲义通常命名为 `lecture_XX.py`，不可执行讲义命名为 `lecture_XX.pdf`；因此，本讲没有单独的 `lecture_05.py` 主程序可逐段解析。

本报告中的“延伸理解”“工程直觉”“学习提示”是为了帮助学习和复习而加入的解释，不代表课程原文逐字内容；涉及课程资料的事实会尽量附上 PDF、原报告或 GitHub 引用。

------

## 1. 学习导读

### 1.1 本讲要解决的核心问题

本讲的核心问题是：

> **大模型训练和推理为什么不能只看“理论计算量”，而必须理解 GPU 的硬件结构、内存层级和数据移动成本？**

课程从一个很实际的困惑开始：为什么同样是矩阵乘法，有些矩阵尺寸会突然变慢？为什么把词表大小从 50,257 padding 到 50,304，虽然增加了无用计算，却可能带来加速？为什么 FlashAttention 没有改变 attention 的数学公式，却能显著降低内存访问和运行时间？原报告把这些问题概括为：GPU 不是“更快的 CPU”，而是一个强依赖并行度、内存层级、数据布局和 kernel 形状的系统。

PDF 第 2 页也明确给出本讲目标：让 CUDA 和 GPU 不再像“魔法”，理解 GPU 什么时候会变慢，以及如何设计快速算法。

### 1.2 本讲在 CS336 课程中的位置

Lecture 5 是 CS336 从语言模型算法转向系统效率的关键一讲。前面几讲更多关注模型、神经网络和训练目标；本讲开始进入 systems portion，讨论 GPU、parallelization、inference 等主题。原报告指出，本讲不是完整 CUDA 编程课，而是建立一个足够准确的 GPU 性能心智模型。

在大模型训练中，算法本身只是问题的一部分。即使模型结构和训练目标固定，能否充分利用 GPU，也会直接影响训练时间、成本、可扩展性和模型规模。换句话说，本讲回答的是：

> **当模型已经确定后，怎样让硬件真正跑满？**

### 1.3 本讲知识结构

PDF 第 4 页把本讲分成三部分：GPU 如何工作、如何理解 GPU 性能，以及如何用这些知识理解 FlashAttention。

| 模块               | 解决的问题                            | 核心概念                                                     | 与后续内容的关系                             |
| ------------------ | ------------------------------------- | ------------------------------------------------------------ | -------------------------------------------- |
| GPU / TPU 硬件模型 | GPU 为什么适合大模型？                | SM、SP、thread、block、warp、memory hierarchy                | 为后续解释 tiling、coalescing、fusion 打基础 |
| GPU 性能模型       | 为什么同一个算子有时快、有时慢？      | roofline model、operational intensity、memory-bound、compute-bound | 解释矩阵乘法吞吐量曲线和优化方向             |
| GPU 优化技巧       | 怎样减少 GPU 的等待和数据搬运？       | low precision、fusion、recomputation、coalescing、tiling     | 构成 FlashAttention 的设计组件               |
| 矩阵乘法性能谜题   | 为什么 padding 可能加速？             | tile size、alignment、wave quantization                      | 把抽象性能模型落实到真实矩阵形状             |
| FlashAttention     | 为什么 exact attention 也能大幅加速？ | IO-aware attention、online softmax、tile-wise recomputation  | 是本讲所有系统思想的综合案例                 |

### 1.4 学习本讲前需要知道什么

学习本讲最好具备以下前置知识：

| 前置知识                  | 为什么需要                                                   |
| ------------------------- | ------------------------------------------------------------ |
| 矩阵乘法                  | 本讲大量性能问题都围绕 dense matmul 展开                     |
| Transformer attention     | FlashAttention 部分需要理解 Q、K、V、softmax 和 attention matrix |
| 基本反向传播              | recomputation / activation checkpointing 与 forward activations 有关 |
| 基本硬件概念              | 需要理解计算单元、cache、显存、带宽、延迟                    |
| PyTorch / CUDA 的基本概念 | 理解 kernel launch、GPU device、tensor placement 更容易      |
| 浮点数精度                | low precision、FP16、FP8、MXFP8、MXFP4 都与数值格式有关      |

学习本讲时最重要的不是记住某个“魔法倍数”，而是建立一种系统直觉：**GPU 性能常常由数据移动、并行粒度和硬件对齐决定，而不仅仅由 FLOPs 决定。**

------

## 2. 第一部分：为什么语言模型需要 GPU，以及 GPU 与 CPU 的根本差异

### 2.1 问题引入：compute 是语言模型 scaling 的燃料

这一节要解决的问题是：**为什么大模型 scaling 必然把我们带到 GPU 和并行计算？**

PDF 第 5 页用 scaling law 的图说明，更多 compute 往往能带来更可预测的语言模型性能提升；更快硬件、更高利用率和更好的并行化本身就能推动模型进步。

但问题在于：compute 从哪里来？

过去几十年，处理器性能提升很大程度上依赖 Dennard scaling：晶体管变小，功耗密度保持可控，时钟频率和单线程性能持续提高。PDF 第 6 页指出，传统 Dennard scaling 从 1980 到 2000 年代后逐渐失效。

这意味着，大模型不能只依赖“单个 CPU 核心越来越快”。新的增长路径是 parallel scaling：让大量计算单元同时工作。PDF 第 7 页强调，GPU parallel scaling 在 10 年内带来了超过 1000 倍的提升，并给出一句核心判断：“There is no LLM scaling without GPU scaling。”

### 2.2 概念定义：CPU 优化 latency，GPU 优化 throughput

直觉上，可以把 CPU 想成“少数很强的工人”，把 GPU 想成“很多轻量工人组成的工厂”。

CPU 更关注单个任务尽快完成，也就是 latency。它拥有更复杂的控制逻辑、更强的分支处理能力、更大的 cache 和更强的单线程执行能力。适合 CPU 的任务通常具有复杂控制流、不规则分支、串行依赖或需要快速响应。

GPU 更关注单位时间处理尽可能多的数据，也就是 throughput。它不追求单个 thread 极快完成，而是让大量 lightweight threads 同时执行相似指令。PDF 第 8 页明确对比：CPU 优化少数快速线程和 latency，GPU 优化大量线程和总吞吐量；GPU 有许多小 ALU，但对 branching、control 和 cache 的支持更少。

### 2.3 方法与原理：为什么 Transformer 适合 GPU？

Transformer 中最重的计算通常是 dense matrix multiplication，例如 attention 中的 $QK^\top$、softmax 后乘 $V$，以及 MLP 中的大矩阵乘法。这些运算有两个特点：

第一，计算结构规则。矩阵乘法中每个输出元素都来自大量乘加操作，非常适合把工作分给许多 threads。

第二，数据并行度高。大量输出元素可以并行计算，只要数据能够及时从内存送到计算单元。

这正好匹配 GPU 的强项：**大规模并行 + 高吞吐矩阵乘法**。

### 2.4 学习提示

本节复习时要抓住一个对比：

| 问题                 | CPU 思路                     | GPU 思路                           |
| -------------------- | ---------------------------- | ---------------------------------- |
| 优化目标             | 让一个或少数任务尽快完成     | 让整体吞吐量最大                   |
| 适合任务             | 分支复杂、串行依赖强         | 大量规则并行计算                   |
| 语言模型中的典型场景 | 数据准备、调度、部分控制逻辑 | 矩阵乘法、attention、MLP、批量推理 |

一句话总结：**GPU 不是更快的 CPU，而是为大规模规则并行计算设计的吞吐机器。**

------

## 3. 第二部分：GPU 硬件模型与编程模型

### 3.1 问题引入：为什么只知道“GPU 很快”还不够？

这一节要解决的问题是：**GPU 内部到底有哪些层级，为什么这些层级决定了代码快慢？**

如果只把 GPU 当作一个黑箱，看到性能曲线中的锯齿、掉点、padding 加速、FlashAttention 加速时，就会觉得它们像魔法。课程的目标是把这种“魔法”拆成几个具体对象：SM、SP、thread、block、warp、shared memory、global memory。

### 3.2 GPU 的执行单元：SM 与 SP

GPU 的基本执行组织可以理解为两层。

第一层是 SM，即 Streaming Multiprocessor。GPU 有许多 SM，每个 SM 可以独立执行一个或多个 block。PDF 第 9 页展示了 GPU execution units：GPU 有许多 SM，每个 SM 进一步包含许多 SP，也就是 streaming processors。

第二层是 SP，它们是真正执行大量 thread 指令的更小计算单元。

直觉上，SM 像一个小工厂，里面有很多工位；整个 GPU 则由许多这样的工厂组成。只要任务足够大、切分足够合理，就可以把很多 block 分派到不同 SM 上并行执行。

### 3.3 GPU 的内存层级：性能优化的核心

这一小节是本讲最重要的硬件直觉之一：**离计算单元越近的内存越快，但容量越小、越昂贵；离计算单元越远的内存容量更大，但访问更慢。**

PDF 第 10 页总结了 GPU memory hierarchy：L1 cache 和 shared memory 位于 SM 内部，L2 cache 在芯片上，global memory 是 GPU 旁边的显存芯片；SRAM 类型的 shared/cache memory 更贵但比 DRAM 类型的 global memory 更快。

常见层级可以这样理解：

| 层级                | 位置              | 特点               | 编程意义                           |
| ------------------- | ----------------- | ------------------ | ---------------------------------- |
| Registers           | 每个 thread 私有  | 最快、容量最小     | 存放 thread 局部变量               |
| Shared memory       | 同一 block 内共享 | 很快、程序员可控制 | tiling 的核心存储                  |
| L1 cache            | SM 内             | 快、自动缓存       | 程序员通常不直接控制               |
| L2 cache            | GPU die 上        | 比 L1 慢但更大     | 多 SM 共享缓存                     |
| Global memory / HBM | GPU 外部显存      | 容量大但慢得多     | 大多数 tensor 存放处，也是瓶颈来源 |

原报告强调的工程原则非常关键：

> 能在 registers / shared memory 中重复使用的数据，就不要反复从 global memory / HBM 读写。

### 3.4 GPU 执行模型：thread、block、warp

PDF 第 11 页明确列出 GPU execution model 的三个关键对象：threads、blocks、warps。Threads 真正“do the work”；blocks 是 threads 的分组，每个 block 在一个 SM 上运行，并拥有自己的 shared memory；warp 是 32 个连续编号 threads 的执行单位。

| 对象   | 直觉解释     | 正式含义                                           | 为什么重要                                      |
| ------ | ------------ | -------------------------------------------------- | ----------------------------------------------- |
| Thread | 一个轻量工人 | 执行具体指令和数据操作的最小并行单元               | 每个 thread 通常处理一小块数据                  |
| Block  | 一个小工队   | 一组 threads，在一个 SM 上运行，共享 shared memory | tiling 和 shared memory reuse 的基本单位        |
| Warp   | 同步执行小组 | 32 个连续编号 threads，执行相同指令                | control divergence 和 coalescing 都与 warp 有关 |

GPU 的执行模型通常称为 SIMT：Single Instruction, Multiple Threads。同一个 warp 内的 threads 执行同一条指令，但作用在不同数据上。它比 CPU 的分支执行更受约束，但换来了极高吞吐量。

### 3.5 GPU 内存模型：为什么跨 block 通信慢？

PDF 第 12 页说明，每个 thread 可以访问自己的 registers，同一 block 内可以访问 shared memory；跨 block 的信息通常需要读写 global memory，因此慢。

这直接解释了为什么很多 GPU 优化围绕 block 展开：如果一个数据 tile 被加载到 shared memory，那么同一 block 内的 threads 可以反复使用它；但如果不同 block 需要交换信息，就往往要绕回 global memory，代价更高。

### 3.6 学习提示

本节最容易混淆的是 shared memory 和 cache。二者都比 global memory 快，但含义不同：

| 概念          | 是否程序员显式控制       | 主要用途                           |
| ------------- | ------------------------ | ---------------------------------- |
| L1 / L2 cache | 通常自动控制             | 自动缓存近期访问数据               |
| Shared memory | 程序员 / kernel 显式控制 | 手动实现 tile reuse、减少 HBM 访问 |

一句话总结：**理解 GPU 性能，必须先理解 thread/block/warp 的执行模型，以及 register/shared/cache/global memory 的层级差异。**

------

## 4. 第三部分：TPU 旁支与 accelerator 的共同结构

### 4.1 问题引入：为什么讲 GPU 时还要提 TPU？

这一节要解决的问题是：**TPU 和 GPU 是完全不同的硬件，还是针对机器学习负载走向相似结构的不同实现？**

课程在 GPU 主线中插入 TPU 讨论，是为了说明一个更一般的规律：如果目标是高效执行机器学习 workloads，硬件往往会趋向某些共同设计。

### 4.2 GPU 与 TPU 的共同点

PDF 第 13 页总结：GPU、TPU 和许多 accelerators 在高层结构上相似，都包含 lightweight control、fast big matmul unit 和 fast memory。不同点包括 accelerator networking，以及 TPU 没有 GPU 中的 warp 概念。

也就是说，它们共享的核心不是品牌，而是：

| 共同结构       | 作用                                           |
| -------------- | ---------------------------------------------- |
| 轻量控制单元   | 不把大量晶体管浪费在复杂分支控制上             |
| 大矩阵乘法单元 | 加速 dense matmul，这是现代 ML 的核心计算      |
| 快速本地内存   | 尽量减少慢速外部 HBM 访问                      |
| 高带宽外部内存 | 存放大模型权重、activation、optimizer state 等 |

### 4.3 容易混淆点：GPU Tensor Core vs TPU Tensor Core

原报告特别提醒：tensor core 在 GPU 和 TPU 语境中含义不同。GPU 中的 Tensor Core 通常指专门的矩阵乘法硬件单元；TPU 中的 Tensor Core 更接近一个 processor / core。

这类术语差异在阅读硬件文档时非常重要。不要只看名字，要看它在具体架构中承担什么功能。

### 4.4 学习提示

本节复习重点是：

> **现代 ML accelerator 的共同目标，是用尽可能高的能效执行大规模矩阵乘法，并尽量减少数据移动。**

TPU 与 GPU 的区别会在并行化和系统互联中继续变得重要；本讲只需要掌握它们在“矩阵乘法 + 本地内存 + 外部 HBM”这一层面的共同结构。

------

## 5. 第四部分：为什么矩阵乘法是现代机器学习的“特权操作”

### 5.1 问题引入：为什么课程反复强调 matmul？

这一节要解决的问题是：**为什么现代模型架构中 dense matmul 如此重要？**

答案不只是“神经网络数学上需要矩阵乘法”，还包括一个硬件事实：现代 GPU 对矩阵乘法有专门硬件支持。

### 5.2 Tensor Cores：矩阵乘法硬件化

早期 GPU 并不是专门为神经网络训练设计的。PDF 第 16 页提到，早期研究者曾利用 programmable shaders 来实现快速矩阵乘法。

后来，NVIDIA GPU 引入 Tensor Cores。PDF 第 17 页写明，Tensor Cores 是 specialized matrix multiplication circuits，matmuls 比其他 floating point ops 快超过 10 倍。

这带来一个架构层面的结论：

> 如果一个模型的主要计算可以表达成大规模 dense matmul，它就更容易吃到现代 GPU 的硬件红利。

### 5.3 Compute scaling 快于 memory scaling

PDF 第 18 页展示了另一个重要趋势：FLOPs 增长快于 memory bandwidth 和 interconnect bandwidth；也就是说，计算单元越来越强，但把数据送到计算单元的能力没有同比例增长。

这就是所谓 memory wall 在大模型训练中的表现。GPU 可能有极高的峰值 FLOPs，但如果数据搬运跟不上，计算单元就会空等。

### 5.4 工程直觉

现代 GPU 优化常常不是“少做计算”，而是“少搬数据”。在某些情况下，多做一点计算反而更快，例如：

| 方法                    | 表面看起来             | 实际系统收益                                        |
| ----------------------- | ---------------------- | --------------------------------------------------- |
| Recomputation           | 重复计算，似乎浪费     | 减少 activation 保存和读取                          |
| FlashAttention backward | backward 中重新算 tile | 避免保存完整 $N^2$ attention matrix                 |
| Padding                 | 增加无用维度           | 改善 tile / warp / memory alignment，提高硬件利用率 |

### 5.5 学习提示

本节最重要的结论是：

> **Tensor Cores 让 dense matmul 成为 GPU 上最值得优化、最容易扩展的核心操作；但 compute 增长快于 memory，意味着真正的瓶颈经常是 data movement。**

------

## 6. 第五部分：GPU 性能框架——Roofline Model

### 6.1 问题引入：怎样判断一个 workload 为什么慢？

这一节要解决的问题是：**一个 GPU kernel 慢，到底是因为算力不够，还是因为内存带宽不够？**

Roofline model 给了一个简洁的答案。它用 workload 的 operational intensity 来判断性能上限来自 compute 还是 memory。

### 6.2 公式：Roofline 上界

Roofline model 可以写为：
$$
\text{Throughput} \le \min(P_{\text{peak}},\; B_{\text{mem}} \cdot I)
$$
其中：

| 符号              | 含义                                                      |
| ----------------- | --------------------------------------------------------- |
| $P_{\text{peak}}$ | 硬件峰值计算吞吐量，例如 peak FLOPs/s                     |
| $B_{\text{mem}}$  | 内存带宽，例如 bytes/s                                    |
| $I$               | operational intensity，即每搬运 1 byte 数据能做多少 FLOPs |

通常定义：
$$
I = \frac{\text{FLOPs}}{\text{Bytes moved}}
$$
如果 $I$ 很小，说明每搬一次数据只能做少量计算，性能受 memory bandwidth 限制；如果 $I$ 足够大，说明每次搬来的数据能被反复计算利用，性能可能进入 compute-bound 区域。原报告用这一模型解释了矩阵乘法吞吐量曲线的总体上升和平坦区域。

### 6.3 直觉解释：斜线区和平坦区

Roofline 图通常有两段：

| 区域                 | 性能瓶颈                 | 直觉                             |
| -------------------- | ------------------------ | -------------------------------- |
| Memory-bound 斜线区  | $B_{\text{mem}} \cdot I$ | 数据搬得慢，计算单元吃不饱       |
| Compute-bound 平坦区 | $P_{\text{peak}}$        | 数据复用足够好，硬件算力成为上限 |

如果一个 kernel 在 memory-bound 区域，单纯增加 FLOPs 峰值没有用；必须减少 bytes moved，或提高每 byte 数据的复用次数。

### 6.4 例子：为什么大矩阵乘法通常更快？

矩阵越大，单个元素读入后可参与的乘加越多，数据复用更充分，因此 operational intensity 通常更高。这样 workload 更容易从 memory-bound 走向 compute-bound。

但课程也强调，矩阵乘法吞吐量曲线并不平滑。PDF 第 20 页展示 square matmul 的性能曲线，出现了大量锯齿、周期性波动和掉点。

Roofline model 能解释“大矩阵总体更快”，但不能解释所有锯齿。要解释这些离散掉点，还需要 tiling、alignment、coalescing 和 wave quantization。

### 6.5 学习提示

复习 roofline 时，不要只背公式。要会回答：

| 问题                                 | 回答重点                                                     |
| ------------------------------------ | ------------------------------------------------------------ |
| 什么是 operational intensity？       | 每搬运 1 byte 数据能做多少 FLOPs                             |
| memory-bound 怎么优化？              | 减少 HBM 访问、提高数据复用、fusion、tiling、低精度          |
| compute-bound 怎么优化？             | 使用 Tensor Cores、提高并行度、选择合适数据类型              |
| 为什么 roofline 不解释所有性能锯齿？ | 它是上层模型，不包含 tile size、SM 数量、alignment 等离散因素 |

一句话总结：**Roofline model 告诉我们，GPU 优化的第一步是判断瓶颈在 compute 还是 memory。**

------

## 7. 第六部分：让 GPU 变快的六个技巧

PDF 第 22 页列出本讲 Part 2 的六个技巧：control divergence、low precision computation、operator fusion、recomputation、coalescing memory 和 tiling。

这一节要解决的问题是：**当我们知道 GPU 可能被 memory 或执行模型限制后，具体有哪些办法让 workload 变快？**

------

### 7.1 Trick 0：Control divergence

#### 这是什么？

Control divergence 指同一个 warp 内的 threads 因为条件分支而走不同路径，导致 GPU 无法让所有 threads 同时有效工作。

PDF 第 23 页说明，GPU 采用 SIMT 模型，同一个 warp 中每个 thread 执行相同指令；conditionals 可以写，但会带来执行模型上的显著开销。

#### 为什么会慢？

CPU 遇到 `if/else` 时，可以让一个核心沿着某个分支执行。GPU 的 warp 中有 32 个 threads，如果一部分 thread 走 `if`，另一部分走 `else`，硬件通常需要串行执行两个分支，并对暂时不属于当前分支的 threads 做 mask。结果就是：一部分 thread 在等。

#### 与 ReLU 的关系

ReLU 可以写成：
$$
\operatorname{ReLU}(x)=\max(0,x)
$$
如果在 GPU kernel 中用大量分支表达它，可能引入 divergence。实际中常用向量化、mask 或专门 kernel 实现，让 warp 内 threads 尽量执行相同指令。

#### 学习提示

Control divergence 不是 memory bottleneck，而是 GPU 执行模型带来的瓶颈。它提醒我们：**GPU 喜欢规则、统一、少分支的工作。**

------

### 7.2 Trick 1：Low precision computation

#### 这是什么？

Low precision 的基本想法是：用更少 bits 表示数值，从而减少数据搬运，并让硬件使用更快的低精度矩阵乘法路径。

PDF 第 24 页把它概括为：如果 bits 更少，需要移动的 bits 也更少。

#### ReLU 例子：FP32 vs FP16

课程用 elementwise ReLU 说明低精度如何降低 memory traffic。对于长度为 $n$ 的向量，计算：
$$
x = \max(0,x)
$$
FP32 情况下，读一个 $x$ 需要 4 bytes，写回结果需要 4 bytes，总共约 8 bytes / FLOP。FP16 情况下，读写各 2 bytes，总共约 4 bytes / FLOP。PDF 第 25 页展示了这个对比。

这里要注意一个易混点：幻灯片用的是 bytes/FLOP，它表示“每做一次计算要搬多少数据”；而 roofline 中常用的 operational intensity 是 FLOPs/byte。二者互为倒数。bytes/FLOP 降低，就意味着 FLOPs/byte 提高。

#### Mixed precision：不是所有地方都低精度

PDF 第 26 页说明，现代 GPU 的许多操作通过 low / mixed precision operations 加速，Tensor Cores 可以用低精度输入和更高精度 accumulator 做矩阵乘法。

常见做法不是“所有东西都变成 FP16/FP8”，而是：

| 计算位置             | 常见精度选择      | 原因                      |
| -------------------- | ----------------- | ------------------------- |
| Matmul 输入          | FP16 / BF16 / FP8 | Tensor Cores 支持，吞吐高 |
| Accumulator          | FP32 或更高精度   | 避免累加误差过大          |
| Softmax / exp        | 通常需要更谨慎    | 容易 overflow / underflow |
| Loss / normalization | 通常保持较高精度  | 数值稳定性关键            |

#### FP8、MXFP8 与 MXFP4

原报告保留了课程对前沿低精度格式的讨论。FP8 并不是一个唯一格式，常见有：

| 格式 | 特点                                           |
| ---- | ---------------------------------------------- |
| E4M3 | exponent 4 bits，mantissa 3 bits，精度更细     |
| E5M2 | exponent 5 bits，mantissa 2 bits，动态范围更大 |

MXFP8 的关键是 multiple scaling factors。PDF 第 27 页写明，MXFP8 使用 E4M3，并且 scale factors 自身是 FP8 E8M0，1 per 32；同时转置不再简单。

PDF 第 29 页展示 MXFP4 可表示的所有值，并说明它使用 1 per 16 scaling 和 E4M3 scaling factors。

#### 学习提示

低精度的核心不是“越低越好”，而是：

> **在不破坏训练稳定性的前提下，用更少 bits 减少 memory traffic，并启用更快的硬件路径。**

------

### 7.3 Trick 2：Operator fusion

#### 这是什么？

Operator fusion 指把多个小算子合并到一个 kernel 中执行，避免中间结果反复写回 global memory 再读出。

课程用“GPU 像工厂”的比喻解释：global memory 像仓库，compute units 像工厂。每个小算子都单独执行，就像每加工一步都把半成品运回仓库，下一步再搬出来，非常浪费。

#### 课堂例子：$\sin^2(x)+\cos^2(x)$

计算：
$$
\sin^2(x)+\cos^2(x)
$$
naive PyTorch computation graph 可能包含 `sin`、`cos`、两个 square 和一个 add。PDF 第 32 页说明，naive 计算会 launch 5 个 CUDA kernels。PDF 第 33 页说明，这 5 个 pointwise operations 可以 fused into a single CUDA kernel call，并且这类简单 fusion 可以由 `torch.compile` 等 compiler 自动完成。 

融合前后的差别：

| 实现方式 | 数据流                                                       |
| -------- | ------------------------------------------------------------ |
| 非融合   | 每个小算子读 global memory，写中间结果，再被下一个算子读     |
| 融合     | 读一次 $x$，在 SM/register/shared memory 附近完成多步计算，最后写一次结果 |

#### 学习提示

Fusion 解决的问题不是减少数学计算量，而是减少 kernel launch 和 HBM traffic。它适合 pointwise、elementwise 或能局部合并的算子；复杂 fusion 可能需要手写 kernel。

------

### 7.4 Trick 3：Recomputation

#### 这是什么？

Recomputation 指 forward 时不保存所有中间 activation，而是在 backward 需要时重新计算一部分中间值。

这听起来像浪费，因为它重复做了计算。但在现代 GPU 上，计算常常比 HBM 读写便宜。因此，用额外 compute 换更少 memory traffic 可能更快。

#### 课堂例子：三个 sigmoid

课程用三个 sigmoid 叠加举例：
$$
x \rightarrow \sigma(x) \rightarrow \sigma(\sigma(x)) \rightarrow \sigma(\sigma(\sigma(x))) = out
$$
普通 backprop 会保存中间 activations，backward 再读取它们。原报告记录，naive forward/backward 总共涉及 8 次 memory reads/writes，arithmetic intensity 很低；recomputation 则丢弃部分 activation，在 backward 中重新 forward，memory accesses 从 8 降到 5，即 5/8 的 memory access。

#### 与 activation checkpointing 的关系

Activation checkpointing 可以看作 recomputation 的一种通用形式：只保存少量 checkpoint，中间层在 backward 时重算。它特别适合深层网络，因为 activation memory 往往随层数和序列长度增长很快。

#### 学习提示

Recomputation 的核心权衡是：

| 付出             | 收益                       |
| ---------------- | -------------------------- |
| 多做一些 compute | 少保存 / 少读取 activation |
| backward 更复杂  | 降低显存压力和 HBM traffic |

一句话总结：**当 compute 便宜、memory 昂贵时，重复计算可能是系统上的最优选择。**

------

### 7.5 Trick 4：Memory coalescing

#### 这是什么？

Memory coalescing 指同一个 warp 内 threads 访问连续或相邻的 memory addresses，使硬件可以用一次或少数几次 DRAM transaction 满足多个 thread 的访问请求。

PDF 第 37 页解释 DRAM global memory 以 burst mode 读取：每次访问往往返回一段连续 bytes。PDF 第 38 页进一步说明，如果同一个 warp 中 threads 的访问都落在同一个 burst section，就称为 coalesced。

#### 为什么重要？

如果 32 个 threads 分别访问相隔很远的地址，硬件可能需要很多次 memory transaction；如果它们访问连续地址，可能只需要一次或少数几次。这会极大影响 memory bandwidth 利用率。

#### 矩阵乘法中的 coalescing

在矩阵乘法中，coalescing 与矩阵 layout、row-major / column-major、thread id 到地址的映射都有关。原报告提醒，不要简单记成“永远按行读”或“永远按列读”。正确理解是：

> **同一个 warp 中的 threads 应尽量访问连续内存地址，并让访问边界与 burst / tile 边界对齐。**

#### 学习提示

Coalescing 的复习重点是 warp：memory access 是以 warp 为重要粒度发生的。你要问的是：

> 同一个 warp 的 32 个 threads，是在一起读一段连续内存，还是在到处乱读？

------

### 7.6 Trick 5：Tiling

#### 这是什么？

Tiling 是本讲最重要的技巧。它把矩阵切成较小 tiles，把 tile 加载到 shared memory，然后在 shared memory 中重复使用，从而减少 global memory 访问。

PDF 第 41 页明确说：cut up the matrix into smaller tiles, and load this into shared memory；优势是 repeated reads now access shared, not global memory，并且 memory access can be coalesced。

#### 矩阵乘法公式

矩阵乘法：
$$
C = AB
$$
其中：
$$
C_{ij} = \sum_{k=1}^{N} A_{ik}B_{kj}
$$
对于每个 $C_{ij}$，都要读取 $A$ 的一行和 $B$ 的一列。如果不复用数据，$A$ 和 $B$ 中的同一元素可能被多次从 global memory 读取。

#### Tiling 如何减少 global memory access？

课程给出的核心结论是：

非 tiled matmul 中，每个输入元素大约从 global memory 读取 $N$ 次：
$$
\text{global reads}_{\text{non-tiled}} \sim N
$$
使用 tile size 为 $T$ 的 tiled matmul 后，每个输入元素从 global memory 读取次数降为：
$$
\text{global reads}_{\text{tiled}} \sim \frac{N}{T}
$$
因此 global memory access 约减少 $T$ 倍：
$$
\frac{N}{N/T}=T
$$
PDF 第 42 页也用 “factor of $T$ reduction in global memory access” 概括这一点。

#### 学习提示

Tiling 不是改变矩阵乘法数学定义，而是改变执行顺序和数据所在位置：

| 问题             | Tiling 的回答                                             |
| ---------------- | --------------------------------------------------------- |
| 数据从哪里来？   | 从 global memory 加载 tile                                |
| 数据在哪里复用？ | shared memory                                             |
| 为什么更快？     | 减少 HBM 访问，提高数据复用                               |
| 为什么难？       | tile size、shared memory 大小、对齐和矩阵维度都会影响性能 |

一句话总结：**Tiling 的本质是把“反复读 HBM”变成“读一次 HBM，在 shared memory 中反复用”。**

------

## 8. 第七部分：矩阵乘法性能谜题——为什么 padding 会变快？

### 8.1 问题引入：为什么更大的矩阵反而更快？

这一节要解决的问题是：**为什么矩阵乘法吞吐量不是矩阵越小越快、越大越慢，而会出现奇怪的锯齿和掉点？**

课程用 square matmul 吞吐量图展示了复杂现象：总体上矩阵变大，吞吐量增加；但曲线出现大量周期性波动、掉点和分层。原报告指出，roofline model 只能解释总体趋势，不能解释所有锯齿；锯齿来自 tiling、alignment、coalescing 和 wave quantization。

### 8.2 Tile size 不整除矩阵尺寸

PDF 第 43 页说明，tile sizes may not divide the matrix size，从而导致低利用率。影响 tile size 的因素包括 coalesced memory access、shared memory size 和 matrix dimension divisibility。

直觉例子：

| 矩阵尺寸         | tile size        | 结果                                |
| ---------------- | ---------------- | ----------------------------------- |
| $256 \times 256$ | $128 \times 128$ | 正好切成 4 个完整 tiles             |
| $257 \times 257$ | $128 \times 128$ | 需要额外边缘 tile，很多位置空着不用 |

虽然 $257$ 只比 $256$ 多 1，但它可能让 kernel 多处理一批利用率很低的边缘 tile。

### 8.3 Memory alignment 与 padding

PDF 第 44 页说明，memory comes in bursts，loading tiles is fast if bursts align with the matrix；若矩阵维度导致布局不对齐，coalesced accesses 可能很难实现，需要 padding。

原报告用 Andrej Karpathy 的 nanoGPT 例子说明：把 vocab size 从 50,257 增加到 50,304，虽然增加了无用维度，却可能带来约 25% speedup。原因不是“更多参数更好”，而是 padding 改善了 kernel 形状、对齐和 occupancy。

### 8.4 Wave quantization

PDF 第 48 页展示了一个具体掉点：矩阵尺寸从 1792 增加到 1793 时，吞吐量突然下降。原因是 tile size 为 $256 \times 128$。

当 $N=1792$：
$$
\frac{1792}{256} \times \frac{1792}{128}
=
7 \times 14
=
98
$$
需要 98 个 tiles。

当 $N=1793$：
$$
\left\lceil \frac{1793}{256} \right\rceil
\times
\left\lceil \frac{1793}{128} \right\rceil
=
8 \times 15
=
120
$$
需要 120 个 tiles。

如果 A100 有 108 个 SM，那么 98 个 tiles 可以一波执行完；120 个 tiles 则需要第二波，而第二波只有 12 个 tiles，导致许多 SM 空闲。这就是 wave quantization。

### 8.5 工程直觉

Padding 加速不是“玄学”，而是硬件离散性造成的结果：

| 因素                         | 为什么会造成性能跳变                             |
| ---------------------------- | ------------------------------------------------ |
| Warp size                    | 线程以 32 为关键调度粒度                         |
| Tile size                    | 矩阵尺寸不能整除 tile 时会产生低利用率边缘块     |
| Burst alignment              | 内存访问不对齐会增加 transaction                 |
| SM 数量                      | tiles 数量不能很好填满 SM waves 时会有 tail wave |
| Tensor Core preferred shapes | 某些维度更容易启用高效硬件路径                   |

### 8.6 学习提示

不要死记 “pad 到 64 就快”。正确复习方式是记住：

> **padding 有时加速，是因为它让矩阵形状更适合硬件的 tile、warp、burst、SM wave 和 Tensor Core 对齐要求。**

------

## 9. 第八部分：FlashAttention——把本讲所有技巧组合起来

### 9.1 问题引入：FlashAttention 到底快在哪里？

这一节要解决的问题是：**FlashAttention 为什么没有改变 attention 数学公式，却能显著提升速度和降低显存访问？**

PDF 第 50 页指出，FlashAttention dramatically accelerates attention，并展示了 attention on GPT-2 中 PyTorch 和 FlashAttention 的 latency 对比，以及 HBM accesses 与 block size 的关系。

原报告强调，FlashAttention 的主要收益来自减少 HBM/global memory traffic，而不是把 attention 变成近似算法。

### 9.2 标准 scaled dot-product attention

标准 attention 写作：
$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V
$$
其中：

| 符号       | 含义                                     |
| ---------- | ---------------------------------------- |
| $Q$        | queries                                  |
| $K$        | keys                                     |
| $V$        | values                                   |
| $d_k$      | key/query 的 hidden dimension            |
| $QK^\top$  | attention scores                         |
| softmax    | 对每个 query 对所有 key 的相关性做归一化 |
| 最后乘 $V$ | 得到 value 的加权和                      |

PDF 第 51 页总结 attention computation：涉及 Q、K、V 的 3 个矩阵乘法，中间夹着 softmax。

### 9.3 Naive attention 的系统问题

Naive attention 通常会显式构造 $N \times N$ attention score matrix，再构造 softmax matrix。序列长度 $N$ 变大时，这带来两个问题：

| 问题     | 代价                                     |
| -------- | ---------------------------------------- |
| 计算量   | attention scores 是 $O(N^2)$             |
| 内存访问 | 中间 $N \times N$ 矩阵需要写入和读出 HBM |

对于长序列，真正昂贵的不只是算 $QK^\top$，而是把巨大中间矩阵 materialize 到 HBM，再读回来继续算。

### 9.4 FlashAttention 的核心思想

PDF 第 52 页说明，FlashAttention Figure 1 本质上就是对 KQV matrix multiply 做 tiling，但问题是 softmax 怎么办。

FlashAttention forward 的核心流程可以概括为：

1. 把 $Q,K,V$ 分块成 tiles；

2. 把当前需要的 tile 加载到 SRAM / shared memory；

3. 在 tile 内计算局部 attention score：
   $$
   S = QK^\top
   $$

4. 不把完整 $S$ 写回 HBM；

5. 在 tile 内融合 exponential、softmax accumulation 和乘 $V$；

6. 用 online softmax 维护跨 tile 的全局归一化；

7. 只把最终 attention output 写回 HBM。

PDF 第 54 页概括了 FlashAttention forward 的三点：tile-wise computation of inner products $S$、fusion of exponential operator、tile-wise computation of softmax via online telescoping sum trick；并说明 backward pass 虽然本讲不展开，但会 tile-by-tile recompute。

### 9.5 为什么 softmax 难以直接 tile？

普通稳定 softmax 写作：
$$
\operatorname{softmax}(x_i)
=
\frac{e^{x_i-m}}{\sum_j e^{x_j-m}},
\quad
m=\max_j x_j
$$
这里的 $m$ 是整行最大值，denominator 也是整行所有元素的和。如果只看一个 tile，就不知道整行最大值，也不知道最终 denominator。因此，不能对每个 tile 独立 softmax 后再简单拼起来。

这就是 FlashAttention 的数学难点：**matmul 容易分块，softmax 的归一化是全局的。**

### 9.6 Online softmax：tile-by-tile 计算 exact softmax

Online softmax 的目标是：一边扫描 tile，一边维护到目前为止的最大值和归一化分母，使最终结果等价于一次性对整行做 softmax。

设第 $j$ 个 tile 中的 scores 为 $s^{(j)}$。维护两个量：

| 符号     | 含义                                  |
| -------- | ------------------------------------- |
| $m_j$    | 到第 $j$ 个 tile 为止见过的最大 score |
| $\ell_j$ | 以 $m_j$ 为基准重标定后的 denominator |

初始化：
$$
m_0=-\infty,\quad \ell_0=0
$$
处理第 $j$ 个 tile：
$$
m_j=
\max\left(
m_{j-1},
\max_{x\in s^{(j)}}x
\right)
$$
分母更新的核心直觉是：如果新 tile 出现了更大的最大值 $m_j$，之前累积的指数项都是以旧基准 $m_{j-1}$ 算的，必须乘上缩放因子 $e^{m_{j-1}-m_j}$，才能换到新的数值尺度下。原报告用 telescoping sum 解释了这一点。

如果同时维护未归一化的 value accumulator $u_j$，可写为：
$$
u_j =
e^{m_{j-1}-m_j}u_{j-1}
+
\sum_{x_t\in s^{(j)}} e^{x_t-m_j}v_t
$$
最后输出：
$$
o_j = \frac{u_j}{\ell_j}
$$
这就是 FlashAttention 可以 tile-by-tile 计算 exact attention 的关键：它没有近似 softmax，而是用在线更新避免 materialize 完整 $N\times N$ attention matrix。

### 9.7 Backward pass：为什么 recomputation 合理？

本讲没有详细展开 FlashAttention backward，但 PDF 第 54 页明确提到 backward 会 tile-by-tile recompute。

这与前面 recomputation 思想一致：forward 不保存完整 attention matrix，backward 需要时重新计算对应 tile。这样多做了一些 compute，但避免了保存和读取 $N^2$ 级别中间 activation。

### 9.8 FlashAttention 与本讲技巧对应表

| 本讲技巧          | FlashAttention 中的体现                               | 学习重点                       |
| ----------------- | ----------------------------------------------------- | ------------------------------ |
| Tiling            | $Q,K,V$ 和 attention scores 分块计算                  | 避免完整 $N^2$ matrix 进入 HBM |
| Fusion            | $QK^\top$、exp、softmax accumulation、乘 $V$ 尽量融合 | 减少中间结果写回               |
| Online softmax    | 维护 running max 和 denominator                       | 解决 softmax 全局归一化问题    |
| Recomputation     | backward 中 tile-wise recompute                       | 用 compute 换 memory           |
| Memory hierarchy  | 尽量使用 SRAM/shared memory                           | 降低 HBM accesses              |
| Roofline thinking | 减少 bytes moved，提高有效 operational intensity      | 从 memory-bound 向更高吞吐移动 |

### 9.9 学习提示

FlashAttention 最容易被误解为“近似 attention”。本讲中的正确理解是：

> **FlashAttention 是 exact attention 的 IO-aware 实现；它改变的是执行方式和数据移动路径，不是 attention 数学定义。**

------

## 10. Python 代码示例详解

### 10.1 文件整体结构

本讲在 GitHub 仓库中没有专属的 `lecture_05.py` 或 notebook。当前仓库文件列表中 Lecture 5 对应 `lecture_05.pdf`；README 说明 executable lectures 命名为 `lecture_XX.py`，non-executable lectures 命名为 `lecture_XX.pdf`。

因此，本节不编造 Lecture 5 的 Python 主流程。原报告中提到两个与本讲主题相关的通用辅助文件：

| 文件          | 与本讲关系                                                 |
| ------------- | ---------------------------------------------------------- |
| `gpu_util.py` | GPU device 选择和 CUDA peak memory 测量                    |
| `facts.py`    | H100 FLOP/s 与 memory bandwidth 常量，可用于 roofline 估算 |

### 10.2 `gpu_util.py`：GPU 设备选择与显存测量

GitHub 中的 `gpu_util.py` 导入 `torch`，定义了两个函数：`cuda_if_available` 和 `get_max_memory_usage`。

#### `cuda_if_available(index: int = 0) -> torch.device`

代码逻辑是：

```
if torch.cuda.is_available():
    return torch.device(f"cuda:{index}")
else:
    return torch.device("cpu")
```

该函数解决的是设备选择问题：如果有 CUDA GPU，就把 tensor / model 放到指定 GPU；否则回退到 CPU。

它对应本讲的 host/device memory 区分。GPU 的 global memory 与 CPU memory 不是同一个空间，训练代码必须明确数据在哪个 device 上。

#### `get_max_memory_usage(func)`

代码逻辑是：

```
if not torch.cuda.is_available():
    return 0

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
func()
return torch.cuda.max_memory_allocated()
```

这个函数用于测量某个函数执行期间的 CUDA peak memory allocation。

它对应本讲的核心系统问题：如果优化目标是减少 HBM/global memory traffic 或显存占用，就需要可测量的 memory usage。FlashAttention、activation checkpointing、low precision 都可以通过类似工具观察显存变化。

### 10.3 `facts.py`：roofline 所需硬件常量

GitHub 中的 `facts.py` 定义了两个 H100 相关常量：

```
h100_flop_per_sec = 1979e12 / 2  # Half without sparsity
h100_bytes_per_sec = 3.35e12
```

这些常量可以分别对应 roofline model 中的 $P_{\text{peak}}$ 和 $B_{\text{mem}}$。

于是可以估算机器 balance：
$$
\frac{P_{\text{peak}}}{B_{\text{mem}}}
=
\frac{1979e12/2}{3.35e12}
\approx 295.37
\text{ FLOPs/byte}
$$
直觉含义是：在这个近似设定下，如果一个 workload 的 operational intensity 明显低于约 295 FLOPs/byte，就很可能难以喂饱 H100 的峰值计算能力，容易 memory-bound。

### 10.4 代码与课程知识点对应表

| 课程知识点                 | 对应代码函数 / 变量                      | 代码功能                         | 理论原理                     | 在课程中的作用                                       |
| -------------------------- | ---------------------------------------- | -------------------------------- | ---------------------------- | ---------------------------------------------------- |
| GPU device 与 CPU fallback | `cuda_if_available(index=0)`             | 自动选择 CUDA 或 CPU             | host/device memory 区分      | 帮助理解 tensor 必须放到正确设备                     |
| 显存占用测量               | `get_max_memory_usage(func)`             | 测量 peak CUDA memory allocation | HBM/global memory 是关键瓶颈 | 可用于观察 recomputation / FlashAttention 的显存收益 |
| Roofline peak compute      | `h100_flop_per_sec`                      | 给出 H100 近似 peak FLOPs        | compute-bound 上限           | 对应 roofline 平坦区域                               |
| Roofline memory bandwidth  | `h100_bytes_per_sec`                     | 给出 H100 memory bandwidth       | memory-bound 上限            | 对应 roofline 斜线区域                               |
| 机器 balance               | `h100_flop_per_sec / h100_bytes_per_sec` | 估算需要多高 FLOPs/byte 才能跑满 | compute-memory gap           | 解释为什么减少 data movement 如此重要                |

### 10.5 学习提示

本讲没有 lecture-specific Python 主文件，因此不要把 `gpu_util.py` 和 `facts.py` 误认为 Lecture 5 的完整实验代码。它们的价值在于提供两个系统学习工具：

1. 如何选择 GPU device；
2. 如何测量显存和估算 roofline。

------

## 11. PDF / 幻灯片 / 图表解析

### 11.1 主要图表总览

| 图表 / 页面     | 展示内容                                 | 支持的课程结论                                   | 对应知识点         |
| --------------- | ---------------------------------------- | ------------------------------------------------ | ------------------ |
| PDF 第 2 页     | matmul 性能曲线与 FlashAttention 图      | 本讲目标是理解 GPU 何时慢、如何写快算法          | 课程导入           |
| PDF 第 4 页     | 三段课程结构                             | 本讲按 GPU、性能、FlashAttention 展开            | 课程主线           |
| PDF 第 8 页     | CPU vs GPU                               | CPU 优化 latency，GPU 优化 throughput            | 硬件模型           |
| PDF 第 9-12 页  | SM、memory hierarchy、thread/block/warp  | GPU 性能由执行模型和内存层级决定                 | GPU anatomy        |
| PDF 第 17 页    | matmul vs non-matmul FLOPs               | Tensor Cores 让 matmul 成为特权操作              | 矩阵乘法           |
| PDF 第 18 页    | compute scaling vs memory scaling        | FLOPs 增长快于 memory bandwidth                  | memory wall        |
| PDF 第 21 页    | roofline model                           | 判断 memory-bound / compute-bound                | 性能模型           |
| PDF 第 25-29 页 | FP32/FP16、Tensor Cores、FP8/MXFP8/MXFP4 | low precision 减少数据搬运并启用硬件加速         | 低精度             |
| PDF 第 30-33 页 | factory metaphor 与 fusion               | fusion 减少中间结果写回                          | operator fusion    |
| PDF 第 34-36 页 | sigmoid recomputation                    | 多做计算可减少 memory access                     | recomputation      |
| PDF 第 37-39 页 | burst mode 与 coalescing                 | warp 访问连续地址更高效                          | memory coalescing  |
| PDF 第 41-44 页 | tiled matmul 与 alignment                | tile reuse 降低 global memory access             | tiling             |
| PDF 第 48 页    | wave quantization                        | tile 数与 SM 数不匹配会导致掉点                  | 矩阵性能谜题       |
| PDF 第 50-54 页 | FlashAttention                           | tiling + online softmax + fusion + recomputation | IO-aware attention |
| PDF 第 55 页    | 全讲总结                                 | 当前 GPU compute 鼓励思考 matmul + data movement | 复习总结           |

PDF 第 55 页明确总结：硬件推动 scaling，低层细节决定什么能 scale；当前 GPU-based compute 强烈鼓励围绕 matmul 和 data movement 思考；仔细考虑 coalescing、tiling、fusion 会带来好性能。

### 11.2 关键图表详细解释

#### 图 1：PDF 第 2 页——本讲目标图

这页同时展示了两个问题：左边是矩阵乘法吞吐量曲线，右边是 FlashAttention 加速图。它不是普通目录页，而是整讲的问题陈述：为什么 GPU 会慢？怎样设计快速算法？

读图时要注意：矩阵乘法曲线的锯齿不是测量噪声，而是后面 tiling、alignment 和 wave quantization 要解释的核心现象。

#### 图 2：PDF 第 9-12 页——GPU anatomy

这些页依次展示 execution units、memory hierarchy、execution model 和 memory model。第 11 页尤其重要，因为它把 thread、block、warp 三个概念放在一起：thread 做工作，block 在 SM 上执行并共享 shared memory，warp 是 32 个连续 threads 的执行单位。

读图时要抓住两条线索：

1. 计算如何被切分到 SM / block / warp；
2. 数据如何在 register / shared memory / global memory 之间移动。

#### 图 3：PDF 第 25-29 页——低精度图

第 25 页用 ReLU 展示 FP32 到 FP16 如何减少 bytes/FLOP；第 26 页展示 mixed precision Tensor Core 计算；第 27-29 页展示 FP8、MXFP8、MXFP4。

读图时最容易误解的是：低精度不是所有计算都低精度。实际训练会根据数值稳定性选择不同精度路径。

#### 图 4：PDF 第 41-42 页——tiling 图

第 41 页展示把矩阵切成 tiles，加载到 shared memory，并分 phases 计算 partial sums；第 42 页给出关键结论：tiled matmul 把每个输入元素从 global memory 读取 $N$ 次降到 $N/T$ 次。

读图时要注意：tiling 的收益不是减少总 FLOPs，而是减少 global memory reads。

#### 图 5：PDF 第 48 页——wave quantization 图

这页解释了 1792 到 1793 的性能掉点。关键不是矩阵只大了 1，而是 tile 数从 98 增加到 120，超过 A100 的 108 个 SM 后需要第二波执行，尾波 SM 利用率很低。

#### 图 6：PDF 第 50-54 页——FlashAttention 图

这些页从 attention recap 出发，先展示 KQV matmul 可以 tiling，再提出 softmax 难题，最后用 online softmax 和 fused forward pass 解决。第 54 页总结 FlashAttention forward 包含 tile-wise inner products、fusion of exponential operator 和 online telescoping softmax。

读图时要把 FlashAttention 看成“本讲技巧合集”，而不是单个算法技巧。

------

## 12. 关键公式与推导汇总

| 公式                                                         | 符号解释                                                     | 适用场景                                       | 直觉解释                               | 代码 / 例子对应                       |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ---------------------------------------------- | -------------------------------------- | ------------------------------------- |
| $\text{Throughput} \le \min(P_{\text{peak}},B_{\text{mem}}I)$ | $P_{\text{peak}}$ 为峰值算力，$B_{\text{mem}}$ 为内存带宽，$I$ 为 operational intensity | 判断 kernel 是 compute-bound 还是 memory-bound | 性能受计算上限和内存上限中的较小者限制 | `facts.py` 中 H100 FLOPs 与 bandwidth |
| $I=\frac{\text{FLOPs}}{\text{Bytes moved}}$                  | FLOPs 为计算量，Bytes moved 为数据移动量                     | roofline 分析                                  | 每搬 1 byte 能做多少计算               | matmul 比 ReLU 通常 $I$ 更高          |
| $\operatorname{ReLU}(x)=\max(0,x)$                           | $x$ 为输入元素                                               | low precision 示例                             | FP16 比 FP32 少搬数据                  | PDF 第 25 页                          |
| $C=AB,;C_{ij}=\sum_k A_{ik}B_{kj}$                           | $A,B,C$ 为矩阵，$k$ 为归约维度                               | dense matmul                                   | 每个输出元素是一个 dot product         | Tensor Cores 加速对象                 |
| $\text{global reads}_{\text{tiled}}\sim N/T$                 | $N$ 为矩阵维度，$T$ 为 tile size                             | tiled matmul                                   | tile 内复用把 HBM 访问减少约 $T$ 倍    | PDF 第 42 页                          |
| $#\text{tiles}=\left\lceil\frac{N}{T_m}\right\rceil\left\lceil\frac{N}{T_n}\right\rceil$ | $T_m,T_n$ 为 tile shape                                      | wave quantization                              | tile 数与 SM 数不匹配会造成尾波浪费    | 1792→1793 案例                        |
| $\operatorname{softmax}(x_i)=\frac{e^{x_i-m}}{\sum_j e^{x_j-m}}$ | $m=\max_jx_j$                                                | stable softmax                                 | 减去最大值避免 overflow                | FlashAttention 的 softmax 难点        |
| $m_j=\max(m_{j-1},\max_{x\in s^{(j)}}x)$                     | $m_j$ 为到第 $j$ 个 tile 的 running max                      | online softmax                                 | 逐 tile 更新全局最大值                 | FlashAttention                        |
| $u_j=e^{m_{j-1}-m_j}u_{j-1}+\sum_{x_t\in s^{(j)}}e^{x_t-m_j}v_t$ | $u_j$ 为未归一化 value accumulator                           | FlashAttention forward                         | 旧累积值换到新最大值尺度下             | online telescoping softmax            |
| $\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$ | $Q,K,V$ 为 attention 输入，$d_k$ 为维度缩放                  | Transformer attention                          | 先算相关性，再用 softmax 加权 value    | FlashAttention 保持该公式不变         |

### 12.1 Roofline 公式的推导直觉

一个 kernel 的吞吐量不能超过硬件峰值算力：
$$
\text{Throughput}\le P_{\text{peak}}
$$
同时，如果它每秒最多只能从内存搬 $B_{\text{mem}}$ bytes，而每 byte 只能支持 $I$ FLOPs，那么由内存带宽决定的计算吞吐上限是：
$$
\text{Throughput}\le B_{\text{mem}}I
$$
两者都必须满足，所以取较小值：
$$
\text{Throughput} \le \min(P_{\text{peak}},B_{\text{mem}}I)
$$
这个公式告诉我们：如果 $I$ 太低，优化方向不是换更强算力，而是减少 bytes moved 或提高数据复用。

### 12.2 Tiling 公式的推导直觉

非 tiled matmul 中，某个输入元素可能被许多输出元素重复需要。如果每次都从 global memory 读，就会造成大量 HBM traffic。

Tiling 的做法是把一个 $T$ 大小的数据块读入 shared memory，然后让 block 内多个 threads 反复使用它。于是原来从 HBM 重复读的 $T$ 次，变成读一次 HBM，再在 shared memory 中读多次。

所以 global memory access 约减少 $T$ 倍。

### 12.3 Online softmax 公式的推导直觉

普通 softmax 要知道整行最大值 $m$。但 tile-by-tile 计算时，未来 tile 可能出现更大值。如果新最大值从 $m_{j-1}$ 变为 $m_j$，旧的指数累计值必须重缩放：
$$
e^{x-m_{j-1}}
=
e^{x-m_j}e^{m_j-m_{j-1}}
$$
等价地，旧 accumulator 需要乘：
$$
e^{m_{j-1}-m_j}
$$
这就是公式中缩放因子的来源。它保证逐 tile 累积结果与一次性 full softmax 等价。

------

## 13. 概念对比与易混淆点

| 易混淆概念                              | 区别                                                         | 联系                       | 常见误解                           | 正确理解                                   |
| --------------------------------------- | ------------------------------------------------------------ | -------------------------- | ---------------------------------- | ------------------------------------------ |
| CPU vs GPU                              | CPU 优化 latency；GPU 优化 throughput                        | 都能执行通用程序           | GPU 是更快 CPU                     | GPU 是 massive parallel throughput machine |
| Thread vs Block vs Warp                 | thread 做工作；block 是 thread 组；warp 是 32-thread 执行单位 | 都属于 GPU execution model | block 和 warp 一样                 | block 是编程分组，warp 是硬件调度粒度      |
| L1 cache vs Shared memory               | L1 自动缓存；shared memory 显式控制                          | 都靠近 SM 且较快           | 二者完全一样                       | shared memory 是 tiling 可控复用的关键     |
| Memory-bound vs Compute-bound           | memory-bound 受带宽限制；compute-bound 受算力限制            | roofline 中两种区域        | FLOPs 高就一定快                   | 数据搬不动时 FLOPs 峰值没有意义            |
| bytes/FLOP vs FLOPs/byte                | 一个是移动成本，一个是 operational intensity                 | 互为倒数                   | 课程中“intensity”说法总是一致      | 阅读时要看单位                             |
| Low precision vs Quantization           | low precision 可用于训练计算；quantization 常指表示压缩      | 都减少 bits                | 低精度越低越好                     | 需要数值稳定和 scale factor                |
| Fusion vs Tiling                        | fusion 合并算子；tiling 分块复用数据                         | 都减少 HBM traffic         | 二者是同一种优化                   | fusion 减少中间写回，tiling 增加局部复用   |
| Recomputation vs 缓存 activation        | recomputation 少存多算；缓存 activation 多存少算             | 都服务 backward            | 重算一定浪费                       | 在 memory 昂贵时重算可能更优               |
| Coalescing vs Alignment                 | coalescing 关注 warp 访问合并；alignment 关注地址/维度边界   | 常一起影响性能             | padding 只是增加无用计算           | padding 可能改善访问合并和 tile 利用率     |
| FlashAttention vs Approximate Attention | FlashAttention 是 exact attention 的 IO-aware 实现           | 都试图降低 attention 成本  | FlashAttention 改了 attention 公式 | 它主要改变执行方式，不改变数学定义         |

------

## 14. 学习拓展与发散理解

### 14.1 与大模型训练实践的关系

**延伸理解：数据准备。** 本讲主要讨论 GPU kernel 和硬件性能，但它也提醒我们，数据 pipeline 不能让 GPU 饿着。如果 CPU 数据加载、tokenization、batch 构造过慢，GPU 即使 kernel 很快也会空等。

**延伸理解：模型结构。** 现代 Transformer 大量使用 dense matmul，是因为它既适合数学表达，也适合 Tensor Core 加速。若模型包含大量不规则稀疏访问或复杂分支，理论 FLOPs 变少并不一定代表实际更快。

**延伸理解：优化策略。** Low precision、activation checkpointing、FlashAttention 都体现同一个系统权衡：用数值表示、计算量和重算，换取更低的显存占用和 HBM traffic。

**延伸理解：训练效率。** 对大模型训练来说，GPU 利用率、memory bandwidth、kernel fusion、通信开销都会转化为训练成本。理解本讲内容有助于判断为什么某个实现“看起来 FLOPs 少但实际慢”。

**延伸理解：推理部署。** 推理中 batch size、sequence length、KV cache layout、attention kernel、量化格式都会影响性能。FlashAttention 的 IO-aware 思想也会影响长上下文推理优化。

### 14.2 与其他 lecture 的联系

本讲向前连接模型 scaling 和 Transformer 训练目标：模型规模增长需要更多 compute，而 compute 的实际可用性取决于 GPU scaling 和硬件利用率。

本讲向后连接 parallelism、distributed training 和 inference：当单卡 GPU 优化不足以支撑更大模型时，就需要张量并行、流水线并行、数据并行、expert parallelism 以及跨设备通信优化。PDF 第 13 页也提示 accelerator networking 会在 parallelism lecture 中讨论。

### 14.3 面试 / 考试 / 作业可能考察角度

| 问题                                          | 考察点                                     | 回答思路                                                     |
| --------------------------------------------- | ------------------------------------------ | ------------------------------------------------------------ |
| 为什么 GPU 适合训练 Transformer？             | GPU throughput、dense matmul、Tensor Cores | 对比 CPU latency 与 GPU throughput，再解释 Transformer 的矩阵乘法结构 |
| 什么是 roofline model？                       | memory-bound / compute-bound               | 写出公式，解释 $P_{\text{peak}}$、$B_{\text{mem}}$、$I$      |
| 为什么 FP16 ReLU 比 FP32 更有利于 bandwidth？ | low precision 与 bytes moved               | FP32 读写 8 bytes/FLOP，FP16 读写 4 bytes/FLOP               |
| 为什么 `if/else` 在 GPU 上可能慢？            | SIMT 与 warp divergence                    | 同一 warp 不同分支会串行执行并 mask                          |
| Operator fusion 为什么能加速？                | 减少中间结果写回                           | 用 $\sin^2(x)+\cos^2(x)$ 例子说明 5 kernels 到 1 kernel      |
| Recomputation 为什么可能更快？                | compute-memory tradeoff                    | 用三个 sigmoid 从 8 次 memory access 到 5 次解释             |
| Tiling 如何降低 global memory access？        | shared memory reuse                        | 推导 $N$ 到 $N/T$ 的访问次数下降                             |
| Padding 为什么能加速？                        | alignment、tile、wave quantization         | 用 vocab padding 或 1792→1793 例子说明                       |
| FlashAttention 为什么是 exact？               | online softmax                             | 说明 running max / denominator 保证结果等价于 full softmax   |
| FlashAttention backward 为什么重算？          | recomputation                              | 避免保存完整 $N^2$ attention matrix                          |

------

## 15. 课程复习总结

### 15.1 本讲核心思想

本讲的核心思想是：**语言模型 scaling 需要大量 compute，但现代 GPU 上真正稀缺的不只是计算能力，而是高效地把数据送到计算单元并复用数据的能力。** 因此，理解 GPU/TPU 硬件模型、memory hierarchy、SIMT、roofline、低精度、fusion、recomputation、coalescing、tiling 和 FlashAttention，是理解大模型系统效率的基础。

PDF 最后一页也将全讲总结为三点：硬件推动 scale，低层细节决定哪些东西能 scale；当前 GPU-based compute 强烈鼓励围绕 matmul 和 data movement 思考；仔细考虑 coalescing、tiling、fusion 会带来好性能。

### 15.2 必须掌握的概念

| 概念                                              | 复习重点                              |
| ------------------------------------------------- | ------------------------------------- |
| CPU vs GPU                                        | latency vs throughput                 |
| SM / SP                                           | GPU 的并行执行结构                    |
| Thread / Block / Warp                             | GPU 编程和硬件调度基本单位            |
| Registers / Shared memory / L1/L2 / Global memory | 内存层级决定数据移动成本              |
| SIMT                                              | 同一 warp 执行相同指令                |
| Control divergence                                | 分支导致 warp 内部分 thread 空等      |
| Tensor Cores                                      | 专门加速矩阵乘法                      |
| Roofline model                                    | 判断 memory-bound / compute-bound     |
| Low precision                                     | 减少 bits、减少搬运、启用低精度硬件   |
| Operator fusion                                   | 合并小算子，减少中间写回              |
| Recomputation                                     | 用计算换内存                          |
| Memory coalescing                                 | warp 内访问连续地址                   |
| Tiling                                            | shared memory 中复用 tile             |
| Alignment / padding                               | 改善硬件利用率                        |
| Wave quantization                                 | tile waves 与 SM 数不匹配导致尾波浪费 |
| FlashAttention                                    | exact attention 的 IO-aware 实现      |

### 15.3 必须理解的公式

| 公式                                                         | 解决的问题                           |
| ------------------------------------------------------------ | ------------------------------------ |
| $\text{Throughput}\le \min(P_{\text{peak}},B_{\text{mem}}I)$ | 性能瓶颈判断                         |
| $I=\frac{\text{FLOPs}}{\text{Bytes moved}}$                  | 衡量数据复用能力                     |
| $C_{ij}=\sum_k A_{ik}B_{kj}$                                 | 矩阵乘法基本形式                     |
| $\text{global reads}_{\text{tiled}}\sim N/T$                 | tiling 为什么减少 HBM 访问           |
| $\text{tiles}=\lceil N/T_m\rceil\lceil N/T_n\rceil$          | wave quantization 分析               |
| $\operatorname{softmax}(x_i)=\frac{e^{x_i-m}}{\sum_j e^{x_j-m}}$ | stable softmax                       |
| $m_j=\max(m_{j-1},\max s^{(j)})$                             | online softmax running max           |
| $u_j=e^{m_{j-1}-m_j}u_{j-1}+\sum e^{x_t-m_j}v_t$             | FlashAttention tile-wise accumulator |
| $\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$ | 标准 attention 数学定义              |

### 15.4 必须看懂的代码

| 代码 / 变量                              | 学习重点                          |
| ---------------------------------------- | --------------------------------- |
| `cuda_if_available(index=0)`             | GPU device 与 CPU fallback        |
| `get_max_memory_usage(func)`             | 如何测量 CUDA peak memory         |
| `h100_flop_per_sec`                      | roofline 中的 peak compute        |
| `h100_bytes_per_sec`                     | roofline 中的 memory bandwidth    |
| `h100_flop_per_sec / h100_bytes_per_sec` | 机器 balance 与 memory-bound 判断 |

### 15.5 一句话总结

**Lecture 5 用 GPU 内存层级和数据移动这条主线，把大模型 compute scaling、矩阵乘法性能异常和 FlashAttention 的 IO-aware 设计连接成了一套完整的系统性能心智模型。**