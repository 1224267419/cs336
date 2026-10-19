# Lecture 6 课程报告：从 GPU Kernel 到 Triton Tiling 的高性能计算讲义

## 0. 资料来源与改写说明

本报告是在用户上传的原始 Lecture 6 中文课程报告基础上进行系统性改写、可读性优化和学习化增强。原始报告标题为 **“Lecture 6 课程报告：GPU Kernels、Triton、Benchmarking/Profile 与 Tiling”**，其主线包括 GPU 硬件回顾、benchmarking / profiling、GeLU kernel fusion、Triton 编程模型、softmax、row sum、tiled matmul 与 ReLU fusion 等内容。 

本次改写遵循以下原则：

- **不重新从零生成**：保留原报告的课程顺序、核心内容、公式、代码解释、图表说明和引用标记。
- **增强学习价值**：对每个重要概念补充“为什么需要它”“它解决什么问题”“代码如何体现它”“复习时抓住什么”。
- **区分课程内容与拓展内容**：凡是为了帮助理解而加入的内容，会用 **学习提示**、**工程直觉**、**补充说明** 或 **延伸理解** 标注，不把它伪装成课程原文。
- **保留原报告引用格式**：原报告中的 `【字幕†source】`、`【Python代码†source】`、`【图表†source】`、`【资料核对†source】` 均尽量保留，用来表示原报告中已有的来源指向。
- **资料核对说明**：我同时核对了 `stanford-cs336/lectures` GitHub 仓库。该仓库 README 显示其为 Spring 2026 CS336 lecture materials，并说明 executable lectures 命名为 `lecture_XX.py`，non-executable lectures 命名为 `lecture_XX.pdf`；仓库文件列表中确实包含 `lecture_06.py`。 `lecture_06.py` 的主流程也与原报告一致：GPU review → benchmarking/profiling → GeLU 对比 → Triton intro → Triton GeLU → softmax → row sum → matmul+ReLU。

需要特别说明：当前可见输入中没有单独上传字幕文件或 Lecture 6 PDF；因此，本报告保留原报告中已有的字幕、Python 代码和图表引用标记，但不会声称我直接读取了独立字幕原文或不存在的 PDF。原报告也明确指出未找到 `lecture_06.pdf`，Lecture 6 主要对应 `lecture_06.py`，并且不展开 XLA 内容，因为报告核对到 `lecture_06.py` 中没有实际 XLA 讲解【资料核对†source】。

------

## 1. 学习导读

### 1.1 本讲要解决的核心问题

Lecture 6 的核心问题是：

> **同样的张量运算，为什么在 GPU 上可能快很多，也可能慢得离谱？我们如何通过 profiling、kernel fusion、Triton 和 tiling，把语言模型中的计算高效映射到 GPU 硬件上？**

上一讲已经从较高层次介绍了 GPU 和性能。本讲进一步进入代码层面：先理解 GPU 的真实硬件约束，再学习如何测量代码性能，随后用 GeLU、softmax、row sum 和 matmul+ReLU 作为例子，逐步展示从 element-wise kernel 到 tiled matrix multiplication 的 Triton 编程思想【字幕†source】【Python代码†source】。

这节课的学习重点不是单纯记住某个 kernel 怎么写，而是建立一种系统直觉：

> 高性能深度学习代码的关键，经常不在于数学公式本身，而在于如何减少 HBM 访问、如何让数据在 SM 附近复用、如何用合适的 block / tile 划分计算。

### 1.2 本讲在 CS336 课程中的位置

CS336 的主题是从零理解和构建语言模型。语言模型训练中，大量时间花在矩阵乘法、softmax、归一化、激活函数等张量运算上。Transformer 的 MLP 层依赖 matmul + activation，attention 依赖 matmul + softmax + matmul，训练时还要反复进行前向、反向和优化器更新。

因此，本讲处在 **模型数学** 与 **训练系统** 的交界处：

- 从数学角度看，GeLU、softmax、matmul 都是熟悉的张量操作；
- 从系统角度看，这些操作必须被拆成 GPU kernel；
- 从性能角度看，真正决定速度的往往是 HBM 读写、kernel launch、memory coalescing、occupancy、tiling 和 fusion。

原报告总结得很准确：本讲是在真实 GPU 硬件约束中理解语言模型中的张量计算，并为后续 FlashAttention、多 GPU 编程和训练系统优化做铺垫【字幕†source】。

### 1.3 本讲知识结构

| 模块                     | 解决的问题                              | 核心概念                                                     | 与后续内容的关系                                             |
| ------------------------ | --------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| GPU 硬件层次             | 为什么内存位置会极大影响性能？          | SM、register、L1/shared memory、L2、HBM                      | 后续所有 kernel 优化都围绕“减少远端 HBM 访问、增加近端复用”展开 |
| GPU 执行模型             | 如何把计算拆成并行执行单元？            | thread、thread block / CTA、grid、warp                       | Triton 的 program instance 基本对应 block-level 思维         |
| 硬件性能细节             | 为什么正确的 kernel 不一定快？          | control divergence、occupancy、bank conflict、memory coalescing、block occupancy | 解释 profile 中性能差异的底层原因                            |
| Benchmarking / Profiling | 如何知道代码慢在哪里？                  | CUDA events、synchronize、PyTorch profiler、Nsight           | 优化前必须先测量，否则容易凭直觉误判                         |
| GeLU 与 kernel fusion    | 为什么多个 PyTorch primitive 会慢？     | naive GeLU、builtin GeLU、`torch.compile`、fusion            | 引出 Triton 手写 kernel 的动机                               |
| Triton 编程模型          | 如何用接近 Python 的方式写 GPU kernel？ | `tl.program_id`、`tl.arange`、`tl.load`、`tl.store`、mask    | 后续 softmax、row sum、matmul 都复用同一模板                 |
| Softmax reduction        | 如何处理一行内需要聚合的操作？          | row-wise reduction、数值稳定、one row one block              | attention softmax 的基础                                     |
| Row sum tiling           | 一行放不进一个 block 时怎么办？         | tile、accumulator、block 内循环                              | 为 matmul tiling 做预热                                      |
| Matmul+ReLU              | 如何提高 arithmetic intensity？         | tiled GEMM、`tl.dot`、strides、activation fusion             | 是深度学习高性能 kernel 的核心范式                           |

### 1.4 学习本讲前需要知道什么

学习本讲最好具备以下前置知识：

1. **基本线性代数**
    需要知道矩阵乘法：
   $$
   C_{mn} = \sum_k A_{mk}B_{kn}
   $$

2. **基本概率与 softmax**
    需要知道 softmax 会把一组 logits 转换成归一化概率。

3. **PyTorch 张量操作**
    需要理解 `x + y`、`x @ y`、`torch.exp`、`torch.nn.functional.gelu` 这类高层表达式。

4. **GPU 异步执行的基本概念**
    需要知道 CPU 发起 GPU 操作后，GPU 通常异步执行，所以计时必须同步。

5. **Transformer 中常见算子**
    GeLU、softmax、matmul 都是 Transformer 训练和推理中的核心操作。

**学习提示**：如果你只从数学角度理解这些算子，会觉得它们很简单；但本讲的重点是从“数据在哪里、怎么移动、怎么复用”的角度重新理解它们。

------

## 2. 第一部分：GPU 硬件层次与编程模型

这一节要解决的问题是：**为什么 GPU kernel 的性能高度依赖硬件层次？为什么同一个数学操作，换一种数据划分方式就可能快很多？**

### 2.1 问题引入：高带宽不等于无限带宽

GPU 的计算能力非常强，但计算单元只有在拿到数据时才能工作。大模型训练中的很多性能瓶颈并不是“不会算”，而是“数据搬得太慢”或“搬了太多次”。

课程首先回顾 GPU 的典型结构：GPU 芯片包含多个 **Streaming Multiprocessors，SMs**；每个 SM 内部有 registers、L1 cache / shared memory；芯片级有 L2 cache；芯片外或旁边有 HBM，即 High Bandwidth Memory【字幕†source】。`lecture_06.py` 中还用表格对比 A100、H100、B200，包括 SM 数量、register 容量、L1/shared memory、L2、HBM 容量和带宽等【Python代码†source】。

### 2.2 概念定义：GPU 的内存层次

直觉上，可以把 GPU 内存层次理解成一套“离计算单元远近不同的存储空间”：
$$
\text{越靠近 SM：越快、越小；越远离 SM：越大、越慢。}
$$

| 层次               | 位置                       | 特点                                             | 学习重点                                     |
| ------------------ | -------------------------- | ------------------------------------------------ | -------------------------------------------- |
| Registers          | 每个 thread 私有           | 最快，容量极小                                   | 保存局部变量、accumulator                    |
| Shared memory / L1 | 每个 SM 内部               | 快，但容量有限；shared memory 可由程序员显式利用 | block 内线程共享数据的关键                   |
| L2 cache           | GPU 芯片级共享             | 比 L1 大，但更慢                                 | 多个 SM 共享的缓存层                         |
| HBM                | GPU 外部或旁边的高带宽内存 | 容量最大，但相对最远、最慢                       | 大多数张量数据存放处，也是性能优化的主要瓶颈 |

字幕中的直觉总结是：large memory 是 slow and far but big；register 和 L1 这类 fast memory 位于 SM 上，local and fast but small【字幕†source】。原报告也指出，`gpu-hardware.png` 展示了 SM 内部的 Reg 与 L1+shared memory、芯片级 L2 以及芯片外 HBM 的分层关系【图表†source】。

### 2.3 GPU 编程模型：thread、thread block、grid

GPU kernel 的基本执行结构可以写成：
$$
\text{Grid} \rightarrow \text{Thread Blocks / CTAs} \rightarrow \text{Threads}
$$

- **Thread**：最小执行单元，通常处理一个或少量数据元素。
- **Thread block / CTA**：一组 threads，可以被调度到同一个 SM，并能共享 shared memory。
- **Grid**：一次 kernel launch 启动的全部 thread blocks。

在 CUDA 的抽象中，你通常会思考“每个 thread 做什么”。而在 Triton 中，更自然的思维方式是“每个 program instance / block 处理哪一块数据”。

### 2.4 为什么需要 thread block，而不只是很多 thread？

如果是 element-wise operation，比如 GeLU：
$$
y_i = f(x_i)
$$
每个元素可以独立计算，thread 之间不需要交流。此时，让每个 thread 处理一个或多个元素就很自然。

但 softmax 和 matmul 不一样：

- softmax 需要对一整行求 max、exp、sum、normalize；
- matmul 中一个输出元素需要沿 $K$ 维累加，多个输出元素还会重复使用同一批输入元素。

如果每个 thread 都直接从 HBM 读自己需要的数据，再把中间结果写回 HBM，就会产生大量重复慢速访问【字幕†source】。

thread block 的作用在这里体现出来：

> 一个 block 被调度到一个 SM 上，block 内 threads 可以通过 shared memory 或 registers 附近的数据结构共享中间结果，从而减少对 HBM 的反复访问。

**工程直觉**：高性能 kernel 的常见套路是：
 先从 HBM 搬一小块数据到 SM 附近，再在 registers / shared memory 中尽量多算，最后把结果写回 HBM。

这也是后面 tiling 的核心。

### 2.5 本节学习提示

本节最重要的结论是：

> GPU 性能优化不是抽象地“多并行”就够了，而是要让数据尽可能少地往返 HBM，并尽可能在 SM 附近被复用。

容易混淆的地方：

- **HBM 带宽很高** 不等于可以随便反复读写；
- **thread 数量多** 不等于性能一定好；
- **shared memory 快** 但容量有限，需要 tile 设计；
- **Triton block-level 编程** 并不等于不用理解底层，只是抽象层次更适合写高性能张量 kernel。

------

## 3. 第二部分：硬件细节如何影响性能

这一节要解决的问题是：**为什么一个“数学上正确”的 kernel 可能性能很差？哪些硬件机制会影响 GPU kernel 的真实速度？**

课程明确强调，programming model 给了我们正确性所需的抽象，但性能对硬件细节非常敏感。`lecture_06.py` 也把 “Interaction between programming model and hardware” 作为单独部分，说明理解硬件是获得高性能的前提【Python代码†source】。

### 3.1 Warp 与 control divergence

#### 直觉解释

虽然我们在编程时说“很多 threads 并行执行”，但硬件不是完全逐个独立调度 thread。NVIDIA GPU 会把 threads 组织成 **warp**，一个 warp 通常包含 32 个 threads【字幕†source】。

一个 warp 内的 threads 通常以 lockstep 方式执行同一条指令。可以把它想象成 32 个学生必须跟着同一个口令做动作。如果一部分学生要做动作 A，另一部分学生要做动作 B，就不能真正同时完成，而要分批执行。

#### 正式定义

**Control divergence** 指同一个 warp 内不同 threads 因条件分支走向不同路径，导致硬件必须串行执行多个分支路径。

例如：

```
if condition:
    A
else:
    B
```

如果一个 warp 内部分 threads 满足 `condition`，部分不满足，那么硬件通常需要先执行 A 路径，再执行 B 路径，并对不属于当前路径的 threads 做 mask。这会降低并行效率【字幕†source】。

#### 学习提示

- control divergence 不是程序错误，而是性能问题；
- 它主要发生在同一个 warp 内部分支不一致时；
- 对 element-wise kernel，如果每个元素走不同分支，可能会降低 warp 执行效率。

### 3.2 Warp scheduler：用多个 warps 隐藏 HBM 延迟

访问 HBM 很慢。一个 warp 执行 `tl.load` 或 global memory load 时，可能需要等待数据返回。为了不让 SM 闲着，SM 上通常会同时驻留多个 warps。

当一个 warp 在等内存时，warp scheduler 可以切换到另一个 ready warp 继续执行。字幕中提到，这种切换可以用来隐藏 HBM 读写延迟【字幕†source】。

**工程直觉**：GPU 的高吞吐来自“用大量并发工作隐藏延迟”。不是每个 warp 每个周期都在计算，但只要有足够多 ready warps，SM 就不容易空转。

### 3.3 Occupancy：不是越高越好

#### 这是什么？

**Occupancy** 通常指一个 SM 上实际 resident warps 数量与硬件最大 resident warps 数量的比例。

直觉上，occupancy 越高，SM 上能切换的 warps 越多，越容易隐藏 HBM 等待。但课程特别提醒：occupancy 并不是越高越好【字幕†source】。

#### 课程例子

原报告给出的例子是：

- 每个 block 有 128 threads；
- 每个 thread 使用 160 registers；
- B200 每个 SM 有 65536 registers；
- 每个 SM 最多 64 warps。

每个 block 消耗的 register 数量是：
$$
R_{\text{block}} = 128 \times 160 = 20480
$$
每个 block 的 warp 数是：
$$
\frac{128}{32} = 4
$$
一个 SM 能放下的 block 数约为：
$$
\left\lfloor \frac{65536}{20480} \right\rfloor = 3
$$
因此 resident warps 数为：
$$
W_{\text{resident}} = 3 \times 4 = 12
$$
occupancy 为：
$$
\text{occupancy} = \frac{12}{64} = 18.75\% \approx 18\%
$$
原报告指出，这说明 register 使用量会限制同一 SM 上能同时驻留的 warps【字幕†source】【Python代码†source】。

#### 为什么不是越高越好？

如果每个 thread 使用更多 registers，occupancy 可能下降。但这可能允许每个 thread 做更多工作，减少 HBM 访问，或者实现 **thread coarsening**，即一个 thread 处理多个元素。这样反而可能提高整体性能【字幕†source】。

**学习提示**：occupancy 是诊断指标，不是最终目标。真正目标是吞吐、延迟和整体执行时间。

### 3.4 Bank conflicts：shared memory 的冲突

Shared memory 虽然快，但不是一个可以任意并发访问的“魔法数组”。硬件上 shared memory 被分成多个 banks。课程中说明 shared memory 有 32 个 banks，每个 bank 宽 4 bytes；每个 cycle 中，一个 bank 只能服务一个非同地址访问【字幕†source】。

如果一个 warp 的多个 threads 同时访问同一个 bank 的不同地址，这些访问就会被串行化，称为 **bank conflict**。

最坏情况：32 个 threads 同时访问映射到同一个 bank 的不同地址，会形成 32-way bank conflict。

**补充说明**：在 matmul 中，既要访问矩阵 A 的行，又要访问矩阵 B 的列。列访问容易造成不连续访问或 bank 冲突，因此实际高性能 GEMM kernel 往往需要复杂的数据布局优化，例如 swizzling【字幕†source】。

### 3.5 Memory coalescing：HBM 访问的合并

**Bank conflict** 发生在 shared memory；**memory coalescing** 发生在 global memory / HBM。

当一个 warp 的 32 个 threads 访问 HBM 时，硬件会尝试把这些访问合并成 cache line transaction。课程中提到典型 transaction 是 128 bytes；如果 32 个 threads 访问连续的 32 个 float32 元素，刚好是 32 × 4 bytes = 128 bytes，可以形成 full coalescing【字幕†source】。

如果 threads 访问分散地址，就可能需要多次 transaction，浪费带宽。

| 概念              | 发生位置            | 问题形式                                  | 优化目标               |
| ----------------- | ------------------- | ----------------------------------------- | ---------------------- |
| Bank conflict     | Shared memory       | 多个 threads 同时访问同一 bank 的不同地址 | 让访问分布到不同 banks |
| Memory coalescing | HBM / global memory | warp 访问分散在多条 cache line            | 让 warp 访问连续地址   |

### 3.6 Block occupancy：SM 数量带来的“尾波”问题

除了 warp occupancy，还有 **block occupancy / wave occupancy**。

假设 B200 有 148 个 SM。如果 launch 160 个 blocks，第一波最多运行 148 个 blocks，剩下 12 个 blocks 要到第二波运行。第二波只有 12 个 blocks，很多 SM 会闲置，形成所谓 wave quantization 或 tail wave 问题【字幕†source】。

这说明：

- block 数量太少可能填不满 GPU；
- block 数量与 SM 数量匹配不好时，最后一波会浪费；
- block size、grid size 不只是正确性参数，也会影响实际性能。

### 3.7 本节学习提示

本节最重要的结论是：

> GPU kernel 的性能由多个层次共同决定：warp 分支是否一致、resident warps 是否足够、shared memory 是否冲突、HBM 访问是否连续、block 数量是否能填满 SM。

复习时要特别区分：

| 易混淆点                             | 正确理解                                                     |
| ------------------------------------ | ------------------------------------------------------------ |
| occupancy 高就一定快                 | 不一定；更多 registers 或 thread coarsening 可能降低 occupancy 但提升复用 |
| bank conflict 和 coalescing 是一回事 | 不是；一个是 shared memory，一个是 HBM                       |
| 写出并行 kernel 就会快               | 不一定；访问模式和硬件资源约束可能让它很慢                   |
| block size 只影响正确性              | 不对；它还影响 occupancy、tail wave、memory pattern 和 shared memory 使用 |

------

## 4. 第三部分：Benchmarking 与 Profiling

这一节要解决的问题是：**优化 GPU 代码之前，如何可靠地知道它到底慢不慢、慢在哪里？**

### 4.1 基本哲学：先测量，再优化

课程给出的优化流程非常朴素但重要：
$$
\text{benchmark/profile}
\rightarrow
\text{modify code}
\rightarrow
\text{benchmark/profile again}
$$
Benchmarking 和 profiling 的目的不是为了生成好看的表格，而是避免“凭感觉优化”。原报告指出，课程在写 Triton kernel 前专门讲 benchmarking/profiling，是为了先知道瓶颈在哪里【字幕†source】【Python代码†source】。

### 4.2 Benchmarking：测端到端时间

#### 这是什么？

**Benchmarking** 回答的问题是：

> 这段代码从开始到结束到底用了多长时间？

它适合用来比较不同实现，例如 naive GeLU、builtin GeLU、compiled GeLU 谁更快；也适合观察性能随维度变化的趋势，例如 matmul 从 256 到 8192 的 scaling【字幕†source】。

#### 课程中的 benchmark 函数

`lecture_06.py` 中的 `benchmark(run)` 体现了正确测 GPU 时间的几个关键点【Python代码†source】：

| 步骤                       | 为什么需要                                            |
| -------------------------- | ----------------------------------------------------- |
| Warmup                     | 第一次运行可能包含编译、缓存初始化等一次性开销        |
| CUDA events                | 用 GPU event 记录 GPU 上的执行时间，减少 CPU 计时误差 |
| `torch.cuda.synchronize()` | GPU 异步执行，不同步会导致 CPU 过早结束计时           |
| 多次 trials                | 减少偶然波动，取平均时间                              |

平均时间可以写成：
$$
\bar{t} = \frac{1}{T}\sum_{i=1}^{T} t_i
$$
其中：

- $T$：测量次数；
- $t_i$：第 $i$ 次 GPU elapsed time；
- $\bar{t}$：平均运行时间。

#### 常见误区

最容易犯的错误是用 CPU 端 `time.time()` 包住 GPU 代码，但不调用 `torch.cuda.synchronize()`。因为 GPU 操作通常是异步提交，CPU 可能只是把任务发给 GPU 就继续往下走，导致测出来的时间偏小。

**学习提示**：只要测 GPU 时间，就要问自己：有没有 warmup？有没有同步？有没有排除一次性编译开销？

### 4.3 Profiling：看时间花在哪里

#### 这是什么？

**Profiling** 回答的问题是：

> 端到端时间里面，具体哪些 kernel 或操作花了时间？

Benchmarking 只能告诉你“慢”；profiling 才能告诉你“为什么慢”。

课程使用 PyTorch profiler，并提到作业中会使用 Nsight 获得更细节的信息【字幕†source】。`lecture_06.py` 中的 `profile(run)` 会收集 CUDA activity，并按 `cuda_time_total` 排序输出表格【Python代码†source】。

### 4.4 从 PyTorch 表达式到真实 CUDA kernel

课程 profile 了两个例子：

1. `a + b`
    高层看只是一个张量加法，但底层会调用 CUDA add kernel。
2. `a @ b`
    高层看只是矩阵乘法，但不同矩阵维度可能调用不同 matmul kernel。kernel 名称里可能包含 Cutlass、SM100、f32、tile shape 等信息【字幕†source】【Python代码†source】。

例如 kernel 名称里的：

- `cutlass`：NVIDIA 线性代数相关库；
- `sm100`：Blackwell 架构；
- `f32`：float32；
- `64x64x16`：某种 tile shape。

**补充说明**：这说明 PyTorch 的高级 API 是很好的生产力工具，但它隐藏了真实执行路径。优化时不能只看 Python 表达式，要看 profiler 里实际 launch 了什么 kernel。

### 4.5 本节学习提示

本节最重要的结论是：

> Benchmark 告诉你“快不快”，profile 告诉你“时间花在哪里”。高性能 GPU 编程必须两者结合。

复习时抓住三句话：

1. GPU 异步执行，所以 benchmark 必须 synchronize。
2. 高层 PyTorch 表达式可能对应多个底层 CUDA kernels。
3. 写 Triton kernel 前，应该先 profile，确认瓶颈真的来自 kernel 数量、HBM 流量或访问模式。

------

## 5. 第四部分：GeLU、`torch.compile` 与 Kernel Fusion

这一节要解决的问题是：**为什么同一个 GeLU 数学函数，naive PyTorch、builtin PyTorch 和 `torch.compile` 的性能会不同？kernel fusion 到底节省了什么？**

### 5.1 GeLU 的数学形式

GeLU 是 Transformer 和大语言模型中常见的激活函数。本讲使用 tanh approximation：
$$
\operatorname{GELU}(x)
\approx
\frac{1}{2}x
\left(
1 + \tanh\left(
\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)
\right)
\right)
$$
符号解释：

| 符号           | 含义                                      |
| -------------- | ----------------------------------------- |
| $x$            | 输入张量中的一个元素                      |
| $x^3$          | 对输入元素做三次方                        |
| $0.044715x^3$  | tanh approximation 中的经验近似项         |
| $\sqrt{2/\pi}$ | 约等于 `0.79788456`，代码中直接使用该常数 |
| 输出           | 与输入形状相同的 element-wise 激活结果    |

`lecture_06.py` 中：

- `naive_gelu(x)`：直接用 PyTorch primitive 写公式；
- `builtin_gelu(x)`：调用 `torch.nn.functional.gelu(x, approximate="tanh")`；
- `compiled_gelu`：由 `torch.compile(naive_gelu)` 生成【Python代码†source】。

### 5.2 三种实现的性能差异

| 实现            | 写法                  | 底层行为                                         | 性能直觉                        |
| --------------- | --------------------- | ------------------------------------------------ | ------------------------------- |
| Naive PyTorch   | 手写公式              | 多个 PyTorch primitive，可能 launch 多个 kernels | 中间结果反复写回 HBM，慢        |
| Builtin GeLU    | 调用 PyTorch 内置函数 | 通常是专门优化的 fused kernel                    | 减少 HBM 往返，快               |
| `torch.compile` | 编译 naive 函数       | 编译器生成 fused Triton kernel                   | 自动 fusion，通常显著快于 naive |

字幕中明确指出，naive GeLU 的 profiler 会看到多个 kernel，例如 binary functor、unary add、tanh 等。每个 kernel 都要从 HBM 读数据，计算后写回 HBM，下一个 kernel 再读回来，因此慢【字幕†source】。

与之相比，builtin GeLU 和 compiled GeLU 都能把多个操作融合成一个 kernel，大致变成：
$$
\text{read } x \text{ once}
\rightarrow
\text{locally compute all scalar ops}
\rightarrow
\text{write } y \text{ once}
$$

### 5.3 Kernel fusion 的本质

GeLU 公式可以被拆成多个 primitive：
$$
x
\rightarrow
x^3
\rightarrow
x + 0.044715x^3
\rightarrow
\tanh(\cdot)
\rightarrow
y
$$
如果每一步都是单独 kernel，那么每一步都需要：
$$
\text{read from HBM}
\rightarrow
\text{compute on SM}
\rightarrow
\text{write to HBM}
$$
Fusion 后，中间结果尽量保留在 registers 或 SM 附近，而不是反复写回 HBM。

#### 为什么这对语言模型重要？

Transformer 中有很多类似的 memory-bound 操作：

- activation；
- bias add；
- dropout；
- residual add；
- normalization 中的 element-wise 部分；
- attention 中某些 mask / scale / softmax 周边操作。

这些操作本身 FLOPs 不一定多，但如果每一步都读写 HBM，就会成为性能瓶颈。

**工程直觉**：kernel fusion 的核心收益不是“数学变少了”，而是“内存搬运少了”。

### 5.4 `torch.compile` 不保证永远最快

课程问答提醒，compiled Triton kernel 不一定总比 builtin CUDA kernel 快。在本例中，compiled Triton kernel 甚至可能比 builtin GeLU 慢一些，性能依赖硬件、编译器优化程度和手写库 kernel 的成熟度【字幕†source】。

这非常重要：`torch.compile` 是强大的自动优化工具，但它不是“无脑最快”按钮。

### 5.5 本节学习提示

本节最重要的结论是：

> Naive PyTorch 写法可能数学上很清楚，但底层可能拆成多个 kernel，导致大量 HBM 往返；kernel fusion 通过把多个操作放进一个 kernel 来减少内存流量。

复习时抓住：

- GeLU 是 element-wise operation；
- naive GeLU 慢主要因为多 kernel、多次 HBM 读写；
- builtin / compiled 快主要因为 fusion；
- `torch.compile` 有帮助，但必须 benchmark/profile 验证。

------

## 6. 第五部分：Triton 编程模型与 GeLU Kernel

这一节要解决的问题是：**如何从 PyTorch 高层操作过渡到手写 GPU kernel？Triton 为什么适合作为学习和实现高性能 kernel 的中间层？**

### 6.1 CUDA vs Triton：per-thread vs per-block

CUDA 的典型思维是：

> 写每个 thread 做什么。

优点是控制非常细；缺点是程序员需要管理很多细节，例如 shared memory、同步、thread-level indexing 等。

Triton 的典型思维是：

> 写每个 program instance / block 做什么。

课程中说，Triton 让你以 thread block 为中心思考：加载一块数据，在 shared memory / registers 附近操作，再写回 global memory【字幕†source】。`lecture_06.py` 的 `triton_introduction()` 也将 Triton 描述为介于 PyTorch 大张量操作和 CUDA 单 thread 操作之间的抽象层【Python代码†source】。

| 层次    | 编程思维                    | 优点                                | 缺点                       |
| ------- | --------------------------- | ----------------------------------- | -------------------------- |
| PyTorch | 整个 tensor 做什么          | 简洁、易写                          | 很难控制 kernel 和内存访问 |
| Triton  | 每个 block / program 做什么 | 适合 tile-level kernel，接近 Python | 仍需理解 GPU 硬件          |
| CUDA    | 每个 thread 做什么          | 控制最细                            | 编写和调试成本高           |

### 6.2 Triton GeLU 外层包装函数

`triton_gelu(x)` 的外层 Python 逻辑是【Python代码†source】：

1. 检查输入是否在 CUDA 上；
2. 检查输入是否 contiguous；
3. 分配输出 `y = torch.empty_like(x)`；
4. 计算元素总数 `num_elements = x.numel()`；
5. 设定 `BLOCK_SIZE = 1024`；
6. 用 `triton.cdiv(num_elements, BLOCK_SIZE)` 计算 block 数；
7. 用 grid `(num_blocks,)` launch `triton_gelu_kernel`。

这里的核心是：输入 tensor 可能有 8192 个元素，不能假设整个 tensor 都放进一个 SM。因此要把它切成多个连续 blocks，每个 Triton program instance 负责一个 block【字幕†source】。

### 6.3 Triton GeLU kernel：从 `pid` 到 `tl.store`

Triton kernel 的基本模板是：
$$
\text{identify block}
\rightarrow
\text{compute offsets}
\rightarrow
\text{load}
\rightarrow
\text{compute}
\rightarrow
\text{store}
$$
具体到 GeLU：

#### 第一步：识别当前 block

$$
\text{pid} = \operatorname{tl.program\_id}(0)
$$

`pid` 是当前 program instance / block 的编号。

#### 第二步：计算当前 block 的起始位置

$$
\text{start} = \text{pid} \times \text{BLOCK\_SIZE}
$$

#### 第三步：构造 offsets

$$
\text{offsets} = \text{start} + [0,1,\dots,\text{BLOCK\_SIZE}-1]
$$

#### 第四步：mask 防止越界

如果 `num_elements` 不能被 `BLOCK_SIZE` 整除，最后一个 block 会越界。因此需要：
$$
\text{mask}_i = (\text{offsets}_i < \text{num\_elements})
$$

#### 第五步：load、compute、store

$$
x = \operatorname{tl.load}(x_{\text{ptr}} + \text{offsets}, \text{mask})
$$

计算 GeLU 后：
$$
\operatorname{tl.store}(y_{\text{ptr}}+\text{offsets}, y, \text{mask})
$$
字幕逐步解释了 `x_ptr`、`y_ptr`、`offsets`、mask、`tl.load` 和 `tl.store` 的含义；原报告指出，`lecture_06.py` 的 `triton_gelu_kernel` 正是按 `pid → start → offsets → mask → load → compute → store` 结构实现【字幕†source】【Python代码†source】。

### 6.4 Triton 中的 GeLU 公式实现

Triton kernel 中没有直接调用 `tanh`，而是使用：
$$
\tanh(a) = \frac{e^{2a}-1}{e^{2a}+1}
$$
其中：
$$
a = 0.79788456 \cdot (x + 0.044715x^3)
$$
所以：
$$
y = 0.5x(1+\tanh(a))
$$
这与前面的 GeLU tanh approximation 对应【Python代码†source】。

### 6.5 Triton 到 PTX：写的是抽象，执行的是编译后代码

课堂中特别指出，`tl.load` 不是 GPU 真的在运行 Python 函数。Triton 代码会被编译成 PTX，再由 GPU 执行【字幕†source】。

`lecture_06.py` 中会打印 `compiled_kernel.asm["ptx"]`。课程观察包括【Python代码†source】：

- `ld.global`：从 global memory / HBM 加载；
- `st.global`：写回 global memory / HBM；
- `%ctaid.x`：block index；
- `%tid.x`：thread index；
- `%r`：integer registers；
- `%f`：floating-point registers；
- compiler 可能做 thread coarsening，让一个 thread 处理多个元素。

**学习提示**：Triton 提高了抽象层次，但不改变底层事实：最终执行的是 GPU 指令，性能仍然受 memory hierarchy、warp、register、coalescing 等影响。

### 6.6 本节学习提示

本节最重要的结论是：

> Triton 的核心学习模板是：确定当前 block 负责哪块数据，构造 offset 和 mask，从 HBM load，局部计算，再 store 回 HBM。

复习时必须看懂：

| Triton 元素                | 含义                                 |
| -------------------------- | ------------------------------------ |
| `tl.program_id(axis=0)`    | 当前 program / block 的编号          |
| `tl.arange(0, BLOCK_SIZE)` | block 内并行处理的 offsets           |
| `mask`                     | 防止最后一个 block 越界              |
| `tl.load`                  | 从 global memory 读取                |
| `tl.store`                 | 写回 global memory                   |
| `tl.exp`                   | 在 block 内做向量化计算              |
| `tl.constexpr`             | 编译期常量，帮助 Triton 生成高效代码 |

------

## 7. 第六部分：Softmax：行级 Reduction

这一节要解决的问题是：**softmax 不是 element-wise 操作，因为一行元素之间需要共同求 max 和 sum；那么它应该如何映射到 Triton block？**

### 7.1 Softmax 的定义与数值稳定性

给定矩阵：
$$
X \in \mathbb{R}^{M \times N}
$$
对第 $i$ 行做 softmax：
$$
\operatorname{softmax}(X_{i,:})_j
=
\frac{\exp(X_{ij})}{\sum_{k=0}^{N-1}\exp(X_{ik})}
$$
为了数值稳定，通常先减去该行最大值：
$$
m_i = \max_j X_{ij}
$$
然后：
$$
\operatorname{softmax}(X_{i,:})_j
=
\frac{\exp(X_{ij}-m_i)}
{\sum_{k=0}^{N-1}\exp(X_{ik}-m_i)}
$$
为什么可以减去 $m_i$？

因为分子分母同时乘以同一个常数因子：
$$
\frac{\exp(X_{ij}-m_i)}
{\sum_k \exp(X_{ik}-m_i)}
=
\frac{\exp(X_{ij})\exp(-m_i)}
{\sum_k \exp(X_{ik})\exp(-m_i)}
=
\frac{\exp(X_{ij})}
{\sum_k \exp(X_{ik})}
$$
减去最大值不改变结果，但可以避免 $\exp$ 输入过大导致 overflow。

课程说明 softmax 会对矩阵每一行 exponentiate and normalize，用于 attention 和生成输出概率【字幕†source】。`lecture_06.py` 中还用例子说明 `[0,0,0]` 会变为均匀分布，`[1,1,-inf]` 会变为 `[1/2,1/2,0]`【Python代码†source】。

### 7.2 Naive PyTorch Softmax 的内存读写

原报告中 `naive_softmax(x)` 拆成五步【Python代码†source】：

1. 每行求 max；
2. 减去 max；
3. exp；
4. 每行求 sum；
5. normalize。

内存访问大致为：

| 步骤     | 操作             | 读       | 写   |
| -------- | ---------------- | -------- | ---- |
| max      | `x.max(dim=1)`   | $MN$     | $M$  |
| subtract | `x - max`        | $MN + M$ | $MN$ |
| exp      | `exp(...)`       | $MN$     | $MN$ |
| sum      | `exp.sum(dim=1)` | $MN$     | $M$  |
| divide   | `exp / sum`      | $MN + M$ | $MN$ |

总读次数：
$$
\text{Reads} = 5MN + M
$$
字幕用简化说法概括为大约 $5MN$ reads 和 $3MN$ writes，并强调这是多个 PyTorch kernel 之间反复读写 HBM 的结果【字幕†source】。

理想情况下，如果一整行能放入一个 block，就可以：
$$
\text{read row once}
\rightarrow
\text{compute max/exp/sum locally}
\rightarrow
\text{write row once}
$$
也就是接近 $MN$ reads 和 $MN$ writes。

### 7.3 为什么 softmax 是 row-wise，而不是 element-wise？

Softmax 的一个输出元素依赖同一行的所有输入元素：

- 要知道全行 max；
- 要知道全行 exp sum；
- 才能算每个元素的归一化结果。

因此，softmax 不是 element-wise。

但不同 row 之间互不依赖，所以它是 row-wise。课程选择：
$$
\text{one row}
\leftrightarrow
\text{one Triton program / block}
$$
字幕明确解释：每一行需要 normalize 和 sum，因此 softmax 不是 element-wise；但它是 row-wise，所以 blocks 之间不需要交互【字幕†source】。

### 7.4 Triton Softmax kernel 结构

`triton_softmax(x)` 的外层逻辑是【Python代码†source】：

- 输入 $x$ 形状为 $(M,N)$；
- 输出 $y$ 与 $x$ 同形状；
- `block_size = triton.next_power_of_2(N)`；
- `num_blocks = M`；
- launch grid 为 `(M,)`；
- 传入 row stride / column stride，用于地址计算。

kernel 内部：

1. `row_idx = tl.program_id(0)`：当前 block 负责哪一行；
2. `col_offsets = tl.arange(0, BLOCK_SIZE)`：该行内列 offset；
3. 用 row stride 计算当前行起始指针；
4. `tl.load(..., other=-float("inf"))`：越界位置填 $-\infty$；
5. `x = x - tl.max(x, axis=0)`；
6. `numerator = tl.exp(x)`；
7. `denominator = tl.sum(numerator, axis=0)`；
8. `y = numerator / denominator`；
9. `tl.store(...)` 写回。

为什么 mask 的 `other` 是 $-\infty$？

因为：
$$
\exp(-\infty)=0
$$
被 mask 的位置不应该影响 softmax 的 sum，所以填 $-\infty$ 是符合数学语义的【Python代码†source】。

### 7.5 图表解释：`triton-softmax.png`

`triton-softmax.png` 展示了 softmax kernel 的映射方式：输入矩阵的一行对应一个 program instance，`pid` 就是 row index；该 program 加载整行，减去最大值，做 exp 和 sum，最后 normalize 并 store【图表†source】。

读图时要注意：

- 图中的一个 block 不是处理一个元素，而是处理一整行；
- `tl.max` 和 `tl.sum` 是 block 内 reduction；
- mask 的作用是处理 `BLOCK_SIZE` 大于真实列数的情况；
- 这个方案要求一整行能放进一个 block，这为下一节 row sum 的问题埋下伏笔。

### 7.6 本节学习提示

本节最重要的结论是：

> Softmax 的并行粒度不是单个元素，而是一行。只要一整行能放入一个 block，就可以在 block 内完成 max、exp、sum、normalize，减少 HBM 往返。

复习时抓住：

- softmax 需要行内通信；
- 不同行之间独立；
- 数值稳定要减去 row max；
- mask 位置填 $-\infty$；
- naive PyTorch softmax 慢在多个 kernel 反复读写 HBM。

------

## 8. 第七部分：Row Sum 与 Baby Tiling：当一行放不进一个 block

这一节要解决的问题是：**如果一行太长，不能一次放进一个 Triton block，应该怎么办？**

### 8.1 问题引入：softmax 的假设会失效

上一节 softmax 假设一整行能放入一个 block。假设一行有 4096 列，而 block size 只有 1024，就不能一次性加载完整行。

字幕在这里从 softmax 切到 row sum，因为 row sum 更简单，适合说明“行太长时如何分块累加”【字幕†source】。

Row sum 定义为：
$$
s_i = \sum_{j=0}^{N-1} X_{ij}
$$
如果 $N > B$，其中 $B$ 是 block size，就把一行拆成多个 tiles：
$$
s_i
=
\sum_{t=0}^{\lceil N/B\rceil-1}
\sum_{r=0}^{B-1}
\mathbf{1}_{tB+r<N}X_{i,tB+r}
$$
符号解释：

| 符号                  | 含义                         |
| --------------------- | ---------------------------- |
| $i$                   | 当前行编号                   |
| $N$                   | 每行列数                     |
| $B$                   | block size / tile width      |
| $t$                   | tile 编号                    |
| $r$                   | tile 内 offset               |
| $\mathbf{1}_{tB+r<N}$ | mask，避免最后一个 tile 越界 |

### 8.2 策略：一个 block 负责一行，block 内循环多个 tiles

课程策略是：

1. 一个 block 仍然负责一整行；
2. 该行被拆成多个 tiles；
3. 每个 thread / lane 维护一个 accumulator；
4. block 内循环处理 tile 0、tile 1、tile 2；
5. 最后对 accumulator 做 reduction，得到该行 sum。

字幕特别提醒：GeLU 中输入向量被切成多个 blocks，每个 block 独立处理一段；row sum 中这些小块不是 blocks，而是 tiles。整个 row 才对应一个 block，block 内部循环处理 tiles【字幕†source】。

### 8.3 Triton Row Sum 代码逻辑

`triton_row_sum_kernel` 的核心结构是【Python代码†source】：

1. `row = tl.program_id(0)`
    当前 block 负责哪一行。
2. `acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)`
    每个 offset 一个 accumulator。
3. `for start in range(0, N, BLOCK_SIZE)`
    沿行方向循环 tiles。
4. `cols = start + tl.arange(0, BLOCK_SIZE)`
    当前 tile 的列索引。
5. `mask = cols < N`
    处理最后一个 tile 越界问题。
6. `x = tl.load(..., mask=mask, other=0.0)`
    加载当前 tile；越界位置填 0。
7. `acc += x`
    累加到 accumulator。
8. `result = tl.sum(acc, axis=0)`
    对 accumulator 做最终 reduction。
9. `tl.store(y_ptr + row, result)`
    写出该行 sum。

这里 mask 的 `other=0.0` 与 softmax 的 `-inf` 不同。原因是 row sum 的加法单位元是 0，越界位置不应该改变求和结果。

### 8.4 图表解释：`triton-row-sum.png`

`triton-row-sum.png` 展示了行太长时的 tiled row sum：一个 block 仍负责一行，但该行被分成多个 tiles；每个 tile 中同一 offset 的值累加到 accumulator，最后通过 reduction 得到行和【图表†source】。

读图时要特别注意：

- **block 是调度单位**：一个 row 对应一个 Triton program；
- **tile 是数据片段**：block 内部循环处理多个 tiles；
- accumulator 可能在 registers 或 shared memory 中，具体由 compiler 决定；
- 最后一次 `tl.sum(acc)` 才把长度为 `BLOCK_SIZE` 的 accumulator 归约成一个标量。

### 8.5 工程意义：为 matmul tiling 做预热

Row sum 是一个 “baby tiling” 例子。它说明：

- 一个 block 不一定只处理一次固定大小的数据；
- 一个 block 可以在 kernel 内部 loop over tiles；
- accumulator 是连接多个 tiles 的关键；
- 当数据无法一次放入 fast memory，就必须分 tile 处理。

这与下一节 matmul 的思想完全一致，只是 matmul 的 tile 是二维的，且需要沿 $K$ 维循环累加。

### 8.6 本节学习提示

本节最重要的结论是：

> 当一行放不进一个 block 时，不是把一行拆给多个 block 直接求最终结果，而是可以让一个 block 负责整行，在 block 内循环处理多个 tiles，并用 accumulator 跨 tile 累加。

复习时抓住：

- block 和 tile 不是同一个概念；
- row sum 的 mask `other` 是 0；
- softmax 的 mask `other` 是 $-\infty$；
- accumulator 是跨 tile 保留中间结果的关键；
- 这节是 matmul tiling 的预备知识。

------

## 9. 第八部分：Matmul+ReLU：Tiling、Arithmetic Intensity 与 Fusion

这一节要解决的问题是：**矩阵乘法为什么必须做 tiling？tiling 如何提高 arithmetic intensity？为什么 matmul 后接 ReLU 很适合 fusion？**

### 9.1 Matmul 的定义

给定：
$$
A \in \mathbb{R}^{M\times K},
\quad
B \in \mathbb{R}^{K\times N}
$$
输出：
$$
C \in \mathbb{R}^{M\times N}
$$
其中：
$$
C_{mn} = \sum_{k=0}^{K-1} A_{mk}B_{kn}
$$
本讲还在 matmul 后接 ReLU：
$$
Y_{mn} = \max(C_{mn}, 0)
$$
字幕解释，线性层后接激活函数在神经网络中很常见，因此 matmul+activation fusion 是自然的工程优化【字幕†source】。

### 9.2 Naive Matmul：正确但 HBM 访问太多

最 naive 的思路是：每个输出元素 $C_{mn}$ 单独计算。对每个 $k$：

1. 从 HBM 读取 $A_{mk}$；
2. 从 HBM 读取 $B_{kn}$；
3. 乘加到 accumulator；
4. 最后写出 $C_{mn}$。

这样当然正确，但会造成大量重复读取。

例如计算 $C_{m,n}$ 和 $C_{m,n+1}$ 都需要同一行 $A_{m,:}$。如果每个输出元素独立从 HBM 读取 $A_{m,:}$，同一批数据会被重复读很多次【字幕†source】。

Naive matmul 的 HBM 读次数近似：
$$
O(MKN)
$$
写次数：
$$
O(MN)
$$
计算量也是：
$$
O(MKN)
$$
所以 arithmetic intensity 近似：
$$
\text{AI}
=
\frac{\text{FLOPs}}{\text{Bytes moved}}
\approx
\frac{O(MKN)}{O(MKN)}
=
O(1)
$$

#### Arithmetic intensity 是什么？

$$
\text{Arithmetic Intensity}
=
\frac{\text{FLOPs}}{\text{Bytes moved}}
$$

它衡量每搬运一个 byte 数据能做多少计算。

- AI 低：经常等内存，memory-bound；
- AI 高：数据复用好，更可能发挥计算单元能力。

### 9.3 Ideal Matmul：如果能把全部 A 和 B 放进 shared memory

理想情况下，如果整个 $A$ 和 $B$ 都能放进 shared memory，那么每个输入元素只需要从 HBM 读取一次：
$$
\text{Reads} = O(MK + KN)
$$
对于方阵 $N \times N$：

- 计算量是 $O(N^3)$；
- 数据读写是 $O(N^2)$；

因此：
$$
\text{AI} \approx O(N)
$$
这比 naive 的 $O(1)$ 好得多。

但现实问题是：矩阵通常太大，无法全部放入 shared memory【字幕†source】。

### 9.4 Tiling：全局像 naive，局部像 ideal

Tiling 的核心思想是：

> 不能把整个矩阵放入 fast memory，就把一个小 tile 放进去；在这个 tile 上尽可能多地复用数据，再进入下一个 tile。

课程图示中，输出矩阵 $C$ 被切成 tiles。每个 output tile 由一个 thread block 负责。对于某个输出 tile，block 沿 $K$ 维循环：

1. 加载 $A$ 的一个 row tile；
2. 加载 $B$ 的一个 column tile；
3. 对这两个 tile 做 `dot`；
4. 累加到 output tile accumulator；
5. 进入下一个 $K$ tile；
6. 最后写出 output tile。

字幕用一句话概括：全局看像 naive approach，但局部看像 idealized approach【字幕†source】。

tiling 后 arithmetic intensity 大约提升到：
$$
\text{AI} \approx O(T)
$$
其中 $T$ 是 tile size。虽然通常达不到 ideal 的 $O(N)$，但比 naive 的 $O(1)$ 已经大幅改善【字幕†source】。

### 9.5 Strides：二维索引如何映射到线性内存

Triton matmul kernel 中大量代码都在做 pointer arithmetic。原因是 tensor 在逻辑上是二维的，但物理内存是一维的。

给定二维索引 $(i,j)$，地址偏移为：
$$
\text{offset}(i,j)
=
i\cdot \text{stride}_0
+
j\cdot \text{stride}_1
$$
如果是 row-major contiguous matrix：
$$
\text{stride}_0 = N,
\quad
\text{stride}_1 = 1
$$
如果是 transpose view，strides 可能变化。字幕解释说，stride 告诉你如何从行列坐标映射到真实内存位置；`lecture_06.py` 中也专门演示了 `x.stride()` 和 `row * stride_row + col * stride_col`【字幕†source】【Python代码†source】。

### 9.6 Triton Matmul+ReLU kernel 结构

`lecture_06.py` 的 `triton_matmul_relu` 设置为【Python代码†source】：
$$
\text{BLOCK\_M}=64,
\quad
\text{BLOCK\_N}=64,
\quad
\text{BLOCK\_K}=32
$$
Grid 为：
$$
\left(
\left\lceil \frac{M}{64} \right\rceil,
\left\lceil \frac{N}{64} \right\rceil
\right)
$$
每个 program instance 负责一个 $64 \times 64$ 的 output tile。

kernel 内部主要步骤：

1. `pid_m = tl.program_id(0)`
    当前 output tile 的 row block index。
2. `pid_n = tl.program_id(1)`
    当前 output tile 的 column block index。
3. 构造 `indices_m`、`indices_n`、`indices_k`。
4. 用 strides 构造 $A$ tile 和 $B$ tile 的 pointer matrix。
5. 初始化 accumulator：

$$
\text{acc} \in \mathbb{R}^{64\times64}
$$

1. 沿 $K$ 维循环，每次加载：

$$
A_{\text{tile}} \in \mathbb{R}^{64\times32}
$$

和：
$$
B_{\text{tile}} \in \mathbb{R}^{32\times64}
$$

1. 做 tile dot：

$$
\text{acc}
\leftarrow
\text{acc}
+
A_{\text{tile}}B_{\text{tile}}
$$

代码中对应：

```
acc += tl.dot(a, b)
```

1. 循环结束后做 ReLU fusion：

$$
\text{acc} = \max(\text{acc}, 0)
$$

代码中对应：

```
tl.maximum(acc, 0.0)
```

1. 最后用 mask 写回 $C$，处理 $M,N$ 不是 block size 整数倍的情况【Python代码†source】。

### 9.7 为什么 matmul+ReLU 适合 fusion？

如果不 fusion，流程可能是：
$$
A @ B
\rightarrow
\text{write } C \text{ to HBM}
\rightarrow
\text{read } C \text{ from HBM}
\rightarrow
\operatorname{ReLU}(C)
\rightarrow
\text{write } Y \text{ to HBM}
$$
如果 fusion，则变成：
$$
A @ B \text{ accumulated in acc}
\rightarrow
\operatorname{ReLU}(\text{acc})
\rightarrow
\text{write final } Y \text{ once}
$$
既然 matmul kernel 已经得到 accumulator，在写回 HBM 之前顺手做 ReLU 非常便宜。这正是 kernel fusion 的典型场景【字幕†source】。

### 9.8 图表解释：`gemm_tiled.png`

`gemm_tiled.png` 展示了 tiled GEMM：矩阵 $A$、$B$、$C$ 被按 tile size $T$ 切分；每个 output tile 通过循环加载 $A$ 和 $B$ 的对应 tile，做局部 multiply-accumulate，然后写回 $C$【图表†source】。

读图时注意：

- 输出矩阵 $C$ 的一个 tile 对应一个 program / block；
- $K$ 维方向需要循环多个 tile；
- accumulator 在所有 $K$ tile 循环中不断累加；
- tile size 越大，数据复用机会越多，但 register/shared memory 压力也越大。

### 9.9 本节学习提示

本节最重要的结论是：

> Matmul tiling 通过让一个 block 负责一个 output tile，并沿 $K$ 维循环复用 A/B tiles，把 arithmetic intensity 从 naive 的 $O(1)$ 提升到 tile size 量级。

复习时抓住：

- naive matmul 慢在重复 HBM 读取；
- ideal matmul 不现实，因为 shared memory 放不下整个矩阵；
- tiling 是现实中的折中；
- `tl.dot` 是 tiled matmul 的核心；
- ReLU 可以在 accumulator 写回前 fusion；
- strides 是正确访问矩阵元素的基础。

------

## 10. Python 代码示例详解

这一节要解决的问题是：**`lecture_06.py` 的代码如何对应课程主线？每个函数服务于哪个学习目标？**

### 10.1 文件整体结构

`lecture_06.py` 是一份 executable lecture。原报告指出，`main()` 按课程顺序调用多个函数，基本对应课堂主线【Python代码†source】。

| 顺序 | 函数                                  | 对应课程部分      | 作用                                                         |
| ---- | ------------------------------------- | ----------------- | ------------------------------------------------------------ |
| 1    | `review_of_gpus()`                    | GPU review        | 回顾 GPU memory hierarchy、thread/block/grid、warps、occupancy、bank conflicts、coalescing |
| 2    | `benchmarking_and_profiling()`        | Benchmark/Profile | 展示如何正确计时、如何 profile PyTorch kernel                |
| 3    | `naive_vs_builtin_vs_compiled_gelu()` | GeLU fusion       | 比较 naive、builtin、compiled GeLU                           |
| 4    | `triton_introduction()`               | Triton intro      | 对比 CUDA 与 Triton 的 mental model                          |
| 5    | `triton_gelu_example()`               | Triton GeLU       | 编写第一个 element-wise Triton kernel                        |
| 6    | `triton_softmax_example()`            | Softmax           | 编写 row-wise reduction kernel                               |
| 7    | `triton_row_sum_example()`            | Row sum tiling    | 处理 row 不 fit in block 的情况                              |
| 8    | `triton_matmul_relu_example()`        | Matmul tiling     | 实现 tiled matmul + ReLU fusion                              |
| 9    | `main()` 尾部总结                     | 课程总结          | 总结编程模型、硬件约束、benchmark/profile、Triton、tiling    |

### 10.2 Benchmark 与 Profile 相关函数

| 函数 / 变量                      | 输入              | 输出           | 关键逻辑                                          | 理论意义                              |
| -------------------------------- | ----------------- | -------------- | ------------------------------------------------- | ------------------------------------- |
| `benchmark(run)`                 | 一个无参 callable | 平均 GPU 时间  | warmup、CUDA event、synchronize、多 trials        | 正确测量 GPU 异步执行时间             |
| `profile(run)`                   | 一个无参 callable | profiler table | warmup 后进入 PyTorch profiler，按 CUDA time 排序 | 看 PyTorch 表达式实际触发哪些 kernels |
| `run_operation1(dim, operation)` | 维度与单输入操作  | callable       | 创建一个随机 tensor，返回闭包                     | 统一 benchmark/profile 单输入操作     |
| `run_operation2(dim, operation)` | 维度与双输入操作  | callable       | 创建两个随机 tensor，返回闭包                     | 统一 benchmark/profile 二元操作       |

这些函数服务于课程哲学：

> 先 benchmark 看到端到端时间，再 profile 看到底层 kernel，再决定是否修改实现或写 Triton kernel。

### 10.3 GeLU 相关代码

| 函数                        | 功能                                           | 对应公式 / 概念                                  | 课堂作用                               |
| --------------------------- | ---------------------------------------------- | ------------------------------------------------ | -------------------------------------- |
| `naive_gelu(x)`             | 用 PyTorch primitive 直接写 tanh approximation | $\frac12x(1+\tanh(\sqrt{2/\pi}(x+0.044715x^3)))$ | 展示 naive 表达式会产生多个 kernel     |
| `builtin_gelu(x)`           | 调用 PyTorch 内置 fused GeLU                   | 同一 GeLU 数学函数                               | 展示标准库中已有优化 kernel            |
| `torch.compile(naive_gelu)` | 编译 naive 函数                                | 自动 kernel fusion                               | 展示编译器可以生成 fused Triton kernel |
| `triton_gelu_kernel`        | 手写 Triton GeLU                               | block-level element-wise kernel                  | 展示 Triton kernel 基本模板            |

这部分的重点不是 GeLU 本身，而是说明：

> 同一个数学函数，不同实现方式可能对应完全不同的 kernel 数量和 HBM 流量【字幕†source】。

### 10.4 Triton kernel 基本模板

从 GeLU、Softmax、Row Sum 到 Matmul，Triton kernel 基本都遵循：
$$
\text{identify block}
\rightarrow
\text{compute offsets/pointers}
\rightarrow
\text{load}
\rightarrow
\text{compute}
\rightarrow
\text{store}
$$

| 代码元素                  | 含义                      | 出现位置               | 作用                                    |
| ------------------------- | ------------------------- | ---------------------- | --------------------------------------- |
| `tl.program_id(axis=0/1)` | 当前 program / block 编号 | 所有 Triton kernels    | 把 grid 中的 block 映射到数据区域       |
| `tl.arange(...)`          | block 内 vector offsets   | 所有 Triton kernels    | 构造一组并行索引                        |
| `mask`                    | 越界保护                  | 非整除 shape 的 kernel | 避免非法读写                            |
| `tl.load`                 | 从 global memory 读取     | 所有 kernels           | 将 HBM 数据带入 kernel                  |
| `tl.store`                | 写回 global memory        | 所有 kernels           | 输出结果                                |
| `tl.max / tl.sum`         | block 内 reduction        | Softmax、Row Sum       | 聚合同一行的多个元素                    |
| `tl.dot`                  | tile matrix multiply      | Matmul                 | 使用 Triton/硬件支持做 tile dot product |

原报告也给出了类似的 Triton 模板表，强调这些元素贯穿 GeLU、Softmax、Row Sum 和 Matmul【Python代码†source】。

### 10.5 关键代码片段解析

#### 10.5.1 `benchmark(run)`

学习重点：

- `run()` 是待测操作；
- warmup 排除首次运行开销；
- CUDA events 记录 GPU 时间；
- synchronize 确保 GPU 完成；
- 多次 trials 求平均。

理论对应：
$$
\bar{t} = \frac{1}{T}\sum_i t_i
$$
这段代码帮助学生理解：GPU 计时不是普通 Python 计时，必须考虑异步执行。

#### 10.5.2 `triton_gelu_kernel`

学习重点：

- `pid` 决定当前 block；
- `offsets` 决定当前 block 处理哪些元素；
- `mask` 处理最后一个不完整 block；
- `tl.load` 和 `tl.store` 显式表达 HBM 读写；
- GeLU 公式在 block 内完成，从而避免多个 PyTorch kernels。

#### 10.5.3 `triton_softmax_kernel`

学习重点：

- `row_idx = tl.program_id(0)` 表示一行一个 block；
- `tl.max` 和 `tl.sum` 是 block 内 reduction；
- `other=-inf` 是为了不让 mask 位置影响 softmax；
- 该 kernel 展示了从 element-wise 进入 row-wise reduction 的关键转变。

#### 10.5.4 `row_sum_kernel`

学习重点：

- 一个 block 仍负责一行；
- `for start in range(0, N, BLOCK_SIZE)` 表示 block 内循环 tiles；
- `acc` 跨 tiles 累加；
- `other=0.0` 符合加法单位元；
- 这是 matmul tiling 的简化版。

#### 10.5.5 `matmul_relu_kernel`

学习重点：

- `pid_m`、`pid_n` 确定 output tile；
- `indices_m`、`indices_n`、`indices_k` 构造 tile 索引；
- strides 把二维索引映射到真实内存；
- `tl.dot(a,b)` 做 tile matrix multiply；
- `acc` 跨 $K$ tiles 累加；
- `tl.maximum(acc, 0.0)` 在写回前 fusion ReLU。

### 10.6 代码与课程知识点对应表

| 课程知识点           | 对应代码函数 / 变量 / 片段                    | 代码功能                                  | 对应理论原理                     | 在课程中的作用                           |
| -------------------- | --------------------------------------------- | ----------------------------------------- | -------------------------------- | ---------------------------------------- |
| GPU memory hierarchy | `review_of_gpus()`；硬件表                    | 对比 registers、L1/shared、L2、HBM        | 快小 vs 慢大                     | 建立后续所有 kernel 优化背景             |
| Warp 与 divergence   | `review_of_gpus()`                            | 说明 32 threads/warp 与 lockstep 执行     | SIMT 执行                        | 解释分支为何损害性能                     |
| Occupancy            | `num_registers_per_thread` 等变量             | 计算 register 限制下的 resident warps     | 资源约束决定并发度               | 说明 register 使用与隐藏延迟的 trade-off |
| Benchmarking         | `benchmark(run)`                              | 正确测量 GPU 时间                         | warmup、CUDA events、sync        | 防止异步执行和首次编译污染结果           |
| Profiling            | `profile(run)`                                | 查看底层 CUDA kernel                      | CUDA activity profiling          | 发现 PyTorch 背后的真实执行路径          |
| Kernel fusion        | `naive_gelu`、`builtin_gelu`、`torch.compile` | 比较多 kernel vs 单 kernel                | 减少 HBM read/write              | 引入 Triton 和 fused kernel 动机         |
| Triton 基本模板      | `triton_gelu_kernel`                          | `pid`、offset、mask、load、compute、store | block-level programming          | 第一个完整 Triton kernel                 |
| Row-wise reduction   | `triton_softmax_kernel`                       | 一行一个 block，做 max/sum                | softmax reduction                | 展示 block 内 reduction                  |
| Baby tiling          | `row_sum_kernel`                              | 一行多个 tiles，accumulate 后 reduce      | 数据不 fit in block 时循环 tiles | 为 matmul tiling 预热                    |
| Tiled matmul         | `matmul_relu_kernel`                          | $64\times64$ output tile，沿 $K$ 循环     | 提高 arithmetic intensity        | 本讲 kernel 优化高潮                     |
| Activation fusion    | `tl.maximum(acc, 0.0)`                        | matmul 后直接 ReLU                        | 写回 HBM 前融合 element-wise op  | 展示工程中常见 fusion 模式               |

------

## 11. PDF / 幻灯片 / 图表解析

这一节要解决的问题是：**原报告中提到的图表分别帮助理解什么？读图时应该看哪些信息？**

当前原报告说明没有找到 Lecture 6 对应 PDF，因此本节解析 `lecture_06.py` 使用的图表资料，而不是 PDF【资料核对†source】。

### 11.1 主要图表总览

| 图表                        | 展示内容                                           | 支持的课程结论                                      | 对应代码或字幕内容                       |
| --------------------------- | -------------------------------------------------- | --------------------------------------------------- | ---------------------------------------- |
| `images/gpu-hardware.png`   | SM、register、L1/shared memory、L2、HBM 的层次结构 | HBM 大但远，SM 内存小但快；优化目标是减少 HBM 访问  | `review_of_gpus()`；GPU memory hierarchy |
| `images/triton-softmax.png` | 一行一个 Triton program / block 的 softmax kernel  | softmax 是 row-wise reduction，而不是 element-wise  | `triton_softmax_kernel`                  |
| `images/triton-row-sum.png` | 一行太长时 block 内循环 tiles 的 row sum           | block 和 tile 不同；block 可循环处理多个 tiles      | `row_sum_kernel`                         |
| `images/gemm_tiled.png`     | tiled GEMM 的 output tile、A/B tile 和累加过程     | tiling 提高 arithmetic intensity，减少重复 HBM 读取 | `matmul_relu_kernel`                     |

### 11.2 GPU hardware 图

这张图展示：

- GPU 芯片上有多个 SM；
- 每个 SM 内有 registers 与 L1/shared memory；
- GPU 芯片级有 L2 cache；
- 芯片外连接 HBM。

图表支持的结论是：

> 高性能 kernel 的目标是把数据尽量搬到 SM 附近，并在写回 HBM 前尽可能多地复用。

读图时最容易误解的是：HBM 带宽高不代表访问代价可以忽略。相对于 register/shared memory，HBM 仍然远得多、慢得多。

### 11.3 Triton Softmax 图

这张图展示：

- 输入矩阵的一行对应一个 program instance；
- `pid` 对应 row index；
- block 加载一行；
- 做 row max、exp、row sum；
- normalize 后 store 回输出。

图表支持的结论是：

> softmax 的自然并行粒度是一行，因为同一行内部需要 reduction，不同行之间相互独立。

读图时注意：这里的 block 不是处理一个元素，而是处理一整行。

### 11.4 Triton Row Sum 图

这张图展示：

- 一个 row 对应一个 block；
- row 被拆成多个 tiles；
- 每个 tile 的对应 offset 被累加到 accumulator；
- 最后通过 `tl.sum(acc)` 得到行和。

图表支持的结论是：

> 当一行不能一次放进 block 时，可以在 block 内 loop over tiles。

读图时最容易混淆的是 block 和 tile：

- block 是 GPU 调度 / Triton program instance；
- tile 是 block 内部处理的数据片段。

### 11.5 Tiled GEMM 图

这张图展示：

- 输出矩阵 $C$ 被切成 tiles；
- 一个 output tile 由一个 block 负责；
- block 沿 $K$ 维加载 $A$ 和 $B$ 的对应 tiles；
- 对 tile 做 multiply-accumulate；
- 最后写回 output tile。

图表支持的结论是：

> tiling 让 $A$ 和 $B$ 的数据在 fast memory 中被复用，从而提高 arithmetic intensity。

读图时注意：tile size 不是越大越好。更大的 tile 增加复用，但也增加 register/shared memory 压力，可能降低 occupancy。

------

## 12. 关键公式与推导汇总

这一节集中整理本讲出现的关键公式，并说明它们与代码和工程直觉的关系。

### 12.1 公式总览表

| 公式                                                         | 符号解释                     | 适用场景             | 直觉解释                   | 代码 / 例子对应                    |
| ------------------------------------------------------------ | ---------------------------- | -------------------- | -------------------------- | ---------------------------------- |
| $\operatorname{GELU}(x)\approx \frac12x(1+\tanh(\sqrt{2/\pi}(x+0.044715x^3)))$ | $x$ 为输入元素               | GeLU 激活函数        | 平滑版非线性激活           | `naive_gelu`、`triton_gelu_kernel` |
| $\tanh(a)=\frac{e^{2a}-1}{e^{2a}+1}$                         | $a$ 为 tanh 输入             | Triton 中实现 GeLU   | 用 exp 实现 tanh           | `tl.exp(2*a)`                      |
| $m_i=\max_j X_{ij}$                                          | 第 $i$ 行最大值              | 数值稳定 softmax     | 减去最大值避免 overflow    | `tl.max(x_row)`                    |
| $\operatorname{softmax}(X_{i,:})*j=\frac{\exp(X*{ij}-m_i)}{\sum_k\exp(X_{ik}-m_i)}$ | 行内归一化                   | attention / 概率输出 | 一行 logits 转概率         | `triton_softmax_kernel`            |
| $\text{Reads}=5MN+M$                                         | $M$ 行、$N$ 列               | naive softmax        | 多 kernel 导致反复 HBM 读  | `naive_softmax`                    |
| $s_i=\sum_{t}\sum_r \mathbf{1}*{tB+r<N}X*{i,tB+r}$           | $t$ tile，$r$ tile 内 offset | tiled row sum        | 一行拆成多个 tiles 累加    | `row_sum_kernel`                   |
| $C_{mn}=\sum_k A_{mk}B_{kn}$                                 | matmul 基本定义              | GEMM                 | 行列点积                   | `tl.dot(a,b)`                      |
| $\text{AI}=\frac{\text{FLOPs}}{\text{Bytes moved}}$          | 算术强度                     | 性能分析             | 每搬一次数据做多少计算     | matmul tiling                      |
| $\text{offset}(i,j)=i\cdot\text{stride}_0+j\cdot\text{stride}_1$ | stride 映射                  | pointer arithmetic   | 二维索引到一维内存         | matmul pointer matrix              |
| $\text{occupancy}=\frac{W_{\text{resident}}}{W_{\max}}$      | resident warps / max warps   | GPU 并发分析         | 衡量 SM 上可切换的 warp 数 | occupancy 例子                     |

### 12.2 GeLU 公式展开

$$
\operatorname{GELU}(x)
\approx
\frac{1}{2}x
\left(
1+
\tanh\left(
\sqrt{\frac{2}{\pi}}(x+0.044715x^3)
\right)
\right)
$$

直觉解释：

- ReLU 是硬截断：负数变 0，正数保留；
- GeLU 是更平滑的激活函数；
- 本讲重点不是 GeLU 的概率解释，而是它由多个 element-wise primitive 组成，很适合展示 kernel fusion。

代码对应：

- `naive_gelu` 逐项写出公式；
- `builtin_gelu` 调用优化实现；
- `triton_gelu_kernel` 在一个 Triton block 中完成所有 scalar operations。

### 12.3 Softmax 数值稳定公式展开

原始 softmax：
$$
\frac{\exp(X_{ij})}{\sum_k \exp(X_{ik})}
$$
数值稳定 softmax：
$$
\frac{\exp(X_{ij}-m_i)}{\sum_k \exp(X_{ik}-m_i)}
$$
其中：
$$
m_i = \max_j X_{ij}
$$
为什么不改变结果？
$$
\frac{\exp(X_{ij}-m_i)}
{\sum_k\exp(X_{ik}-m_i)}
=
\frac{\exp(X_{ij})\exp(-m_i)}
{\sum_k\exp(X_{ik})\exp(-m_i)}
=
\frac{\exp(X_{ij})}
{\sum_k\exp(X_{ik})}
$$
代码对应：

- `tl.max` 计算 $m_i$；
- `x_row = x_row - tl.max(x_row, axis=0)`；
- `tl.exp`；
- `tl.sum`；
- divide；
- store。

### 12.4 Row Sum tiled 公式展开

$$
s_i
=
\sum_{t=0}^{\lceil N/B\rceil-1}
\sum_{r=0}^{B-1}
\mathbf{1}_{tB+r<N}X_{i,tB+r}
$$

直觉解释：

- $N$ 太大，不能一次处理；
- 每次处理长度为 $B$ 的 tile；
- mask 确保最后一个 tile 不越界；
- accumulator 保留跨 tile 的部分和。

代码对应：

```
for start in range(0, N, BLOCK_SIZE):
    cols = start + tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(..., mask=mask, other=0.0)
    acc += x
result = tl.sum(acc, axis=0)
```

### 12.5 Matmul arithmetic intensity 推导

Naive matmul：
$$
\text{FLOPs} = O(MKN)
$$
HBM 读取也近似：
$$
\text{Bytes moved} = O(MKN)
$$
所以：
$$
\text{AI} \approx O(1)
$$
Ideal matmul 若能把 $A$、$B$ 全部放入 shared memory：
$$
\text{Reads} = O(MK + KN)
$$
对于方阵：
$$
\text{AI} \approx \frac{O(N^3)}{O(N^2)} = O(N)
$$
Tiled matmul：
$$
\text{AI} \approx O(T)
$$
其中 $T$ 是 tile size【字幕†source】。

**学习提示**：tiling 的目标就是在现实硬件容量限制下，尽可能接近 ideal matmul 的数据复用效果。

------

## 13. 概念对比与易混淆点

这一节用于复习时排雷。

| 易混淆概念                              | 区别                                                   | 联系                                  | 常见误解                       | 正确理解                                      |
| --------------------------------------- | ------------------------------------------------------ | ------------------------------------- | ------------------------------ | --------------------------------------------- |
| Thread vs Thread block                  | thread 是最小执行单元；block 是一组 threads            | block 内 threads 可共享 shared memory | 只要 thread 多就快             | block 设计决定数据共享和 tiling               |
| Block vs Tile                           | block 是调度单位；tile 是数据片段                      | 一个 block 可处理一个或多个 tiles     | row sum 中每个 tile 都是 block | row sum 中一个 row 是 block，row 内多个 tiles |
| Warp occupancy vs Block occupancy       | 前者看 resident warps；后者看 blocks 是否填满 SM waves | 都影响硬件利用率                      | occupancy 越高越好             | occupancy 是指标，不是最终目标                |
| Bank conflict vs Memory coalescing      | 前者发生在 shared memory；后者发生在 HBM               | 都与访问模式有关                      | 都叫内存访问冲突，所以一样     | 优化位置和方法不同                            |
| Benchmarking vs Profiling               | benchmark 测总时间；profiling 看时间分布               | 优化时二者都需要                      | benchmark 快就知道原因         | 只有 profile 才知道底层 kernel                |
| PyTorch 表达式 vs CUDA kernel           | Python 层表达式可能对应多个 kernel                     | PyTorch 最终调用底层 kernel           | 一行 PyTorch 就是一个 kernel   | 必须 profile 才知道                           |
| `torch.compile` vs builtin kernel       | compile 自动优化；builtin 可能是手写高度优化           | 都可能 fusion                         | compile 一定最快               | 必须 benchmark/profile                        |
| Element-wise vs Reduction               | element-wise 各元素独立；reduction 需要聚合            | reduction 常在 block 内完成           | softmax 是 element-wise        | softmax 需要行内 max/sum                      |
| Softmax mask `-inf` vs Row sum mask `0` | softmax 用 $-\infty$，row sum 用 0                     | 都是 mask 的 `other`                  | mask 值随便填                  | `other` 必须符合数学单位元                    |
| Triton 抽象 vs PTX 执行                 | Triton 是高级写法；PTX 是底层指令                      | Triton 编译到 PTX                     | 写 Triton 就不用懂硬件         | 性能仍取决于硬件细节                          |

### 13.1 Block 与 tile

原报告特别强调：GeLU 中输入向量被切成多个 blocks；Row Sum 中一个 row 对应一个 block，但 row 内部被切成多个 tiles【字幕†source】。

一句话区分：

> block 是“谁来执行”，tile 是“执行时处理哪一块数据”。

### 13.2 Occupancy 不是最终目标

Occupancy 高有助于隐藏延迟，但如果为了提高 occupancy 而让每个 thread 工作太少、数据复用太差，也可能变慢。thread coarsening 可能降低 occupancy，却提高整体效率【字幕†source】。

### 13.3 Mask 的 `other` 要符合操作语义

| 操作    | mask 的 `other` | 原因                          |
| ------- | --------------- | ----------------------------- |
| Softmax | $-\infty$       | $\exp(-\infty)=0$，不影响 sum |
| Row sum | $0$             | 0 是加法单位元                |
| Matmul  | $0$             | 0 不影响 dot product 累加     |

### 13.4 XLA 不应被强行加入本讲

原报告说明，虽然标题中提到 XLA，但当前字幕和 `lecture_06.py` 中没有实际 XLA 讲解，因此不将其作为课程知识点展开【资料核对†source】。

这体现了一个重要原则：**资料中没有明确讲的内容，不应编造成课程结论。**

------

## 14. 学习拓展与发散理解

本节是帮助理解的拓展，不代表课程原文。

### 14.1 与大模型训练实践的关系

**工程直觉：大模型训练不是只拼 FLOPs，还要拼数据移动效率。**

本讲内容直接影响以下实践问题：

| 实践环节    | 本讲相关内容                | 实际意义                                               |
| ----------- | --------------------------- | ------------------------------------------------------ |
| MLP 层训练  | matmul + GeLU               | matmul tiling 和 activation fusion 影响吞吐            |
| Attention   | softmax、row-wise reduction | softmax 的内存访问和 reduction 方式影响 attention 性能 |
| Kernel 优化 | Triton、fusion、tiling      | 减少 HBM 往返，提高 arithmetic intensity               |
| 性能诊断    | benchmark/profile           | 定位是 kernel launch、HBM 读写还是访问模式瓶颈         |
| 编译优化    | `torch.compile`             | 自动 fusion 可能显著加速，但需要验证                   |
| 推理部署    | fused kernels               | 降低 latency，减少中间 tensor materialization          |
| 训练成本    | arithmetic intensity        | 更好利用 GPU，减少浪费，降低训练时间和成本             |

### 14.2 与其他 lecture 的联系

**补充说明**：在 CS336 的整体主线中，本讲是从模型计算走向训练系统的关键桥梁。

可能的联系包括：

- 前面 lecture 中已经介绍语言模型、Transformer 或训练目标时，很多操作在数学上是 matmul、softmax、activation；
- 本讲解释这些操作在 GPU 上如何真正执行；
- 后续涉及 FlashAttention、多 GPU、分布式训练或系统优化时，仍会反复用到 memory hierarchy、tiling、fusion 和 profiling 的思想。

谨慎地说，本讲提供的是后续训练系统优化的基础语言：
 **block、tile、HBM、shared memory、arithmetic intensity、fusion、profile**。

### 14.3 面试 / 考试 / 作业可能考察角度

| 问题                                              | 考察点                         | 回答思路                                                     |
| ------------------------------------------------- | ------------------------------ | ------------------------------------------------------------ |
| 为什么 naive GeLU 比 builtin / compiled GeLU 慢？ | kernel fusion、HBM 读写        | naive 拆成多个 kernels，中间结果反复写回 HBM；builtin/compiled fusion 成单 kernel |
| 为什么 softmax 不是 element-wise？                | row-wise reduction             | 每个输出依赖整行 max 和 sum                                  |
| 为什么 softmax 可以一行一个 block？               | block 内 reduction，不同行独立 | 行内需要通信，行间不需要通信                                 |
| row 长度超过 block size 怎么办？                  | baby tiling                    | 一个 block 负责一行，block 内循环多个 tiles，用 accumulator 累加 |
| matmul tiling 如何提高 AI？                       | arithmetic intensity           | tile 中数据被复用，减少重复 HBM 读取                         |
| 为什么 benchmark GPU 代码要 synchronize？         | GPU 异步执行                   | CPU 提交任务不等于 GPU 完成任务                              |
| profiling 能看到什么？                            | 底层 kernel                    | 能看到 PyTorch 表达式实际调用哪些 CUDA kernels               |
| control divergence 为什么慢？                     | warp lockstep                  | 同一 warp 不同分支需要串行执行                               |
| bank conflict 和 memory coalescing 区别？         | 内存层次                       | bank conflict 在 shared memory；coalescing 在 HBM/global memory |
| occupancy 越高越好吗？                            | 性能 trade-off                 | 不一定；低 occupancy 可能换来更多 per-thread work 或数据复用 |
| Triton 和 CUDA 最大区别是什么？                   | 编程抽象                       | CUDA 写 thread，Triton 写 block/program                      |
| 为什么 matmul+ReLU 可以 fusion？                  | 写回前 element-wise op         | accumulator 写回 HBM 前直接 ReLU，避免额外读写               |

------

## 15. 课程复习总结

### 15.1 本讲核心思想

Lecture 6 的主线是：**把语言模型中的张量计算放到真实 GPU 硬件约束中理解**。课程从 GPU memory hierarchy、thread/block/grid 编程模型出发，解释 warp、occupancy、bank conflict、memory coalescing 和 block occupancy 如何影响性能；随后强调 benchmark/profile 是优化前必须做的测量；接着用 GeLU 展示 kernel fusion 如何减少 HBM 读写；最后通过 Triton 逐步实现 element-wise GeLU、row-wise softmax、row-too-long 的 tiled row sum，以及 tiled matmul+ReLU【字幕†source】【Python代码†source】。

### 15.2 必须掌握的概念

- SM、register、L1/shared memory、L2、HBM；
- thread、thread block / CTA、grid、warp；
- control divergence、warp scheduler、occupancy；
- bank conflict、memory coalescing、block occupancy；
- benchmarking、profiling、CUDA events、synchronize；
- kernel fusion、`torch.compile`、PTX；
- Triton 的 `tl.program_id`、`tl.arange`、`tl.load`、`tl.store`、mask；
- row-wise reduction、tiling、arithmetic intensity；
- matrix strides、tiled matmul、`tl.dot`。

### 15.3 必须理解的公式

1. **GeLU tanh approximation**
    解决激活函数实现问题，也用于展示 element-wise fusion。
2. **数值稳定 softmax**
    解决 exponent overflow 问题，并展示 row-wise reduction。
3. **Naive softmax HBM 访问量**
    说明多个 PyTorch primitive 为什么会造成大量内存往返。
4. **Row sum tiled 公式**
    说明一行放不进 block 时如何拆 tiles。
5. **Matmul 公式**
    是深度学习最核心计算之一。
6. **Arithmetic intensity**
    衡量数据复用效率，是理解 tiling 的关键。
7. **Stride 地址公式**
    是写 Triton matmul pointer arithmetic 的基础。
8. **Occupancy 计算**
    说明 registers 等资源如何限制并发。

### 15.4 必须看懂的代码

| 代码                                            | 学习重点                                            |
| ----------------------------------------------- | --------------------------------------------------- |
| `benchmark(run)`                                | warmup、CUDA events、synchronize、多次 trials       |
| `profile(run)`                                  | 看底层 CUDA kernels 和耗时分布                      |
| `naive_gelu` / `builtin_gelu` / `compiled_gelu` | 理解同一数学函数不同 kernel 行为                    |
| `triton_gelu_kernel`                            | 掌握 Triton 基本模板                                |
| `triton_softmax_kernel`                         | 理解 row-wise reduction 和 mask `-inf`              |
| `row_sum_kernel`                                | 理解 block 内循环 tiles 和 accumulator              |
| `matmul_relu_kernel`                            | 理解 tiled matmul、strides、`tl.dot` 和 ReLU fusion |

### 15.5 复习时的一条主线

可以用下面这条线串起整讲：
$$
\text{GPU memory hierarchy}
\rightarrow
\text{benchmark/profile}
\rightarrow
\text{发现 HBM / kernel bottleneck}
\rightarrow
\text{kernel fusion}
\rightarrow
\text{Triton block-level programming}
\rightarrow
\text{reduction}
\rightarrow
\text{tiling}
\rightarrow
\text{higher arithmetic intensity}
$$

### 15.6 一句话总结

> **高性能语言模型系统不是只写出数学公式，而是要把计算切成适合 GPU 的 blocks 和 tiles，用 benchmark/profile 找瓶颈，用 fusion 和 tiling 减少 HBM 往返，并在硬件约束下提高 arithmetic intensity。**