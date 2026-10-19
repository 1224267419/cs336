# Lecture 7 课程报告：从通信原语到多 GPU 并行训练的系统化理解

## 0. 资料来源与改写说明

本报告是在用户上传的原始报告 `lec7.md` 基础上进行的系统性改写，目标是将原报告优化为更适合学习、复习和长期保存的中文讲义。原报告主题为 **Lecture 7：多 GPU 并行训练、通信原语与分布式 MLP 实现**，内容覆盖多 GPU 通信基础、硬件互连、PyTorch Distributed、通信 benchmark，以及数据并行、张量并行、流水线并行三类训练策略。 

本次改写参考了原报告中已有的字幕引用、Python 代码引用、运行输出引用和图表引用。需要特别说明的是：本次对话中没有单独提供原始字幕文件，因此本文中的 `【字幕†source】` 仍沿用原报告已经标注的字幕来源；本文没有把未提供的字幕内容重新当作可逐句核验的原始材料。原报告中已给出的 `【Python代码†source】`、`【运行输出†source】`、`【图表/Python代码†source】` 也会继续保留，并在需要时进行更清楚的解释。

我还核对了 Stanford CS336 lectures 仓库。仓库页面显示这是 Spring 2026 CS336 的 lecture materials，并且 README 说明可执行讲义命名为 `lecture_XX.py`，非可执行讲义命名为 `lecture_XX.pdf`；文件列表中有 `lecture_07.py`，相邻 lecture 中则有若干 PDF 文件，例如 `lecture_08.pdf` 等。 进一步核对 `lecture_07.py` 后可以看到，本讲代码结构确实按“通信原语 → 硬件/NCCL/PyTorch → benchmark → 数据并行/张量并行/流水线并行”的顺序展开，并且 `main()` 中明确将本讲主题写成跨多个 GPU 的 parallelism。 运行输出文件也验证了 all-reduce、reduce-scatter、all-gather、data parallelism、tensor parallelism、pipeline parallelism 示例的实际输出。

本文中的内容来源边界如下：

| 内容类型                             | 处理方式                                                     |
| ------------------------------------ | ------------------------------------------------------------ |
| 原报告已有主线、概念、公式、代码说明 | 保留并重新组织为学习讲义                                     |
| 原报告已有引用                       | 尽量保留原引用格式，如 `【字幕†source】`、`【Python代码†source】` |
| GitHub `lecture_07.py` 中的代码结构  | 用于核对和补充代码解释                                       |
| 运行输出                             | 用于解释通信操作和并行训练示例的实际行为                     |
| 课程外补充解释                       | 明确标注为“补充说明”“延伸理解”“工程直觉”或“学习提示”         |
| PDF / 幻灯片                         | 未发现 `lecture_07.pdf`；本讲图表主要来自 `lecture_07.py` 中的 `image(...)` 调用 |

------

## 1. 学习导读

这一节先回答三个问题：**本讲为什么重要？它在 CS336 中处于什么位置？学习时应该抓住哪条主线？**

### 1.1 本讲要解决的核心问题

Lecture 7 要解决的核心问题是：

> 当单张 GPU 的显存或算力不足以高效训练大语言模型时，如何把训练任务分布到多张 GPU 上，并让这些 GPU 在通信代价可控的前提下协同工作？

这个问题看起来是“多用几张 GPU”，但真正的难点不是启动更多 GPU，而是：数据、参数、激活、梯度、优化器状态到底应该**复制**、**分片**，还是**重计算**？不同 GPU 之间需要传输哪些张量？什么时候传输？通过什么通信原语传输？这些传输是否会抵消多 GPU 带来的计算加速？

字幕中把本讲从上一讲自然引出：上一讲关注单 GPU 内部如何通过 kernel、fusion、tiling 等手段减少数据搬运；本讲把这个思想扩展到多 GPU，核心仍然是避免数据传输瓶颈，只是“数据移动”的距离从 GPU 内部存储层级扩大到了 GPU 与 GPU、节点与节点之间【字幕†source】。

### 1.2 本讲在 CS336 课程中的位置

Lecture 7 是课程从“单卡优化”进入“分布式训练系统”的转折点。

在单 GPU 上，优化重点通常是：

- 如何提高 Tensor Core 利用率；
- 如何减少 HBM 访问；
- 如何通过 kernel fusion、tiling、shared memory 等手段减少内存瓶颈。

进入多 GPU 后，问题变为：

- 每张 GPU 存哪些模型状态？
- 不同 GPU 之间什么时候同步？
- 哪些通信可以用 all-reduce，哪些需要 all-gather 或 reduce-scatter？
- 数据并行、张量并行、流水线并行分别适合什么硬件拓扑？
- 通信带宽和延迟如何影响大模型训练效率？

`lecture_07.py` 中也明确把本讲主题概括为跨多个 GPU 的 parallelism，并指出统一主题是组织计算以避免数据传输瓶颈【Python代码†source】。

### 1.3 本讲知识结构

| 模块                  | 解决的问题                                   | 核心概念                                                     | 与后续内容的关系                        |
| --------------------- | -------------------------------------------- | ------------------------------------------------------------ | --------------------------------------- |
| 多 GPU 层次化通信视角 | 为什么多 GPU 训练不是简单加卡                | HBM、NVLink、NVSwitch、InfiniBand、Ethernet                  | 为理解通信瓶颈打基础                    |
| Collective operations | 多 GPU 之间如何表达通信模式                  | broadcast、scatter、gather、reduce、all-gather、reduce-scatter、all-reduce、all-to-all | 是 DDP、FSDP/ZeRO、张量并行、MoE 的基础 |
| 硬件互连与 NCCL       | 为什么同样的通信操作在不同机器上速度差异很大 | PCIe、NVLink、RDMA、NCCL、PyTorch Distributed                | 决定并行策略是否可行                    |
| 通信 benchmark        | 如何量化通信性能                             | 有效带宽、同步、barrier、CUDA synchronize                    | 判断通信是否成为训练瓶颈                |
| 数据并行              | 如何按 batch 切分训练                        | DDP、梯度 all-reduce                                         | 最基础、最模块化的并行训练方式          |
| 张量并行              | 如何按层内矩阵维度切分模型                   | column tensor parallel、activation all-gather                | 适合高速互连域内的模型分片              |
| 流水线并行            | 如何按网络深度切分模型                       | pipeline stage、micro-batch、pipeline bubble                 | 适合更大模型和更慢互连，但要处理气泡    |

### 1.4 学习本讲前需要知道什么

学习本讲不要求掌握完整的分布式系统细节，但最好具备以下前置知识：

| 前置知识                  | 为什么需要                                                   |
| ------------------------- | ------------------------------------------------------------ |
| 张量形状与矩阵乘法        | 理解 MLP、张量并行中的列切分、activation 拼接                |
| 反向传播与梯度            | 理解 DDP 为什么同步梯度而不是同步 loss                       |
| 优化器基本过程            | 理解参数、梯度、优化器状态为什么占显存                       |
| GPU 存储层次              | 理解 HBM、cache、shared memory 与跨 GPU 通信的速度差异       |
| PyTorch 基础              | 理解 `loss.backward()`、`optimizer.step()`、`torch.distributed` 调用 |
| Transformer 中 MLP 的作用 | 本讲用深层 MLP 作为训练示例，因为课程指出 MLP 是 Transformer 中重要的计算瓶颈【Python代码†source】 |

------

## 2. 第一部分：从单 GPU 到多 GPU——通信视角与 collective operations

本节要解决的问题是：**多 GPU 程序中，不同设备之间到底在“说什么语言”？为什么需要 broadcast、all-reduce、all-gather 这些通信原语？**

### 2.1 问题引入：多 GPU 训练中的“距离”变远了

在单 GPU 内部，计算单元从寄存器、shared memory、L2 cache、HBM 中读取数据。上一讲已经说明，即使 HBM 很快，相对于计算单元来说仍然可能成为瓶颈。

到了多 GPU 训练，层次进一步扩展：

1. 单 GPU 内部：L1 cache / shared memory，最快；
2. 单 GPU 内部：HBM；
3. 单节点多 GPU：NVLink / NVSwitch；
4. 多节点多 GPU：InfiniBand / Ethernet，最慢。

Python 讲义明确列出了这个 generalized hierarchy【Python代码†source】。从学习角度看，这个层次告诉我们：**多 GPU 并行不是只考虑 FLOPs，还必须考虑数据在多远的地方。**

### 2.2 概念定义：rank 与 world size

在分布式训练中，课程先引入两个基本词：

- **rank**：参与分布式程序的一个进程或设备编号。本讲中可以近似理解为一张 GPU。
- **world size**：总共有多少个 rank，也就是一共有多少个参与者。

例如 world size 为 4 时，rank 可以是 0、1、2、3。后续所有通信操作都以 rank 为单位描述：谁发送、谁接收、谁保存结果、谁参与归约。

**学习提示**：
 不要把 rank 机械理解成“物理 GPU 编号”。在本讲的简化语境中 rank 基本等同于 GPU，但更一般地说，rank 是分布式程序里的进程编号。一个节点可以有多个 rank，一个 rank 通常绑定一张 GPU。

### 2.3 方法与原理：为什么需要 collective operations

如果手写多 GPU 通信，一种朴素做法是让 GPU 0 给 GPU 1 发送某个张量，再让 GPU 1 给 GPU 2 发送另一个张量。这样会很快变得混乱。

Collective operations 的思想是：不直接描述底层点对点细节，而是描述一种**整体通信模式**。例如：

- “把 rank 0 的数据复制给所有 rank”就是 broadcast；
- “把所有 rank 的梯度求和，并让每个 rank 都得到结果”就是 all-reduce；
- “先把所有 rank 的结果归约，再把不同片段分发给不同 rank”就是 reduce-scatter。

这样做有两个好处：

1. 程序员表达的是高层通信语义；
2. NCCL 等底层库可以根据硬件拓扑选择更优的实际通信路径。

课程强调，与其手动管理点对点通信，不如使用 collective operation 表达常见模式，让系统和通信库完成优化【字幕†source】。

### 2.4 核心通信原语详解

#### 2.4.1 Broadcast：一份数据复制给所有 rank

**直觉解释**：
 broadcast 就像老师把一份讲义发给全班。源 rank 持有原始数据，所有其他 rank 接收同样的数据。

**正式定义**：
 如果源 rank 为 $s$，源数据为 $x_s$，则：
$$
y_r = x_s,\quad \forall r \in \{0,1,\ldots,p-1\}
$$
其中 $p$ 是 world size。

**课程例子**：
 Python 讲义用 rank 0 上的 `tensor([0., 1, 2, 3])` 作为输入，broadcast 后所有 rank 都得到同样的张量【Python代码†source】。

**实际意义**：
 broadcast 常用于初始化，例如 rank 0 读取 checkpoint 后，把参数广播给所有 GPU。字幕也说明它通常不是训练主循环中最高频的操作，而更常见于初始化阶段【字幕†source】。

**学习提示**：
 本讲的 broadcast 是分布式通信操作，不是 NumPy/PyTorch 中张量形状自动扩展的 broadcasting。二者名字相似，但语义不同。

------

#### 2.4.2 Scatter 与 Gather：拆分与收集

**Scatter 直觉**：
 把一份大张量切成多块，分别发给不同 rank。

若 rank 0 有：
$$
x = [x_0, x_1, \ldots, x_{p-1}]
$$
scatter 后：
$$
y_r = x_r
$$
课程示例中，rank 0 的 `[0,1,2,3]` 被分给四个 rank 后，rank 0 得到 `[0]`，rank 1 得到 `[1]`，以此类推【Python代码†source】。

**Gather 直觉**：
 gather 是 scatter 的逆过程。每个 rank 持有一小块数据，最终把它们收集到某个目标 rank。

如果各 rank 分别有：
$$
x^{(0)}=[0],\quad x^{(1)}=[1],\quad x^{(2)}=[2],\quad x^{(3)}=[3]
$$
gather 到 rank 0 后：
$$
y^{(0)}=[0,1,2,3]
$$
课堂问答中特别澄清：gather 或 reduce 的目标 rank 不一定固定为 rank 0，rank 0 只是示例中常用的目标【字幕†source】。

**学习提示**：
 Scatter 和 gather 主要帮助理解后面的 reduce-scatter 和 all-gather。真正训练中最常出现的往往不是单独的 scatter/gather，而是它们的 all 版本或与 reduce 结合后的版本。

------

#### 2.4.3 Reduce：跨 rank 做归约

**直觉解释**：
 reduce 就是把多个 rank 上的数据合并成一个结果。合并方式可以是求和、求最大值、求最小值等。

**正式定义**：
$$
y_s = \bigoplus_{r=0}^{p-1} x_r
$$
其中：

- $s$ 是目标 rank；
- $\oplus$ 是归约操作，例如 sum、min、max；
- $x_r$ 是 rank $r$ 上的输入。

**课程例子**：
 如果四个 rank 分别有 0、1、2、3，用 sum reduce 到 rank 0，则输出为：
$$
0+1+2+3=6
$$
Python 讲义中也用 `tensor([6.])` 表示这个结果【Python代码†source】。

**学习提示**：
 reduce 只把结果放到目标 rank。后面的 all-reduce 则让每个 rank 都拿到归约后的完整结果。

------

#### 2.4.4 All-gather：每个 rank 都得到完整拼接结果

**直觉解释**：
 gather 是“收集到某一个人手里”，all-gather 是“每个人都收集一份完整结果”。

若四个 rank 分别持有：
$$
x^{(0)}=0,\quad x^{(1)}=1,\quad x^{(2)}=2,\quad x^{(3)}=3
$$
all-gather 后每个 rank 都得到：
$$
y^{(r)}=[0,1,2,3],\quad \forall r
$$
**训练意义**：
 all-gather 在模型状态被分片时非常重要。例如每张 GPU 只保存一部分参数，forward 时可能需要 all-gather 参数分片，从而临时拼出完整参数参与计算。原报告也指出，后续 FSDP/ZeRO 类方法会反复出现 gather、scatter、再 gather、再 scatter 的模式【字幕†source】。

**工程直觉**：
 all-gather 的代价在于它让每个 rank 都得到完整结果，因此会增加通信量和临时显存占用。它适合“每个设备接下来都需要完整张量”的场景，不适合只需要局部结果的场景。

------

#### 2.4.5 Reduce-scatter：先归约，再分散保存

**直觉解释**：
 reduce-scatter 可以理解为：先把多个 rank 的数据按位置相加，再把最终结果的不同位置分给不同 rank 保存。

课程示例中四个 rank 的输入是：
$$
x^{(0)}=[0,1,2,3]
$$
更完整地看，四个 rank 分别是：
$$
x^{(0)}=[0,1,2,3]
$$
先按列求和：
$$
[0+1+2+3,\ 1+2+3+4,\ 2+3+4+5,\ 3+4+5+6]
=
[6,10,14,18]
$$
然后把结果分散保存：
$$
y^{(0)}=6,\quad y^{(1)}=10,\quad y^{(2)}=14,\quad y^{(3)}=18
$$
Python 代码和运行输出都展示了这一点：reduce-scatter 后四个 rank 分别得到 6、10、14、18【Python代码†source】【运行输出†source】。

**训练意义**：
 reduce-scatter 对分片训练非常关键。backward 后，不同 rank 可能各自产生梯度贡献；如果最终每个 rank 只需要保存梯度的一部分，就不必 all-reduce 到完整梯度，而可以 reduce-scatter：既完成梯度归约，又保持结果分片存储。

------

#### 2.4.6 All-reduce：所有 rank 都得到归约结果

**直觉解释**：
 all-reduce 是分布式训练中最常见的通信操作之一。它把所有 rank 的输入归约，然后把完整归约结果复制给所有 rank。

**公式**：
$$
y_r = \bigoplus_{i=0}^{p-1} x_i,\quad \forall r
$$
对 sum 来说，就是每个 rank 都得到所有 rank 输入之和。

**重要关系**：
$$
\text{all-reduce} = \text{reduce-scatter} + \text{all-gather}
$$
也就是说：

1. reduce-scatter 先得到归约后的分片；
2. all-gather 再把这些分片收集到每个 rank；
3. 于是每个 rank 都有完整归约结果。

课程代码直接写出这个关系，运行输出也显示 reduce-scatter 后各 rank 得到 6、10、14、18，而 all-gather 后每个 rank 都得到 `[6,10,14,18]`【Python代码†source】【运行输出†source】。

**训练意义**：
 数据并行 DDP 的核心就是梯度 all-reduce。每张 GPU 处理不同数据，产生不同梯度；all-reduce 把这些梯度求平均或求和，让所有 GPU 使用同一个梯度更新参数【字幕†source】。

------

#### 2.4.7 All-to-all：最一般的路由通信

**直觉解释**：
 all-to-all 是最一般的通信模式：每个 rank 都可以给每个其他 rank 发送不同片段。

平衡切分时，它看起来像矩阵转置。若输入按 rank 排成矩阵：
$$
\begin{bmatrix}
0 & 1 & 2 & 3\\
4 & 5 & 6 & 7\\
8 & 9 & 10 & 11\\
12 & 13 & 14 & 15
\end{bmatrix}
$$
那么 rank 0 可能得到第一列：
$$
[0,4,8,12]
$$
rank 1 得到第二列，以此类推。

**训练意义**：
 all-to-all 对 Mixture-of-Experts 很重要。MoE 中每个 token 可能被路由到不同 expert，而 expert 又可能分布在不同 rank 上，所以需要把 token 激活发送到对应 expert 所在的设备【字幕†source】。

**学习提示**：
 all-to-all 很强大，但也容易成为通信瓶颈。因为它涉及多对多数据交换，对负载均衡和网络拓扑都比较敏感。

### 2.5 本节小结

本节最重要的结论是：**collective operations 是多 GPU 训练的通信语言**。后续所有并行策略都可以用这些通信原语解释：

- DDP 依赖 all-reduce；
- FSDP/ZeRO 依赖 all-gather 和 reduce-scatter；
- 张量并行常用 all-gather 或 reduce-scatter；
- MoE 常用 all-to-all；
- pipeline parallelism 则更常见 send/recv 这种点对点通信。

------

## 3. 第二部分：硬件互连、NCCL、PyTorch Distributed 与通信 benchmark

本节要解决的问题是：**同样是 all-reduce，为什么在不同机器上速度会差很多？代码中如何实际调用这些通信操作？又如何测量通信性能？**

### 3.1 问题引入：通信原语只是抽象，硬件决定实际速度

上一节讲的是通信“语义”。但真实训练中，all-reduce 的快慢不仅取决于张量大小，还取决于 GPU 如何连接。

课程先对比传统连接和现代数据中心连接：

| 场景                   | 典型互连                     | 直觉                             |
| ---------------------- | ---------------------------- | -------------------------------- |
| 同一台普通服务器内 GPU | PCIe                         | 可以通信，但比 NVLink 慢         |
| 不同普通节点           | Ethernet                     | 通常更慢，且传统路径可能经过 CPU |
| 现代单节点多 GPU       | NVLink / NVSwitch            | 高带宽、适合频繁通信             |
| 多节点训练             | InfiniBand / Ethernet / RoCE | 影响跨节点并行效率               |

原报告中列出典型数值：PCIe v7.0 x16 理论带宽约 242 GB/s；B200 NVLink 5.0 总带宽约 1.8 TB/s；B200 HBM 带宽约 8 TB/s；InfiniBand 示例约 0.05 TB/s【Python代码†source】。这些数字的重点不是死记，而是理解相对层级：**HBM 很快，NVLink 比 HBM 慢但仍很快，跨节点网络通常更慢。**

### 3.2 RDMA：绕过 CPU 的直接内存访问

**直觉解释**：
 传统网络通信往往需要 CPU 参与。GPU 的数据先到 CPU，CPU 构造网络包，再交给网卡。这会增加延迟和 CPU 负担。

**正式定义**：
 RDMA，即 Remote Direct Memory Access，允许一个设备直接读写另一个设备的内存，尽量绕过 CPU。

字幕中解释说，传统 Ethernet 需要 GPU 把数据交给 CPU，CPU 再构造网络包并送到 NIC；RDMA 则可以绕过 CPU，减少延迟【字幕†source】。Python 讲义也写到，InfiniBand 支持 RDMA，而标准 Ethernet 不支持；RoCE 则让 Ethernet 也能实现类似 RDMA 的能力【Python代码†source】。

**学习提示**：
 RDMA 不是某一根线，也不是某一种 GPU。它更像是一种通信能力或机制。NVLink、InfiniBand、RoCE 等硬件/协议可以提供或支持类似的直接访问能力。

### 3.3 NCCL：把 collective operation 变成真实通信

NCCL 是 NVIDIA Collective Communication Library。它的作用是把高层 collective operation 翻译成底层 GPU 间通信。

当代码调用：

```
dist.all_reduce(tensor=data, op=dist.ReduceOp.SUM)
```

程序员看到的是 all-reduce；NCCL 需要处理的是：

- 当前机器有几张 GPU；
- GPU 之间通过什么连接；
- 是否有 NVLink、PCIe、NVSwitch；
- 是否跨节点；
- 应该用 ring、tree 或其他策略；
- 如何启动 GPU kernel 发送和接收数据。

课程说明 NCCL 会探测硬件拓扑、优化 GPU 间通信路径，并启动 GPU kernels 来发送/接收数据【字幕†source】【Python代码†source】。

**工程直觉**：
 NCCL 的价值在于让程序员不用手写底层通信拓扑。但这不意味着通信成本消失了。上层算法仍然要尽量减少不必要通信，或者让通信与计算重叠。

### 3.4 PyTorch Distributed：课程使用的编程接口

PyTorch Distributed 提供高层通信接口，例如：

```
dist.all_reduce(...)
dist.reduce_scatter_tensor(...)
dist.all_gather_into_tensor(...)
dist.barrier()
```

Python 讲义中说明，`torch.distributed` 支持不同后端：GPU 上常用 NCCL，CPU 上可以使用 gloo；它也支持更高层的 FSDP 等算法，但本课为了从底层理解分布式训练，主要使用原始通信原语【Python代码†source】。

### 3.5 `setup()`、`spawn()` 与 `barrier()`

#### `setup(rank, world_size)`

课程代码中的 `setup()` 完成分布式环境初始化：

```
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "15623"

if torch.cuda.is_available():
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
else:
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
```

这段代码做了三件事：

1. 指定 master 地址；
2. 指定 master 端口；
3. 根据是否有 CUDA 选择 NCCL 或 gloo 后端。

**学习重点**：
 `MASTER_ADDR` 和 `MASTER_PORT` 用于协调进程发现彼此，不等于所有数据都通过 master 传输。真实张量通信会走 NCCL 或对应后端【Python代码†source】。

#### `spawn(func, world_size, ...)`

`spawn()` 用于启动多个进程，每个进程运行同一个函数，但 rank 不同。这是典型的 SPMD 风格：Single Program, Multiple Data。

课程代码在正常执行时使用 `mp.spawn`，在 edtrace 追踪时则关闭实际分布式函数，只运行 rank 0 路径，以便生成可视化讲义【Python代码†source】。

#### `barrier()`

`dist.barrier()` 是同步屏障。所有 rank 都到达 barrier 后，程序才继续向下执行。

字幕解释说，多进程异步运行时打印顺序可能混乱；barrier 可以让输出更清楚，但过多 barrier 会让快的 rank 等慢的 rank，从而降低效率【字幕†source】。

**学习提示**：
 barrier 很适合教学和 benchmark，但真实训练中要谨慎使用。同步点越多，越容易破坏并行效率。

### 3.6 PyTorch 集体通信示例

#### all-reduce 示例

每个 rank 构造：
$$
\text{data}^{(r)} = [0,1,2,3] + r
$$
所以四个 rank 的输入是：
$$
[0,1,2,3]
$$
代码调用：

```
dist.all_reduce(tensor=data, op=dist.ReduceOp.SUM, async_op=False)
```

按列求和后，每个 rank 得到：
$$
[6,10,14,18]
$$
运行输出显示四个 rank 在 all-reduce 后都得到 `tensor([ 6., 10., 14., 18.])`【运行输出†source】。

#### reduce-scatter + all-gather 示例

reduce-scatter 后：
$$
rank 0 \to 6,\quad rank 1 \to 10,\quad rank 2 \to 14,\quad rank 3 \to 18
$$
随后 all-gather 把这些分片收集到每个 rank，每个 rank 又得到：
$$
[6,10,14,18]
$$
这从代码和输出上说明：
$$
\text{all-reduce} = \text{reduce-scatter} + \text{all-gather}
$$

### 3.7 通信 benchmark 与有效带宽

课程代码对大张量运行 all-reduce 和 reduce-scatter，并用 `torch.cuda.synchronize()` 和 `dist.barrier()` 保证计时准确【Python代码†source】。

为什么需要这两个同步？

- CUDA kernel 通常是异步启动的，Python 进入下一行时 GPU 可能还没算完；
- 多进程之间也异步运行，不同 rank 可能不同时到达计时代码。

因此 benchmark 中需要：

```
torch.cuda.synchronize()
dist.barrier()
```

#### all-reduce 有效带宽公式

代码定义：
$$
S = \text{data.element\_size()} \times \text{data.numel()}
$$
其中：

- $S$ 是单个 rank 上张量的字节数；
- $p$ 是 world size；
- $T$ 是当前 rank 观测到的通信耗时。

课程代码中 all-reduce 的有效带宽近似为：
$$
B_{\text{eff}}
=
\frac{2S(p-1)}{pT}
$$
这里的 $2$ 来自发送与接收，$(p-1)$ 来自每个 rank 与其他 rank 的通信规模，分母中的 $pT$ 对应代码中用 world size 乘以单 rank 观测时间得到总传输时间【Python代码†source】。

运行输出中，all-reduce 对 104,857,600 个元素的张量耗时约 1.38–1.60 ms，有效带宽约 366–426 GB/s【运行输出†source】。

#### reduce-scatter 有效带宽公式

对于 reduce-scatter，代码定义输入数据字节数：
$$
D = \text{input.element\_size()} \times \text{input.numel()}
$$
有效带宽近似为：
$$
B_{\text{eff}}
=
\frac{D(p-1)}{pT}
$$
这里没有 all-reduce 中的 2 倍，因为 reduce-scatter 可以看作 all-reduce 的一半通信过程。课程代码注释也说明：all-reduce 相比 reduce-scatter 移动 2 倍数据，花 2 倍时间，因此测得带宽可以相近【Python代码†source】。

运行输出中，reduce-scatter 耗时约 2.39–2.61 ms，测得带宽约 450–490 GB/s【运行输出†source】。

### 3.8 本节小结

本节最重要的结论是：**通信原语决定“做什么”，硬件和 NCCL 决定“怎么做得快”，benchmark 决定“到底有多快”。**

复习时要抓住三点：

1. 同样的 all-reduce，在 PCIe、NVLink、InfiniBand 上性能差异很大；
2. PyTorch Distributed 提供接口，NCCL 负责底层实现；
3. benchmark 必须处理 CUDA 异步和多进程异步，否则测得时间不可信。

------

## 4. 第三部分：数据并行 Data Parallelism

本节要解决的问题是：**如果模型能放进每张 GPU，但想用更多 GPU 加速训练，最简单的并行方式是什么？**

答案是：数据并行。

### 4.1 问题引入

假设一张 GPU 可以放下完整模型，但训练数据很多，batch 很大。最自然的想法是：每张 GPU 处理一部分 batch，然后一起更新模型。

这就是数据并行的核心思想：

- 数据切分；
- 模型复制；
- 梯度同步；
- 参数保持一致。

### 4.2 概念定义

**直觉解释**：
 数据并行就像多个学生分别做同一份试卷的不同题目，最后把答案汇总成一份统一反馈。每个 rank 看到的数据不同，但模型结构和初始参数相同。

**正式定义**：
 给定全局 batch：
$$
B = B_0 \cup B_1 \cup \cdots \cup B_{p-1}
$$
rank $r$ 只处理本地 batch：
$$
B_r
$$
每个 rank 都有完整参数副本：
$$
\theta^{(0)}=\theta^{(1)}=\cdots=\theta^{(p-1)}
$$
forward/backward 后，各 rank 得到本地梯度：
$$
g_r = \nabla_\theta L_r(\theta)
$$
DDP 的关键是对梯度求平均：
$$
g = \frac{1}{p}\sum_{r=0}^{p-1} g_r
$$
然后每个 rank 用同一个 $g$ 更新参数：
$$
\theta_{t+1}^{(r)}
=
\operatorname{OptimizerStep}(\theta_t^{(r)}, g)
$$
只要初始参数相同、每步梯度同步相同、优化器状态演化一致，所有 rank 的参数就会保持一致。

### 4.3 方法与原理

数据并行的流程可以分成五步：

1. **切分 batch**：rank $r$ 只拿全局 batch 的一部分；
2. **复制模型**：每个 rank 都有完整 MLP 参数；
3. **本地 forward**：每个 rank 只在自己的数据上计算 loss；
4. **本地 backward**：每个 rank 得到自己的梯度；
5. **all-reduce 梯度**：把梯度平均后，每个 rank 执行相同 optimizer step。

课程代码中，唯一和普通单卡训练明显不同的地方就是 backward 后多了一步：

```
for param in params:
    dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG)
```

这正是 DDP 的核心【Python代码†source】。

### 4.4 代码解析

数据生成：

```
batch_size = 128
num_dim = 1024
data = torch.randn(batch_size, num_dim)
```

world size 为 4 时，每个 rank 的本地 batch size 是：
$$
128 / 4 = 32
$$
代码切分逻辑为：

```
local_batch_size = batch_size // world_size
start_index = rank * local_batch_size
end_index = start_index + local_batch_size
data = data[start_index:end_index]
```

每个 rank 创建完整 MLP 参数：

```
params = [get_init_params(num_dim, num_dim, rank) for layer in range(num_layers)]
optimizer = torch.optim.AdamW(params, lr=1e-3)
```

forward 对应多层 MLP：

```
x = x @ param
x = F.gelu(x)
loss = x.square().mean()
```

用公式表示，第 $\ell$ 层为：
$$
X_{\ell+1} = \operatorname{GELU}(X_\ell W_\ell)
$$
loss 是最终激活平方均值：
$$
L = \operatorname{mean}(X_L^2)
$$
**补充说明**：
 这个 loss 不是语言模型的 next-token 交叉熵损失，而是教学用的简单目标。它的作用是让 MLP forward/backward 能真实运行，从而展示分布式通信结构。

### 4.5 运行结果如何理解

运行输出显示，不同 rank 的 loss 不同，例如 rank 0、1、2、3 的 loss 各不相同；这是因为它们看到的是不同数据 shard。与此同时，输出中的参数摘要保持一致，说明梯度 all-reduce 后各 rank 执行了相同参数更新【运行输出†source】。

这正是数据并行的关键现象：

- loss 可以不同；
- 本地梯度可以不同；
- 同步后的平均梯度相同；
- 参数更新后保持一致。

### 4.6 容易误解的地方

| 误解                             | 正确理解                                                     |
| -------------------------------- | ------------------------------------------------------------ |
| 数据并行会把模型也切开           | 不会。数据并行中每个 rank 保存完整模型                       |
| 每个 rank 的 loss 应该一样       | 不一定。每个 rank 看到的数据不同，loss 可以不同              |
| all-reduce 同步的是参数          | DDP 核心同步的是梯度；参数通过相同 optimizer step 保持一致   |
| batch size 可以小于 world size   | 通常没有意义。至少要让每个 rank 有数据                       |
| batch size 不需要整除 world size | 工业系统可以处理不整除，但本讲教学代码用 `int_divide` 要求整除【Python代码†source】 |

### 4.7 学习提示

数据并行是最容易掌握的并行方式，因为它对模型代码侵入最小。复习时记住一句话：

> 数据并行按 batch 切数据、复制完整模型，并在 backward 后用 all-reduce 平均梯度。

------

## 5. 第四部分：张量并行 Tensor Parallelism

本节要解决的问题是：**如果单张 GPU 放不下完整层参数，或者希望把单层矩阵乘法分摊到多张 GPU 上，该怎么办？**

答案是：张量并行。

### 5.1 问题引入

数据并行要求每张 GPU 都保存完整模型。如果模型太大，单卡放不下完整参数、梯度或优化器状态，数据并行就不够了。

张量并行换了一个切分方向：不再按 batch 切数据，而是按层内张量维度切模型参数。

课程中展示的是 **column tensor parallelism**：把每一层权重矩阵按列切分。

### 5.2 概念定义

**直觉解释**：
 假设一层 MLP 的权重矩阵很宽，可以把这个矩阵按列切成几块。每张 GPU 负责计算一部分输出维度。最后把各 GPU 的局部输出拼起来，就得到完整输出。

**正式定义**：
 设某层权重为：
$$
W_\ell \in \mathbb{R}^{d \times d}
$$
按列切成 $p$ 份：
$$
W_\ell = [W_{\ell,0}, W_{\ell,1}, \ldots, W_{\ell,p-1}]
$$
其中：
$$
W_{\ell,r} \in \mathbb{R}^{d \times d/p}
$$
输入激活为：
$$
X_\ell \in \mathbb{R}^{B \times d}
$$
rank $r$ 计算局部输出：
$$
Z_{\ell,r}=X_\ell W_{\ell,r}
$$
局部输出形状是：
$$
Z_{\ell,r} \in \mathbb{R}^{B \times d/p}
$$
然后通过 all-gather 收集所有局部输出，并沿 hidden dimension 拼接：
$$
X_{\ell+1}
=
[Z_{\ell,0}, Z_{\ell,1}, \ldots, Z_{\ell,p-1}]
\in \mathbb{R}^{B \times d}
$$

### 5.3 方法与原理

张量并行的 forward 大致是：

1. 所有 rank 都拿到输入激活 $X_\ell$；
2. 每个 rank 保存当前层权重的一部分列 $W_{\ell,r}$；
3. 每个 rank 计算自己的局部输出 $X_\ell W_{\ell,r}$；
4. 对局部输出做 all-gather；
5. 拼接成完整 hidden dimension；
6. 进入下一层。

课程代码中：

```
local_num_dim = int_divide(num_dim, world_size)
params = [get_init_params(num_dim, local_num_dim, rank) for layer in range(num_layers)]
```

这说明每个 rank 的参数形状是：
$$
d \times \frac{d}{p}
$$
forward 中：

```
x = x @ params[layer]
x = F.gelu(x)

activations = [
    torch.empty(batch_size, local_num_dim, device=cuda_if_available(rank))
    for _ in range(world_size)
]

dist.all_gather(tensor_list=activations, tensor=x)
x = torch.cat(activations, dim=1)
```

这段代码正对应“局部矩阵乘法 → all-gather 激活 → 拼回完整维度”的过程【Python代码†source】。

### 5.4 张量并行为什么更侵入模型代码

数据并行几乎不改模型 forward；它只是在 backward 后同步梯度。

张量并行则不同。它必须修改模型内部计算逻辑：

- 权重矩阵如何切；
- 每个 rank 计算哪一块；
- 激活何时 all-gather；
- backward 中梯度如何 reduce-scatter；
- 哪些层需要列并行，哪些层需要行并行。

字幕中也指出，张量并行需要“进入模型内部”修改矩阵乘法和激活通信逻辑，因此不像数据并行那样模块化【字幕†source】。

### 5.5 backward 中的 all-gather / reduce-scatter 对偶关系

课堂问答中提到：forward 中使用 all-gather 收集激活，backward 中通常会出现与之对应的 reduce-scatter 或类似通信操作【字幕†source】。

直觉是：

- forward 需要把分散的局部输出拼成完整激活；
- backward 需要把完整激活梯度分配回相关参数分片，并对来自不同 rank 的贡献做归约。

本讲代码把 tensor parallelism 的 backward 留作 homework exercise，因此这里不把完整工业实现写成课程已实现内容【Python代码†source】。

**补充说明**：
 实际大模型中，张量并行通常会配合更复杂的行并行、列并行、sequence parallelism、通信/计算重叠等技术。本讲只展示最小教学版本，用来理解“层内切分 + 激活通信”的基本思想。

### 5.6 硬件要求

张量并行每一层都可能通信激活。因此，它对互连速度要求很高。

课程总结中明确指出：tensor parallelism 通常需要非常快的互连，例如 NVLink【Python代码†source】。字幕也强调，张量并行一般不应跨越较慢的互连域，因为每层都通信，代价会迅速放大【字幕†source】。

### 5.7 学习提示

复习张量并行时抓住三句话：

1. 它按模型层内维度切分，而不是按 batch 切分；
2. column tensor parallel 中，每个 rank 计算部分输出维度；
3. 由于每层都要通信激活，它强依赖高速互连。

------

## 6. 第五部分：流水线并行 Pipeline Parallelism

本节要解决的问题是：**如果模型太深，可以把不同层放到不同 GPU 上吗？如果可以，如何减少等待？**

答案是：流水线并行。

### 6.1 问题引入

张量并行按层内宽度切分参数；流水线并行按网络深度切分模型。

如果一个模型有很多层，可以让：

- rank 0 负责前几层；
- rank 1 负责中间几层；
- rank 2 负责更后面的层；
- 最后一个 rank 负责输出层。

输入数据先经过 rank 0，产生激活后传给 rank 1，再传给后续 rank。这就是 pipeline parallelism。

### 6.2 概念定义

**直觉解释**：
 流水线并行类似工厂流水线。每个工人负责一道工序，产品依次流过各个工位。只要流水线填满，多个工位可以同时处理不同产品。

**正式定义**：
 设模型有 $L$ 层，world size 为 $p$。每个 rank 负责：
$$
L_{\text{local}} = L/p
$$
层。

rank $r$ 负责的层可以写成：
$$
rL_{\text{local}},\ rL_{\text{local}}+1,\ \ldots,\ (r+1)L_{\text{local}}-1
$$
如果 batch 被切成 $m$ 个 micro-batch：
$$
X = [X^{(0)}, X^{(1)}, \ldots, X^{(m-1)}]
$$
rank $r$ 对 micro-batch $j$ 的计算可以写成：
$$
H_{r,j} = f_r(H_{r-1,j})
$$
其中 $f_r$ 表示 rank $r$ 持有的层段。

### 6.3 代码实现

课程代码中先按层数切分：

```
local_num_layers = int_divide(num_layers, world_size)

local_params = [
    get_init_params(num_dim, num_dim, rank)
    for layer in range(local_num_layers)
]
```

这表示每个 rank 拿到若干完整的 $d \times d$ 层参数【Python代码†source】。

然后把 batch 切成 micro-batch：

```
micro_batch_size = int_divide(batch_size, num_micro_batches)

if rank == 0:
    micro_batches = data.chunk(chunks=num_micro_batches, dim=0)
else:
    micro_batches = [
        torch.empty(micro_batch_size, num_dim, device=cuda_if_available(rank))
        for _ in range(num_micro_batches)
    ]
```

每个 rank 对每个 micro-batch 执行：

1. 如果不是 rank 0，就从前一个 rank 接收激活：

```
dist.recv(tensor=x, src=rank - 1)
```

1. 执行本地层：

```
for param in local_params:
    x = x @ param
    x = F.gelu(x)
```

1. 如果不是最后一个 rank，就把激活发给下一个 rank：

```
dist.send(tensor=x, dst=rank + 1)
```

运行输出中可以看到，rank 0 多次把形状为 `32x1024` 的 micro-batch 激活发送给 rank 1【运行输出†source】。

### 6.4 Pipeline bubble 与 micro-batch

**Pipeline bubble** 指流水线中的空闲时间。

如果整个 batch 一次性通过 rank 0，再传给 rank 1，那么在 rank 0 计算时 rank 1 空闲；rank 1 计算时 rank 0 可能又空闲。这种等待就是 bubble。

micro-batch 的作用是把一个大 batch 切成多个小块。rank 0 处理完第一个 micro-batch 后，就可以把它传给 rank 1，同时自己开始处理第二个 micro-batch。这样流水线更快填满，设备空闲时间更少【字幕†source】。

**工程直觉**：
 micro-batch 数越多，通常 bubble 越小，但调度、通信和内存管理也更复杂。实际系统还需要考虑 backward pipeline、梯度累积、激活保存与重计算等问题。

### 6.5 通信与计算重叠

本讲代码是 naive pipeline 版本，没有实现 communication/computation overlap。Python 讲义也明确写到：未处理通信与计算重叠，因此不能消除 pipeline bubbles【Python代码†source】。

理想情况下，一个 rank 可以在：

- 计算当前 micro-batch；
- 异步接收下一个 micro-batch；
- 异步发送上一个 micro-batch；

之间进行重叠。PyTorch 中可以使用异步 send/recv，例如 `isend`、`irecv`，但这会增加调度复杂度【字幕†source】。

### 6.6 学习提示

流水线并行要抓住两个关键词：

- **depth sharding**：按层切模型；
- **bubble**：设备等待导致效率下降。

复习时记住一句话：

> 流水线并行把不同层放在不同 rank 上，用 micro-batch 让多个 rank 同时工作，但必须处理 pipeline bubble 和通信/计算重叠问题。

------

## 7. Python 代码示例详解

本节要解决的问题是：**`lecture_07.py` 如何把前面的概念变成可以运行的分布式训练示例？**

### 7.1 文件整体结构

`lecture_07.py` 是一份可执行讲义。它既用 `edtrace.text`、`image`、`link` 组织课程内容，也用 PyTorch 真正运行分布式通信和训练示例。GitHub 中的 `lecture_07.py` 文件展示了该结构。

| 代码部分       | 主要函数                                                     | 对应课程内容                                                 | 作用                                                  |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ----------------------------------------------------- |
| 课程入口       | `main()`                                                     | 全讲大纲                                                     | 串联通信原语、硬件、PyTorch、benchmark 和三种并行训练 |
| 通信原语       | `collective_operations()`                                    | broadcast、scatter、gather、reduce、all-gather、reduce-scatter、all-reduce、all-to-all | 用小张量展示通信语义                                  |
| 硬件           | `hardware()`                                                 | PCIe、NVLink、NVSwitch、InfiniBand、Ethernet、RDMA、NCCL     | 解释不同互连层次                                      |
| PyTorch 分布式 | `torch_distributed()`、`collective_operations_main()`        | 实际调用 `torch.distributed`                                 | 展示 all-reduce、reduce-scatter、all-gather           |
| benchmark      | `benchmarking()`、`all_reduce()`、`reduce_scatter()`         | 有效带宽测量                                                 | 对大张量通信计时                                      |
| 数据并行       | `data_parallelism()`、`data_parallelism_main()`              | DDP 基本思想                                                 | 切 batch，同步梯度                                    |
| 张量并行       | `tensor_parallelism()`、`tensor_parallelism_main()`          | column tensor parallel                                       | 切权重列，all-gather 激活                             |
| 流水线并行     | `pipeline_parallelism()`、`pipeline_parallelism_main()`      | pipeline + micro-batch                                       | 切层，用 send/recv 传激活                             |
| 工具函数       | `setup()`、`cleanup()`、`spawn()`、`get_init_params()`、`int_divide()` | 分布式运行基础                                               | 初始化、启动进程、初始化参数、保证整除                |

### 7.2 `main()` 如何组织课程主线

`main()` 的顺序很重要。它不是随意堆代码，而是按学习逻辑组织：

1. 先说明从单 GPU 到多 GPU；
2. 引入数据距离和通信瓶颈；
3. 讲 collective operations；
4. 讲硬件互连；
5. 讲 PyTorch Distributed 和 NCCL；
6. 做 benchmark；
7. 用 MLP 展示数据并行、张量并行、流水线并行；
8. 最后总结不同并行方式的适用场景。

原报告指出，`main()` 中依次调用 `collective_operations()`、`hardware()`、`torch_distributed()`、`benchmarking()`，再调用 `data_parallelism()`、`tensor_parallelism()`、`pipeline_parallelism()`【Python代码†source】。这也与 GitHub `lecture_07.py` 的结构一致。

### 7.3 关键函数逐段解析

#### `setup(rank, world_size)`

| 项目         | 说明                                         |
| ------------ | -------------------------------------------- |
| 输入         | 当前 rank，总 world size                     |
| 输出         | 无显式返回值                                 |
| 作用         | 初始化分布式 process group                   |
| 关键变量     | `MASTER_ADDR`、`MASTER_PORT`、backend        |
| 对应课程概念 | 分布式程序中每个 rank 需要加入同一通信组     |
| 学习重点     | master 地址负责协调，不代表数据都经过 master |

#### `spawn(func, world_size, ...)`

| 项目         | 说明                                          |
| ------------ | --------------------------------------------- |
| 输入         | 要运行的函数、world size、函数参数            |
| 输出         | 启动多个进程并等待结束                        |
| 作用         | 让每个 rank 执行同一段函数                    |
| 对应课程概念 | SPMD 编程模型                                 |
| 学习重点     | rank 不同导致每个进程处理的数据、参数或层不同 |

#### `collective_operations_main(rank, world_size)`

| 项目     | 说明                                            |
| -------- | ----------------------------------------------- |
| 输入     | rank、world size                                |
| 输出     | 打印通信前后结果                                |
| 作用     | 实际调用 all-reduce、reduce-scatter、all-gather |
| 对应公式 | all-reduce = reduce-scatter + all-gather        |
| 学习重点 | 通过真实输出理解通信语义，而不是只背定义        |

#### `data_parallelism_main(rank, world_size, data, num_layers, num_steps)`

| 项目     | 说明                                                  |
| -------- | ----------------------------------------------------- |
| 输入     | rank、world size、数据、层数、训练步数                |
| 输出     | 打印 loss 和参数摘要                                  |
| 作用     | 演示最小 DDP 训练                                     |
| 关键代码 | `data[start:end]`、`dist.all_reduce(param.grad, AVG)` |
| 对应公式 | $g=\frac1p\sum_r g_r$                                 |
| 学习重点 | 每个 rank loss 不同，但同步梯度后参数一致             |

#### `tensor_parallelism_main(rank, world_size, data, num_layers)`

| 项目     | 说明                                                      |
| -------- | --------------------------------------------------------- |
| 输入     | rank、world size、数据、层数                              |
| 输出     | 打印 forward 激活摘要                                     |
| 作用     | 演示 column tensor parallel forward                       |
| 关键代码 | `params = d x d/p`、`dist.all_gather`、`torch.cat(dim=1)` |
| 对应公式 | $W=[W_0,\ldots,W_{p-1}]$                                  |
| 学习重点 | 每层局部计算后必须通信激活                                |

#### `pipeline_parallelism_main(rank, world_size, data, num_layers, num_micro_batches)`

| 项目     | 说明                                         |
| -------- | -------------------------------------------- |
| 输入     | rank、world size、数据、层数、micro-batch 数 |
| 输出     | 打印 send 激活信息                           |
| 作用     | 演示 pipeline forward                        |
| 关键代码 | `data.chunk`、`dist.recv`、`dist.send`       |
| 对应公式 | $H_{r,j}=f_r(H_{r-1,j})$                     |
| 学习重点 | 按层切分模型，用 micro-batch 减少 bubble     |

### 7.4 关键代码片段解析

#### DDP 梯度同步

```
loss.backward()

for param in params:
    dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG)

optimizer.step()
```

这段代码对应课程中数据并行的核心。每个 rank 先对本地数据做 backward，得到本地梯度；随后用 all-reduce 求平均梯度；最后每个 rank 用同一个平均梯度更新参数。

**理论对应**：
$$
g = \frac{1}{p}\sum_{r=0}^{p-1} g_r
$$
**学习重点**：
 DDP 不是把每个 rank 的 loss 先平均再 backward，而是每个 rank 独立 backward，然后同步梯度。

------

#### Tensor parallel 激活 all-gather

```
x = x @ params[layer]
x = F.gelu(x)

activations = [
    torch.empty(batch_size, local_num_dim, device=cuda_if_available(rank))
    for _ in range(world_size)
]

dist.all_gather(tensor_list=activations, tensor=x)
x = torch.cat(activations, dim=1)
```

这段代码把矩阵乘法拆成多个局部输出。每个 rank 只计算 $d/p$ 个输出维度，随后 all-gather 收集所有局部输出，再拼接回完整 hidden dimension。

**理论对应**：
$$
XW = [XW_0, XW_1, \ldots, XW_{p-1}]
$$
**学习重点**：
 张量并行不是减少总计算量，而是把同一层计算分摊到多张 GPU，同时引入每层通信。

------

#### Pipeline send/recv

```
if rank - 1 >= 0:
    dist.recv(tensor=x, src=rank - 1)

for param in local_params:
    x = x @ param
    x = F.gelu(x)

if rank + 1 < world_size:
    dist.send(tensor=x, dst=rank + 1)
```

这段代码体现 pipeline parallelism 的核心：每个 rank 只负责部分层，输入激活来自前一个 rank，输出激活传给下一个 rank。

**理论对应**：
$$
H_{r,j}=f_r(H_{r-1,j})
$$
**学习重点**：
 流水线并行的通信是相邻 stage 之间的激活传递，和 DDP 中所有 rank 参与的 all-reduce 不同。

### 7.5 代码与课程知识点对应表

| 课程知识点        | 对应代码函数 / 变量 / 片段                                 | 代码功能                       | 对应理论原理                 | 在课程中的作用            |
| ----------------- | ---------------------------------------------------------- | ------------------------------ | ---------------------------- | ------------------------- |
| rank / world size | 多数分布式函数参数                                         | 标识当前进程与总进程数         | SPMD 分布式模型              | 所有通信操作的基础        |
| all-reduce        | `dist.all_reduce(...)`                                     | 对所有 rank 张量归约并复制结果 | $\forall r,\ y_r=\sum_i x_i$ | DDP 梯度同步核心          |
| reduce-scatter    | `dist.reduce_scatter_tensor(...)`                          | 归约后分散保存                 | reduce + scatter             | 分片梯度/状态基础         |
| all-gather        | `dist.all_gather_into_tensor(...)`、`dist.all_gather(...)` | 收集分片到每个 rank            | gather to all                | 参数或激活重构            |
| benchmark         | `sent_bytes / total_duration`                              | 估算有效带宽                   | 数据量 / 时间                | 量化通信瓶颈              |
| 数据并行          | `data[start:end]` + `all_reduce(param.grad, AVG)`          | 切 batch，同步梯度             | $g=\frac1p\sum_r g_r$        | 最基础 DDP 示例           |
| 张量并行          | `params = d x d/p` + `all_gather` + `cat(dim=1)`           | 切权重列，拼接激活             | column tensor parallel       | 展示层内分片              |
| 流水线并行        | `local_params` + `data.chunk` + `recv/send`                | 切层并传 micro-batch           | depth sharding               | 展示 pipeline bubble 动机 |
| 可复现实验        | `torch.random.manual_seed(0)`                              | 保证初始化一致                 | 相同初值                     | 防止 DDP 参数初始不同     |
| 整除约束          | `int_divide(a,b)`                                          | 要求可均匀切分                 | 均衡分片                     | 简化教学实现              |

------

## 8. PDF / 幻灯片 / 图表解析

本节要解决的问题是：**本讲有哪些图表？它们分别帮助理解什么？**

### 8.1 PDF 情况

原报告指出，仓库中未找到 `lecture_07.pdf`，Lecture 7 对应的是 `lecture_07.py`；GitHub 仓库 README 也说明可执行讲义命名为 `lecture_XX.py`，非可执行讲义命名为 `lecture_XX.pdf`，文件列表中确实存在 `lecture_07.py`。 因此，本讲的图表主要来自 Python 讲义中的 `image(...)` 调用，而不是单独 PDF 幻灯片。

### 8.2 主要图表总览

| 图表 / 文件                       | 展示内容                | 支持的课程结论                           | 对应代码或字幕内容                                      |
| --------------------------------- | ----------------------- | ---------------------------------------- | ------------------------------------------------------- |
| `images/gpu-node-overview.png`    | 多 GPU 节点结构         | 多 GPU 通信有层次，跨设备通信比 HBM 更慢 | `main()`、`hardware()` 中调用【图表/Python代码†source】 |
| `images/ranks.png`                | rank 与 world size 示意 | collective operations 的参与者坐标系     | `collective_operations()`【图表/Python代码†source】     |
| `images/data-parallelism.png`     | 按 batch 切分数据       | 数据并行切数据、不切模型                 | `data_parallelism()`【图表/Python代码†source】          |
| `images/tensor-parallelism.png`   | 按层内宽度切参数        | 张量并行切矩阵列，需要通信激活           | `tensor_parallelism()`【图表/Python代码†source】        |
| `images/pipeline-parallelism.png` | 按网络深度切层          | 流水线并行切层，并传递 micro-batch 激活  | `pipeline_parallelism()`【图表/Python代码†source】      |

### 8.3 GPU node overview 图

这张图服务于课程开头的核心类比：在单 GPU 中，计算单元和 HBM 之间有距离；在多 GPU 中，不同 GPU、不同节点之间也有距离。

读图时应关注：

- GPU 内部有自己的 HBM；
- 单节点内多个 GPU 可能通过 NVLink / NVSwitch 连接；
- 跨节点通信可能经过 InfiniBand 或 Ethernet；
- 越跨越更远层级，通信通常越慢。

**容易误解的地方**：
 这张图不是在展示某个具体训练算法，而是在展示硬件拓扑。它回答的是“通信为什么有快慢”，不是“具体怎么切模型”。

### 8.4 ranks 图

ranks 图展示多个 rank 和 world size。它是理解后续所有 collective operations 的基础。

读图时要看：

- 每个 rank 是一个通信参与者；
- world size 是参与者总数；
- broadcast、scatter、all-reduce 等操作都在这些 rank 之间发生。

**学习提示**：
 如果 rank/world size 没理解清楚，后面的公式会很抽象。复习时可以始终用 4 个 rank：0、1、2、3 来想象通信过程。

### 8.5 data parallelism 图

data parallelism 图展示的是按 batch dimension 切分数据。

读图重点：

- 数据被切成多个子 batch；
- 每个 rank 拿一个子 batch；
- 每个 rank 保留完整模型；
- backward 后用 all-reduce 同步梯度。

这张图支持的结论是：数据并行对模型代码侵入最小，因为模型本身没有被切开。

### 8.6 tensor parallelism 图

tensor parallelism 图展示的是按层内参数切分。

读图重点：

- 每个 rank 拿到每层权重的一部分；
- 输入激活通常需要被所有 rank 使用；
- 局部输出需要 all-gather 后拼接；
- 每层都可能通信，因此需要高速互连。

这张图支持的结论是：张量并行比数据并行更复杂，因为它要修改模型内部矩阵乘法。

### 8.7 pipeline parallelism 图

pipeline parallelism 图展示的是按网络深度切分层。

读图重点：

- 每个 rank 负责连续的一段层；
- 激活从前一个 stage 传到后一个 stage；
- batch 可以切成 micro-batches；
- 没有良好调度时会产生 pipeline bubble。

这张图支持的结论是：流水线并行的核心难点不是如何切层，而是如何减少不同 stage 的等待时间。

------

## 9. 关键公式与推导汇总

本节要解决的问题是：**复习时应集中掌握哪些公式？这些公式分别对应哪些代码和课程概念？**

### 9.1 Collective operations 公式表

| 操作           | 公式                                         | 符号解释                       | 适用场景                    | 直觉解释                    |
| -------------- | -------------------------------------------- | ------------------------------ | --------------------------- | --------------------------- |
| Broadcast      | $y_r=x_s,\ \forall r$                        | $s$ 为源 rank                  | 参数初始化、checkpoint 广播 | 一份数据复制给所有人        |
| Scatter        | 若 $x=[x_0,\ldots,x_{p-1}]$，则 $y_r=x_r$    | $p$ 为 world size              | 分发数据或分片              | 一个大张量拆给多张卡        |
| Gather         | $y_s=[x_0,\ldots,x_{p-1}]$                   | $s$ 为目标 rank                | 收集结果                    | scatter 的逆过程            |
| Reduce         | $y_s=\bigoplus_{r=0}^{p-1}x_r$               | $\oplus$ 为 sum/min/max 等     | 梯度或统计量归约            | 多份数据合成一份            |
| All-gather     | $y_r=[x_0,\ldots,x_{p-1}],\ \forall r$       | 所有 rank 都得到完整拼接       | 参数/激活重构               | 每个人都 gather             |
| Reduce-scatter | $y_j=\bigoplus_r x_{r,j}$，结果放到 rank $j$ | 先按维度归约，再分散           | 分片梯度/优化器状态         | reduce 后不复制，只分片保存 |
| All-reduce     | $y_r=\bigoplus_i x_i,\ \forall r$            | 每个 rank 都得到归约结果       | DDP 梯度同步                | reduce 后复制给所有人       |
| All-to-all     | $y_j=[x_{0,j},x_{1,j},\ldots,x_{p-1,j}]$     | 每个 rank 给每个目标发不同片段 | MoE routing                 | 平衡时像矩阵转置            |

### 9.2 all-reduce = reduce-scatter + all-gather

先做 reduce-scatter：
$$
z_j = \sum_{r=0}^{p-1} x_{r,j}
$$
rank $j$ 保存 $z_j$。

再做 all-gather：
$$
y_r=[z_0,z_1,\ldots,z_{p-1}],\quad \forall r
$$
于是每个 rank 都得到完整归约结果，这正是 all-reduce。

代码和运行输出都展示了这一点：reduce-scatter 后各 rank 得到 6、10、14、18；all-gather 后每个 rank 都得到 `[6,10,14,18]`【Python代码†source】【运行输出†source】。

### 9.3 DDP 梯度平均公式

本地 loss：
$$
L_r(\theta)=\frac{1}{|B_r|}\sum_{x\in B_r}\ell(x;\theta)
$$
本地梯度：
$$
g_r = \nabla_\theta L_r(\theta)
$$
平均梯度：
$$
g=\frac{1}{p}\sum_{r=0}^{p-1}g_r
$$
参数更新：
$$
\theta_{t+1}^{(r)}
=
\operatorname{OptimizerStep}(\theta_t^{(r)}, g)
$$
代码中：

```
dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG)
```

正是实现平均梯度【Python代码†source】。

### 9.4 Column tensor parallel 公式

权重列切分：
$$
W_\ell=[W_{\ell,0},W_{\ell,1},\ldots,W_{\ell,p-1}]
$$
局部计算：
$$
Z_{\ell,r}=X_\ell W_{\ell,r}
$$
拼接输出：
$$
X_{\ell+1}
=
[Z_{\ell,0},Z_{\ell,1},\ldots,Z_{\ell,p-1}]
$$
代码中 `params[layer]` 的形状是 `num_dim x local_num_dim`，`dist.all_gather` 收集激活，`torch.cat(..., dim=1)` 拼回完整维度【Python代码†source】。

### 9.5 Pipeline micro-batch 公式

batch 切成 micro-batches：
$$
X=[X^{(0)},X^{(1)},\ldots,X^{(m-1)}]
$$
rank $r$ 持有层段 $f_r$，对 micro-batch $j$：
$$
H_{r,j}=f_r(H_{r-1,j})
$$
代码中 rank 0 用 `data.chunk(chunks=num_micro_batches, dim=0)` 切分数据，后续 rank 分配空 activation buffer，并用 `recv` / `send` 接收和发送激活【Python代码†source】。

### 9.6 有效带宽公式

all-reduce：
$$
B_{\text{eff}}
=
\frac{2S(p-1)}{pT}
$$
reduce-scatter：
$$
B_{\text{eff}}
=
\frac{D(p-1)}{pT}
$$
其中：

| 符号             | 含义                              |
| ---------------- | --------------------------------- |
| $S$              | all-reduce 中单个 rank 张量字节数 |
| $D$              | reduce-scatter 输入总字节数       |
| $p$              | world size                        |
| $T$              | 当前 rank 测得的通信耗时          |
| $B_{\text{eff}}$ | 代码定义下的有效带宽              |

**学习提示**：
 这些公式是本讲 benchmark 代码中的带宽估算方式，不是所有通信拓扑下唯一的理论公式。复习时更应理解“数据量 / 时间”的思想，以及 all-reduce 通常可分解为 reduce-scatter 与 all-gather。

------

## 10. 概念对比、易混淆点与拓展理解

本节要解决的问题是：**哪些概念容易混？这些知识如何连接大模型训练实践？考试、面试或作业可能怎么考？**

### 10.1 易混淆概念对比

| 易混淆概念                           | 区别                                                  | 联系                                                         | 常见误解                                  | 正确理解                                     |
| ------------------------------------ | ----------------------------------------------------- | ------------------------------------------------------------ | ----------------------------------------- | -------------------------------------------- |
| Broadcast vs 张量 broadcasting       | 前者是跨设备通信，后者是张量形状扩展                  | 都有“一变多”的直觉                                           | 以为二者是同一机制                        | 本讲 broadcast 是 collective operation       |
| Gather vs All-gather                 | gather 只到目标 rank；all-gather 每个 rank 都得到结果 | all-gather 可看作对所有 rank 做 gather                       | 以为 gather 后所有 rank 都有结果          | 只有 all-gather 才是所有 rank 都有完整结果   |
| Reduce vs All-reduce                 | reduce 到某个目标；all-reduce 到所有 rank             | all-reduce 可由 reduce + broadcast 或 reduce-scatter + all-gather 实现 | 以为 reduce 后所有 rank 都有结果          | DDP 用的是 all-reduce                        |
| Reduce-scatter vs All-reduce         | reduce-scatter 结果分片保存；all-reduce 结果完整复制  | all-reduce = reduce-scatter + all-gather                     | 以为 reduce-scatter 只是更慢的 all-reduce | reduce-scatter 更适合分片状态                |
| Data parallel vs Tensor parallel     | 前者切 batch，后者切层内参数                          | 都是并行训练策略                                             | 以为二者都不改模型                        | Tensor parallel 通常需要改模型 forward       |
| Tensor parallel vs Pipeline parallel | 前者切宽度，后者切深度                                | 都能降低单卡模型状态压力                                     | 以为二者通信代价类似                      | Tensor parallel 通信更频繁，通常需要高速互连 |
| NCCL vs PyTorch Distributed          | PyTorch 是高层接口，NCCL 是底层 GPU 通信库            | PyTorch GPU 后端常用 NCCL                                    | 以为 NCCL 是训练框架                      | NCCL 负责 collective 的底层实现              |
| rank vs GPU                          | rank 是进程/设备编号；本讲中近似为 GPU                | 通常一个 rank 绑定一张 GPU                                   | 以为 rank 永远等于物理 GPU ID             | 更一般地 rank 是分布式进程编号               |

### 10.2 与大模型训练实践的关系

#### 数据准备

数据并行要求不同 rank 处理不同数据 shard。实际训练中，不应像教学代码那样先在每个 rank 拿到完整 data 再切分；更常见做法是每个 rank 从数据加载器中直接读取自己负责的数据 shard。

**延伸理解**：
 在大规模训练中，数据加载本身也可能成为瓶颈。数据并行不仅要求 GPU 通信高效，也要求数据管线能持续供给每个 rank。

#### 模型训练

本讲三种并行方式对应三种基本切分方向：

- batch 维度：data parallelism；
- width 维度：tensor parallelism；
- depth 维度：pipeline parallelism。

现代大模型训练通常不是只用一种，而是组合使用。例如：

- 节点内用 tensor parallel；
- 节点间用 data parallel；
- 超大模型再叠加 pipeline parallel；
- 参数、梯度、优化器状态再用 FSDP/ZeRO 分片。

这些组合思想与本讲的通信原语直接相关，但具体组合策略属于更完整训练系统设计，本文不把它写成 Lecture 7 已完整实现的内容。

#### 优化策略

DDP 中 all-reduce 梯度是关键同步点。如果 all-reduce 太慢，GPU 会等待通信，计算资源空闲。

张量并行中每层都可能通信激活，如果互连慢，层间通信会吞掉矩阵乘法加速收益。

流水线并行中如果 micro-batch 太少，pipeline bubble 会导致 stage 空闲；如果 micro-batch 太多，又会带来调度和内存开销。

#### 工程成本

硬件拓扑直接影响并行策略：

| 硬件环境                      | 更适合的策略                       | 原因                                      |
| ----------------------------- | ---------------------------------- | ----------------------------------------- |
| 单节点多 GPU，NVLink/NVSwitch | 张量并行、数据并行                 | 高频通信较可承受                          |
| 多节点 InfiniBand             | 数据并行、流水线并行、部分张量并行 | 带宽较高但仍需控制通信                    |
| 较慢 Ethernet                 | 粗粒度流水线或较少同步             | 高频 all-gather / all-reduce 可能代价过高 |

### 10.3 与其他 lecture 的联系

**与上一讲的联系**：
 上一讲讨论单 GPU 内部优化，核心是减少内存访问和提高 kernel 效率。本讲把“少搬数据”的思想扩展到多 GPU：少跨 GPU 通信，或者让通信发生在更快的互连域内【字幕†source】。

**与后续内容的联系**：
 原报告提到下一讲可能进一步讨论 FSDP/ZeRO，这与本讲中的 all-gather、reduce-scatter 直接相关【Python代码†source】。可以把本讲看作分布式训练的基础语言课：先学会通信原语，后面才能理解更复杂的模型状态分片和训练系统。

**与 CS336 主线的关系**：
 CS336 关注从零理解语言模型训练。模型结构、数据、优化目标固然重要，但当模型和数据规模变大后，训练能否进行往往取决于系统能力。本讲正是把语言模型训练从“算法能不能写出来”推进到“系统能不能跑起来”。

### 10.4 面试 / 考试 / 作业可能考察角度

| 问题                                               | 考察点                | 回答思路                                                     |
| -------------------------------------------------- | --------------------- | ------------------------------------------------------------ |
| 解释 all-reduce、reduce-scatter、all-gather 的关系 | collective operations | all-reduce 可以分解为 reduce-scatter + all-gather            |
| 为什么 DDP 同步梯度而不是同步参数？                | 数据并行原理          | 初始参数相同，同步平均梯度后 optimizer step 相同，参数自然保持一致 |
| 数据并行的显存瓶颈是什么？                         | 模型状态复制          | 每个 rank 保存完整参数、梯度、优化器状态                     |
| column tensor parallel 的 forward 如何实现？       | 张量切分              | 按列切 $W$，局部算 $XW_r$，all-gather 激活后 concat          |
| 为什么 tensor parallel 需要高速互连？              | 通信频率              | 每层都可能通信激活，跨慢网络代价太高                         |
| pipeline parallel 为什么需要 micro-batch？         | pipeline bubble       | micro-batch 让流水线更快填满，减少 stage 空闲                |
| `barrier()` 为什么不能随便加？                     | 同步开销              | barrier 会让快 rank 等慢 rank，影响并行效率                  |
| benchmark 为什么要 `torch.cuda.synchronize()`？    | CUDA 异步执行         | 不同步会导致 Python 计时提前结束，低估真实耗时               |
| all-to-all 在什么场景重要？                        | MoE routing           | token 需要路由到不同 rank 上的 expert                        |
| 本讲代码为什么只实现部分 backward？                | 教学边界              | 重点是理解通信结构，tensor/pipeline backward 留作练习        |

------

## 11. 课程复习总结

### 11.1 本讲核心思想

Lecture 7 的核心思想是：**分布式训练不是简单地“多加 GPU”，而是在显存、计算和通信之间做系统性权衡。** 课程先用 rank/world size 和 collective operations 建立通信语言，再用 NVLink、NVSwitch、InfiniBand、RDMA、NCCL 解释通信为什么有快慢，最后通过 MLP 示例展示三种基本并行方向：按 batch 切的数据并行、按 width 切的张量并行、按 depth 切的流水线并行【字幕†source】【Python代码†source】。

### 11.2 必须掌握的概念

复习时必须掌握：

- rank；
- world size；
- broadcast；
- scatter；
- gather；
- reduce；
- all-gather；
- reduce-scatter；
- all-reduce；
- all-to-all；
- NCCL；
- RDMA；
- PyTorch Distributed；
- DDP；
- tensor parallelism；
- pipeline parallelism；
- micro-batch；
- pipeline bubble；
- 有效通信带宽。

其中最核心的是：

> DDP 的关键是梯度 all-reduce；FSDP/ZeRO 的方向是 all-gather + reduce-scatter；tensor parallelism 需要高速互连；pipeline parallelism 需要减少 bubble。

### 11.3 必须理解的公式

| 公式                                                        | 解决的问题             | 复习重点                     |
| ----------------------------------------------------------- | ---------------------- | ---------------------------- |
| $y_r=x_s,\ \forall r$                                       | broadcast              | 一份数据复制到所有 rank      |
| $y_s=\bigoplus_r x_r$                                       | reduce                 | 多个 rank 归约到一个目标     |
| $y_r=[x_0,\ldots,x_{p-1}],\ \forall r$                      | all-gather             | 每个 rank 得到完整拼接       |
| $y_j=\bigoplus_r x_{r,j}$                                   | reduce-scatter         | 归约后分片保存               |
| $\text{all-reduce}=\text{reduce-scatter}+\text{all-gather}$ | all-reduce 分解        | 理解 DDP 与 FSDP/ZeRO 的连接 |
| $g=\frac1p\sum_r g_r$                                       | DDP 梯度平均           | 为什么参数保持一致           |
| $W=[W_0,\ldots,W_{p-1}]$                                    | column tensor parallel | 每个 rank 负责部分输出维度   |
| $H_{r,j}=f_r(H_{r-1,j})$                                    | pipeline micro-batch   | 激活如何跨 stage 流动        |
| $B_{\text{eff}}=\frac{2S(p-1)}{pT}$                         | all-reduce benchmark   | 如何估算通信有效带宽         |

### 11.4 必须看懂的代码

| 代码片段                                                 | 学习重点                  |
| -------------------------------------------------------- | ------------------------- |
| `setup(rank, world_size)`                                | 如何初始化 process group  |
| `mp.spawn(...)`                                          | 如何启动多个 rank         |
| `dist.all_reduce(...)`                                   | DDP 梯度同步核心          |
| `dist.reduce_scatter_tensor(...)`                        | 分片归约基础              |
| `dist.all_gather_into_tensor(...)`                       | 参数/激活重构基础         |
| `data[start_index:end_index]`                            | 数据并行按 batch 切分     |
| `params = get_init_params(num_dim, local_num_dim, rank)` | 张量并行按列切参数        |
| `torch.cat(activations, dim=1)`                          | 拼接局部 hidden dimension |
| `dist.recv(...)` / `dist.send(...)`                      | pipeline stage 间传激活   |
| `torch.cuda.synchronize()` + `dist.barrier()`            | benchmark 中处理异步执行  |

### 11.5 最后复习提示

本讲最容易混淆的不是公式，而是不同并行方式的切分方向：

| 并行方式   | 切什么       | 复制什么      | 主要通信                    | 最大难点               |
| ---------- | ------------ | ------------- | --------------------------- | ---------------------- |
| 数据并行   | batch        | 完整模型      | 梯度 all-reduce             | 显存复制、通信同步     |
| 张量并行   | 层内参数维度 | 数据/部分激活 | all-gather / reduce-scatter | 高频通信、高速互连需求 |
| 流水线并行 | 网络深度     | 每段完整层    | send/recv 激活              | pipeline bubble 和调度 |

### 11.6 一句话总结

Lecture 7 把“让单 GPU 少搬数据”的思想扩展为“让多 GPU 少通信、会通信、按硬件拓扑通信”，并用 PyTorch 原语从零构建了数据并行、张量并行和流水线并行的最小工作模型。