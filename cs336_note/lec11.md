# Lecture 11 课程报告：Scaling in Practice——大模型扩展、超参数调优与 muP 参数化细节

## 0. 资料来源与改写说明

本报告基于本次对话中可见的两份资料整理而成：一份英文字幕文本文件，文件开头标注为 “Subtitle”，内容记录了教师对 Lecture 11 的逐段讲解；另一份为 `lecture_11.pdf`，标题为 **Scaling – Case Study and Details**。我没有看到单独的、已生成好的中文原始课程报告，因此以下内容不是对某份中文报告逐句润色，而是基于字幕与 PDF 资料，按照用户要求整理成一份中文学习讲义。字幕显示本讲从“继续 scaling journey”开始，课程目标是讨论 scaling laws 的高级细节、真实开源模型中的 scaling 实践、学习率与 batch size 调优、优化器和初始化在规模变化下的敏感性。

GitHub 仓库中，CS336 lecture 文件分为可执行的 `lecture_XX.py` 与不可执行的 `lecture_XX.pdf`；当前仓库文件列表中 Lecture 11 对应的是 `lecture_11.pdf`，没有看到 `lecture_11.py`。课程官网的日程也把 Lecture 11 标为 **Scaling laws [Tatsu]**，课程资料为 `lecture_11.pdf`。因此，本报告的“代码示例”部分主要解释 PDF 中的算法伪代码、实验流程和 scaling recipe，而不是解析某个 Python 文件。 

本报告中的引用规则如下：
 “【字幕】”表示内容来自上传的字幕文本；“【PDF】”表示内容来自上传的 Lecture 11 PDF；“【GitHub / 课程官网】”表示来自公开课程仓库或课程官网。凡是用于学习理解的补充解释，会明确标注为“补充说明”“延伸理解”或“工程直觉”，不把它伪装成课程原文。

------

## 1. 学习导读

### 1.1 本讲要解决的核心问题

本讲要回答的问题是：**当我们真正把语言模型从小规模实验扩展到大规模训练时，scaling law 还能不能可靠地指导模型规模、数据规模、学习率、batch size、优化器和参数化方式的选择？**

前面课程已经讲过 Kaplan scaling law、Chinchilla scaling law 等经典结果。本讲进一步追问：这些规律在 MiniCPM、DeepSeek、Qwen、Kimi K2、Hunyuan、LLaMA 3、MiniMax 等真实模型训练案例中如何使用？哪些结论稳定？哪些只是经验拟合？哪些超参数会随规模漂移？如何降低调参成本？

教师在开头明确指出，本讲会覆盖 scaling law 的更高级细节、真实构建语言模型时的重要实践，以及初始化、学习率、batch size、优化器这些随规模变化而敏感的内容。【字幕】

### 1.2 本讲在 CS336 课程中的位置

CS336 的主线是从零构建语言模型：tokenization、PyTorch 与计算资源、Transformer 架构、注意力与 MoE、GPU 与并行系统、scaling laws、inference、evaluation、data、post-training 等。课程官网显示 Lecture 9 是 scaling laws，Lecture 10 是 inference，Lecture 11 继续 scaling laws，随后进入 evaluation 和 data。

因此，Lecture 11 的作用是把前面较“理想化”的 scaling law 推向真实训练场景。它不是重新介绍语言模型损失函数，而是讨论：在预算有限、模型很大、训练很贵的情况下，如何用小规模实验预测大规模训练；如何选择学习率、batch size、模型大小、数据量、稀疏度、优化器和参数化方式。

### 1.3 本讲知识结构

| 模块               | 解决的问题                                           | 核心概念                                             | 与后续内容的关系                              |
| ------------------ | ---------------------------------------------------- | ---------------------------------------------------- | --------------------------------------------- |
| 课程动机           | scaling law 在真实大模型训练中是否可靠               | Chinchilla、IsoFLOP、超参数漂移                      | 为后续模型规模选择与训练预算规划打基础        |
| MiniCPM 案例       | 如何用 muP 和 WSD 降低调参成本                       | muP、稳定学习率、WSD、critical batch                 | 引出“稳定超参数”的路线                        |
| DeepSeek 案例      | 不用 muP 时如何直接拟合学习率和 batch size           | grid search、power law、IsoFLOP                      | 引出“直接拟合超参数 scaling law”的路线        |
| 近期模型案例       | scaling law 如何服务 MoE、架构选择和 downstream 预测 | Qwen、Kimi K2、Hunyuan、LLaMA 3、MiniMax             | 展示 scaling law 已经成为模型发布中的工程工具 |
| StepFun 超参数研究 | LR / batch size 到底该依赖哪些变量                   | batch-size scaling、LR scaling、N 与 D               | 给出更系统的经验观察                          |
| 优化器 scaling     | 小规模优化器收益能否外推到大模型                     | AdamW、Muon、Chinchilla ratio、scale dependence      | 提醒算法创新必须做 scaling 检验               |
| muP 深入推导       | 为什么某些参数化能让学习率跨规模稳定                 | activation scale、feature learning、fan-in / fan-out | 连接初始化理论与大模型训练实践                |
| 总结               | scaling in the wild 为什么既像科学又像手艺           | 稳定性、经验拟合、WSD、无银弹                        | 帮助形成真实工程判断                          |

### 1.4 学习本讲前需要知道什么

学习本讲前最好具备以下基础：

| 前置知识               | 为什么需要                                                   |
| ---------------------- | ------------------------------------------------------------ |
| 概率与统计             | scaling law 本质是对实验点做统计拟合，需要理解误差、外推和置信区间 |
| 梯度下降与学习率       | 本讲大量讨论 learning rate 如何随模型规模变化                |
| batch size 与优化噪声  | critical batch、optimal batch 都和梯度噪声及计算效率相关     |
| Transformer 基本结构   | muP 会区分 embedding、attention、MLP、LM head 等参数类型     |
| Chinchilla scaling law | 本讲多次讨论模型参数量与 token 数之间的最优 tradeoff         |
| GPU 训练成本意识       | WSD 的核心动机是减少重复训练，节省 scaling experiment 的计算成本 |
| 矩阵范数和 SVD         | muP 推导和 Muon 优化器都涉及矩阵谱范数、奇异值和正交化       |

------

## 2. 第一部分：为什么要重新讨论 scaling in practice？

### 2.1 问题引入

课程一开始提出三个问题：Chinchilla 的 scaling 方法是否真的有效？在拟合 scaling law 时能不能省计算？我们是否应该选择某些架构或参数化方式，让模型更容易稳定扩展？PDF 第 2 页把这些问题作为本讲动机列出。【PDF】

这三个问题背后有一个共同矛盾：**大模型训练太贵，所以不能靠“训练很多大模型”来调参；但小模型实验又不一定能可靠外推到大模型。**

因此，本讲的核心不是“scaling law 是否优雅”，而是“scaling law 在工程上怎样被用来减少错误决策”。

### 2.2 概念定义：scaling law 的工程含义

**直觉解释**：scaling law 就像训练大模型前的“预算地图”。我们先在便宜的小模型、小数据、小 compute 上训练许多点，然后拟合出损失、参数量、数据量、计算量之间的规律，最后外推到真正昂贵的大训练。

**正式理解**：在语言模型预训练中，常见 scaling law 会描述 loss 与模型参数量 $N$、训练 token 数 $D$、计算量 $C$、batch size $B$、learning rate $\eta$ 等变量之间的关系。例如 Chinchilla 风格会关心给定 compute 下如何分配模型参数和训练数据。

**它解决的问题**：
 如果我们有固定训练预算，应该训练更大的模型还是用更多 token？如果模型规模扩大 10 倍，学习率是否还应该一样？batch size 是否也要变大？如果选择 MoE，稀疏度应该多高？如果更换优化器，小规模 benchmark 的收益是否可靠？

### 2.3 方法与原理：本讲的两条路线

本讲反复对比两种路线：

| 路线       | 代表案例           | 核心思想                                                     | 优点               | 风险                                            |
| ---------- | ------------------ | ------------------------------------------------------------ | ------------------ | ----------------------------------------------- |
| 稳定化路线 | MiniCPM / muP      | 改变初始化、残差缩放、每类参数学习率，让最优学习率尽量不随规模变化 | 减少学习率调参成本 | 实现复杂，理论假设与现代 Transformer 不完全匹配 |
| 拟合路线   | DeepSeek / StepFun | 不强行让超参数不变，而是在多个小规模实验中直接拟合 LR 和 batch 的 scaling law | 直接、经验有效     | 拟合质量依赖 grid 位置、数据、架构和 optimizer  |

教师在讲 DeepSeek 时总结得很清楚：处理敏感超参数可以尝试稳定它们，也可以尝试直接拟合它们随规模变化的 scaling law；DeepSeek 更代表后者。【字幕】

### 2.4 学习提示

本节最重要的结论是：**scaling law 不只是预测 loss 的曲线，它已经成为大模型训练中选择学习率、batch size、模型大小、数据量、MoE 稀疏度、优化器和架构的工程工具。**

最容易误解的是：scaling law 看起来像确定性的数学定律，但现实中它依赖数据配方、架构细节、优化器、学习率 schedule 和实验规模。教师最后也强调，wild scaling 很 messy，没有 silver bullet。【字幕】

------

## 3. 第二部分：MiniCPM——用 muP 和 WSD 降低 scaling 成本

### 3.1 问题引入

MiniCPM 是本讲第一个详细案例。PDF 将 MiniCPM 描述为 2024 年来自清华团队的小型高性能语言模型，重点在于它做了细致的 scaling 计算，并使用 muP 稳定和简化 scaling；PDF 也提醒它并不一定是严格意义上的 SOTA，但包含很多 scaling 方面的经验教训。【PDF】

字幕中教师进一步解释：MiniCPM 在当时的 1B 到 2B 参数区间非常有竞争力，虽然今天已有 Gemma 等更强小模型，但 MiniCPM 作为 scaling 案例仍然很有学习价值。【字幕】

### 3.2 MiniCPM 的核心策略

MiniCPM 不是直接训练大量 1.5B 模型来暴力调参，而是先训练一系列更小的模型，建立 scaling ladder，再外推到目标模型。PDF 第 9 页显示 MiniCPM 使用 muP 初始化、固定模型宽高比例，然后逐步扩大整体模型规模；最大 scaling-ladder 模型与最终模型之间大约有 5 倍差距，optimal batch、learning rate、token-to-size ratio 都通过 scaling analysis 拟合。【PDF】

这体现了一个真实训练中的原则：**真正的大模型训练应该尽量“一次成功”，前期小规模实验的目标就是减少大规模训练时的超参数错误。**

### 3.3 muP：让最优学习率尽量跨规模稳定

#### 3.3.1 直觉解释

普通参数化下，模型变宽或变深后，激活值、梯度、更新量的尺度可能变化。这样原来在小模型上合适的学习率，到大模型上可能太大或太小。

muP，即 maximum update parametrization，想做的事情是：**通过调整初始化、残差缩放、输出头缩放以及不同参数的学习率，让模型变大时最优学习率仍然大致保持不变。**

教师在字幕里说，muP 的目标就是希望“optimal learning rate is the same as I scale up and down”。【字幕】

#### 3.3.2 MiniCPM 中的 muP 操作

PDF 第 8 页列出 MiniCPM 应用 tensor program / muP 技术时的具体操作：【PDF】

| 操作                         | 具体含义                                          | 学习重点                                 |
| ---------------------------- | ------------------------------------------------- | ---------------------------------------- |
| Embedding output scaling     | 将 embedding 输出乘以 `scale_emb`                 | 控制输入到后续网络的激活尺度             |
| Residual connection scaling  | 每层残差增量乘以 `scale_depth / sqrt(num_layers)` | 避免层数增加后残差累积过大               |
| Tensor initialization        | 二维张量参数初始化标准差按宽度比例缩放            | 控制权重矩阵输出尺度                     |
| Tensor learning-rate scaling | 对二维张量参数使用按宽度比例调整的学习率          | 这是 muP 中最不像常规训练的部分          |
| LM head scaling              | 对输出 logits 按宽度比例缩放                      | 避免输出层随宽度变化导致概率分布尺度漂移 |

PDF 给出的 MiniCPM 设置示例为：`scale_emb = 12`，`scale_depth = 1.4`，`init_std = 0.1`，`lr = 0.01`。这些数值本身不是普适配方，重要的是背后的原则：不同参数类型可能需要不同的初始化和学习率缩放。

#### 3.3.3 图表解释：最优学习率是否真的稳定？

PDF 第 10 页展示 MiniCPM 的 learning-rate sweep：横轴是 learning rate，纵轴是 loss，每条曲线对应不同模型规模。图中最小 loss 大致落在相似学习率附近，说明在 MiniCPM 设定下，muP 确实让 optimal LR 的漂移变小。【PDF】

字幕中教师评价这是一个“very clean”的成功案例：多种模型规模下，loss minimum 基本落在 $10^{-2}$ 附近；最小模型略有偏移，但整体很接近。【字幕】

**学习提示**：muP 的价值不是让所有超参数都不需要调，而是优先降低最敏感的学习率调参成本。

### 3.4 optimal batch：batch size 仍然需要随目标 loss 变化

即使 muP 稳定了学习率，batch size 仍然不是常数。MiniCPM 对不同模型规模、数据量、batch size 做训练实验。PDF 第 11 页说明：三个模型规模 9M、30M、170M 中，横轴是 batch size，纵轴是数据量或训练进度，颜色表示 loss；每一列竖直点代表固定 batch size 的一条训练曲线，红线试图找出每个数据规模下的最低 loss 点，也就是对应的 optimal batch size。【PDF】

PDF 第 12 页进一步沿用 Kaplan 2020 的 critical batch 思路，把 optimal batch size 与 final loss 画在一起，结论是：**loss 越低，即模型训练得越好、目标越难，optimal batch size 越大，并且呈现近似幂律趋势。**【PDF】

#### 补充说明：为什么 loss 越低 batch 越大？

从优化直觉看，训练早期梯度信号很强，小 batch 也能提供足够方向信息；训练后期模型接近较优区域，梯度噪声相对更显著，增大 batch 可以减少噪声，提高有效训练效率。但 batch 也不能无限大，因为过大 batch 会降低单位 compute 的收益。

### 3.5 WSD learning rate：用学习率 schedule 节省 Chinchilla 实验成本

#### 3.5.1 问题：为什么 Chinchilla 分析很贵？

如果使用 cosine learning-rate schedule，每个训练长度都有不同的 decay 形状。假设你先训练 4M sequences，再想扩展到 8M sequences，不能简单从 4M 终点继续训练，因为 8M run 的 cosine schedule 从一开始就应该不同。PDF 第 13 页因此指出：Chinchilla-style scaling 需要从头训练，不只是 early stop；这会让拟合 scaling law 的成本从近似 $n$ 变成类似 $n^2$。【PDF】

#### 3.5.2 WSD 的结构

WSD 是 warmup-stable-decay：
$$
\eta(t)=
\begin{cases}
\text{warmup from }0\text{ to }\eta_{\max}, & 0 \le t < T_w \\
\eta_{\max}, & T_w \le t < T_s \\
\text{decay from }\eta_{\max}\text{ to a small final LR}, & T_s \le t \le T
\end{cases}
$$
PDF 第 14 页说明：与 cosine 不同，WSD 把学习率拆成 warmup、stable、decay 三段；在 Chinchilla-style 分析中，可以从 stable phase 的末端 checkpoint 重启训练，而不用每次从头训练。【PDF】

字幕中教师进一步解释：warmup 通常是固定步数而不是训练长度比例；stable phase 保持学习率恒定；最后 decay phase 通常占总训练的 10% 到 20%，最后降到最大学习率的约 10% 左右。这样如果想延长训练，只需回滚到 stable checkpoint，继续 stable，再重新 decay，而不是全部重跑。【字幕】

#### 3.5.3 WSD 与 cosine 的关系

PDF 第 15 页显示，WSD 在 stable phase 看起来可能比 cosine 慢，但进入 decay phase 后 loss 会快速下降，最终可以接近 cosine 表现；图上还标注 decay 约 10%。【PDF】

教师补充说，很多人经验上认为 cosine 在一些场景中稍好，但 WSD 在许多情况下“basically as good”，并且更适合重复延长训练、做数据维度 sweep。【字幕】

**工程直觉**：WSD 的价值不只是性能，而是实验复用性。它把“为了不同训练长度不断重跑”变成“从稳定阶段 checkpoint 多次接 decay”，极大降低 scaling experiment 成本。

### 3.6 Chinchilla-type analysis：数据量与模型大小的 tradeoff

MiniCPM 在有了 WSD 后，尝试寻找 optimal data-to-model size ratio。PDF 第 16 页给出 Chinchilla 风格的联合拟合公式：【PDF】
$$
L(N,D)=C_N N^{-\alpha}+C_D D^{-\beta}+L_0
$$
其中：

| 符号           | 含义                                                   |
| -------------- | ------------------------------------------------------ |
| $L(N,D)$       | 给定模型参数量 $N$ 和训练数据量 $D$ 时的 loss          |
| $N$            | 模型参数量，通常可理解为非 embedding 参数量            |
| $D$            | 训练 token 数或数据规模                                |
| $C_N, C_D$     | 分别控制模型规模和数据规模贡献的常数                   |
| $\alpha,\beta$ | 模型规模和数据规模带来的收益指数                       |
| $L_0$          | irreducible loss，即无法靠扩大模型或数据消除的损失下界 |

给定 compute 近似为：
$$
C \approx 6ND
$$
则可以进一步求给定计算预算下的最优模型大小 $N_{\text{opt}}$ 和数据规模 $D_{\text{opt}}$。PDF 第 16 页展示了 MiniCPM 使用 method 1，即 lower envelope，以及 method 3，即 joint fit。PDF 第 18 页说明 MiniCPM 的 joint fit 得到很高的数据-模型比例。【PDF】

字幕中教师提醒：MiniCPM 得到的联合拟合指数与 Chinchilla 原论文差异较大；MiniCPM 论文据此认为应该用比 Chinchilla 更多的文本，但教师表示不确定这是真实结论，还是拟合方式相对原 Chinchilla 有些 strange。【字幕】

### 3.7 MiniCPM 小结

MiniCPM 给本讲贡献了两个核心工具：

1. **muP**：通过参数化和初始化，让 learning rate 尽量跨规模稳定；
2. **WSD**：通过 warmup-stable-decay schedule，让不同训练长度的 scaling experiment 可以复用 checkpoint。

复习时要抓住：MiniCPM 不是重点在“这个模型本身多强”，而是重点在“它展示了一套减少大规模训练调参成本的工程方法”。

------

## 4. 第三部分：DeepSeek——直接拟合 LR / batch 的 scaling law

### 4.1 问题引入

DeepSeek 是第二个详细案例。PDF 第 19 页把 DeepSeek 描述为另一个做了细致 scaling analysis 的 2024 语言模型案例，并提到 7B 和 67B 参数模型在开放模型中表现较强。【PDF】

DeepSeek 与 MiniCPM 的关键区别是：**DeepSeek 不采用 muP 去稳定学习率，而是直接在多个规模上做 grid search，然后拟合 optimal learning rate 和 optimal batch size 随 compute 变化的规律。**

字幕中教师将 DeepSeek 称为 open world 中执行得很好的 scaling analysis 之一，认为它体现了“fit the scaling law”的路线。【字幕】

### 4.2 DeepSeek 的 scaling strategy：batch + LR

PDF 第 20 页写得很直接：DeepSeek 的 scaling strategy 是“不使用 muP，直接估计 optimal batch / LR”。【PDF】

具体做法是：

1. 在小规模 compute 上训练很多模型；
2. 同时 sweep learning rate 和 batch size；
3. 对每个 compute / model setting 找到最低 terminal loss；
4. 收集 near-optimal 点；
5. 在 log-log 坐标中拟合 batch size 和 learning rate 随 compute 的 power law；
6. 外推到大模型训练。

### 4.3 DeepSeek 的公式

PDF 第 21 页给出 DeepSeek 风格的经验拟合：【PDF】
$$
\eta_{\text{opt}} = 0.3118 \cdot C^{-0.1250}
$$
其中：

| 符号                | 含义                        |
| ------------------- | --------------------------- |
| $C$                 | 非 embedding training FLOPs |
| $\eta_{\text{opt}}$ | 最优学习率                  |
| $B_{\text{opt}}$    | 最优 batch size             |

这两个公式表达的趋势是：compute 越大，最优 batch size 越大；最优 learning rate 越小。

### 4.4 图表解释：为什么学习率拟合更可疑？

PDF 第 21 页左图是 batch size scaling curve，右图是 learning rate scaling curve。batch size 的灰点整体更像一条线，而 learning rate 的点更分散。PDF 页脚甚至写着 “Learning rate fit looks a bit questionable”。【PDF】

教师在字幕中也表达类似判断：batch 的拟合看起来合理，learning rate 的线性拟合并不是特别令人信服；如果 grid search 的网格没有覆盖真正最优区域，会有较大 quantization error，导致拟合看起来有些任意。【字幕】

**学习提示**：batch size 通常更容易拟合出稳定趋势；learning rate 更敏感，网格过粗、数据不同、架构不同、schedule 不同都可能影响结论。

### 4.5 DeepSeek 中的 WSD-style learning rate

DeepSeek 也使用 WSD-like schedule 来做 Chinchilla 分析。PDF 第 22 页说明：DeepSeek 使用快速 warmup，加两个 10% decay steps，并且整体表现似乎能匹配 cosine learning rate。【PDF】

字幕中教师指出，DeepSeek 的两个 decay phase 是一个有点 weird 的变体，后续没有特别流行，但重点仍然是：WSD 是有效的，并可以用于复现 Chinchilla-style analysis。【字幕】

### 4.6 IsoFLOP 与最终 loss 预测

PDF 第 23 页显示 DeepSeek 使用 Chinchilla method 2，即 straightforward IsoFLOP-style analysis，来选择模型大小和数据大小的 tradeoff。PDF 第 24 页显示拟合出的 scaling model 能够大致预测最终 7B 和 67B 模型的 loss。【PDF】

字幕中教师总结：这说明 Chinchilla laws 已经被其他团队在较大 compute 范围复现过，因此整体分析方法是 generally sound；你训练小模型、拟合 power law，最终大模型结果可以接近预测值。【字幕】

### 4.7 DeepSeek 小结

DeepSeek 的路线可以总结为：
$$
\text{小规模 grid search} \rightarrow \text{找 near-optimal LR / batch} \rightarrow \text{拟合 scaling law} \rightarrow \text{外推到大模型}
$$
与 MiniCPM 相比，DeepSeek 不试图让学习率不变，而是承认学习率和 batch size 会随规模变化，然后直接拟合变化规律。

复习时要抓住：**MiniCPM 是“让超参数更不变”，DeepSeek 是“让超参数变化可预测”。**

------

## 5. 第四部分：近期模型中的 scaling recipes

### 5.1 Qwen：LR / batch scaling 逐渐成为标准配方

PDF 第 25 页指出，Qwen 2.5 使用 batch 和 hyperparameter scaling fits；Qwen 3 继续使用类似的 LR / batch scaling。PDF 也暗示这些新论文给出的细节比 MiniCPM 和 DeepSeek 少。【PDF】

字幕中教师解释原因：到 Qwen 2.5 / Qwen 3 时，很多模型团队已经默认会做 batch size、learning rate 的 scaling experiment，所以论文里不再详细展开。【字幕】

**学习提示**：当一项技术在论文中变得“不详细”，往往不代表不重要，而是它已经变成业内默认 recipe。

### 5.2 Kimi K2：用 sparsity scaling law 选择 MoE 稀疏度

PDF 第 26 页展示 Kimi K2 的 sparsity scaling law：对于 MoE 模型，团队会在不同 FLOPs 下比较不同稀疏度的 validation loss，从而确定合适的 sparsity level。【PDF】

字幕中教师解释：当模型转向 MoE 时，关键问题变成“应该激活多少 active parameters”。Kimi K2 通过不同 sparsity level 的 scaling experiment 发现，越稀疏在给定 FLOP 下 loss 越好，但收益会递减，于是选择 sparsity 48，因为此后收益开始 diminishing returns。【字幕】

**工程直觉**：dense 模型的 scaling law 主要在 $N,D,C$ 之间做 tradeoff；MoE 模型还要加入“总参数量 vs active parameters vs sparsity”的 tradeoff。

### 5.3 Hunyuan：MoE active parameter ratio

PDF 第 27 页显示 Hunyuan 做了针对 MoE active parameter size 的 IsoFLOP-style scaling，并给出 optimal ratio 约为 96:1，即 data to active parameter ratio。【PDF】

这类分析说明：MoE 的“参数量”不能只看总参数，还要看每个 token 激活多少参数。对训练成本和 loss 来说，active parameters 通常更直接。

### 5.4 LLaMA 3：从 log loss 到 downstream accuracy

PDF 第 28 页展示 LLaMA 3 的 scaling laws：一方面有 IsoFLOP-style scaling，标注了 39:1 的 token-to-model ratio；另一方面展示 compute-to-downstream scaling，即 log probability / normalized NLL 与 benchmark accuracy 之间可能存在 sigmoid-like 关系。【PDF】

字幕中教师提醒，不应把右图的 sigmoid 曲线当作唯一真理，因为有系统偏差；但它说明 log loss 与 downstream accuracy 在许多情形下有紧密联系。【字幕】

**补充说明**：这很重要，因为预训练 scaling law 通常只预测 loss，但用户和 benchmark 更关心 accuracy、pass@k、win rate、instruction following 等下游指标。LLaMA 3 的图是在尝试把“训练损失”连接到“实际任务表现”。

### 5.5 MiniMax-01：用 scaling law 做架构选择

PDF 第 29 页展示 MiniMax-01 的 architecture scaling laws：比较 softmax attention、lightning attention、hybrid-lightning 等架构在不同 compute 下的 loss、model size 与 token 数。【PDF】

字幕中教师解释，MiniMax 的问题是：如果使用 lightning attention、full softmax attention 或 hybrid 架构，随着 scale 变大，哪一种会明显变差？实验结论大致是三者可比，因此 scaling plots 被用来支持实际系统中采用 hybrid architecture。【字幕】

### 5.6 本部分小结

PDF 第 30 页对近期 scaling recipes 做了很清楚的总结：【PDF】

| 模型 / 路线       | scaling 重点                                                 |
| ----------------- | ------------------------------------------------------------ |
| DeepSeek          | 假设多数 Transformer 超参数随 scale 稳定；对 batch / LR 做 scaling analysis；用 IsoFLOP 决定 model sizing；用 piecewise-linear schedule 降低 Chinchilla 成本 |
| MiniCPM           | 用 muP 让 Transformer 与 LR 更 scale-invariant；用 piecewise-linear schedule 获取 Chinchilla method 3 的 curve-fitting 样本 |
| Qwen              | LR / batch scaling，但公开细节较少                           |
| Kimi K2           | MoE sparsity scaling                                         |
| LLaMA 3 / Hunyuan | 主要展示 IsoFLOP scaling                                     |
| MiniMax           | 用 scaling law 支持 architecture choice                      |

这一部分最重要的学习价值是：**scaling law 已经从“预测模型 loss 的研究工具”变成“模型设计决策工具”。**

------

## 6. 第五部分：StepFun——学习率与 batch size 到底该依赖什么？

### 6.1 问题引入

前面 MiniCPM 和 DeepSeek 给出两种路线，但仍有一个更基础的问题：learning rate 和 batch size 的 scaling law 到底应该依赖哪些变量？是 terminal loss？compute？模型参数量？数据量？稀疏度？数据配方？

PDF 第 32 页引入 StepFun 的大规模 scaling law study，核心问题是：当模型规模变化时，如何设置 LR 和 batch parameters？【PDF】

### 6.2 不同观点：loss、compute、N/D

PDF 第 33 页展示了不同团队提出的 LR / batch scaling 公式，包含 OpenAI law、Microsoft law、DeepSeek law、MiniCPM law、Meituan law 和 StepFun law。该页下方总结出三类观点：critical batch 把 batch 看作 loss 的函数；DeepSeek 把 batch 和 LR 看作 compute 的幂律函数；StepFun 则尝试使用更细的变量，例如模型大小 $N$ 和数据大小 $D$。【PDF】

PDF 表格中 StepFun 的经验式大致为：
$$
\eta \approx 1.79N^{-0.713}D^{0.307}
$$
这表达了两个趋势：

1. learning rate 同时依赖模型大小和数据量；
2. batch size 主要依赖数据量 $D$。

### 6.3 方法：纯经验 grid search

StepFun 的做法类似 DeepSeek：训练大量模型，在不同模型规模、不同数据量下，对 learning rate 和 batch size 做高分辨率 grid search，然后观察 loss surface。PDF 第 34 页将其描述为 purely empirical grid search；第 35 页展示 loss over batch / LR 的曲面较平滑，预训练 loss 下 LR 和 batch 的 minimizer 可以较清楚地识别。【PDF】

字幕中教师也强调：这些 contour plot 很有帮助，因为它们显示 hyperparameter landscape 在某些 slice 上近似 smooth and convex；如果曲面很 jagged，那么 grid search 是否能可靠找最优就会很可疑。【字幕】

### 6.4 观察 1：loss 对 LR / batch 大致凸

PDF 第 35 页的主要结论是：对于 pre-training losses，LR / batch 的 minimizer 可以被 cleanly identified。【PDF】

**直觉解释**：如果学习率太小，模型学得慢；如果太大，训练不稳定或 loss 变差。batch size 也类似：太小噪声大、吞吐效率差；太大可能浪费 compute 或导致优化效率下降。中间存在一个较优区域。

### 6.5 观察 2：batch size 主要依赖数据量

PDF 第 36 页显示，在 Chinchilla-style joint scaling 下，batch size 主要依赖 dataset size；不同模型大小的点大致落在同一趋势上。学习率则更复杂：固定模型大小时，数据越多最优学习率越高；模型越大最优学习率越低。但 PDF 也提示，这个 LR 与 $D$ 的正相关可能比较 fragile，尤其在换成 WSD 等 schedule 时可能变化。【PDF】

字幕中教师强调，optimal batch size 似乎只主要依赖总数据量，这是一个非常有趣的 general scaling trend；而 learning rate 的数据依赖更反直觉，也更可能脆弱。【字幕】

### 6.6 观察 3：对 MoE 和数据集有一定迁移，但不是完全不变

PDF 第 37 页展示 StepFun 方法对 MoE 和其他数据集的 generalization。字幕中教师补充：当切换训练数据时，optimal LR 和 optimal batch 会有一些移动，这意味着具体 scaling law 的常数和指数可能依赖数据配方。【字幕】

### 6.7 学习提示

StepFun 部分最重要的结论是：

- batch size 可能主要随数据量 $D$ 增大；
- learning rate 同时受模型大小、数据量、schedule 和 optimizer 影响；
- 所有超参数 scaling law 都不应被当作绝对真理；
- 在自己的数据、架构、optimizer 上，最好至少做局部验证。

------

## 7. 第六部分：优化器 scaling 与 Muon

### 7.1 问题引入

优化器本身也会随规模变化。PDF 第 31 页提出问题：optimizer choices / tuning can be tricky and scale sensitive；我们应该如何选择不同优化器？【PDF】

这部分的核心警告是：**小模型 benchmark 上看起来很强的优化器，不一定在大模型上仍然保持同样收益。**

### 7.2 优化器调参常见问题

PDF 第 39 页指出，不同优化器可能需要不同超参数；如果 AdamW 的学习率没有调好，可能会显得比别的算法差很多；如果 Lion 的 weight decay 没调好，也会误判其效果。【PDF】

字幕中教师解释：超参数之间确实会交互，但完整 grid search 维度会指数爆炸，所以实践中通常优先 grid 最敏感的 learning rate 和 batch size，然后对 weight decay 等做单变量 sweep。【字幕】

### 7.3 scale dependence：compute 与 Chinchilla ratio 都要看

PDF 第 40 页提醒，算法开发时要检查两个维度：

1. **compute scale**：模型和数据一起变大时，收益是否保持；
2. **Chinchilla ratio**：数据量与参数量的比例变化时，算法是否仍然有效。【PDF】

字幕中教师强调，Chinchilla ratio 是很大的 confounder。有些算法可能在 over-parameterized 情况下更好，有些可能在大量数据 per parameter 的情况下更好。因此只看模型大小轴是不够的。【字幕】

### 7.4 scaling trend 也可能突然失效

PDF 第 41 页展示一个例子：看起来很好的 scaling trend 在更大规模上突然 blow up。图中例子来自 Cautious AdamC + sqrt batch-size scaling of learning rates，后来需要更小心的参数化、scaling 和 optimizer 修改来修复。【PDF】

这说明：即使 scaling curve 在多个数量级内很好，也不保证继续外推不会失败。

### 7.5 Muon：矩阵参数优化器

#### 7.5.1 直觉解释

普通 optimizer 通常把参数当成一组坐标来处理。但 Transformer 中很多参数是矩阵：attention projection、MLP weight 等。Muon 的核心想法是：**矩阵参数应该用矩阵的方式优化，而不是完全当成独立坐标。**

PDF 第 42 页写道：Muon 是 optimizer for matrix-valued parameters；NewtonSchultz 近似把矩阵 $B_t=USV^\top$ 正交化为 $UV^\top$。【PDF】

#### 7.5.2 Muon 更新公式

Muon 的简化算法可以写成：
$$
G_t = \nabla_\theta \ell(\theta_{t-1})
$$
如果对 $B_t$ 做奇异值分解：
$$
B_t = U S V^\top
$$
Muon 理想上将所有奇异值归一化，得到：
$$
O_t = U V^\top
$$
然后更新参数：
$$
\theta_t=\theta_{t-1}-\eta O_t
$$
其中：

| 符号         | 含义                                 |
| ------------ | ------------------------------------ |
| $G_t$        | 当前梯度                             |
| $B_t$        | 带 momentum 的更新矩阵               |
| $\mu$        | momentum 系数                        |
| $U,S,V^\top$ | SVD 的左奇异向量、奇异值、右奇异向量 |
| $O_t$        | 正交化后的更新方向                   |
| $\eta$       | 学习率                               |

字幕中教师解释：如果某些奇异值很大，就缩回 unit；如果很小，就放大到 unit。因此 Muon 类似在 spectral norm 意义下做“归一化更新”。这只对矩阵有意义，因此 vector-valued 参数通常仍可使用 AdamW。Newton-Schulz 不是直接做昂贵 SVD，而是用矩阵乘法近似正交化，这更适合 GPU。【字幕】

### 7.6 Muon 是否真的适合大规模？

PDF 第 43 页把三个证据放在一起：NanoGPT speedrun 很小规模、scaling study、更大模型 Kimi K2；页脚总结是 scaling gains 很难测量，但 Muon 显然 “works” at scale。【PDF】

字幕中教师的态度很谨慎：Muon 在小规模 benchmark 上收益明显；一些 scaling study 认为收益随规模增大而下降；但 Kimi K2 用 Muon 完整训练，说明 Muon 在大规模上至少是可行的。是否比 AdamW 更好，由于 Kimi K2 没有大规模 ablation，还不能下定论。【字幕】

### 7.7 学习提示

Muon 部分复习时要抓住三点：

1. Muon 是 matrix optimizer，不是普通逐坐标 optimizer；
2. Newton-Schulz 的作用是用矩阵乘法近似正交化，避免直接 SVD；
3. 小规模收益、scaling study、真实大模型训练之间可能给出不同信号。

------

## 8. 第七部分：muP 深入理解——为什么参数化能稳定学习率？

### 8.1 问题引入

PDF 第 44 页进入 maximum update parametrization 的深入部分：muP 的吸引力在于 scale-invariant hyperparameter tuning，如果成立，小模型调出来的学习率就能迁移到大模型。【PDF】

教师在字幕中说，muP 有点神秘，不同论文和实现并不完全一致，但它有一组共同核心思想，并且在实践中通常有效。【字幕】

### 8.2 CerebrasGPT：muP 的经验支持

PDF 第 45 页显示 CerebrasGPT 训练了 0.1B 到 13B 模型，采用 Chinchilla recipe，并发现使用 muP parametrization 让 scaling 更稳定。【PDF】

字幕中教师补充，CerebrasGPT 的 muP scaling law predictions 更接近实际模型，而 non-muP 模型的 loss prediction 波动更大。【字幕】

### 8.3 muP 的两个核心不变量

PDF 第 46 页给出 muP 的核心假设。随着网络宽度 $n_l$ 变化，应满足：【PDF】
$$
\text{A1: activations at initialization should remain } \Theta(1)
$$
更具体地说，如果每个 individual activation 是 $\Theta(1)$，那么整层 activation vector 的范数应为：
$$
\|h_l\|_2 = \Theta(\sqrt{n_l})
$$

#### 直觉解释

- A1 说：模型变宽时，初始化后的激活值不应爆炸，也不应消失。
- A2 说：训练一步后，模型内部表示应该有非零、非爆炸的改变，这样网络能学习 feature。
- 这和 neural tangent kernel 极限不同；NTK 中宽度趋于无穷时 feature 变化可能消失，而 muP 希望保留 feature learning。

字幕中教师明确解释：feature learning 指做一步梯度更新后，网络中的东西发生显著变化，而且这个变化量应当是宽度的固定函数；如果变化随宽度消失，就不是我们想要的大网络学习行为。【字幕】

### 8.4 condition A1 推导：如何选择初始化尺度？

考虑一个深线性网络：
$$
h_l = W_l h_{l-1}
$$
设：
$$
W_l \sim \mathcal{N}(0,\sigma^2 I_{n_l\times n_{l-1}})
$$
随机矩阵的 operator norm 大致满足：
$$
\|W_l\|_* \approx \sigma(\sqrt{n_{l-1}}+\sqrt{n_l})
$$
于是：
$$
\|h_l\|_2 \approx \|W_l\|_* \|h_{l-1}\|_2
$$
为了让 $\|h_l\|_2=\Theta(\sqrt{n_l})$，PDF 第 47 页选择：【PDF】
$$
\sigma =
\frac{\sqrt{n_l}}{\sqrt{n_{l-1}}}
\left(\sqrt{n_l}+\sqrt{n_{l-1}}\right)^{-1}
$$
它可以理解为：
$$
\sigma = \Theta\left(
\frac{1}{\sqrt{n_{l-1}}}
\min\left(1,\sqrt{\frac{n_l}{n_{l-1}}}\right)
\right)
$$
如果假设上一层：
$$
\|h_{l-1}\|_2=\Theta(\sqrt{n_{l-1}})
$$
则：
$$
\|W_l\|_* \approx \sqrt{\frac{n_l}{n_{l-1}}}
$$
从而：
$$
\|h_l\|_2
\approx
\sqrt{\frac{n_l}{n_{l-1}}}
\cdot
\sqrt{n_{l-1}}
=
\sqrt{n_l}
$$
这说明初始化尺度选择的目标是：**每层输出 activation vector 的范数随层宽以合理方式增长，而单个 activation 的尺度保持 $\Theta(1)$。**

### 8.5 condition A2 推导：如何选择学习率尺度？

现在考虑权重更新。PDF 第 48 页给出 SGD 在线性层上的更新形式：【PDF】
$$
\Delta W_l = -\eta_l \nabla_{h_l}\ell \, h_{l-1}^{\top}
$$
这可以理解为“loss 对当前层输出的梯度”与“上一层 activation”的外积，因此是 rank-one update。

激活变化为：
$$
\Delta h_l
=
W_l\Delta h_{l-1}
+
\Delta W_l(h_{l-1}+\Delta h_{l-1})
$$
这里有两类贡献：

1. 上一层 activation 已经变了，通过旧权重 $W_l$ 传到当前层；
2. 当前层权重本身变了，通过 $\Delta W_l$ 影响当前层输出。

为了满足 A2，我们希望：
$$
\|\Delta h_l\|_2 = \Theta(\sqrt{n_l})
$$
这要求：
$$
\|\Delta W_l\|_* \sqrt{n_{l-1}} = \Theta(\sqrt{n_l})
$$
因此：
$$
\|\Delta W_l\|_* = \Theta\left(\sqrt{\frac{n_l}{n_{l-1}}}\right)
$$
PDF 第 49 页进一步通过 loss update 的尺度关系，得到 SGD 下学习率应满足：【PDF】
$$
\eta_l = \Theta\left(\frac{n_l}{n_{l-1}}\right)
$$
而对于 Adam，PDF 的 recap 指出会出现不同缩放，常见结论是对 Adam 使用约 $1/n_{l-1}$ 的 layer-specific learning-rate scaling。【PDF】

### 8.6 muP mini recap

PDF 第 50 页总结了 baby muP 的规则：【PDF】

| 项目         | muP 风格                                                     | 标准参数化                 |
| ------------ | ------------------------------------------------------------ | -------------------------- |
| 初始化标准差 | $\Theta\left(\frac{1}{\sqrt{n_{l-1}}}\min(1,\sqrt{n_l/n_{l-1}})\right)$ | $\frac{1}{\sqrt{n_{l-1}}}$ |
| SGD 学习率   | $\Theta(n_l/n_{l-1})$                                        | $\Theta(1)$                |
| Adam 学习率  | 常出现与 fan-in 相关的 layer-specific 缩放                   | 通常设为常数               |

**学习提示**：muP 不是单一公式，而是一套“让激活和更新在宽度变化下保持合理尺度”的参数化原则。

### 8.7 muP 对现代语言模型是否鲁棒？

PDF 第 53 页列出现代 LM 中偏离 muP 理论的组件：SwiGLU、squared ReLU、不同 batch size、zero attention initialization、RMSNorm gains、Lion 等 exotic optimizers、regularizers。【PDF】

PDF 第 54 页指出，RMSNorm learnable gains 会破坏 muP transfer，但这些 gains 可以在较小性能损失下移除。PDF 第 56 页指出 strong weight decay，尤其是 0.1 这种强 weight decay，可能是显著的 muP failure case。【PDF】

字幕中教师也总结：大多数现实偏差不一定会破坏 muP，但 learnable RMSNorm gain、Lion 这种 sign-gradient optimizer、较大的 decoupled weight decay 都可能造成 muP failure。【字幕】

### 8.8 muP 是否有用？

PDF 第 57 页结论是：muP 总体上似乎有用，因为 standard parametrization 更不稳定；当前证据显示 muP parametrization / initialization 可能更容易调参。【PDF】

字幕中教师最后给出谨慎判断：如果你想稳定 learning rate，muP 很有用；它是控制 hyperparameter drift 的工具之一，但不是唯一正确答案，也不是已经完全解决的问题。【字幕】

------

## 9. Python 代码示例详解

### 9.1 本讲是否有 Python 文件？

本讲没有看到对应的 `lecture_11.py`。GitHub 仓库说明可执行讲义命名为 `lecture_XX.py`，不可执行讲义命名为 `lecture_XX.pdf`；仓库文件列表显示 Lecture 10 是 `lecture_10.py`，Lecture 11 是 `lecture_11.pdf`，Lecture 12 是 `lecture_12.py`。因此，本讲主要是 PDF 讲义与图表分析，而不是 Python 代码讲解。

### 9.2 本讲中最接近“代码”的算法：Muon 伪代码

虽然没有 Python 文件，但 PDF 第 42 页给出 Muon 的算法伪代码。它的功能可以按代码逻辑理解为：

| 步骤          | 伪代码含义                                         | 理论作用                       |
| ------------- | -------------------------------------------------- | ------------------------------ |
| 初始化        | $B_0 \leftarrow 0$                                 | 初始化 momentum buffer         |
| 计算梯度      | $G_t \leftarrow \nabla_{\theta}\ell(\theta_{t-1})$ | 得到当前 loss 对参数的梯度     |
| momentum 更新 | $B_t \leftarrow \mu B_{t-1}+G_t$                   | 平滑梯度方向                   |
| 矩阵正交化    | $O_t \leftarrow \text{NewtonSchulz5}(B_t)$         | 近似把矩阵更新方向正交化       |
| 参数更新      | $\theta_t \leftarrow \theta_{t-1}-\eta O_t$        | 使用正交化后的方向更新矩阵参数 |

### 9.3 WSD schedule 的伪代码理解

WSD 虽然不是代码，但很容易写成训练 loop 的逻辑：

```
for step in training_steps:
    if step < warmup_steps:
        lr = linear_warmup(step)
    elif step < stable_end:
        lr = max_lr
    else:
        lr = decay_schedule(step)

    optimizer.step(lr)

    if step is in stable_phase:
        save_checkpoint_for_possible_reuse()
```

这段伪代码对应课程中的工程重点：stable phase 的 checkpoint 可以复用，后续做不同训练长度或数据量 sweep 时，只需从 checkpoint 继续训练并重新 decay。

### 9.4 代码 / 算法与课程知识点对应表

| 课程知识点                  | 对应算法 / 变量 / 片段                 | 功能                             | 对应理论原理                           | 在课程中的作用                              |
| --------------------------- | -------------------------------------- | -------------------------------- | -------------------------------------- | ------------------------------------------- |
| WSD learning rate           | warmup / stable / decay                | 降低不同训练长度实验的重复成本   | 学习率 schedule 与 checkpoint reuse    | 支撑 MiniCPM / DeepSeek 的 Chinchilla 分析  |
| DeepSeek LR / batch scaling | $\eta_{\text{opt}}, B_{\text{opt}}, C$ | 用小规模实验拟合大规模超参数     | 幂律外推                               | 代表“拟合超参数变化”的路线                  |
| StepFun grid search         | LR × batch contour                     | 找 loss surface 的最优点         | 经验优化曲面                           | 观察 batch 和 LR 的 scaling trend           |
| Muon                        | $B_t, O_t, \text{NewtonSchulz}$        | 对矩阵参数做谱意义上的归一化更新 | 矩阵正交化 / spectral norm             | 展示优化器也有 scale dependence             |
| muP 初始化                  | $\sigma$, fan-in, fan-out              | 控制 activation scale            | A1：初始化激活保持 $\Theta(1)$         | 解释为什么参数化影响 learning-rate transfer |
| muP 学习率                  | $\eta_l$                               | 控制 update scale                | A2：一步更新后激活变化保持 $\Theta(1)$ | 解释 scale-invariant hyperparameter tuning  |

------

## 10. PDF / 幻灯片 / 图表解析

### 10.1 主要图表总览

| 页面    | 展示内容                                                   | 支持的课程结论                                           |
| ------- | ---------------------------------------------------------- | -------------------------------------------------------- |
| p2      | 本讲三大问题                                               | scaling law 要被放到真实训练实践中检验                   |
| p4      | initialization、optimizer、LR / batch 的 scale sensitivity | 超参数和优化器随规模变化敏感                             |
| p8      | MiniCPM 的 muP 操作表                                      | muP 通过多种缩放控制激活和学习率                         |
| p10     | MiniCPM LR sweep                                           | muP 可让最优 LR 在不同规模下相对稳定                     |
| p11-p12 | optimal batch 实验                                         | batch size 随 target loss / 数据规模呈规律变化           |
| p14-p15 | WSD vs cosine                                              | WSD 可复用 stable checkpoint，并在 decay 后追回 loss     |
| p20-p21 | DeepSeek grid search 与 LR / batch scaling                 | 直接拟合超参数 scaling law 是另一条路线                  |
| p23-p24 | DeepSeek IsoFLOP 与 final loss prediction                  | 小规模 scaling 可以预测大模型 loss                       |
| p26-p29 | Kimi K2、Hunyuan、LLaMA 3、MiniMax                         | scaling law 可用于 MoE 稀疏度、downstream 预测和架构选择 |
| p33-p37 | StepFun 超参数 scaling                                     | batch 主要依赖数据量，LR 依赖更复杂                      |
| p38-p43 | 优化器 scaling 与 Muon                                     | 优化器收益可能随规模变化                                 |
| p46-p50 | muP 理论推导                                               | 从 activation 和 update 尺度推出初始化和学习率缩放       |
| p53-p57 | muP 鲁棒性                                                 | RMSNorm gain、Lion、strong weight decay 等可能破坏 muP   |
| p58     | recap                                                      | scaling in practice 的挑战与解决方案                     |

### 10.2 关键图表详细解释

#### 图表 1：MiniCPM LR sweep

这张图想回答：使用 muP 后，不同模型规模的最优 learning rate 是否仍然变化很大？

横轴是 learning rate，纵轴是 loss，不同曲线对应不同模型大小。理想情况下，如果 muP 成功，各曲线最低点应出现在相近 learning rate。图中确实大致如此，因此支持“muP 稳定 LR”的结论。【PDF】

容易误解的地方：这不代表所有架构、所有 optimizer、所有数据配方中 muP 都会让 LR 完全不变，只说明在 MiniCPM 的实验设置中效果较好。

#### 图表 2：optimal batch size vs final loss

这组图想回答：batch size 是否也能通过 scaling law 预测？

PDF 第 11 页中，横轴是 batch size，纵轴是数据量或训练进度，颜色表示 loss；红线表示每个数据规模下的最优 batch。PDF 第 12 页进一步把 optimal batch 与 final loss 对应起来，显示 loss 越低，optimal batch 越大。【PDF】

容易误解的地方：optimal batch 不是“越大越好”，而是在目标 loss、模型规模、数据量和系统效率之间的最优折中。

#### 图表 3：WSD learning rate

这张图想回答：如何减少 Chinchilla-style scaling experiment 的重复训练？

横轴是 iteration 或 tokens，纵轴是 learning rate / loss。WSD 在 warmup 后进入长 stable phase，最后快速 decay。它允许从 stable checkpoint 重启，因此可以用同一长训练 run 生成不同训练长度的数据点。【PDF】

容易误解的地方：WSD 在 stable phase 的 loss 可能看起来落后 cosine，但进入 decay 后会快速下降，因此不能只看中途 loss 判断 schedule 好坏。

#### 图表 4：DeepSeek LR / batch scaling

这张图想回答：不用 muP 时，能否直接拟合 optimal LR / batch？

左图 batch size 随 compute 增大而增大，拟合较清楚；右图 learning rate 随 compute 增大而减小，但点更分散，教师与 PDF 都认为 LR fit 有些 questionable。【PDF】

容易误解的地方：幂律公式看起来精确，但它来自有限 grid search；如果网格过粗或数据变了，指数和常数可能变化。

#### 图表 5：StepFun contour

这张图想回答：LR / batch 超参数空间是否足够平滑，使得 grid search 有意义？

如果 loss surface 是平滑凸状，那么最优点可以较可靠地识别；如果非常 jagged，则小规模拟合很难外推。PDF 第 35 页显示 StepFun 的 loss surface 在实验范围内较平滑。【PDF】

#### 图表 6：Muon scaling

这张图想回答：Muon 在小规模和大规模之间的收益是否一致？

左侧 NanoGPT speedrun 显示 Muon 小规模表现好；中间 scaling study 显示相对 AdamW 的 speedup 随模型规模增大而下降；右侧 Kimi K2 训练曲线说明 Muon 至少可以用于大规模训练。【PDF】

正确理解是：Muon “能 work at scale” 不等于“已证明大规模优于 AdamW”。

------

## 11. 关键公式与推导汇总

| 公式                                                         | 符号解释                                             | 适用场景                       | 直觉解释                                      | 代码 / 例子对应             |
| ------------------------------------------------------------ | ---------------------------------------------------- | ------------------------------ | --------------------------------------------- | --------------------------- |
| $L(N,D)=C_NN^{-\alpha}+C_DD^{-\beta}+L_0$                    | $N$：模型参数；$D$：数据量；$\alpha,\beta$：收益指数 | Chinchilla-style joint fit     | loss 来自模型不足、数据不足和不可约误差三部分 | MiniCPM Chinchilla method 3 |
| $C\approx 6ND$                                               | $C$：训练 compute                                    | dense Transformer 训练成本估算 | 参数越多、token 越多，compute 近似线性增加    | IsoFLOP 分析                |
| $\eta_{\text{opt}}=0.3118C^{-0.1250}$                        | $\eta$：学习率；$C$：FLOPs                           | DeepSeek LR scaling            | compute 越大，学习率越小                      | DeepSeek grid search        |
| $B_{\text{opt}}=0.2920C^{0.3271}$                            | $B$：batch size                                      | DeepSeek batch scaling         | compute 越大，最优 batch 越大                 | DeepSeek batch fit          |
| $\eta \approx 1.79N^{-0.713}D^{0.307}$                       | $N$：模型大小；$D$：数据量                           | StepFun LR scaling             | 模型越大 LR 越小；数据越多 LR 可能越大        | StepFun empirical law       |
| $B \approx 0.58D^{0.571}$                                    | $D$：数据量                                          | StepFun batch scaling          | batch 主要随训练数据量增长                    | StepFun observation         |
| $B_t=\mu B_{t-1}+G_t$                                        | $B_t$：momentum update；$G_t$：梯度                  | Muon                           | 平滑当前梯度                                  | Muon algorithm              |
| $B_t=USV^\top \rightarrow O_t=UV^\top$                       | $S$：奇异值                                          | Muon matrix update             | 将矩阵更新方向正交化                          | Newton-Schulz approximation |
| $h_l=W_lh_{l-1}$                                             | $h_l$：第 $l$ 层激活                                 | muP 推导                       | 研究权重尺度如何影响激活尺度                  | condition A1                |
| $\sigma=\Theta\left(\frac{1}{\sqrt{n_{l-1}}}\min(1,\sqrt{n_l/n_{l-1}})\right)$ | $\sigma$：初始化标准差                               | muP 初始化                     | 让单个 activation 保持 $\Theta(1)$            | muP init                    |
| $\Delta W_l=-\eta_l\nabla_{h_l}\ell h_{l-1}^\top$            | $\Delta W_l$：权重更新                               | muP A2                         | 线性层 SGD 更新是梯度与激活的外积             | condition A2                |
| $\eta_l=\Theta(n_l/n_{l-1})$                                 | $\eta_l$：层学习率                                   | muP SGD learning-rate scaling  | 控制一步更新后 activation change 的尺度       | muP LR                      |

------

## 12. 概念对比与易混淆点

| 易混淆概念                                 | 区别                                                         | 联系                                       | 常见误解                               | 正确理解                                                     |
| ------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------ | -------------------------------------- | ------------------------------------------------------------ |
| muP vs Muon                                | muP 是参数化 / 初始化 / LR scaling；Muon 是优化器            | 都试图控制 scale 下的更新行为              | 名字相似就以为是一类方法               | muP 解决超参数跨规模迁移，Muon 解决矩阵参数更新方式          |
| WSD vs cosine                              | WSD 有 stable phase；cosine 从头到尾依赖总训练长度           | 都是 learning-rate schedule                | WSD 一定比 cosine 好                   | WSD 的优势主要是 checkpoint 复用和 scaling experiment 成本   |
| optimal batch vs global batch              | optimal batch 是理论 / 经验最优；global batch 是实际训练设置 | 实际 global batch 应尽量靠近 optimal batch | batch 越大越好                         | batch 受优化效率、通信成本、收敛速度共同限制                 |
| LR stability vs LR scaling law             | 前者让 LR 尽量不变；后者拟合 LR 如何变化                     | 都是控制 LR 漂移                           | 两条路线互斥                           | 实际可以混用：先用更稳参数化，再做局部 scaling fit           |
| loss scaling vs downstream scaling         | loss 是预训练目标；downstream 是任务表现                     | loss 通常与下游能力相关                    | loss 降低一定等价于所有 benchmark 提升 | 关系可能是非线性的，并受评估集和 post-training 影响          |
| dense 参数量 vs MoE active 参数量          | dense 每 token 激活大部分参数；MoE 只激活部分专家            | 都影响训练 loss 和 compute                 | MoE 只看总参数量                       | scaling 时更应关注 active parameters 和 sparsity             |
| 小规模 optimizer speedup vs 大规模 speedup | 小 benchmark 中收益可能很大；大规模可能缩小                  | 小规模实验仍能产生想法                     | 小规模赢就一定大规模赢                 | optimizer 是 scale-sensitive，必须检查 compute 与 Chinchilla ratio |
| scaling law vs 物理定律                    | scaling law 是经验拟合；物理定律有更强普适性                 | 都可能呈现幂律                             | 拟合直线就能无限外推                   | scaling in the wild 有艺术性，需要验证和风险控制             |

------

## 13. 学习拓展与发散理解

### 13.1 与大模型训练实践的关系

#### 数据准备

本讲说明，数据规模 $D$ 不只是影响 loss，也可能影响 optimal batch 和 learning rate。StepFun 的结果尤其强调 batch size 主要依赖 dataset size。对实际训练来说，数据配方变化后，不能假设原来的 LR / batch scaling 一定继续有效。

#### 模型训练

MiniCPM 和 DeepSeek 分别代表两种训练策略：一种通过 muP 减少超参数漂移，一种通过 small-scale grid search 拟合漂移规律。实际团队常常需要结合两者：先选一个稳定参数化，再用小规模实验校准学习率、batch、weight decay 等。

#### 优化策略

WSD 说明 schedule 本身也可以是节省 compute 的工具。它不是单纯为了更低 loss，而是为了让 scaling experiment 能复用中间 checkpoint。

#### 训练效率

Muon 说明优化器可能带来训练速度收益，但这种收益必须在 scale 上验证。矩阵正交化、Newton-Schulz、矩阵乘法友好性都体现了算法和系统效率之间的耦合。

#### 架构选择

Kimi K2、Hunyuan、MiniMax 的例子说明 scaling law 可以帮助选择 MoE 稀疏度、active parameter ratio、attention 变体和 hybrid architecture。也就是说，scaling law 不只回答“模型多大”，还回答“模型结构该怎么选”。

### 13.2 与其他 lecture 的联系

Lecture 9 讲 classical scaling laws，本讲把这些方法推进到真实模型训练。Lecture 10 是 inference，本讲中 LLaMA 3 的 compute-to-downstream scaling 也为后续 evaluation / inference 成本讨论提供背景。Lecture 12 evaluation 会更关心下游指标如何衡量模型能力；Lecture 13 和 14 data 会进一步影响本讲提到的数据配方和 $D$ 的作用；Lecture 15 之后的 post-training 则会带来新的 scaling 问题。

课堂中有学生问 post-training 会如何改变这些 scaling 分析。教师回答说，这仍是开放问题，目前没有很好的 fully integrated scaling 方法能把 post-training 完整纳入；post-training 有时会改变 pre-training 应该如何做。【字幕】

### 13.3 面试 / 考试 / 作业可能考察角度

| 问题                                            | 考察点                       | 回答思路                                                     |
| ----------------------------------------------- | ---------------------------- | ------------------------------------------------------------ |
| MiniCPM 和 DeepSeek 的 scaling 策略有什么区别？ | 两条路线对比                 | MiniCPM 用 muP 稳定 LR，DeepSeek 直接拟合 LR / batch scaling |
| 为什么 WSD 能降低 Chinchilla 分析成本？         | schedule 与 checkpoint reuse | stable phase 可复用，只需多次 decay                          |
| 为什么 batch size 可能随 loss 降低而增大？      | critical batch 直觉          | 后期梯度噪声相对更重要，大 batch 更有效                      |
| DeepSeek 的 learning-rate fit 为什么可疑？      | grid search 与拟合误差       | LR 点分散，网格过粗会有 quantization error                   |
| StepFun 发现 batch size 主要依赖什么？          | 超参数 scaling 变量          | 主要依赖 dataset size $D$                                    |
| Muon 与 AdamW 的根本区别是什么？                | optimizer 类型               | Muon 对矩阵参数做谱归一化 / 正交化，AdamW 是坐标级 adaptive optimizer |
| 为什么 Muon 小规模好不代表大规模一定好？        | scale dependence             | optimizer gains 会随 compute 和 Chinchilla ratio 变化        |
| muP 的 A1 / A2 分别是什么意思？                 | muP 理论基础                 | A1 控制初始化激活，A2 控制一步更新后的激活变化               |
| RMSNorm gain 为什么可能破坏 muP？               | 现实架构偏离理论             | learnable gain 引入额外尺度自由度，影响 transfer             |
| scaling law 为什么既像科学又像艺术？            | 实践判断                     | 曲线拟合有规律，但外推依赖数据、架构、optimizer 和 schedule  |

------

## 14. 课程复习总结

### 14.1 本讲核心思想

本讲的主线是：经典 scaling laws 已经从理论和论文图表进入真实大模型工程，但真实训练中最难的不只是拟合 loss，而是控制随规模漂移的超参数、优化器和架构选择。MiniCPM 通过 muP 和 WSD 尝试让学习率稳定、降低实验成本；DeepSeek 通过大量小规模 grid search 直接拟合 learning rate 和 batch size；近期 Qwen、Kimi K2、Hunyuan、LLaMA 3、MiniMax 等模型则展示 scaling law 如何服务 MoE 稀疏度、active parameter、downstream prediction 和 architecture decision。最后，muP 的推导说明参数化本身可以通过控制激活和更新尺度来改善跨规模调参，但它仍然不是银弹。

### 14.2 必须掌握的概念

- Chinchilla-style scaling
- IsoFLOP analysis
- optimal batch / critical batch
- learning-rate scaling
- WSD learning-rate schedule
- muP / maximum update parametrization
- feature learning
- hyperparameter drift
- StepFun LR / batch empirical scaling
- optimizer scale dependence
- Muon matrix optimizer
- MoE sparsity scaling
- active parameters
- compute-to-downstream scaling

### 14.3 必须理解的公式

最重要的公式包括：
$$
L(N,D)=C_NN^{-\alpha}+C_DD^{-\beta}+L_0
$$
它描述模型规模和数据规模对 loss 的共同影响。
$$
C\approx 6ND
$$
它是 dense Transformer 训练 compute 的常见近似。
$$
\eta_{\text{opt}}=0.3118C^{-0.1250},\quad B_{\text{opt}}=0.2920C^{0.3271}
$$
它代表 DeepSeek 直接拟合超参数 scaling law 的路线。
$$
B_t=USV^\top \rightarrow UV^\top
$$
它是 Muon 对矩阵更新方向做正交化的核心思想。
$$
\text{A1: }h_l\text{ at init remains }\Theta(1),\quad
\text{A2: }\Delta h_l\text{ after one step remains }\Theta(1)
$$
它是 muP 推导的两个基础不变量。

### 14.4 必须看懂的代码 / 算法

本讲没有对应 Python lecture 文件，最重要的是看懂三个“算法级”流程：

| 算法 / 流程                    | 学习重点                                                     |
| ------------------------------ | ------------------------------------------------------------ |
| WSD schedule                   | 为什么 stable checkpoint 可复用，如何降低 Chinchilla sweep 成本 |
| DeepSeek / StepFun grid search | 如何从小规模实验找 LR / batch minima，再拟合 scaling law     |
| Muon optimizer                 | 如何对矩阵参数做 momentum + Newton-Schulz 正交化更新         |

### 14.5 一句话总结

**Lecture 11 的核心是：大模型 scaling 不只是画 power law 曲线，而是在真实训练中用小规模实验、稳定参数化、学习率 schedule 和优化器分析，尽可能降低大规模训练的超参数风险。**