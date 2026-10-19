# Lecture 03 课程报告：现代语言模型架构与超参数选择的工程逻辑

## 0. 资料来源与改写说明

本报告是对你上传的原始课程报告 `lec3.md` 的系统性改写与学习化增强，原报告标题为 **Lecture 03 课程报告：LM Architecture and Hyperparameters**。本次改写保留原报告的课程顺序、核心知识点、公式、代码截图解释、PDF 图表说明和引用标记。 

根据 Stanford CS336 官方课程页，Spring 2025 的 Lecture 03 是 **“Architectures, hyperparameters (Tatsu)”**，课程材料为 lecture 3 PDF；GitHub 仓库当前也显示该讲对应文件为 `lecture_03.pdf`，而不是独立的 `lecture_03.py`。

需要说明的是：本次对话中我能直接读取的是你上传的原始报告 `lec3.md`，而不是单独的字幕文件或 PDF 文件。因此，正文中出现的 **【字幕†source】**、**【PDF†source】**、**【PDF/代码截图†source】** 是对原报告已有引用格式的保留与整理；我不会把新增的解释性内容伪装成课程原文。凡是为了帮助理解而补充的内容，会用 **补充说明**、**延伸理解**、**工程直觉** 或 **学习提示** 标明。

本报告的目标不是重新生成一份百科式总结，而是把 Lecture 03 改写成一份可长期复习的中文学习讲义。你可以把它理解为：

> 以原报告为事实基础，以课程授课顺序为主线，把“现代语言模型为什么这样设计”讲清楚。

------

## 1. 学习导读

### 1.1 本讲要解决的核心问题

这一讲要解决的问题不是“Transformer 的基本公式是什么”，而是更贴近真实训练场景的问题：

> 当我们真的要实现并训练一个现代语言模型时，面对 LayerNorm 放在哪里、用 RMSNorm 还是 LayerNorm、FFN 用 ReLU/GeLU 还是 SwiGLU、位置编码用什么、head dimension 怎么设、词表多大、是否使用 dropout、怎样避免训练崩溃、怎样降低推理 KV cache 成本等大量设计选择时，应该采用哪些默认方案？为什么？

授课者开场强调，本讲采取的是一种 **survey lens**：不是试图从纯理论推导出唯一最优架构，而是观察现代大模型和技术报告中已经形成共识的选择。换句话说，这一讲更像是“现代 LM 架构经验规律课”，而不是“Transformer 基础定义课”。【字幕†source】

### 1.2 本讲在 CS336 课程中的位置

CS336 的总体目标是带学生从零理解并构建语言模型，涵盖预训练数据、Transformer 构建、训练、评估和部署等环节；官方课程介绍也说明课程会带学生走完整个语言模型创建过程。

Lecture 03 位于课程早期。前两讲已经介绍课程概览、tokenization、PyTorch / einops、FLOPs、memory、arithmetic intensity 等基础；本讲接着进入 **模型架构与超参数**，为后续实现标准 Transformer 语言模型打基础。官方课程页也显示 Assignment 1 要实现 tokenizer、model architecture 和 optimizer 等组成部分。

从课程主线看，本讲承上启下：

| 课程位置           | 已经铺垫的内容                                     | 本讲接上的问题                                     | 后续会继续用到的思想                                 |
| ------------------ | -------------------------------------------------- | -------------------------------------------------- | ---------------------------------------------------- |
| Lecture 01-02 之后 | tokenization、PyTorch、FLOPs、memory、GPU 计算直觉 | 如何选择现代 Transformer 架构默认值                | 系统优化、并行训练、scaling laws、inference          |
| Assignment 1 附近  | 要自己实现 Transformer LM                          | 为什么现代实现常用 pre-norm、RMSNorm、RoPE、SwiGLU | 后续所有模型训练与系统分析都会默认这些设计           |
| 后续 lecture 之前  | 已知道 Transformer 基本组件                        | 开始理解架构选择背后的稳定性与效率取舍             | MoE、attention alternatives、inference、long context |

### 1.3 本讲知识结构

| 模块                        | 解决的问题                                       | 核心概念                                                  | 与后续内容的关系                       |
| --------------------------- | ------------------------------------------------ | --------------------------------------------------------- | -------------------------------------- |
| 原始 Transformer 到现代变体 | 为什么今天的 LM 不再完全照搬 Vaswani Transformer | post-norm、ReLU、sine embedding vs pre-norm、SwiGLU、RoPE | 建立“现代默认架构”的整体图景           |
| Norm 位置与 RMSNorm         | 为什么 LayerNorm 放置位置会影响训练稳定性        | residual stream、pre-norm、non-residual norm、RMSNorm     | 后续训练稳定性和优化分析的基础         |
| FFN 与 GLU                  | 为什么现代模型倾向使用 SwiGLU / GeGLU            | ReLU、GeLU、GLU、SwiGLU、参数匹配                         | 连接模型表达能力、参数量和计算成本     |
| 位置编码与 RoPE             | Transformer 如何知道 token 顺序                  | sine、absolute、relative、RoPE、旋转内积                  | 长上下文与 attention 变体的前置知识    |
| 超参数经验规律              | hidden size、FFN size、head 数、深宽如何选择     | `d_ff/d_model`、head dimension、aspect ratio、vocab size  | scaling 和训练预算设计的重要输入       |
| 正则化与稳定性              | 为什么大模型训练更怕崩溃而不是单纯过拟合         | dropout、weight decay、z-loss、QK norm、soft-capping      | 大规模预训练工程实践核心               |
| 推理与长上下文              | 为什么 attention 成本在部署时变成瓶颈            | KV cache、MQA、GQA、sliding window attention              | inference、serving、长上下文模型的基础 |

### 1.4 学习本讲前需要知道什么

学习本讲最好先具备以下基础：

| 前置知识                                 | 为什么需要                                                   |
| ---------------------------------------- | ------------------------------------------------------------ |
| Transformer 基本结构                     | 本讲默认你知道 attention、FFN、residual connection、LayerNorm 是什么 |
| 向量和矩阵乘法                           | RoPE、Q/K/V projection、multi-head attention 都用矩阵形式表达 |
| 概率与 softmax                           | 输出分布、attention weights、z-loss、softmax 稳定性都依赖这些概念 |
| 梯度下降与训练稳定性                     | pre-norm、gradient spike、weight decay 的讨论都和优化有关    |
| GPU / memory / arithmetic intensity 直觉 | RMSNorm、bias 删除、GQA/MQA、KV cache 都不是纯数学选择，而是系统选择 |
| 基本语言模型训练目标                     | 理解“架构选择如何服务 next-token prediction”需要知道 LM 训练目标 |

**学习提示：**
 这一讲最重要的学习方式，是不要把每个设计选择当作孤立技巧。应当反复问：这个选择是在改善表达能力、训练稳定性、系统效率、推理成本，还是长上下文能力？

------

## 2. 第一部分：从原始 Transformer 到现代 LLaMA-like Transformer

这一部分要解决的问题是：**为什么现代语言模型的 Transformer block 和原始 Transformer 长得不一样？**

### 2.1 问题引入：原始 Transformer 的默认值不再是现代默认值

课程从 Vaswani 等人的原始 Transformer 出发。原始 Transformer 中有几个经典默认选择：

1. 位置编码使用 sine/cosine；
2. FFN 激活函数使用 ReLU；
3. LayerNorm 采用 post-norm；
4. FFN 和 attention projection 中通常保留 bias。

原报告中给出了 sine 位置编码的一部分形式：
$$
PE_{(pos,2i)} = \sin(pos / 10000^{2i/d_{\text{model}}})
$$
这里：

- `pos` 表示 token 在序列中的位置；
- `i` 表示 embedding 维度中的坐标索引；
- `d_model` 是模型隐藏维度；
- 不同维度使用不同频率的正弦/余弦函数，让模型能区分不同位置。

这类设计在原始 Transformer 时代非常重要，但现代自回归语言模型通常已经改成：

- 用 **pre-norm** 或更一般的 non-residual norm；
- 用 **RMSNorm** 替代 LayerNorm；
- 用 **RoPE** 替代简单的 absolute / sine embedding；
- 用 **SwiGLU / GeGLU** 替代 ReLU / GeLU；
- 线性层和 norm 中经常删除 bias。【PDF†source】

字幕中也说明，Assignment 1 会要求实现这些现代差异：把 LayerNorm 移到 transformer block 前面，实现 RoPE，实现 SwiGLU / GeLU 而不是 ReLU；这很大程度上复制了 LLaMA 的选择，而许多现代模型也采取类似做法。【字幕†source】

### 2.2 概念定义：什么叫“现代 Transformer 默认架构”？

直觉上，现代 Transformer 默认架构不是一套严格数学定理，而是一组经过大量大模型训练经验筛选出来的工程默认值。

更正式地说，本讲讨论的现代 LM 架构，是指以 decoder-only Transformer 为主干，在 block 内采用如下设计的一类模型：

```
token embeddings
    ↓
多层 Transformer block
    - residual stream
    - pre-norm / RMSNorm
    - causal self-attention + RoPE
    - SwiGLU / GeGLU FFN
    - 通常 no bias
    ↓
LM head / output softmax
```

这类模型的目标仍然是语言模型的 next-token prediction，但它的很多设计不是为了让单个公式更漂亮，而是为了让大规模训练更稳定、GPU 运行更高效、推理更便宜。

### 2.3 方法与原理：为什么不能只靠理论推导？

本讲的方法论是：**现代 LM 架构空间太大，不能指望靠穷举或纯理论一次性找到最优设计。**

原报告指出，课程团队没有足够 compute 去覆盖所有架构，因此采取“观察别人怎么做”的经验归纳方式。大量 2024-2025 模型和技术报告提供了观察趋势的机会，例如 vocab size、norm type、position embedding 等都能在模型之间形成明显趋势。【字幕†source】

这说明本讲关注的问题不是：

> “我们能不能证明 RoPE 一定最优？”

而是：

> “为什么大多数强模型最后都用了 RoPE、RMSNorm、SwiGLU、GQA？这些选择背后分别对应什么训练或系统问题？”

### 2.4 课堂例子与解释

课程把原始 Transformer 和现代简单变体并列展示。讲义中前几页对比了：

| 组件      | 原始 Transformer          | 现代 LLaMA-like 默认值       | 变化背后的动机                          |
| --------- | ------------------------- | ---------------------------- | --------------------------------------- |
| Norm 位置 | post-norm                 | pre-norm / non-residual norm | 保持 residual stream 稳定               |
| Norm 类型 | LayerNorm                 | RMSNorm                      | 减少低 arithmetic intensity 操作        |
| FFN 激活  | ReLU                      | SwiGLU / GeGLU               | 增强表达能力和训练效果                  |
| 位置编码  | sine/cosine add embedding | RoPE                         | 让 attention score 更自然地依赖相对位置 |
| bias      | 常见                      | 常删除                       | 减少参数移动和潜在稳定性问题            |

【PDF†source】

**学习提示：**
 本节复习时要抓住一个主线：现代 LM 架构不是对原始 Transformer 的完全否定，而是在同一骨架上，为了大规模训练和部署做出的系统性调整。

------

## 3. 第二部分：LayerNorm 位置、RMSNorm 与删除 bias

这一部分要解决的问题是：**为什么 norm 的位置和形式会成为现代大模型训练的关键？为什么许多模型会删掉看似无害的 bias？**

------

### 3.1 Pre-norm vs post-norm：现代 LM 的关键共识

#### 3.1.1 直觉解释

Transformer block 里有一条贯穿所有层的主通道，通常叫 **residual stream**。你可以把它想象成模型内部的信息高速公路。每一层 attention 或 FFN 不是完全重写表示，而是在当前表示上加一个增量：
$$
x_{l+1} = x_l + F(x_l)
$$
这里：

- `x_l` 是第 `l` 层输入；
- `F(x_l)` 是 attention 或 FFN 计算出的更新量；
- `x_l + F(x_l)` 表示把新信息加回原来的表示。

如果每一层都对这条主通道进行强归一化，就像每过一个路口都强制重新缩放一次车流，信号传播和梯度传播都会变得复杂。现代 LM 的核心直觉是：

> 保持 residual stream 尽量干净，让信息和梯度能沿主路径顺畅流动。

字幕中将这一点概括为：**keep your residual stream clean**。【字幕†source】

#### 3.1.2 正式定义

原始 Transformer 的 post-norm 写法是：
$$
x_{l+1} = \operatorname{LayerNorm}(x_l + F(x_l))
$$
也就是说，先做 residual addition，再对结果做 LayerNorm。这样 LayerNorm 直接作用在主 residual stream 上。

现代常见的 pre-norm 写法是：
$$
x_{l+1} = x_l + F(\operatorname{LayerNorm}(x_l))
$$
也就是说，Norm 只放在计算分支 `F` 的输入处，主 residual stream 中的 `x_l` 可以直接加到下一层。

#### 3.1.3 为什么 pre-norm 更稳定？

从前向传播看，pre-norm 让 `x_l` 有一条不被 norm 改写的路径直接传到高层。

从反向传播看，梯度也可以更直接地沿 residual connection 传回底层。post-norm 中，梯度每层都要穿过 LayerNorm，而 LayerNorm 的雅可比矩阵可能改变梯度方向和尺度，因此容易出现 gradient attenuation 或 gradient spike。

原报告指出，早期研究强调 pre-norm 可以减少 warmup 需求；而在现代大模型中，更重要的理由是深层网络稳定性、更少 gradient spike，以及能承受更大学习率。【PDF†source】

#### 3.1.4 课堂例子与解释

讲义展示了 Xiong 2020 和 Salazar & Nguyen 的实验图：pre-norm 在无 warmup 或稳定性实验中有更好的 loss / BLEU 表现，并且 gradient spike 更小。【字幕†source】

这个例子的代表性在于：它不是说明 pre-norm 一定在所有任务上性能最高，而是说明当模型变深、训练变贵、训练崩溃代价变大时，稳定性本身就成为核心目标。

#### 3.1.5 学习提示

| 复习点           | 正确理解                                                  |
| ---------------- | --------------------------------------------------------- |
| pre-norm 的重点  | 不是“Norm 一定要放在前面”，而是不要破坏主 residual stream |
| post-norm 的问题 | Norm 作用在 residual addition 之后，可能影响深层梯度传播  |
| 工程意义         | 大模型训练中，稳定性比小幅性能差异更重要                  |
| 常见误解         | pre-norm 不是因为公式更优雅，而是因为训练更稳             |

------

### 3.2 Double norm / non-residual postnorm：不是所有 “post” 都错

这一节要解决一个容易误解的问题：**既然 post-norm 不稳定，是不是所有计算之后的 norm 都不好？**

答案是否定的。课程指出，真正的问题不是 “norm 在计算前还是计算后”，而是 **norm 是否放进了主 residual stream**。

如果 norm 放在计算之后，但仍然只作用于分支输出，而不破坏主 residual stream，那么它可能依然有效。近年来一些模型使用的 non-residual postnorm 或 double norm 就体现了这个思路。

讲义中提到 Grok、Gemma 2、OLMo 2 等近期模型采用了这类结构：LayerNorm 不必一定在 computation 之前，也可以在 computation 之后，只要不要直接破坏主 residual signal path。【PDF†source】

**工程直觉：**
 当训练出现不稳定时，一个粗糙但常见的经验做法是“多加 norm”：在 block 前、计算后、甚至 Q/K 上加 norm。这个做法听起来不够理论化，但在大模型训练中常常有效。【字幕†source】

**学习提示：**
 复习时不要机械记成 “pre 好、post 坏”。更准确的理解是：

> 现代趋势是 non-residual norm：norm 可以在不同位置，但最好不要反复污染主 residual stream。

------

### 3.3 LayerNorm vs RMSNorm

这一节要解决的问题是：**为什么现代大模型越来越常用 RMSNorm，而不是原始 Transformer 常见的 LayerNorm？**

#### 3.3.1 LayerNorm 的公式与含义

LayerNorm 对每个 token 的 hidden vector 做均值和方差归一化：
$$
\operatorname{LayerNorm}(x)
=
\frac{x-\mathbb{E}[x]}
{\sqrt{\operatorname{Var}[x]+\epsilon}}
\odot \gamma+\beta
$$
其中：

- `x` 是某个 token 的 hidden state，维度通常为 `d_model`；
- `E[x]` 是该 token hidden vector 各维度的均值；
- `Var[x]` 是该 hidden vector 各维度的方差；
- `ε` 是防止除零的小常数；
- `γ` 是可学习缩放参数；
- `β` 是可学习偏移参数；
- `⊙` 表示逐元素乘法。

LayerNorm 的直觉是：把每个 token 的隐藏表示调整到相对稳定的尺度和中心位置，避免某些层的激活过大或过小。

#### 3.3.2 RMSNorm 的公式与含义

RMSNorm 不减均值，也通常没有 bias，只根据 root mean square 控制尺度：
$$
\operatorname{RMSNorm}(x)
=
\frac{x}
{\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2+\epsilon}}
\odot \gamma
$$
其中：

- `d` 是 hidden dimension；
- `x_i` 是 hidden vector 的第 `i` 个分量；
- 分母是 hidden vector 的 root mean square；
- `γ` 是可学习缩放参数。

RMSNorm 保留了控制激活尺度的功能，但去掉了均值中心化和偏移项。

#### 3.3.3 为什么 RMSNorm 更流行？

原报告总结了课程给出的现代解释：RMSNorm 的表达能力损失通常很小，但它更快，因为它少了均值计算，也少了 bias 参数移动。【PDF†source】

这背后有一个重要系统观点：

> FLOPs 少不等于 runtime 少；某些操作虽然 FLOPs 很少，但因为频繁读写内存，实际运行时间可能很高。

LayerNorm / RMSNorm 这类统计归一化操作的 arithmetic intensity 较低。GPU 最擅长大矩阵乘法，不擅长频繁搬运小块数据。如果均值减法和 bias 对模型能力贡献不大，删除它们就可能提升训练速度。【字幕†source】

#### 3.3.4 RMSNorm 与 LayerNorm 的对比

| 维度           | LayerNorm                | RMSNorm                                | 学习重点                           |
| -------------- | ------------------------ | -------------------------------------- | ---------------------------------- |
| 是否减均值     | 是                       | 否                                     | RMSNorm 不做中心化                 |
| 是否归一化尺度 | 是                       | 是                                     | 二者都控制激活大小                 |
| 是否有 bias    | 通常有 `β`               | 通常没有                               | RMSNorm 更简洁                     |
| 系统成本       | 更高                     | 更低                                   | RMSNorm 减少统计计算和参数移动     |
| 现代模型趋势   | GPT1/2/3、OPT 等使用较多 | LLaMA、PaLM、Chinchilla、T5 等使用较多 | 现代默认更偏 RMSNorm【PDF†source】 |

#### 3.3.5 学习提示

| 容易误解                        | 正确理解                                                     |
| ------------------------------- | ------------------------------------------------------------ |
| RMSNorm 比 LayerNorm 理论上更强 | 不一定。RMSNorm 更流行主要是因为够用且更高效                 |
| LayerNorm FLOPs 很少所以不重要  | FLOPs 少不代表 runtime 少，memory access 可能很贵            |
| RMSNorm 只是小改动              | 在大规模训练中，小的 kernel / memory 改动可能带来显著 wall-clock 差异 |

------

### 3.4 删除 bias term

这一节要解决的问题是：**为什么现代 Transformer 常常删除 linear layer 和 norm 中的 bias？**

原始 FFN 可以写成：
$$
FFN(x)=\max(0,xW_1+b_1)W_2+b_2
$$
现代非 gated FFN 常被简化为：
$$
FFN(x)=\sigma(xW_1)W_2
$$
也就是去掉 `b_1` 和 `b_2`。

原报告指出，讲义第 18 页说明现代 Transformer 多数没有 bias term，原因包括 memory、与 RMSNorm 类似的系统效率，以及优化稳定性。【PDF†source】

#### 工程直觉

bias 参数本身占参数量不大，但它会带来额外读写和 kernel 处理。对小模型这可能不明显；对大模型，所有这种“看起来很小”的内存移动都会累积。

字幕进一步解释：bias 在 Transformer 中通常并不非常有用，却会带来额外参数移动；某些情况下还可能诱发稳定性问题，所以现代模型常把 linear 和 RMSNorm 中的 bias 都删掉。【字幕†source】

#### 学习提示

删除 bias 不是因为 bias 在神经网络中永远没用，而是因为在现代 LM 的典型架构和训练规模下：

> 它的收益往往小于系统成本和潜在稳定性成本。

------

## 4. 第三部分：激活函数、GLU 与 FFN

这一部分要解决的问题是：**为什么现代语言模型的 FFN 不再只是 ReLU 或 GeLU，而是大量采用 SwiGLU / GeGLU？**

------

### 4.1 从 ReLU 到 GeLU，再到 SwiGLU / GeGLU

Transformer block 中的 FFN 通常负责对每个 token 的 hidden representation 做逐位置非线性变换。它不混合序列位置，主要增强每个 token 表示的非线性表达能力。

普通 ReLU FFN：
$$
FF(x)=\max(0,xW_1)W_2
$$
GeLU FFN：
$$
FF(x)=\operatorname{GELU}(xW_1)W_2
$$
其中 GeLU 可以理解为一种平滑版 ReLU：它不是简单地把负数截断为 0，而是根据输入大小平滑地控制通过程度。

讲义列出了 ReLU、GeLU、Swish、ELU、GLU、GeGLU、ReGLU、SeLU、SwiGLU、LiGLU 等激活函数，并提出问题：这些函数是什么？大家实际用什么？它们到底重要吗？【PDF†source】

原报告总结的模型趋势是：

| 激活 / FFN 类型 | 代表模型                                          |
| --------------- | ------------------------------------------------- |
| ReLU            | Original Transformer、T5、Gopher、Chinchilla、OPT |
| GeLU            | GPT1/2/3、GPT-J、GPT-NeoX、BLOOM                  |
| SwiGLU / GeGLU  | LLaMA、PaLM、T5 v1.1，以及 2023 年后的多数模型    |

【PDF†source】

字幕补充说，使用 ReLU 或 GeLU 也能训练出合理语言模型，例如 GPT-3 使用 GeLU；但现代强模型几乎都使用 gated linear unit，也就是 SwiGLU 或 GeGLU。【字幕†source】

------

### 4.2 GLU：门控为什么有效？

#### 4.2.1 直觉解释

普通 FFN 像是“先生成一批特征，再整体投影回 hidden dimension”。GLU 的直觉则是：不仅要生成候选特征，还要同时生成一个门控向量，决定哪些特征应该通过、哪些应该被抑制。

可以把它类比成：

- 一条分支产生内容；
- 另一条分支产生开关；
- 内容和开关逐元素相乘；
- 最后再投影回模型维度。

#### 4.2.2 ReGLU 公式

普通 ReLU FFN：
$$
FF(x)=\max(0,xW_1)W_2
$$
ReGLU 加入一个额外线性门控项 `xV`：
$$
FF_{\text{ReGLU}}(x)
=
\left(\max(0,xW_1)\otimes xV\right)W_2
$$
其中：

- `x` 是输入 hidden state；
- `W_1` 产生候选特征；
- `V` 产生门控向量；
- `⊗` 是逐元素乘法；
- `W_2` 把中间表示投影回 `d_model`。

讲义第 22 页说明，GLU 修改的是 FF layer 的第一部分，用额外线性项 `xV` 逐元素调制激活输出，因此会多一个参数矩阵 `V`。【PDF†source】

#### 4.2.3 GeGLU 与 SwiGLU

GeGLU：
$$
FFN_{\text{GeGLU}}(x,W,V,W_2)
=
(\operatorname{GELU}(xW)\otimes xV)W_2
$$
SwiGLU 使用 Swish 激活。Swish 定义为：
$$
\operatorname{Swish}(x)=x\cdot \sigma(x)
$$
因此 SwiGLU 可以理解为：
$$
FFN_{\text{SwiGLU}}(x)
=
(\operatorname{Swish}(xW)\otimes xV)W_2
$$
讲义第 23 页指出，T5 v1.1、mT5、LaMDA、Phi3、Gemma 系列偏向 GeGLU；LLaMA 1/2/3、PaLM、Mistral、OLMo 和 2023 年后多数模型偏向 SwiGLU。【PDF†source】

------

### 4.3 为什么 GLU 要把 `d_ff` 缩小到约 `8/3 d_model`？

这是本讲非常重要的一个公式推导。

普通 FFN 有两个矩阵：
$$
W_1 \in \mathbb{R}^{d_{\text{model}}\times d_{ff}},
\quad
W_2 \in \mathbb{R}^{d_{ff}\times d_{\text{model}}}
$$
参数量约为：
$$
d_{\text{model}}d_{ff}+d_{ff}d_{\text{model}}
=
2d_{\text{model}}d_{ff}
$$
如果传统设置是：
$$
d_{ff}=4d_{\text{model}}
$$
那么普通 FFN 参数量约为：
$$
2d_{\text{model}}\cdot 4d_{\text{model}}
=
8d_{\text{model}}^2
$$
GLU 多了一个门控矩阵 `V`，所以参数量约为：
$$
3d_{\text{model}}d'_{ff}
$$
为了让 GLU 和普通 FFN 参数量接近，令：
$$
3d_{\text{model}}d'_{ff}
=
8d_{\text{model}}^2
$$
得到：
$$
d'_{ff}
=
\frac{8}{3}d_{\text{model}}
\approx 2.67d_{\text{model}}
$$
这就是为什么 gated FFN 常把 `d_ff` 从 `4d_model` 缩到大约 `8/3 d_model`。字幕也明确说，因为 GLU 多一个 `V`，如果要保持参数量相同，应把 feedforward dimension 缩小 `2/3`。【字幕†source】

#### 学习提示

| 问题                   | 回答                                   |
| ---------------------- | -------------------------------------- |
| GLU 为什么多参数？     | 多了门控矩阵 `V`                       |
| 为什么 `d_ff` 要缩小？ | 为了和普通 FFN 保持相近参数量和计算量  |
| `8/3` 从哪里来？       | 令 `3 d_model d_ff' = 8 d_model^2`     |
| 复习重点               | 能从参数量匹配角度推导，而不是死记数值 |

------

### 4.4 Serial vs parallel layers

这一节要解决的问题是：**attention 和 MLP 是应该串行计算，还是并行计算？**

标准 Transformer block 是串行结构：
$$
x' = x + \operatorname{Attention}(\operatorname{Norm}(x))
$$
然后再经过 MLP 分支。parallel layer 则试图同时计算 attention 和 MLP：
$$
y
=
x
+
\operatorname{MLP}(\operatorname{LayerNorm}(x))
+
\operatorname{Attention}(\operatorname{LayerNorm}(x))
$$
讲义第 27-28 页说明，GPT-J、PaLM、GPT-NeoX 等少数模型使用 parallel layers。如果实现得好，可以共享 LayerNorm，并融合 matrix multiplication，从而带来系统优化。【PDF†source】

字幕解释为什么它后来不再主流：parallel 结构虽然可能提升系统利用率，但相当于减少了有效深度；当 serial block 优化得足够好后，parallel 的系统收益可能不值得表示能力损失。【字幕†source】

#### 学习提示

| 设计           | 优点                                      | 缺点             | 现代趋势       |
| -------------- | ----------------------------------------- | ---------------- | -------------- |
| serial block   | 表达深度更自然，主流实现成熟              | 计算顺序更长     | 多数模型仍使用 |
| parallel block | 可共享 norm、融合矩阵乘法，系统上可能更快 | 可能降低有效深度 | 少数模型使用   |

------

## 5. 第四部分：位置编码与 RoPE

这一部分要解决的问题是：**Transformer 的 self-attention 本身不理解顺序，那么现代语言模型如何把位置信息注入 attention？为什么 RoPE 成为主流？**

------

### 5.1 为什么 Transformer 必须加入位置信息？

Self-attention 的核心是 token 之间的 inner product。如果没有位置编码，模型只能看到一组 token 表示，却不知道它们的顺序。换句话说，attention 本身对 token 顺序近似 permutation-invariant。

字幕明确解释：attention 是 inner product，可以 shuffle；没有 position embedding 时，attention 结果无法体现位置。【字幕†source】

因此，语言模型必须加入位置信息。讲义列出四类位置编码：

| 方法                | 思想                                               | 典型模型                            |
| ------------------- | -------------------------------------------------- | ----------------------------------- |
| Sine embeddings     | 给 token embedding 加正弦余弦位置向量              | Original Transformer                |
| Absolute embeddings | 每个位置有自己的可学习向量                         | GPT1/2/3、OPT                       |
| Relative embeddings | 在 attention 计算中加入相对位置项                  | T5、Gopher、Chinchilla              |
| RoPE                | 对 query/key 做旋转，使 attention 内积依赖相对位置 | GPT-J、PaLM、LLaMA，多数 2024+ 模型 |

【PDF†source】

------

### 5.2 RoPE 的目标：让 attention score 真正依赖相对位置

RoPE 希望构造一个位置相关函数 `f(x,i)`，满足：
$$
\langle f(x,i), f(y,j)\rangle
=
g(x,y,i-j)
$$
这个公式表达的是：

- `x` 和 `y` 是两个 token 的内容表示；
- `i` 和 `j` 是两个 token 的绝对位置；
- `i-j` 是相对距离；
- attention score 应该只依赖 token 内容和相对位置，而不是绝对位置。

讲义第 31 页明确给出这个目标，并指出 sine embedding 有非相对 cross terms，absolute embedding 显然不是 relative，传统 relative embedding 虽然相对，但不是一个 inner product factorization。【PDF†source】

字幕中的直觉例子是：如果 “A” 和 “apple” 一起出现，无论在句首还是句尾，只要它们相对位置一样，RoPE 希望它们的内积关系保持一致。【字幕†source】

------

### 5.3 RoPE 的几何直觉：旋转保持相对角度

在二维空间中，令位置 `i` 对应旋转矩阵 `R_i`。对 query 和 key 分别做旋转：
$$
q_i = R_i W_q x_i
$$
attention score 为：
$$
q_i^\top k_j
=
x_i^\top W_q^\top R_i^\top R_j W_k x_j
$$
旋转矩阵满足：
$$
R_i^\top R_j = R_{j-i}
$$
因此：
$$
q_i^\top k_j
=
x_i^\top W_q^\top R_{j-i} W_k x_j
$$
这就说明 attention score 只依赖相对位置 `j-i`，而不是绝对位置 `i` 和 `j`。

#### 课堂例子

课程用 “we know that” 和 “of course we know” 说明 RoPE：

- 在 “we know that” 中，`we` 在位置 0，`know` 在位置 1；
- 在 “of course we know” 中，`we` 在位置 2，`know` 在位置 3；
- 两个句子中，`we` 和 `know` 的绝对位置不同，但相对距离都是 1。

RoPE 让二者旋转后保持相同相对角度，因此 inner product 能捕捉相对关系。【字幕†source】

讲义第 32 页也用图展示了这个例子：position-independent embedding、句子 “we know that” 的旋转、句子 “of course we know” 的旋转，并强调 inner product 对整体旋转不变。【PDF†source】

------

### 5.4 高维 RoPE：把坐标两两配对旋转

在高维空间中，RoPE 选择一个简单而有效的方法：把 `d` 维向量拆成 `d/2` 个二维坐标对，每对使用不同频率旋转。

直觉上：

- 高频维度旋转快，更适合捕捉近距离关系；
- 低频维度旋转慢，更适合捕捉长距离关系。

讲义第 33 页总结为：“Just pair up the coordinates and rotate them in 2d”。【PDF†source】

字幕补充说，RoPE 的论文可能用复数形式解释，但直觉上就是反复使用二维旋转；RoPE 的 sine/cosine 看起来像原始 sine embedding，但二者关键区别是：

| 方法           | 操作方式 | 作用位置        | 结果                             |
| -------------- | -------- | --------------- | -------------------------------- |
| sine embedding | 加法     | token embedding | 可能出现内容和位置的 cross terms |
| RoPE           | 乘法旋转 | query/key       | 让 QK 内积依赖相对位置           |

【字幕†source】

------

### 5.5 RoPE 在 attention 中的实现位置

RoPE 不是只在输入 embedding 上加一次，而是在每一层 attention 中对 query 和 key 应用。

PDF 第 35 页的代码截图显示流程大致为：

1. 先计算 Q/K/V projection；
2. reshape 成 multi-head attention 需要的形状；
3. 根据 position IDs 生成 RoPE 的 cos/sin；
4. 对 query_states 和 key_states 调用 `apply_rotary_pos_emb`；
5. 最后进入标准 multi-head self-attention。【PDF†source】

字幕解释：这样做是为了每次 attention 计算都强制位置不变性；cos/sin 基于 position IDs 生成，并应用到 query 和 key，而不是 value。【字幕†source】

#### 学习提示

| 复习点                          | 一句话解释                                         |
| ------------------------------- | -------------------------------------------------- |
| 为什么只旋转 Q/K？              | attention score 来自 QK 内积，位置关系应影响 score |
| 为什么不旋转 V？                | V 是被加权汇聚的内容，不直接决定 attention 权重    |
| RoPE 和 sine embedding 最大区别 | RoPE 是乘法旋转，不是加法位置向量                  |
| RoPE 的核心公式                 | `R_i^T R_j = R_{j-i}`                              |

------

## 6. 第五部分：超参数经验规律

这一部分要解决的问题是：**模型宽度、深度、FFN 中间维度、head dimension、词表大小等超参数有没有经验默认值？哪些选择比较敏感，哪些比较 forgiving？**

------

### 6.1 `d_ff / d_model`：经典的 4 倍规则与 GLU 的 8/3 规则

FFN 有两个关键维度：

- `d_model`：主 hidden size；
- `d_ff`：FFN 中间层维度。

传统经验规律是：
$$
d_{ff}=4d_{\text{model}}
$$
讲义第 37 页把它称为一个 surprisingly consensus hyperparameter，并说这几乎总是成立，只有少数例外。【PDF†source】

对于 GLU，因为多一个门控矩阵 `V`，参数匹配后常用：
$$
d_{ff}
=
\frac{8}{3}d_{\text{model}}
\approx 2.67d_{\text{model}}
$$
讲义第 38 页列出 PaLM、Mistral 7B、LLaMA-2 70B、Qwen 14B、DeepSeek 67B、Yi 34B、T5 v1.1 等模型的比例，大多落在 2.5 到 3.5 附近。【PDF†source】

#### T5 的极端例外

T5 11B 使用：
$$
d_{ff}=65536,\quad d_{\text{model}}=1024
$$
倍率为 64。讲义第 39 页说这是非常大胆的设置，理由是更大的矩阵乘法可能更高效地利用 TPU；但后续 T5 v1.1 回到更标准的 2.5 GeGLU multiplier，说明 64 倍虽然能工作，但很可能不是最优。【PDF†source】

字幕还指出，Kaplan 2020 的实验显示 `d_ff/d_model` 在大约 1 到 10 之间有较平坦的好区域，所以 2.6、3.5、4 这些值都在安全区；超过很大范围才会明显恶化 loss。【字幕†source】

#### 学习提示

| 设置                 | 适用情况                    | 复习重点         |
| -------------------- | --------------------------- | ---------------- |
| `d_ff = 4 d_model`   | 普通 ReLU / GeLU FFN        | 经典规则         |
| `d_ff ≈ 8/3 d_model` | GLU / SwiGLU / GeGLU        | 参数量匹配       |
| 极大 `d_ff`          | 特殊系统考虑，如 TPU 利用率 | 能工作不代表最优 |

------

### 6.2 Head dimension 与 number of heads

多头注意力常见规则是：
$$
h\cdot d_{\text{head}}=d_{\text{model}}
$$
即：
$$
d_{\text{head}}
=
\frac{d_{\text{model}}}{h}
$$
其中：

- `h` 是 attention head 数；
- `d_head` 是每个 head 的维度；
- `d_model` 是总 hidden size。

讲义第 42-43 页指出，这不是数学上必须成立；也可以让 head dimension 大于 `d_model/h`，但大多数模型仍遵守这个比例。表中 GPT-3、T5 v1.1、LLaMA2 等 ratio 为 1，T5 和 LaMDA 等 Google 模型是例外。【PDF†source】

字幕也说，head dimension 是一个相对 forgiving 的超参数，很多模型 ratio 大约在 1 附近，少数 Google 模型有明显偏离。【字幕†source】

#### 学习提示

不要把 `h * d_head = d_model` 理解成数学定律。它更像一个工程默认值：实现方便、并行自然、参数和张量形状整齐。

------

### 6.3 Aspect ratio：模型该深还是该宽？

课程定义 aspect ratio 为：
$$
\frac{d_{\text{model}}}{n_{\text{layer}}}
$$
它反映模型随规模变大时更偏“宽”还是更偏“深”。

- `d_model` 越大，模型每层越宽；
- `n_layer` 越大，模型层数越深；
- aspect ratio 越大，表示相对更宽；
- aspect ratio 越小，表示相对更深。

讲义第 44 页展示多个模型的比例：BLOOM 205、T5 v1.1 171、PaLM 156、GPT3/OPT/Mistral/Qwen/OLMo 3 为 128、LLaMA/LLaMA2 为 102、Gemma 3 为 87、Gemma 4 为 61、T5 11B 为 33。【PDF†source】

字幕解释，现代模型通常不走极端深也不走极端宽，很多落在每层约 100 个 hidden width 的量级。太深会导致系统并行困难和更高 latency；宽度更容易做 tensor parallel，而深度切分通常进入 pipeline parallel，会带来更复杂的并行问题。【字幕†source】

讲义第 51 页总结：aspect ratio 有 100-200 的宽泛好区间，系统因素往往决定具体值。【PDF†source】

#### 工程直觉

宽模型更容易把同一层的大矩阵乘法切到多个 GPU 上；深模型则更依赖 pipeline parallel。pipeline parallel 会带来 bubble、调度和通信复杂度，因此实际设计常常偏向系统可训练性，而不是单纯理论表达能力。

------

### 6.4 Vocabulary size：单语、多语与生产模型

词表大小决定 tokenizer 可以直接表示多少 token。讲义第 47 页展示典型范围：

| 类型            | 典型范围   | 例子                                                         |
| --------------- | ---------- | ------------------------------------------------------------ |
| 单语模型        | 30k-50k    | Original Transformer 37k、GPT 40,257、GPT2/3 50,257、T5 32,128、LLaMA 32,000 |
| 多语 / 生产系统 | 100k-250k+ | mT5 250k、PaLM 256k、GPT-4 100,276、Gemma 4 262,144、DeepSeek 100k、Qwen 15B 152,064 |

【PDF†source】

讲义结论是：单语 vocab 不需要很大，但多语模型需要更大 vocab 覆盖更多语言与符号空间。【PDF†source】

字幕还补充了课堂问答：如果是 multimodal 模型，取决于图像等模态如何 tokenization；如果图像也被 token 化，则可能需要单独较大的 image tokenizer vocab。【字幕†source】

#### 学习提示

| 词表太小                                     | 词表太大                                            |
| -------------------------------------------- | --------------------------------------------------- |
| 文本被切得过碎，序列变长，训练和推理成本增加 | embedding / output softmax 变大，参数和计算成本增加 |
| 对多语和稀有符号不友好                       | token 覆盖更强，但可能浪费词表容量                  |

------

### 6.5 Dropout、weight decay 与“正则化”的反直觉

#### 6.5.1 标准直觉

在传统机器学习中，dropout 和 weight decay 常被称为正则化方法，主要用于减少 overfitting。

但语言模型预训练往往使用海量数据。在 compute-constrained setting 下，模型可能甚至无法多次遍历训练数据。字幕中授课者说，在 compute-constrained language modeling 中，overfitting 几乎不是问题，因此有些人甚至只看 training loss。【字幕†source】

#### 6.5.2 实际做法

讲义第 49 页显示，旧模型常使用 dropout：Original Transformer、GPT2、T5、GPT3、OPT 等都有 0.1 dropout；新模型如 PaLM、LLaMA 更倾向不用 dropout，但仍使用 weight decay。LLaMA weight decay 为 0.1，Qwen 14B 则 dropout 与 weight decay 都为 0.1。【PDF†source】

#### 6.5.3 关键结论：weight decay 不一定主要是防过拟合

字幕解释，weight decay 在 LLM 预训练中常常不是为了减少 overfitting，而是与 optimizer 和 learning-rate schedule 相互作用，改善优化动态。Andriushchenko et al. 2023 的观察表明，在 single-pass 训练中，不同 weight decay 下 train / validation loss 并没有显示典型 overfitting 控制效果；但 weight decay 与 cosine LR decay 联合时，较强 weight decay 可能最终收敛到更好 loss。【字幕†source】

#### 学习提示

本节最重要的反直觉是：

> 一个机制在教科书里叫“正则化”，但在大模型预训练中，它可能主要是在改变优化轨迹，而不是控制训练集和验证集之间的 gap。

------

## 7. 第六部分：稳定训练技巧

这一部分要解决的问题是：**为什么现代大模型训练如此强调稳定性？softmax 为什么危险？z-loss、QK norm、soft-capping 分别解决什么问题？**

------

### 7.1 为什么稳定性成为核心问题？

讲义第 52 页展示了 loss 和 gradient norm 出现大量 spike 的训练曲线，并写道：“Don’t train models that look like the blue curve!”【PDF†source】

字幕强调，模型越大，训练越昂贵，稳定性越重要。如果训练中途 loss 爆炸，模型质量可能无法恢复，已经消耗的资金和算力也可能浪费。【字幕†source】

#### 学习提示

在小模型实验中，loss spike 可能只是一次失败运行；在大模型预训练中，loss spike 可能意味着数十万甚至数百万美元级别的浪费。因此稳定训练技巧不是“锦上添花”，而是“能不能完成训练”的前提。

------

### 7.2 Softmax 是危险区域

语言模型里至少有两个重要 softmax：

1. **输出 softmax**：把 vocabulary logits 转成 token 概率分布；
2. **attention softmax**：把 attention logits 转成 attention weights。

softmax 形式为：
$$
\operatorname{softmax}(z_i)
=
\frac{e^{z_i}}{\sum_j e^{z_j}}
$$
这里有两个数值风险：

- 指数 `e^{z_i}` 可能爆炸；
- 分母如果数值不稳定，可能导致除法问题。

字幕明确说，softmax 有两个危险因素：exponential 和 division by zero，因此是模型稳定性的 danger zone。【字幕†source】

------

### 7.3 输出 softmax 稳定性：z-loss

输出 softmax 中：
$$
P(x)=\frac{e^{U_x}}{Z}
$$
其中：
$$
Z=\sum_v e^{U_v}
$$
`Z` 是 softmax normalizer。如果 `log Z` 过大或过小，loss 可能不稳定。

z-loss 的思想是在目标中加入惩罚项：
$$
L
=
\sum_i
\left[
-\log P(x_i)
+
\alpha(\log Z(x_i))^2
\right]
$$
其中：

- `-log P(x_i)` 是标准负对数似然；
- `Z(x_i)` 是该样本对应的 softmax normalizer；
- `α` 是一个小权重；
- `(\log Z)^2` 惩罚 normalizer 偏离稳定范围。

直觉是让 `log Z` 靠近 0，也就是让 `Z` 靠近 1，从而稳定 softmax normalizer。

讲义第 54 页写到，PaLM 使用过 z-loss，其他例子包括 Baichuan 2、DCLM、OLMo 2、OLMo 3。【PDF†source】

字幕解释：softmax 对整体加常数是过参数化的，可以通过惩罚 `log Z` 偏离 0 来稳定 normalizer；这就是 Jacob Devlin 2014 提出的 z-loss trick。【字幕†source】

#### 学习提示

| 问题                      | z-loss 的作用             |
| ------------------------- | ------------------------- |
| 输出 logits 整体漂移      | 惩罚 `log Z`              |
| softmax normalizer 不稳定 | 让 `Z` 靠近稳定范围       |
| 是否改变主训练目标        | 是，在 NLL 外加了小稳定项 |
| 适用位置                  | 输出 vocabulary softmax   |

------

### 7.4 Attention softmax 稳定性：QK norm

标准 attention logits 为：
$$
S
=
\frac{QK^\top}{\sqrt{d_k}}
$$
如果 Q 和 K 的范数不断变大，`S` 也会变大，softmax 可能进入饱和区域。饱和后 attention 权重可能过于尖锐，梯度也可能不稳定。

QK norm 的做法是在 Q/K 进入 softmax 前先做归一化：
$$
S_{ij}
=
\frac{
\operatorname{Norm}(q_i)^\top
\operatorname{Norm}(k_j)
}{
\sqrt{d_k}
}
$$
这里：

- `q_i` 是第 `i` 个 query；
- `k_j` 是第 `j` 个 key；
- `Norm` 可以是 LayerNorm 或 RMSNorm；
- 归一化之后，QK 内积尺度更可控。

讲义第 55 页说明，query 和 keys 在进入 softmax 前被 Layer/RMS normed；例子包括 DCLM、OLMo2、Gemma 2、Qwen3、OLMo 3、Gemma 4；该技巧最早来自 vision 和 multimodal models。【PDF†source】

字幕直观解释为：在 Q/K 做矩阵乘法之前“扔一个 LayerNorm”，让输入 softmax 的 logits 尺度大致一致，避免 attention degeneracy。【字幕†source】

#### 学习提示

QK norm 控制的是 **softmax 前的 logits 尺度**，不是 softmax 后的 attention weights。复习时要把这一点和 dropout、attention mask 区分开。

------

### 7.5 Logit soft-capping

logit soft-capping 比 QK norm 更强。它直接把进入 softmax 的 logits 限制到某个范围。例如：
$$
\tilde z
=
c\tanh(z/c)
$$
其中：

- `z` 是原始 logit；
- `c` 是 cap 值；
- `tanh` 的输出范围在 `[-1,1]`；
- 因此 `\tilde z` 的范围在 `[-c,c]`。

讲义第 56 页说明，soft-capping 通过 Tanh 把 logits 限制到最大值，防止 logits 爆炸，但可能带来性能问题。【PDF†source】

字幕补充：Gemma 2、3、4 使用了 logit soft cap；NVIDIA 的系统比较中，QK norm 可能略有收益，因为可提高 learning rate；但 soft-capping 单独使用可能损失性能，因为它限制了模型表达非常自信 softmax 信号的能力。【字幕†source】

#### 学习提示

| 方法         | 控制对象                    | 强度 | 潜在副作用                 |
| ------------ | --------------------------- | ---- | -------------------------- |
| z-loss       | 输出 softmax normalizer     | 中等 | 改变 loss                  |
| QK norm      | attention logits 的输入尺度 | 中等 | 增加 norm 操作             |
| soft-capping | logits 绝对范围             | 强   | 可能限制模型表达高置信分布 |

------

## 8. 第七部分：Attention 变体、GQA/MQA 与长上下文

这一部分要解决的问题是：**训练时 attention 成本和推理时 attention 成本为什么不同？为什么 MQA/GQA 能降低推理成本？长上下文模型为什么常混合 full attention 和 sliding window attention？**

------

### 8.1 为什么从训练转向部署？

前面讨论的是训练和模型结构；课程后半转向部署和 attention 成本。字幕说，训练好大模型后要服务用户，抽象上要付两种成本：

1. FLOPs；
2. memory accesses。

这两者都会影响 latency、utilization 和系统效率。【字幕†source】

#### 工程直觉

训练时通常有较大 batch 和完整序列，矩阵乘法容易做大，GPU 利用率较好。推理时尤其是 incremental decoding，每次只生成一个 token，需要频繁读取模型参数和 KV cache，memory access 可能成为瓶颈。

------

### 8.2 Prefill / training 时 attention 的 arithmetic intensity

讲义第 59 页给出 prefill / training 类似场景的近似：

- hidden dim：`d`
- batch：`b`
- sequence length：`n`
- heads：`h`
- head dim：`k=d/h`

总算术操作近似：
$$
O(bnd^2)
$$
总内存访问近似：
$$
O(bnd + bhn^2 + d^2)
$$
arithmetic intensity 较高，因此可以让 GPU 保持忙碌。【PDF†source】

#### 学习提示

prefill 阶段和训练阶段更像“大批量矩阵计算”，GPU 更容易吃满；这和逐 token decoding 的瓶颈非常不同。

------

### 8.3 Incremental decoding 与 KV cache

生成时不能一次性并行生成所有 token，而是一步一步：
$$
x_1 \rightarrow x_2 \rightarrow x_3 \rightarrow \cdots
$$
如果每生成一个新 token 都重新计算所有历史 token 的 K/V，会非常浪费。因此需要 **KV cache**：保存过去 token 的 key 和 value，使新 token 只需与历史 KV 交互，而不是重新计算所有历史 attention。

讲义第 60 页说明，incremental generation 需要 step by step，通过 KV cache 增量更新 attention。【PDF†source】

但 KV cache 也带来问题：incremental decoding 的 arithmetic intensity 变差。讲义第 61 页给出：
$$
\text{memory access}
\approx
O(bn^2d+nd^2)
$$
这意味着需要大 batch、短 sequence 或很大的 `d` 才能效率高；小模型长上下文部署会受到 memory access 限制。【PDF†source】

字幕也解释，KV cache 虽节省 compute，但每一步都要读参数和历史 KV，导致 memory access pattern 不理想。【字幕†source】

------

### 8.4 MQA：多个 query，共享 key/value

Multi-Query Attention 的想法是：

- query heads 保持多个；
- key/value heads 减少到 1 组或很少几组；
- 因为 KV cache 主要存 K/V，所以这样能大幅减少内存移动。

讲义第 62 页写道，MQA 的 key idea 是 multiple queries, but just one dimension for keys and values，因此 KV cache 中要搬运的项目更少。【PDF†source】

字幕解释：普通 MHA 每个 head 有不同 Q/K/V；MQA 让所有 heads 共享 K 和 V，只让 Q 不同。这会显著减少 KV cache 大小和内存访问，提高推理效率；但代价是表达能力下降，因为所有 query heads 只能看同一组 K/V。【字幕†source】

#### 学习提示

MQA 的本质不是减少 query，而是减少 **key/value cache 的头数**。因为推理时读 KV cache 很贵，所以减少 KV heads 很有价值。

------

### 8.5 GQA：在 MHA 和 MQA 之间取折中

Grouped Query Attention 不把 K/V 减到 1 组，而是减到少数几组。若 query heads 为 `h`，key/value heads 为 `g`，则：
$$
1 < g < h
$$
GQA 提供一个可调旋钮：

- `g/h` 越小，KV cache 越小，推理越高效；
- 但表达能力可能下降；
- 相比 MQA，GQA 保留更多 K/V 多样性；
- 相比完整 MHA，GQA 明显降低 KV cache 成本。

讲义第 63 页称 GQA 是 “Don’t go all the way to one dimension of KV”，可以控制 expressiveness 和 inference efficiency。【PDF†source】

字幕总结说，GQA 在实践中 trade-off 非常好：相比完整 MHA，它推理成本低很多，性能接近；相比 MQA，它性能损失更小。因此现代模型几乎都采用 GQA 结构。【字幕†source】

#### MHA / MQA / GQA 对比

| 方法 | Query heads | Key/Value heads | KV cache 成本 | 表达能力     |
| ---- | ----------- | --------------- | ------------- | ------------ |
| MHA  | `h`         | `h`             | 最高          | 最强         |
| MQA  | `h`         | `1`             | 最低          | 可能损失较大 |
| GQA  | `h`         | `g`, `1<g<h`    | 中等偏低      | 接近 MHA     |

------

### 8.6 Sliding window attention 与 full/local interleaving

完整 attention 的成本随上下文长度二次增长：
$$
O(n^2)
$$
当上下文变长时，这个成本会迅速变高。Sliding window attention 限制每个 token 只关注固定窗口内的 token，把 attention matrix 从 full matrix 变成 banded / local pattern。

讲义第 65 页说明，sparse / sliding window attention 通过结构化 attention 在 expressiveness 和 runtime 之间做折中，例子包括 GPT-3、GPT-OSS、Gemma4。【PDF†source】

当前趋势是 **full attention 与 cheap/local attention 交替**。讲义第 66 页以 Cohere Command A 为例：每 4 层有一层 full attention；长程信息可通过 NoPE，全局/局部结构结合，短程信息通过 RoPE + SWA；LLaMA 4、Gemma 3、Gemma 4、OLMo 3 使用 SWA + Full RoPE。【PDF†source】

字幕讲得更细：每四层一层 full attention，三层 sliding window attention；局部层不断聚合局部信息，full 层提供全局通信。这样可以在长上下文性能和推理成本之间取得平衡。Qwen3.5 / Qwen 3 Next 则用 gated delta net 等 cheap layer 与 full attention 交替，体现了“混合架构”趋势。【字幕†source】

#### 学习提示

| 设计                     | 解决的问题                | 代价            |
| ------------------------ | ------------------------- | --------------- |
| full attention           | 全局 token 通信           | `O(n^2)` 成本高 |
| sliding window attention | 降低长上下文成本          | 单层只能看局部  |
| full/local interleaving  | 局部高效 + 周期性全局通信 | 结构更复杂      |

------

## 9. Python 代码示例详解

这一部分要解决的问题是：**Lecture 03 是否有独立 Python 文件？如果没有，原报告中的代码应如何学习？**

### 9.1 文件整体结构：本讲没有独立 `lecture_03.py`

原报告指出，GitHub 仓库中 Lecture 03 对应的是 `lecture_03.pdf`，而不是 `lecture_03.py`。官方课程页也把第 3 讲材料列为 lecture 3 PDF，GitHub 当前显示的对应文件为 `lecture_03.pdf`。

因此，本讲不存在可逐段解析的完整 Python 脚本，也没有 `main()`、函数定义或讲义生成流程可解释。原报告中的代码解析对象是 **PDF 第 35 页 RoPE 在 attention 模块中的代码截图**，不是独立 `.py` 文件。【PDF†source】

### 9.2 RoPE 代码截图逻辑解析

PDF 第 35 页展示的是典型 attention 模块中插入 RoPE 的位置。原报告整理出的代码大致如下：

```
query_states = self.q_proj(hidden_states)
key_states = self.k_proj(hidden_states)
value_states = self.v_proj(hidden_states)

query_states = query_states.view(
    bsz, q_len, self.num_heads, self.head_dim
).transpose(1, 2)

key_states = key_states.view(
    bsz, q_len, self.num_key_value_heads, self.head_dim
).transpose(1, 2)

value_states = value_states.view(
    bsz, q_len, self.num_key_value_heads, self.head_dim
).transpose(1, 2)

cos, sin = self.rotary_emb(value_states, position_ids)

query_states, key_states = apply_rotary_pos_emb(
    query_states, key_states, cos, sin
)
```

#### 逐段解释

第一步，`hidden_states` 是输入 hidden representation，通常形状为：
$$
(batch\_size, seq\_len, d_{\text{model}})
$$
第二步，`q_proj`、`k_proj`、`v_proj` 是三个线性投影：
$$
Q=XW_q,\quad K=XW_k,\quad V=XW_v
$$
它们把同一个 hidden state 投影成三种角色：

| 张量      | 作用                                    |
| --------- | --------------------------------------- |
| Q / query | 当前 token 用什么信息去“查询”其他 token |
| K / key   | 每个历史 token 提供什么索引信息供匹配   |
| V / value | 被 attention 权重加权汇聚的实际内容     |

第三步，`view(...).transpose(1, 2)` 把张量改成 multi-head attention 需要的形状：
$$
(b,n,d)
\rightarrow
(b,h,n,d_{\text{head}})
$$
这里：

- `b` 是 batch size；
- `n` 是 sequence length；
- `h` 是 head 数；
- `d_head` 是每个 head 的维度。

第四步，`num_heads` 与 `num_key_value_heads` 分开，说明代码兼容 GQA/MQA：query heads 可以多于 key/value heads。

第五步，`self.rotary_emb(value_states, position_ids)` 生成每个位置的 cos/sin。虽然函数调用中传入 `value_states` 可能用于推断 shape、device 或 dtype，但 RoPE 真正应用到的是 query 和 key。

第六步，`apply_rotary_pos_emb(query_states, key_states, cos, sin)` 对 Q/K 做旋转：
$$
Q'=R_{\text{pos}}Q,\quad K'=R_{\text{pos}}K
$$
第七步，旋转后的 Q/K 进入普通 attention：
$$
\operatorname{softmax}
\left(
\frac{Q'K'^\top}{\sqrt{d_k}}
\right)V
$$
字幕说明，RoPE 需要基于 position IDs 生成 cosine/sine angles，并应用到 queries 和 keys；可以用矩阵乘法实现，也可以手动旋转；它应该在每个 attention operation 中做，而不是只在最底层 embedding 处做一次。【字幕†source】

### 9.3 代码与课程知识点对应表

| 课程知识点         | 对应代码函数 / 变量 / 片段                                 | 代码功能                                                     | 对应理论原理                 | 在课程中的作用                  |
| ------------------ | ---------------------------------------------------------- | ------------------------------------------------------------ | ---------------------------- | ------------------------------- |
| Q/K/V projection   | `q_proj`, `k_proj`, `v_proj`                               | 从 hidden states 得到 Q、K、V                                | `Q=XW_q, K=XW_k, V=XW_v`     | 标准 self-attention 起点        |
| Multi-head reshape | `view(...).transpose(1, 2)`                                | 把 `(batch, seq, hidden)` 改成 `(batch, heads, seq, head_dim)` | 多头注意力按 head 并行计算   | 支持 MHA/GQA/MQA                |
| GQA/MQA 兼容       | `num_heads` vs `num_key_value_heads`                       | Q heads 和 KV heads 可以不同                                 | 减少 KV cache 内存访问       | 与后半 GQA/MQA 部署效率呼应     |
| RoPE 参数          | `cos`, `sin`, `position_ids`                               | 生成位置相关旋转角度                                         | 不同位置对应不同旋转矩阵     | 注入相对位置信息                |
| RoPE 应用          | `apply_rotary_pos_emb(query_states, key_states, cos, sin)` | 只旋转 Q/K，不旋转 V                                         | `R_i^T R_j = R_{j-i}`        | 让 attention score 依赖相对位置 |
| 后续 attention     | `softmax(QK^T / sqrt(d_k)) V`                              | 执行注意力加权汇聚                                           | scaled dot-product attention | 保持 Transformer 主结构不变     |

【PDF/代码截图†source】

### 9.4 学习提示

这段代码的复习重点不是背 API，而是看懂三件事：

1. RoPE 插在 Q/K projection 之后、attention score 之前；
2. Q heads 和 KV heads 分离，说明代码已经支持 GQA/MQA；
3. RoPE 不改变 attention 的基本形式，只改变 Q/K 的几何结构。

------

## 10. PDF / 幻灯片 / 图表解析

这一部分要解决的问题是：**本讲 PDF 中的图表分别服务于哪些课程结论？读图时应该抓住什么？**

### 10.1 主要图表总览

| 页码 / 图表 | 展示内容                                | 图表结构                                                     | 支持的课程结论                                               |
| ----------- | --------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 第 3-4 页   | 原始 Transformer vs 简单现代变体        | 第 3 页列 sine/cosine、ReLU、post-norm；第 4 页列 pre-LN、RoPE、SwiGLU、no bias | 本讲从“原始默认值”转向“现代默认值”【PDF†source】             |
| 第 10-12 页 | pre-norm / post-norm 与梯度稳定性       | 图示 residual path、validation loss、BLEU、gradient expectation、gradient spikes | pre-norm 不破坏 residual path，梯度传播更稳定【PDF†source】  |
| 第 14-17 页 | LayerNorm vs RMSNorm                    | 公式、FLOPs/runtime 表、Narang 实验表                        | RMSNorm 减少数据移动，可能带来 runtime 和性能收益【PDF†source】 |
| 第 21-25 页 | ReLU、GeLU、GLU 变体                    | 激活函数公式和 Shazeer / Narang 实验表                       | GLU 系列通常带来稳定小幅收益，成为现代共识【PDF†source】     |
| 第 30-35 页 | 位置编码与 RoPE                         | 位置编码公式、RoPE 相对内积条件、旋转图、代码截图            | RoPE 通过旋转 Q/K 让 attention 依赖相对位置【PDF†source】    |
| 第 37-51 页 | 超参数表和经验规律                      | `d_ff/d_model`、head ratio、aspect ratio、vocab size、dropout / weight decay | 许多看似复杂的超参数存在经验安全区【PDF†source】             |
| 第 52-56 页 | 稳定训练技巧                            | loss spike 图、z-loss 公式、QK norm 图、soft-capping         | 大规模训练中稳定性是核心工程目标【PDF†source】               |
| 第 59-64 页 | GQA/MQA 与推理成本                      | arithmetic intensity、KV cache、MQA/GQA 示意图               | KV cache 降 compute，但 memory access 成瓶颈；GQA 是推理效率与表达力折中【PDF†source】 |
| 第 65-66 页 | sliding window 与 interleaved attention | full attention 与 local/SWA 层交替                           | 长上下文模型常混合 full attention 与 cheap/local attention【PDF†source】 |

### 10.2 关键图表详细解释

#### 图表 1：原始 Transformer vs 现代变体

这组图表想回答的问题是：**现代语言模型到底改了 Transformer 的哪些默认值？**

读图时不要只看组件名称，而要看到背后的方向：

- position encoding 从加法式位置向量走向 RoPE；
- FFN 从 ReLU/GeLU 走向 GLU；
- norm 从 post-norm 走向 pre-norm / non-residual norm；
- bias 从默认保留走向默认删除。

这组图表是本讲的入口：后面每一节都在解释这些变化为什么发生。

#### 图表 2：pre-norm / post-norm 与梯度稳定性

这组图表想回答的问题是：**LayerNorm 放置位置为什么影响训练？**

读图时重点看：

- residual path 是否被 norm 截断；
- validation loss 或 BLEU 是否稳定；
- gradient expectation 是否在深层出现异常；
- gradient spike 是否更频繁。

这组图支持的结论是：pre-norm 的核心优势是稳定 residual stream，让深层模型更容易训练。【PDF†source】

#### 图表 3：LayerNorm vs RMSNorm runtime

这组图表想回答的问题是：**为什么一个 FLOPs 很少的操作会影响 runtime？**

读图时要区分：

- FLOPs 占比；
- runtime 占比；
- arithmetic intensity；
- memory movement。

它支持的结论是：RMSNorm 不只是数学简化，更是系统效率选择。矩阵乘法虽然 FLOPs 大，但 GPU 擅长；统计归一化虽然 FLOPs 小，却可能因为内存访问低效而拖慢训练。【PDF†source】

#### 图表 4：GLU 实验表

这组图表想回答的问题是：**SwiGLU / GeGLU 的收益是否只是经验噪声？**

读图时重点看不同激活函数在相近参数量下的 loss / performance 对比。它支持的结论不是“GLU 是唯一能训练的 FFN”，而是“GLU 变体通常有比较一致的小幅收益，因此成为现代默认值”。【PDF†source】

#### 图表 5：RoPE 旋转图与代码截图

这组图表想回答的问题是：**RoPE 如何让内积依赖相对位置？代码里应该在哪里插入 RoPE？**

读图时重点看：

- 同一 token 在不同绝对位置上被旋转到不同角度；
- 两个 token 的相对角度决定 QK 内积；
- RoPE 作用在 Q/K，而不是 V；
- 代码中 RoPE 位于 Q/K projection 之后、attention softmax 之前。

它支持的结论是：RoPE 把相对位置信息编码进 attention score 的几何结构中。【PDF†source】

#### 图表 6：GQA/MQA 与 KV cache

这组图表想回答的问题是：**为什么推理时要减少 K/V heads？**

读图时要抓住：

- MHA 每个 head 都有自己的 K/V；
- MQA 所有 query heads 共享一组 K/V；
- GQA 把 query heads 分组，共享较少数量的 K/V；
- KV cache 的内存访问是 incremental decoding 的瓶颈。

它支持的结论是：GQA 是现代模型在表达能力和推理效率之间的常见折中。【PDF†source】

------

## 11. 关键公式与推导汇总

| 公式                                           | 符号解释                                            | 适用场景               | 直觉解释                        | 代码 / 例子对应     |
| ---------------------------------------------- | --------------------------------------------------- | ---------------------- | ------------------------------- | ------------------- |
| `x_{l+1}=x_l+F(x_l)`                           | `x_l` 是 residual stream，`F` 是 attention/FFN 更新 | Transformer block      | 每层只增量修改表示              | residual connection |
| `x_{l+1}=LN(x_l+F(x_l))`                       | post-norm                                           | 原始 Transformer       | Norm 直接作用在 residual stream | post-norm 图示      |
| `x_{l+1}=x_l+F(LN(x_l))`                       | pre-norm                                            | 现代 LM                | 保持 residual stream clean      | pre-norm 图示       |
| `LayerNorm(x)=((x-E[x]) / sqrt(Var[x]+ε))⊙γ+β` | 均值、方差、缩放、偏移                              | 激活归一化             | 控制中心和尺度                  | LayerNorm           |
| `RMSNorm(x)=x / sqrt((1/d)∑x_i^2+ε)⊙γ`         | 只用 RMS 控制尺度                                   | 现代 LM norm           | 不减均值，减少计算和参数移动    | RMSNorm             |
| `FFN(x)=σ(xW_1)W_2`                            | 普通 FFN                                            | token-wise 非线性变换  | 生成中间特征再投回 hidden       | ReLU / GeLU FFN     |
| `GLU=(σ(xW)⊗xV)W_2`                            | `V` 是门控矩阵                                      | gated FFN              | 内容分支 × 门控分支             | SwiGLU / GeGLU      |
| `d_ff'=8/3 d_model`                            | GLU 参数匹配                                        | FFN 超参数             | 多一个矩阵，所以中间维度缩小    | SwiGLU 设置         |
| `<f(x,i),f(y,j)>=g(x,y,i-j)`                   | 只依赖相对位置                                      | RoPE 目标              | 平移不改变相对关系              | RoPE                |
| `R_i^T R_j=R_{j-i}`                            | 旋转矩阵组合                                        | RoPE 推导              | 相对角度决定内积                | RoPE 旋转图         |
| `Attention=softmax(QK^T/sqrt(d_k))V`           | 标准 attention                                      | MHA/GQA/MQA            | QK 决定权重，V 提供内容         | attention 代码      |
| `L=Σ[-logP(x_i)+α(logZ(x_i))^2]`               | z-loss                                              | 输出 softmax 稳定      | 惩罚 normalizer 漂移            | z-loss              |
| `S_ij=Norm(q_i)^T Norm(k_j)/sqrt(d_k)`         | QK norm                                             | attention softmax 稳定 | 控制 logits 尺度                | QK norm             |
| `\tilde z=c tanh(z/c)`                         | soft-capping                                        | logit 限幅             | 防止 logits 爆炸                | Gemma-style trick   |
| `1<g<h`                                        | GQA 中 KV heads 数                                  | 推理优化               | 介于 MHA 和 MQA 之间            | GQA                 |

------

## 12. 概念对比与易混淆点

| 易混淆概念                       | 区别                                                         | 联系                                    | 常见误解                                      | 正确理解                                         |
| -------------------------------- | ------------------------------------------------------------ | --------------------------------------- | --------------------------------------------- | ------------------------------------------------ |
| pre-norm vs post-norm            | pre-norm 把 norm 放在计算分支前；post-norm 放在 residual addition 后 | 都是 Transformer block 的 norm 位置选择 | 以为“前”一定好，“后”一定坏                    | 关键是 norm 是否破坏主 residual stream           |
| LayerNorm vs RMSNorm             | LayerNorm 减均值并除方差；RMSNorm 只按 RMS 缩放              | 都用于控制激活尺度                      | 以为 RMSNorm 理论上更强                       | RMSNorm 更流行主要因为够用且系统效率好           |
| no bias vs 有 bias               | no bias 删除线性层偏移                                       | 都能构建可训练模型                      | 以为 bias 一定必要                            | 在现代 LM 中 bias 收益小，可能不值得额外内存成本 |
| ReLU / GeLU vs SwiGLU / GeGLU    | GLU 多一个门控分支                                           | 都是 FFN 非线性                         | 以为不用 GLU 就不能训练                       | 非 GLU 也能训练，但 GLU 现代收益更稳定           |
| RoPE vs sine embedding           | sine 是加法位置向量；RoPE 是 Q/K 旋转                        | 都用 sin/cos                            | 以为二者本质一样                              | RoPE 目标是让 attention score 依赖相对位置       |
| relative embedding vs RoPE       | relative embedding 常向 logits 加偏置；RoPE 是 inner product factorization | 都表达相对位置                          | 以为所有相对位置方法等价                      | RoPE 的几何结构不同                              |
| dropout vs weight decay          | dropout 随机屏蔽激活；weight decay 惩罚权重                  | 都常被叫正则化                          | 以为它们主要都是防过拟合                      | LLM 预训练中 weight decay 常主要改变优化动态     |
| z-loss vs QK norm                | z-loss 作用于输出 softmax；QK norm 作用于 attention logits   | 都提升稳定性                            | 以为都是同一个 softmax trick                  | 它们控制不同 softmax                             |
| MQA vs GQA                       | MQA 共享所有 K/V；GQA 共享部分 K/V                           | 都降低 KV cache 成本                    | 以为越省 KV 越好                              | 还要考虑表达能力损失                             |
| sliding window vs full attention | sliding window 只看局部；full attention 看全局               | 都是 attention pattern                  | 以为 sliding window 能完全替代 full attention | 现代长上下文常交替使用两者                       |

------

## 13. 学习拓展与发散理解

本节是为帮助学习加入的补充，不把这些内容伪装成课程原文。

### 13.1 与大模型训练实践的关系

**补充说明：** 本讲几乎覆盖了现代 decoder-only LM 的“默认架构清单”。如果你要从零实现一个小型 LLaMA-like 语言模型，可以把本讲内容转化为如下工程决策：

| 实践环节        | 本讲相关内容                                 | 实际意义                            |
| --------------- | -------------------------------------------- | ----------------------------------- |
| 模型 block 设计 | pre-norm、RMSNorm、serial block              | 降低训练崩溃概率                    |
| FFN 实现        | SwiGLU / GeGLU、`8/3 d_model`                | 在参数量相近下提升表达能力          |
| 位置编码        | RoPE                                         | 支持相对位置建模和较好外推直觉      |
| attention 实现  | GQA/MQA 兼容                                 | 为后续推理部署降低 KV cache 成本    |
| 超参数选择      | `d_ff/d_model`、head dimension、aspect ratio | 减少盲目搜索空间                    |
| 稳定训练        | z-loss、QK norm、soft-capping                | 避免 loss spike 和 softmax 数值问题 |
| 长上下文        | sliding window + full attention              | 在长序列成本和表达力之间折中        |

### 13.2 与其他 lecture 的联系

**学习提示：** 本讲与前后内容的联系可以这样理解：

- 与前两讲的联系：前两讲讲 tokenization、PyTorch、FLOPs、memory、arithmetic intensity；本讲开始把这些系统直觉应用到架构选择，例如 RMSNorm、no bias、GQA。
- 与 Assignment 1 的联系：Assignment 1 要实现标准 Transformer LM，本讲解释了现代实现中为什么使用 pre-norm、RoPE、SwiGLU 等。
- 与后续系统 lecture 的联系：GQA/MQA、KV cache、arithmetic intensity 会在推理和系统优化中继续出现。
- 与 scaling laws 的联系：超参数不是随便选的；`d_ff/d_model`、深宽比、head dimension 都会影响在固定 compute 下的训练效率。
- 与评估和部署的联系：训练稳定性、推理 latency、长上下文能力最终都会影响模型能否实际使用。

### 13.3 面试 / 考试 / 作业可能考察角度

| 问题                                                         | 考察点                      | 回答思路                                                     |
| ------------------------------------------------------------ | --------------------------- | ------------------------------------------------------------ |
| 为什么现代 LM 大多使用 pre-norm？                            | residual stream、梯度稳定性 | 说明 post-norm 会让 norm 作用在主残差流上，而 pre-norm 保持信号路径干净 |
| RMSNorm 相比 LayerNorm 去掉了什么？为什么可能更快？          | norm 公式、系统效率         | 去掉均值中心化和 bias，减少低 arithmetic intensity 操作      |
| 推导 GLU 下 `d_ff=8/3 d_model`                               | 参数量计算                  | 普通 FFN 参数量 `8d_model^2`，GLU 有 3 个矩阵，令二者相等    |
| 为什么 RoPE 的 attention score 只依赖相对位置？              | 旋转矩阵推导                | 写出 `R_i^T R_j=R_{j-i}`                                     |
| QK norm 和 z-loss 分别控制什么？                             | softmax 稳定性              | z-loss 控制输出 softmax normalizer；QK norm 控制 attention logits 尺度 |
| 为什么 weight decay 在 LLM 预训练中不一定是传统正则化？      | optimization dynamics       | 数据足够大时 overfitting 不是主因，weight decay 可能主要改变优化轨迹 |
| MQA 为什么能降低 KV cache 成本？                             | inference memory access     | 多个 query heads 共享 K/V，减少缓存大小                      |
| GQA 为什么是 MHA 和 MQA 的折中？                             | 表达力与效率 trade-off      | 保留少数 KV groups，降低成本同时减少性能损失                 |
| 为什么长上下文模型常交替 full attention 和 sliding window attention？ | 长上下文成本                | sliding window 降成本，full attention 周期性全局通信         |
| 设计一个新 LM 时哪些默认值可以直接采用？                     | 现代架构共识                | pre-norm、RMSNorm、SwiGLU、RoPE、GQA、合理 `d_ff` 和 head dimension |

------

## 14. 课程复习总结

### 14.1 本讲核心思想

本讲的核心思想是：

> 现代大语言模型架构不是从一个单一优雅理论直接推导出来的，而是在表达能力、训练稳定性、GPU 系统效率、推理成本和长上下文需求之间形成的一组经验折中。

字幕结尾也总结：跨模型观察会看到许多共性和好默认值，同时也会看到在 context handling、position embeddings、tokenization 等方面仍存在明显差异。【字幕†source】

### 14.2 必须掌握的概念

复习本讲时，至少要掌握以下概念：

- residual stream 与 “keep your residual stream clean”；
- post-norm、pre-norm、non-residual norm、double norm；
- LayerNorm 与 RMSNorm 的公式和差异；
- 删除 bias 的系统动机；
- ReLU、GeLU、SwiGLU、GeGLU、GLU；
- GLU 中 `d_ff = 8/3 d_model` 的参数量推导；
- RoPE 的相对位置内积目标；
- RoPE 的旋转推导 `R_i^T R_j = R_{j-i}`；
- `d_ff/d_model`、head dimension、aspect ratio；
- vocab size 与单语 / 多语模型差异；
- dropout 和 weight decay 在 LLM 预训练中的角色；
- z-loss、QK norm、logit soft-capping；
- KV cache、MQA、GQA；
- sliding window attention 与 full/local interleaving。

### 14.3 必须理解的公式

| 公式                                  | 解决的问题                             |
| ------------------------------------- | -------------------------------------- |
| `x_{l+1}=x_l+F(Norm(x_l))`            | pre-norm 如何保持 residual stream 稳定 |
| `RMSNorm(x)=x/sqrt((1/d)∑x_i^2+ε)⊙γ`  | RMSNorm 如何只控制尺度                 |
| `FF_GLU(x)=(σ(xW)⊗xV)W_2`             | GLU 如何通过门控增强 FFN               |
| `d_ff'=8/3 d_model`                   | GLU 参数量如何与普通 FFN 匹配          |
| `<f(x,i),f(y,j)>=g(x,y,i-j)`          | RoPE 想实现的相对位置目标              |
| `R_i^T R_j=R_{j-i}`                   | RoPE 为什么只依赖相对位置              |
| `softmax(QK^T/sqrt(d_k))V`            | attention 主公式                       |
| `L=Σ[-logP(x_i)+α(logZ(x_i))^2]`      | z-loss 如何稳定输出 softmax            |
| `S_ij=Norm(q_i)^TNorm(k_j)/sqrt(d_k)` | QK norm 如何稳定 attention softmax     |
| `\tilde z=c tanh(z/c)`                | soft-capping 如何限制 logits           |

### 14.4 必须看懂的代码

本讲没有独立 `lecture_03.py`，但必须看懂 PDF 第 35 页 RoPE 代码截图中的几件事：

1. `q_proj/k_proj/v_proj` 如何从 hidden states 得到 Q/K/V；
2. `view(...).transpose(1,2)` 如何把张量改成 multi-head attention 形状；
3. `num_heads` 和 `num_key_value_heads` 分离如何支持 GQA/MQA；
4. `rotary_emb(..., position_ids)` 如何生成 cos/sin；
5. `apply_rotary_pos_emb(query_states, key_states, cos, sin)` 为什么只作用于 Q/K；
6. 旋转后的 Q/K 如何进入标准 attention。

【PDF/代码截图†source】

### 14.5 一句话总结

**Lecture 03 的主线是：从原始 Transformer 出发，通过观察现代大模型的共同架构选择，理解哪些默认值已经成为共识，哪些选择由训练稳定性和 GPU 系统效率驱动，哪些部分正在为推理成本和长上下文需求继续演化**