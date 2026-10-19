# Lecture 1 & Lecture 2 课程报告：从语言模型技术栈到资源账本——Overview、Tokenization、PyTorch 与 Resource Accounting



## 0. 资料来源与改写说明

本报告基于你上传的原始中文课程报告 `lec1,2.md` 改写而成。原报告覆盖 **CS336 Lecture 1: Overview, Tokenization** 与 **Lecture 2: PyTorch / einops / Resource Accounting**，并已整理了字幕主线、GitHub Python 可执行讲义、公式、代码结构和图表说明。 

本次改写还核对了 Stanford CS336 官方 GitHub 仓库 `stanford-cs336/lectures`。该仓库说明 Spring 2026 CS336 lectures 是 Stanford “Language Modeling from Scratch”的课程资料，且可执行讲义命名为 `lecture_XX.py`，非可执行讲义命名为 `lecture_XX.pdf`。仓库目录中明确列出 `lecture_01.py`、`lecture_02.py`，随后才出现 `lecture_03.pdf`、`lecture_04.pdf` 等文件，因此 Lecture 1/2 本报告以 Python 可执行讲义和其中嵌入的图片为主要 GitHub 资料，而不伪造 `lecture_01.pdf` 或 `lecture_02.pdf`。

引用格式说明：

- **【字幕†原报告】**：表示该内容来自原报告中已整理的字幕主线；本轮未单独提供可逐行核对的字幕文件，因此不新增伪造字幕引用。
- **【Python代码】**：表示该内容来自 GitHub 中 `lecture_01.py` 或 `lecture_02.py` 的代码讲义。
- **【图表/代码】**：表示图表不是单独 PDF，而是 Python 讲义中通过 `image(...)` 调用的图片或可视化。
- **延伸理解 / 学习提示 / 工程直觉**：表示为了帮助学习而加入的解释，不把它伪装成课程原文。

------

## 1. 学习导读

### 1.1 本讲要解决的核心问题

这两讲共同回答一个问题：

> **如果要真正理解并训练语言模型，应该先从哪里开始？**

Lecture 1 的答案是：先理解整门课的目标——**通过从零构建来获得底层理解**，然后从语言模型输入管线的第一步 **tokenization** 开始。

Lecture 2 的答案是：理解模型之前，还必须会算账——**每个 tensor 占多少显存，每次矩阵乘法做多少 FLOPs，训练一步为什么近似是 $6ND$，以及 GPU 瓶颈到底在计算还是内存移动**。

这两讲合起来构成 CS336 的入口：Lecture 1 讲“为什么要构建”和“模型看见什么输入单位”，Lecture 2 讲“构建时资源怎么花、瓶颈在哪里”。

### 1.2 本讲在 CS336 课程中的位置

CS336 的主线不是单纯介绍 Transformer，而是训练学生形成一种能力：

> **在固定 compute、memory、data 和时间预算下，设计并训练尽可能好的语言模型。**

Lecture 1 给出整门课地图：basics、systems、scaling laws、data、alignment。`lecture_01.py` 的 `main()` 按 `welcome()`、`why_this_course_exists()`、`current_lm_landscape()`、`course_syllabus()`、`tokenization()` 的顺序组织，最后进入 tokenization 单元，说明本讲从课程动机逐步过渡到第一个技术主题。【Python代码】

Lecture 2 则进入 resource accounting。`lecture_02.py` 的 `main()` 明确写出本讲目标：从上一讲的 overview/tokenization 转到 resource accounting，并依次讲 tensors、tensor memory、GPU、einops、FLOPs、arithmetic intensity、deep network、gradients、optimizer、train loop、gradient accumulation 和 activation checkpointing。【Python代码】

### 1.3 本讲知识结构

| 模块                            | 解决的问题                                        | 核心概念                                                     | 与后续内容的关系                                             |
| ------------------------------- | ------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Lecture 1：课程哲学             | 为什么要从零构建语言模型                          | from scratch、leaky abstractions、mechanics / mindset / intuitions | 决定整门课不是“调用 API”，而是理解 tokenizer、模型、优化器、系统和数据 |
| Lecture 1：效率主线             | 为什么 fixed resources 下效率最重要               | efficiency × resources、bitter lesson                        | 后续 systems、scaling laws、data filtering 都围绕效率展开    |
| Lecture 1：Tokenization         | 如何把字符串变成模型可处理的整数序列              | encode/decode、round trip、compression ratio                 | Assignment 1 的第一部分，也是语言模型输入管线的起点          |
| Lecture 1：BPE                  | 如何兼顾可逆性、固定词表和较短序列                | byte-level vocab、pair counting、merge、vocab / merges       | 现代 LLM tokenizer 的重要基础                                |
| Lecture 2：Tensor 与 dtype      | 训练中到底存了什么，占多少显存                    | tensor rank、shape、numel、element_size、fp32/fp16/bf16      | 后续模型实现、显存估算、混合精度训练的基础                   |
| Lecture 2：einops               | 如何写出不容易搞错维度的张量代码                  | einsum、reduce、rearrange、命名维度                          | Transformer attention、MLP、multi-head reshape 的代码表达基础 |
| Lecture 2：FLOPs 与 MFU         | 如何估算计算量与硬件利用率                        | FLOPs、FLOP/s、MFU、benchmark                                | 后续 kernel、profiling、训练速度优化的基础                   |
| Lecture 2：Arithmetic intensity | 为什么有些操作算力很高却跑不满 GPU                | memory-bound、compute-bound、roofline                        | 解释 elementwise、matvec、matmul、inference decode 的瓶颈    |
| Lecture 2：训练资源账本         | forward/backward/optimizer/activations 如何占资源 | $6ND$、optimizer state、activation memory                    | 大模型训练预算、batch size、checkpointing、parallelism 的基础 |
| Lecture 2：显存优化             | 显存不够时怎么办                                  | gradient accumulation、activation checkpointing              | 后续大模型训练系统和分布式训练的前置知识                     |

### 1.4 学习本讲前需要知道什么

本讲不要求已经会完整训练 LLM，但最好具备以下基础：

- **概率与语言模型直觉**：知道语言模型通常预测 token 序列的概率。
- **Python / PyTorch 基础**：知道 tensor、shape、dtype、autograd、optimizer 的基本含义。
- **矩阵乘法**：知道 $XW$ 的维度规则和点积计算方式。
- **反向传播直觉**：知道 backward 会计算参数梯度。
- **Transformer 基本印象**：知道 Transformer 中常见 batch、sequence、heads、hidden 等维度。
- **GPU 基本概念**：知道 GPU 有显存 HBM、算力 FLOP/s、内存带宽 bytes/s。

------

# Part A：Lecture 1——Overview 与 Tokenization

## 2. 第一部分：为什么要“从零构建”语言模型

### 2.1 问题引入

Lecture 1 从一个现实问题开始：现在很多研究者和工程师可以直接通过 GPT、Claude、Gemini 等 API 使用强大模型，但这也带来一个风险——人们越来越熟悉 prompt，却越来越不理解底层技术栈。【字幕†原报告】

这种抽象层当然提升了生产力，但对于基础研究而言，只会调用模型是不够的。因为一旦问题涉及 tokenizer、architecture、optimizer、training data、parallelism、inference system 或 alignment objective，prompt 层面的调节就无法触及核心。

### 2.2 概念定义：from scratch 到底是什么意思

**直觉解释**：from scratch 不是“所有东西都不用库”，而是“亲手实现最能帮助理解的关键部件”。

**正式理解**：课程希望学生通过构建 tokenizer、Transformer、loss、optimizer、training loop、resource accounting 等核心组件，理解语言模型从数据到训练再到推理的完整链条。

`lecture_01.py` 的开头也体现了这个思想：课程保持 “from scratch” 哲学，同时优先讲 high value-per-time concepts，并加入 mixture of experts、long-context、agents 等现代语言模型内容。【Python代码】

### 2.3 为什么 prompt 不够

只会 prompt，意味着只能在模型外部改变输入形式。它适合应用开发，但不适合回答这些底层问题：

- tokenizer 为什么这样切词？
- 为什么某个架构更省显存或更快？
- 为什么某个 optimizer 更稳定？
- 为什么同样 FLOPs 下，有的模型训练更好？
- 为什么 inference decode 经常 memory-bound？
- 为什么数据过滤和去重会影响训练效率？

**学习提示**：CS336 的目标不是否定 API，而是让学生知道 API 背后发生了什么；只有理解底层，才能判断何时应该信任抽象、何时必须撕开抽象层。

### 2.4 Frontier models 的工业化与不透明

Lecture 1 强调，frontier models 的训练已经高度工业化。原报告提到，GPT-4 训练成本曾被报道为上亿美元量级，frontier 模型的架构、数据、训练细节往往不完全公开。【字幕†原报告】

这意味着课程不能简单复刻闭源大模型训练流程。课程更现实的目标是：在小模型和公开资料上学到可迁移的 **mechanics** 和 **mindset**。

### 2.5 Mechanics、Mindset、Intuitions

课程把学习内容分成三类：

| 类型       | 这是什么                                                     | 能否从课程中学到 | 复习重点                         |
| ---------- | ------------------------------------------------------------ | ---------------- | -------------------------------- |
| Mechanics  | 机制，例如 Transformer、BPE、model parallelism、kernel、optimizer | 可以较好学习     | 知道组件如何工作、输入输出是什么 |
| Mindset    | 思维方式，例如认真做 resource accounting、profile、benchmark、scaling | 可以重点训练     | 遇到问题先估算资源和瓶颈         |
| Intuitions | 大规模建模直觉，例如哪些数据/模型决策在 frontier scale 有效  | 只能部分学习     | 小模型结论不能盲目推广到大模型   |

**延伸理解**：
 小模型实验适合学习“机制”和“方法论”，但不一定能复现大模型能力的全部现象。例如，某些能力可能在规模达到一定阈值后才显著出现；某些瓶颈在小模型上不是瓶颈，到大模型上才变得关键。

### 2.6 Bitter lesson 的正确理解

课程用一个简化关系组织全课：
$$
\text{accuracy} = \text{efficiency} \times \text{resources}
$$
其中：

- $\text{accuracy}$：最终模型表现；
- $\text{resources}$：投入的 compute、data、memory、bandwidth、工程时间等；
- $\text{efficiency}$：单位资源产生有效性能的能力。

这里的重点不是“算法不重要”，而是：

> **能随规模扩展、能高效使用资源的算法才重要。**

**学习提示**：大规模训练中，5% 的效率差距也可能对应非常大的成本差异。因此从 Lecture 1 开始，课程就不断强调：不要只问“模型能不能训练”，还要问“资源是否被有效利用”。

------

## 3. 第二部分：语言模型发展脉络与课程地图

### 3.1 从 n-gram 到 neural language models

Lecture 1 用语言模型历史说明：今天的 LLM 不是凭空出现的，而是经历了从统计语言模型到神经网络语言模型、再到 foundation models 的演进。【字幕†原报告】

原报告将发展脉络整理为：
$$
\text{n-gram} \rightarrow \text{LSTM / neural LM} \rightarrow \text{seq2seq / attention}
\rightarrow \text{Transformer} \rightarrow \text{foundation models} \rightarrow \text{chat / agents}
$$
`lecture_01.py` 的 `current_lm_landscape()` 也列出 Shannon、n-gram、LSTM、Bengio neural LM、seq2seq、Adam、attention、Transformer、MoE、model parallelism 等关键节点。【Python代码】

### 3.2 从 fine-tuning 到 prompting，再到 agents

课程强调，过去十年人们对语言模型的使用方式不断变化：
$$
\text{BERT时代：fine-tune}
\rightarrow
\text{GPT-3时代：prompt}
\rightarrow
\text{ChatGPT时代：conversation}
\rightarrow
\text{Agent时代：act autonomously}
$$
这条变化说明，语言模型已经不只是一个“文本补全器”，而正在变成可以参与复杂任务、工具调用和长期交互的系统组件。【字幕†原报告】

**延伸理解**：
 这也是为什么课程不仅讲 architecture，还要讲 data、systems、alignment 和 agents。现代语言模型的能力不只来自参数量，也来自数据管线、训练目标、系统效率和后训练过程。

### 3.3 课程五大模块

Lecture 1 给出全课程地图：

| Assignment   | 模块         | 核心目标                                                     | 本讲如何铺垫                                      |
| ------------ | ------------ | ------------------------------------------------------------ | ------------------------------------------------- |
| Assignment 1 | Basics       | BPE tokenizer、Transformer、loss、optimizer、training loop、resource accounting | Lecture 1/2 直接服务于 A1                         |
| Assignment 2 | Systems      | kernels、parallel training、inference、profiling             | Lecture 2 开始讲 memory/compute                   |
| Assignment 3 | Scaling laws | 用小实验预测大规模训练                                       | Lecture 1 引入 $C=6ND$ 和 scaling recipe          |
| Assignment 4 | Data         | evaluation、curation、filtering、deduplication、mixing       | Lecture 1 强调数据决定模型能力                    |
| Assignment 5 | Alignment    | DPO、GRPO、RLHF / weak supervision                           | Lecture 1 说明从 next-token prediction 到偏好优化 |

`lecture_01.py` 的 `course_syllabus()` 中，Assignment 1 明确包括实现 BPE tokenizer、Transformer、cross-entropy loss、AdamW optimizer、training loop，以及做 resource accounting；同一部分还把 expressivity、stability、efficiency 作为高层设计原则。【Python代码】

------

## 4. 第三部分：Tokenization——语言模型输入管线的第一层

### 4.1 问题引入

语言模型不能直接处理任意 Unicode 字符串。模型内部通常处理的是整数 token 序列，因此需要 tokenizer 在字符串和整数序列之间转换。

直觉上，tokenizer 就像“文本到模型输入”的翻译器：
$$
\text{raw string}
\rightarrow
\text{tokens}
\rightarrow
\text{model}
$$
如果 tokenizer 设计不好，模型看到的序列可能过长、词表可能过大，或者文本信息可能在编码/解码中丢失。

### 4.2 Tokenizer 的正式定义

Tokenizer 的核心接口是：
$$
\text{encode}: \text{string} \rightarrow [\text{int}]
$$
理想情况下应该满足：
$$
\text{decode}(\text{encode}(x)) = x
$$
这叫 **round trip**。如果 round trip 不成立，说明 tokenizer 在编码和解码过程中破坏了原文本信息。

`lecture_01.py` 中的 `Tokenizer` 是抽象接口，要求实现 `encode()` 与 `decode()`；讲义还明确说，tokenizer 负责在 raw inputs / bytes 和 token integer sequences 之间转换。【Python代码】

### 4.3 Compression ratio：为什么序列长度很重要

课程定义：
$$
\text{compression ratio}
=
\frac{\#\text{UTF-8 bytes}}{\#\text{tokens}}
$$
符号解释：

- $\#\text{UTF-8 bytes}$：原始字符串编码成 UTF-8 后的字节数；
- $\#\text{tokens}$：tokenizer 编码后得到的 token 数；
- compression ratio 越高，表示平均每个 token 承载更多 byte，序列更短。

`lecture_01.py` 的 `get_compression_ratio()` 正是用 UTF-8 bytes 数除以 token 数；代码还说明，压缩率越大，序列越短，而这对 Transformer 很重要，因为 attention 在序列长度上的复杂度近似是二次的。【Python代码】

**学习提示**：
 不要把 compression ratio 简单理解为“越高越好”。如果通过无限增大 vocabulary size 来压缩序列，很多 token 会非常稀疏，模型很难学好。因此 tokenizer 设计本质上是在 **序列长度** 和 **词表大小 / 稀疏性** 之间折中。

### 4.4 真实 tokenizer 的“奇怪现象”

课程提到现代 tokenizer 会出现一些不符合人类直觉的现象：

| 现象                                      | 直觉解释                     | 学习重点                             |
| ----------------------------------------- | ---------------------------- | ------------------------------------ |
| `" world"` 可能是一个 token               | 前导空格和词一起出现频率高   | token 不等于 word                    |
| 句首 `"hello"` 和中间 `" hello"` 可能不同 | 是否带空格会影响 byte 序列   | tokenizer 是统计系统，不是语言学词典 |
| 数字常被每几位切开                        | 数字组合非常多，整词表征稀疏 | 数字能力受 tokenizer 影响            |
| 多语言 tokenizer 词表很大                 | 要覆盖更多脚本和语言片段     | 多语言建模对 tokenizer 要求更高      |

`lecture_01.py` 的 `tokenization_examples()` 明确列出：词与前导空格可能在同一个 token 中，句首词和中间词表示不同，数字会按几位一组切分，并展示用 `tiktoken` 的 `o200k_base` 进行 encode/decode round trip。【Python代码】

------

## 5. 第四部分：三种朴素 tokenizer 为什么不够好

### 5.1 Character tokenizer

**直觉解释**：每个 Unicode 字符就是一个 token。
$$
\text{encode}(s)
=
[\operatorname{ord}(c_1), \operatorname{ord}(c_2), \dots]
$$
其中：

- $c_i$：字符串中的第 $i$ 个字符；
- $\operatorname{ord}(c_i)$：字符对应的 Unicode code point。

优点：

- 简单；
- 可逆；
- 不需要复杂训练。

缺点：

- Unicode 字符数量非常大；
- 很多字符极少出现，词表利用率差；
- 字符粒度太细，序列可能很长；
- 对现代 Transformer 来说计算效率低。

`lecture_01.py` 中 `character_tokenizer()` 用 `ord()` 编码、`chr()` 解码，并指出 Unicode 字符规模大、很多字符稀有，是 large vocabulary 与 low compression ratio 的“双输”。【Python代码】

### 5.2 Byte tokenizer

**直觉解释**：把字符串先转成 UTF-8 bytes，每个 byte 是一个 token。
$$
\text{encode}(s)=\text{list}(\text{UTF-8 bytes}(s))
$$
优点：

- 基础词表固定为 256；
- 任意 Unicode 字符串都可以表示；
- 可逆；
- 没有 UNK 问题。

缺点：

- compression ratio 为 1；
- 序列非常长；
- attention 成本随序列长度二次增长，因此效率差。

`lecture_01.py` 中 `byte_tokenizer()` 明确说明 byte 可以表示 256 个值，compression ratio 为 1，并指出这种压缩率会导致序列太长，不适合 Transformer 的有限 context length。【Python代码】

### 5.3 Word tokenizer

**直觉解释**：按照空格或正则表达式把文本切成词。

原报告中的代码示例使用：

```
regex.findall(r"\w+|.", string)
```

它会把连续字母数字组成一个 chunk，其余字符单独处理。

优点：

- token 往往具有语义；
- compression ratio 较好；
- 与人类语言直觉接近。

缺点：

- 词表可能非常大；
- 新词、罕见词、拼写变体会造成开放词表问题；
- 测试时遇到未知词需要 UNK；
- UNK 会丢失信息，也会影响 perplexity 计算。

`lecture_01.py` 中 `word_tokenizer()` 用 `regex.findall(r"\w+|.", string)` 演示词级切分，并指出 word tokenizer 压缩率好但词表可能巨大，且未知词需要特殊 UNK token。【Python代码】

### 5.4 三者对比小结

| 方法      | 词表大小   | 序列长度 | 可逆性   | 主要问题                 |
| --------- | ---------- | -------- | -------- | ------------------------ |
| Character | 大         | 中等偏长 | 通常可逆 | Unicode 字符多且稀疏     |
| Byte      | 256        | 很长     | 可逆     | compression ratio 太低   |
| Word      | 可能无限大 | 较短     | 依赖实现 | 词表开放、UNK、稀有词    |
| BPE       | 可控       | 较短     | 可逆     | 贪心启发式，不是全局最优 |

------

## 6. 第五部分：Byte Pair Encoding（BPE）

### 6.1 BPE 要解决什么问题

BPE 的目标是在 byte tokenizer 和 word tokenizer 之间取得折中：

- 保留 byte-level 的开放词汇能力；
- 不需要 UNK；
- 保持 encode/decode 可逆；
- 用频繁出现的 byte sequence 组成更长 token；
- 让常见片段更短、稀有片段回退到 byte 或更小 token；
- 控制 vocabulary size；
- 缩短 Transformer 看到的序列。

`lecture_01.py` 中 `bpe_tokenizer()` 说明 BPE 最初由 Philip Gage 用于数据压缩，后来被用于神经机器翻译，再被 GPT-2 使用；讲义给出的核心直觉是：常见 byte 序列由一个 token 表示，稀有序列由多个 token 表示，并从每个 byte 作为 token 开始，反复合并最常见相邻 pair。【Python代码】

### 6.2 BPE 训练算法

给定训练字符串 $s$，BPE 的训练过程可以写成：

1. 把字符串转成 UTF-8 byte 序列；
2. 初始 vocabulary 包含 256 个 byte token；
3. 统计当前 token 序列中所有相邻 pair 的出现次数；
4. 找到最高频 pair；
5. 分配一个新 token id；
6. 用新 token 替换所有非重叠出现的该 pair；
7. 把新 token 对应的 byte 串加入 vocabulary；
8. 重复若干次。

形式化地，若当前 token 序列为：
$$
[i_1, i_2, \dots, i_n]
$$
统计所有相邻 pair：
$$
(i_j, i_{j+1})
$$
选择：
$$
p^\* = \arg\max_p \operatorname{count}(p)
$$
然后分配：
$$
v_{\text{new}} = 256 + t
$$
并用 $v_{\text{new}}$ 替换 $p^\*$ 的出现。

`lecture_01.py` 的 `train_bpe()` 正是这样实现：初始 `indices = list(map(int, string.encode("utf-8")))`，`vocab = {x: bytes([x]) for x in range(256)}`，每轮用 `count_adjacent_pairs(indices)` 统计 pair，用 `max(counts, key=counts.get)` 找最高频 pair，再设置 `new_index = 256 + i`，更新 `merges` 与 `vocab`，最后调用 `merge(...)` 更新 token 序列。【Python代码】

### 6.3 BPE 的关键数据结构

```
@dataclass(frozen=True)
class BPETokenizerParams:
    vocab: dict[int, bytes]
    merges: dict[tuple[int, int], int]
```

| 结构                   | 含义                              | 作用                                           |
| ---------------------- | --------------------------------- | ---------------------------------------------- |
| `vocab`                | token id → byte string            | decode 时把 token 还原为 bytes                 |
| `merges`               | `(old_id_1, old_id_2)` → `new_id` | encode 时按照训练得到的合并规则压缩 token 序列 |
| `indices`              | 当前 token 序列                   | BPE 训练和编码时不断被合并                     |
| `count_adjacent_pairs` | 相邻 pair 计数                    | 找到最值得合并的局部结构                       |
| `merge`                | 替换 pair                         | 把高频相邻 pair 压成新 token                   |

**学习提示**：
 BPE 的核心不是“理解词义”，而是“统计上压缩常见 byte 片段”。它能产生类似词或子词的单位，是因为语言数据中的常见片段本身具有统计稳定性。

### 6.4 BPE 的使用例子

原报告中保留了课程例子：

```
params = train_bpe("the cat in the hat", num_merges=3)
tokenizer = BPETokenizer(params)
indices = tokenizer.encode("the quick brown fox")
reconstructed_string = tokenizer.decode(indices)
assert string == reconstructed_string
```

这个例子说明：

- tokenizer 可以在一段文本上训练；
- 训练得到的 `vocab` 和 `merges` 可以用于新文本；
- 即使新文本中出现训练文本没有完整出现过的词，也能回退到 byte 或较小片段；
- 最后仍然需要 round trip 保证不丢信息。

`lecture_01.py` 中的 `bpe_tokenizer()` 正是用 `"the cat in the hat"` 训练 3 次 merge，然后编码 `"the quick brown fox"` 并检查解码后等于原字符串。【Python代码】

### 6.5 Assignment 1 中要进一步优化的地方

课程中的 BPE 实现是教学版，不是工业级实现。`lecture_01.py` 说明 Assignment 1 会要求学生改进：当前 `encode()` 会遍历所有 merges，需要只遍历相关 merges；还要检测并保留 special tokens，例如 `<|endoftext|>`；使用 pre-tokenization，例如 GPT-2 tokenizer regex；并尽量提高实现速度。【Python代码】

**工程直觉**：
 BPE 真正在大规模训练中使用时，速度非常重要。因为 tokenization 发生在整个数据管线前端，如果 tokenizer 太慢，会拖慢数据预处理和训练输入供给。

------

# Part B：Lecture 2——PyTorch、einops 与 Resource Accounting

## 7. 第六部分：Resource accounting 的目标

### 7.1 问题引入

Lecture 2 从两个数量级问题开始：

1. 用 1024 张 H100 在 15T tokens 上训练 70B 参数模型，大约需要多久？
2. 8 张 H100 使用 AdamW，不做特别优化，最多能装下多大模型？

这两个问题的重点不是得到完美精确答案，而是训练 **napkin math** 能力：先用简单公式抓住数量级，再判断方案是否可行。

`lecture_02.py` 的 `motivating_questions()` 对第一个问题使用 `total_flops = 6 * 70e9 * 15e12`，假设 H100 dense bf16 FLOP/s 为 `1979e12 / 2`、MFU 为 0.5，并估算 1024 张卡每天提供的 FLOPs；第二个问题用 H100 80GB 显存和 AdamW 每参数 `2 + 2 + (4 + 4)` bytes 估算参数上限，并明确说明未计入 activations，因此只是 upper bound。【Python代码】

### 7.2 Resource accounting 是什么

Resource accounting 是训练过程的资源账本，主要回答：

- 参数占多少显存？
- 梯度占多少显存？
- optimizer state 占多少显存？
- activations 占多少显存？
- forward 做多少 FLOPs？
- backward 做多少 FLOPs？
- optimizer step 额外占多少 compute 和 memory？
- 当前操作是 memory-bound 还是 compute-bound？
- 实际 FLOP/s 和硬件峰值差距有多大？

**一句话总结**：
 Resource accounting 是把“训练模型很贵”变成可以计算、可以比较、可以优化的账本。

### 7.3 为什么它重要

在小模型训练中，资源估算不准可能只是慢一点；在大模型训练中，估算错误可能导致：

- 模型放不进显存；
- batch size 过小，训练不稳定；
- GPU 跑不满，训练成本浪费；
- optimizer state 或 activation memory 爆炸；
- 训练时间和预算严重偏离预期。

`lecture_02.py` 的 `main()` 把本讲 takeaway 分为 mechanics、mindset、intuitions：mechanics 是 PyTorch semantics，mindset 是 resource accounting，intuitions 是理解资源如何花掉，没有“ML magic”。【Python代码】

------

## 8. 第七部分：Tensor、shape、dtype 与 memory

### 8.1 Tensor 是训练中的统一存储单位

深度学习训练中的几乎所有对象都是 tensor：

- data；
- parameters；
- gradients；
- optimizer state；
- activations。

`lecture_02.py` 的 `tensors_basics()` 明确列出这些对象都由 tensors 存储，并展示 rank 1 vector、rank 2 matrix、rank 3 tensor，以及 Transformer 中常见的 rank 4 tensor。【Python代码】

### 8.2 Transformer 中的 rank-4 tensor

课程示例：

```
B = 32  # batch size
S = 16  # sequence length
H = 16  # number of heads
D = 64  # hidden dimension per head
x = torch.zeros(B, S, H, D)
```

数学表示为：
$$
x \in \mathbb{R}^{B \times S \times H \times D}
$$
符号解释：

- $B$：batch size，一次处理多少样本；
- $S$：sequence length，每个样本有多少 token；
- $H$：attention heads 数；
- $D$：每个 head 的 hidden dimension。

**学习提示**：
 很多 Transformer bug 都来自维度搞错。后面引入 einops，就是为了让这些维度从“位置编号”变成“有名字的语义维度”。

### 8.3 Tensor memory 公式

任意 tensor 的显存占用可以用：
$$
M(x)
=
\operatorname{numel}(x)
\times
\operatorname{element\_size}(x)
$$
其中：

- $\operatorname{numel}(x)$：tensor 元素个数；
- $\operatorname{element\_size}(x)$：每个元素占多少 bytes。

`lecture_02.py` 中 `get_memory_usage(x)` 就返回 `x.numel() * x.element_size()`；讲义用一个 $4 \times 8$ 的 float32 tensor 验证：元素数 $4\times 8=32$，每个元素 4 bytes，总共 128 bytes。【Python代码】

### 8.4 fp32、fp16、bf16

| dtype | 每元素 bytes | 优点                                            | 风险 / 代价                                 |
| ----- | ------------ | ----------------------------------------------- | ------------------------------------------- |
| fp32  | 4            | 精度和动态范围较好，传统 baseline               | 显存和带宽开销大                            |
| fp16  | 2            | 节省显存，提高吞吐                              | dynamic range 差，容易 underflow / overflow |
| bf16  | 2            | 保留接近 fp32 的 exponent range，更适合深度学习 | mantissa 少，分辨率较低                     |

`lecture_02.py` 中展示了 fp16 的 underflow：`torch.tensor([1e-8], dtype=torch.float16)` 会变成 0；而 bf16 同样占 2 bytes，却因为 exponent range 接近 fp32，更适合深度学习训练中的大动态范围数值。【Python代码】

**学习提示**：
 深度学习中，很多时候更需要“数值不要溢出或下溢”，而不是每个数字都有非常高的小数精度。因此 bf16 在 LLM 训练中非常常见。

### 8.5 Mixed precision

混合精度训练的典型思想：

| 对象             | 常用 dtype | 原因                              |
| ---------------- | ---------- | --------------------------------- |
| parameters       | bf16       | 节省参数显存                      |
| activations      | bf16       | activation 很大，bf16 通常足够    |
| gradients        | bf16       | 减少梯度显存与通信量              |
| optimizer states | fp32       | 一阶/二阶统计长期累积，需要更稳定 |

**工程直觉**：
 optimizer state 常常不是主要 FLOPs 瓶颈，却是显存瓶颈。AdamW 通常要保存一阶矩和二阶矩，所以每参数额外显存很可观。

### 8.6 Tensor 在 CPU 与 GPU 之间

PyTorch 默认在 CPU 上创建 tensor。要使用 GPU，需要显式移动：

```
x = x.to(device)
```

或者在 GPU device context 中创建。原报告强调，如果忘记把 tensor 移到 GPU，就不会获得 GPU 并行加速。【字幕†原报告】

------

## 9. 第八部分：einops——给维度命名，减少张量错误

### 9.1 为什么需要 einops

传统 PyTorch 写法常见：

```
z = x @ y.transpose(-2, -1)
```

问题是：读代码的人必须记住 `-2`、`-1` 分别是什么维度。维度一多，尤其是 batch、sequence、heads、hidden 混在一起时，很容易写错。

`lecture_02.py` 明确说 einops 是一个通过命名维度操作 tensors 的库，并用 `x @ y.transpose(-2, -1)` 展示传统写法容易搞错维度的问题。【Python代码】

### 9.2 einsum：用命名维度表达矩阵乘法

普通矩阵乘法：
$$
X \in \mathbb{R}^{\text{seq1}\times \text{hidden}},
\quad
Y \in \mathbb{R}^{\text{hidden}\times \text{seq2}}
$$
输出：
$$
Z \in \mathbb{R}^{\text{seq1}\times \text{seq2}}
$$
einops 写法：

```
z = einsum(x, y, "seq1 hidden, hidden seq2 -> seq1 seq2")
```

这里 `hidden` 出现在输入中，但没有出现在输出中，因此被求和消去。这就是 Einstein summation notation 的思想。

更复杂的 batch 版本：

```
z = einsum(
    x, y,
    "batch seq1 hidden, batch seq2 hidden -> batch seq1 seq2"
)
```

或者使用 `...`：

```
z = einsum(x, y, "... seq1 hidden, ... seq2 hidden -> ... seq1 seq2")
```

`lecture_02.py` 中正是这样组织 `einops_einsum()`，并指出未出现在输出中的维度会被 summed over。【Python代码】

### 9.3 reduce：命名维度上的归约

传统写法：

```
y = x.sum(dim=-1)
```

einops 写法：

```
y = reduce(x, "... hidden -> ...", "sum")
```

含义：把 `hidden` 维度求和，保留其他所有维度。

### 9.4 rearrange：拆分与合并维度

Transformer 中经常需要把一个 hidden 维度拆成 heads 和 per-head hidden：

```
x = rearrange(
    x,
    "... (heads hidden1) -> ... heads hidden1",
    heads=2
)
```

处理完成后再合并：

```
x = rearrange(x, "... heads hidden2 -> ... (heads hidden2)")
```

`lecture_02.py` 的 `einops_rearrange()` 演示了把 `total_hidden` 拆成 `heads * hidden1`，做变换后再合并回去。【Python代码】

**学习提示**：
 einops 不只是“代码更漂亮”，它让公式、代码和张量 shape 对齐。学 Transformer 时，能用命名维度解释每一步，是避免 attention 维度错误的关键。

------

## 10. 第九部分：FLOPs、FLOP/s 与 MFU

### 10.1 FLOPs 与 FLOP/s 的区别

| 术语   | 含义                                           | 例子                             |
| ------ | ---------------------------------------------- | -------------------------------- |
| FLOPs  | floating-point operations，计算量              | 训练一个模型总共做了多少浮点运算 |
| FLOP/s | floating-point operations per second，硬件速度 | H100 每秒能做多少浮点运算        |

`lecture_02.py` 明确区分：FLOPs 是计算量，FLOP/s 是硬件速度；它们读音相近但含义不同。【Python代码】

### 10.2 矩阵乘法 FLOPs

设：
$$
X \in \mathbb{R}^{B \times D},
\quad
W \in \mathbb{R}^{D \times K},
\quad
Y = XW \in \mathbb{R}^{B \times K}
$$
每个输出元素：
$$
Y_{bk}
=
\sum_{d=1}^{D} X_{bd} W_{dk}
$$
对每个 $(b,d,k)$ 三元组，有一次乘法和一次加法，所以：
$$
\text{FLOPs}
\approx
2BDK
$$
`lecture_02.py` 中 `tensor_operations_flops()` 直接使用 `actual_num_flops = 2 * B * D * K`，并解释每个 $(i,j,k)$ triple 有一次 multiplication 和一次 addition。【Python代码】

### 10.3 实际 FLOP/s

如果逻辑计算量是 $\text{FLOPs}$，实际耗时是 $t$，则：
$$
\text{actual FLOP/s}
=
\frac{\text{FLOPs}}{t}
$$
这就是 benchmark 的基本思路：先知道理论操作数，再测 wall-clock time。

### 10.4 MFU：Model FLOPs Utilization

MFU 定义为：
$$
\text{MFU}
=
\frac{\text{actual FLOP/s}}
{\text{promised FLOP/s}}
$$
其中：

- actual FLOP/s：实际跑出来的吞吐；
- promised FLOP/s：硬件 spec sheet 上给出的峰值吞吐。

`lecture_02.py` 中给出 MFU 定义，并指出通常 MFU $\ge 0.5$ 已经相当不错；随后引出问题：为什么 MFU 不能更接近 1？【Python代码】

**学习提示**：
 MFU 低不一定说明代码“写得很烂”。如果 workload 本身 arithmetic intensity 很低，硬件瓶颈可能是内存带宽而不是算力。

------

## 11. 第十部分：Arithmetic intensity 与 roofline analysis

### 11.1 为什么 GPU 算力跑不满

一次 GPU 计算通常包含：

1. 从 HBM 把输入搬到计算单元；
2. 执行算术操作；
3. 把输出写回 HBM。

所以耗时取决于两类资源：

- accelerator speed：FLOP/s；
- memory bandwidth：bytes/s。

`lecture_02.py` 的 `arithmetic_intensity()` 正是用这个流程解释，并用 H100 dense FLOP/s 与 memory bandwidth 举例。【Python代码】

### 11.2 Hardware intensity

硬件强度：
$$
I_{\text{hardware}}
=
\frac{\text{peak FLOP/s}}
{\text{memory bandwidth}}
$$
含义：每搬运 1 byte，硬件理论上能做多少 FLOPs。

### 11.3 Algorithm arithmetic intensity

算法强度：
$$
I_{\text{algo}}
=
\frac{\text{FLOPs}}
{\text{bytes moved}}
$$
判断规则：
$$
I_{\text{algo}} < I_{\text{hardware}}
\Rightarrow
\text{memory-bound}
$$

### 11.4 ReLU：典型 memory-bound 操作

ReLU：
$$
y = \max(x, 0)
$$
假设 $x$ 有 $n$ 个 bf16 元素，每个 2 bytes：

- 读 $x$：$2n$ bytes；
- 写 $y$：$2n$ bytes；
- FLOPs：约 $n$ 次比较。

所以：
$$
I_{\text{ReLU}}
=
\frac{n}{4n}
=
0.25
$$
这远低于现代 GPU 的 hardware intensity，因此 ReLU 是 memory-bound。`lecture_02.py` 的 `arithmetic_intensity_relu()` 用 `bytes = (2 * n) + (2 * n)`、`flops = n` 计算，并判断 ReLU 是 memory-bound。【Python代码】

### 11.5 GELU：计算更多，但仍可能 memory-bound

GELU 近似为：
$$
\text{GELU}(x)
=
0.5x
\left(
1+\tanh
\left(
\sqrt{\frac{2}{\pi}}
(x+0.044715x^3)
\right)
\right)
$$
GELU 每个元素做的计算比 ReLU 多，但它仍然是逐元素操作，读写 bytes 没有显著减少。因此孤立 GELU 的瓶颈仍可能是内存移动，而不是算术。

**学习提示**：
 这解释了一个反直觉现象：GELU 算术比 ReLU 复杂，但单独看 kernel，它未必慢很多，因为两者都可能被 memory bandwidth 限制。

### 11.6 Matrix-vector product 与 inference decode

Matrix-vector product 的 arithmetic intensity 也较低，因为每次只处理一个向量，权重矩阵读入成本很高，但复用不充分。

这和 inference decode 有直接关系：decode 阶段一次生成一个 token，许多操作更接近 matrix-vector product，因此经常 memory-bound。【字幕†原报告】

### 11.7 Matrix multiplication：为什么训练更容易 compute-bound

对大矩阵乘法 $XW$，若 $X,W \in \mathbb{R}^{n\times n}$：
$$
\text{FLOPs} \approx 2n^3
$$
读写数据大约是：
$$
\text{bytes}
=
2n^2 + 2n^2 + 2n^2
=
6n^2
$$
所以：
$$
I_{\text{matmul}}
\approx
\frac{2n^3}{6n^2}
=
\frac{n}{3}
$$
当 $n$ 足够大，matmul 的 arithmetic intensity 会很高，更容易 compute-bound。这也是 Transformer training 大量依赖大矩阵乘法时，可以较好利用 GPU 算力的原因。

### 11.8 Roofline plot 如何读

Roofline plot：

- 横轴：arithmetic intensity；
- 纵轴：achieved FLOP/s；
- 左侧：memory-bound 区域，性能随内存带宽线性上升；
- 右侧：compute-bound 区域，性能受峰值算力限制；
- 转折点：memory-bound 到 compute-bound 的 kink。

原报告中的近似关系：
$$
\text{MFU}
=
\min
\left(
1,
\frac{I_{\text{algo}}}{I_{\text{hardware}}}
\right)
$$
它表达的直觉是：如果算法强度低于硬件强度，MFU 会被内存带宽压低；如果算法强度足够高，才可能接近算力上限。

------

## 12. 第十一部分：训练中的 memory 与 compute accounting

### 12.1 Deep network 示例

Lecture 2 用一个简化网络作为训练资源估算对象：
$$
h_{\ell+1}
=
\operatorname{ReLU}(h_{\ell}W_{\ell})
$$
其中每层有一个 $D \times D$ 权重矩阵，共 $L$ 层。因此总参数量：
$$
N = D^2L
$$
`lecture_02.py` 中 `Block` 是 linear + ReLU，`DeepNetwork` 用 `nn.ModuleList([Block(dim) for i in range(num_layers)])` 堆叠 $L$ 层，并断言参数量是 $(D \times D) \times L$。【Python代码】

### 12.2 Autograd 直觉

课程先用简单例子说明 PyTorch 自动求导：
$$
y = 0.5(x \cdot w - 5)^2
$$
若：
$$
x=[1,2,3],
\quad
w=[1,1,1]
$$
则：
$$
x \cdot w = 6
$$
梯度为：
$$
\nabla_w y
=
(x \cdot w - 5)x
=
1 \cdot [1,2,3]
=
[1,2,3]
$$
`lecture_02.py` 中 `gradients_basics()` 用 `loss.backward()` 验证 `w.grad == torch.tensor([1, 2, 3])`。【Python代码】

### 12.3 Backward FLOPs 推导

考虑一层：
$$
h_2 = h_1 W_2
$$
其中：
$$
h_1 \in \mathbb{R}^{B \times D},
\quad
W_2 \in \mathbb{R}^{D \times D},
\quad
h_2 \in \mathbb{R}^{B \times D}
$$
forward FLOPs：
$$
2BD^2
$$
backward 需要两部分：
$$
\frac{\partial L}{\partial h_1}
=
\frac{\partial L}{\partial h_2} W_2^\top
$$
以及：
$$
\frac{\partial L}{\partial W_2}
=
h_1^\top
\frac{\partial L}{\partial h_2}
$$
每个也是一次矩阵乘法，各约：
$$
2BD^2
$$
所以 backward FLOPs：
$$
4BD^2
$$
总计：
$$
\text{forward} + \text{backward}
=
2BD^2 + 4BD^2
=
6BD^2
$$
推广到所有参数：
$$
\text{training step FLOPs}
\approx
6
\times
(\#\text{data points})
\times
(\#\text{parameters})
$$
`lecture_02.py` 的 `gradients_flops()` 用 `einsum` 手动计算 `h1_grad` 与 `w2_grad`，并总结：forward 是 $2 \times$ data points $\times$ parameters，backward 是 $4 \times$，总计是 $6 \times$ data points $\times$ parameters；讲义还说明该近似对短上下文 Transformer 也有参考价值。【Python代码】

### 12.4 Optimizer state：以 AdaGrad 为例

AdaGrad 存储每个参数历史梯度平方和：
$$
G_t = G_{t-1} + g_t^2
$$
参数更新：
$$
\theta_t
=
\theta_{t-1}
-
\eta
\frac{g_t}{\sqrt{G_t+\epsilon}}
$$
`lecture_02.py` 中 `AdaGrad.step()` 对每个参数读取梯度 `grad`，取出或初始化 `g2`，执行 `g2 += torch.square(grad)`，保存到 `state["g2"]`，再用 `p.data -= lr * grad / torch.sqrt(g2 + 1e-5)` 更新参数。【Python代码】

### 12.5 训练显存分解

训练时总显存大致包含：
$$
M
=
M_{\text{params}}
+
M_{\text{grads}}
+
M_{\text{optimizer}}
+
M_{\text{activations}}
$$
课程示例中假设 parameters、gradients、activations 使用 bf16，每元素 2 bytes；optimizer state 常用 fp32，AdaGrad 存二阶统计约 4 bytes/parameter，Adam 存一阶与二阶矩约 8 bytes/parameter。【Python代码】

**学习提示**：
 训练显存不是只由参数决定。很多初学者只算模型权重，忽略 gradients、optimizer states 和 activations，最后会发现估算的模型“理论上放得下”，实际训练却 OOM。

------

## 13. 第十二部分：训练循环、梯度累积与 activation checkpointing

### 13.1 标准训练循环

标准 PyTorch training loop：

1. 获取 batch；
2. forward；
3. 计算 loss；
4. `loss.backward()`；
5. `optimizer.step()`；
6. `optimizer.zero_grad(set_to_none=True)`。

`lecture_02.py` 中 `train_loop()` 正是这个结构：获取数据，模型 forward，计算 MSE loss，执行 backward，optimizer step，然后 zero gradients。【Python代码】

### 13.2 Gradient accumulation

问题：大 batch size 可能提升训练稳定性，但 activation memory 随 batch size 增长。

若 activation memory 近似为：
$$
M_{\text{act}}
=
2BDL
$$
其中：

- 2：bf16 每元素 2 bytes；
- $B$：batch size；
- $D$：hidden dimension；
- $L$：层数。

当显存不够时，可以把大 batch 切成 micro-batches：

1. 每个 micro-batch 做 forward/backward；
2. 不清空梯度，让梯度累积；
3. 累积到等效 batch size 后再 `optimizer.step()`；
4. 然后 zero gradients。

`lecture_02.py` 中 `gradient_accumulation()` 明确说 large batch size 可以改善训练稳定性，但 activation memory scales with batch size；gradient accumulation 通过 micro batches 计算并累积梯度，每隔 `batch_size / micro_batch_size` 步更新参数并清空梯度。【Python代码】

**容易误解**：
 Gradient accumulation 主要降低的是单次 micro-batch 的 activation memory，不会减少总 FLOPs。

### 13.3 Activation checkpointing

训练 backward 需要用到 forward 中间 activations。如果每层都保存，activation memory 是 $O(L)$。

Activation checkpointing / gradient checkpointing / rematerialization 的思想是：

- forward 只保存部分层的 activations；
- backward 需要中间 activations 时，从最近 checkpoint 重新计算；
- 用额外 compute 换更少 memory。

`lecture_02.py` 中 `activation_checkpointing()` 明确写出：forward pass 只保留部分层 activations，backward pass 从最近 checkpoint 重算缺失 activations，核心哲学是 tradeoff memory for compute；`DeepNetworkCheckpointed.forward()` 的关键实现是 `torch.utils.checkpoint.checkpoint(layer, x)`。【Python代码】

复杂度权衡：

| 策略                     | Activation memory | 额外计算       |
| ------------------------ | ----------------- | -------------- |
| 保存每层 activation      | $O(L)$            | 无重算         |
| 完全不保存 activation    | $O(1)$            | $O(L^2)$ 重算  |
| 每 $\sqrt{L}$ 层保存一次 | $O(\sqrt{L})$     | 约 $O(L)$ 重算 |

`lecture_02.py` 也列出这三种情况：store each layer 是 $O(L)$ memory 且无重算；store no activations 是 $O(1)$ memory 与 $O(L^2)$ compute；store every $\sqrt{L}$ layers 是 $O(\sqrt{L})$ memory 与 $O(L)$ recomputation。【Python代码】

------

## 14. Python 代码示例详解

### 14.1 Lecture 1：`lecture_01.py` 文件结构

| 函数 / 类                  | 作用                                                   | 对应课程概念                                   | 学习重点                  |
| -------------------------- | ------------------------------------------------------ | ---------------------------------------------- | ------------------------- |
| `main()`                   | 组织 Lecture 1 顺序                                    | overview → landscape → syllabus → tokenization | 看懂课程主线              |
| `why_this_course_exists()` | 解释 from scratch、leaky abstractions、frontier models | 课程哲学                                       | 为什么不能只会 prompt     |
| `current_lm_landscape()`   | 梳理 LM 发展历史                                       | LM landscape                                   | 从历史理解范式变化        |
| `course_syllabus()`        | 串联五个 assignment                                    | 课程地图                                       | CS336 的整体结构          |
| `Tokenizer`                | 抽象 encode/decode 接口                                | tokenizer correctness                          | round trip                |
| `CharacterTokenizer`       | Unicode code point tokenizer                           | 朴素分词失败案例                               | 大词表、低效率            |
| `ByteTokenizer`            | UTF-8 byte tokenizer                                   | byte-level tokenization                        | 小词表、长序列            |
| `get_compression_ratio()`  | 计算 bytes/token                                       | tokenizer 效率                                 | 序列长度与 attention 成本 |
| `train_bpe()`              | 训练 BPE merges 与 vocab                               | BPE 算法                                       | 高频 pair 贪心合并        |
| `count_adjacent_pairs()`   | 统计相邻 pair                                          | BPE 训练步骤                                   | 找最高频合并候选          |
| `merge()`                  | 替换 pair 为新 token                                   | BPE 单轮合并                                   | token 序列如何缩短        |
| `BPETokenizer`             | encode/decode BPE token                                | BPE 使用                                       | 固定词表 + 可逆           |

### 14.2 Lecture 2：`lecture_02.py` 文件结构

| 函数 / 类                    | 作用                                   | 对应课程概念            | 学习重点                     |
| ---------------------------- | -------------------------------------- | ----------------------- | ---------------------------- |
| `main()`                     | 组织 Lecture 2 顺序                    | resource accounting     | 从 tensor 到 training memory |
| `motivating_questions()`     | 70B/15T/H100 时间估算，8 H100 显存估算 | napkin math             | 训练数量级估算               |
| `tensors_basics()`           | tensor rank、shape、Transformer rank-4 | tensor 基础             | shape 语义                   |
| `tensors_memory()`           | fp32/fp16/bf16/AMP/fp8/fp4             | dtype 与显存            | precision tradeoff           |
| `tensors_on_gpus()`          | CPU/GPU device placement               | GPU 使用                | tensor 必须在正确设备上      |
| `einops_einsum()`            | 命名维度矩阵乘法                       | einsum                  | 减少维度错误                 |
| `einops_reduce()`            | 命名维度归约                           | reduce                  | sum/mean/max 的语义化写法    |
| `einops_rearrange()`         | 拆分与合并维度                         | rearrange               | heads 与 hidden 的 reshape   |
| `tensor_operations_flops()`  | FLOPs、FLOP/s、MFU                     | compute accounting      | 矩阵乘法 $2BDK$              |
| `arithmetic_intensity()`     | memory-bound / compute-bound           | roofline                | 判断瓶颈                     |
| `gradients_flops()`          | backward FLOPs 推导                    | $6ND$                   | backward 是 forward 约 2 倍  |
| `AdaGrad`                    | 自定义 optimizer                       | optimizer state         | 更新公式与显存               |
| `train_loop()`               | 标准训练循环                           | PyTorch training        | forward/backward/update      |
| `gradient_accumulation()`    | micro-batch 累积                       | 显存优化                | 保持 effective batch size    |
| `activation_checkpointing()` | 重算 activation                        | memory-compute tradeoff | 用计算换显存                 |
| `DeepNetworkCheckpointed`    | checkpoint 版网络                      | checkpoint 实现         | `torch.utils.checkpoint`     |

### 14.3 代码与课程知识点对应表

| 课程知识点               | 对应代码函数 / 变量 / 片段              | 代码功能                       | 对应理论原理       | 在课程中的作用                |
| ------------------------ | --------------------------------------- | ------------------------------ | ------------------ | ----------------------------- |
| Tokenizer 接口           | `Tokenizer.encode/decode`               | 字符串 ↔ token ids             | 离散化输入         | 所有 tokenizer 的共同抽象     |
| Round trip               | `assert string == reconstructed_string` | 检查可逆                       | 信息不丢失         | tokenizer correctness         |
| Compression ratio        | `get_compression_ratio()`               | 计算 bytes/token               | 序列长度效率       | 连接 attention 成本           |
| BPE training             | `train_bpe()`                           | 高频 pair 合并                 | 贪心压缩           | 现代 tokenizer 基础           |
| Tensor memory            | `get_memory_usage(x)`                   | `numel * element_size`         | 显存估算           | resource accounting 起点      |
| bf16 vs fp16             | `torch.bfloat16`, `torch.float16`       | 展示 underflow / dynamic range | 数值稳定性         | 混合精度训练                  |
| einsum                   | `einsum(..., "batch seq hidden ...")`   | 命名维度运算                   | Einstein summation | 写清楚 Transformer 维度       |
| MatMul FLOPs             | `2 * B * D * K`                         | 矩阵乘法计算量                 | 点积乘加           | compute accounting            |
| MFU                      | `actual / promised`                     | 实际吞吐 / 峰值吞吐            | 硬件利用率         | 判断训练效率                  |
| Arithmetic intensity     | `flops / bytes`                         | 判断瓶颈                       | roofline           | memory-bound vs compute-bound |
| Training FLOPs           | `6 * B * num_parameters`                | 训练一步 FLOPs                 | forward + backward | LLM scaling 估算              |
| AdaGrad state            | `state["g2"]`                           | 累计梯度平方                   | 自适应学习率       | optimizer memory              |
| Gradient accumulation    | `micro_batch_size`                      | 累积 micro-batch 梯度          | 显存换步数         | 支持大 effective batch        |
| Activation checkpointing | `torch.utils.checkpoint.checkpoint`     | backward 重算                  | 显存换计算         | 大模型训练常用技巧            |

------

## 15. 图表 / 代码讲义图片解析

由于 Lecture 1/2 没有单独 PDF，本节解析 Python 讲义中调用的图片与图表。仓库目录显示 Lecture 1/2 是 `.py` 文件，Lecture 3 起才出现 `.pdf`；README 也说明 executable lectures 命名为 `lecture_XX.py`，non-executable lectures 命名为 `lecture_XX.pdf`。【图表/代码】

| 图表 / 图片                          | 出现位置                          | 展示内容                                  | 支持的课程结论                      | 读图重点                        |
| ------------------------------------ | --------------------------------- | ----------------------------------------- | ----------------------------------- | ------------------------------- |
| `gpt4-no-details.png`                | Lecture 1 motivation              | GPT-4 technical report 不公开关键细节     | frontier models 不透明              | 不能依赖闭源细节学习底层机制    |
| `roller-flops.png`                   | Lecture 1 frontier scale          | attention / MLP FLOPs 比例随规模变化      | 小模型瓶颈不一定代表大模型瓶颈      | 规模变化会改变资源分布          |
| `wei-emergence-plot.png`             | Lecture 1 emergence               | 能力随规模出现非线性变化                  | 小模型不一定观察到大模型能力        | 不要过度外推小实验              |
| `tokenized-example.png`              | Tokenization                      | 文本被切成 token                          | tokenizer 决定模型看到的基本单位    | token 不等于 word               |
| `compute-memory.png`                 | Lecture 1/2 systems               | compute 与 memory 分离                    | 数据移动会限制性能                  | 解释 memory-bound               |
| `prefill-decode.png`                 | Lecture 1 systems overview        | inference 的 prefill 与 decode            | decode 更容易 memory-bound          | prefill 可并行，decode 逐 token |
| `chinchilla-isoflop.png`             | Scaling laws overview             | 不同 FLOPs budget 下的 optimal model size | 选择 $N$ 与 $D$ 要按 compute budget | 连接 $C=6ND$                    |
| `fp32.png` / `fp16.png` / `bf16.png` | Lecture 2 tensor memory           | 浮点数位布局                              | dtype 决定 memory 与 dynamic range  | bf16 与 fp16 的差别             |
| `cpu-gpu.png`                        | Lecture 2 tensors on GPUs         | CPU/GPU memory 与 compute                 | tensor 必须在 GPU 上计算            | device placement                |
| roofline plot                        | Lecture 2 arithmetic intensity    | intensity 与 achieved FLOP/s              | 判断 memory/compute bound           | kink 是瓶颈转折点               |
| `deep-network.png`                   | Lecture 2 gradients/checkpointing | 多层网络 forward/backward                 | 推导 parameter、activation、FLOPs   | 连接 $6ND$ 与 checkpointing     |

------

## 16. 关键公式与推导汇总

| 公式                                                         | 符号解释                                                    | 适用场景              | 直觉解释                        | 代码 / 例子对应                               |
| ------------------------------------------------------------ | ----------------------------------------------------------- | --------------------- | ------------------------------- | --------------------------------------------- |
| $\text{accuracy} = \text{efficiency} \times \text{resources}$ | resources 是 compute/data/memory；efficiency 是单位资源产出 | 全课程主线            | 大规模下效率决定成本            | Lecture 1 framing                             |
| $\text{decode}(\text{encode}(x))=x$                          | $x$ 是原字符串                                              | tokenizer correctness | 编码解码不能丢信息              | `assert string == reconstructed_string`       |
| $\text{compression ratio}=\#\text{bytes}/\#\text{tokens}$    | bytes 是 UTF-8 bytes；tokens 是 token 数                    | tokenizer 评估        | token 越少，attention 越省      | `get_compression_ratio()`                     |
| $p^\*=\arg\max_p \operatorname{count}(p)$                    | $p$ 是相邻 token pair                                       | BPE merge             | 每轮合并最高频 pair             | `count_adjacent_pairs()` + `max()`            |
| $M(x)=\operatorname{numel}(x)\operatorname{element\_size}(x)$ | 元素数 × 每元素 bytes                                       | tensor memory         | 显存由 shape 和 dtype 决定      | `get_memory_usage()`                          |
| $\text{FLOPs}=2BDK$                                          | $B,D,K$ 是 matmul 维度                                      | linear layer / matmul | 每个三元组一次乘加              | `2 * B * D * K`                               |
| $\text{MFU}=\text{actual FLOP/s}/\text{promised FLOP/s}$     | actual 是实测；promised 是硬件峰值                          | 性能评估              | 衡量算力利用率                  | `actual_flop_per_sec / promised_flop_per_sec` |
| $I_{\text{algo}}=\text{FLOPs}/\text{bytes moved}$            | 算法每搬 1 byte 做多少计算                                  | roofline              | 判断瓶颈                        | arithmetic intensity                          |
| $I_{\text{ReLU}}=n/(4n)=0.25$                                | bf16 读写共 $4n$ bytes                                      | elementwise ops       | 计算少、搬数据多                | ReLU memory-bound                             |
| $I_{\text{matmul}}\approx n/3$                               | 大矩阵乘法                                                  | training matmul       | $O(n^3)$ 计算复用 $O(n^2)$ 数据 | matmul compute-bound                          |
| $G_t=G_{t-1}+g_t^2$                                          | $G_t$ 是累计平方梯度                                        | AdaGrad               | 历史梯度大则步长变小            | `state["g2"]`                                 |
| $\theta_t=\theta_{t-1}-\eta g_t/\sqrt{G_t+\epsilon}$         | $\eta$ 学习率；$\epsilon$ 稳定项                            | AdaGrad update        | 自适应缩放梯度                  | `p.data -= ...`                               |
| $\text{training FLOPs}\approx6\times\#\text{data points}\times\#\text{parameters}$ | 6 来自 forward 2 + backward 4                               | 训练预算              | backward 约为 forward 2 倍      | `gradients_flops()`                           |
| $M_{\text{act}}=2BDL$                                        | 2 是 bf16 bytes；$B,D,L$ 是 batch、hidden、layers           | activation memory     | batch 越大、层越多显存越大      | gradient accumulation / checkpointing         |

------

## 17. 概念对比与易混淆点

| 易混淆概念                                 | 区别                                            | 联系                         | 常见误解                  | 正确理解                                  |
| ------------------------------------------ | ----------------------------------------------- | ---------------------------- | ------------------------- | ----------------------------------------- |
| token vs word                              | token 是模型单位，word 是语言学单位             | token 可能近似子词或词       | token 一定是单词          | token 可能包含空格、byte、子词            |
| byte tokenizer vs BPE                      | byte 每 byte 一个 token；BPE 合并高频 byte 序列 | BPE 从 byte vocab 开始       | byte tokenizer 已经完美   | byte 可逆但序列太长                       |
| compression ratio vs vocab size            | 前者衡量 bytes/token；后者衡量 token 类型数     | 增大词表可提升压缩           | 压缩率越高越好            | 词表太大会导致稀疏                        |
| mechanics vs intuitions                    | mechanics 是机制；intuitions 是经验判断         | 都服务于建模                 | 小模型 intuition 必然迁移 | mechanics 更可迁移，intuition 要谨慎      |
| FLOPs vs FLOP/s                            | FLOPs 是计算量；FLOP/s 是速度                   | 训练时间 = 计算量 / 速度     | 两者可混用                | 必须区分“多少计算”和“多快计算”            |
| fp16 vs bf16                               | 都是 2 bytes；bf16 dynamic range 更大           | 都用于混合精度               | fp16 和 bf16 差不多       | bf16 更抗 underflow                       |
| memory-bound vs compute-bound              | 前者瓶颈是数据移动；后者瓶颈是计算              | 由 arithmetic intensity 判断 | GPU 利用率低就是代码差    | workload 本身可能不适合跑满算力           |
| matvec vs matmul                           | matvec 数据复用少；matmul 数据复用高            | 都是线性代数操作             | 推理 decode 和训练一样    | decode 常更 memory-bound                  |
| gradient accumulation vs 大 batch 原生训练 | accumulation 分 micro-batch 累积梯度            | 目标是同等 effective batch   | accumulation 减少总计算   | 它主要减少单次 activation memory          |
| activation checkpointing vs 免费省显存     | checkpointing 需要重算                          | 用 compute 换 memory         | checkpointing 没有代价    | 它降低显存但增加计算                      |
| optimizer state vs parameters              | optimizer state 是训练额外状态                  | Adam/AdaGrad 都要存状态      | 模型显存只看参数          | 训练还要算 gradients、states、activations |

------

## 18. 学习拓展与发散理解

### 18.1 与大模型训练实践的关系

**数据准备**：
 Tokenization 决定原始文本如何变成 token 序列。tokenizer 的 compression ratio 会影响有效 context length、训练 FLOPs 和数据吞吐。BPE 这种可逆、开放词汇、压缩率较好的方法，是大模型数据管线的重要基础。

**模型训练**：
 Lecture 2 的 tensor memory、dtype、training FLOPs 公式直接用于判断一个模型能否训练。参数量只是显存账本的一部分，真正训练时还要保存 gradients、optimizer states 和 activations。

**优化策略**：
 bf16、mixed precision、gradient accumulation、activation checkpointing 都是现代训练常用手段。它们不是“附加技巧”，而是让大模型训练可行的必要条件。

**训练效率**：
 MFU 和 arithmetic intensity 帮助判断瓶颈。大矩阵乘法容易 compute-bound，而 elementwise、matvec、decode 等可能 memory-bound。理解这一点，才能解释为什么 kernel fusion、tiling、FlashAttention、continuous batching 等系统优化重要。

**工程成本**：
 课程反复强调 fixed resources。训练大模型时，算错 FLOPs 或显存不是小问题，而是会直接影响预算、时间和可行性。

### 18.2 与其他 lecture 的联系

- Lecture 1 的 tokenization 会在 Assignment 1 中继续使用，因为 Transformer 的输入就是 token ids。
- Lecture 2 的 $6ND$、MFU、arithmetic intensity 会在 systems 和 scaling laws 中反复出现。
- Scaling laws 需要用 FLOPs budget 思考模型大小 $N$ 和训练 tokens $D$ 的选择。
- Data lecture 会继续讨论：不是所有 token 都同样有价值，高质量数据可以提高单位 compute 的训练收益。
- Alignment lecture 会在 next-token prediction 之后，引入偏好优化和 weak supervision，但仍然离不开 inference 与 training systems 的资源账本。

### 18.3 面试 / 考试 / 作业可能考察角度

| 问题                                                 | 考察点                  | 回答思路                                                 |
| ---------------------------------------------------- | ----------------------- | -------------------------------------------------------- |
| 为什么 tokenizer 需要 round trip？                   | tokenizer correctness   | 如果 decode(encode(x)) 不等于 x，说明文本信息被破坏      |
| Character、byte、word tokenizer 分别有什么问题？     | tokenizer tradeoff      | character 词表大；byte 序列长；word 词表开放且有 UNK     |
| BPE 为什么可以没有 UNK？                             | byte-level fallback     | 因为初始 vocab 包含所有 byte，任何字符串都可分解为 bytes |
| compression ratio 为什么影响 Transformer 成本？      | sequence length         | attention 近似二次复杂度，token 越少越省                 |
| 矩阵乘法 $X_{B\times D}W_{D\times K}$ 有多少 FLOPs？ | compute accounting      | 每个 $(b,d,k)$ 一乘一加，所以 $2BDK$                     |
| 为什么训练一步近似 $6ND$？                           | forward/backward FLOPs  | forward 约 $2ND$，backward 约 $4ND$                      |
| 为什么 ReLU 可能 memory-bound？                      | arithmetic intensity    | 每元素只做少量计算，但要读写数据                         |
| 为什么大 matmul 更容易 compute-bound？               | 数据复用                | 计算 $O(n^3)$，数据 $O(n^2)$                             |
| AdamW 训练显存为什么比参数显存大很多？               | optimizer state         | 还要存 gradients、一阶矩、二阶矩、activations            |
| Gradient accumulation 减少了什么？                   | memory optimization     | 减少单次 micro-batch activation memory，不减少总 FLOPs   |
| Activation checkpointing 的代价是什么？              | memory-compute tradeoff | 显存下降，但 backward 需要重算 activation                |

------

## 19. 课程复习总结

### 19.1 本讲核心思想

Lecture 1 和 Lecture 2 合起来建立了 CS336 的入门框架：
 **理解语言模型不能只停留在 prompt 或 API 层，而要从输入表示、训练目标、代码实现和资源账本逐层向下理解。Tokenization 决定模型看到什么基本单位，resource accounting 决定模型能否在给定硬件和预算下高效训练。**

### 19.2 必须掌握的概念

- from scratch philosophy；
- leaky abstractions；
- mechanics / mindset / intuitions；
- efficiency × resources；
- tokenizer encode / decode；
- round trip；
- compression ratio；
- character / byte / word tokenizer 的优缺点；
- BPE 的 pair counting、merge、vocab、merges；
- tensor rank、shape、dtype；
- memory = numel × element_size；
- fp32、fp16、bf16、mixed precision；
- einsum、reduce、rearrange；
- FLOPs vs FLOP/s；
- MFU；
- arithmetic intensity；
- memory-bound vs compute-bound；
- roofline analysis；
- training FLOPs $\approx 6ND$；
- optimizer state；
- gradient accumulation；
- activation checkpointing。

### 19.3 必须理解的公式

1. **Tokenizer 可逆性**

$$
\text{decode}(\text{encode}(x)) = x
$$

解决问题：检查 tokenizer 是否破坏原始文本。

1. **Compression ratio**

$$
\frac{\#\text{UTF-8 bytes}}{\#\text{tokens}}
$$

解决问题：衡量 tokenizer 是否有效缩短序列。

1. **Tensor memory**

$$
M(x)=\operatorname{numel}(x)\times\operatorname{element\_size}(x)
$$

解决问题：估算 tensor 占多少显存。

1. **MatMul FLOPs**

$$
2BDK
$$

解决问题：估算 linear layer 或 projection 的计算量。

1. **MFU**

$$
\text{MFU}
=
\frac{\text{actual FLOP/s}}
{\text{promised FLOP/s}}
$$

解决问题：衡量硬件利用率。

1. **Arithmetic intensity**

$$
I_{\text{algo}}
=
\frac{\text{FLOPs}}
{\text{bytes moved}}
$$

解决问题：判断操作是 memory-bound 还是 compute-bound。

1. **Training FLOPs**

$$
\text{training step FLOPs}
\approx
6
\times
(\#\text{data points})
\times
(\#\text{parameters})
$$

解决问题：估算训练一步或整个训练 run 的计算预算。

1. **Activation memory**

$$
M_{\text{act}}
=
2BDL
$$

解决问题：理解 batch size、hidden dimension、层数如何影响训练显存。

### 19.4 必须看懂的代码

- `Tokenizer.encode()` / `Tokenizer.decode()`：理解 tokenizer 接口；
- `get_compression_ratio()`：理解 bytes/token；
- `train_bpe()`：理解 BPE 从 byte 开始、反复合并最高频 pair；
- `merge()`：理解 token 序列如何被局部压缩；
- `get_memory_usage()`：理解 tensor 显存公式；
- `einsum()` 示例：理解命名维度矩阵乘法；
- `tensor_operations_flops()`：理解 $2BDK$ 和 MFU；
- `arithmetic_intensity_relu()` / `arithmetic_intensity_matmul()`：理解 memory-bound 与 compute-bound；
- `gradients_flops()`：理解 backward FLOPs 为什么约为 forward 两倍；
- `AdaGrad.step()`：理解 optimizer state；
- `gradient_accumulation()`：理解 micro-batch 累积；
- `DeepNetworkCheckpointed.forward()`：理解 activation checkpointing。

### 19.5 一句话总结

> **Lecture 1 告诉我们为什么要从零理解语言模型，并从 tokenizer 这个输入抽象开始；Lecture 2 则把训练语言模型的资源消耗变成可计算的账本，让我们能判断模型是否放得下、训练是否跑得动、瓶颈到底在计算还是内存。**