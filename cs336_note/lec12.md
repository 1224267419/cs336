# Lecture 12 课程报告：大语言模型 Evaluation——从困惑度到真实世界评测的系统思维

## 0. 资料来源与改写说明

本报告是在你上传的原始 Lecture 12 中文课程报告基础上进行系统性改写。原报告说明其依据包括 lecture 12 字幕和 Stanford CS336 GitHub 仓库中的 `lecture_12.py`，并指出该 Python 文件按顺序组织了本讲主题：什么是“好模型”、perplexity、考试型 benchmark、聊天型 benchmark、agentic benchmark、纯推理 benchmark、安全 benchmark、真实性、有效性，以及如何思考 evaluation 本身。 

我还核对了 Stanford CS336 当前公开仓库：仓库 README 说明 CS336 lectures 包含可执行讲义 `lecture_XX.py`，文件列表中包含 `lecture_12.py`；`lecture_12.py` 的主流程也确实按 `what_is_good()`、`perplexity()`、`exam_benchmarks()`、`chat_benchmarks()`、`agentic_benchmarks()`、`pure_reasoning_benchmarks()`、`safety_benchmarks()`、`realism()`、`validity()`、`how_to_think_about_evaluation()` 的顺序组织本讲内容。

本报告中的引用约定如下：

- **原报告依据**：使用系统文件引用，如 ``。
- **课程代码来源**：用 `【Python代码†lecture_12.py】` 标注，并在首次出现处用可点击网页引用补充。
- **字幕内容**：原始报告多次转述“字幕中说/字幕提到”，但本轮没有额外提供可逐行检索的字幕文件。因此，本报告会保留这些字幕转述，不新增伪造的逐行字幕引用。
- **延伸内容**：凡是为了帮助理解而加入的背景、类比、学习提示、工程直觉，都会明确标注为“补充说明”“延伸理解”“工程直觉”或“学习提示”，不把它们伪装成课程原文。

------

## 1. 学习导读

### 1.1 本讲要解决的核心问题

本讲的核心问题是：
$$
\boxed{
\text{给定一个训练好的语言模型，我们如何判断它到底有多好？}
}
$$
这看起来像一个简单的工程流程：准备 prompt，把 prompt 发给模型，得到 response，然后计算 accuracy。但课程强调，evaluation 远不只是“算分”。它会成为整个 AI 研究和产品开发的 **North Star**：研究者会追逐高分，模型开发者会根据评测反馈改进模型，用户和公司会用评测结果做选择，政策制定者也可能依据评测判断风险与收益。原报告将本讲的核心挑战概括为：把抽象目标转化为具体指标。

换句话说，evaluation 的真正难点是：
$$
\text{抽象能力}
\longrightarrow
\text{具体任务}
\longrightarrow
\text{数据集或环境}
\longrightarrow
\text{评分规则}
\longrightarrow
\text{可比较的结论}
$$
例如，“模型会推理”“模型有帮助”“模型安全”“模型适合真实工作流”都是抽象说法。只有当它们被转化为具体题目、具体交互环境、具体 judge 或具体指标时，才变成可评测对象。

### 1.2 本讲在 CS336 课程中的位置

Lecture 12 位于课程主线的关键转折点。前面课程已经讲过训练语言模型所需的基础：架构、训练、系统、并行、scaling laws、inference 等；接下来课程会进入“数据”主题。而在讨论“应该用什么数据训练模型”之前，必须先回答“我们希望模型表现出什么行为”。原报告也指出，数据会塑造模型行为：训练代码数据会让模型更擅长代码，训练 DNA 序列并不必然让模型擅长英语。

因此，本讲是连接 **训练技术** 与 **训练目标** 的桥梁：

- 如果不知道如何评价模型，就不知道数据是否有效；
- 如果不知道评测指标是否可靠，就很难解释 benchmark 分数；
- 如果不知道评测对象是模型、方法还是 agent 系统，就容易误读排行榜；
- 如果不知道评测是否贴近真实使用，就可能高估模型部署价值。

### 1.3 本讲知识结构

| 模块                    | 解决的问题                          | 核心概念                                                | 与后续内容的关系                        |
| ----------------------- | ----------------------------------- | ------------------------------------------------------- | --------------------------------------- |
| 什么是“好模型”          | 为什么不存在唯一好模型定义          | benchmark、成本、人类偏好、用户使用量                   | 说明 evaluation 是目标设计问题          |
| Perplexity              | 如何用概率分布自然评价语言模型      | NLL、cross-entropy、PPL、OOD eval                       | 连接预训练目标、scaling laws 与数据分布 |
| Perplexity 的理论与局限 | 为什么 PPL 有数学吸引力，又为何不够 | KL 分解、真实分布、conditional PPL                      | 解释为什么需要任务型 benchmark          |
| 考试型 benchmark        | 如何评测知识和标准题能力            | MMLU、MMLU-Pro、GPQA、HLE、accuracy                     | 展示 benchmark 难度随模型能力演化       |
| 聊天型 benchmark        | 如何评测开放式回答                  | Chatbot Arena、Elo、AlpacaEval、WildBench               | 连接人类偏好、LLM judge、rubric         |
| Agentic benchmark       | 如何评测“模型能做什么”              | SWE-bench、Terminal-Bench、CyBench、MLE-bench、scaffold | 引入工具使用、长程任务和系统评测        |
| 纯推理 benchmark        | 能否隔离 reasoning 与知识           | ARC-AGI、归纳规则、fluid intelligence                   | 讨论推理能力与记忆/知识的关系           |
| Safety benchmark        | 安全到底测什么                      | HarmBench、AIR-Bench、jailbreak、GCG                    | 连接模型部署风险与对抗性评测            |
| Realism / validity      | 分数是否贴近真实且可信              | ecological validity、contamination、fresh/private evals | 形成现代 evaluation 的科学检查框架      |
| 统计可靠性与评测对象    | 分数是否有显著性，评测的是谁        | confidence interval、method/model/system                | 总结如何科学解释 benchmark              |

### 1.4 学习本讲前需要知道什么

本讲不要求深入掌握模型训练实现，但理解以下概念会很有帮助：

| 前置知识             | 为什么需要                                                   |
| -------------------- | ------------------------------------------------------------ |
| 概率分布与条件概率   | Perplexity、语言模型自回归分解、logprob scoring 都依赖条件概率 |
| 负对数似然与交叉熵   | PPL 本质上是平均 NLL 的指数变换                              |
| KL 散度              | 理解为什么真实分布下最优模型是 `p = t`                       |
| accuracy / win rate  | 多选 benchmark 和聊天评测的基础指标                          |
| 置信区间             | 判断 benchmark 小幅差异是否有统计意义                        |
| 基础语言模型训练流程 | 理解 evaluation 为什么影响数据选择和训练方向                 |
| 简单 agent 概念      | 理解 SWE-bench、Terminal-Bench 等为什么不只评测 LM 本体      |

------

## 2. 第一部分：Evaluation 是什么，以及“模型好”到底是什么意思

这一部分要解决的问题是：**为什么 evaluation 不是简单算分，而是把抽象目标变成具体指标的过程？**

### 2.1 问题引入

课程从一个看似简单的问题开始：
$$
\text{Given a model, how good is it?}
$$
直觉上，我们可能以为只要找几个 benchmark，跑一下分数，就能判断模型好坏。但课程马上指出，“好”本身没有唯一含义。模型可能 benchmark 高但很贵；可能便宜但不够可靠；可能在考试题上强但真实聊天体验差；也可能在用户中受欢迎但并不总是事实准确。原报告列举了几类视角：benchmark 分数、推理成本、人类偏好、用户是否真的愿意使用并付费。

### 2.2 概念定义：Evaluation

**直觉解释**：Evaluation 就像给模型设计一场考试。但问题在于，这场考试不是天然存在的，而是我们根据关心的能力人为设计出来的。

**正式定义**：在本讲语境中，evaluation 是将一个抽象 construct，例如“推理能力”“聊天质量”“安全性”“真实世界可用性”，转化为一组可执行任务、数据、环境、评分规则和指标的过程。

可以写成：
$$
\text{Evaluation}
=
f(
\text{goal},
\text{task},
\text{data/environment},
\text{scoring rule}
)
$$
其中：

- `goal` 是想测什么；
- `task` 是让模型做什么；
- `data/environment` 是模型面对的输入、上下文和工具环境；
- `scoring rule` 是如何把模型输出转化为分数。

### 2.3 方法与原理：从目标到指标

课程的关键思想是：
$$
\text{目标不同}
\Rightarrow
\text{评测设计不同}
\Rightarrow
\text{结论也不同}
$$
例如：

| 目标           | 可能的评测方式                | 主要风险                         |
| -------------- | ----------------------------- | -------------------------------- |
| 测知识         | MMLU、GPQA、HLE               | 题目污染、考试能力不等于真实能力 |
| 测聊天质量     | Chatbot Arena、AlpacaEval     | 偏好偏差、长度偏差、讨好型回答   |
| 测任务完成能力 | SWE-bench、Terminal-Bench     | scaffold 混入、环境和测试不充分  |
| 测安全         | HarmBench、AIR-Bench          | 安全定义依赖文化、法律和场景     |
| 测真实部署价值 | GDPVal、MedHELM、真实用户数据 | 隐私风险、分布不可控             |

所以，evaluation 不只是技术问题，也是设计问题。一个 benchmark 高分并不自动意味着“模型整体更好”，只能说明它在该 benchmark 定义的任务、数据分布和评分规则下表现更好。

### 2.4 课堂例子与解释

原报告提到几种“好模型”定义：

1. **benchmark 高分**：适合研究论文比较，但容易被刷榜。
2. **benchmark 高且便宜**：适合实际部署，因为成本决定是否可规模化使用。
3. **人类更偏好回答**：适合开放式聊天模型，但偏好可能混合风格、长度和事实性。
4. **用户愿意使用并付费**：接近市场选择，但用户选择不一定等于能力最强。

这些例子共同说明：goodness 是多维的，不存在一个天然、唯一、永远正确的模型质量指标。

### 2.5 代码 / 图表对应说明

`lecture_12.py` 在主函数中先说明前面课程已经覆盖训练 LM 的架构、训练、系统和 scaling，然后提出“缺失的一块是训练什么数据”，再引出 evaluation：在讨论数据之前，必须先讨论希望模型具备什么行为。文件随后调用 `what_is_good()`，展开“什么是好模型”的多种视角。

对应图表示例包括：

| 图表来源               | 展示内容         | 想说明什么                         |
| ---------------------- | ---------------- | ---------------------------------- |
| Artificial Analysis 图 | benchmark 与成本 | 模型好坏不只看能力，还要看推理成本 |
| Arena AI leaderboard   | 人类偏好排名     | 真实用户偏好可以作为一种评测信号   |
| OpenRouter 排名        | 用户调用量或流量 | 用户选择也是一种市场型评价指标     |

### 2.6 学习提示

本节最重要的不是记住某个 leaderboard，而是记住：
$$
\boxed{
\text{Evaluation 的第一步不是算分，而是说清楚你想测什么。}
}
$$
最容易误解的地方是：把 benchmark 分数当成模型能力的完整定义。正确理解是：benchmark 只是在特定任务、特定样本、特定规则下的局部观察。

------

## 3. 第二部分：Perplexity——语言模型最自然的评测方式

这一部分要解决的问题是：**如果语言模型本质上是概率分布，那么最自然的评测指标是什么？**

### 3.1 问题引入

语言模型最基础的能力是给文本序列分配概率。训练时，我们希望模型给真实文本更高概率；评测时，自然也可以看模型是否给测试文本高概率。这就是 perplexity 的出发点。

原报告指出，`lecture_12.py` 将 perplexity 写成：
$$
(1 / p(D))^{1/|D|}
$$
也就是测试数据概率的长度归一化倒数；它衡量模型给测试数据分配了多少概率质量。

### 3.2 概念定义：语言模型作为概率分布

**直觉解释**：语言模型在每个位置都在猜下一个 token。猜得越准，给真实 token 的概率越高；猜得越犹豫，困惑度越高。

**正式定义**：给定 token 序列
$$
x = (x_1, x_2, \dots, x_T)
$$
自回归语言模型将联合概率分解为：
$$
p_\theta(x_1,\dots,x_T)
=
\prod_{t=1}^{T}
p_\theta(x_t \mid x_{<t})
$$
其中：

- `x_t` 表示第 `t` 个 token；
- `x_{<t}` 表示当前位置之前的所有 token；
- `p_\theta(x_t | x_{<t})` 表示参数为 `θ` 的模型在上下文 `x_{<t}` 下预测真实 token `x_t` 的概率。

这条公式是语言模型训练、困惑度评测和 next-token prediction 的共同基础。

### 3.3 方法与原理：NLL 与 Perplexity

如果把测试集拼成长度为 `M` 的 token 序列，则平均负对数似然为：
$$
\text{NLL}(D)
=
-\frac{1}{M}
\sum_{i=1}^{M}
\log p_\theta(x_i \mid x_{<i})
$$
Perplexity 定义为平均 NLL 的指数：
$$
\text{PPL}(D)
=
\exp
\left(
-\frac{1}{M}
\sum_{i=1}^{M}
\log p_\theta(x_i \mid x_{<i})
\right)
$$
如果使用 base-2 log，则可写为：
$$
\text{PPL}(D)
=
2^{
-\frac{1}{M}
\sum_i
\log_2 p_\theta(x_i \mid x_{<i})
}
$$
**直觉解释**：

- `PPL = 10`：模型平均像是在 10 个等概率候选 token 中选择；
- `PPL = 2`：模型平均更确定；
- PPL 越低，模型对测试文本越不“困惑”。

### 3.4 Perplexity、likelihood 与 cross-entropy 的关系

最大似然训练希望最大化训练数据概率：
$$
\max_\theta
\prod_{i=1}^{M}
p_\theta(x_i \mid x_{<i})
$$
取 log 后等价于：
$$
\max_\theta
\sum_i
\log p_\theta(x_i \mid x_{<i})
$$
乘以负号后，就是最小化负对数似然：
$$
\min_\theta
-\sum_i
\log p_\theta(x_i \mid x_{<i})
$$
因此：
$$
\text{降低 NLL}
\iff
\text{降低 cross-entropy}
\iff
\text{降低 perplexity}
$$
因为指数函数单调递增，最小化平均 NLL 与最小化 PPL 在排序意义上等价。原报告也强调，传统语言模型研究长期使用这种方式：在某个数据集 train split 上训练，在 test split 上看 perplexity，例如 Penn Treebank、WikiText-103、One Billion Word Benchmark。

### 3.5 In-distribution 与 out-of-distribution evaluation

传统语言模型评测范式是：
$$
\text{train on train split}
\rightarrow
\text{test on test split}
$$
此时训练集和测试集来自同一分布：
$$
q_{\text{train}} = q_{\text{test}}
$$
这叫 **in-distribution evaluation**。

GPT-2 之后，范式发生变化：模型先在大规模 WebText 之类的通用语料上训练，再 zero-shot 测标准数据集。此时：
$$
q_{\text{train}} \neq q_{\text{test}}
$$
这叫 **out-of-distribution evaluation**。原报告指出，这更接近今天 foundation model 的实际情况：模型在大规模混合互联网数据上预训练，再迁移到各种下游任务。

### 3.6 代码 / 图表对应说明

`lecture_12.py` 的 `perplexity()` 部分依次组织了这些内容：

| 课程知识点      | 对应代码内容                     | 代码功能         | 理论原理             | 学习重点                  |
| --------------- | -------------------------------- | ---------------- | -------------------- | ------------------------- |
| LM 是概率分布   | `text("- Recall: ... p(x) ...")` | 提醒语言模型定义 | 序列概率建模         | PPL 的对象是概率分布      |
| Perplexity 定义 | `text("- Perplexity (1/p(D))^(1/ | D                | ) ...")`             | 给出 PPL 形式             |
| 传统评测        | Penn Treebank、WikiText-103、1BW | 列经典数据集     | in-distribution eval | 训练/测试同分布           |
| GPT-2 范式      | WebText + zero-shot              | 展示 OOD eval    | transfer evaluation  | foundation model 常见范式 |

【Python代码†lecture_12.py】

### 3.7 学习提示

本节最重要的公式是：
$$
\text{PPL}(D)
=
\exp(\text{average NLL})
$$
复习时要抓住三点：

1. PPL 是语言模型最自然的指标，因为语言模型本身就是概率分布；
2. PPL 与训练目标高度一致，因为训练也是最小化 NLL；
3. PPL 不等于真实任务能力，因此后续需要 benchmark、chat eval、agent eval 等更任务化的评测方式。

------

## 4. 第三部分：“Perplexity is all you need”的理论吸引力与局限

这一部分要解决的问题是：**为什么有人相信只要不断降低 perplexity 就能得到强模型？为什么这种观点又不充分？**

### 4.1 问题引入

课程提出一个有启发性的观点：如果存在真实数据分布 `t(x)`，模型分布是 `p(x)`，那么在真实分布下，最优 perplexity 会在 `p = t` 时达到。如果模型真的学到了真实分布，它就可以通过条件概率完成各种任务：
$$
p(\text{solution} \mid \text{problem})
$$
原报告称，字幕中也提到最低 perplexity 是真实分布的 entropy，并且在 `p = t` 时实现。

### 4.2 公式与推导：交叉熵分解

设真实分布为 `t(x)`，模型分布为 `p(x)`。测试样本来自 `t`。模型的期望负对数似然是交叉熵：
$$
H(t,p)
=
\mathbb{E}_{x\sim t}[-\log p(x)]
=
-\sum_x t(x)\log p(x)
$$
交叉熵可以分解为：
$$
H(t,p)
=
H(t)
+
\mathrm{KL}(t \Vert p)
$$
其中：
$$
H(t)
=
-\sum_x t(x)\log t(x)
$$
是真实分布的熵；
$$
\mathrm{KL}(t \Vert p)
=
\sum_x
t(x)
\log
\frac{t(x)}{p(x)}
$$
是从真实分布 `t` 到模型分布 `p` 的 KL 散度。

因为：
$$
\mathrm{KL}(t \Vert p) \ge 0
$$
并且当且仅当 `p(x)=t(x)` 几乎处处成立时取 0，所以：
$$
H(t,p) \ge H(t)
$$
因此，最小可能的交叉熵是真实分布熵 `H(t)`，最优模型满足：
$$
p = t
$$
对应的最小 perplexity 为：
$$
\text{PPL}_{\min}
=
\exp(H(t))
$$
如果使用 base-2：
$$
\text{PPL}_{\min}
=
2^{H_2(t)}
$$
原报告已经给出了这一推导，并将其总结为“把 perplexity 压低就逼近真实分布”的数学基础。

### 4.3 直觉解释

这个推导说明：如果测试数据真的来自世界的真实分布，那么能最好预测测试文本的模型，也就是最接近真实分布的模型。

**类比**：假设你要预测一个城市每天的天气。如果你的概率分布和真实天气机制完全一致，那么长期来看，你的 log loss 最低。语言模型也是类似：如果它学到了真实语言和世界文本的生成机制，就能把概率质量放在真正会出现的文本上。

### 4.4 Perplexity 为什么“超过你需要的东西”

课程马上指出，perplexity 也可能评估了很多你并不关心的东西。例子是：

> Stanford was founded in 1885.

预测 “1885” 体现世界知识；但预测 “was” 或 “founded” 这样的常见 token，未必直接对应我们想测的能力。原报告指出，perplexity 会对所有 token 的预测错误收取 bits，而很多 token 只是 incidental。

这带来一个问题：
$$
\text{PPL 低}
\not\Rightarrow
\text{在你关心的任务上一定强}
$$
例如，一个模型可能非常擅长预测常见语法和高频词，从而 PPL 很低，但它在复杂推理、工具使用、安全拒答或专业工作流中仍可能表现不足。

### 4.5 Conditional perplexity

为了解决“评测了太多无关 token”的问题，可以只评测 response 部分的条件困惑度：
$$
\text{PPL}(y\mid x)
=
\exp
\left(
-\frac{1}{|y|}
\sum_{j=1}^{|y|}
\log p_\theta(y_j \mid x, y_{<j})
\right)
$$
其中：

- `x` 是 prompt；
- `y` 是 response；
- `y_j` 是 response 的第 `j` 个 token；
- `p_\theta(y_j | x, y_{<j})` 是模型在 prompt 和已生成回答前缀下预测真实回答 token 的概率。

它的学习意义是：不要把用户 prompt 或不相关上下文也纳入评分，而是只问“模型对正确回答是否有高概率”。原报告也指出，`lecture_12.py` 明确写到可以测量 `p(response | prompt)^(1/|response|)` 形式的 conditional perplexity。

### 4.6 “伪装成 accuracy 的 perplexity”

课程还指出，一些 benchmark 表面上是 accuracy 或 multiple choice，本质仍然接近 next-token / sentence completion。典型例子包括：

- **LAMBADA**：给定长上下文，预测最后缺失的词；
- **HellaSwag**：从多个候选中选择最自然的句子补全。

对多选 sentence completion，可以为每个候选 `c` 计算：
$$
s(c)
=
\sum_{j=1}^{|c|}
\log
p_\theta(c_j \mid \text{context}, c_{<j})
$$
然后选择：
$$
c^\*
=
\arg\max_{c\in C}
s(c)
$$
如果正确选项得分最高，就算答对。这说明很多“下游 benchmark”并没有完全脱离语言建模，而是在用任务形式包装概率续写能力。

### 4.7 Perplexity leaderboard 的信任问题

Perplexity 评测有一个特殊问题：它要求模型返回合法概率或 log probability。课程提醒，如果有人提交一个“假语言模型”，总是返回 logprob 0，也就是概率 1，就会得到极低 perplexity，但这显然不是合法概率分布，因为所有可能序列的概率不能都等于 1。原报告也保留了这个 warning。

这与普通下游任务不同：
$$
\text{普通任务：}
\quad
\text{response} = \text{LM}(\text{prompt})
$$
普通任务只看输出是否正确，可以把模型当黑盒。Perplexity 则要相信模型暴露的概率分布，因此更需要代码审计、开源权重、可复现实验或可信执行环境。

### 4.8 学习提示

本节复习时要记住一句话：
$$
\boxed{
\text{Perplexity 在数学上很干净，但在任务语义上不总是充分。}
}
$$
它适合评估“模型是否能解释文本分布”，但不一定适合评估“模型是否真的有帮助、安全、会推理、能完成真实任务”。

------

## 5. 第四部分：考试型 Benchmark——从 MMLU 到 HLE

这一部分要解决的问题是：**如何用标准题评测模型的知识、推理和专业能力？**

### 5.1 问题引入

考试题适合作为 benchmark，因为它有几个优点：

- 可以控制学科；
- 可以控制难度；
- 通常有明确正确答案；
- 容易自动评分；
- 可以跨模型比较。

原报告也指出，课程认为考试可以 carefully control subject and difficulty，并设计 unambiguous correct answers。

### 5.2 基本指标：Accuracy

对于 `K` 选一的多选题，最简单指标是 accuracy：
$$
\text{Accuracy}
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbf{1}[\hat{a}_i = a_i]
$$
其中：

- `N` 是题目数量；
- `\hat{a}_i` 是模型对第 `i` 道题的预测答案；
- `a_i` 是正确答案；
- 指示函数等于 1 表示答对，等于 0 表示答错。

随机猜测的期望正确率是：
$$
\mathbb{E}[\text{Accuracy}_{\text{random}}]
=
\frac{1}{K}
$$
因此：

- 4 选 1 的 chance level 是 25%；
- 10 选 1 的 chance level 是 10%。

### 5.3 生成答案 vs logprob scoring

考试型 benchmark 常见两种评测方式。

第一种是 **生成式回答**：

```
Question: ...
A. ...
B. ...
C. ...
D. ...
Answer:
```

模型生成 “A/B/C/D”，评测器抽取最终答案。

第二种是 **logprob scoring**：
$$
\hat{a}
=
\arg\max_{a\in\{A,B,C,D\}}
\log p_\theta(a \mid \text{prompt})
$$
如果每个选项不是一个字母，而是一段较长文本，就需要长度归一化：
$$
\hat{a}
=
\arg\max_a
\frac{1}{|y_a|}
\sum_{j=1}^{|y_a|}
\log p_\theta(y_{a,j}\mid \text{prompt}, y_{a,<j})
$$
否则长选项因为 token 更多，联合概率通常更小，可能被系统性惩罚。原报告也强调，answer extraction 本身会影响评测结果，语言模型评测可能对此非常敏感。

### 5.4 MMLU：从语言模型到通用任务求解器

MMLU 全称 Massive Multitask Language Understanding，包含 57 个学科。课程特别指出，虽然名字里有 “Language Understanding”，但它更像是在测知识和推理，而不只是语言理解。原报告也保留了 `lecture_12.py` 中“MMLU 包含 57 个 subject，并用 few-shot prompting 在 GPT-3 上评测”的信息。

MMLU 的历史意义在于，它帮助人们把语言模型从“会续写流畅文本的模型”重新理解为“可能成为 general-purpose task solver 的模型”。

典型 few-shot prompt 结构是：

```
The following are questions about high school mathematics.

Question: ...
Answer: B

Question: ...
Answer: D

Question: [test question]
Answer:
```

**学习重点**：few-shot prompting 不是更新模型参数，而是在上下文中给模型示例，让模型通过 in-context learning 适应任务格式。

### 5.5 MMLU-Pro：benchmark 饱和后的升级

当 MMLU 被强模型逐渐接近饱和后，MMLU-Pro 作为更难版本被提出。原报告总结了它的三点改动：

1. 去掉 noisy / trivial questions；
2. 从 4 个选项扩展到 10 个选项；
3. 使用 chain-of-thought 评测，让模型有更充分推理空间。

`lecture_12.py` 还记录模型准确率在 MMLU-Pro 上下降 16% 到 33%，因此它暂时不那么饱和。

**直觉解释**：把 4 选 1 扩展到 10 选 1，不只是降低随机猜中的概率，也减少了模型通过浅层模式匹配碰巧命中的机会。

### 5.6 GPQA：Google-Proof QA

GPQA，即 Graduate-Level Google-Proof Q&A，试图构造 PhD 级别、难以直接搜索解决的问题。原报告记录了几个关键数字：题目由 61 名 Upwork 上的 PhD contractors 编写，PhD experts 达到 65% accuracy，非专家在 30 分钟 Google 辅助下约 34%，GPT-4 约 39%。

GPQA 的重要性不是只在于“题难”，还在于它试图对抗一个现代语言模型评测问题：如果题目答案可以轻易在网上搜到，而模型训练数据又来自互联网，那么模型可能并不是“推理出答案”，而是“记住或近似记住答案”。

### 5.7 Humanity’s Last Exam：继续推高难度

Humanity’s Last Exam，简称 HLE，是更激进的考试型 benchmark。原报告写到，它包含 2500 道题，覆盖多模态、多学科，题型包括多选和短答，并通过奖金、署名、frontier LLM filtering 和多阶段 review 激励与筛选题目。

HLE 也引出一个现代 benchmark 的共同问题：
$$
\boxed{
\text{Benchmark 有 shelf life。}
}
$$
当模型在某个 benchmark 上快速接近饱和后，研究者会提出更难的新 benchmark；但新 benchmark 也可能很快被追上，甚至被污染。因此 evaluation 是与模型能力共同演化的动态过程。

### 5.8 学习提示

考试型 benchmark 的复习重点：

| 应抓住的问题     | 正确理解                                    |
| ---------------- | ------------------------------------------- |
| 为什么考试题有用 | 可控、可比、易评分                          |
| 为什么考试题不够 | 远离真实使用，且可能污染或饱和              |
| MMLU 的意义      | 把 LM 推向多学科知识和任务求解评测          |
| MMLU-Pro 的意义  | benchmark 饱和后提高难度                    |
| GPQA 的意义      | 尝试减少“可 Google 搜到”的题目              |
| HLE 的意义       | 继续推高难度，并暴露 benchmark 生命周期问题 |

------

## 6. 第五部分：聊天型 Benchmark——开放式回答如何评价？

这一部分要解决的问题是：**当问题没有唯一标准答案时，如何评价模型回答质量？**

### 6.1 问题引入

现实用户很少问模型标准多选题。用户更可能问：

> 我想做甜菜山羊奶酪沙拉，哪些香草适合？哪些不适合？

这类问题没有唯一 ground truth，不能简单做 exact match，也不能只算 accuracy。原报告指出，开放式问题没有 ground truth，不能检查字符串相等。

因此，开放式评测的核心问题变成：
$$
\text{什么叫“更好”的回答？}
$$
可能标准包括：

- 是否事实正确；
- 是否满足用户意图；
- 是否表达清楚；
- 是否有帮助；
- 是否避免幻觉；
- 是否不过度迎合；
- 是否安全；
- 是否长度合适；
- 是否有结构。

### 6.2 Chatbot Arena / Arena AI：人类偏好与 Elo

Chatbot Arena 的基本流程是：

1. 真实用户输入 prompt；
2. 系统随机选择两个匿名模型回答；
3. 用户判断哪个回答更好；
4. 收集成对偏好数据；
5. 用 Elo 或类似模型估计全局排名。

每条偏好数据可以写成：
$$
(A,B,y)
$$
其中：

- `A` 和 `B` 是两个模型；
- `y=1` 表示 A 胜过 B；
- `y=0` 表示 B 胜过 A。

Elo 概率模型是：
$$
P(A \text{ beats } B)
=
\frac{1}{1+10^{(R_B-R_A)/400}}
$$
其中：

- `R_A` 是模型 A 的 rating；
- `R_B` 是模型 B 的 rating；
- rating 差越大，高分模型获胜概率越高。

对一组成对比较数据 `\mathcal{D}`，可以最大化 log-likelihood：
$$
\mathcal{L}(R)
=
\sum_{(i,j)\in \mathcal{D}}
\left[
y_{ij}\log p_{ij}
+
(1-y_{ij})\log(1-p_{ij})
\right]
$$
其中：
$$
p_{ij}
=
\frac{1}{1+10^{(R_j-R_i)/400}}
$$
原报告指出，Chatbot Arena 的好处是 prompt 来自真实用户；但也有用户分布不可控、bias、spammers、leaderboard gaming、style 和 correctness 混淆、sycophancy 等问题。

### 6.3 AlpacaEval：LLM-as-a-judge 与 win rate

AlpacaEval 的核心思想是：让被评测模型 `M` 和 baseline 模型 `B` 对同一 instruction 生成回答，再用一个强模型作为 judge 判断谁更好。

指标是相对 baseline 的 win rate：
$$
\text{WinRate}(M,B)
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbf{1}
[
\text{Judge}(r_i^M,r_i^B)=M
]
$$
其中：

- `r_i^M` 是模型 `M` 对第 `i` 个 instruction 的回答；
- `r_i^B` 是 baseline 的回答；
- judge 判断 `M` 更好则计 1。

问题在于，LLM judge 也有偏差。原报告提到，早期 AlpacaEval 发现 LLM judge 偏好更长回答，导致 leaderboard gaming；后续版本用回归方法对长度偏差进行 debias。

一个简化的长度去偏模型可以写成：
$$
\text{score}_i
=
\alpha
+
\beta_1 \cdot \text{quality}_i
+
\beta_2 \cdot \text{length}_i
+
\epsilon_i
$$
如果 `\beta_2 > 0` 反映的是长度偏好而非真实质量，就可以做：
$$
\text{adjusted score}
=
\text{raw score}
-
\hat{\beta}_{\text{length}}\cdot \text{length}
$$
**学习重点**：课程真正强调的不是这个具体回归公式，而是：
$$
\boxed{
\text{Metric 本身也需要被 evaluation。}
}
$$
也就是说，我们不仅要问模型好不好，还要问“这个评分器是否真的测到了我关心的质量”。

### 6.4 WildBench：用 checklist / rubric 明确评测标准

WildBench 的思路是为每个 prompt 生成 checklist，让 judge 按任务特定标准评价回答。原报告记录，WildBench 从 100 万人类-聊天机器人对话中采样 1024 个例子，使用 GPT-4 turbo 作为 judge，并引入 checklist。

以甜菜山羊奶酪沙拉为例，一个 rubric 可以包括：
$$
\begin{aligned}
&\text{是否推荐了适配甜菜和山羊奶酪的香草？}\\
&\text{是否解释了风味搭配理由？}\\
&\text{是否指出不适合的香草？}\\
&\text{是否避免过度武断？}\\
&\text{是否结构清楚？}
\end{aligned}
$$
这样，judge 不再只回答“哪个更好”这种模糊问题，而是根据任务相关标准逐项判断。

### 6.5 学习提示

开放式评测最容易混淆的是“用户喜欢”和“回答正确”。一个更长、更自信、更会迎合用户的回答可能赢得偏好投票，但未必事实更准确。

本节复习时抓住：
$$
\boxed{
\text{开放式评测没有唯一答案，所以需要偏好、judge、rubric 和偏差控制。}
}
$$

------

## 7. 第六部分：Agentic Benchmark——从“模型说什么”到“模型做什么”

这一部分要解决的问题是：**当模型能使用工具、运行代码、修改文件、长期迭代时，我们到底在评测模型，还是评测整个 agent 系统？**

### 7.1 问题引入

前面的 chat evaluation 主要评估模型“说什么”。但很多真实任务要求模型“做什么”：修 bug、运行命令、训练模型、攻击或防御安全环境、提交结果。这类任务需要工具使用、状态观察、错误修正和多步规划。

课程将 agent 定义为：
$$
\text{Agent}
=
(\text{LM}, \text{tools}, \text{memory}, \text{policy/scaffold})
$$
原报告也保留了这个思想：agent 是语言模型加上 scaffold，即决定如何调用语言模型、使用哪些工具、如何管理上下文和行动循环的逻辑。

### 7.2 Agent 的形式化过程

在第 `t` 步，agent 观察环境状态：
$$
o_t
$$
结合历史：
$$
h_t
$$
通过策略或 scaffold 调用语言模型生成动作：
$$
a_t
=
\pi_\theta(o_t,h_t)
$$
环境执行动作后返回新观察：
$$
o_{t+1}
=
E(a_t)
$$
最终根据任务是否完成评分。

**直觉解释**：chat benchmark 像“口试”，agentic benchmark 更像“实操考试”。模型不只是回答“应该怎么做”，而是要真的操作环境并交付结果。

### 7.3 SWE-bench：代码修复与单元测试

SWE-bench 的任务是：给定代码库和 GitHub issue 描述，让 agent 提交 patch / PR；评测方式是看单元测试是否通过。原报告记录，SWE-bench 包含 2294 个任务，来自 12 个 Python repositories，输入是 codebase + issue description，输出是 PR，指标是 unit tests。

基本 solve rate 可以写为：
$$
\text{SolveRate}
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbf{1}
[
\text{tests pass after patch}_i
]
$$
更严格地，若任务 `i` 有原始失败测试集 `F_i` 和回归测试集 `R_i`，希望：
$$
\forall f\in F_i,\quad
f(\text{patched code})=\text{pass}
$$
并且：
$$
\forall r\in R_i,\quad
r(\text{patched code})=\text{pass}
$$
也就是既修复 bug，又不破坏已有功能。

### 7.4 Terminal-Bench：通用终端环境任务

Terminal-Bench 把环境设定为计算机终端。模型通过 shell 命令完成任务。原报告记录，`lecture_12.py` 写到它使用 terminal environments，因为简单且通用；任务由 93 名贡献者众包，229 个任务中 89 个构成 Terminal-Bench 2.0。

它评测的是更通用的工具使用能力：
$$
\text{read files}
\rightarrow
\text{run commands}
\rightarrow
\text{observe errors}
\rightarrow
\text{revise plan}
\rightarrow
\text{finish task}
$$

### 7.5 CyBench：网络安全 CTF

CyBench 包含 Capture-the-Flag 任务。agent 可以查看源代码、访问 web server、运行命令，目标是找到 flag。原报告记录，CyBench 有 40 个 CTF tasks，并用 first-solve time 作为难度度量。

任务成功率可以写成：
$$
\text{ASR}_{\text{ctf}}
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbf{1}
[
\text{flag}_i \text{ extracted}
]
$$
这里的 `ASR` 可以理解为攻击任务成功率。但课程也提醒，网络安全任务天然具有 dual-use：同样能力既可以用于攻击系统，也可以用于防御、渗透测试和漏洞修复。

### 7.6 MLE-bench：机器学习工程任务

MLE-bench 让 agent 参加 Kaggle 风格任务：读取数据、理解描述、写代码、训练模型、提交结果。原报告记录它包含 75 个 Kaggle competitions，需要 training models、processing data 等。

典型工作流是：
$$
\text{understand task}
\rightarrow
\text{inspect data}
\rightarrow
\text{build baseline}
\rightarrow
\text{train}
\rightarrow
\text{evaluate}
\rightarrow
\text{iterate}
\rightarrow
\text{submit}
$$
这类 benchmark 的难点在于，它测的不是一次回答，而是长程任务规划和迭代改进能力。

### 7.7 Scaffold 为什么重要

课程非常强调：agentic evaluation 不只是评测语言模型，也是在评测 scaffold。原报告保留了 `lecture_12.py` 中列出的 scaffold 要素：
$$
\begin{aligned}
&\text{Explicit planning: todo list}\\
&\text{Hierarchical delegation: sub-agents}\\
&\text{Persistent memory: read/write files}\\
&\text{Extreme context engineering}
\end{aligned}
$$
因此，评测时必须说明对象是：
$$
\text{Evaluation target}
=
\text{LM only}
\quad
\text{or}
\quad
\text{LM + scaffold + tools}
$$
如果不声明，就无法判断高分来自模型本体，还是来自工程系统。

### 7.8 学习提示

本节最重要的一句话是：
$$
\boxed{
\text{Agentic benchmark 测的是完整行动系统，不只是语言模型本体。}
}
$$
复习时尤其注意：SWE-bench、Terminal-Bench、CyBench、MLE-bench 的指标看起来都是 solve rate / pass rate，但背后的环境、工具、scaffold 和任务长度完全不同。

------

## 8. 第七部分：纯推理 Benchmark——ARC-AGI 与 reasoning / knowledge 分离

这一部分要解决的问题是：**有没有办法尽量隔离模型的推理能力，而不是测它记住了多少世界知识？**

### 8.1 问题引入

前面的 benchmark 大多混合了语言理解、世界知识、专业知识和推理。比如 GPQA 需要专业知识，MMLU 需要学科知识，SWE-bench 需要代码库理解和工具使用。ARC-AGI 则试图更集中地测 reasoning / fluid intelligence。

原报告指出，ARC-AGI 任务旨在 100% 可由人类解决，但对 AI 有挑战；每个任务都是特殊的，记忆事实或旧题帮助不大。

### 8.2 概念定义：从少数例子归纳规则

ARC 类任务通常给出少数输入-输出网格例子：
$$
\{(X_1,Y_1),\dots,(X_k,Y_k)\}
$$
模型需要推断一个变换规则：
$$
\{(X_1,Y_1),\dots,(X_k,Y_k)\}
\rightarrow
f
$$
然后对新输入 `X_*` 生成：
$$
Y_* = f(X_*)
$$
这里的核心不是知道某个事实，而是从少量例子中抽象出规则。

### 8.3 方法与原理

ARC-AGI 的评测对象更接近：

- 模式识别；
- 抽象规则归纳；
- 组合变换；
- 少样本推断；
- 多步推理。

它试图减少模型通过互联网记忆取胜的空间。但课程也强调，“纯推理”很难完全与知识分离，因为任何任务都带有人类设计先验。

### 8.4 意义与局限

原报告记录，课程指出 ARC-AGI-1 在早期预训练语言模型上几乎没有进展；后来 reasoning models 出现后，成绩明显提升。课程也强调，ARC-AGI 限制在人类可解范围内，因此不覆盖超人推理，比如开放数学问题或竞赛金牌级推理。

**延伸理解**：ARC-AGI 的价值在于，它提醒我们不要把“知识覆盖率”误认为“推理能力”。但它也不能完全定义智能，因为真实智能还包括长期记忆、工具使用、语言交流、目标管理、实验设计等能力。

### 8.5 学习提示

复习时抓住：
$$
\boxed{
\text{ARC-AGI 的目标不是测知识量，而是尽量测少样本规则归纳能力。}
}
$$
最容易误解的是把它叫作“纯智能测量”。更谨慎的说法是：它是一个试图减少知识依赖、突出推理短板的 benchmark。

------

## 9. 第八部分：Safety Benchmark——安全到底测什么？

这一部分要解决的问题是：**AI safety 不是一个单一分数，那么安全 benchmark 到底在测哪些风险？**

### 9.1 问题引入

“安全”听起来像一个明确目标，但实际上非常复杂。模型可能会：

- 回答有害请求；
- 幻觉事实；
- 过度迎合用户；
- 协助违法行为；
- 放大偏见；
- 让用户丧失批判思考；
- 在不同法律文化环境中表现不一致。

原报告也指出，安全高度依赖政治、法律、社会规范和国家差异，不同风险需要不同关注。

### 9.2 HarmBench：有害请求拒答

HarmBench 是常见安全评测类型：给模型有害 prompt，看模型是否拒绝。原报告记录，HarmBench 基于 510 个违反法律或规范的 harmful behaviors。

可以定义拒答率：
$$
\text{RefusalRate}
=
\frac{1}{N}
\sum_i
\mathbf{1}
[
\text{model refuses harmful request}_i
]
$$
也可以定义 attack success rate：
$$
\text{ASR}
=
\frac{1}{N}
\sum_i
\mathbf{1}
[
\text{model complies with harmful request}_i
]
$$
粗略地：
$$
\text{ASR}
=
1-\text{RefusalRate}
$$
但现实中情况更复杂：回答可能部分遵从、模糊拒绝、先拒绝后给出关键细节，或给出安全替代方案。因此安全评测通常需要更细致的 judge 或 rubric。

### 9.3 AIR-Bench：更系统的风险分类

HarmBench 主要覆盖 harmful instruction refusal，而 AIR-Bench 试图更全面地构建 AI 风险 taxonomy。原报告记录，AIR-Bench 基于 regulatory frameworks 和 company policies，taxonomy 包含 314 个风险类别和 5694 个 prompts。

这说明：
$$
\text{safety}
\neq
\text{refuse harmful prompt only}
$$
安全包括多种风险类型：
$$
\text{harmful instruction},
\quad
\text{hallucination},
\quad
\text{sycophancy},
\quad
\text{crime assistance},
\quad
\text{inequality},
\quad
\text{loss of critical thinking}
$$

### 9.4 Jailbreaking 与 GCG

课程提到 jailbreak：模型经过安全训练后会拒绝有害请求，但攻击者可能构造特殊 prompt 绕过安全机制。原报告记录，GCG，即 Greedy Coordinate Gradient，可以自动优化 prompt 来绕过 safety，并且可能从 open-weight model 转移到 closed model。

可以抽象为寻找一个 adversarial suffix `s`：
$$
s^\*
=
\arg\max_s
\log p_\theta(y_{\text{harm}} \mid x_{\text{harm}}, s)
$$
其中：

- `x_harm` 是有害请求；
- `s` 是附加的对抗性后缀；
- `y_harm` 是攻击者希望模型生成的有害回答。

GCG 的简化思想是每次替换 suffix 中某个 token，使目标 log probability 增大：
$$
s_{t+1}
=
\text{UpdateOneToken}(s_t)
$$
并希望：
$$
\log p_\theta(y_{\text{harm}}\mid x,s_{t+1})
>
\log p_\theta(y_{\text{harm}}\mid x,s_t)
$$
**安全说明**：这里的学习重点不是如何实施攻击，而是理解为什么安全评测必须包含对抗性 prompt。普通 harmful prompt 上安全，不代表在 adversarial setting 下安全。

### 9.5 学习提示

本节复习时要记住：
$$
\boxed{
\text{安全评测不是一道拒答题，而是一组情境化、对抗性、多风险维度的评测。}
}
$$
最容易误解的是把高拒答率等同于高安全性。过度拒答也会损害有益性，因此安全评测常常需要同时看 harmlessness 与 helpfulness。

------

## 10. 第九部分：Realism / Ecological Validity——评测是否贴近真实使用？

这一部分要解决的问题是：**benchmark 分数高，是否代表真实用户和真实职业场景中也表现好？**

### 10.1 生态有效性

Ecological validity 指评测能否捕捉真实世界使用。原报告指出，考试型 benchmark 如 GPQA 距离真实使用较远；Chatbot Arena 的 prompt 来自真人，但用户分布不可控。`lecture_12.py` 也把 ecological validity 定义为 evaluation 捕捉 real-world use 的程度。

可以把它理解为评测分布与真实分布的差异：
$$
q_{\text{eval}}(x,y)
\quad
\text{vs.}
\quad
q_{\text{real}}(x,y)
$$
如果二者差别很大，那么 benchmark 分数再高，也未必代表真实用户体验好。

### 10.2 GDPVal：按职业任务构造评测

原报告记录，GDPVal 覆盖美国 GDP 前 9 个行业中的 44 种职业，任务由平均约 14 年经验的专业人士创建。

这类评测与考试题不同。它不问“模型会不会答标准题”，而是问“模型能不能完成真实职业任务”。例如：

- 护士可能关心病历摘要与沟通；
- 房地产经纪人可能关心合同、市场分析和客户沟通；
- 视频编辑可能关心素材整理、剪辑建议和工作流辅助。

### 10.3 MedHELM：医学评测从考试走向临床任务

医学评测尤其说明了 realism 的重要性。模型通过医学考试，并不意味着可以直接临床部署。原报告记录，MedHELM 包含 121 个 clinical tasks，由 29 位 clinicians 提供，混合私有和公共数据集。

关键公式是：
$$
\text{exam competence}
\neq
\text{real-world professional competence}
$$
考试能测知识，但真实临床任务还涉及上下文、责任、风险、沟通、隐私和 workflow。

### 10.4 Clio：真实用户数据与隐私张力

课程还提到 Anthropic 的 Clio：模型开发者拥有真实用户数据，但不能直接公开查看和发布用户对话。因此可以用语言模型对用户数据做聚合分析，分享总体模式。原报告也总结了 realism 与 privacy 的张力。

可以写成：
$$
\text{more realism}
\quad
\Longleftrightarrow
\quad
\text{more privacy risk}
$$
最真实的数据是实际 query stream，但实际 query stream 又最敏感。因此真实评测必须与隐私保护共同设计。

### 10.5 学习提示

本节最重要的结论是：
$$
\boxed{
\text{越真实的评测越有用，但也越难控制、越难公开、越容易涉及隐私。}
}
$$
复习时要区分：

- **difficulty**：题难不难；
- **realism**：像不像真实使用；
- **validity**：是否真的测到了目标能力。

------

## 11. 第十部分：Validity——评测是否科学有效？

这一部分要解决的问题是：**即使 benchmark 看起来合理，我们怎么知道分数可信？**

### 11.1 Train-test contamination

在 foundation model 时代，train-test overlap / contamination 是最重要的评测有效性问题之一。原报告指出，传统机器学习中 ImageNet、SQuAD 等有清晰 train-test split；而今天模型训练在互联网和大量未公开数据上，外部评测者不知道 test set 是否进入训练数据。

显式污染可以写成：
$$
D_{\text{test}}
\subset
D_{\text{train}}
$$
但更常见的是隐式污染：
$$
D_{\text{test}} = g(S),
\qquad
S \subset D_{\text{train}}
$$
也就是说，测试题本身可能没出现在训练集中，但题目的来源材料、解法、近似副本或讨论帖已经出现在训练数据中。

### 11.2 四种应对路线

原报告总结了课程中列出的几种处理污染的方法。

| 路线               | 方法                                     | 直觉                                     | 局限                                 |
| ------------------ | ---------------------------------------- | ---------------------------------------- | ------------------------------------ |
| 从模型行为推断污染 | 利用题目顺序 exchangeability 等信号      | 如果模型记得原始顺序，可能见过 benchmark | 只能发现部分污染                     |
| 建立报告规范       | 模型提供方声明 train-test overlap        | 像报告置信区间一样报告数据重叠风险       | 依赖提供方诚实和审计                 |
| fresh evals        | 使用训练 cutoff 之后的新网页、代码、论文 | 降低旧数据污染风险                       | 时间戳不总可靠，新内容可能复制旧资料 |
| private evals      | 使用内部代码库、私有文本、未公开任务     | 不容易进入训练数据                       | 不公开导致难复现、难比较             |

### 11.3 Dataset quality：测试题本身可能有错

即使没有污染，benchmark 本身也可能质量不高。原报告提到 SWE-bench 之后出现 SWE-bench Verified，因为原始任务和单元测试存在问题；GSM8K、MMLU 等也有人审计出题目错误。

对于 agentic benchmark，问题更严重，因为它不只是题目文本，还包括：

- 完整环境；
- 依赖版本；
- 测试用例；
- 工具链；
- 文件系统状态；
- agent trace；
- 判分脚本。

因此课程建议：做 benchmark 或跑 benchmark 时，一定要看模型输出和 agent trace，确认测到的是你想测的东西。

### 11.4 统计可靠性：置信区间

如果 benchmark 有 `N` 道独立题，模型答对 `k` 道，则 accuracy 点估计为：
$$
\hat{p}
=
\frac{k}{N}
$$
近似 95% 置信区间为：
$$
\hat{p}
\pm
1.96
\sqrt{
\frac{\hat{p}(1-\hat{p})}{N}
}
$$
对于 win rate 也是类似：
$$
\widehat{\text{WinRate}}
=
\frac{k}{N}
$$
标准误差约为：
$$
\text{SE}
=
\sqrt{
\frac{\hat{p}(1-\hat{p})}{N}
}
$$
原报告也指出，当 `N` 很小时，两个模型差 1% 或 2% 可能没有统计意义。

### 11.5 学习提示

本节复习时记住：
$$
\boxed{
\text{一个 benchmark 分数可信，需要同时考虑污染、题目质量和统计不确定性。}
}
$$
最容易误解的是只看 leaderboard 上的一个点估计。正确做法是看样本量、置信区间、数据来源、评测规则、是否污染、是否有人工审计。

------

## 12. 第十一部分：我们到底在评测什么——方法、模型还是系统？

这一部分要解决的问题是：**同一个 benchmark 分数，到底归功于算法、模型、数据、prompt、工具，还是完整产品系统？**

### 12.1 问题引入

课程最后提出一个非常关键的问题：评测对象到底是什么？

在 foundation model 之前，很多 benchmark 评测的是方法：
$$
\text{fixed data}
+
\text{fixed evaluation}
+
\text{different algorithms}
$$
也就是说，数据集和评测规则固定，研究者比较的是算法。

但今天的语言模型评测往往是在评测完整模型或系统：
$$
\text{model}
+
\text{data}
+
\text{post-training}
+
\text{prompting}
+
\text{tools}
+
\text{scaffold}
+
\text{inference strategy}
$$
原报告也称这是一种 “anything goes” 的评测语境。

### 12.2 规则必须说清楚

每个评测都应回答：
$$
\text{What is allowed?}
$$
例如：

| 规则问题                        | 为什么重要                   |
| ------------------------------- | ---------------------------- |
| 是否允许 chain-of-thought       | 会显著影响推理题表现         |
| 是否允许工具                    | 会改变知识、计算、代码能力   |
| 是否允许检索                    | 会影响 factual QA 与专业任务 |
| 是否允许多次采样                | 会提高 pass@k，但成本也变高  |
| 是否允许人工后处理              | 可能让结果不再代表自动系统   |
| 评测 base model 还是 chat model | post-training 会显著改变行为 |
| 评测 LM 还是 agent system       | scaffold 可能贡献大量性能    |

### 12.3 学习提示

本讲最后的核心结论是：
$$
\boxed{
\text{There is no one true evaluation. Choose what you are trying to measure.}
}
$$
原报告也把这句话作为总结，强调不存在唯一万能评测。好的 evaluation 必须说明：想测什么、为什么这个 benchmark 能代表它、评分是否可靠、是否贴近真实使用、是否存在污染，以及评测对象到底是模型还是完整系统。

------

## 13. Python 代码示例详解

### 13.1 文件整体结构

本讲对应的主要代码文件是 Stanford CS336 仓库中的 `lecture_12.py`。当前公开仓库 README 说明可执行讲义命名为 `lecture_XX.py`，并给出用 `python execute.py -m lecture_01` 编译讲义的方式；仓库文件列表中也包含 `lecture_12.py`。

`lecture_12.py` 不是传统意义上的训练代码，而是**可执行讲义脚本**。它通过 `text()`、`image()`、`link()` 等函数组织课程内容、图表和链接。它的作用更像“课程内容生成器”，而不是模型训练脚本。

主流程可以概括为：

```
def main():
    text("## Lecture 12: evaluation")
    ...
    what_is_good()
    perplexity()
    exam_benchmarks()
    chat_benchmarks()
    agentic_benchmarks()
    pure_reasoning_benchmarks()
    safety_benchmarks()
    realism()
    validity()
    how_to_think_about_evaluation()
```

【Python代码†lecture_12.py】

### 13.2 主要函数列表

| 函数                              | 对应课程部分 | 主要内容                                                | 学习重点                          |
| --------------------------------- | ------------ | ------------------------------------------------------- | --------------------------------- |
| `main()`                          | 全讲入口     | 引出 evaluation，并按顺序调用各模块                     | 理解本讲主线                      |
| `what_is_good()`                  | 什么是好模型 | benchmark、成本、人类偏好、用户使用量                   | “好模型”没有唯一标准              |
| `perplexity()`                    | 困惑度       | PPL、传统 LM 数据集、GPT-2 OOD、conditional PPL         | PPL 的数学基础和局限              |
| `exam_benchmarks()`               | 考试型评测   | MMLU、MMLU-Pro、GPQA、HLE                               | 从标准题测知识和推理              |
| `chat_benchmarks()`               | 聊天评测     | Chatbot Arena、Elo、AlpacaEval、WildBench               | 开放式回答如何评分                |
| `agentic_benchmarks()`            | Agent 评测   | SWE-bench、Terminal-Bench、CyBench、MLE-bench、scaffold | 从“说”到“做”                      |
| `pure_reasoning_benchmarks()`     | 纯推理       | ARC-AGI                                                 | reasoning 与 knowledge 的分离尝试 |
| `safety_benchmarks()`             | 安全         | HarmBench、AIR-Bench、GCG、dual-use                     | 安全的多维性与对抗性              |
| `realism()`                       | 真实性       | GDPVal、MedHELM、Clio                                   | 评测是否贴近真实使用              |
| `validity()`                      | 有效性       | contamination、fresh/private evals、dataset quality     | 分数是否可信                      |
| `how_to_think_about_evaluation()` | 总结         | 评测目的、评测对象、规则                                | 没有唯一 evaluation               |

### 13.3 关键代码片段解析

#### 13.3.1 `what_is_good()`：把“好模型”拆成多个视角

该函数通过文本、链接和图片展示：

- benchmark score；
- benchmark score + cost；
- 人类偏好；
- 用户是否真实使用和付费。

它对应本讲开头的核心思想：evaluation 不只是机械算分，而是要先明确“goodness”定义。

**学习重点**：不要把某个排行榜当成模型质量的完整定义。排行榜只代表一种测量视角。

#### 13.3.2 `perplexity()`：从概率分布到 PPL

该函数从语言模型是 `p(x)` 开始，给出 PPL 形式：
$$
(1/p(D))^{1/|D|}
$$
然后讲传统数据集、in-distribution evaluation、GPT-2 的 out-of-distribution evaluation，以及 “Perplexity is all you need” 的信仰与局限。

**对应公式**：
$$
\text{PPL}(D)
=
\exp
\left(
-\frac{1}{M}
\sum_{i=1}^{M}
\log p_\theta(x_i \mid x_{<i})
\right)
$$
**学习重点**：代码中的 `text()` 不是在计算 PPL，而是在组织讲义内容；数学含义需要通过公式理解。

#### 13.3.3 `chat_benchmarks()`：从开放式回答到 pairwise preference

该函数先给出甜菜山羊奶酪沙拉例子，说明开放式回答没有唯一标准答案；随后介绍 Chatbot Arena、Elo、AlpacaEval 和 WildBench。

**对应公式**：
$$
P(A \text{ beats } B)
=
\frac{1}{1+10^{(R_B-R_A)/400}}
$$
**学习重点**：开放式回答质量很难直接打分，pairwise comparison 通常比绝对分数信号更强，但仍有偏差。

#### 13.3.4 `agentic_benchmarks()`：评测完整行动系统

该函数引入 SWE-bench、Terminal-Bench、CyBench、MLE-bench，并特别强调 agent scaffold，包括 planning、sub-agents、persistent memory 和 context engineering。

**对应概念**：
$$
\text{Agent}
=
(\text{LM}, \text{tools}, \text{memory}, \text{policy/scaffold})
$$
**学习重点**：agentic benchmark 的高分不能简单归因于 LM 本体，必须声明 scaffold 和工具规则。

### 13.4 代码与课程知识点对应表

| 课程知识点            | 对应代码函数 / 变量 / 片段    | 代码功能                           | 对应理论原理                     | 在课程中的作用                 |
| --------------------- | ----------------------------- | ---------------------------------- | -------------------------------- | ------------------------------ |
| Evaluation 的核心挑战 | `what_is_good()`              | 展示多个“好模型”定义               | 抽象 construct → concrete metric | 奠定全讲视角                   |
| Perplexity            | `perplexity()`                | 讲解 PPL 与传统 LM 评测            | NLL、cross-entropy、概率分布     | 连接训练目标与评测             |
| OOD evaluation        | `perplexity()` 中 GPT-2 部分  | 展示从 WebText 到标准数据集        | 分布迁移                         | 说明 foundation model 评测范式 |
| Exam benchmark        | `exam_benchmarks()`           | MMLU、GPQA、HLE 等                 | accuracy、few-shot prompting     | 测知识和标准题能力             |
| Chat evaluation       | `chat_benchmarks()`           | Arena、AlpacaEval、WildBench       | pairwise preference、Elo、judge  | 测开放式回答质量               |
| Agent evaluation      | `agentic_benchmarks()`        | SWE-bench 等                       | task success、tool use           | 测模型行动能力                 |
| Pure reasoning        | `pure_reasoning_benchmarks()` | ARC-AGI                            | rule induction                   | 分离 reasoning 与 knowledge    |
| Safety                | `safety_benchmarks()`         | HarmBench、AIR-Bench、GCG          | refusal、ASR、adversarial prompt | 测安全与对抗鲁棒性             |
| Realism               | `realism()`                   | GDPVal、MedHELM、Clio              | ecological validity              | 测真实使用代表性               |
| Validity              | `validity()`                  | contamination、fresh/private evals | 科学评测可靠性                   | 判断分数是否可信               |

------

## 14. PDF / 幻灯片 / 图表解析

### 14.1 资料状态说明

本轮资料中没有单独提供 `lecture_12.pdf`。当前公开 CS336 仓库中，Lecture 12 对应的是 `lecture_12.py`，而不是 PDF 文件；该 Python 文件通过 `image()` 引入多张课程图表。因此，本节将“PDF / 幻灯片 / 图表解析”理解为对 `lecture_12.py` 中图表的学习化解析。

### 14.2 主要图表总览

| 图表 / 页面                | 展示内容                         | 支持的课程结论                    | 对应课程部分      |
| -------------------------- | -------------------------------- | --------------------------------- | ----------------- |
| Artificial Analysis        | 模型 benchmark 与成本            | 好模型不只看分数，也要看推理成本  | 什么是好模型      |
| Arena AI leaderboard       | 人类偏好排名                     | 真实用户偏好可作为评测信号        | 聊天型 benchmark  |
| OpenRouter 排名            | 用户实际使用量                   | 市场选择也是一种评价视角          | 什么是好模型      |
| GPT-2 perplexity 图        | GPT-2 在标准数据集 zero-shot PPL | OOD evaluation 成为新范式         | Perplexity        |
| LAMBADA 图                 | cloze / fill-in-the-blank        | 部分 benchmark 是 PPL in disguise | PPL 局限          |
| HellaSwag 图               | 多选句子补全                     | 多选也可以测语言续写概率          | PPL 局限          |
| MMLU 图                    | 多学科多选题                     | LM 可作为通用任务求解器           | 考试 benchmark    |
| MMLU-Pro 图                | 更难的 MMLU 变体                 | benchmark 饱和后需要升级          | 考试 benchmark    |
| GPQA 图                    | PhD 级 Google-proof 问题         | 困难 benchmark 设计成本高         | 考试 benchmark    |
| HLE pipeline / results     | 高难题目构建与结果               | benchmark 有生命周期              | 考试 benchmark    |
| Arena beets 图             | 沙拉开放式回答比较               | 开放式回答无唯一标准答案          | 聊天 benchmark    |
| AlpacaEval correlation 图  | 与 Arena 的相关性                | metric 本身也要被评估             | LLM-as-a-judge    |
| WildBench 图               | checklist / rubric               | 结构化标准提升评测可靠性          | 聊天 benchmark    |
| SWE-bench 图               | 代码修复任务                     | agent 能力需要真实环境            | Agentic benchmark |
| Terminal-Bench 图          | 终端任务                         | 通用工具使用能力                  | Agentic benchmark |
| CyBench 图                 | CTF 任务                         | 网络安全能力有 dual-use           | Agentic / Safety  |
| MLE-bench 图               | Kaggle 任务                      | 长程 ML 工程能力                  | Agentic benchmark |
| ARC-AGI 图                 | 网格推理任务                     | 尽量隔离 reasoning                | 纯推理            |
| AIR-Bench 图               | 风险 taxonomy                    | safety 是多维的                   | Safety            |
| GDPVal / MedHELM / Clio 图 | 真实职业、临床、用户数据         | realism 与 privacy 张力           | Realism           |
| contamination 图           | train-test overlap 检测          | 分数有效性需要检查                | Validity          |

### 14.3 关键图表详细解释

#### 14.3.1 GPT-2 perplexity 图

这类图通常比较 GPT-2 在不同标准数据集上的 zero-shot PPL。它要回答的问题是：一个在 WebText 上训练的大模型，能否迁移到标准语言模型数据集？

读图时注意：

- 横向通常是不同数据集；
- 纵向是 PPL；
- PPL 越低表示模型越能解释该数据集；
- 小数据集上迁移收益可能更明显，大数据集上不一定。

**学习提示**：这张图服务于 in-distribution 到 out-of-distribution evaluation 的范式转变。

#### 14.3.2 MMLU / MMLU-Pro 图

MMLU 图表通常展示不同模型在多个学科或总体上的 accuracy。MMLU-Pro 图则强调难度升级，例如去噪、增加选项、使用 chain-of-thought。

读图时不要只看总分，还要问：

- 哪些学科强？
- 哪些学科弱？
- 是否接近饱和？
- 是否与真实任务相关？
- 是否可能污染？

#### 14.3.3 Chatbot Arena 图

Arena 图展示模型在成对人类偏好下的排名。它支持的结论是：真实用户偏好可以作为开放式回答质量的一个信号。

但读图时最容易误解的是：Arena 高分不一定表示事实性最高。它可能混合了风格、长度、语气、自信程度、用户群体偏差和 sycophancy。

#### 14.3.4 SWE-bench / Terminal-Bench 图

这类图通常展示 agent 在真实代码库或终端任务中的 solve rate。它们的意义是：模型能力不再只是“输出文本”，而是“在环境中完成任务”。

读图时要特别问：

- 评测的是 base LM、chat model，还是完整 agent？
- 使用了哪些工具？
- 是否允许多轮执行？
- 是否有人工提示？
- 单元测试是否充分？

#### 14.3.5 AIR-Bench / Safety 图

AIR-Bench 图展示安全风险 taxonomy，支持课程结论：AI safety 不是一个简单二分类问题。

读图时要注意：风险类别越细，评测越能覆盖多样安全问题；但也越依赖政策、文化和法律背景。

------

## 15. 关键公式与推导汇总

| 公式                                                       | 符号解释                          | 适用场景           | 直觉解释                        | 代码 / 例子对应 |
| ---------------------------------------------------------- | --------------------------------- | ------------------ | ------------------------------- | --------------- |
| `p_\theta(x_1,\dots,x_T)=\prod_t p_\theta(x_t\mid x_{<t})` | `x_t` 为 token，`x_{<t}` 为上下文 | 自回归语言模型     | 序列概率分解成逐 token 条件概率 | Perplexity      |
| `NLL=-\frac{1}{M}\sum_i \log p_\theta(x_i\mid x_{<i})`     | `M` 是 token 数                   | 语言模型训练和评测 | 正确 token 概率越低，损失越高   | 训练目标        |
| `PPL=\exp(NLL)`                                            | PPL 是平均 NLL 指数               | LM 评测            | 平均“候选困惑数”                | `perplexity()`  |
| `H(t,p)=H(t)+KL(t                                          |                                   | p)`                | `t` 真实分布，`p` 模型分布      | PPL 理论分析    |
| `PPL(y                                                     | x)=\exp(-\frac{1}{                | y                  | }\sum_j\log p(y_j               | x,y_{<j}))`     |
| `Accuracy=\frac{1}{N}\sum_i 1[\hat a_i=a_i]`               | `\hat a_i` 预测，`a_i` 正解       | 多选 benchmark     | 答对比例                        | MMLU、GPQA、HLE |
| `P(A beats B)=1/(1+10^{(R_B-R_A)/400})`                    | `R_A,R_B` 为 Elo rating           | Chatbot Arena      | rating 差决定偏好胜率           | Arena           |
| `SolveRate=\frac{1}{N}\sum_i 1[\text{tests pass}]`         | `N` 为任务数                      | SWE-bench 等       | 完成任务比例                    | Agentic eval    |
| `ASR=\frac{1}{N}\sum_i 1[\text{model complies}]`           | ASR 为攻击成功率                  | 安全 / jailbreak   | 有害请求成功绕过比例            | HarmBench / GCG |
| `\hat p \pm 1.96\sqrt{\hat p(1-\hat p)/N}`                 | `\hat p` 点估计，`N` 样本量       | 统计可靠性         | 分数有不确定性                  | 置信区间        |

### 15.1 重点推导：为什么最优 cross-entropy 在 `p=t`

$$
H(t,p)
=
-\sum_x t(x)\log p(x)
$$

加减 `\sum_x t(x)\log t(x)`：
$$
H(t,p)
=
-\sum_x t(x)\log t(x)
+
\sum_x t(x)\log\frac{t(x)}{p(x)}
$$
得到：
$$
H(t,p)
=
H(t)+\mathrm{KL}(t||p)
$$
由于：
$$
\mathrm{KL}(t||p)\ge 0
$$
所以：
$$
H(t,p)\ge H(t)
$$
当且仅当：
$$
p=t
$$
时取等号。

**一句话理解**：如果测试数据来自真实分布，那么最能预测它的模型就是最接近真实分布的模型。

------

## 16. 概念对比与易混淆点

| 易混淆概念                        | 区别                                             | 联系                           | 常见误解               | 正确理解                             |
| --------------------------------- | ------------------------------------------------ | ------------------------------ | ---------------------- | ------------------------------------ |
| Perplexity vs accuracy            | PPL 评概率，accuracy 评答案对错                  | 多选可用 logprob 转成 accuracy | PPL 低就一定任务强     | PPL 是自然但不充分的 LM 指标         |
| In-distribution vs OOD eval       | 前者 train/test 同分布，后者不同分布             | 都可用 PPL 或任务分数          | OOD 分数低一定模型差   | 也可能是分布不匹配                   |
| MMLU vs MMLU-Pro                  | MMLU-Pro 更难、更多选项、去噪                    | 都是考试型 benchmark           | MMLU 饱和后就没意义    | 仍可用于历史比较，但不能单独代表能力 |
| Human judge vs LLM judge          | 前者来自人类偏好，后者自动化便宜                 | 都可做 pairwise eval           | LLM judge 一定客观     | LLM judge 也有长度、风格和模型偏差   |
| Chatbot Arena vs AlpacaEval       | Arena 用真人偏好，AlpacaEval 常用 LLM judge      | 都评开放式回答                 | Arena 分数就是事实性   | Arena 混合用户偏好和回答风格         |
| Agent benchmark vs chat benchmark | agent 需要行动和工具，chat 主要输出文本          | 都可能用 LM                    | agent 高分就是 LM 高分 | scaffold 和工具贡献很大              |
| Safety refusal vs safety          | 拒答只是安全一部分                               | 有害请求评测常用拒答           | 拒答越多越安全         | 过度拒答会损害 helpfulness           |
| Difficulty vs realism             | 难度是题难，真实性是像真实任务                   | 高质量 benchmark 需要二者平衡  | 越难越真实             | HLE 很难但未必贴近日常使用           |
| Validity vs leaderboard score     | validity 问分数是否可信                          | 分数需要有效性支撑             | 排名靠前就代表真实更强 | 需要检查污染、样本量、题目质量       |
| Model vs system                   | model 是权重，system 包括 prompt、工具、scaffold | 部署时常评 system              | 不说明规则也能公平比较 | 必须明确 allowed resources           |

------

## 17. 学习拓展与发散理解

### 17.1 与大模型训练实践的关系

**补充说明：Evaluation 会反过来塑造训练数据。**

如果某个 benchmark 被广泛重视，模型开发者就会倾向于收集相关数据、设计相关 post-training、优化相关 prompt 或工具使用能力。这不是坏事，但会带来两种风险：

1. **过拟合 benchmark**：模型越来越会考试，但不一定真实更有用；
2. **目标偏移**：模型优化的是可测指标，而不是原本抽象目标。

在大模型训练实践中，本讲内容影响至少五个环节：

| 训练 / 部署环节 | Evaluation 的影响                            |
| --------------- | -------------------------------------------- |
| 数据准备        | 决定需要收集什么能力相关数据                 |
| 预训练          | PPL 和 validation loss 用于监控训练          |
| 后训练          | chat eval、安全 eval 影响 SFT / RLHF 数据    |
| 系统设计        | agent benchmark 推动工具使用和 scaffold      |
| 部署监控        | realism 和 private eval 用于真实场景质量检查 |

### 17.2 与其他 lecture 的联系

**补充说明**：本讲与前后 lecture 的联系可以理解为：

- 与前面 scaling laws 的联系：PPL 是 scaling laws 中常见的平滑指标，但本讲提醒 PPL 不等于所有能力；
- 与 inference 的联系：模型质量不只看分数，也看推理成本、latency、工具使用；
- 与后续 data lecture 的联系：在决定训练数据前，要知道希望模型具备什么行为；
- 与 alignment lecture 的联系：chat preference、安全拒答、LLM judge、sycophancy 都会进入 alignment 讨论；
- 与 systems lecture 的联系：agentic benchmark 强调工具环境、执行成本、系统 scaffold。

### 17.3 面试 / 考试 / 作业可能考察角度

| 问题                                   | 考察点         | 回答思路                                        |
| -------------------------------------- | -------------- | ----------------------------------------------- |
| 为什么 perplexity 是语言模型自然指标？ | LM 是概率分布  | 说明 NLL、PPL、next-token prediction            |
| PPL 低为什么不等于真实任务强？         | 指标局限       | PPL 评所有 token，不区分任务相关性              |
| 推导 `H(t,p)=H(t)+KL(t                 |                | p)`                                             |
| MMLU 和 MMLU-Pro 有什么差别？          | benchmark 演化 | 去噪、10 选项、CoT、降低饱和                    |
| Chatbot Arena 的优缺点是什么？         | 人类偏好评测   | 真实 prompt、有偏差、style/correctness 混淆     |
| LLM-as-a-judge 有什么风险？            | judge bias     | 长度偏差、模型偏差，需要 debias 和 sanity check |
| 为什么 agentic eval 不只是评 LM？      | 系统边界       | scaffold、tools、memory、prompt 都影响结果      |
| 什么是 train-test contamination？      | validity       | 显式污染和隐式污染都要考虑                      |
| 为什么 private eval 有用？             | 污染控制       | 不易进入训练数据，但难公开复现                  |
| 为什么 benchmark 有 shelf life？       | 动态演化       | 模型变强、数据泄露、题目饱和导致 benchmark 失效 |

------

## 18. 课程复习总结

### 18.1 本讲核心思想

CS336 Lecture 12 的核心不是介绍某一个固定 benchmark，而是训练一种 evaluation 思维方式：任何评测都必须先说明想测什么，再把抽象目标转化为具体任务、数据、环境和评分规则。Perplexity 是语言模型最自然、数学上最清楚的指标，但它不足以覆盖真实用户需求；考试型、聊天型、agentic、推理、安全、真实世界和有效性评测分别覆盖不同维度。最终，不存在唯一万能评测，只有与目标匹配、规则清楚、统计可靠、尽量真实且有效的评测设计。

### 18.2 必须掌握的概念

- Evaluation：抽象目标到具体指标的映射；
- Perplexity：平均 NLL 的指数；
- In-distribution / OOD evaluation；
- Conditional perplexity；
- MMLU、MMLU-Pro、GPQA、HLE；
- Chatbot Arena、Elo、AlpacaEval、WildBench；
- Agent scaffold；
- SWE-bench、Terminal-Bench、CyBench、MLE-bench；
- ARC-AGI；
- HarmBench、AIR-Bench、GCG；
- Ecological validity；
- Train-test contamination；
- Fresh eval / private eval；
- Dataset quality；
- Confidence interval；
- Method vs model vs system。

### 18.3 必须理解的公式

| 公式                                               | 解决的问题                 |
| -------------------------------------------------- | -------------------------- |
| `p(x)=\prod_t p(x_t                                | x_{<t})`                   |
| `PPL=\exp(-\frac{1}{M}\sum_i\log p(x_i             | x_{<i}))`                  |
| `H(t,p)=H(t)+KL(t                                  |                            |
| `Accuracy=\frac{1}{N}\sum_i1[\hat a_i=a_i]`        | 多选 benchmark 如何评分    |
| `P(A beats B)=1/(1+10^{(R_B-R_A)/400})`            | Arena 如何从偏好得到排名   |
| `SolveRate=\frac{1}{N}\sum_i1[\text{task solved}]` | agentic benchmark 如何评分 |
| `\hat p \pm 1.96\sqrt{\hat p(1-\hat p)/N}`         | 分数不确定性如何估计       |

### 18.4 必须看懂的代码

| 代码部分               | 学习重点                                   |
| ---------------------- | ------------------------------------------ |
| `main()`               | 全讲顺序：从好模型定义到 validity 总结     |
| `what_is_good()`       | 多视角定义模型质量                         |
| `perplexity()`         | PPL 的地位、传统范式和局限                 |
| `exam_benchmarks()`    | MMLU 到 HLE 的难度演化                     |
| `chat_benchmarks()`    | 开放式回答的偏好评测                       |
| `agentic_benchmarks()` | LM + scaffold + tools 的系统评测           |
| `safety_benchmarks()`  | 安全评测的多维与对抗性                     |
| `realism()`            | 真实世界评测与隐私张力                     |
| `validity()`           | 污染、fresh/private evals、dataset quality |

### 18.5 一句话总结

$$
\boxed{
\text{Evaluation 不是寻找唯一分数，而是为你真正关心的模型行为设计可信、清晰、可解释的测量方式。}
}
$$





