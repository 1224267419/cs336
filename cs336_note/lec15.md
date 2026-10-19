# Lecture 15 课程报告：预训练之后——从 SFT 到 RLHF、PPO 与 DPO 的后训练学习讲义

## 0. 资料来源与改写说明

本报告是在你上传的原始课程报告 `lec15.md` 基础上进行的系统性改写、结构优化和学习化增强，原报告主题为 **CS336 Lecture 15: After Pretraining / Mid-Post Training**。本次改写保留原报告中的课程主线、公式、代码片段、PDF 页码标记、字幕引用标记和核心结论。

关于 GitHub 资料，当前 `stanford-cs336/lectures` 仓库文件列表中确实包含 `lecture_15.pdf`，并且仓库 README 说明可执行讲义命名为 `lecture_XX.py`，非可执行讲义命名为 `lecture_XX.pdf`；因此 Lecture 15 在仓库中对应的是 PDF 讲义，而不是独立的 `lecture_15.py`。本报告只将这一点用于核对资料类型，不额外编造不存在的 Python 文件内容。

本报告中的引用格式沿用原报告：

- **【字幕†source】**：表示原报告中标注为来自课程字幕的内容；
- **【PDF p.X†source】**：表示原报告中标注为来自 Lecture 15 PDF 对应页码的内容；
- **【PDF代码片段†source】**：表示原报告中标注为来自 PDF 内部展示的 Python 或伪代码片段；
- **延伸理解 / 补充说明 / 工程直觉 / 学习提示**：表示为了帮助学习而加入的解释，不把它们伪装成课程原文。

------

## 1. 学习导读

### 1.1 本讲要解决的核心问题

Lecture 15 讨论的是：**一个已经完成大规模预训练的 base model，如何被改造成更有用、更安全、更会遵循指令的 assistant model？**

预训练让模型获得大量语言、知识、代码和推理能力，但这些能力并不会自动以“用户想要的形式”稳定展现出来。后训练的目标，就是在预训练模型已有能力的基础上，通过 **SFT、mid-training、RLHF、PPO、DPO、安全调优和偏好优化** 等方法，让模型更接近 ChatGPT / InstructGPT 这类产品形态。【字幕†source】【PDF p.2-p.6†source】

一句话说，本讲的核心问题是：

> **预训练给模型“能力”，后训练决定模型“如何表现这些能力”。**

### 1.2 本讲在 CS336 课程中的位置

CS336 的主线是从零理解和构建语言模型。前面的 lecture 更多关注模型结构、预训练数据、训练系统、优化与评估；Lecture 15 则进入一个更产品化、更复杂也更“脏”的阶段：**after pretraining**。

本讲和大模型训练实践的关系非常直接：

| 课程阶段   | 本讲对应的问题                                      | 实际意义                               |
| ---------- | --------------------------------------------------- | -------------------------------------- |
| 预训练之后 | base model 如何变成 instruction-following model     | 决定模型能否作为助手使用               |
| 数据设计   | SFT / RLHF 数据如何构造                             | 决定模型学到什么风格、知识和安全边界   |
| 优化算法   | PPO / DPO 如何利用偏好数据                          | 决定模型如何从“模仿答案”转向“优化偏好” |
| 安全与对齐 | 如何减少有害输出和过度拒答                          | 决定模型能否可靠部署                   |
| 评估与风险 | 如何避免 length bias、reward hacking、mode collapse | 决定模型是否真的变好，而不只是指标变高 |

### 1.3 本讲知识结构

| 模块         | 解决的问题                           | 核心概念                                                     | 与后续内容的关系                                  |
| ------------ | ------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------- |
| 后训练动机   | 为什么预训练不够                     | base model、assistant behavior、post-training                | 引出 SFT 和 RLHF                                  |
| SFT          | 如何用示范数据教模型听指令           | supervised fine-tuning、next-token prediction、response mask | 后续 RLHF 通常从 SFT 模型开始                     |
| SFT 数据演化 | 什么样的数据能诱导 assistant 行为    | FLAN、Self-Instruct、Alpaca、OpenAssistant、tool use         | 说明数据比算法更关键                              |
| SFT 风险     | 为什么好数据也可能带来坏行为         | chattiness、tail knowledge、hallucination、safety SFT        | 连接偏好优化和安全调优                            |
| Mid-training | 预训练和后训练边界为何模糊           | two-phase training、instruction data mixing                  | 解释现代 base model 为什么不再“纯”                |
| RLHF         | 如何从“模仿参考答案”转向“优化偏好”   | pairwise feedback、reward model、policy                      | 引出奖励模型、PPO、DPO                            |
| PPO          | 如何用强化学习优化语言模型           | KL regularization、policy gradient、clipping                 | 经典 RLHF 方法，但工程复杂                        |
| DPO          | 如何不用显式 reward model 做偏好优化 | implied reward、preference loss、reference policy            | 现代开放模型常用方法之一                          |
| 副作用       | 偏好优化为什么可能把模型带坏         | reward hacking、overoptimization、mode collapse、calibration loss | 连接下一讲 reasoning / RLVR 中的 exploration 问题 |

### 1.4 学习本讲前需要知道什么

学习本讲最好具备以下前置知识：

- **语言模型训练目标**：next-token prediction、交叉熵、负对数似然；
- **概率建模**：条件概率、log probability、softmax；
- **深度学习优化**：梯度下降、反向传播、AdamW、学习率调度；
- **基础强化学习**：policy、reward、policy gradient、KL 正则；
- **Transformer / LLM 基础**：prompt、completion、logits、token；
- **模型评估直觉**：benchmark、human preference、length bias；
- **大模型安全基本概念**：拒答、有害请求、误拒、幻觉。

学习提示：本讲最容易混淆的是 **SFT 和 RLHF 的目标不同**。SFT 是“模仿参考答案”，RLHF / DPO / PPO 是“根据偏好信号调整概率分布”。

------

## 2. 第一部分：为什么需要后训练——从 base model 到 assistant model

### 2.1 问题引入

课程从一个现实问题开始：我们已经有了 GPT-3 级别的大规模预训练模型，为什么还需要额外的后训练？

直觉上，预训练模型像一个读过大量文本的系统。它可能知道很多东西，也可能能写代码、做摘要、回答问题，但它不一定知道：

- 用户现在是在请求帮助；
- 应该直接回答还是解释过程；
- 哪些问题应该拒绝；
- 答案应该多长；
- 是否应该使用工具；
- 如何稳定遵循复杂指令；
- 如何避免输出有害或虚假的内容。

因此，从 GPT-3 式 base model 到 ChatGPT 式 assistant model，中间缺少一个关键环节：**把潜在能力转化为可控行为**。【字幕†source】【PDF p.2-p.6†source】

### 2.2 概念定义

#### Base model

直觉解释：base model 是主要通过大规模文本预测训练出来的模型。它擅长“继续写文本”，但不一定擅长“作为助手服务用户”。

正式理解：base model 的主要目标是建模文本分布：
$$
p_\theta(x_t \mid x_{<t})
$$
它学习的是在上下文 `x_{<t}` 给定时，下一个 token `x_t` 出现的概率。

#### Post-training

直觉解释：post-training 是在预训练之后继续塑造模型行为的过程。

正式理解：post-training 包括 SFT、RLHF、安全调优、偏好优化、工具调用训练等。它不一定主要增加模型的底层知识，而是让模型更稳定地表现出人类想要的行为。【字幕†source】

#### Mid-training

直觉解释：mid-training 位于“纯预训练”和“正式后训练”之间。它通常在预训练后期混入更高质量、更结构化或更接近指令的数据。

正式理解：课程指出，现代模型常在预训练尾部或第二阶段训练中混入 instruction-tuning data、高质量领域数据、问答数据或 UltraChat 等，使得所谓 base model 已经带有一定 assistant-like 行为。【字幕†source】【PDF p.29-p.30†source】

### 2.3 方法与原理

后训练解决的问题可以分成三层：

第一层是 **行为抽取**：模型可能已经在预训练中见过代码、解释、问答、教程、对话。SFT 的作用是告诉模型：“当用户用指令形式提问时，请把这些能力组织成 assistant-style response。”

第二层是 **偏好优化**：人类不一定能写出最优答案，但通常能比较两个答案谁更好。这引出 RLHF：用 pairwise preference 或 reward model 优化模型行为。

第三层是 **安全边界控制**：模型不仅要有用，还要避免被用于诈骗、垃圾邮件、钓鱼、恶意攻击或危险内容生成。【字幕†source】【PDF p.22-p.26†source】

### 2.4 为什么后训练很 “messy”

课程强调，后训练并不是一个纯粹由算法决定效果的阶段。公开论文通常会介绍 SFT、PPO、DPO 等方法，但真正影响模型行为的往往是数据：

- 什么 prompt 被采样；
- 谁来写参考答案；
- 答案应该多长；
- 是否包含引用；
- 是否包含工具调用；
- 安全边界如何定义；
- 标注者偏好如何统一；
- 如何避免模型学到“看似正确”的引用格式但实际幻觉。

课程指出，ChatGPT 之后，frontier lab 往往不会公开完整的 post-training 数据、标注规范和数据配比，因此这个领域在公开资料中显得尤其不透明。【字幕†source】【PDF p.5†source】

**学习提示**

后训练不是“在预训练后再训练一点点”这么简单。它是一个由数据、标注、偏好、奖励、安全策略和优化算法共同决定的系统工程。

------

## 3. 第二部分：SFT——用示范答案教模型遵循指令

### 3.1 问题引入

课程首先介绍 SFT，因为它是 InstructGPT 流程中的第一步：先收集 demonstration data，再用这些示范答案做监督微调；之后才进入 comparison data、reward model 和 PPO 等阶段。【PDF p.6†source】

SFT 要解决的问题是：

> 给定用户指令，如何让模型更倾向于输出人类希望看到的 assistant response？

### 3.2 概念定义

SFT，全称 **Supervised Fine-Tuning**，即监督微调。

训练样本通常是：
$$
(x, y)
$$
其中：

- `x` 表示用户指令、上下文、多轮对话历史或系统提示；
- `y` 表示期望模型生成的回答；
- `π_θ` 表示待训练语言模型；
- `θ` 表示模型参数。

SFT 的本质仍然是 next-token prediction，只不过训练分布从普通互联网文本换成了 instruction-response 数据。【PDF p.32†source】

### 3.3 SFT 损失函数

SFT 的基本目标是最大化参考答案 `y` 在模型下的概率，等价于最小化 completion token 的负对数似然：
$$
\mathcal{L}_{\text{SFT}}(\theta)
=
-\mathbb{E}_{(x,y)\sim \mathcal{D}_{\text{SFT}}}
\sum_{t=1}^{|y|}
\log \pi_\theta(y_t \mid x, y_{<t})
$$
符号解释：

| 符号                       | 含义                                      |
| -------------------------- | ----------------------------------------- |
| `\mathcal{D}_{\text{SFT}}` | SFT 训练数据集                            |
| `x`                        | prompt、用户指令或对话上下文              |
| `y`                        | assistant response，即参考答案            |
| `y_t`                      | 参考答案中的第 `t` 个 token               |
| `y_{<t}`                   | 当前 token 之前的答案 token               |
| `π_θ(y_t \mid x, y_{<t})`  | 模型在上下文下预测正确 token 的概率       |
| `-\log`                    | 负对数似然，正确 token 概率越低，损失越大 |

直觉解释：如果参考答案中下一个 token 是 “because”，而模型只给它很低概率，那么损失就会很大；训练会推动模型以后在类似上下文中更倾向于生成这个 token。

### 3.4 Response mask：为什么有时只训练回答部分

在 instruction tuning 中，常见做法是只对 assistant response 部分计算损失，而不对 prompt 部分计算损失。原报告给出的 masked SFT 形式是：
$$
\mathcal{L}_{\text{SFT-mask}}(\theta)
=
-\mathbb{E}_{(x,y)}
\sum_t m_t \log \pi_\theta(s_t \mid s_{<t})
$$
其中：

- `s_t` 表示 prompt 与 response 拼接后的第 `t` 个 token；
- `m_t = 1` 表示该 token 属于 assistant response，需要计算损失；
- `m_t = 0` 表示该 token 属于 prompt，不计入损失。

为什么要这样做？

直觉上，prompt 是用户给模型看的输入，而不是希望模型学习“生成用户指令”。如果训练时对 prompt 也计算损失，模型会同时学习“如何写用户问题”和“如何回答问题”。这在纯语言建模中合理，但在 assistant 微调中通常更希望模型专注于学习回答行为。

不过课程也提到，在某些 mid-training 数据中，prompt 不一定被 mask，训练形式可能更接近普通 pretraining，即同时预测 prompt 与 response。【字幕†source】

### 3.5 SFT 的直观含义：不是创造能力，而是抽取行为

课程中一个重要观点是：SFT 更像是在 **抽取预训练模型已经具备的能力**，而不是硬塞全新能力。

例如，预训练模型可能已经读过大量 Python 教程、Stack Overflow 问答、数学解释和代码片段。SFT 的作用是让模型学会：当用户说 “Find the average number in a list.” 时，应该组织成一个清晰的 Python 函数，而不是随机续写某段网页文本。【PDF代码片段†source】

原报告给出的 Alpaca 代码例子是：

```
def avg_list(nums):
    total = 0
    for n in nums:
        total += n
    return total / len(nums)

print(avg_list([4, 3, 6, 10, 8]))  # Output 6.4
```

这个例子说明，SFT 数据不是在教模型从零理解“平均数”或 Python 循环，而是在强化一种行为模式：

1. 读懂自然语言指令；
2. 判断用户想要代码；
3. 写出函数；
4. 正确遍历列表；
5. 返回总和除以长度；
6. 给出示例调用和输出。【PDF代码片段†source】

**学习提示**

SFT 的核心不是“多训练一点”，而是“改变训练分布”。它把模型从互联网文本续写模式，推向 instruction-following assistant 模式。

------

## 4. 第三部分：SFT 数据如何演化——从 FLAN 到工具调用与安全调优

### 4.1 问题引入

有了 SFT 目标函数之后，下一个问题是：**SFT 数据应该长什么样？**

课程按时间线介绍了开放世界 SFT 数据的发展，包括 FLAN、Self-Instruct、Alpaca、ShareGPT / Vicuna、OpenAssistant、WizardLM、Tulu3、Nemotron 以及工具调用数据。【PDF p.9†source】【字幕†source】

这些数据集的变化说明，后训练并不是只有“公式和代码”，更关键的是“示范行为”的设计。

### 4.2 FLAN：把传统 NLP 数据集改写成指令

FLAN 的思想是：NLP 社区已有大量监督数据集，可以把它们统一转成 instruction-output 格式。

原报告列出的例子包括：

- 从邮件正文预测 subject line；
- 给新闻文本做分类；
- 对文章写 highlights；
- 根据结构化餐厅信息生成一句话。【PDF p.10-p.14†source】

直觉上，FLAN 像是把很多 benchmark 任务包装成自然语言指令。它的优点是任务覆盖广、监督信号明确；缺点是这些数据仍然偏 benchmark-like，不太像真实用户和聊天助手之间的自然对话。

课程指出，这类数据虽然有监督价值，但用户并不想“和 NLP benchmark 对话”。【PDF p.10-p.14†source】

**补充说明**

FLAN 适合教模型理解“指令—输出”格式，但不一定能教会模型真实产品中需要的语气、上下文组织、解释深度和安全边界。

### 4.3 Self-Instruct：让模型自己生成训练数据

Self-Instruct 的核心思想是利用已有模型生成新的 instruction-response 数据。课程认为这个方向非常 forward-looking，因为它预示了后续 synthetic data 的重要性：模型越强，越可以作为数据生成器。【字幕†source】

直觉上，这是一个自举过程：

1. 先有一个还不错的模型；
2. 让它生成大量指令和回答；
3. 过滤、清洗、改写这些数据；
4. 再用这些数据训练模型。

这种思路后来在很多开放模型、偏好数据和合成推理数据中反复出现。

### 4.4 Alpaca：蒸馏 ChatGPT 风格

Alpaca 使用 ChatGPT traces 生成更自然的 input-output pair。课程强调，这类数据的回答更长、更像聊天、更符合用户对 assistant 的期待；当它用于原始 LLaMA 这样的强预训练模型时，可以较可靠地诱导出 ChatGPT-like 行为。【字幕†source】【PDF p.11†source】

这里体现了一个关键结论：

> **SFT 不是单独起作用的。强预训练模型 + 合适的 SFT 数据，才能诱导出强 instruction-following 行为。**

如果底座模型本身很弱，SFT 数据再像 ChatGPT，也可能只是学到表面格式；如果底座模型很强，少量高质量 SFT 数据就可能显著改变行为。

### 4.5 OpenAssistant：更长、更复杂、更像专家回答

OpenAssistant 代表了更人类驱动、更高质量、更长答案的 instruction data。课程用 “monopsony” 例子说明，OpenAssistant 回答可能包含详细解释和学术引用，例如 Bivens & Mishel 的论文引用。【PDF p.12/p.19†source】

这个例子的重要性不在于 monopsony 本身，而在于它揭示了 SFT 的一个风险：模型可能同时学习到知识内容和输出格式。

如果训练数据中有很多“看起来像学术引用”的回答，模型可能学会：

- 用户问复杂概念时要给长解释；
- 用户要求引用时要输出引用格式；
- 答案看起来越正式越好。

但如果模型内部并不知道某个引用是否真实，它可能学到引用格式，却在具体文献上产生幻觉。课程把这称为知识与格式的纠缠问题。【字幕†source】

**学习提示**

SFT 数据越复杂，越不只是“教答案”。它也在教模型风格、格式、语气、引用习惯和事实输出边界。

### 4.6 Tool-use 与 agentic behavior：从聊天到工具调用

现代 SFT 数据越来越强调 tool use 和 agentic behavior。原报告提到，PDF 中 Nemotron-SFT-OpenCode-v1 的例子展示 assistant 不只是生成普通文本，还会生成结构化 JSON 和 `tool_calls`，例如调用 `"skill"` 或写入 todo list。【PDF p.13-p.14†source】

这意味着 SFT 的目标正在扩展：

| 传统 SFT     | 现代 tool-use / agentic SFT       |
| ------------ | --------------------------------- |
| 回答用户问题 | 判断是否需要调用工具              |
| 生成自然语言 | 生成 JSON / 函数调用格式          |
| 单轮问答     | 多步任务分解                      |
| 直接输出结果 | 维护任务状态、todo list、工具结果 |
| 学聊天风格   | 学 agent 工作流                   |

从工程角度看，工具调用 SFT 训练的是“协议遵循能力”：模型不仅要知道答案，还要知道何时把意图表示成结构化动作。

### 4.7 SFT 数据中的关键变化因素

课程总结了 SFT 数据中几个真正影响模型行为的因素。

#### 4.7.1 Chattiness：更像助手，不一定更有能力

FLAN 类数据通常是短答案、分类标签、摘要、结构化输出；Alpaca 和 OpenAssistant 则更长、更详细、更像人类助手。课程把这种变化称为 chattiness。【PDF p.14-p.16†source】

但更健谈不等于更强。模型更会列项目符号、更会解释、更礼貌，不一定代表数学、事实或代码能力真的提升。

#### 4.7.2 Length bias：长答案容易被偏好

课程指出，人类和 GPT-based evaluator 都可能偏好更长、更详细、带 bullet points 的回答。某些 SFT 数据可能提升 AlpacaEval 等偏好指标，却不明显提升 MMLU、GSM、BBH、TyDiQA、Code-Eval 等标准 benchmark。【字幕†source】【PDF p.17-p.18†source】

这说明：
$$
\text{Preference score} \uparrow
\not\Rightarrow
\text{Capability} \uparrow
$$
**学习提示**

复习时要区分两件事：
 模型“看起来更会回答”是 style improvement；模型“真的更会解题”才是 capability improvement。

#### 4.7.3 Tail knowledge：事实正确也可能带来幻觉

课程提出一个重要经验法则：在 SFT 阶段微调模型“不知道的事实”，可能增加幻觉。【字幕†source】【PDF p.19-p.21†source】

原因是 SFT 不只是教事实，还教行为模式。假设训练数据里有很多带引用的长答案，模型可能学到“遇到引用请求就输出引用”。如果它并不知道真实文献，就可能输出虚假引用。

课程中提到 tail knowledge 没有严格形式化定义，可能用 Wikipedia 条目长度、出现频率等 proxy 衡量模型是否大概率知道某个事实。【字幕†source】【PDF p.19-p.21†source】

**工程直觉**

如果目标是让模型知道更多事实，SFT 不是唯一也不总是最好的方法。对于事实可靠性，检索增强、工具调用、拒绝未知问题、偏好反馈和事实核验数据都可能更合适。

#### 4.7.4 Safety SFT：降低有害输出，也要避免误拒

面向用户部署时，模型必须处理 misinformation、scams、spam、spearphishing 等滥用风险。课程指出，post-training 团队在某种意义上是防止模型被滥用的最后一道防线。【字幕†source】【PDF p.22-p.26†source】

安全调优需要同时降低两个指标：
$$
\text{Violation Rate} \downarrow
$$
即有害请求通过率要下降。
$$
\text{False Refusal Rate} \downarrow
$$
即正常请求不能被错误拒绝。

课程举例说，用户问 “how do I kill a Python process” 时，模型不能因为看到 “kill” 就拒绝，因为这是正常编程问题。【字幕†source】

课程还指出，约 500 个 Alpaca-style safety examples 就能明显降低有害指令遵循率；但如果要实现细粒度安全边界，仍需要长尾数据和更大规模标注。【PDF p.26-p.27†source】

**学习提示**

安全 SFT 的难点不只是“拒绝坏请求”，而是学会区分上下文：同一个词在不同场景下可能完全无害。

------

## 5. 第四部分：SFT 训练方法与 Mid-training

### 5.1 问题引入

课程在方法部分说 SFT “最无聊”，意思是：相比数据设计的复杂性，SFT 的训练代码本身非常接近普通语言模型训练循环。【PDF代码片段†source】

### 5.2 SFT 的训练循环

PDF 中展示了典型 PyTorch 风格训练流程：

```
from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
```

这段代码对应 SFT 的核心计算：

| 代码                            | 作用                    | 对应理论                |
| ------------------------------- | ----------------------- | ----------------------- |
| `model.train()`                 | 切换训练模式            | 启用 dropout 等训练行为 |
| `batch = {k: v.to(device) ...}` | 把数据放到 GPU / device | 准备 mini-batch         |
| `outputs = model(**batch)`      | 前向传播                | 计算 logits 和 loss     |
| `loss = outputs.loss`           | 取交叉熵损失            | SFT 负对数似然          |
| `loss.backward()`               | 反向传播                | 计算 `∇_θ L_SFT`        |
| `optimizer.step()`              | 更新参数                | 梯度下降 / AdamW        |
| `lr_scheduler.step()`           | 更新学习率              | 控制训练动态            |
| `optimizer.zero_grad()`         | 清空梯度                | 避免梯度跨 batch 累积   |
| `progress_bar.update(1)`        | 更新进度条              | 工程监控                |

从数学上看，这就是对 SFT loss 做随机梯度下降或 AdamW 优化。【PDF代码片段†source】

### 5.3 代码与公式的对应关系

SFT 公式是：
$$
\mathcal{L}_{\text{SFT}}(\theta)
=
-\mathbb{E}_{(x,y)}
\sum_{t=1}^{|y|}
\log \pi_\theta(y_t \mid x, y_{<t})
$$
代码中的 `outputs.loss` 通常就是模型内部根据 `input_ids`、`labels` 和可能的 `attention_mask` / label mask 计算出来的 token-level cross entropy。

也就是说：

| 数学对象         | 代码对象                               | 含义                    |
| ---------------- | -------------------------------------- | ----------------------- |
| `x`              | prompt tokens / input_ids 中的用户部分 | 输入指令                |
| `y`              | labels 中的 assistant response         | 参考答案                |
| `π_θ`            | `model`                                | 当前语言模型            |
| `-\log π_θ(...)` | cross entropy                          | 正确 token 的负对数概率 |
| `m_t`            | label mask 或 `-100` label             | 控制哪些 token 计入损失 |

**学习提示**

SFT 的训练代码看起来普通，但关键在数据：prompt-response 如何拼接、哪些 token 计入 loss、数据比例如何设置，都会显著影响模型行为。

### 5.4 Mid-training / Two-phase training

课程强调，现代模型中 pretraining 和 post-training 的边界正在模糊。越来越多模型会在预训练尾部的 decay phase 或第二阶段训练中混入高质量数据、问答数据、instruction data、UltraChat 等，然后再做一个较短的正式 SFT。【字幕†source】【PDF p.29-p.30†source】

典型流程可以理解为：

```
General Pretraining
        ↓
High-quality / Instruction-mixed Mid-training
        ↓
Shorter Final SFT
        ↓
Preference Optimization / Safety Tuning
```

这样做的作用包括：

- 放大 instruction data 的影响；
- 降低单独 SFT 导致的 catastrophic forgetting 风险；
- 让模型在进入后训练前已经更像 assistant；
- 使“base model”概念变得不再纯粹，因为很多所谓 base model 已经见过 chat / instruction 数据。【字幕†source】

**补充说明**

在现代训练流程中，模型能力不是在某一个阶段突然产生的。预训练、mid-training、SFT 和 RLHF 会共同塑造最终模型。因此比较不同模型时，不能只看参数量，也要看训练阶段和数据配方。

------

## 6. 第五部分：从 SFT 到 RLHF——从模仿答案到优化偏好

### 6.1 问题引入

SFT 的限制在于：它只能模仿参考答案。如果参考答案不是最优，或者人类很难直接写出最优答案，SFT 就会受限。

课程用 **G-V gap**，即 generation-verification gap，解释为什么 RLHF 必要：很多任务中，验证一个答案好不好，比自己生成一个好答案更容易。【字幕†source】

例如：

- 写数学证明很难，但检查证明是否有漏洞相对容易；
- 写高质量代码很难，但 code review 能指出明显问题；
- 写完美摘要很难，但比较两个摘要谁更好相对容易。

### 6.2 概念转变：SFT 是 imitation，RLHF 是 optimization

SFT 的目标可以写成：
$$
\hat{p}(y\mid x) \approx p^*(y\mid x)
$$
它是在拟合参考行为分布。

RLHF 的目标是：
$$
\max_{\pi_\theta}
\mathbb{E}_{y\sim \pi_\theta(\cdot\mid x)}
[R(x,y)]
$$
它把语言模型看成 policy，用 reward 衡量输出好坏，然后优化 policy。【PDF p.32/p.50†source】

核心区别如下：

| 维度     | SFT                        | RLHF                                    |
| -------- | -------------------------- | --------------------------------------- |
| 学习信号 | 参考答案                   | 偏好 / 奖励                             |
| 训练方式 | 模仿学习                   | 强化学习 / 偏好优化                     |
| 模型角色 | 条件语言模型               | policy                                  |
| 优点     | 简单稳定                   | 能优化人类偏好                          |
| 风险     | 模仿错误答案、学到表面格式 | reward hacking、mode collapse、校准下降 |

课程特别强调，RLHF 后语言模型不再只是一个拟合真实文本分布的 probabilistic model。理论上，如果某个回答 reward 最高，策略可能对每个 prompt 坍缩到一个固定高 reward 输出，从而丢失多样性。【PDF p.32/p.50†source】

### 6.3 Pairwise feedback 数据

RLHF 通常从 SFT 后模型开始，对同一个 prompt 采样多个回答，再让标注者比较哪个更好。最常见的数据形式是：
$$
(x, y_w, y_l)
$$
其中：

- `x` 是 prompt；
- `y_w` 是 winner / chosen response；
- `y_l` 是 loser / rejected response。

课程提到，采样通常可以使用 temperature 1，因为 SFT 后模型已经有一定多样性。【字幕†source】【PDF p.35-p.36†source】

### 6.4 标注目标：helpful, truthful, harmless

InstructGPT 风格标注指南通常要求评价回答是否：

- **Helpful**：清楚、有用、符合用户意图，不过度冗长；
- **Truthful**：避免错误、幻觉和虚假信息；
- **Harmless**：拒绝危险、有害或不当请求。

Google Bard 早期标注示例也有类似维度，包括 helpfulness、presentation、factuality 等。【字幕†source】【PDF p.37-p.38†source】

**学习提示**

RLHF 的关键不是“标注者喜欢哪个答案”这么简单，而是要定义什么叫“好答案”。如果定义模糊，模型会学习到混乱偏好。

### 6.5 标注者不是中性的

课程强调，不同标注者的背景会改变模型行为。教育背景、文化背景、政治倾向、专业程度都会影响他们对“好回答”的判断。Santurkar 等研究显示，RLHF 后模型观点可能向标注者群体偏移；Hosking 等研究显示，非专家标注者更容易偏好 formatting，而专家更关注 factuality 和 inconsistency。【字幕†source】【PDF p.43-p.44†source】

这说明 RLHF 数据质量取决于：

- 谁在标注；
- 标注者是否有专业知识；
- 标注规范是否清晰；
- 标注者是否认真检查事实；
- 标注者是否被长度、格式、礼貌程度误导。

### 6.6 众包与专家标注的难点

课程指出，众包 RLHF 数据存在很多现实问题：

- 很难找到高质量、可验证的标注者；
- 很难保证标注者真的做事实检查；
- 很难防止标注者偷偷用 AI 生成标注；
- 大规模低薪标注存在劳动伦理问题；
- 医学、法律、代码等任务越来越需要专家标注，但成本很高。【PDF p.39-p.42†source】

**工程直觉**

对于开放域聊天，普通标注者可能足够评价语气和有用性；但对于医学、法律、科学、复杂代码，非专家标注可能会奖励“看起来很专业但实际错误”的回答。

### 6.7 LM-generated feedback：用强模型做标注者

课程指出，在开放模型追赶 frontier model 的场景中，AI feedback 已经非常常见。GPT-4 级别模型可以作为不错的 pairwise feedback system，在系统级排序上接近人类标注。UltraFeedback、Zephyr、OLMo、Tulu3 等流程都大量使用模型生成或模型偏好数据。【字幕†source】【PDF p.45-p.46†source】

但课程也区分了两种情况：

- 如果目标是追赶已有 frontier model，模型标注很有用；
- 如果目标是推动 frontier，或需要律师、医生、科学家等真实世界专业知识，仍然需要人类专家数据。【字幕†source】

### 6.8 Constitutional AI 与 self-training

Anthropic 的 Constitutional AI 被课程用作 self-training 示例：模型先生成有害 prompt 的回答，再根据 constitution 生成 critique 和 revision，最终训练出更安全的模型。【PDF p.47†source】【字幕†source】

这个过程展示了一种自举式后训练循环：

```
模型生成回答
   ↓
根据规则批评回答
   ↓
根据批评改写回答
   ↓
把改写结果用于训练
```

**学习提示**

RLHF 数据不一定全部来自人类手写，也可以来自强模型、规则、constitution 或 self-training 流程。但这些反馈信号本身也会带来偏差。

### 6.9 RLHF 的长度效应

课程提醒，偏好评价容易产生 length effect。人类或模型 judge 可能持续偏好更长、更详细的回答，因此只优化偏好分数可能把模型推向冗长输出。【PDF p.48†source】【字幕†source】

这和 SFT 中的 chattiness / style bias 是一脉相承的。

------

## 7. 第六部分：奖励模型、PPO 与 DPO

### 7.1 奖励模型：把 pairwise preference 转成 reward

在 RLHF 中，常先训练一个 reward model：
$$
r_\phi(x,y)
$$
它接收 prompt `x` 和 response `y`，输出一个标量 reward。

给定 winner `y_w` 和 loser `y_l`，常用 Bradley-Terry / logistic preference 模型：
$$
P(y_w \succ y_l \mid x)
=
\sigma(r_\phi(x,y_w)-r_\phi(x,y_l))
$$
其中 sigmoid 函数是：
$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$
奖励模型训练损失为：
$$
\mathcal{L}_{\text{RM}}(\phi)
=
-\mathbb{E}_{(x,y_w,y_l)\sim \mathcal{D}}
\log \sigma(r_\phi(x,y_w)-r_\phi(x,y_l))
$$
直觉解释：

- 如果人类偏好 `y_w`，reward model 应该给 `y_w` 更高分；
- 如果 `r_\phi(x,y_w) - r_\phi(x,y_l)` 越大，模型认为 `y_w` 更好的概率越接近 1；
- 如果 reward model 给 loser 更高分，loss 会很大，训练会纠正它。【PDF p.52†source】【字幕†source】

### 7.2 PPO 的目标：最大化 reward，同时不要偏离参考模型太远

InstructGPT / Stiennon 风格 RLHF 目标可写为：
$$
\max_{\pi_\theta}
\mathbb{E}_{x,y\sim\pi_\theta}
\left[
r_\phi(x,y)
\right]
-
\beta
D_{\text{KL}}
\left(
\pi_\theta(\cdot\mid x)
\|
\pi_{\text{ref}}(\cdot\mid x)
\right)
$$
符号解释：

| 符号       | 含义                        |
| ---------- | --------------------------- |
| `π_θ`      | 正在优化的策略模型          |
| `π_ref`    | 参考模型，通常是 SFT 后模型 |
| `r_φ(x,y)` | 奖励模型给出的 reward       |
| `β`        | KL 惩罚强度                 |
| `D_KL`     | 新策略与参考策略之间的距离  |

KL 项的作用是防止模型为了 reward 走得太远，产生语言退化、reward hacking 或奇怪模板。课程将其解释为：“我想最大化 reward，但又要保持接近 pretrained / SFT model。”【字幕†source】【PDF p.51-p.52†source】

### 7.3 Policy gradient identity

PPO 的基础是 policy gradient。若目标为：
$$
J(\theta)=\mathbb{E}_{z\sim p_\theta}[R(z)]
=
\sum_z p_\theta(z)R(z)
$$
则梯度为：
$$
\nabla_\theta J(\theta)
=
\sum_z R(z)\nabla_\theta p_\theta(z)
$$
利用恒等式：
$$
\nabla_\theta p_\theta(z)
=
p_\theta(z)\nabla_\theta \log p_\theta(z)
$$
得到：
$$
\nabla_\theta \mathbb{E}_{p_\theta}[R(z)]
=
\mathbb{E}_{p_\theta}
[
R(z)\nabla_\theta \log p_\theta(z)
]
$$
直觉解释：这像是“带 reward 权重的 SFT”。高 reward 样本的 log probability 会被提高，低 reward 样本的概率会被压低。【PDF p.53†source】【字幕†source】

### 7.4 从 TRPO 到 PPO

普通 policy gradient 的问题是每次更新都需要新采样，rollout 成本高。TRPO 的思路是复用 rollout，但约束新旧策略不要相差太远：
$$
\max_\theta
\mathbb{E}
\left[
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}
A_t
\right]
$$
约束为：
$$
\mathbb{E}
[
D_{\text{KL}}
(
\pi_{\theta_{\text{old}}}(\cdot\mid s_t)
\|
\pi_\theta(\cdot\mid s_t)
)
]
\le \delta
$$
PPO 用 clipping 近似这个约束。定义概率比：
$$
\rho_t(\theta)=
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}
$$
PPO clipping objective 为：
$$
\mathcal{L}^{\text{CLIP}}(\theta)
=
\mathbb{E}_t
\left[
\min
\left(
\rho_t(\theta)A_t,
\text{clip}(\rho_t(\theta),1-\epsilon,1+\epsilon)A_t
\right)
\right]
$$
直觉上，PPO 不允许新策略一次性把某个动作概率推得太远，因此训练更稳定。但课程也指出，在语言模型 RLHF 中，PPO 非常 finicky，工程复杂度比 SFT 高得多。【PDF p.53†source】【字幕†source】

### 7.5 为什么想摆脱 PPO

课程列出许多避免 on-policy RL 的尝试：

- 给 chosen / rejected 加 `[GOOD]` / `[BAD]` 控制 token；
- 只在 preferred output 上做 SFT；
- 训练 reward model，再筛选高 reward 输出做 SFT；
- 采样大量输出，只拿最优输出训练。【PDF p.54-p.55†source】【字幕†source】

这些方法直觉上合理，但通常不能完全替代 PPO。于是出现了 DPO：希望保留偏好优化效果，同时去掉显式 reward model 和 on-policy rollout。

### 7.6 DPO 的直觉

DPO，即 **Direct Preference Optimization**。

它的核心直觉是：

- 对 winner response `y_w`：提高 log probability；
- 对 loser response `y_l`：降低 log probability；
- 更新强度由当前模型对偏好判断的错误程度决定。

课程将其概括为：

> “positive gradient on good, negative gradient on bad”。【PDF p.55/p.58†source】【字幕†source】

但 DPO 不是简单的“好答案做 SFT，坏答案反向 SFT”，它的权重来自 KL-regularized RLHF 目标的推导。

### 7.7 DPO 推导：从 KL-regularized RLHF 出发

DPO 从以下目标开始：
$$
\max_{\pi}
\mathbb{E}_{x\sim \mathcal{D},y\sim \pi(y\mid x)}
\left[
r(x,y)
\right]
-
\beta
D_{\text{KL}}
[
\pi(y\mid x)
\|
\pi_{\text{ref}}(y\mid x)
]
$$
假设 `π` 不是神经网络参数化策略，而是所有可能策略的集合，即 nonparametric assumption。此时最优策略有闭式解：
$$
\pi_r(y\mid x)
=
\frac{1}{Z(x)}
\pi_{\text{ref}}(y\mid x)
\exp\left(\frac{1}{\beta}r(x,y)\right)
$$
其中 `Z(x)` 是归一化常数。【PDF p.56†source】

对上式取对数并解出 reward：
$$
r(x,y)
=
\beta
\log
\frac{\pi_r(y\mid x)}
{\pi_{\text{ref}}(y\mid x)}
+
\beta \log Z(x)
$$
这说明，如果某个回答相对 reference policy 被当前策略显著提高概率，那么它对应更高 implied reward。【PDF p.56†source】

### 7.8 DPO loss

把 implied reward 代入 pairwise preference loss，可以得到 DPO loss：
$$
\mathcal{L}_{\text{DPO}}(\pi_\theta;\pi_{\text{ref}})
=
-
\mathbb{E}_{(x,y_w,y_l)\sim \mathcal{D}}
\left[
\log
\sigma
\left(
\beta
\log
\frac{\pi_\theta(y_w\mid x)}
{\pi_{\text{ref}}(y_w\mid x)}
-
\beta
\log
\frac{\pi_\theta(y_l\mid x)}
{\pi_{\text{ref}}(y_l\mid x)}
\right)
\right]
$$
这里 `Z(x)` 在 winner 和 loser 的 reward 差中抵消，所以不需要显式训练 reward model。【PDF p.57†source】

DPO 的关键步骤是：

1. 作 nonparametric assumption，使 policy 与 reward 有闭式关系；
2. 用 policy 参数化 implied reward；
3. 用监督式 pairwise loss 优化 preference，从而优化 policy。【PDF p.57†source】

### 7.9 DPO 梯度的学习含义

令 implied reward 为：
$$
\hat{r}_\theta(x,y)
=
\beta
\log
\frac{\pi_\theta(y\mid x)}
{\pi_{\text{ref}}(y\mid x)}
$$
DPO 梯度可直观写成：
$$
\nabla_\theta \mathcal{L}_{\text{DPO}}
\propto
-
\beta
\sigma(\hat{r}_\theta(x,y_l)-\hat{r}_\theta(x,y_w))
\left[
\nabla_\theta \log \pi_\theta(y_w\mid x)
-
\nabla_\theta \log \pi_\theta(y_l\mid x)
\right]
$$
其中：

- `∇_θ log π_θ(y_w | x)`：提高 winner 概率；
- `-∇_θ log π_θ(y_l | x)`：降低 loser 概率；
- `σ(\hat{r}_l-\hat{r}_w)`：如果模型当前把 loser 估得太好，更新权重更大；如果模型已经明显偏好 winner，更新较小。

这解释了课程中 “scaled by prediction error of the implied reward model” 的说法。【PDF p.58†source】【字幕†source】

### 7.10 DPO 变体与经验判断

课程提到，LLaMA 等开放模型流程中使用了 DPO，并可与 expert iteration / rejection sampling 外循环结合。DPO 后还有许多变体，例如：

- **SimPO**：去掉 reference policy 项，改变归一化或偏好 margin；
- **Length-normalized DPO**：按长度归一化，减少 length hacking。【字幕†source】【PDF p.59-p.62†source】

课程提醒，不应把 PPO、DPO 或某个变体视为绝对优越。经验结果高度依赖数据、实现细节、reference policy、超参数和评估方式。

### 7.11 RLHF 的副作用

#### Overoptimization / reward overfitting

课程最后重点提醒，RLHF 最大风险之一是过度优化 reward model。优化过强时，模型可能学会 exploit learned reward，而不是真正变好。表现为 reward model 分数继续上升，但真实人类偏好可能下降。【字幕†source】【PDF p.63-p.65†source】

KL regularizer 的作用就在于限制模型不要偏离 reference policy 太远：
$$
-\beta D_{\text{KL}}(\pi_\theta\|\pi_{\text{ref}})
$$
如果没有这个约束，模型容易产生 reward hacking、语言退化、过度迎合 evaluator 风格等问题。【字幕†source】

#### Mode collapse / entropy loss

RLHF 把模型从“拟合分布”推向“最大化 reward”。如果某类回答 reward 高，模型可能把概率集中到少数模板化输出上，导致多样性下降。这就是 mode collapse 或 entropy loss。【字幕†source】【PDF p.64†source】

#### Calibration loss

课程指出，RLHF 后模型不再默认是良好校准的 probabilistic model。输出概率、置信度和真实正确率之间可能失去原有关系。这对后续 RLVR / reasoning models 很重要，因为推理模型需要探索多种解法，而 entropy 与 exploration 是关键资源。【字幕†source】

**学习提示**

RLHF 能让模型更符合偏好，但也可能让模型变得更模板化、更不校准、更会迎合指标。后训练的难点是优化行为，同时不破坏模型的探索性和可靠性。

------

## 8. Python 代码示例详解

### 8.1 文件整体结构说明

原报告和当前 GitHub 核对结果均显示，Lecture 15 对应的是 `lecture_15.pdf`，不是独立的 `lecture_15.py`。因此本讲的代码解析主要来自 PDF 中展示的代码片段和伪代码，而不是一个可执行 Python lecture 文件。

本讲出现的代码主要服务于两个学习目标：

1. 用 Alpaca 示例说明 SFT 如何把自然语言指令映射为代码输出；
2. 用训练循环伪代码说明 SFT 的优化过程本质上仍是普通语言模型训练。

### 8.2 Alpaca 代码示例：从自然语言指令到 Python 函数

代码片段：

```
def avg_list(nums):
    total = 0
    for n in nums:
        total += n
    return total / len(nums)

print(avg_list([4, 3, 6, 10, 8]))  # Output 6.4
```

#### 函数名称

```
avg_list
```

#### 输入参数

`nums`：一个数字列表，例如 `[4, 3, 6, 10, 8]`。

#### 输出结果

列表元素的平均值。

对于示例：
$$
\frac{4+3+6+10+8}{5}
=
\frac{31}{5}
=
6.2
$$
补充说明：原报告代码注释写作 `Output 6.4`，但按给出的列表 `[4, 3, 6, 10, 8]` 计算结果应为 `6.2`。这里不修改原报告中的代码引用内容，但在学习讲义中提醒读者注意这个数值不一致。该处可能是原 PDF 片段、原报告转录或示例列表与注释之间存在不一致；原报告未提供更进一步来源核验。

#### 关键变量

| 变量                | 含义             |
| ------------------- | ---------------- |
| `nums`              | 输入列表         |
| `total`             | 当前累计和       |
| `n`                 | 遍历到的当前元素 |
| `len(nums)`         | 列表长度         |
| `total / len(nums)` | 平均值           |

#### 实现逻辑

1. 初始化 `total = 0`；
2. 遍历列表中每个数字；
3. 将每个数字累加到 `total`；
4. 用总和除以列表长度；
5. 返回平均值。

#### 对应课程概念

这个例子对应 SFT 中的 instruction-following 行为：用户给出自然语言任务，模型输出结构化代码。它说明 SFT 可以诱导模型把预训练中已有的 Python 和数学知识组织为“可执行回答”。【PDF代码片段†source】

#### 学习重点

不要把这个例子理解为“Alpaca 数据教会模型 Python 平均数算法”。更准确地说，它教模型在看到自然语言编程指令时，应该用 assistant-style 的方式给出代码答案。

### 8.3 SFT 训练循环代码解析

代码片段：

```
from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
```

#### 整体作用

这段代码实现了 SFT 的标准训练流程：从 dataloader 取 batch，前向计算 loss，反向传播，更新参数，更新学习率，并清空梯度。【PDF代码片段†source】

#### 关键函数与变量

| 函数 / 变量             | 输入                    | 输出                | 作用               |
| ----------------------- | ----------------------- | ------------------- | ------------------ |
| `train_dataloader`      | SFT 数据集              | batch               | 提供 mini-batch    |
| `batch.items()`         | batch 字典              | key-value 对        | 遍历输入张量       |
| `.to(device)`           | CPU tensor              | GPU / device tensor | 把数据移到训练设备 |
| `model(**batch)`        | token ids、labels、mask | logits、loss        | 前向传播           |
| `outputs.loss`          | 模型输出                | 标量 loss           | SFT 训练目标       |
| `loss.backward()`       | loss                    | 梯度                | 计算参数梯度       |
| `optimizer.step()`      | 梯度                    | 更新后的参数        | 执行优化           |
| `lr_scheduler.step()`   | 当前 step               | 新学习率            | 调整学习率         |
| `optimizer.zero_grad()` | 已有梯度                | 清零梯度            | 准备下一步         |

### 8.4 代码与课程知识点对应表

| 课程知识点      | 对应代码函数 / 变量 / 片段               | 代码功能                       | 对应理论原理                        | 在课程中的作用                          |
| --------------- | ---------------------------------------- | ------------------------------ | ----------------------------------- | --------------------------------------- |
| SFT             | `outputs.loss`                           | 计算 response token 负对数似然 | 最大似然 / 交叉熵                   | 说明 SFT 本质仍是 next-token prediction |
| Response mask   | `labels` / mask，原报告未给出具体变量    | 控制哪些 token 计入 loss       | masked language modeling style loss | 区分 prompt 与 assistant response       |
| 梯度下降        | `loss.backward()`                        | 计算梯度                       | `∇_θ L_SFT`                         | 把公式变成可训练过程                    |
| 参数更新        | `optimizer.step()`                       | 更新模型参数                   | SGD / AdamW                         | 让模型提高参考答案概率                  |
| 学习率调度      | `lr_scheduler.step()`                    | 改变学习率                     | 训练稳定性控制                      | 防止训练过快或过慢                      |
| Alpaca 代码生成 | `avg_list(nums)`                         | 从指令生成函数                 | instruction-following               | 展示 SFT 数据如何诱导 ChatGPT-like 行为 |
| 工具调用数据    | `tool_calls`、JSON，原报告未给出完整代码 | 生成结构化动作                 | agentic behavior                    | 从聊天模型走向工具使用模型              |

------

## 9. PDF / 幻灯片 / 图表解析与关键公式汇总

### 9.1 主要 PDF 内容总览

| PDF 页面 / 范围 | 展示内容                                                  | 支持的课程结论                                            | 对应代码或字幕内容                                           |
| --------------- | --------------------------------------------------------- | --------------------------------------------------------- | ------------------------------------------------------------ |
| p.2-p.6         | After Pretraining、InstructGPT 流程                       | 预训练后需要 SFT 与 RLHF                                  | 课程开头关于 GPT-3 到 ChatGPT 的转变【字幕†source】          |
| p.9-p.14        | FLAN、Alpaca、OpenAssistant、Nemotron 等 SFT 数据         | SFT 数据从 benchmark-like 走向聊天、工具调用和 agent 行为 | Alpaca 平均数代码、OpenAssistant monopsony 例子【PDF代码片段†source】 |
| p.14-p.18       | chattiness、长度、偏好指标与 benchmark                    | 风格提升不等于能力提升                                    | 长答案和 bullet points 可能提升偏好评价【字幕†source】       |
| p.19-p.21       | tail knowledge 与复杂事实                                 | SFT 灌输模型未知事实可能增加幻觉                          | 引用格式与知识纠缠问题【字幕†source】                        |
| p.22-p.27       | Safety SFT                                                | 降低 violation rate，同时避免 false refusal               | “kill a Python process” 例子【字幕†source】                  |
| p.29-p.30       | Mid-training / two-phase training                         | pretraining 与 post-training 边界模糊                     | instruction data mixing【字幕†source】                       |
| p.32            | SFT 与 RLHF 的概念区别                                    | SFT 是 imitation，RLHF 是 optimization                    | `\hat p(y                                                    |
| p.35-p.48       | Pairwise feedback、标注者、AI feedback、Constitutional AI | RLHF 数据收集复杂且有偏差                                 | helpful / truthful / harmless 标注框架                       |
| p.51-p.53       | KL 正则化 RLHF、reward model、policy gradient、PPO        | PPO 通过 reward + KL 训练 policy                          | PPO 公式推导                                                 |
| p.54-p.62       | DPO 动机、推导、变体                                      | DPO 直接用偏好数据优化 policy                             | implied reward、DPO loss                                     |
| p.63-p.65       | RLHF 副作用与 recap                                       | 过优化、reward hacking、mode collapse、校准下降           | 课程总复习【字幕†source】                                    |

### 9.2 图表阅读提示

原报告没有逐图提供横轴、纵轴、颜色和曲线细节，因此这里不补编具体图像属性。下面只根据原报告中记录的 PDF 页码和课程解释，给出阅读这些图表时应抓住的问题。

#### InstructGPT 流程图

这类图通常要回答：从 base model 到 instruction-following model 的训练流程是什么？

阅读时应关注三个阶段：

1. 收集 demonstration data；
2. 做 SFT；
3. 收集 comparison data，训练 reward model，再用 PPO 优化 policy。【PDF p.6†source】

学习重点：SFT 和 RLHF 不是互斥关系，而是流水线中的不同阶段。

#### SFT 数据演化图 / 示例页

这些页面展示 FLAN、Alpaca、OpenAssistant、Nemotron 等数据。它们要说明的不是某个单一数据集最好，而是 SFT 数据从传统 NLP 任务逐步演化到真实对话、长答案、专家风格和工具调用。【PDF p.9-p.14†source】

学习重点：数据格式决定模型行为。不同 SFT 数据会诱导不同 assistant 风格。

#### Chattiness 与 benchmark 对比图

这类图要说明：模型在偏好评价中变好，不一定在传统 benchmark 上变强。【PDF p.17-p.18†source】

阅读时应注意：

- 偏好指标可能受长度和格式影响；
- benchmark 更关注任务正确性；
- 两者都重要，但不能互相替代。

#### RLHF 过优化图

课程最后关于 reward overfitting / overoptimization 的图表，要说明 reward model 分数持续上升时，真实人类偏好可能反而下降。【PDF p.63-p.65†source】

学习重点：reward model 是代理目标，不是真实目标。过度优化代理目标会导致 reward hacking。

### 9.3 关键公式汇总表

| 公式                                                         | 符号解释                         | 适用场景                             | 直觉解释                 | 代码 / 例子对应           |
| ------------------------------------------------------------ | -------------------------------- | ------------------------------------ | ------------------------ | ------------------------- |
| `\mathcal{L}_{SFT}=-E\sum_t \log π_θ(y_t                     | x,y_{<t})`                       | `x` prompt，`y` response，`π_θ` 模型 | SFT                      | 提高参考答案 token 概率   |
| `\mathcal{L}_{SFT-mask}=-E\sum_t m_t \log π_θ(s_t            | s_{<t})`                         | `m_t` 控制是否计入 loss              | response-only SFT        | 只训练 assistant 回答部分 |
| `\max E[R(x,y)]`                                             | `R` 奖励函数                     | RLHF 抽象目标                        | 生成高 reward 回答       | reward model / PPO        |
| `P(y_w \succ y_l)=σ(r(x,y_w)-r(x,y_l))`                      | winner / loser reward 差         | 奖励模型训练                         | winner 分数应高于 loser  | pairwise preference       |
| `\max E[r_φ]-βD_{KL}(π_θ                                     |                                  | π_ref)`                              | reward + KL 惩罚         | PPO / RLHF                |
| `∇E[R]=E[R∇\log p_θ]`                                        | policy gradient identity         | 强化学习优化                         | 高 reward 样本提高概率   | PPO 基础                  |
| `ρ_t=π_θ(a_t                                                 | s_t)/π_old(a_t                   | s_t)`                                | 新旧策略概率比           | PPO                       |
| `L^{CLIP}=E[\min(ρA,clip(ρ)A)]`                              | `A` advantage                    | PPO                                  | 限制单步更新过大         | PPO 稳定训练              |
| `π_r(y                                                       | x)=1/Z·π_ref(y                   | x)\exp(r/β)`                         | 最优策略闭式解           | DPO 推导                  |
| `\hat r_θ=β\log(π_θ/π_ref)`                                  | policy 隐含 reward               | DPO                                  | 概率提升代表 reward 提升 | DPO loss                  |
| `L_DPO=-E\log σ(β\log π_θ(y_w)/π_ref(y_w)-β\log π_θ(y_l)/π_ref(y_l))` | winner 与 loser 的相对 log ratio | DPO                                  | 提高好答案，降低坏答案   | pairwise preference loss  |

### 9.4 特别重要公式展开：SFT 与 DPO 的差异

SFT 是单样本学习：
$$
-\log \pi_\theta(y_w\mid x)
$$
它只告诉模型：“这个答案应该更可能。”

DPO 是成对学习：
$$
-\log \sigma
\left(
\beta
\log
\frac{\pi_\theta(y_w\mid x)}
{\pi_{\text{ref}}(y_w\mid x)}
-
\beta
\log
\frac{\pi_\theta(y_l\mid x)}
{\pi_{\text{ref}}(y_l\mid x)}
\right)
$$
它告诉模型：“相对于参考模型，你应该更偏向 winner，而不是 loser。”

核心差别：

| 方法 | 学习信号            | 是否需要 loser         | 是否需要 reference model | 是否显式训练 reward model |
| ---- | ------------------- | ---------------------- | ------------------------ | ------------------------- |
| SFT  | 好答案              | 不需要                 | 不需要                   | 不需要                    |
| PPO  | reward              | 可由 reward model 学来 | 通常需要                 | 通常需要                  |
| DPO  | winner / loser 偏好 | 需要                   | 需要                     | 不需要                    |

------

## 10. 概念对比、易混淆点与学习拓展

### 10.1 易混淆概念对比表

| 易混淆概念                              | 区别                                                         | 联系                                    | 常见误解                  | 正确理解                                    |
| --------------------------------------- | ------------------------------------------------------------ | --------------------------------------- | ------------------------- | ------------------------------------------- |
| Pretraining vs SFT                      | Pretraining 学通用文本分布；SFT 学指令回答行为               | SFT 建立在预训练能力上                  | SFT 会创造所有能力        | SFT 更多是抽取和塑造已有能力                |
| SFT vs RLHF                             | SFT 模仿示范答案；RLHF 优化偏好奖励                          | RLHF 通常从 SFT 模型开始                | RLHF 只是更高级 SFT       | RLHF 把模型当 policy 优化                   |
| Helpful vs Truthful                     | Helpful 强调有用；Truthful 强调真实                          | 好答案通常两者都需要                    | 长解释就是 helpful        | 长但错的答案不 truthful                     |
| Harmless vs False refusal               | Harmless 要拒绝有害请求；false refusal 是误拒正常请求        | 安全调优要两者平衡                      | 安全模型越保守越好        | 过度拒答会损害可用性                        |
| Chattiness vs Capability                | Chattiness 是回答风格；capability 是真实能力                 | 偏好评价可能混淆两者                    | 回答更长就是能力更强      | 长度和格式可能只是 style                    |
| Reward model score vs Human preference  | reward score 是学习到的代理目标；human preference 是真实目标 | reward model 试图预测人类偏好           | reward 越高一定越好       | 过度优化 reward 可能变坏                    |
| PPO vs DPO                              | PPO on-policy、复杂；DPO supervised-style、直接偏好优化      | 都试图利用偏好数据                      | DPO 总是优于 PPO          | 效果依赖数据、实现和评估                    |
| DPO vs SFT                              | DPO 同时推高 winner、压低 loser；SFT 只推高参考答案          | DPO 梯度有 SFT 影子                     | DPO 就是反向 SFT          | DPO 来自 KL-regularized RLHF 推导           |
| Base model vs Chat model                | base model 主要续写文本；chat model 会遵循指令               | chat model 通常由 base model 后训练得到 | base model 完全没见过指令 | 现代 base model 可能已混入 instruction 数据 |
| Knowledge injection vs Behavior shaping | 注入事实是知识问题；塑造回答方式是行为问题                   | SFT 会同时影响二者                      | 多放事实数据一定减少幻觉  | 模型未知事实的 SFT 可能增加幻觉             |

### 10.2 与大模型训练实践的关系

#### 数据准备

本讲最核心的实践启示是：后训练数据不是越多越好，而是要精心设计。不同数据会影响模型不同方面：

- FLAN 类数据增强任务泛化；
- Alpaca 类数据诱导 ChatGPT-like 行为；
- OpenAssistant 类数据提升长答案和专家式解释；
- safety 数据塑造拒答边界；
- tool-use 数据训练结构化动作；
- pairwise preference 数据训练偏好判断。

#### 模型训练

SFT 的训练代码简单，但数据 mask、样本配比、学习率、训练步数都会影响结果。RLHF / PPO 的工程复杂度更高，需要处理 rollout、reward model、KL 控制和训练稳定性。DPO 简化了流程，但仍对 reference policy、beta、数据质量和长度偏差敏感。

#### 优化策略

PPO 和 DPO 的共同目标是让模型偏向更好的回答，但它们都不能脱离数据质量讨论。课程反复强调，post-training 的 leverage 很多来自数据，而不只是算法。【字幕†source】

#### 评估方法

本讲提醒我们，不应只看 AlpacaEval 或模型 judge 分数。更长、更有条理的回答可能获得偏好指标提升，但不代表 MMLU、GSM、BBH、TyDiQA、Code-Eval 等能力指标同步提升。【字幕†source】【PDF p.17-p.18†source】

#### 安全部署

安全 SFT 和 RLHF 要平衡 violation rate 与 false refusal rate。一个可部署模型不仅要拒绝诈骗、垃圾邮件、钓鱼等有害请求，也不能把正常编程问题误判为危险请求。【字幕†source】【PDF p.22-p.26†source】

### 10.3 与其他 lecture 的联系

原报告没有提供完整前后 lecture 内容，因此这里只做谨慎连接。

- 与前面 lecture 的关系：本讲建立在预训练、数据、优化和语言模型损失函数基础之上。SFT 损失本质上仍是前面语言模型训练目标的延续。
- 与数据相关 lecture 的关系：本讲进一步说明，数据不仅影响模型“知道什么”，还影响模型“如何回答”。
- 与系统优化 lecture 的关系：SFT 训练循环简单，但大规模后训练仍依赖分布式训练、GPU 资源、优化器和训练稳定性。
- 与后续 reasoning / RLVR 的关系：课程提到 RLHF 后的 entropy loss 和 calibration loss 会影响推理模型，因为推理训练需要探索多种解法。【字幕†source】

### 10.4 面试 / 考试 / 作业可能考察角度

| 问题                                      | 考察点                                            | 回答思路                                       |
| ----------------------------------------- | ------------------------------------------------- | ---------------------------------------------- |
| 为什么预训练模型还需要 SFT？              | base model 与 assistant model 的区别              | 预训练学续写，SFT 学指令响应行为               |
| 写出 SFT loss 并解释 response mask        | SFT 公式                                          | 说明 `x`、`y`、`m_t`，为什么只训练 response    |
| 为什么 SFT 可能增加幻觉？                 | tail knowledge 与格式学习                         | 模型学到引用格式，但未必知道事实               |
| FLAN 和 Alpaca 数据有什么区别？           | SFT 数据演化                                      | FLAN benchmark-like，Alpaca 更 chat-like       |
| 为什么 long answer 会影响偏好评估？       | length bias                                       | 人类和模型 judge 可能偏好长答案                |
| RLHF 为什么需要 pairwise feedback？       | G-V gap                                           | 人类生成难，比较相对容易                       |
| 奖励模型如何从 pairwise preference 学习？ | Bradley-Terry loss                                | winner reward 应高于 loser                     |
| PPO 中 KL 正则有什么作用？                | RLHF 稳定性                                       | 防止偏离 reference policy，减少 reward hacking |
| DPO 为什么不需要显式 reward model？       | implied reward 推导                               | reward 可由 `β log πθ/πref` 表示               |
| DPO 和 PPO 谁更好？                       | 经验判断                                          | 不能绝对化，取决于数据和实现                   |
| RLHF 的副作用有哪些？                     | overoptimization、mode collapse、calibration loss | reward 代理目标可被 exploit                    |
| Safety SFT 的两类错误是什么？             | violation 与 false refusal                        | 既不能放过坏请求，也不能误拒好请求             |

------

## 11. 课程复习总结

### 11.1 本讲核心思想

Lecture 15 的主线是：预训练模型虽然拥有广泛能力，但要成为有用、安全、可控的 assistant，还需要后训练。SFT 用示范答案让模型学会遵循指令；mid-training 让预训练和后训练边界变得模糊；RLHF 用偏好数据和奖励模型优化模型行为；PPO 是经典但复杂的强化学习方法；DPO 则用偏好损失直接优化 policy，避免显式 reward model 和 on-policy rollout。与此同时，后训练会带来 length bias、hallucination、reward hacking、mode collapse 和 calibration loss 等副作用。

### 11.2 必须掌握的概念

- **Post-training**：预训练之后塑造模型行为的训练阶段；
- **SFT**：用 prompt-response 示范数据做监督微调；
- **Response mask**：只对 assistant response token 计算损失；
- **Instruction data**：让模型学习指令遵循的数据；
- **Chattiness**：模型回答更长、更像聊天的风格变化；
- **Tail knowledge**：模型可能不知道的长尾事实；
- **Safety SFT**：用安全样本降低有害输出；
- **False refusal**：模型错误拒绝正常请求；
- **Mid-training**：在预训练后期或第二阶段混入高质量 / 指令数据；
- **RLHF**：利用人类或模型偏好优化语言模型；
- **Pairwise feedback**：比较 winner 和 loser response；
- **Reward model**：把回答映射为 reward 的模型；
- **PPO**：带 KL 控制的策略优化方法；
- **DPO**：直接用偏好数据优化 policy 的方法；
- **Reward hacking**：模型利用奖励模型漏洞而不是真正变好；
- **Mode collapse**：模型输出多样性下降；
- **Calibration loss**：概率和真实正确率之间关系变差。

### 11.3 必须理解的公式

#### SFT loss

$$
\mathcal{L}_{\text{SFT}}(\theta)
=
-\mathbb{E}_{(x,y)}
\sum_{t=1}^{|y|}
\log \pi_\theta(y_t \mid x, y_{<t})
$$

解决的问题：让模型提高参考答案 token 的概率。

#### Masked SFT loss

$$
\mathcal{L}_{\text{SFT-mask}}(\theta)
=
-\mathbb{E}_{(x,y)}
\sum_t m_t \log \pi_\theta(s_t \mid s_{<t})
$$

解决的问题：只训练 assistant response，不训练 prompt。

#### Reward model preference loss

$$
\mathcal{L}_{\text{RM}}(\phi)
=
-\mathbb{E}
\log \sigma(r_\phi(x,y_w)-r_\phi(x,y_l))
$$

解决的问题：把 pairwise preference 转成可优化的 reward model。

#### KL-regularized RLHF objective

$$
\max_{\pi_\theta}
\mathbb{E}[r_\phi(x,y)]
-
\beta D_{\text{KL}}(\pi_\theta\|\pi_{\text{ref}})
$$

解决的问题：既提高 reward，又防止模型偏离参考模型太远。

#### DPO loss

$$
\mathcal{L}_{\text{DPO}}
=
-
\mathbb{E}
\log
\sigma
\left(
\beta
\log
\frac{\pi_\theta(y_w\mid x)}
{\pi_{\text{ref}}(y_w\mid x)}
-
\beta
\log
\frac{\pi_\theta(y_l\mid x)}
{\pi_{\text{ref}}(y_l\mid x)}
\right)
$$

解决的问题：不用显式 reward model，直接利用 winner / loser 偏好优化 policy。

### 11.4 必须看懂的代码

#### Alpaca 平均数代码

学习重点：理解 SFT 如何将自然语言指令映射为代码生成行为，而不是从零教会模型算法。【PDF代码片段†source】

#### SFT 训练循环

学习重点：理解 `outputs.loss`、`loss.backward()`、`optimizer.step()` 如何把 SFT loss 变成参数更新。【PDF代码片段†source】

### 11.5 最重要的易错点

1. **SFT 不是万能知识注入工具**：它更适合抽取和塑造已有能力。
2. **长答案不一定更强**：偏好指标可能奖励 length 和 formatting。
3. **安全不是简单拒答**：还要降低 false refusal。
4. **RLHF 不等于模型真的懂人类价值**：它优化的是偏好数据和 reward model。
5. **DPO 不一定总优于 PPO**：算法效果依赖数据和实现。
6. **Reward model 不是终极真理**：过度优化会导致 reward hacking。
7. **后训练后模型可能不再校准**：概率分布的解释要更谨慎。

### 11.6 一句话总结

**Lecture 15 的核心是：后训练通过 SFT 抽取预训练能力，通过 RLHF / PPO / DPO 优化人类偏好，并通过安全与数据设计控制模型行为；但它也会引入风格偏差、幻觉、奖励过拟合、多样性下降和校准损失等系统性风险。**