# Lecture 9 课程报告：Scaling Laws Basics——用小规模实验预测大模型训练行为

## 0. 资料来源与改写说明

本报告是对你上传的原始中文课程报告 `lec9.md` 的系统性改写，目标是保留原报告的课程主线、公式、课堂例子、PDF 图表说明和引用，同时把它整理成更适合学习、复习和长期保存的中文讲义。原报告的主线已经覆盖 B200 训练场景、data scaling、数据混合、重复数据、critical batch size、Chinchilla/Kaplan 分歧、overtraining 等内容，本次改写在此基础上增强解释层次和学习结构。

本次对话中可直接读取的课程资料包括：原始报告 `lec9.md` 和课件 `lecture_09.pdf`。本次没有直接收到单独的字幕文件，因此原报告中已有的 `【字幕†source】` 会作为“原报告转引的课堂字幕来源”保留；新增核对主要基于你上传的 PDF 以及公开课程页面 / GitHub 仓库。你上传的 PDF 标题为 Lecture 9, “Scaling Laws - Basics”，共 57 页。

我也核对了 Stanford CS336 Spring 2026 官方课程页面：该课程页面标明课程为 “CS336: Language Modeling from Scratch”，Spring 2026；课程 schedule 中第 9 讲是 “Scaling laws [Tatsu]”，课程材料对应 `lecture_09.pdf`。 GitHub 仓库 `stanford-cs336/lectures` 当前文件列表中也包含 `lecture_09.pdf`；仓库 README 说明可执行讲义命名为 `lecture_XX.py`，非可执行讲义命名为 `lecture_XX.pdf`，因此本讲没有可逐段解析的 `lecture_09.py` 代码文件。

本报告中的引用约定如下：

| 引用形式                         | 含义                                                         |
| -------------------------------- | ------------------------------------------------------------ |
| `【字幕†source】`                | 原报告中已有的字幕引用，本次未直接拿到字幕文件，故作为原报告转引保留 |
| `【PDF†lecture_09.pdf, p.X】`    | 来自你上传的 Lecture 9 PDF 第 X 页                           |
| ``                               | 系统文件引用，用于指向上传文件或检索到的文件片段             |
| ``                               | 网页核对引用，用于课程主页或 GitHub 仓库信息                 |
| “延伸理解 / 工程直觉 / 学习提示” | 为了帮助理解而加入的解释，不伪装成课程原文                   |

------

## 1. 学习导读

### 1.1 本讲要解决的核心问题

这一讲要解决的问题非常工程化：**如果你已经有了大量 GPU、分布式训练系统和预训练数据集，下一步到底应该训练什么样的大语言模型？**

这不是一个单纯的“模型越大越好”的问题。你需要同时决定：模型多大、训练多少 token、用什么架构、batch size 多大、学习率如何迁移、是否重复数据、是否过度训练、以及如何避免昂贵的大规模训练失败。Lecture 9 的核心工具就是 **scaling laws**：先在小规模实验上观察规律，再把这些规律外推到大规模训练。PDF 第 4 页把 scaling laws 定义为“simple, predictive rules for model performance”，并对比了旧做法“在大模型上调参”和新做法“在小模型上调参再外推”。【PDF†lecture_09.pdf, p.4】

### 1.2 本讲在 CS336 课程中的位置

Lecture 9 位于 parallelism / systems 课程之后、inference 课程之前。也就是说，前面已经讲过如何让模型在 GPU 和多机环境中跑起来，本讲开始回答：**有了系统能力之后，如何理性地选择训练方案？**

官方 schedule 显示 Lecture 8 是 Parallelism，Lecture 9 是 Scaling laws，Lecture 10 是 Inference，Lecture 11 仍然是 Scaling laws。这说明 scaling laws 在 CS336 中连接了训练系统、模型设计、数据工程和推理部署，是课程中从“能训练”走向“知道该训练什么”的关键环节。

### 1.3 本讲知识结构

| 模块                      | 解决的问题                                               | 核心概念                                              | 与后续内容的关系                                    |
| ------------------------- | -------------------------------------------------------- | ----------------------------------------------------- | --------------------------------------------------- |
| 动机：为什么 scaling 重要 | 大模型训练为什么不能靠盲目试错？                         | 小模型实验、外推、工程成本                            | 为后面所有 scaling 方法建立工程背景                 |
| 历史背景                  | scaling laws 是不是 LLM 时代才有？                       | sample complexity、learning curves、early NLP scaling | 说明 scaling law 与统计学习理论和早期经验曲线的关系 |
| Data scaling              | 数据量增加时 loss 如何下降？                             | power law、log-log linearity、irreducible error       | 后续数据收集、数据过滤、重复数据策略的基础          |
| 估计理论直觉              | 为什么会出现幂律？                                       | 均值估计、非参数估计、维度相关指数                    | 帮助理解 exponent 为什么可能很小                    |
| 数据组成与重复            | 不同数据源、重复数据会如何影响 scaling？                 | data mixture、offset vs slope、effective data         | 连接预训练数据工程和后续数据课程                    |
| 模型工程 scaling          | 架构、优化器、深宽比、batch size、learning rate 怎么选？ | architecture scaling、critical batch size、μP         | 连接超参数迁移和大模型训练稳定性                    |
| Data-model joint scaling  | 固定训练算力下，应增大模型还是增大数据？                 | joint scaling、IsoFLOP、Kaplan、Chinchilla            | 直接影响训练预算分配                                |
| 训练最优 vs 部署最优      | Chinchilla 最优是否就是生产最优？                        | train-optimal、deployment-optimal、overtraining       | 连接推理成本和模型产品化                            |

### 1.4 学习本讲前需要知道什么

学习本讲不需要完整掌握 scaling law 论文细节，但需要以下前置知识：

| 前置知识                   | 本讲用到它的地方                                       |
| -------------------------- | ------------------------------------------------------ |
| 基本概率与统计             | 高斯均值估计、方差、均方误差、样本复杂度               |
| 对数与幂律                 | 理解为什么 power law 在 log-log 图上变成直线           |
| 机器学习泛化误差           | 理解 sample complexity 与 data scaling 的联系          |
| 语言模型 loss / perplexity | 理解 scaling law 的纵轴通常是 test loss 或 perplexity  |
| Transformer 基础           | 理解架构 scaling、层数、宽度、attention head 等超参数  |
| 梯度下降和 batch size      | 理解 critical batch size 与梯度噪声                    |
| GPU / 分布式训练常识       | 理解为什么 batch size、compute budget 和推理成本很重要 |

------

## 2. 第一部分：为什么要认真对待 Scaling？

### 2.1 问题引入：10,000 块 B200，一个月

这一节要解决的问题是：**为什么大模型训练需要 scaling laws，而不是直接照抄现有模型配置或在大模型上反复试错？**

老师用一个很具体的工程场景开场：假设朋友给你 10,000 块 B200 GPU 用一个月，让你训练一个很好的开源语言模型。你已经有基础设施团队和分布式训练框架，也已经准备好预训练数据集，现在真正的问题是：**要跑一个大模型，但到底跑哪一个？** PDF 第 2 页也用这个场景说明，本讲所在的位置正是 “Run a big model (but which one??)”。【PDF†lecture_09.pdf, p.2】

这个例子背后的意思是：当训练成本巨大时，错误的模型选择不只是“效果差一点”，而可能意味着数百万美元训练预算被浪费。模型设计空间又非常大：深还是宽？多少 attention heads？用什么非线性？是否用某个已有 LLM 的超参数？PDF 第 3 页明确把这些选择称为 “Scaling isn’t easy”。【PDF†lecture_09.pdf, p.3】

### 2.2 概念定义：什么是 scaling laws？

直觉上，scaling law 就是：**用小规模实验画出趋势线，然后预测大规模训练会发生什么。**

更正式地说，本讲中的 scaling laws 指一类简单、可预测的经验函数关系，它们把模型性能与资源变量联系起来。这里的资源变量可以是：

- 训练数据量；
- 模型参数量；
- 训练计算量；
- batch size；
- 模型深度或宽度；
- 训练 token 数。

最常见的形式是幂律：
$$
\mathrm{Loss}(x) \approx A x^{-\alpha} + C
$$
其中，$x$ 可以是数据量、参数量或计算量；$A$ 是比例常数；$\alpha$ 是 scaling exponent；$C$ 是不可约误差或极限损失。

当 $C$ 可以忽略时，取对数得到：
$$
\log \mathrm{Loss}(x) \approx \log A - \alpha \log x
$$
这意味着在 log-log 图上，loss 与资源规模之间近似是一条直线。PDF 第 15 页展示了 Kaplan 2020 语言模型 data scaling 图：dataset size 与 test loss 在 log-log 图上近似线性，因此被称为 “scale-free” 或 “power law”。【PDF†lecture_09.pdf, p.15】

### 2.3 方法与原理：旧做法 vs 新做法

旧做法是：直接在大模型上调超参数。这个做法昂贵、低效，而且一旦失败代价很高。

新做法是：

1. 训练一批小模型；
2. 在小模型上观察 loss 随数据、参数、compute 或超参数变化的规律；
3. 拟合 scaling law；
4. 外推到大模型；
5. 选择最有希望的大规模训练配置。

PDF 第 42 页把这个流程总结为三步：train a few smaller models，establish a scaling law，select optimal hyperparameter based on the prediction。【PDF†lecture_09.pdf, p.42】

**工程直觉：** scaling laws 的价值不在于它能保证预测永远准确，而在于它让昂贵决策变得可实验、可拟合、可比较。对于大模型训练团队来说，这相当于把“拍脑袋选配置”变成“用低成本实验支持高成本决策”。

### 2.4 课堂例子与解释

课堂开场的 B200 场景很有代表性，因为它把 scaling laws 放在真实的大模型训练链条中：

| 阶段           | 已经解决的问题         | 本讲关注的问题                       |
| -------------- | ---------------------- | ------------------------------------ |
| Infra          | 如何让大规模训练跑起来 | 不再是本讲重点                       |
| Data           | 如何构建预训练数据     | 本讲会讨论数据量、数据组成、重复数据 |
| Model training | 具体训练什么模型       | 本讲核心问题                         |

这说明 scaling laws 不是孤立数学工具，而是大模型训练工程中的决策层工具：系统团队解决“怎么跑”，数据团队解决“用什么数据”，scaling laws 帮助解决“跑什么、跑多大、跑多久”。

### 2.5 代码 / PDF 对应说明

本讲对应 GitHub 资料是 `lecture_09.pdf`，不是可执行 Python 文件。GitHub 仓库当前列出了 `lecture_09.pdf`，README 也说明非可执行讲义命名为 `lecture_XX.pdf`。 因此本节没有 `【Python代码†source】` 可引用；所有课程内容主要来自 PDF 和原报告中的字幕转引。

### 2.6 学习提示

本节最重要的结论是：**scaling laws 的出发点不是理论优雅，而是工程必要性。** 当大模型训练成本极高时，我们需要用小规模实验预测大规模行为。

容易误解的地方是：不要把 scaling law 理解为“自然界定律”。它主要是经验拟合工具，只有在合适的规模区间、合适的指标和合理的实验设置下才可靠。

一句话总结：**Scaling laws 是把大模型训练中的高风险决策，尽量转化为低成本可预测实验的方法。**

------

## 3. 第二部分：历史背景——从 Sample Complexity 到 Neural Scaling

### 3.1 问题引入

这一节要解决的问题是：**scaling laws 是不是大语言模型时代才突然出现的新概念？**

答案是否定的。老师把 scaling laws 放回更长的机器学习历史中：理论家很早就在研究“训练数据量增加时，误差如何下降”；早期工程论文也很早就在用小数据学习曲线预测大数据性能。PDF 第 5 页把本部分标题列为 “Scaling laws, history and background”，并强调 data scaling 可以看成 empirical sample complexities。【PDF†lecture_09.pdf, p.5】

### 3.2 Sample complexity：理论上界与经验 loss

在统计学习理论中，我们常问：给定训练样本数 $m$，学习到的假设 $\hat h$ 的误差会比最优假设差多少？PDF 第 6 页给出了有限假设类中的典型泛化界：
$$
\epsilon(\hat h)
\le
\min_{h\in H}\epsilon(h)
+
2\sqrt{
\frac{1}{m}\log\frac{2k}{\delta}
}
$$
符号含义如下：

| 符号                       | 含义                 |
| -------------------------- | -------------------- |
| $H$                        | 假设类               |
| $k$                        | 假设类大小           |
| $m$                        | 训练样本数           |
| $\delta$                   | 失败概率或置信参数   |
| $\epsilon(\hat h)$         | 学到的假设的误差     |
| $\min_{h\in H}\epsilon(h)$ | 假设类中最优可能误差 |

这个公式说明，样本数 $m$ 越大，上界中的复杂度项越小。它不是现代 LLM scaling law 的实际 loss 曲线，因为它是上界，不是实际训练出来的 test loss；但它给出一个重要直觉：**误差可以随样本量按某种函数形式系统下降。** PDF 第 6 页也明确提醒，这类结果是 upper bounds, not actual realized loss values。【PDF†lecture_09.pdf, p.6】

### 3.3 1993 年 learning curves：早期数据 scaling 思想

PDF 第 7 页展示了 1993 年的 “Learning Curves: Asymptotic Values and Rate of Convergence”。这类工作已经在做一件和现代 scaling laws 很相似的事情：当完整训练很昂贵时，先在小训练集上训练，拟合 learning curve，再预测更大训练集上的 test error。【PDF†lecture_09.pdf, p.7】

**直觉解释：** 这和今天训练 LLM 的逻辑几乎一样。区别只是横轴从几千、几万样本变成了上万亿 token，纵轴从分类错误率变成了语言模型 loss，成本从普通训练变成了大规模 GPU 集群训练。

### 3.4 NLP 中的数据规模研究

Lecture 9 还回顾了 NLP 中更接近现代语言模型的早期 scaling 结果。

PDF 第 8 页展示 Banko & Brill 2001 的结果：随着训练语料增加，不同方法在 confusion set disambiguation 上的性能都出现规律性提升，并提出应重新考虑“花时间改算法”与“花资源收集语料”之间的投入取舍。【PDF†lecture_09.pdf, p.8】

PDF 第 9 页展示 Kolachina et al. 2012 对机器翻译数据规模与下游性能的曲线拟合，比较指数、幂律等函数族，并展示 power law relation between data and downstream performance。【PDF†lecture_09.pdf, p.9】

PDF 第 10-11 页介绍 Hestness et al. 2017，这是较早的大规模 neural scaling 工作之一，覆盖机器翻译、语言模型、语音识别等任务，并已经讨论了 predictable scaling、emergence、scaling by compute、speed = accuracy 等后来在大模型时代变得非常重要的思想。【PDF†lecture_09.pdf, p.10-p.11】

### 3.5 学习提示

本节要记住两点。

第一，scaling laws 不是突然出现的魔法。它和 sample complexity、learning curves、数据规模研究之间有连续历史。

第二，现代 LLM scaling laws 的新意不只是“误差随数据下降”，而是它在大规模工程决策中变成了核心工具：数据、模型、compute、超参数、部署成本都可以被放进 scaling 分析框架。

------

## 4. 第三部分：Data Scaling Laws——数据量如何决定性能？

### 4.1 问题引入

这一节要解决的问题是：**如果其他条件固定，只增加训练数据量，语言模型的 test loss 会如何变化？**

这也是最基础的 scaling law。数据越多，模型通常越好；但课程关心的不是这句常识，而是更精确的问题：

- 这种提升是否可预测？
- 曲线是什么形状？
- 为什么常常是幂律？
- 什么时候幂律会失效？
- 这对大模型数据收集有什么意义？

### 4.2 概念定义：data scaling law

PDF 第 14 页给出定义：data scaling law 是一个把 dataset size $n$ 映射到 error 的简单公式。该页还展示了 Hestness 风格的三段曲线：small data region、power-law region、irreducible error region。【PDF†lecture_09.pdf, p.14】

可以把这三段理解为：

| 区域         | 现象                             | 为什么重要                       |
| ------------ | -------------------------------- | -------------------------------- |
| 小数据区     | 数据太少，模型表现接近随机或很差 | 这里不一定符合稳定幂律           |
| 幂律区       | 数据增加带来规律性 loss 下降     | scaling law 最有用的区间         |
| 不可约误差区 | 继续加数据收益变小，曲线变平     | 需要考虑噪声、任务极限或模型限制 |

**常见误区：** 看到 log-log 图上有一段直线，不代表所有 scale 上都能无限外推。幂律通常只在中间区间近似成立。

### 4.3 公式与推导：log-log 直线为什么等价于幂律？

关键公式是：
$$
\mathrm{Error}(n) = A n^{-\alpha} + C
$$
其中：

| 符号     | 含义                                         |
| -------- | -------------------------------------------- |
| $n$      | 训练数据量，例如 token 数                    |
| $A$      | 比例常数，影响曲线整体高度                   |
| $\alpha$ | scaling exponent，表示数据增加时误差下降速度 |
| $C$      | 不可约误差或渐近损失                         |

当 $C$ 很小或暂时忽略时：
$$
\mathrm{Error}(n) \approx A n^{-\alpha}
$$
两边取对数：
$$
\log \mathrm{Error}(n)
\approx
\log A - \alpha \log n
$$
这就是一条直线：

- 横轴：$\log n$；
- 纵轴：$\log \mathrm{Error}$；
- 斜率：$-\alpha$；
- 截距：$\log A$。

因此，PDF 第 15 页中 “loss and dataset size is linear on a log-log plot” 的意思就是：loss 与数据量之间近似满足 power law。【PDF†lecture_09.pdf, p.15】

### 4.4 均值估计例子：最简单的 scaling law

为了说明幂律并不神秘，课程用高斯均值估计作为 toy example。设：
$$
x_1,\ldots,x_n \sim \mathcal N(\mu,\sigma^2)
$$
任务是估计均值：
$$
\hat\mu = \frac{1}{n}\sum_{i=1}^n x_i
$$
标准结果是：
$$
\mathbb E[(\hat\mu-\mu)^2] = \frac{\sigma^2}{n}
$$
取对数：
$$
\log \mathrm{Error}
=
-\log n + 2\log \sigma
$$
这就是斜率为 $-1$ 的 log-log scaling law。PDF 第 17 页也明确写道：more generally, any polynomial rate $1/n^\alpha$ is a scaling law。【PDF†lecture_09.pdf, p.17】

**直觉解释：** 样本越多，平均值越稳定。误差不是线性下降，而是以 $1/n$ 的形式下降。log-log 图把这种多项式下降变成直线。

### 4.5 为什么神经网络 exponent 往往很小？

经典均值估计和线性回归常常有接近 $1/n$ 的误差下降，也就是 log-log 斜率接近 $-1$。但 PDF 第 18 页指出，机器翻译、语音、语言建模中的 neural scaling exponent 往往远小于 1，例如 $-0.1$ 或 $-0.3$ 量级。【PDF†lecture_09.pdf, p.18】

课程用非参数估计给出直觉。假设输入 $x_i$ 均匀分布在二维单位方块中：
$$
y_i = f(x_i) + \mathcal N(0,1)
$$
任务是估计任意平滑函数 $f(x)$。一种朴素方法是把空间切成小格子，每个格子内平均样本。PDF 第 19 页给出直观结论：在 $d$ 维中，误差可以粗略写成：
$$
\mathrm{Error}(n) \approx n^{-1/d}
$$
于是：
$$
\log \mathrm{Error}
=
-\frac{1}{d}\log n + C
$$
这说明维度越高，指数越小，误差下降越慢。【PDF†lecture_09.pdf, p.19】

**延伸理解：** 神经网络虽然是参数模型，但它能近似非常复杂的函数，因此在 scaling 行为上可能更像灵活的非参数估计器。这可以帮助理解为什么 LLM loss 随数据增加下降得很稳定，但下降速度并不像简单均值估计那样快。

### 4.6 Intrinsic dimensionality 的解释与限制

PDF 第 20 页提到一种观点：scaling exponent $\alpha$ 可能和数据的 intrinsic dimensionality 有关。直觉是，如果数据实际上位于某个低维流形上，那么学习难度可能由这个内在维度控制，而不是原始输入维度控制。【PDF†lecture_09.pdf, p.20】

但课程也提醒：intrinsic dimension 的估计本身并不可靠，因此这不是严密定理，只是帮助理解 exponent 的一个心智模型。

### 4.7 学习提示

本节最重要的结论是：**data scaling law 的核心不是“数据越多越好”，而是 loss 随数据量增加常常呈现可拟合、可外推的幂律关系。**

复习时抓住三件事：

1. power law 在 log-log 图上是直线；
2. scaling exponent 决定误差下降速度；
3. 幂律只在一定区间成立，接近不可约误差时会变平。

------

## 5. 第四部分：数据组成、重复与过滤——Data Scaling 的工程扩展

### 5.1 问题引入

这一节要解决的问题是：**真实训练中，我们不只关心“有多少数据”，还要关心“是什么数据、是否重复、怎么过滤”。**

PDF 第 21 页把问题从 dataset size 扩展到 dataset composition：如何用小模型选择最优数据混合？是否重复数据？如何在数据质量与重复率之间平衡？【PDF†lecture_09.pdf, p.21】

这对大模型训练非常关键，因为高质量数据不是无限的。当 compute 变得越来越多时，模型可能不得不重复使用有限数据，或者降低过滤强度以获得更多 token。

### 5.2 数据组成：通常改变 offset，而不是 slope

PDF 第 22 页给出一个重要结论：data composition affects the offset, not the slope。【PDF†lecture_09.pdf, p.22】

这句话的意思是：不同数据源或数据混合比例可能让 loss 曲线整体上移或下移，但斜率可能近似不变。用公式表示，如果两个数据混合方案分别满足：
$$
L_1(n) \approx A_1 n^{-\alpha} + C
$$
它们有相似的 $\alpha$，但 $A_1$ 与 $A_2$ 不同，那么 log-log 图上就是两条大致平行的线。

**工程意义：** 如果某种数据混合在小模型上有更低截距，而斜率又相似，那么它在大模型上也可能继续更好。这解释了为什么很多团队会用小模型快速 sweep 数据配比，然后把表现最好的配比用于更大模型。

### 5.3 数据混合例子：News 与 Wikipedia

原报告转述了课堂中的 News 与 Wikipedia 数据混合例子：可以训练一批小模型，尝试不同数据源比例，观察 loss 随数据量或模型规模的变化，再预测大规模训练时哪种 mixture 最优。【字幕†source，原报告转引】

这个例子想说明：data mixture 不是一个只能靠直觉定的选择。它可以被实验化：把不同 mixture 当作不同候选方案，比较它们的 scaling 曲线。

但 PDF 第 23 页也提醒，实践中的 data mixture selection via scaling 很难。一个自然想法是建立数据 scaling laws；另一个经验做法是直接选小规模上表现最好的数据集。【PDF†lecture_09.pdf, p.23】

### 5.4 重复数据：effective data 与递减收益

当高质量数据有限时，训练往往会重复数据。问题是：重复一遍数据是否等价于新增一遍 fresh data？

答案是否定的。PDF 第 24 页展示了 data repetition 下的 scaling law，并给出 effective data 公式：【PDF†lecture_09.pdf, p.24】
$$
D'
=
U_D
+
U_D R_D^*
\left(
1-e^{-R_D/R_D^*}
\right)
$$
符号含义如下：

| 符号    | 含义                         |
| ------- | ---------------------------- |
| $D'$    | effective data，有效数据量   |
| $U_D$   | unique tokens，唯一 token 数 |
| $R_D$   | repetition，重复次数         |
| $R_D^*$ | 控制重复收益饱和速度的常数   |

直觉上，第一次看到数据价值最高；第二次、第三次仍然有价值；但重复越来越多后，边际收益会逐渐下降。原报告还转述课堂讨论：在某些标准训练 recipe 下，重复到约 4 个 epoch 内可能损失不大，但继续重复会明显偏离 fresh-data 假设下的 scaling 曲线。【字幕†source，原报告转引】

**常见误区：** 不要把“训练 token 数”简单等同于“有效数据量”。如果 token 来自大量重复数据，它对 loss 的贡献会低于同等数量的新数据。

### 5.5 Compute unbounded setting：scaling law 可能 break

PDF 第 25 页给出两个提醒：

1. scaling laws can break if blindly applied；
2. scaling laws are lower bounds, so you can potentially do better.【PDF†lecture_09.pdf, p.25】

这意味着 scaling law 不是机械公式。比如在无限 compute、有限数据的场景中，只靠重复训练不一定持续改善；你可能需要正则化、ensemble、改变训练目标或改进数据选择。

### 5.6 数据过滤必须随 scale 自适应

PDF 第 26 页给出核心结论：given repeated data is less valuable, data selection should be adaptive to scale。【PDF†lecture_09.pdf, p.26】

直觉是：

- 小 compute 时，强过滤高质量数据可能最好，因为你不会重复太多；
- 大 compute 时，如果仍只用少量高质量数据，就会大量重复；
- 因此在更大规模训练中，可能需要放宽过滤，引入更多中等质量但不重复的数据。

**工程直觉：** 数据质量不是一个固定阈值，而是与 compute budget、重复率、模型大小共同决定的动态变量。

### 5.7 学习提示

本节最重要的结论是：**数据 scaling 不只是“更多数据”，而是“更多有效、多样、低重复、适合当前规模的数据”。**

复习时要抓住三个关键词：

- **mixture**：数据组成可能主要影响截距；
- **repetition**：重复数据有价值，但边际收益递减；
- **adaptive filtering**：数据过滤策略应随训练规模变化。

------

## 6. 第五部分：Scaling Laws for Model Engineering

### 6.1 问题引入

这一节要解决的问题是：**如何用 scaling laws 设计大语言模型的架构和超参数？**

PDF 第 28 页把本节动机写得很直接：如何高效设计 huge LMs？包括 LSTMs vs Transformers、Adam vs SGD，以及有限资源下是训练更久、更大模型、收集更多数据还是获取更多 GPU。【PDF†lecture_09.pdf, p.28】

本节的思想是：如果某些超参数选择在小规模上呈现稳定趋势，就可以在大规模训练前预测其效果。

### 6.2 架构选择：Transformers vs LSTMs

PDF 第 30 页提出问题：Are transformers better than LSTMs? 暴力方法是花费数千万美元训练一个 LSTM 版 GPT-3；scaling law 方法则是在小规模上比较 LSTM 与 Transformer 的 loss-parameter 曲线。【PDF†lecture_09.pdf, p.30】

图中横轴是 non-embedding parameters，纵轴是 test loss。Transformer 曲线整体低于 LSTM 曲线，说明在相似参数规模下 Transformer loss 更低；如果这种趋势能外推，就支持在大规模训练中选择 Transformer。

**学习重点：** 架构比较不能只看一个小模型单点。更重要的是看曲线斜率和截距：

| 曲线特征           | 含义                 |
| ------------------ | -------------------- |
| 截距更低           | 同等规模下表现更好   |
| 斜率更陡           | 随规模增加改善更快   |
| 小规模好但斜率差   | 大规模可能反而被超越 |
| 小规模略差但斜率好 | 大规模可能值得押注   |

### 6.3 多架构比较：趋势比单点更重要

PDF 第 31 页展示 Tay et al. 2022 的 cross-architecture scaling。它提醒我们：现代架构论文中的“新方法是否有效”，不应只看某个规模上的指标，而应看不同 FLOPs 下的 scaling 趋势。【PDF†lecture_09.pdf, p.31】

**延伸理解：** 这也是为什么一些 efficient attention 或新架构虽然在小规模上有吸引力，但未必能成为主流大模型架构。大规模训练关心的是“继续放大后是否仍然好”。

### 6.4 优化器选择：Adam vs SGD

PDF 第 32 页展示 Hestness 2017 中 Adam 与 SGD 的 scaling 曲线，并注明这是 pre-transformer 时代的 recurrent highway nets。【PDF†lecture_09.pdf, p.32】

课程强调一个有趣现象：优化器从 SGD 换到 Adam 是很大的训练干预，但 scaling 曲线的斜率可能相似，差异更多体现在截距。

这说明很多工程改动可能不会改变“随数据或规模改善的指数”，但会让整条曲线整体下降。对工程来说，这仍然很重要：同样规模下 loss 更低，就意味着同样预算能训练出更好模型。

### 6.5 深度、宽度与 Transformer 形状超参数

PDF 第 33 页讨论层数：1 层与 2 层差异很大，但在 $10^7$ 参数以下，继续增加层数的收益递减。【PDF†lecture_09.pdf, p.33】

PDF 第 34 页进一步讨论 feed-forward ratio、aspect ratio、attention head dimension 等 Transformer 形状超参数。图中结论是：在固定 non-embedding 参数量时，许多形状选择只造成温和 loss 差异；换言之，存在一个较宽的可行区域，而不是只有唯一精确最优点。【PDF†lecture_09.pdf, p.34】

**学习提示：** 深度和宽度不是孤立变量。模型规模变大时，最优层数、隐藏维度、head 维度可能一起变化。Scaling law 的作用是帮助我们找到“随规模相对稳定”的形状规律。

### 6.6 参数不是都等价：embedding、softmax、MoE

PDF 第 35 页提醒：not all parameters are made equal。embedding 参数和 non-embedding 参数在 scaling 图中行为不同。【PDF†lecture_09.pdf, p.35】

这点非常重要，因为后面的 Kaplan vs Chinchilla 分歧就部分来自参数计数方式。若横轴用 total parameters、non-embedding parameters、active parameters，可能会得到不同的 scaling 关系。

PDF 第 36 页还提到 Mixture-of-Experts 里的 “value of parameters”：MoE 中 total parameters 与 active parameters 解耦，一个 token 只激活部分专家，因此参数规模的含义比 dense Transformer 更复杂。【PDF†lecture_09.pdf, p.36】

**常见误区：** 不要以为“参数量”是一个无歧义变量。Scaling law 的横轴怎么定义，会直接影响拟合结果和工程结论。

### 6.7 Critical batch size：batch size 的 scaling law

batch size 影响两个目标：

1. 优化效率；
2. 系统并行效率。

小 batch 噪声大，但每步便宜；大 batch 梯度更稳定，也利于数据并行，但过大后会有递减收益。PDF 第 37 页把 critical batch size 定义为进入 diminishing returns 前的最小样本数。【PDF†lecture_09.pdf, p.37】

更精确定义见 PDF 第 38 页。固定一个 target loss，对不同 batch size 训练，记录达到目标 loss 所需：

- step 数 $S$；
- examples 数 $E$。

拟合关系：
$$
\frac{S}{S_{\min}} - 1
=
\left(
\frac{E}{E_{\min}} - 1
\right)^{-1}
$$
然后定义：
$$
B_{\mathrm{crit}}
=
\frac{E_{\min}}{S_{\min}}
$$
其中：

| 符号                | 含义                                         |
| ------------------- | -------------------------------------------- |
| $S$                 | 达到目标 loss 所需 step 数                   |
| $E$                 | 达到目标 loss 所需 examples 数               |
| $S_{\min}$          | 拟合得到的最少 step 极限                     |
| $E_{\min}$          | 拟合得到的最少 examples 极限                 |
| $B_{\mathrm{crit}}$ | 平衡 step 与 examples 的 critical batch size |

PDF 第 39 页进一步展示：目标 loss 越小，critical batch size 越大。【PDF†lecture_09.pdf, p.39】

**直觉解释：** 训练早期，梯度信号强，batch 不需要太大；训练后期，模型接近更低 loss，梯度噪声更影响优化方向，因此更大的 batch 更有价值。

### 6.8 Learning rate、μP 与 scale-aware LR

除了 batch size，learning rate 也是随规模迁移时最危险的超参数之一。PDF 第 40 页展示 μP 和 scale-aware learning rate 的思想：如果天真放大模型，最优 learning rate 会随规模移动；因此需要与 scale 匹配的初始化和学习率缩放。【PDF†lecture_09.pdf, p.40】

可以把这里理解为两种策略：

| 策略                    | 做法                                                   | 优点             | 风险                 |
| ----------------------- | ------------------------------------------------------ | ---------------- | -------------------- |
| 直接外推 LR             | 在小模型上测最优 LR 如何随 scale 变化，再预测大模型 LR | 思路直接         | 需要多规模 sweep     |
| μP / scale-aware 参数化 | 重新设计初始化与 LR 缩放，使不同规模最优 LR 更稳定     | 有利于超参数迁移 | 实现和理论理解更复杂 |

**工程直觉：** 大模型训练失败常常不是因为架构错，而是因为学习率、warmup、batch size 等 scale-sensitive 超参数没有正确迁移。

### 6.9 上游 loss 与下游性能可能不一致

PDF 第 41 页提醒：downstream scaling can often be much less predictable。图中显示 log-perplexity 与 SuperGLUE accuracy 的排序并不完全一致。【PDF†lecture_09.pdf, p.41】

这说明 scaling laws 最稳定的对象通常是预训练 loss 或 perplexity，而不是所有下游 benchmark。预训练 loss 好通常有帮助，但不能保证每个下游任务都最好。

### 6.10 学习提示

本节最重要的结论是：**模型工程中的很多选择可以在小规模上通过 scaling 趋势提前评估。**

复习时抓住：

- 架构比较看曲线，不只看单点；
- 优化器和数据组成常改变截距；
- 参数计数方式会影响结论；
- batch size 和 learning rate 必须随 scale 谨慎迁移；
- 下游指标比预训练 loss 更 noisy。

------

## 7. 第六部分：Data-Model Joint Scaling 与 Chinchilla

### 7.1 问题引入：更多数据还是更大模型？

这一节要解决的问题是：**给定固定训练 compute，应该训练更大的模型，还是训练较小模型但喂更多数据？**

PDF 第 43 页用一句话引入：Do we need more data or bigger models? 同页还指出，小模型上喂太多数据会浪费，因为 loss 很快变平；反过来，大模型如果数据太少，也会 under-trained。【PDF†lecture_09.pdf, p.43】

这就是 joint data-model scaling 的核心：loss 同时由数据量和模型规模决定。

### 7.2 联合 scaling 公式

Rosenfeld et al. 2020 的简化形式是：
$$
\mathrm{Error}
=
n^{-\alpha}
+
m^{-\beta}
+
C
$$
其中：

| 符号     | 含义                      |
| -------- | ------------------------- |
| $n$      | 数据量                    |
| $m$      | 模型规模                  |
| $\alpha$ | 数据方向 scaling exponent |
| $\beta$  | 模型方向 scaling exponent |
| $C$      | 不可约误差或渐近项        |

Kaplan 2020 的形式在 PDF 第 43 页写作：
$$
\mathrm{Error}
=
\left[
m^{-\alpha}
+
n^{-1}
\right]^\beta
$$
这两个公式表达同一个基本思想：**模型不够大时，模型容量限制 loss；数据不够多时，数据限制 loss；真正的最优需要同时平衡二者。**【PDF†lecture_09.pdf, p.43】

### 7.3 固定 compute 下的资源分配

训练 FLOPs 粗略上与模型参数量和训练 token 数的乘积相关。因此固定 compute 时，不能同时无限增加模型和数据。我们需要在约束下优化：
$$
\min_{m,n} \; L(m,n)
\quad
\text{s.t.}
\quad
\mathrm{Compute}(m,n) \le C
$$
PDF 第 44 页展示 Rosenfeld 的 joint scaling 可以从小数据、小模型区域拟合 exponent，再预测更大区域的误差。【PDF†lecture_09.pdf, p.44】

**工程直觉：** 这一步把 scaling law 从“预测 loss”推进到“做预算分配”。它不只是告诉你大模型会更好，而是告诉你在同样钱数下，怎样的模型-数据组合更划算。

### 7.4 Kaplan 结论与 Chinchilla 反转

PDF 第 45 页写出 Kaplan 的 compute-optimal 结论：
$$
N_{\mathrm{opt}} = C^{0.73}
$$
其中 $N_{\mathrm{opt}}$ 是最优参数规模，$D_{\mathrm{opt}}$ 是最优数据量，$C$ 是训练 compute。这个结论意味着：compute 增加时，参数量增长快于数据量，tokens per parameter 会下降。【PDF†lecture_09.pdf, p.45】

Chinchilla 则提出相反方向的修正：许多已有模型过大、训练 token 不足；更好的 compute-optimal 策略是让参数量和数据量大致一起增长。PDF 第 46 页列出 Chinchilla 三种方法的 exponent，其中方法 1 和方法 2 都接近：
$$
N_{\mathrm{opt}}\propto C^{0.5}
$$
这就是后来常说的约 20 tokens per parameter 经验规则的来源之一。【PDF†lecture_09.pdf, p.46】

### 7.5 Chinchilla 的三种方法

#### 方法 1：Minimum over training curves

PDF 第 47 页展示方法 1：训练不同大小模型，记录所有 training curves，然后取每个 FLOP 水平下 loss 最低的点。这个 lower envelope 形成一条 power law。【PDF†lecture_09.pdf, p.47】

直觉上，这相当于问：如果我有某个 FLOP budget，在已经跑过的所有模型中，哪个模型达到的 loss 最低？

#### 方法 2：IsoFLOP profiles

PDF 第 48 页展示方法 2：固定一系列 FLOP budget，在每个预算下改变参数量和训练 token 数，让总 FLOPs 保持不变。每个预算会得到一条 terminal loss 对参数量的 U 形或凸形曲线；曲线最低点就是该预算下的最优模型大小。【PDF†lecture_09.pdf, p.48】

这是本讲非常重要的方法。它的优点是实验设计清晰：固定成本，扫自由度，找最低点。

#### 方法 3：Joint fits

PDF 第 49 页展示方法 3：在模型大小和数据大小网格上跑一批模型，用 least squares 拟合 joint scaling law。【PDF†lecture_09.pdf, p.49】

这种方法更依赖函数形式假设。如果函数形式或数据范围有偏，拟合结果可能偏离真实最优。

### 7.6 Kaplan 与 Chinchilla 为什么差这么多？

PDF 第 50 页明确提出问题：为什么两者都拟合 joint scaling laws，却有这么大差异？【PDF†lecture_09.pdf, p.50】

PDF 第 51 页给出第一类解释：

- Kaplan removed last layer parameter from the count；
- very small compute budgets 下 warmup 太高；
- 如果 batch / LR 调得合适，decay 本身可能不是关键。【PDF†lecture_09.pdf, p.51】

PDF 第 52 页给出第二类解释：non-embedding vs total parameter choice 加上 small nonlinearities 足以造成 scaling coefficient 差异。【PDF†lecture_09.pdf, p.52】

**学习重点：** Kaplan vs Chinchilla 的故事不是“一个对一个错”这么简单。它说明 scaling law 对实验细节非常敏感：参数怎么数、warmup 怎么设、batch size 和 learning rate 是否调好、小模型是否真的处在可外推区间，都会改变结论。

### 7.7 Chinchilla 方法 3 的后续修正

PDF 第 53 页提到，Chinchilla 原论文中的 method 3 可能存在问题。后续作者做 data forensics，恢复原始数据并重新拟合，得到更接近方法 1 和方法 2 的结果。【PDF†lecture_09.pdf, p.53】

这个补充非常适合用来理解 scaling laws 的真实性边界：即使是高影响力论文中的 scaling fit，也可能因为拟合细节、数据恢复或函数形式选择而被修正。

### 7.8 Train-optimal 不等于 deployment-optimal

PDF 第 54 页给出本讲最后一个重要警告：Chinchilla aims to tell you what gives the best model for fixed training compute，但真实部署中大部分 compute 是 inference，所以我们可能应该 “over-train”。【PDF†lecture_09.pdf, p.54】

PDF 第 54 页列出 tokens per parameter 的例子：

| 模型        | tokens / parameter |
| ----------- | ------------------ |
| GPT-3       | 2                  |
| Chinchilla  | 20                 |
| LLaMA 65B   | 22                 |
| Llama 2 70B | 29                 |
| Mistral 7B  | 110                |
| Llama 3 70B | 215                |

含义是：当模型会被大量调用时，推理成本可能远大于训练成本。此时训练一个更小但训练更久的模型，可能比训练一个参数更大的 train-optimal 模型更划算。

### 7.9 IsoFLOP everywhere

PDF 第 55 页展示 IsoFLOP 不只用于自回归语言模型，也可用于 diffusion 和 MoE scaling。课程总结说，IsoFLOP 方法容易执行、通常比较干净：固定 FLOP budget，扫参数或其他自由度，找到每条曲线的最低点。【PDF†lecture_09.pdf, p.55】

### 7.10 学习提示

本节最重要的结论是：**compute-optimal scaling 是大模型训练中最直接影响预算分配的 scaling law 应用。**

复习时要抓住：

- joint scaling 同时考虑数据和模型；
- Kaplan 更偏向参数增长，Chinchilla 更强调数据同步增长；
- IsoFLOP 是理解 compute-optimal 的核心实验设计；
- train-optimal 不等于 deployment-optimal；
- scaling law 的结论对实验细节高度敏感。

------

## 8. Python 代码示例详解：本讲无对应 Python 文件

### 8.1 文件整体结构

本讲没有对应的 `lecture_09.py` 或 `lecture_09.ipynb`。GitHub 仓库中 Lecture 9 对应的是 `lecture_09.pdf`；README 说明可执行讲义命名为 `lecture_XX.py`，非可执行讲义命名为 `lecture_XX.pdf`。

因此，本节不会伪造 `【Python代码†source】`。为了帮助学习，下面只把课程中的方法整理为“算法流程”。这些流程不是仓库中的 Python 源码，而是对 PDF 和原报告内容的结构化复习。

### 8.2 课程方法的算法化流程

| 课程知识点          | 算法流程                                                     | 输入                        | 输出                          | 理论原理                      | 来源                              |
| ------------------- | ------------------------------------------------------------ | --------------------------- | ----------------------------- | ----------------------------- | --------------------------------- |
| 单变量 data scaling | 固定训练 recipe，改变数据量 $n$，记录 test loss，拟合 $\log L$ 对 $\log n$ 的直线 | 数据量、loss                | exponent $\alpha$、截距       | $L(n)\approx A n^{-\alpha}+C$ | 【PDF†lecture_09.pdf, p.14-p.15】 |
| 数据混合选择        | 对不同数据源比例训练小模型，比较 scaling 曲线                | mixture ratio、loss         | 候选最优 mixture              | 数据组成常改变 offset         | 【PDF†lecture_09.pdf, p.21-p.23】 |
| 重复数据评估        | 改变 repetition，估计 effective data                         | unique tokens、重复次数     | $D'$                          | 重复数据边际收益递减          | 【PDF†lecture_09.pdf, p.24】      |
| Critical batch size | 固定 target loss，扫 batch size，记录 $S,E$，拟合 $S_{\min},E_{\min}$ | batch size、steps、examples | $B_{\mathrm{crit}}$           | 平衡 step 与 examples         | 【PDF†lecture_09.pdf, p.37-p.39】 |
| IsoFLOP             | 固定 FLOP budget，改变模型参数和训练 token 数，取 loss 最小点 | FLOPs、params、tokens       | compute-optimal params/tokens | 固定成本下优化资源分配        | 【PDF†lecture_09.pdf, p.48】      |
| Joint scaling fit   | 在模型大小与数据大小网格上训练，拟合 loss surface            | $m,n,L$                     | $\alpha,\beta,C$ 等参数       | 联合 data-model scaling       | 【PDF†lecture_09.pdf, p.49】      |

### 8.3 关键流程解析：IsoFLOP 的“伪代码思维”

虽然本讲没有 Python 文件，但 IsoFLOP 可以用如下思维理解：

1. 选定一组 compute budgets，例如 $C_1,C_2,\ldots,C_k$；
2. 对每个 $C_i$，枚举不同模型大小 $N$；
3. 根据固定 FLOPs 约束确定对应训练 token 数 $D$；
4. 训练这些模型，记录 terminal loss；
5. 在每个 $C_i$ 下找到 loss 最低的 $N_i^*$；
6. 拟合 $N_i^*$ 与 $C_i$ 的 power law；
7. 外推到更大 compute budget。

**学习重点：** IsoFLOP 的本质不是某段代码，而是一种实验设计原则：**固定预算，扫自由度，找最优点，再拟合最优点随预算变化的规律。**

### 8.4 课程知识点与“可实现变量”对应表

| 课程知识点      | 实现中会记录的变量                          | 代码功能如果实现应做什么            | 对应理论原理                | 在课程中的作用     |
| --------------- | ------------------------------------------- | ----------------------------------- | --------------------------- | ------------------ |
| data scaling    | `dataset_size`, `test_loss`                 | 训练不同数据量模型并拟合 log-log 线 | power law                   | 预测更多数据的收益 |
| model scaling   | `num_params`, `loss`                        | 训练不同参数规模模型                | loss-parameter scaling      | 比较架构或规模     |
| data mixture    | `mix_ratio`, `source_loss`                  | sweep 数据源比例                    | offset vs slope             | 选择预训练数据配比 |
| data repetition | `unique_tokens`, `epochs`, `effective_data` | 估计重复带来的有效数据              | diminishing returns         | 判断是否重复数据   |
| critical batch  | `batch_size`, `steps`, `examples`           | 拟合 $S_{\min},E_{\min}$            | gradient noise scale        | 选择训练 batch     |
| LR scaling / μP | `width`, `learning_rate`, `loss`            | 比较不同规模的最优 LR               | scale-aware parametrization | 迁移小模型超参数   |
| IsoFLOP         | `flops`, `params`, `tokens`, `loss`         | 固定 FLOPs 找 loss 最小点           | compute-optimal scaling     | 分配训练预算       |

------

## 9. PDF / 幻灯片 / 图表解析

### 9.1 主要图表总览

| PDF 页面  | 展示内容                                                     | 支持的课程结论                             | 对应课程部分       |
| --------- | ------------------------------------------------------------ | ------------------------------------------ | ------------------ |
| p.2-p.4   | B200 场景、scaling law 定义、旧调参 vs 小模型外推            | 大模型训练需要预测工具                     | 动机               |
| p.6-p.11  | sample complexity、1993 learning curves、Banko & Brill、Kolachina、Hestness | scaling law 有长期历史                     | 历史背景           |
| p.14-p.20 | data scaling、log-log linearity、均值估计、非参数估计、intrinsic dimension | 数据 scaling 的经验规律和理论直觉          | Data scaling       |
| p.21-p.26 | 数据组成、数据混合、重复数据、有限数据过滤                   | 数据工程也可以 scaling 化                  | 数据扩展           |
| p.28-p.42 | 架构、优化器、深宽比、参数价值、critical batch、LR、下游风险 | scaling laws 可服务模型工程                | Model engineering  |
| p.43-p.53 | joint scaling、Kaplan、Chinchilla 三方法、差异解释           | compute-optimal scaling 对训练预算至关重要 | Data-model scaling |
| p.54-p.57 | overtraining、IsoFLOP everywhere、课程总结                   | 训练最优不等于部署最优                     | 总结与部署         |

### 9.2 关键图表详细解释

#### 图表 1：Data scaling 的三段曲线

PDF 第 14 页的图用横轴表示 training data set size，纵轴表示 generalization error。图中从左到右分为 small data region、power-law region、irreducible error region。【PDF†lecture_09.pdf, p.14】

读图重点：

- 左侧数据太少，模型还没有进入稳定学习区间；
- 中间区域近似直线，是 scaling law 最可用的区域；
- 右侧接近不可约误差，继续加数据收益变小。

**容易误解的地方：** scaling law 不是说所有数据规模上都是直线，而是中间有一段近似 power-law region。

#### 图表 2：语言模型 loss 与 dataset size 的 log-log 直线

PDF 第 15 页横轴是 dataset size，纵轴是 test loss。图上直线说明 loss 与 dataset size 在 log-log 图中近似线性。【PDF†lecture_09.pdf, p.15】

这张图对应公式：
$$
L(D)\approx A D^{-\alpha}+C
$$
它想说明：语言模型训练中的数据收益可以被简单函数近似描述，因此可以用于预测更大数据量下的 loss。

#### 图表 3：均值估计与非参数估计

PDF 第 17 页展示均值估计的 $1/n$ 收敛；PDF 第 19 页展示非参数估计中的 $n^{-1/d}$ 形式。【PDF†lecture_09.pdf, p.17, p.19】

这两页共同回答：为什么 scaling law 经常是 power law？因为许多估计误差天然以多项式速率下降。

读图重点：

- 均值估计是低维、简单任务，斜率可接近 $-1$；
- 非参数估计更灵活、更复杂，指数依赖维度；
- 神经网络 exponent 较小可以用“复杂函数学习”作直觉解释。

#### 图表 4：Distribution shift 与数据组成

PDF 第 22 页展示不同数据源比例下的 excess error 曲线，以及 expected error intercept 随数据源比例变化的图。【PDF†lecture_09.pdf, p.22】

读图重点：

- 不同 mixture 的曲线大致平行；
- 主要差异体现在截距；
- 最优数据配比可能对应最低 intercept。

这支持课程结论：数据组成可能不显著改变 slope，但会改变整体 loss 水平。

#### 图表 5：重复数据与 effective data

PDF 第 24 页展示 repeating examples 对 scaling 的影响，并给出 effective data 公式。【PDF†lecture_09.pdf, p.24】

读图重点：

- 重复数据初期仍然有收益；
- 继续重复后收益递减；
- 假设重复数据等价于 fresh data 会高估训练收益。

#### 图表 6：Transformer vs LSTM

PDF 第 30 页图中横轴是 non-embedding parameters，纵轴是 test loss。Transformer 曲线低于 LSTM 曲线。【PDF†lecture_09.pdf, p.30】

这张图想回答：是否值得把 LSTM 扩展到 GPT-3 规模？Scaling law 方法给出的答案是：小规模趋势已经显示 Transformer 更优，因此没有必要用巨额预算训练大型 LSTM 来试错。

#### 图表 7：Critical batch size

PDF 第 37-39 页展示小 batch 和大 batch 的优化路径、training speed 与 batch/noise scale 的关系，以及 critical batch size 随目标 loss 变化的趋势。【PDF†lecture_09.pdf, p.37-p.39】

读图重点：

- 小 batch 噪声大；
- 大 batch 能减少噪声，但超过某点收益递减；
- loss 目标越低，critical batch 越大。

#### 图表 8：Chinchilla 三种方法

PDF 第 46 页表格列出 Chinchilla 三种 scaling law fitting 方法得到的 exponent；第 47-49 页分别展示 lower envelope、IsoFLOP、joint fit。【PDF†lecture_09.pdf, p.46-p.49】

读图重点：

- 方法 1：取训练曲线 envelope；
- 方法 2：固定 FLOPs，扫模型大小；
- 方法 3：拟合二维 loss surface；
- 方法 1/2 更接近 $0.5/0.5$，与 Kaplan 的 $0.73/0.27$ 不同。

#### 图表 9：Overtraining 与 tokens per parameter

PDF 第 54 页列出 GPT-3、Chinchilla、LLaMA、Mistral、Llama 3 等模型的 tokens per parameter。【PDF†lecture_09.pdf, p.54】

读图重点：

- Chinchilla 的 20:1 是训练 compute 最优附近的经验规则；
- 现代部署模型常常训练更多 token；
- 使用量越大，越值得付出前期训练成本来降低推理成本。

------

## 10. 关键公式与推导汇总

| 公式                                                         | 符号解释                                              | 适用场景                     | 直觉解释                         | 对应课程内容  |
| ------------------------------------------------------------ | ----------------------------------------------------- | ---------------------------- | -------------------------------- | ------------- |
| $\epsilon(\hat h)\le \min_{h\in H}\epsilon(h)+2\sqrt{\frac{1}{m}\log\frac{2k}{\delta}}$ | $m$ 样本数，$k$ 假设类大小，$\delta$ 置信参数         | 有限假设类泛化界             | 样本越多，上界越小               | PDF p.6       |
| $L(n)=A n^{-\alpha}+C$                                       | $n$ 数据量，$\alpha$ scaling exponent，$C$ 不可约误差 | data scaling                 | 数据增加带来幂律下降             | PDF p.14-p.15 |
| $\log L(n)\approx \log A-\alpha\log n$                       | log-log 变换                                          | 读 scaling 图                | 幂律在 log-log 图上是直线        | PDF p.15      |
| $\mathbb E[(\hat\mu-\mu)^2]=\sigma^2/n$                      | $\hat\mu$ 样本均值，$\sigma^2$ 方差                   | 均值估计                     | 样本均值误差随 $1/n$ 下降        | PDF p.17      |
| $\mathrm{Error}(n)\approx n^{-1/d}$                          | $d$ 维度                                              | 非参数估计                   | 维度越高，收敛越慢               | PDF p.19      |
| $D'=U_D+U_D R_D^*(1-e^{-R_D/R_D^*})$                         | $U_D$ 唯一 token，$R_D$ 重复次数                      | 重复数据                     | 重复数据收益递减                 | PDF p.24      |
| $\frac{S}{S_{\min}}-1=\left(\frac{E}{E_{\min}}-1\right)^{-1}$ | $S$ steps，$E$ examples                               | critical batch 拟合          | 平衡 step 与样本效率             | PDF p.38      |
| $B_{\mathrm{crit}}=E_{\min}/S_{\min}$                        | critical batch size                                   | batch size 选择              | 过小浪费 step，过大浪费 examples | PDF p.38      |
| $\mathrm{Error}=n^{-\alpha}+m^{-\beta}+C$                    | $n$ 数据量，$m$ 模型大小                              | joint scaling                | 同时考虑数据和模型限制           | PDF p.43      |
| $N_{\mathrm{opt}}=C^{0.73},D_{\mathrm{opt}}=C^{0.27}$        | Kaplan compute-optimal                                | 固定 compute 预算            | 参数增长快于数据增长             | PDF p.45      |
| $N_{\mathrm{opt}}\propto C^{0.5},D_{\mathrm{opt}}\propto C^{0.5}$ | Chinchilla 近似结论                                   | compute-optimal LLM training | 参数和数据大致同步增长           | PDF p.46      |

### 10.1 重点推导：幂律为什么在 log-log 图上是直线？

从：
$$
L(n)=A n^{-\alpha}
$$
两边取对数：
$$
\log L(n)=\log(A n^{-\alpha})
$$
利用 $\log(ab)=\log a+\log b$：
$$
\log L(n)=\log A+\log(n^{-\alpha})
$$
再利用 $\log(n^{-\alpha})=-\alpha\log n$：
$$
\log L(n)=\log A-\alpha\log n
$$
这就是一条直线。斜率是 $-\alpha$，截距是 $\log A$。

**学习提示：** 看到 log-log 图上直线时，应立即想到 power law；看到斜率时，应想到 scaling exponent；看到平行曲线时，应想到“斜率相似、截距不同”。

### 10.2 重点推导：critical batch size 的含义

公式：
$$
\frac{S}{S_{\min}} - 1
=
\left(
\frac{E}{E_{\min}} - 1
\right)^{-1}
$$
把 step efficiency 和 example efficiency 放在同一曲线上。若 batch 太小，$S$ 很大，训练 step 多，系统并行效率差；若 batch 太大，$E$ 很大，虽然 step 少，但每一步消耗太多样本，样本效率差。

因此：
$$
B_{\mathrm{crit}}=\frac{E_{\min}}{S_{\min}}
$$
表示一个平衡点：batch 尽量大以利用并行，但还没有进入严重 diminishing returns 区间。

------

## 11. 概念对比与易混淆点

| 易混淆概念                              | 区别                                                   | 联系                           | 常见误解                    | 正确理解                                |
| --------------------------------------- | ------------------------------------------------------ | ------------------------------ | --------------------------- | --------------------------------------- |
| scaling law vs 理论定律                 | scaling law 多是经验拟合；理论定律有严格前提           | 理论可启发函数形式             | 以为 scaling law 永远成立   | 只在特定区间和实验设置下可靠            |
| power law vs log-linear                 | power law 是原始尺度关系；log-linear 是取 log 后的形态 | 二者等价                       | 看到直线就无限外推          | 必须检查是否处在 power-law region       |
| slope vs intercept                      | slope 决定随规模改善速度；intercept 决定整体水平       | 二者共同决定 loss              | 只看小规模单点              | 要看曲线趋势                            |
| data size vs effective data             | data size 是训练 token 数；effective data 考虑重复折扣 | 重复数据仍有部分价值           | 重复 token 等于 fresh token | 重复数据边际收益递减                    |
| total params vs non-embedding params    | total 包含所有参数；non-embedding 排除 embedding 等    | 都可用于 scaling 图            | 参数量怎么数都一样          | 参数计数会影响 scaling 结论             |
| train-optimal vs deployment-optimal     | 前者固定训练 compute；后者考虑推理成本                 | 都是资源优化                   | Chinchilla 一定是生产最优   | 大量服务时常需要 overtrain              |
| pretraining loss vs downstream accuracy | loss 更平滑可预测；下游任务更 noisy                    | 预训练 loss 通常相关           | perplexity 好就一定下游好   | 下游 scaling 可能不一致                 |
| critical batch vs 最大 batch            | critical batch 是收益递减前的平衡点                    | 都影响系统吞吐                 | batch 越大越好              | 超过 critical batch 会浪费样本          |
| 小模型最优超参数 vs 大模型最优超参数    | 有些可迁移，有些随 scale 变化                          | scaling law 试图预测迁移       | 小模型调好即可直接放大      | LR、warmup、batch 需要 scale-aware 调整 |
| IsoFLOP vs joint fit                    | IsoFLOP 固定预算扫自由度；joint fit 拟合二维函数       | 都用于 compute-optimal scaling | joint fit 一定更全面        | 函数形式错误时 joint fit 可能偏         |

------

## 12. 学习拓展与发散理解

### 12.1 与大模型训练实践的关系

**延伸理解：数据准备。**
 本讲说明，数据工作不能只看“总 token 数”。高质量数据、数据多样性、重复率、过滤强度都会影响 scaling。实际训练中，数据团队需要根据 compute budget 动态决定是否扩展数据来源、降低过滤强度或控制重复率。

**延伸理解：模型训练。**
 模型团队不能只追求最大参数量。Kaplan 与 Chinchilla 的差异说明，固定训练预算下模型过大可能 under-trained；更小模型配更多 token 可能更优。

**延伸理解：优化策略。**
 critical batch size 和 μP 说明，batch size、learning rate、warmup 不是简单复制已有模型配置即可。它们需要随规模调整，否则 scaling law 拟合本身都会被错误超参数污染。

**延伸理解：评估方法。**
 预训练 loss 是 scaling law 最稳定的目标，但部署模型最终关心的是下游任务、用户体验、推理延迟和成本。因此 scaling law 应与评估和部署约束结合使用。

**延伸理解：工程成本。**
 如果模型预计会被大量调用，overtraining 更有意义：用更多训练 token 换取更小参数量，从而降低长期推理成本。这正是 train-optimal 与 deployment-optimal 的区别。

### 12.2 与其他 lecture 的联系

本讲与 CS336 其他内容的关系可以这样理解：

| 课程方向                              | 与本讲的关系                                                 |
| ------------------------------------- | ------------------------------------------------------------ |
| PyTorch / FLOPs / resource accounting | scaling law 的横轴常常是 compute，需要理解 FLOPs 和资源计算  |
| Transformer architecture              | 架构 scaling 需要理解深度、宽度、attention heads 等结构变量  |
| Parallelism / systems                 | large batch、分布式训练、GPU 利用率都影响 scaling 实验的可执行性 |
| Data lectures                         | 数据混合、过滤、去重、重复数据与本讲 data scaling 紧密相连   |
| Inference                             | deployment-optimal 与推理成本直接相关                        |
| 后续 scaling lecture                  | 官方 schedule 显示 Lecture 11 也属于 Scaling laws，可能继续讨论更高级的 scaling 主题。 |

### 12.3 面试 / 考试 / 作业可能考察角度

| 问题                                          | 考察点                  | 回答思路                                            |
| --------------------------------------------- | ----------------------- | --------------------------------------------------- |
| 为什么 power law 在 log-log 图上是直线？      | 对数与幂律              | 从 $L=A n^{-\alpha}$ 取 log 推导                    |
| 均值估计为什么是 scaling law？                | 统计估计误差            | 推导 $\mathbb E[(\hat\mu-\mu)^2]=\sigma^2/n$        |
| 为什么 neural scaling exponent 可能远小于 1？ | 非参数估计直觉          | 用 $n^{-1/d}$ 解释复杂函数学习更慢                  |
| 数据组成如何影响 scaling curve？              | slope vs intercept      | 说明 mixture 可能改变 offset 而非 slope             |
| 重复数据为什么不等于 fresh data？             | effective data          | 用递减收益解释 $D'$ 公式                            |
| 如何设计 IsoFLOP 实验？                       | compute-optimal scaling | 固定 FLOPs，扫模型参数，找 loss 最低点              |
| Kaplan 与 Chinchilla 为什么结论不同？         | scaling law 敏感性      | 参数计数、warmup、batch/LR、小非线性                |
| Chinchilla 20:1 是否永远最优？                | train vs deployment     | 固定训练 compute 最优不等于生产推理最优             |
| critical batch size 如何定义？                | 优化与系统效率          | 解释 $S,E,S_{\min},E_{\min}$ 和 $B_{\mathrm{crit}}$ |
| 为什么下游 scaling 更难预测？                 | 评估噪声与迁移          | 预训练 loss 平滑，下游任务受更多因素影响            |

------

## 13. 课程复习总结

### 13.1 本讲核心思想

Lecture 9 的核心思想是：**大语言模型训练太昂贵，不能靠大规模盲目试错；scaling laws 通过小规模实验建立资源与 loss 之间的规律，帮助我们预测大规模训练行为，并在数据、模型、compute 和超参数之间做工程决策。**

这条主线从 data scaling 出发，先说明数据量与误差之间常出现 log-log 线性关系；再用均值估计和非参数估计解释幂律直觉；接着扩展到数据混合、重复数据、模型架构、优化器、batch size 和 learning rate；最后用 Kaplan vs Chinchilla 展示 compute-optimal scaling 的威力与风险。

### 13.2 必须掌握的概念

- scaling law；
- power law；
- log-log linearity；
- scaling exponent；
- intercept vs slope；
- data scaling；
- data mixture；
- effective data；
- repeated data diminishing returns；
- model scaling；
- architecture scaling；
- critical batch size；
- scale-aware learning rate；
- μP；
- joint data-model scaling；
- IsoFLOP；
- Kaplan scaling；
- Chinchilla scaling；
- train-optimal；
- deployment-optimal；
- overtraining。

### 13.3 必须理解的公式

| 公式                                                         | 解决的问题                       |
| ------------------------------------------------------------ | -------------------------------- |
| $L(n)=A n^{-\alpha}+C$                                       | 数据量或规模如何影响 loss        |
| $\log L(n)=\log A-\alpha\log n$                              | 为什么 log-log 图上是直线        |
| $\mathbb E[(\hat\mu-\mu)^2]=\sigma^2/n$                      | 最简单的统计 scaling law         |
| $\mathrm{Error}(n)\approx n^{-1/d}$                          | 为什么复杂任务 exponent 可能很小 |
| $D'=U_D+U_D R_D^*(1-e^{-R_D/R_D^*})$                         | 重复数据的有效数据量             |
| $B_{\mathrm{crit}}=E_{\min}/S_{\min}$                        | 如何选择 batch size              |
| $\mathrm{Error}=n^{-\alpha}+m^{-\beta}+C$                    | 数据与模型联合 scaling           |
| $N_{\mathrm{opt}}=C^{0.73},D_{\mathrm{opt}}=C^{0.27}$        | Kaplan compute-optimal 结论      |
| $N_{\mathrm{opt}}\propto C^{0.5},D_{\mathrm{opt}}\propto C^{0.5}$ | Chinchilla 近似结论              |

### 13.4 必须看懂的“代码 / 实验流程”

本讲没有对应 Python 源码，但必须看懂这些实验流程：

1. **单变量 scaling fit**：改变数据量、参数量或 compute，拟合 log-log 直线；
2. **data mixture sweep**：用小模型比较不同数据配比；
3. **data repetition analysis**：用 effective data 理解重复数据收益；
4. **critical batch sweep**：固定 target loss，扫 batch size；
5. **IsoFLOP profiles**：固定 FLOPs，扫模型大小和 token 数；
6. **joint scaling fit**：在模型大小和数据大小网格上拟合 loss surface。

### 13.5 一句话总结

**Scaling laws 是用小规模实验建立可外推的幂律规律，从而指导大模型训练中的数据、模型、compute 和超参数决策；但它们只有在正确计量、正确调参、正确区间和正确目标下才可靠。**