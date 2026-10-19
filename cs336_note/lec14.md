# Lecture 14 课程报告：Data II——从原始网页到后训练数据的数据工程管线

## 0. 资料来源与改写说明

本报告是在你上传的原始 Lecture 14 中文课程报告基础上进行系统性改写。原报告已经明确说明：Lecture 14 对应的是 Stanford CS336 `lectures` 仓库中的 **`lecture_14.py` 可执行讲义**，而不是单独 PDF；课程主线围绕 `transformation()`、`filtering()`、`deduplication()`、`data_mixing()` 和 `post_training_data()` 五个部分展开。 GitHub 仓库页面也显示 CS336 Spring 2026 的 lecture materials 中包含 `lecture_14.py`，并说明可执行讲义命名为 `lecture_XX.py`。

本次改写遵循以下边界：

- **课程资料内容**：来自原报告中的字幕整理、课堂例子、公式、代码解释，以及 GitHub `lecture_14.py` 中可核对的函数、代码片段和图像调用。
- **引用保留**：原报告中的“【字幕】”“【Python代码】”引用标记继续保留；系统可核对来源用文件引用和网页引用补充。
- **延伸内容**：凡是为了帮助理解而加入的背景、类比、工程直觉、复习提示，均标注为“补充说明”“延伸理解”或“学习提示”，不伪装成课程原文。
- **PDF / 图表说明**：Lecture 14 没有单独 PDF；图表来自 `lecture_14.py` 中通过 `image(...)` 插入的可执行讲义图像。以下图表解析会说明图表服务的课程问题，但不会把未直接读出的坐标轴细节伪装成课程原文。`lecture_14.py` 的 `main()` 函数确实按顺序调用了 `transformation()`、`filtering()`、`deduplication()`、`data_mixing()` 和 `post_training_data()`。

------

## 1. 学习导读

### 1.1 本讲要解决的核心问题

Lecture 14 讨论的是：**当我们已经从互联网、代码仓库、PDF、论坛等来源拿到大量原始数据之后，如何把它们变成可以真正用于大模型训练的数据？**

这句话听起来像工程问题，但它实际上贯穿了模型质量、训练成本、安全性、泛化能力和后训练效果。原始数据不是“干净文本”，它可能是 HTML、PDF、代码目录或大量重复、低质、有毒、语言混杂的文本；因此必须经过 **转换、过滤、去重、混合**，最后再为 SFT、reasoning、agentic coding 等后训练任务构造数据。原报告也将本讲总结为数据管线的五个部分：transformation、filtering、deduplication、mixing 和 post-training data。

### 1.2 本讲在 CS336 课程中的位置

本讲是 CS336 中关于数据的第二讲。上一讲更关注数据来源：例如 live services、GitHub、网页、PDF、论坛如何通过 dump 或 crawl 变成原始数据；本讲则从“已经拿到原始数据”开始，讨论如何让数据真正适合训练语言模型。【字幕】

它连接了课程中的多个主线：

- 对 **预训练** 来说，本讲解释如何得到高质量、多样、低重复的 token；
- 对 **训练效率** 来说，本讲解释如何避免把 GPU 时间浪费在 boilerplate、重复文本和低价值数据上；
- 对 **泛化与记忆** 来说，本讲解释为什么 deduplication 和 decontamination 很重要；
- 对 **数据混合** 来说，本讲解释为什么不能只凭直觉把 Wikipedia、Common Crawl、GitHub、books、papers 等来源拼在一起；
- 对 **后训练** 来说，本讲解释 SFT / reasoning / agent 数据如何通过真实环境、合成任务和强 teacher 模型构造。

### 1.3 本讲知识结构

| 模块                | 解决的问题                                           | 核心概念                                                     | 与后续内容的关系                                 |
| ------------------- | ---------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------ |
| Data Transformation | 原始 HTML / PDF / 代码目录如何变成可训练文本         | boilerplate removal、linearization、OCR、layout loss         | 为 filtering 和 deduplication 提供输入           |
| Filtering           | 如何从海量 raw data 中筛出目标质量、语言或领域的数据 | target data、raw data、scoring function、KenLM、fastText     | 决定训练数据的质量与领域分布                     |
| Deduplication       | 如何删除完全重复和近似重复数据                       | hash、Jaccard、MinHash、LSH                                  | 降低训练浪费、减少 memorization、辅助去污染      |
| Data Mixing         | 多个数据源如何按比例训练                             | mixture distribution、epoching、UniMax、RegMix、simulated epoching | 决定模型能力结构和过拟合风险                     |
| Post-training Data  | 如何构造任务相关数据                                 | environments、tasks/prompts、teacher responses、synthetic / semi-synthetic data | 支撑 SFT、reasoning、coding agent 和工具使用能力 |

### 1.4 学习本讲前需要知道什么

学习本讲最需要的前置知识不是 Transformer 细节，而是以下几类基础：

1. **概率与语言模型基础**：理解 $p(x)$、$p(y \mid x)$、perplexity、负对数似然。
2. **分类器基础**：理解线性分类器、bag-of-words、阈值过滤。
3. **集合与哈希基础**：理解集合交并、哈希碰撞、时间复杂度。
4. **大规模训练直觉**：知道训练 token 数、数据源大小、epoch 数之间的关系。
5. **工程系统直觉**：理解为什么 Web-scale 数据处理必须强调速度、可并行化和近似算法。

**学习提示**：本讲的关键词不是“某个单一算法”，而是 **trade-off**。几乎每个部分都在平衡：质量 vs. 数量、速度 vs. 精度、去噪 vs. 保留信息、低重复 vs. 多样性、小规模实验 vs. 大规模训练。

------

## 2. 第一部分：数据转换——从原始格式到可训练文本

这一节要解决的问题是：**互联网原始数据为什么不能直接训练语言模型？我们如何把 HTML、PDF、代码目录等复杂格式变成线性的 token 序列？**

### 2.1 问题引入

课程首先强调：raw data does not come as text。Common Crawl 里的网页不是干净段落，而是 HTML；PDF 不是语义化文档，而是页面布局；GitHub 不是一段文本，而是目录、文件、依赖、fork 和元数据的集合。【字幕】原报告也指出，Common Crawl、网页、GitHub、arXiv 等来源中的原始数据可能是 HTML、PDF、代码仓库目录或其他半结构化内容。

因此，大模型训练前的第一步不是“训练”，而是 **data transformation**：把复杂来源转换成模型可以读取的 token 序列。

补充说明：语言模型看到的是一维 token 序列。无论网页原本有几栏、表格如何对齐、标题在哪里、图片旁边的说明是什么，最终都要变成类似
$$
x = (x_1, x_2, \dots, x_n)
$$
这样的序列。这一步天然会丢失一部分结构信息。

### 2.2 概念定义

**直觉解释**：数据转换就像把一本带排版、图片、表格和脚注的教材“抄写”成一段纯文本。抄得越好，模型越容易学到正文内容；抄得越差，模型就会学到导航栏、广告、乱码、页眉页脚，甚至错误的表格顺序。

**正式定义**：Data transformation 是将原始数据对象
$$
r \in \mathcal{R}
$$
转换为文本或 token 序列
$$
x = (x_1, x_2, \dots, x_n)
$$
的过程。这里的 $r$ 可以是 HTML 页面、PDF 文件、代码仓库、论坛帖子或其他结构化对象。

它解决的问题是：训练语言模型需要统一的数据表示，而真实世界的数据格式高度异构。

### 2.3 HTML 到文本：去除 boilerplate 与线性化

HTML 是 Web 数据的主要格式。本讲把 HTML-to-text 作为 transformation 的核心例子。

#### 2.3.1 Boilerplate removal

Boilerplate 指网页上大量重复但不一定有训练价值的内容，例如：

- 导航栏；
- 菜单；
- 广告；
- 页眉页脚；
- cookie banner；
- 推荐列表；
- 版权声明；
- 侧边栏。

课程指出，常见做法是移除 footer、header、menu 等，再提取正文。但这并不是完全机械的任务，因为“什么是正文”并不总是清楚：某些导航、目录或页面结构本身可能对模型理解网页也有价值。【字幕】

**容易误解的地方**：不要以为 boilerplate 一定是垃圾。对普通语言建模来说，广告和导航栏大多是噪声；但对某些网页理解、搜索、浏览器代理任务来说，页面结构可能也有价值。因此 transformation 的目标要服务于训练目标。

#### 2.3.2 Linearization：把结构压平成序列

HTML 有 DOM 层次结构，也有视觉布局；但语言模型通常训练在一维 token 序列上。因此必须把网页压平：
$$
\text{HTML / DOM / layout}
\longrightarrow
(x_1, x_2, \dots, x_n)
$$
这一步是 **lossy process**：网页中标题层级、表格对齐、图片位置、左右栏关系、脚注关联等都可能部分丢失。原报告明确指出，HTML-to-text 必须 linearize，因此 inherently lossy。

**工程直觉**：数据转换不是“有没有解析出来”这么简单，而是“解析出的文本是否保留了最重要的语义关系”。一个表格如果被错误线性化，模型可能学到完全错误的键值对应关系。

#### 2.3.3 表格与图片

课程特别指出表格很难处理。简单表格可以转成 Markdown；复杂表格、嵌套表格、跨行跨列内容则很难保持结构。【字幕】

对于图片，传统文本模型无法直接使用图像内容，除非做 OCR、captioning 或使用多模态模型。因此 transformation 也涉及是否要：

- 丢弃图片；
- 提取 alt text；
- 使用 OCR；
- 使用 VLM 生成描述；
- 保留图文关系。

### 2.4 PDF 到文本：高价值但高成本

PDF 与 HTML 的核心区别在于：HTML 至少有一定语义标签，例如 `<h1>`、`<p>`、`<table>`；PDF 更像“页面绘制指令”，强调字符在哪里显示，而不是这段文字在语义上属于标题、正文还是表格。原报告指出，PDF 比网页更容易缺失 layout / semantic structure，因为 PDF 的目标主要是布局显示。

PDF 处理面临的问题包括：

1. **文件可能被截断**：Common Crawl 中大 PDF 可能下载不完整，需要 recrawl；
2. **可能是扫描件**：扫描 PDF 本质是图片，需要 OCR；
3. **版式复杂**：多栏论文、公式、脚注、页眉页脚、表格都会影响解析；
4. **成本高**：用 OCR、VLM 或专门工具处理 PDF，比普通 HTML 解析昂贵得多。

`lecture_14.py` 中也提到 FinePDFs：其来源包括 Common Crawl，需要 recrawl truncated PDFs，并使用 RolmOCR、VLM 或 Docling 做 OCR / 文本化，同时还要 cleanup 和 filtering。【Python代码】

**学习提示**：PDF 的价值和难度同时更高。价值在于 PDF 往往是论文、教材、报告等较正式内容；难度在于它的语义结构不天然暴露给解析器。

### 2.5 代码 / PDF 对应说明

| 课程知识点       | `lecture_14.py` 对应位置                        | 代码 / 讲义作用                                       | 学习重点                      |
| ---------------- | ----------------------------------------------- | ----------------------------------------------------- | ----------------------------- |
| 原始数据不是文本 | `transformation()`                              | 列出 HTML、PDF、code directories                      | 理解为什么需要 transformation |
| HTML-to-text     | `transformation()`                              | 讲 boilerplate removal、images、tables、linearization | 明白转换是 lossy              |
| 规则工具         | `trafilatura`、`resiliparse`、`jusText`、`lynx` | 说明大规模处理常依赖快速规则系统                      | 速度和准确率都重要            |
| FinePDFs         | `transformation()` 中的 FinePDFs 部分           | 说明 PDF recrawl、OCR、cleanup、filtering             | PDF 高质量但处理成本高        |

### 2.6 学习提示

本节最重要的结论是：**数据转换决定模型看到的“世界版本”是什么样的。**

最容易混淆的地方是：把 transformation 当成简单清洗。实际上它是有损编码过程，会决定正文、结构、表格、图片和代码目录如何被表示。

复习时抓住三句话：

1. 原始数据通常不是纯文本；
2. HTML / PDF / 代码目录都需要专门转换；
3. 转换越差，后面的 filtering、deduplication 和 training 都会被污染。

------

## 3. 第二部分：数据过滤——从海量文本中找到“想要的数据”

这一节要解决的问题是：**当我们已经把原始数据转换成文本之后，如何从巨大 raw corpus 中筛选出目标语言、目标质量或目标领域的数据？**

### 3.1 问题引入

完成 transformation 后，我们得到的是“文本”，但还不是“好训练数据”。里面可能有：

- 非目标语言；
- 乱码；
- SEO 垃圾；
- 广告文本；
- 毒性内容；
- 重复模板；
- 与目标任务无关的内容；
- 低质量代码或错误数学内容。

课程把 filtering 抽象为一个统一框架：给定少量目标数据 $T$ 和大量原始数据 $R$，从 $R$ 中找到一个与 $T$ 相似的子集 $T'$。原报告也写出这一框架：少量高质量 target data 加上海量 raw data，目标是筛出 $T' \subseteq R$。

### 3.2 概念定义：目标数据、原始数据与打分函数

给定原始数据集：
$$
R = \{x_1, x_2, \dots, x_N\}
$$
其中 $R$ 很大，可能达到几十万亿或上百万亿 tokens。

给定目标数据集：
$$
T = \{t_1, t_2, \dots, t_m\}
$$
其中 $T$ 通常较小但质量较高，例如 Wikipedia、数学文本、优质代码、无毒评论等。

目标是找到：
$$
T' \subseteq R
$$
使得：
$$
T' \approx T
$$
也就是 $T'$ 在某种目标意义上“像” $T$。

常见流程是：

1. 基于 $T$ 和 / 或 $R$ 训练一个模型；
2. 得到一个 scoring function：

$$
s(x)
$$

1. 对每个 $x \in R$ 打分；
2. 根据阈值保留：

$$
x \in T'
\Longleftrightarrow
s(x) \ge \tau
$$

其中 $\tau$ 是过滤阈值。

**直觉解释**：过滤器像一个高速阅卷员。它不能逐字理解所有互联网文本，但必须足够快地判断：“这篇文章像不像我想要的高质量目标数据？”

### 3.3 过滤算法的两个核心要求

课程强调 filtering algorithm 有两个 desiderata。

#### 3.3.1 要能泛化

如果目标数据 $T$ 是 Wikipedia，我们不是只想保留 Wikipedia，而是想找到更多类似 Wikipedia 的高质量网页。因此过滤器不能只是“记住目标数据”，而要学习“什么样的数据像目标数据”。

#### 3.3.2 要极快

过滤器要跑在 Web-scale 数据上，可能是 $10^{14}$ tokens 级别。因此实际系统常使用 KenLM、fastText、线性分类器、bag-of-words 特征等较轻量方法，而不是对每篇文档都用大模型深度推理。

**工程直觉**：过滤器不一定要最聪明，但必须能跑完。一个完美但太慢的过滤器，在大规模数据管线里不可用。

------

### 3.4 生成式过滤：KenLM / n-gram 语言模型

生成式过滤的思想是：只对目标数据 $T$ 建模，得到目标域语言模型：
$$
p_T(x)
$$
然后用它判断一个新样本 $x$ 是否像目标数据。

若文本为：
$$
x = (w_1, w_2, \dots, w_n)
$$
n-gram 模型估计：
$$
p_T(x)
=
\prod_{i=1}^{n}
p_T(w_i \mid w_{i-k+1}, \dots, w_{i-1})
$$
对于 5-gram KenLM，当前位置 $w_i$ 的概率由前 4 个词近似决定：
$$
p_T(w_i \mid w_{i-4}, w_{i-3}, w_{i-2}, w_{i-1})
$$
实际过滤时常用平均负对数似然或 perplexity：
$$
\text{NLL}(x)
=
-\frac{1}{n}
\sum_{i=1}^{n}
\log p_T(w_i \mid \text{history})
$$
如果：
$$
\text{PPL}(x) < \tau
$$
就认为 $x$ 更像目标域，保留。

**符号解释**：

- $x$：候选文档或文本片段；
- $w_i$：第 $i$ 个词或 token；
- $T$：目标数据；
- $p_T$：在目标数据上训练出的语言模型；
- $\tau$：过滤阈值；
- PPL 越低，表示目标模型越容易预测该文本，因此该文本越像目标域。

**课堂例子：OpenMathText**

OpenMathText 的目标是从 Common Crawl 中筛选数学文本。课程和代码中提到的流程包括：规则过滤 LaTeX 命令、用 KenLM 在 ProofPile 上训练并按 perplexity 过滤、再训练 fastText 分类器判断 mathematical writing，并根据是否包含 LaTeX 使用不同阈值。原报告指出，GitHub 代码中更精确写到该流程 produced 14.7B tokens，并用于训练 1.4B 模型，效果优于 20x data 的模型。

**学习提示**：生成式过滤回答的是：“这段文本有多像目标数据自己会生成的文本？”

------

### 3.5 判别式过滤：fastText / 线性分类器

判别式过滤直接训练一个二分类器：
$$
y =
\begin{cases}
1, & x \in T \\
0, & x \in R_{\text{sample}}
\end{cases}
$$
训练后得到：
$$
s(x) = P(y=1 \mid x)
$$
保留规则为：
$$
x \in T'
\Longleftrightarrow
P(y=1 \mid x) \ge \tau
$$
如果使用 bag-of-words 或 n-gram 特征：
$$
\phi(x) \in \mathbb{R}^d
$$
线性分类器可写成：
$$
z = w^\top \phi(x) + b
$$
再通过 sigmoid 得到概率：
$$
P(y=1 \mid x)
=
\sigma(w^\top \phi(x) + b)
$$
保留条件为：
$$
\sigma(w^\top \phi(x) + b) \ge \tau
$$
**直觉解释**：生成式方法问“这篇文章像不像目标域写出来的”；判别式方法问“这篇文章更像目标数据，还是更像随机 raw data？”

**代码对应**：`lecture_14.py` 中明确把 simple classifier / fastText 的 score 写成 $p(T \mid x)$，并指出 keep examples with score above threshold。

------

### 3.6 过滤任务一：语言识别

语言识别的目标是判断文本是否属于目标语言。例如训练英文模型时，可以定义：
$$
s_{\text{lang}}(x)
=
P(\text{English} \mid x)
$$
并保留：
$$
s_{\text{lang}}(x) \ge \tau
$$
课程提到 fastText language identification 支持 176 种语言，并使用 Wikipedia、Tatoeba、SETimes 等多语资源训练；Dolma 中保留 $p(\text{English}) \ge 0.5$ 的网页。原报告也整理了这一点。

**容易误解的地方**：语言识别看似简单，但 code-switching、方言、转写、乱码、多语混排都会让任务变复杂。对多语模型来说，问题也不是“删掉非英文”，而是如何控制各语言比例。

------

### 3.7 过滤任务二：质量过滤

“质量”不是一个绝对概念，而是与目标相关。原报告特别强调：如果目标是通用百科知识，高质量可能接近 Wikipedia；如果目标是数学能力，高质量可能是数学文本；如果目标是代码教学，高质量可能是教育性强的 Python 代码。

#### 3.7.1 GPT-3：用高质量源做正样本

GPT-3 过滤可以放入统一框架：

- 正样本：Wikipedia、WebText、Books；
- 负样本：Common Crawl；
- 模型：线性分类器；
- 结果：根据分类器分数保留文档。

原报告指出，GitHub 代码中 positives 更具体写为 Wikipedia、WebText2、Books1、Books2，negatives 为 CommonCrawl，并包含按 classifier score 随机保留文档的 `keep_document(score)` 代码片段。

#### 3.7.2 LLaMA / RedPajama：Wikipedia 引用页

LLaMA / RedPajama 的思路不是直接把 Wikipedia 当正样本，而是把 **Wikipedia 引用的网页** 当正样本。直觉是：被 Wikipedia 引用的页面更可能是可信、高质量信息来源。原报告也指出，课程将其正样本定义为 pages referenced by Wikipedia，负样本为 CommonCrawl。

#### 3.7.3 phi-1：用强模型先生成目标，再训练便宜分类器

phi-1 的例子很重要，因为它展示了一种现代数据过滤模式：
$$
\text{强模型 / 少量昂贵标注}
\rightarrow
\text{高质量目标集}
\rightarrow
\text{便宜分类器大规模推广}
$$
流程是：

1. 原始数据 $R$ 是 The Stack 中的 Python 子集；
2. 设计 prompt：判断代码是否有助于学生学习基础编程概念；
3. 用 GPT-4 对 100K 子集分类；
4. 将 GPT-4 判断为正例的数据作为目标集 $T$；
5. 用这些数据训练便宜分类器，例如 Random Forest；
6. 用便宜分类器扩展到整个 $R$。

这说明目标数据 $T$ 不一定来自人工专家，也可以来自强 teacher model 的判断。【字幕】原报告也明确总结为 expensive classifier → cheap classifier。

------

### 3.8 过滤任务三：毒性过滤

毒性过滤同样可以写成分类问题：
$$
y =
\begin{cases}
1, & \text{toxic} \\
0, & \text{non-toxic}
\end{cases}
$$
或者反过来定义 $y=1$ 表示 non-toxic。训练分类器后，可以过滤掉 toxic 内容。课程提到 Jigsaw Toxic Comments 数据集，它来自 Wikipedia talk pages，并标注 toxic、severe toxic、obscene、threat、insult、identity hate 等标签。

**补充说明**：毒性过滤并不是越强越好。如果分类器误删少数群体讨论、医学文本、社会议题或反仇恨内容，模型会失去对重要语境的理解。因此安全过滤也需要评估偏差和误删率。

------

### 3.9 过滤阈值没有唯一最优：scale-dependent effect

一个关键难点是：过滤阈值 $\tau$ 没有全局最优。

直觉上：

- 如果训练 token 少，应该选择更高质量、更小的数据；
- 如果训练 token 多，只用很小的高质量数据集会导致多次 epoch；
- 多次重复同一小数据源可能导致 overfitting；
- 因此训练越久，可能越需要更多低质量但多样的数据。

设阈值为 $\tau$ 时保留下来的数据集大小为 $|D_\tau|$，训练总 token 数为 $N_{\text{train}}$，平均 epoch 数为：
$$
E_\tau
=
\frac{N_{\text{train}}}{|D_\tau|}
$$
当 $\tau$ 太高，$|D_\tau|$ 变小，$E_\tau$ 变大，过拟合风险上升。
 当 $\tau$ 太低，$|D_\tau|$ 变大，但平均数据质量下降。

所以过滤阈值本质上是在平衡：
$$
\text{质量}
\quad vs. \quad
\text{多样性}
\quad vs. \quad
\text{重复训练风险}
$$
原报告也指出，课程明确说不存在单一最优阈值；训练更久时可能需要更多低质量数据，训练较短时更偏向少量高质量数据。

### 3.10 学习提示

本节要记住三点：

1. Filtering 的核心框架是：少量目标数据 $T$ + 大量 raw data $R$ → 筛选 $T'\subseteq R$。
2. 过滤器既要能泛化，又要足够快。
3. 阈值选择依赖训练规模，不能脱离 $N_{\text{train}}$、数据集大小和 epoch 风险单独讨论。

------

## 4. 第三部分：数据去重——Exact Deduplication、Jaccard、MinHash 与 LSH

这一节要解决的问题是：**即使数据已经转换和过滤，为什么还要去重？如何在 Web-scale 上找到完全重复和近似重复内容？**

### 4.1 问题引入

互联网数据中有大量重复：

1. **Exact duplicates**：完全重复，例如镜像站、GitHub fork、复制粘贴文件；
2. **Near duplicates**：只差少量 token，例如模板文章、许可证文本、格式差异、页眉页脚变化。

原报告提到，课程中还举了 C4 数据集里某个商品描述重复出现约 61,000 次的例子，说明 Web 数据中的重复可能非常严重。

去重的目的包括：

- 减少训练 token，节省 GPU；
- 避免模型过度记忆重复内容；
- 降低隐私、版权和 memorization 风险；
- 辅助 decontamination，确保测试集不进入训练集。

### 4.2 去重设计空间

课程将 deduplication 分成三个设计问题：

| 问题               | 可选方案                                                   | 直觉           |
| ------------------ | ---------------------------------------------------------- | -------------- |
| 去重单位是什么？   | sentence、paragraph、document、代码文件、repo、n-gram span | 先决定比较对象 |
| 如何定义 match？   | exact match、共享子片段、共享比例、Jaccard 阈值            | 决定什么算重复 |
| 找到重复后怎么办？ | 删除所有、保留一个、保留质量最高一份、局部删除重复片段     | 决定处理策略   |

原报告也保留了这一 design space：item、match、action 三个问题。

**学习提示**：去重不是一个单一算法，而是一组设计选择。不同训练目标可能需要不同粒度：网页文本可按文档去重，代码可能需要按文件或 repo 去重。

------

### 4.3 核心挑战：不能做 $O(n^2)$ 比较

如果数据集中有 $n$ 个文档，两两比较需要：
$$
\binom{n}{2}
=
\frac{n(n-1)}{2}
=
O(n^2)
$$
Web-scale 数据下这不可行。因此课程强调，deduplication 必须依赖哈希、sketch、LSH 等近似或线性可扩展方法。原报告也指出，deduplication 本质上是 item-to-item comparison，但不能做 $n^2$，必须用线性时间算法。

------

### 4.4 Hash function：精确去重的基础

哈希函数可以写成：
$$
h: \mathcal{X} \rightarrow \mathcal{H}
$$
它把一个对象 $x \in \mathcal{X}$ 映射到较小的 hash value $h(x)$。

哈希碰撞定义为：
$$
h(x) = h(y),
\quad
x \ne y
$$
课程区分：

- **cryptographic hash**：如 SHA-256，抗碰撞强，但慢；
- **non-cryptographic hash**：如 MurmurHash、CityHash，不抗恶意碰撞，但快，适合大规模哈希表和数据处理。

`lecture_14.py` 使用 `mmh3.hash("hello")` 演示 MurmurHash。【Python代码】

------

### 4.5 精确去重代码示例

讲义中的精确去重示例逻辑是：

```
items = ["Hello!", "hello", "hello there", "hello", "hi", "bye"]

hash_items = itertools.groupby(
    sorted(items, key=mmh3.hash),
    key=mmh3.hash
)

deduped_items = [next(group) for h, group in hash_items]
```

这段代码做了四件事：

1. 对每个字符串计算 hash；
2. 按 hash 排序；
3. 使用 `groupby` 将相同 hash 的 item 分到一起；
4. 每组只保留一个。

原报告指出，这种写法类似 MapReduce，容易并行化和扩展；缺点是无法处理 near duplicate，例如 `"hello"` 和 `"hello "` 会被视为不同。

**课堂例子：C4 三句 span 去重**

C4 的 exact deduplication 中：

- item：three-sentence spans；
- match：exact match；
- action：remove all but one。

课程提醒：如果删除的是文档中间的三句话，可能破坏文档连贯性。

------

### 4.6 近似去重：Jaccard similarity

为处理 near duplicates，课程引入 Jaccard similarity。

给定两个集合 $A, B$：
$$
J(A,B)
=
\frac{|A \cap B|}{|A \cup B|}
$$
其中：

- $A \cap B$：两个集合共有的元素；
- $A \cup B$：两个集合所有不同元素；
- $J(A,B) \in [0,1]$。

例子：
$$
A = \{1,2,3,4\},
\quad
B = \{1,2,3,5\}
$$
则：
$$
A \cap B = \{1,2,3\},
\quad
A \cup B = \{1,2,3,4,5\}
$$
所以：
$$
J(A,B)
=
\frac{3}{5}
=
0.6
$$
课程用这个例子说明 Jaccard：0 表示完全无重叠，1 表示完全相同；如果两个文档 Jaccard 高于某个阈值，例如 0.99，就可以认为它们是 near duplicates。

------

### 4.7 MinHash：用哈希估计 Jaccard

直接计算所有文档对的 Jaccard 仍然需要大量两两比较。MinHash 的核心是：用哈希碰撞概率近似 Jaccard。

给定集合 $S$ 和随机哈希函数 $h$：
$$
\text{MinHash}_h(S)
=
\min_{x \in S} h(x)
$$
代码形式是：

```
def minhash(S: set[str], seed: int):
    return min(mmh3.hash(x, seed) for x in S)
```

MinHash 的关键性质是：
$$
P[
\text{MinHash}_h(A)
=
\text{MinHash}_h(B)
]
=
J(A,B)
$$
**推导直觉**：

把 $A \cup B$ 中所有元素随机排列。MinHash 等价于看随机排列中最先出现的元素。

- 如果最先出现的元素属于 $A \cap B$，则它同时是 $A$ 和 $B$ 的最小哈希元素，MinHash 相等；
- 如果最先出现的元素只属于 $A$ 或只属于 $B$，则 MinHash 不相等。

因此碰撞概率就是：
$$
\frac{|A \cap B|}{|A \cup B|}
=
J(A,B)
$$
课程用 $A=\{1,2,3,4\}$、$B=\{1,2,3,5\}$ 的例子说明：如果 1、2、3 先出现，两者 MinHash 相同；如果 4 或 5 先出现，则不同。原报告也完整保留了这个推导。

**学习提示**：普通 hash 希望不同对象尽量不要碰撞；MinHash 反过来希望“越相似越容易碰撞”。这是近似去重的核心思想。

------

### 4.8 LSH：把相似度概率变成阈值判断

单个 MinHash 只能保证：
$$
P[\text{collision}] = J(A,B)
$$
但实际去重想要的是：
$$
J(A,B) \ge \tau
\Rightarrow
\text{high probability collision}
$$
因此课程引入 Locality-Sensitive Hashing, LSH。

#### 4.8.1 Banding 结构

使用 $n$ 个独立哈希函数，分成 $b$ 个 bands，每个 band 有 $r$ 个 hash functions：
$$
n = b \cdot r
$$
两个文档被认为 collide 的规则是：

> 存在某一个 band，使得该 band 内所有 $r$ 个 hash values 都相同。

这形成一个 AND-OR 结构：

- band 内部：所有 hash 都相同，AND；
- band 之间：任意一个 band 匹配即可，OR。

原报告也指出，课程称这是 band 内 AND、多个 band 之间 OR 的结构。

#### 4.8.2 LSH 碰撞概率公式

设：
$$
s = J(A,B)
$$
单个 MinHash 相等概率为 $s$。

一个 band 中 $r$ 个 hash 全部相等的概率为：
$$
s^r
$$
一个 band 不匹配的概率为：
$$
1 - s^r
$$
所有 $b$ 个 bands 都不匹配的概率为：
$$
(1 - s^r)^b
$$
因此至少一个 band 匹配的概率为：
$$
P_{\text{collision}}(s;b,r)
=
1 - (1 - s^r)^b
$$
代码对应：

```
def get_prob_collision(sim, b, r):
    prob_match = sim ** r
    prob_collision = 1 - (1 - prob_match) ** b
    return prob_collision
```

原报告也将此函数与公式一一对应。

#### 4.8.3 $b$ 和 $r$ 的作用

公式：
$$
P_{\text{collision}}(s;b,r)
=
1-(1-s^r)^b
$$
会产生类似 S-shaped curve 的效果。

- 增大 $r$：band 内匹配更难，阈值更严格，曲线向右移动；
- 增大 $b$：匹配机会更多，召回更高，曲线向左移动。

原报告也指出，课程中讲到 increasing $r$ makes matching harder，而 increasing $b$ gives more matching chances。

#### 4.8.4 相变阈值

课程给出一个现实设置：
$$
b = 20,
\quad
r = 450
$$
近似相变阈值：
$$
\tau
\approx
\left(\frac{1}{b}\right)^{1/r}
$$
对应代码：

```
threshold = (1 / b) ** (1 / r)
```

当 $b=20, r=450$：
$$
\tau
\approx
(1/20)^{1/450}
\approx
0.99336
$$
在该阈值附近：
$$
s^r = \frac{1}{b}
$$
碰撞概率约为：
$$
P
=
1-\left(1-\frac{1}{b}\right)^b
$$
当 $b=20$：
$$
P
\approx
1-(0.95)^{20}
\approx
0.6415
$$
接近：
$$
1-\frac{1}{e}
\approx
0.632
$$
原报告指出，课程中也说相变中心碰撞概率约为 0.64。

### 4.9 学习提示

本节复习时抓住一条链：
$$
\text{Exact dedup}
\rightarrow
\text{Jaccard}
\rightarrow
\text{MinHash}
\rightarrow
\text{LSH}
$$

- Hash 解决 exact dedup；
- Jaccard 定义 near duplicate；
- MinHash 用碰撞概率估计 Jaccard；
- LSH 用 banding 把相似度变成近似阈值判断。

------

## 5. 第四部分：数据混合——多源数据如何按比例训练

这一节要解决的问题是：**当我们已经有多个数据源时，训练时应该按什么比例采样 Wikipedia、Common Crawl、GitHub、books、papers、math、PDFs 等数据？**

### 5.1 问题引入

大模型通常不是只在一个数据源上训练，而是在多个 sources 上训练。原报告指出，课程中明确说语言模型训练在 multiple data sources 上进行，关键问题是如何 combine sources。`lecture_14.py` 中给出例子：

```
sources = {"Wikipedia", "CC", "GitHub"}
p = {"Wikipedia": 0.3, "CC": 0.5, "GitHub": 0.2}
```



### 5.2 Data mixture 的数学定义

设数据源集合为：
$$
S = \{s_1, s_2, \dots, s_K\}
$$
数据混合就是在这些来源上定义一个概率分布：
$$
p(s_i) \ge 0,
\quad
\sum_{i=1}^{K} p(s_i)=1
$$
训练时先采样来源：
$$
s \sim p(s)
$$
然后从对应 source 采样 sequence 或 batch。

**课堂细节**：字幕中回答学生问题时说明，通常不是每个 token 单独采样来源，而是每个 sequence 来自某个 mixture component；batch 中可以混合多个来源以降低方差。【字幕】原报告也保留了这一点。

------

### 5.3 三种基础混合策略

#### 5.3.1 手动混合：vibes-based

人工设定：
$$
p(\text{Wikipedia})=0.3,
\quad
p(\text{CC})=0.5,
\quad
p(\text{GitHub})=0.2
$$
优点是简单；缺点是主观、难复现、难优化。

#### 5.3.2 均匀采样

$$
p(s_i)=\frac{1}{K}
$$

优点：每个 source 都有存在感。
 缺点：小数据源可能被严重过采样，大数据源可能被低采样。

#### 5.3.3 按 token 数比例采样

$$
p(s_i)
=
\frac{N_i}{\sum_j N_j}
$$

其中 $N_i$ 是 source $s_i$ 的 token 数。

优点：不容易过采样小数据源。
 缺点：巨大但低质量的数据源可能吃掉大部分训练预算；小而高质量的数据源权重可能太低。

原报告也指出，proportional mixing 看似理性，但如果有 huge low-quality dataset，它会占据大量 token，不一定最优。

------

### 5.4 数据混合中的 epoch 问题

课程用一个例子说明：不能只看 mixture 权重，还要看每个 source 的大小。

设低质量数据源大小为：
$$
N_{\text{low}}
=
10T
=
10 \times 10^{12}
$$
高质量数据源大小为：
$$
N_{\text{high}}
=
10B
=
10 \times 10^9
$$
训练总 token 为：
$$
N_{\text{train}}
=
1T
=
10^{12}
$$
朴素混合：
$$
p_{\text{low}} = 0.5,
\quad
p_{\text{high}} = 0.5
$$
低质量 source 的 epoch 数：
$$
E_{\text{low}}
=
\frac{p_{\text{low}} N_{\text{train}}}{N_{\text{low}}}
=
\frac{0.5 \times 10^{12}}{10 \times 10^{12}}
=
0.05
$$
高质量 source 的 epoch 数：
$$
E_{\text{high}}
=
\frac{p_{\text{high}} N_{\text{train}}}{N_{\text{high}}}
=
\frac{0.5 \times 10^{12}}{10 \times 10^9}
=
50
$$
也就是说，高质量小数据源会被重复训练 50 次。课程指出，这种情况最好是浪费 compute，最坏会 overfit。原报告也保留了这一推导与课程结论。

**学习提示**：mixture weight 不是全部。真正要检查的是每个 source 被训练了多少 epoch。

------

### 5.5 UniMax：给每个 source 设置 epoch cap

UniMax 的核心思想是：可以接近均匀采样，但给每个 source 设置最大 epoch 数上限。

设 epoch cap 为 $C$，则要求：
$$
\frac{p(s)N_{\text{train}}}{N_s}
\le
C
$$
等价于：
$$
p(s)N_{\text{train}}
\le
C N_s
$$
原报告中写道，UniMax 的想法是 sample sources uniformly but with a hard cap $C$ on number of epochs；达到 cap 后，该 source 不再继续提供 token，转向其他数据源。

**工程直觉**：UniMax 是一个 safety net。它防止小语种、小领域、高质量但小的数据源被无限重复训练。

------

### 5.6 Regression-based Mixing / RegMix / Olmix

Regression-based mixing 的目标是更系统地选择 mixture，而不是靠直觉。

基本流程：

1. 在小模型上采样多个 mixture：

$$
p^{(1)}, p^{(2)}, \dots, p^{(M)}
$$

1. 每个 mixture 训练一个 proxy model；
2. 在目标指标上得到 loss：

$$
L^{(i)} = L(p^{(i)})
$$

1. 拟合回归模型：

$$
\hat{L}(p) \approx L(p)
$$

1. 优化得到：

$$
p^*
=
\arg\min_p \hat{L}(p)
$$

1. 用 $p^*$ 训练大模型。

原报告指出，课程描述为：在小规模模型上尝试不同 mixtures，训练一组小模型，得到 target metric loss；再拟合 regression model，把 mixture weights 映射到 loss；最后优化出大模型 mixture。

#### 两个“信念跳跃”

课程提醒，这种方法依赖两个希望：

1. 回归模型在最优点附近仍然准确；
2. 小模型最优 mixture 能迁移到大模型。

这两个都不一定成立。优化可能把 mixture 推到训练分布之外；小模型训练 token 少时偏好的高质量小数据源，到了大模型长训练时可能导致严重 epoching。

------

### 5.7 Simulated epoching：让小规模实验模拟大规模数据稀缺

为了解决小规模实验和大规模训练之间的数据稀缺差异，课程介绍 simulated epoching。

设小规模实验 token 数为：
$$
N_{\text{small}}
=
10B
$$
大规模训练 token 数为：
$$
N_{\text{large}}
=
1T
$$
下采样比例：
$$
\rho
=
\frac{N_{\text{small}}}{N_{\text{large}}}
=
\frac{10 \times 10^9}{10^{12}}
=
0.01
$$
将所有 source 的 token 数按 $\rho$ 下采样：
$$
N_s'
=
\rho N_s
$$
对应代码：

```
ratio = small_run_tokens / large_run_tokens
downsampled_source_token_counts = {
    s: count * ratio
    for s, count in source_token_counts.items()
}
```

这样，如果某个 mixture 在大规模训练中会导致小高质量 source 被过度 epoch，那么在下采样的小实验中也会暴露出来。原报告也明确解释了这一点。

### 5.8 学习提示

本节最重要的公式是：
$$
E_s
=
\frac{p(s)N_{\text{train}}}{N_s}
$$
它告诉我们：数据混合不是只看比例，而是看比例、训练 token 数和 source 大小共同决定的 epoch 数。

------

## 6. 第五部分：后训练数据——从 task-agnostic 到 task-dependent

这一节要解决的问题是：**预训练数据通常是通用文本，而 SFT、reasoning、agentic coding 等后训练需要更任务相关的数据；这些数据如何构造？**

### 6.1 问题引入

前面四部分主要处理 pretraining / mid-training 数据：网页、代码、书籍、论文、数学文本等。这些数据大多是 task-agnostic。后训练数据则更像评测任务：它包含环境、任务提示、目标回答或轨迹。

`lecture_14.py` 给出的 recipe 是：

1. Define a set of environments；
2. Define a set of tasks / prompts；
3. Collect responses from a strong model teacher。

原报告也保留了这三步。

形式化地，可以写成：
$$
e \sim \mathcal{E}
$$
其中 $e$ 是环境，例如 GitHub repo、数学题库、科学问答集合。

然后构造训练样本：
$$
(e, q, a)
$$
其中：

- $e$：环境；
- $q$：任务或 prompt；
- $a$：teacher 或人类给出的回答 / 轨迹。

### 6.2 OpenThoughts：数学、科学与代码推理数据

OpenThoughts 的目标是构造高质量 reasoning post-training 数据。课程讲义写到：

- 使用 QwQ-32B 作为 teacher；
- 得到 1.2M examples；
- 问题来自 27 个 human 和 synthetic sources，例如 StackExchange、NuminaMath、Chemistry 等。

原报告还保留了字幕中的经验发现：多来源并不一定越多越好；每个 prompt 采样多个 responses 有帮助；更强模型不一定是更好 teacher；QwQ-32B 在该设置下比 DeepSeek-R1 更适合作 teacher；basic answer filtering 没有帮助。

**学习提示**：后训练数据不只是“答案越多越好”。teacher 是否适合学生模型学习、推理轨迹是否清晰、prompt 来源是否有效，都会影响最终效果。

### 6.3 SWE-smith：从 GitHub repo 自动生成软件工程任务

SWE-smith 面向 agentic coding。它不是让模型写一个孤立函数，而是让模型在真实代码仓库环境中完成软件工程任务。

课程讲义写到：给定 repository，用语言模型生成 tasks，例如引入 bug；128 个 GitHub repositories 产生 50K tasks。原报告也将这类数据定义为 semi-synthetic：
$$
\text{真实环境} + \text{合成任务}
$$
也就是 repo 真实，任务由模型生成。

### 6.4 SWE-Zero：不依赖执行反馈的大规模 agent trajectories

软件工程任务的难点是环境复杂：

- repo 依赖很重；
- Docker image 难构建；
- 旧 PR 需要回滚环境；
- 很多 repo 根本跑不起来。

SWE-Zero 的关键观察是：强模型即使没有 execution feedback，也能解决不少 SWE tasks。课程讲义写到 SWE-Zero 生成 300K 不需要 repository-specific execution 的 agent trajectories，来自 150K GitHub PRs，并使用 OpenHands scaffold，同时移除未来 git commits 防止 agent “git hacking”。

字幕进一步说明，SWE-Zero prompt 限制模型不能运行 Python code，只能用 `sed`、`grep` 等基本操作；之后从大 Qwen 模型蒸馏，并过滤掉仍试图执行代码的轨迹。【字幕】

**工程直觉**：如果每个 repo 都要完整配置环境，数据规模会被基础设施成本限制。SWE-Zero 的路线是：牺牲一部分执行真实性，换取大规模轨迹数据。

### 6.5 SWE-rebench 与 SWE-ZERO-12M-trajectories

课程讲义还提到：

- **SWE-rebench**：21K interactive Python SWE tasks，来自 3.4K GitHub repositories，450K PRs from GitHub and GitHub Archive；
- **SWE-ZERO-12M-trajectories**：将 SWE-Zero 扩展到 12M agent trajectories，使用 SWE-rebench-v2 tasks，包括 32K executable tasks 和 120K nonexecutable tasks，并使用 mini-coder-1.7B 与 mini-swe-agent scaffold。

原报告也保留了这些数字与说明。

### 6.6 学习提示

后训练数据的关键不是“从网页里抽文本”，而是构造：
$$
\text{环境}
+
\text{任务}
+
\text{高质量响应 / 轨迹}
$$
这与评测非常相似：好的后训练数据往往看起来像模型将来要解决的任务。

------

## 7. Python 代码示例详解

### 7.1 文件整体结构

Lecture 14 对应的 GitHub 文件是 `lecture_14.py`。仓库说明 CS336 的可执行讲义命名为 `lecture_XX.py`，且文件列表包含 `lecture_14.py`。

`lecture_14.py` 的主结构如下：

| 函数                           | 课程主题   | 作用                                                       |
| ------------------------------ | ---------- | ---------------------------------------------------------- |
| `main()`                       | 课程总控   | 按顺序调用五个主题函数                                     |
| `transformation()`             | 数据转换   | 解释 HTML / PDF / code directories 如何变成文本            |
| `filtering()`                  | 数据过滤   | 定义 target $T$、raw $R$、scoring function 和应用          |
| `deduplication()`              | 数据去重   | 组织 exact duplicates、near duplicates 和设计空间          |
| `hash_functions()`             | 哈希基础   | 解释 hash function、collision、MurmurHash                  |
| `exact_deduplication()`        | 精确去重   | 用 hash + sort + groupby 保留一个副本                      |
| `jaccard_minhash()`            | 近似去重   | 定义 Jaccard、MinHash，并用随机 hash 估计相似度            |
| `locality_sensitive_hashing()` | LSH        | 用 banding 结构把相似度变成近似阈值判断                    |
| `data_mixing()`                | 数据混合   | 解释 mixture、epoching、UniMax、RegMix、simulated epoching |
| `post_training_data()`         | 后训练数据 | 列举 OpenThoughts、SWE-smith、SWE-Zero 等数据构造方式      |

原报告也用表格总结了这些代码片段和课程知识点的对应关系。

------

### 7.2 关键函数逐段解析

#### 7.2.1 `transformation()`

**输入 / 输出**：这是讲义展示函数，没有传统意义上的函数输入输出；它通过 `text(...)`、`image(...)`、`link(...)` 展示课程内容。

**作用**：说明 raw data 不天然是文本，而是 HTML、PDF 或代码目录；解释 HTML-to-text、FinePDFs、OCR、cleanup、filtering 等处理。

**对应课程概念**：data transformation、boilerplate removal、linearization、PDF OCR。

**学习重点**：理解 transformation 是后续 filtering 和 training 的前提。错误转换会把噪声带入整个训练管线。

#### 7.2.2 `filtering()`

**作用**：将过滤抽象为：给定 target data $T$ 和 raw data $R$，寻找 $R$ 中与 $T$ 相似的子集。

**关键变量 / 概念**：

| 变量 / 概念 | 含义                           |
| ----------- | ------------------------------ |
| $T$         | 目标数据，表示“好数据长什么样” |
| $R$         | 海量原始数据                   |
| $T'$        | 从 $R$ 中筛出的候选目标子集    |
| `score(x)`  | 过滤分数                       |
| `threshold` | 保留阈值                       |
| KenLM       | 生成式目标域模型               |
| fastText    | 快速判别式分类器               |

**对应公式**：
$$
x \in T'
\Longleftrightarrow
s(x) \ge \tau
$$
**代码服务的课程问题**：如何把“质量”或“目标领域”转化为可执行的大规模筛选规则。

#### 7.2.3 `exact_deduplication()`

**作用**：演示用 hash 做 exact deduplication。

**核心逻辑**：

```
hash_items = itertools.groupby(
    sorted(items, key=mmh3.hash),
    key=mmh3.hash
)
deduped_items = [next(group) for h, group in hash_items]
```

**对应理论**：
$$
h(x)=h(y)
\Rightarrow
x,y \text{ 可能是同一个 item}
$$
在教学例子中，hash 相同的字符串被分到同一组，每组保留一个。

**学习重点**：这种方法高效、可并行，但只适合 exact duplicates，不能识别近似重复。

#### 7.2.4 `jaccard_minhash()`

**作用**：从 Jaccard similarity 开始，引出 MinHash。

**关键代码**：

```
def compute_jaccard(A, B):
    intersection = len(A & B)
    union = len(A | B)
    return intersection / union
```

对应公式：
$$
J(A,B)
=
\frac{|A \cap B|}{|A \cup B|}
$$
MinHash 代码：

```
def minhash(S: set[str], seed: int):
    return min(mmh3.hash(x, seed) for x in S)
```

对应性质：
$$
P[\text{MinHash}(A)=\text{MinHash}(B)]
=
J(A,B)
$$
**学习重点**：MinHash 把集合相似度估计变成哈希碰撞统计。

#### 7.2.5 `locality_sensitive_hashing()`

**作用**：解释如何通过 $b$ 个 bands 和每个 band 中 $r$ 个 hash functions，使碰撞概率出现阈值化效果。

**核心函数**：

```
def get_prob_collision(sim, b, r):
    prob_match = sim ** r
    prob_collision = 1 - (1 - prob_match) ** b
    return prob_collision
```

对应公式：
$$
P_{\text{collision}}(s;b,r)
=
1-(1-s^r)^b
$$
**学习重点**：LSH 的 banding 结构把“相似度越高越容易碰撞”变成“超过阈值后高概率碰撞”。

#### 7.2.6 `data_mixing()`

**作用**：解释多数据源混合、epoching、UniMax、regression-based mixing 和 simulated epoching。

**关键公式**：
$$
E_s
=
\frac{p(s)N_{\text{train}}}{N_s}
$$
**关键代码**：

```
ratio = small_run_tokens / large_run_tokens
downsampled_source_token_counts = {
    s: count * ratio
    for s, count in source_token_counts.items()
}
```

**学习重点**：数据混合的难点不是设置一个概率分布，而是控制每个 source 的重复训练次数。

------

### 7.3 代码与课程知识点对应表

| 课程知识点         | 对应代码函数 / 变量 / 片段                    | 代码功能                               | 对应理论原理                   | 在课程中的作用                   |
| ------------------ | --------------------------------------------- | -------------------------------------- | ------------------------------ | -------------------------------- |
| 数据管线主线       | `main()`                                      | 调用五个主题函数                       | 数据从转换到后训练的流水线     | 帮助学生把课程结构串起来         |
| 原始数据转换       | `transformation()`                            | 展示 HTML、PDF、code directories       | 原始格式到 token 序列          | 说明训练前必须预处理             |
| 过滤框架           | `filtering()`                                 | 定义 target $T$、raw $R$、score        | 相似性筛选                     | 统一语言识别、质量过滤、毒性过滤 |
| 生成式过滤         | `score(x)=p_T(x)`、KenLM                      | 用目标域 LM 打分                       | 低 perplexity 表示更像目标域   | 解释 OpenMathText                |
| 判别式过滤         | `score(x)=p(T \mid x)`、fastText              | 训练正负样本分类器                     | 二分类概率                     | 解释 GPT-3 / LLaMA / phi-1       |
| 精确去重           | `mmh3.hash`、`groupby`                        | 按 hash 分组保留一个                   | exact match                    | 快速去除完全重复                 |
| Jaccard            | `compute_jaccard(A,B)`                        | 计算交并比                             | 集合相似度                     | 定义 near duplicate              |
| MinHash            | `minhash(S, seed)`                            | 用碰撞频率估计 Jaccard                 | $P[h(A)=h(B)]=J(A,B)$          | 近似去重基础                     |
| LSH                | `get_prob_collision(sim,b,r)`                 | 计算 banding 碰撞概率                  | $1-(1-s^r)^b$                  | 将相似度阈值化                   |
| 数据混合 epoch     | `low_num_epochs`、`high_num_epochs`           | 计算 source epoch 数                   | $E_s=p(s)N/N_s$                | 揭示小数据源过采样               |
| Simulated epoching | `ratio = small_run_tokens / large_run_tokens` | 下采样 source sizes                    | 小规模模拟大规模数据稀缺       | 改善 mixture 迁移                |
| 后训练数据         | `post_training_data()`                        | 展示 OpenThoughts、SWE-smith、SWE-Zero | 环境 + 任务 + teacher response | 连接 SFT / agent 训练            |

------

## 8. PDF / 幻灯片 / 图表解析

### 8.1 图表来源说明

Lecture 14 没有单独 PDF；图表主要由 `lecture_14.py` 中的 `image(...)` 调用插入。GitHub 原始代码中包含 `images/dclm-wet.png`、`images/raw-target-schema.png`、`images/data-filtering-scale.png`、`images/regmix.png`、`images/data-mixing-methods.png`、`images/openthoughts-sources.png`、`images/openthoughts-pipeline.png`、`images/swe-smith.png`、`images/swezero-noexec.png`、`images/swezero-prompt.png`、`images/swezero-results.png`、`images/swe-rebench.png` 等图像调用。

由于原报告未提供这些图像的完整坐标轴与视觉细节，以下解析侧重说明它们在课程中的功能，而不伪造具体图中未确认的信息。

### 8.2 主要图表总览

| 图表 / 代码调用                             | 展示内容                                   | 支持的课程结论                                    | 对应章节 |
| ------------------------------------------- | ------------------------------------------ | ------------------------------------------------- | -------- |
| `dclm-wet.png`                              | HTML-to-text / Web extraction 相关质量问题 | transformation accuracy matters                   | 数据转换 |
| `raw-target-schema.png`                     | raw data 与 target data 的过滤框架         | filtering 是从 raw $R$ 中找类似 target $T$ 的子集 | 数据过滤 |
| `data-filtering-scale.png`                  | filtering 的 scale-dependent effect        | 阈值没有单一最优，取决于训练规模                  | 数据过滤 |
| LSH 曲线图像                                | 不同 $b,r$ 下碰撞概率变化                  | banding 让相似度产生近似阈值                      | 数据去重 |
| `marin-token-viewer.png`                    | 多来源 token 规模                          | 训练数据由多个 sources 构成                       | 数据混合 |
| `regmix.png`                                | regression-based mixing 流程               | 小模型试 mixture，再拟合 mixture-to-loss          | 数据混合 |
| `data-mixing-methods.png`                   | 多种 mixing 方法对比                       | 不同策略有不同权衡                                | 数据混合 |
| `openthoughts-sources.png`                  | OpenThoughts 数据来源                      | reasoning 数据由多来源 prompts 构成               | 后训练   |
| `openthoughts-pipeline.png`                 | OpenThoughts 构造流程                      | teacher responses 与采样策略重要                  | 后训练   |
| `swe-smith.png`                             | repo → synthetic SWE tasks                 | semi-synthetic coding tasks                       | 后训练   |
| `swezero-noexec.png` / `swezero-prompt.png` | 无执行反馈的 SWE-Zero 设定                 | 减少环境构建成本                                  | 后训练   |
| `swezero-results.png`                       | SWE-Zero 效果展示                          | teacher trajectory 可用于蒸馏                     | 后训练   |
| `swe-rebench.png`                           | SWE-rebench 任务来源                       | 真实 PR 和 repo 可构造交互式任务                  | 后训练   |

### 8.3 关键图表阅读方式

#### 8.3.1 Filtering schema 图

这类图想回答的问题是：**少量高质量 target data 如何帮助我们筛选海量 raw data？**

读图时应关注：

- raw data $R$：大而杂；
- target data $T$：小而高质量；
- scoring function $s(x)$：把“像不像目标数据”变成分数；
- threshold $\tau$：决定保留多少数据。

容易误解的地方：target data 不是最终训练全集，而是过滤器学习“什么叫好数据”的参照物。

#### 8.3.2 Data filtering scale 图

这类图想说明：**过滤阈值的好坏依赖训练规模。**

读图重点不是某个固定阈值，而是观察随着训练 token 增加：

- 过高阈值会让数据太少，重复 epoch；
- 过低阈值会引入更多低质量数据；
- 最优过滤强度随训练规模变化。

#### 8.3.3 LSH 曲线图

LSH 图想说明：**通过 banding，可以把平滑的相似度碰撞概率变成近似阈值函数。**

读图时关注：

- 横向通常对应相似度 $s=J(A,B)$；
- 纵向对应碰撞概率；
- 增大 $r$：更严格，曲线向右；
- 增大 $b$：更宽松，曲线向左。

#### 8.3.4 RegMix 图

RegMix 图想说明：**不能直接在大模型上试所有 mixture，所以先用小模型实验拟合 mixture-to-loss。**

读图顺序应是：
$$
\text{sample mixtures}
\rightarrow
\text{train proxy models}
\rightarrow
\text{evaluate losses}
\rightarrow
\text{fit regression}
\rightarrow
\text{optimize mixture}
\rightarrow
\text{train large model}
$$
读图时最容易忽略的是：小模型最优 mixture 不一定能直接迁移到大模型，因此需要考虑 scale-dependent effects 和 simulated epoching。

------

## 9. 关键公式与推导汇总

| 公式                                              | 符号解释                                       | 适用场景                   | 直觉解释                     | 代码 / 例子对应              |
| ------------------------------------------------- | ---------------------------------------------- | -------------------------- | ---------------------------- | ---------------------------- |
| $x \in T' \Longleftrightarrow s(x)\ge \tau$       | $x$：候选样本；$s(x)$：分数；$\tau$：阈值      | 过滤                       | 分数高于门槛就保留           | `filtering()`                |
| $s_{\text{gen}}(x)=p_T(x)$                        | $p_T$：目标域 LM                               | 生成式过滤                 | 越像目标数据，概率越高       | KenLM                        |
| $s_{\text{clf}}(x)=P(y=1\mid x)$                  | $y=1$：正样本                                  | 判别式过滤                 | 分类器判断是否像好数据       | fastText                     |
| (\text{NLL}(x)=-\frac1n\sum_i\log p_T(w_i         | \text{history}))                               | $w_i$：词；history：上下文 | n-gram / LM filtering        | 目标模型越容易预测，NLL 越低 |
| (J(A,B)=\frac{                                    | A\cap B                                        | }{                         | A\cup B                      | })                           |
| $\text{MinHash}_h(S)=\min_{x\in S}h(x)$           | $S$：集合；$h$：随机 hash                      | MinHash                    | 看随机排列中最先出现的元素   | `minhash`                    |
| $P[\text{MinHash}(A)=\text{MinHash}(B)] = J(A,B)$ | 碰撞概率等于 Jaccard                           | 估计相似度                 | 越相似越容易碰撞             | 100 seeds 估计               |
| $P_{\text{collision}}(s;b,r)=1-(1-s^r)^b$         | $s$：Jaccard；$b$：bands；$r$：每 band hash 数 | LSH                        | band 内 AND，band 间 OR      | `get_prob_collision`         |
| $\tau \approx (1/b)^{1/r}$                        | $b,r$：LSH 参数                                | LSH 相变                   | 碰撞概率快速上升的位置       | $b=20,r=450$                 |
| $E_s=\frac{p(s)N_{\text{train}}}{N_s}$            | $p(s)$：source 权重；$N_s$：source 大小        | 数据混合                   | 一个 source 被重复训练多少次 | high source 50 epochs        |
| $\rho=\frac{N_{\text{small}}}{N_{\text{large}}}$  | 小 / 大训练规模比例                            | simulated epoching         | 让小实验模拟大训练的数据稀缺 | source downsampling          |

原报告也集中列出了这些公式，作为本讲核心数学结构。

------

## 10. 概念对比与易混淆点

| 易混淆概念                             | 区别                                                         | 联系                               | 常见误解                      | 正确理解                               |
| -------------------------------------- | ------------------------------------------------------------ | ---------------------------------- | ----------------------------- | -------------------------------------- |
| Transformation vs. Filtering           | Transformation 改格式；Filtering 选数据                      | Filtering 依赖 transformation 输出 | 以为转换后就能训练            | 转换后仍需质量筛选                     |
| Boilerplate removal vs. 内容过滤       | 前者去网页模板；后者按质量 / 语言 / 毒性筛选                 | 都减少噪声                         | 把 boilerplate 当唯一噪声     | 噪声有格式噪声、内容噪声、重复噪声     |
| 生成式过滤 vs. 判别式过滤              | 生成式估计 $p_T(x)$；判别式估计 $P(T\mid x)$                 | 都得到 score                       | 以为 fastText 一定比 KenLM 弱 | 关键是任务、速度、目标数据             |
| 语言识别 vs. 质量过滤                  | 语言识别判断语言；质量过滤判断目标价值                       | 都可用分类器                       | 把英文都当好数据              | 英文网页也可能低质                     |
| Exact duplicate vs. Near duplicate     | 完全相同 vs. 高度相似                                        | 都会浪费训练                       | 只做 exact dedup 就够         | Web 数据大量 near duplicate            |
| Hash vs. MinHash                       | Hash 用于精确匹配；MinHash 用于相似度估计                    | 都依赖哈希                         | 以为哈希碰撞总是坏事          | MinHash 中碰撞是有意设计的             |
| Jaccard vs. LSH                        | Jaccard 是相似度定义；LSH 是查找相似项的方法                 | LSH 依赖 Jaccard / MinHash         | 以为 LSH 直接算精确相似度     | LSH 是近似候选生成方法                 |
| $b$ vs. $r$                            | $b$ 增加匹配机会；$r$ 增加匹配难度                           | 共同塑造 S 曲线                    | 只调一个参数                  | $b,r$ 要一起看                         |
| Uniform mixing vs. Proportional mixing | 前者每源等概率；后者按 token 数                              | 都是 baseline                      | 以为 proportional 一定合理    | 大低质源会占太多预算                   |
| Mixture weight vs. Epoch count         | weight 是采样概率；epoch count 是重复训练次数                | $E_s=p(s)N/N_s$                    | 只看 $p(s)$                   | 小数据源即使权重不大也可能被重复很多次 |
| RegMix vs. Simulated epoching          | RegMix 拟合 mixture-to-loss；simulated epoching 调整小实验数据规模 | 可结合使用                         | 以为小模型最优可直接迁移      | 需要模拟大规模 epoch 行为              |
| Synthetic vs. Semi-synthetic data      | synthetic 全部合成；semi-synthetic 真实环境 + 合成任务       | 都用于后训练                       | 合成数据一定低质量            | teacher、环境和过滤决定质量            |

------

## 11. 学习拓展与发散理解

### 11.1 与大模型训练实践的关系

**延伸理解：数据管线决定模型的“训练世界”。**

本讲内容在大模型训练中对应以下实践问题：

| 实践环节   | 本讲对应内容                            | 影响                                                  |
| ---------- | --------------------------------------- | ----------------------------------------------------- |
| 数据准备   | transformation                          | 决定 raw data 是否变成可用文本                        |
| 训练质量   | filtering                               | 决定模型学到的是高质量知识还是低质噪声                |
| 训练效率   | deduplication                           | 避免 GPU 重复训练相同内容                             |
| 记忆风险   | deduplication / decontamination         | 降低模型背诵训练数据和污染测试集                      |
| 能力结构   | data mixing                             | 决定模型更擅长百科、代码、数学还是长尾语言            |
| 过拟合控制 | epoch-aware mixing                      | 避免小数据源被过度重复                                |
| 后训练能力 | post-training data                      | 决定模型是否会推理、写代码、使用工具、完成 agent 任务 |
| 工程成本   | PDF OCR、repo environment、filter speed | 决定数据管线是否能在真实规模运行                      |

### 11.2 与其他 lecture 的联系

**补充说明**：不具体编造其他 lecture 内容的前提下，可以把本讲放入 CS336 的整体主线中理解：

- 与前面关于数据来源的 lecture 相连：上一讲关注数据从 live services 到 dump / crawl，本讲关注 dump / crawl 之后如何处理；
- 与模型训练 lecture 相连：训练损失、优化器和模型结构只有在数据质量足够好时才有意义；
- 与 scaling law 相连：数据数量、质量、重复度和混合比例都会影响 loss scaling；
- 与 evaluation 相连：decontamination 和后训练数据构造都与评测任务高度相关；
- 与 systems 相连：Web-scale filtering、deduplication、OCR、repo 环境构建都依赖工程系统能力。

### 11.3 面试 / 考试 / 作业可能考察角度

| 问题                               | 考察点                   | 回答思路                                                     |
| ---------------------------------- | ------------------------ | ------------------------------------------------------------ |
| 为什么 raw web data 不能直接训练？ | transformation           | HTML / PDF / 代码目录不是干净文本，需要去 boilerplate、OCR、linearization |
| Filtering 的统一数学框架是什么？   | 目标数据与 raw data      | 给定 $T,R$，训练 score $s(x)$，保留 $s(x)\ge\tau$            |
| KenLM 和 fastText 过滤有什么区别？ | 生成式 vs 判别式         | KenLM 学 $p_T(x)$，fastText 学 $P(T\mid x)$                  |
| 为什么过滤阈值没有单一最优？       | scale-dependent effect   | 训练越久越需要更多数据，否则高质量小集会被重复 epoch         |
| 为什么 exact dedup 不够？          | near duplicate           | 格式变化、模板文本、轻微改写不会被 exact match 捕捉          |
| Jaccard similarity 如何定义？      | 集合相似度               | (                                                            |
| MinHash 为什么能估计 Jaccard？     | 随机排列推导             | 最小元素落在交集的概率等于 Jaccard                           |
| LSH 的 $b,r$ 分别有什么作用？      | banding                  | $r$ 使匹配更难，$b$ 增加匹配机会                             |
| 数据混合为什么要看 epoch？         | source size 与训练 token | $E_s=p(s)N_{\text{train}}/N_s$                               |
| UniMax 解决什么问题？              | 小数据源过采样           | 对每个 source 设置 epoch cap                                 |
| RegMix 的风险是什么？              | 小到大迁移               | 回归模型可能外推错误，小模型最优不一定适合大模型             |
| SWE-Zero 为什么不依赖执行反馈？    | 后训练数据工程           | 避免 repo 环境构建成本，用强模型生成可蒸馏轨迹               |

------

## 12. 课程复习总结

### 12.1 本讲核心思想

Lecture 14 的核心思想是：**训练数据不是越多越好，而是要通过系统数据管线，把原始、异构、嘈杂、重复、多来源的数据变成高质量、多样、低重复、比例合理并适合训练目标的数据。**

这条主线贯穿五步：
$$
\text{Transformation}
\rightarrow
\text{Filtering}
\rightarrow
\text{Deduplication}
\rightarrow
\text{Mixing}
\rightarrow
\text{Post-training Data}
$$
原报告最后也将本讲总结为：先把 HTML/PDF/代码等原始数据转换为文本，再用目标数据训练快速过滤器，随后用哈希、Jaccard、MinHash 和 LSH 做 Web-scale 去重，接着通过 epoch-aware 的混合策略平衡多源数据，最后用真实、半合成或全合成任务与 teacher 模型构造后训练数据。

### 12.2 必须掌握的概念

- **Data transformation**：把 HTML、PDF、代码目录转换为文本 / token 序列。
- **Boilerplate removal**：移除导航栏、广告、页眉页脚等重复网页模板。
- **Linearization**：将层次结构和视觉布局压平成一维序列。
- **Filtering**：从 raw data $R$ 中筛出类似 target data $T$ 的子集。
- **KenLM / generative filtering**：用目标域 LM 判断文本像不像目标数据。
- **fastText / discriminative filtering**：训练分类器估计 $P(T\mid x)$。
- **Scale-dependent threshold**：过滤阈值随训练 token 数变化。
- **Exact deduplication**：用 hash 删除完全重复内容。
- **Jaccard similarity**：用集合交并比定义近似重复。
- **MinHash**：用碰撞概率估计 Jaccard。
- **LSH**：用 banding 结构近似查找高相似文档。
- **Data mixture**：在多个 sources 上定义采样分布。
- **Epoching**：某个 source 被重复训练的次数。
- **UniMax**：给 source 设置 epoch cap。
- **Regression-based mixing**：用小模型实验拟合 mixture-to-loss。
- **Simulated epoching**：让小实验模拟大训练的数据稀缺。
- **Post-training data**：环境 + 任务 + teacher response / trajectory。

### 12.3 必须理解的公式

1. 过滤保留规则：

$$
x \in T'
\Longleftrightarrow
s(x)\ge\tau
$$

它解决“哪些样本保留”的问题。

1. Jaccard similarity：

$$
J(A,B)
=
\frac{|A\cap B|}{|A\cup B|}
$$

它解决“两个文档有多像”的问题。

1. MinHash 性质：

$$
P[\text{MinHash}(A)=\text{MinHash}(B)]
=
J(A,B)
$$

它解决“如何用哈希估计相似度”的问题。

1. LSH 碰撞概率：

$$
P_{\text{collision}}(s;b,r)
=
1-(1-s^r)^b
$$

它解决“如何把相似度变成阈值式候选查找”的问题。

1. 数据源 epoch 数：

$$
E_s
=
\frac{p(s)N_{\text{train}}}{N_s}
$$

它解决“某个 source 是否会被过度重复训练”的问题。

1. Simulated epoching 下采样比例：

$$
\rho
=
\frac{N_{\text{small}}}{N_{\text{large}}}
$$

它解决“小规模 mixture 实验如何模拟大规模训练”的问题。

### 12.4 必须看懂的代码

| 代码片段                             | 学习重点                   |
| ------------------------------------ | -------------------------- |
| `main()` 调用五个主题函数            | 理解课程顺序               |
| `filtering()` 中 $T,R,T'$ 框架       | 理解过滤抽象               |
| `keep_document(score)`               | 理解按分数随机 / 阈值保留  |
| `mmh3.hash(...)`                     | 理解哈希用于 exact dedup   |
| `groupby(sorted(...))`               | 理解 MapReduce-style 去重  |
| `compute_jaccard(A,B)`               | 理解交并比                 |
| `minhash(S, seed)`                   | 理解随机哈希估计 Jaccard   |
| `get_prob_collision(sim,b,r)`        | 理解 LSH banding 概率      |
| `low_num_epochs` / `high_num_epochs` | 理解 mixture 与 epoch 风险 |
| `downsampled_source_token_counts`    | 理解 simulated epoching    |
| `post_training_data()`               | 理解后训练数据构造 recipe  |

### 12.5 一句话总结

Lecture 14 讲的是：**大模型训练数据的质量来自一整套数据工程与算法管线——先把复杂原始数据转换成文本，再用快速过滤器筛质量，用哈希、MinHash 和 LSH 去重，用 epoch-aware mixture 平衡多源数据，最后用真实、半合成和合成任务构造后训练数据；整节课的核心是数据质量、多样性、规模、计算效率和过拟合风险之间的系统权衡**