# CS336 Assignment 4: Data 中文结构化报告

> 视角：CS336 课程助教 + LLM 数据工程师 + 数据质量审查者。  
> 目标：把 Assignment 4 当成一个真实的 LLM pretraining data pipeline 项目来理解，而不是仅仅为了通过单元测试。  
> 约束：本报告只做结构化解析、实现路线、测试边界和数据工程解释；不提供可直接提交的实现代码。

---

## 目录

1. [Assignment 基本信息](#1-assignment-基本信息)
2. [PDF / Handout 结构化解析](#2-pdf--handout-结构化解析)
3. [代码结构解析](#3-代码结构解析)
4. [前置知识](#4-前置知识)
5. [推荐完成顺序](#5-推荐完成顺序)
6. [学习路线](#6-学习路线)

---

# 1. Assignment 基本信息

## 1.1 Assignment 名称

**CS336 Assignment 4 (data): Filtering Language Modeling Data**  
版本：**26.0.1**  
课程时间：**Spring 2026**

这个作业的核心不是训练一个全新的模型架构，而是构建和评估一个面向语言模型预训练的数据处理流水线：从 Common Crawl 网页数据出发，经过文本抽取、语言过滤、隐私与安全过滤、质量过滤、去重、tokenization，最终用固定训练脚本训练 GPT-2 small-shaped 模型，并在 Paloma C4 100 domains validation subset 上比较验证损失。

## 1.2 主题方向

主题可以概括为：**Web-scale pretraining data filtering and evaluation**。

它覆盖真实 LLM 数据工程中的几个关键环节：

- **Web data extraction**：从 HTML/WARC 中抽取可用于训练的纯文本。
- **Filtering**：基于语言、PII、harmful content、启发式质量规则、质量分类器过滤数据。
- **Deduplication**：去除 exact line duplicates 和 fuzzy document duplicates。
- **Tokenization and serialization**：用 GPT-2 tokenizer 把文本序列化为 `np.uint16` binary file。
- **Data-centric model evaluation**：固定模型和训练过程，只优化训练数据，观察验证损失和生成质量。

## 1.3 最终目标

最终目标有两层：

第一层是通过 required tests，确保基础函数行为正确，包括 HTML extraction、language identification、PII masking、toxicity/NSFW classification、Gopher filters、quality classifier sanity check、exact dedup 和 MinHash dedup。

第二层更重要：**构造一个高质量训练语料，使固定训练脚本训练出的语言模型在 Paloma C4 100 domains validation data 上取得尽可能低的 validation loss / perplexity**。换言之，本作业把“数据质量”作为主要优化对象，而不是把精力放在改模型、改优化器或调训练脚本上。

## 1.4 与真实 LLM pretraining data pipeline 的关系

这个 assignment 是真实 LLM 预训练数据管线的压缩版：

```text
Common Crawl WARC / WET
        ↓
HTML decoding + text extraction
        ↓
manual inspection + data understanding
        ↓
language identification / English filtering
        ↓
PII masking
        ↓
harmful content filtering: NSFW + toxic speech
        ↓
quality filtering: Gopher rules + trained quality classifier
        ↓
exact line deduplication
        ↓
fuzzy document deduplication: MinHash + LSH
        ↓
large-scale WET processing with parallelism
        ↓
manual inspection of kept / discarded documents
        ↓
GPT-2 tokenization + append <|endoftext|>
        ↓
fixed Transformer LM training
        ↓
Paloma C4 validation loss / perplexity
        ↓
iterate on data pipeline
```

这与 C4、Dolma、RefinedWeb、FineWeb 等开放数据集的思路一致：原始网页数据本身并不等于训练数据，真正决定模型行为的是一系列 extraction、filtering、deduplication、mixture design 和 evaluation 决策。

## 1.5 重点代码目录

### `cs336_data/`

这是本作业主要让学生写代码的模块。当前 starter repo 里非常少，重点是让你自己组织 extraction、filtering、dedup、pipeline scripts。已有文件包括：

- `common.py`：决定共享数据路径，本地时用 `local-shared-data`，Modal 远端时用 `/shared-data`。
- `wet_files.py`：用于生成英文过滤后的 WET files，包含 `is_english` TODO，以及 `EnglishWetFiles` 数据生成逻辑。
- `modal_utils.py`：Modal app、volume、image 构建和 volume mounts；需要设置 `SUNET_ID`。

### `cs336_basics/`

包含 staff-provided 训练模型实现，是 Assignment 1 模型的优化版本。它不是本作业的主要实现区域，而是用于固定训练。主要文件包括：

- `data.py`：从 tokenized binary memmap 中采样 batch。
- `model.py`：Transformer LM，包括 RoPE、RMSNorm、SwiGLU、causal self-attention、generation、from_pretrained。
- `optimizer.py`：cosine learning-rate schedule。
- `train_config.py`：固定模型和训练超参数。
- `ddp_utils.py`：DDP 辅助函数。

### `tests/`

测试全部通过 `tests/adapters.py` 的 hook 调用你的实现。你不需要让测试直接 import 你的内部函数，而是要把 adapter 接到你的实现上。测试不是完整质量保证，只是行为边界和 sanity check。

### `tests/fixtures/`

包含小型 HTML、quality、dedup 样例，是理解隐藏约束的关键：

- `moby.html` / `moby_extracted.txt`：要求 extraction 输出完全匹配。
- `low_quality_cc.txt` / `high_quality_wiki_reference.txt`：质量分类器 sanity check。
- `documents_with_line_duplicates/` 与 `documents_line_deduplicated/`：exact line dedup 目标行为。
- `documents_with_fuzzy_duplicates/`：MIT license near-duplicate，用于 MinHash fuzzy dedup。

## 1.6 下载、训练、生成脚本

### 下载脚本：`scripts/download_data.py`

用于下载 offline-needed files 和 full data。

- offline data 包括：Paloma C4 validation binary、sample WARC/WET、fastText language ID model、Dolma hate speech / NSFW models。
- full data 包括：Wikipedia external URLs 和 English-filtered WET files。

学生路径通常依赖 `/shared-data` 和 Modal；非学生路径需要本地下载，并且 full WET download 前需要先实现 `wet_files.py` 里的 `is_english`。

### 训练脚本：`scripts/train.py`

训练脚本是固定的。你只传入 `--train-bin` 指向 GPT-2 tokenizer 处理后的 `np.uint16` binary training data。

最终 Modal 训练路径：

```sh
uv run modal run scripts/train.py --train-bin /root/data/your_data.bin
```

脚本会使用固定模型、固定优化器、固定 batch 和固定 validation set。它的功能是评估你的数据，而不是给你自由改训练过程。

### 生成脚本：`scripts/generate_with_gpt2_tok.py`

用于从训练好的 checkpoint 采样文本，帮助检查模型是否生成合理语言。它加载 GPT-2 tokenizer 和 `BasicsTransformerLM.from_pretrained(model_path)`，支持 prompt、temperature、top-k、max_new_tokens、num_samples。

## 1.7 哪些文件不应修改

严格来说，作业允许你组织自己的代码，但以下内容不应作为“提高成绩”的手段去修改：

- **不要修改训练逻辑**：README 和 handout 都强调 leaderboard / final run 必须使用固定训练脚本。不要改 `scripts/train.py`、模型结构或优化过程来降低 validation loss。
- **不要修改 validation data**：`/shared-data/tokenized_paloma_c4_100_domains_validation.bin` 是固定验证集，不要改；更不能把 validation text 复制进 training data。
- **不要依赖改 tests 通过**：可以临时写更多测试或本地改测试 debug，但最终实现应通过原始测试。
- **不要把核心 TODO 写进 AGENTS.md 所禁止的 AI-generated solution 形式**：本报告只分析，不替你写提交代码。
- **`modal_utils.py` 只做必要配置**：通常只设置 `SUNET_ID`；非学生 Modal 路线 README 提到需移除 shared data volume 的 `environment_name`。
- **`cs336_basics/` 通常视为 staff fixed implementation**：它的用途是固定训练和评估，不是优化对象。

## 1.8 提交和测试方式

### 测试

本地测试核心命令：

```sh
uv run pytest -v ./tests
```

针对单个问题可以用：

```sh
uv run pytest -k test_extract_text_from_html_bytes
uv run pytest -k test_identify_language
uv run pytest -k test_mask_emails
uv run pytest -k test_mask_phones
uv run pytest -k test_mask_ips
uv run pytest -k test_classify_nsfw
uv run pytest -k test_classify_toxic_speech
uv run pytest -k test_gopher
uv run pytest -k test_classify_quality
uv run pytest -k test_exact_line_deduplication
uv run pytest -k test_minhash_deduplication
```

### 打包

`test_and_make_submission.sh` 会：

1. 运行 `uv run pytest -v ./tests --junitxml=test_results.xml || true`；
2. 生成 `cs336-assignment4-submission.zip`；
3. 排除 `.venv`、cache、`.git`、`.bin`、`.pt`、`.warc.gz`、`.warc.wet.gz`、`.txt`、`.json`、log 等大文件或临时文件。

注意：脚本即使测试失败也会继续打包，因为用了 `|| true`。因此你不能只看 zip 是否生成，要看 pytest 输出和 `test_results.xml`。

### 提交内容

handout 要求提交：

- `writeup.pdf`：回答所有 written questions。
- `code.zip`：包含你写的代码。

---

# 2. PDF / Handout 结构化解析

## 2.1 Assignment overview

handout 开头明确：你要获得过滤 web crawls 并生成 language modeling data 的实践经验。要实现三类基础能力：

1. 把 Common Crawl HTML 转为文本。
2. 用 harmful content、PII、质量规则等方法过滤文本。
3. 对训练数据去重。

要运行的部分是训练语言模型，比较不同数据处理决策对性能的影响。代码组织上，`cs336_basics/*` 是训练模型，`cs336_data/*` 是你写作业代码，`tests/*.py` 是必须通过的测试，`README.md` 是环境和目录说明。

## 2.2 Common Crawl / WARC / WAT / WET

### Common Crawl 是什么

Common Crawl 是公开网页抓取语料。LLM 预训练经常以 Common Crawl 作为原始数据来源，但 Common Crawl 只是原料，不是干净训练集。

### WARC

WARC 是 **Web ARChive format**。它包含原始网页抓取数据：

- page IDs；
- URLs；
- request / response metadata；
- HTTP headers；
- server / crawl metadata；
- 原始网页内容，例如 HTML bytes。

WARC 最接近浏览器或 crawler 实际抓到的“原始页面”。它适合做自定义 extraction，因为你可以重新解析 HTML、过滤脚本、抽取主内容。

### WAT

WAT 是 **Web Archive Transformation**。它从 WARC 派生，主要存高层 JSON metadata，例如页面标题、链接列表等。这个 assignment 只是介绍它，不要求核心实现。

### WET

WET 是 **Web Extracted Text**。它包含从 HTML 中抽出的纯文本，但仍然带有 WARC-like record headers，例如 `Content-Length`。WET 更轻、更便于文本处理，但 extraction 已经由 Common Crawl 做过，质量不一定适合 LM。

### WARC 与 WET 的核心区别

| 维度 | WARC | WET |
|---|---|---|
| 内容 | 原始网页抓取，包括 HTML、headers、metadata | HTML 已抽取后的文本 |
| 信息量 | 最大，保留网页结构 | 较小，丢失标签结构、链接上下文、布局信息 |
| 适合任务 | 自定义 HTML extraction、网页级分析 | 大规模文本过滤、训练语料构建 |
| 风险 | 解析复杂、编码复杂、噪声大 | extraction 质量不可控，可能保留 boilerplate 或丢失正文 |
| 对训练影响 | 可做更精细清洗 | 更快但受 upstream extractor 决策限制 |

## 2.3 手动数据检查：`look_at_cc`

这个问题要求你先看 raw Common Crawl，不急着写代码。它的教学目的很重要：真实数据工程不能只靠单元测试，必须先看数据。

你要做的事情：

- 查看 sample WARC 的第一个网页。
- 记录 URL，判断是否仍可访问。
- 观察 raw HTML 是否能看出页面内容。
- 查看对应 WET record，比较文本抽取结果。
- 手动看 25 个 WET records，记录语言、域名、页面类型、质量备注。
- 回答看到第几个样本时遇到你认为 high-quality 的网页。

这一步会让你直观看到：很多网页不是正文文章，而是论坛导航、菜单、广告、登录页、模板页、404、商品列表或低质量自动生成页面。对 LLM 来说，这些内容可能会让模型学到网页模板和垃圾 token，而不是有用知识。

## 2.4 HTML text extraction

HTML extraction 的任务是：给定 raw HTML bytes，返回可训练的纯文本。

handout 建议使用：

- `resiliparse.extract.html2text.extract_plain_text` 做 HTML-to-text；
- `resiliparse.parse.encoding.detect_encoding()` 在 UTF-8 解码失败时做编码检测；
- `fastwarc.warc.ArchiveIterator` 遍历 WARC。

### 为什么 HTML extraction 会影响训练质量

HTML extraction 是整个 pipeline 的入口。如果 extraction 做差，后续过滤再强也只能修补一部分。

常见影响包括：

- **保留过多 boilerplate**：菜单、footer、copyright、cookie banner、登录提示会大量重复，模型会学到模板文本。
- **正文丢失**：过强 extraction 会删掉真正有价值的段落，让数据量和知识覆盖下降。
- **结构污染**：HTML 标签、导航列表、脚本残留会增加无意义 token，降低有效训练 token 比例。
- **编码错误**：乱码会让 tokenizer 产生异常 token 分布，影响训练稳定性和语言质量。
- **链接文字误判**：链接、按钮、tag 可能是有用分类信息，也可能只是噪声，取决于任务目标。

## 2.5 Language identification

Web 数据是多语言的。作业要求用 fastText `lid.176.bin` 做语言识别，返回语言 ID 和置信分数。

### 为什么 language filtering 重要

在有限 compute budget 下，如果目标是训练英文模型或优化 C4 English validation loss，混入大量非英文文本会：

- 稀释英文训练 token；
- 增加 tokenizer fragmentation；
- 让模型生成时出现语言混杂；
- 对低资源语言产生不稳定或偏见行为；
- 让下游产品在用户期望英文时输出其他语言。

但语言过滤也有风险：

- code-mixed 文档可能被误删；
- 短文本置信度不可靠；
- 低资源语言和非标准方言更容易被误判；
- 过强 English threshold 可能降低覆盖面。

因此 writeup 要通过人工检查估计 English fraction 和阈值，而不是盲目相信 classifier。

## 2.6 PII masking

PII masking 要替换三类信息：

- email address → `|||EMAIL_ADDRESS|||`
- phone number → `|||PHONE_NUMBER|||`
- IPv4 address → `|||IP_ADDRESS|||`

### 安全意义

网页中有大量可联系或识别个人的信息。如果原样训练，语言模型可能 memorise 并在生成时泄露真实个人信息。PII masking 通过替换敏感片段降低隐私泄漏风险。

### 误伤与漏检风险

PII regex 很难完美：

- **false positive**：把订单号、产品编号、日期、版本号、数学表达式误判为电话或 IP。
- **false negative**：漏掉变体格式，例如 `name [at] domain [dot] com`、国际电话号码、带 extension 的号码、IPv6、空格拆开的邮箱。
- **上下文丢失**：替换后句子可能失去语义，例如技术文档中的 IP 地址其实是教学示例。
- **replacement token 过多**：模型可能频繁生成 `|||EMAIL_ADDRESS|||` 这类 mask token，影响自然生成。

因此 PII masking 是安全与数据保真之间的 trade-off。

## 2.7 Harmful content classification

作业关注两类 harmful content：

- NSFW：pornography、profanity、disturbing content 等。
- toxic speech：rude、disrespectful、unreasonable language，可能使人离开讨论。

要求使用 Dolma project 发布的 fastText Jigsaw classifiers：

- `dolma_fasttext_nsfw_jigsaw_model.bin`
- `dolma_fasttext_hatespeech_jigsaw_model.bin`

### False positive / false negative 风险

Harmful content 过滤尤其不能只看单元测试。

**False positives** 可能删除：

- 新闻、历史、医学、法律材料中对有害内容的讨论；
- 反歧视语境中的引用；
- 少数群体 reclaim 的词；
- 文学作品中的粗口或暴力描写；
- 安全研究、内容审核研究材料。

**False negatives** 可能保留：

- 拼写变体、规避词、谐音词；
- 隐晦仇恨表达；
- 多语言 toxic content；
- 上下文依赖的骚扰内容；
- classifier 训练域外的 NSFW 页面。

如果过度过滤，模型会缺少理解和拒答 harmful content 的语境；如果过滤不足，模型可能模仿 toxic style。真实 pipeline 通常需要阈值调优、人工抽检、分层采样和下游安全评估。

## 2.8 Gopher quality filters

Gopher filters 是一组简单可解释的启发式质量规则。作业要求实现其中一部分：

| 规则 | 含义 | 训练影响 |
|---|---|---|
| word count < 50 或 > 100,000 删除 | 太短通常是导航/广告/碎片，太长可能是 dump、日志、拼接异常 | 控制样本粒度，减少极端文档 |
| mean word length 不在 3 到 10 字符之间删除 | 平均词长太短可能是符号/菜单，太长可能是乱码/拼接 token | 过滤乱码、代码化或异常文本 |
| 超过 30% 行以 `...` 结尾删除 | 大量省略号常见于截断列表、搜索结果、低质量 snippet | 减少不完整片段和预览页 |
| 少于 80% words 含至少一个字母删除 | 数字、符号、表格、乱码比例过高 | 保证自然语言比例 |

这些规则的优点是简单、快、可解释；缺点是无法判断语义质量。例如一篇短诗、代码文档、数学文章可能被误删；一篇很长但平庸的 SEO 文章可能通过。

## 2.9 Quality classifier

Gopher rules 只能捕捉表面特征。quality classifier 试图学习“像高质量参考页面”的文本。

handout 的思路：

- 正样本：从 Wikipedia 页面外链抓取的网页，假设这些链接更可信、更高质量。
- 负样本：随机 Common Crawl 页面，代表普通 web noise。
- 训练 fastText classifier，输出 quality score。
- 通过阈值在 precision 与 recall 之间权衡。

### 为什么需要正负样本

分类器需要对比才能学习决策边界。只有正样本，它不知道哪些模式只是“网页共同特征”；只有负样本，它不知道目标质量是什么。正样本提供目标分布，负样本提供背景噪声。二者质量会直接影响 classifier：如果正样本也包含垃圾，模型会学坏；如果负样本太容易，模型只学到浅层域名或长度差异，泛化不好。

## 2.10 Exact line deduplication

exact line dedup 的目标是删除语料中完全重复的行。典型重复包括：

- `Home` / `Menu` / `Login`；
- copyright footer；
- cookie banner；
- forum navigation；
- generated 404 text。

作业要求两遍处理：

1. 第一遍统计整个 corpus 每一行出现次数；为了省内存，用 line hash 做 key。
2. 第二遍重写每个文件，只保留出现次数为 1 的行。

### exact dedup 与 fuzzy dedup 的区别

exact line dedup 是**行级别、完全匹配**。如果两个页面都含同一句 header，它会从所有页面删除该行。它不能处理只差一年、作者名、空格、标点的近似重复。

## 2.11 MinHash fuzzy deduplication

fuzzy dedup 处理“几乎一样但不完全一样”的文档，例如不同项目的 MIT license。

### Jaccard similarity

把文档转成 word n-gram set 后，两个集合的 Jaccard similarity 是：

```text
|S ∩ T| / |S ∪ T|
```

如果 similarity 超过阈值，就认为是近似重复。

### MinHash 的作用

直接比较所有文档对是 O(n²)，不可扩展。MinHash 把 n-gram set 压缩成 fixed-length signature，并且两个 signature 相同列的比例近似 Jaccard similarity。

### LSH 的作用

LSH 把 signature 分成 bands。只要两个文档在某个 band 的 hash 完全相同，就进入 candidate duplicate bucket。这样不必比较所有文档对，只比较可能相似的候选。

### bands 与 recall / precision

在固定 `num_hashes` 下：

- bands 更多、每个 band 更短 → 更容易 bucket match → recall 高、precision 低；
- bands 更少、每个 band 更长 → 更难 match → precision 高、recall 低。

### normalization 的意义

handout 特别要求 fuzzy dedup 前 normalize：lowercase、去 punctuation、normalize whitespace、去 accents、NFD Unicode normalization。这能提高 recall，因为模板重复经常只差大小写、标点、空白或重音。

## 2.12 Large-scale filtering

在第 4 部分，你要处理 `/shared-data/english-wet-data` 下已经做过 English filtering 的 2,500 个 WET files。它们的 English probability 至少 70%。

目标不是实现某个固定 filter，而是自由组合你前面实现的 primitives，并报告每个 filter 保留/删除多少 examples。

建议关注：

- streaming 读取 WET，避免一次性加载所有数据；
- multiprocessing / concurrent.futures 并行处理多个 WET files；
- 分阶段记录统计：input docs、PII modified docs、toxicity removed docs、Gopher removed docs、quality removed docs、dedup removed docs；
- 保留 discarded examples 供人工检查；
- 估算 2,500 files 处理时间，并外推到整个 Common Crawl。

## 2.13 Filtered data inspection

这一步要求你不要只看 metrics，而要看样本文本：

- 随机取 5 个最终保留 examples，判断是否适合作为 C4 100 domains 目标的 LM training data。
- 随机取 5 个被过滤或修改的 WET examples，说明是哪一步删掉/修改，判断是否合理。
- 根据观察迭代 pipeline，并在 writeup 记录尝试。

这一步是数据工程中非常关键的 human-in-the-loop 过程。没有人工检查，validation loss 下降也可能来自数据泄漏、模板过拟合或分布偶然匹配。

## 2.14 Tokenization

作业要求用 GPT-2 tokenizer：

- 对 filtered data encode 成 integer token IDs；
- 每个 document 后 append GPT-2 end-of-sequence token `<|endoftext|>`；
- flatten 所有 IDs；
- 转成 `np.uint16`；
- 用 `ids_array.tofile(output_path)` 序列化。

### 为什么要 append `<|endoftext|>`

`<|endoftext|>` 是文档边界信号。没有它，训练时随机取连续 context 会把两个无关网页拼接成一个连续段落，模型会学到错误跨文档转移。加 EOS 能：

- 告诉模型一个文档结束；
- 降低跨网页拼接噪声；
- 与 GPT-2 tokenizer / vocab 约定一致；
- 让 generation 更容易学会何时停止。

### 为什么使用 `np.uint16`

GPT-2 vocab size 是 50,257，最大 token ID 小于 65,535，因此 `uint16` 足够，且比 `int32` 节省空间。训练脚本用 `np.memmap(..., dtype=np.uint16)` 读取，所以格式必须兼容。

## 2.15 Fixed training script

训练脚本固定的原因是：**隔离变量**。

如果允许修改模型结构、学习率、batch size、训练步数或 validation 过程，就无法判断 validation loss 的变化到底来自数据质量还是训练技巧。本作业的研究问题是：在相同模型和相同训练预算下，哪种数据处理 pipeline 能得到更好的 LM？

固定配置包括：

- vocab size：50,257；
- context length：512；
- d_model：1024；
- d_ff：3072；
- layers：24；
- heads：16；
- dtype：bfloat16；
- train batch size：每进程 128；
- train steps：16,384；
- eval interval：2,000；
- eval iterations：1,000；
- final Modal run：8 × B200 GPUs；
- 训练 token 规模约：8 × 128 × 512 × 16,384 ≈ 8.59B sampled tokens。

## 2.16 Paloma C4 validation evaluation

验证目标是 **Paloma benchmark 的 C4 100 domains subset**。作业提供了已经用 GPT-2 tokenizer tokenized 的验证集：

```text
/shared-data/tokenized_paloma_c4_100_domains_validation.bin
```

你可以查看 validation data 的样子，用它设计过滤器或分类器，但不允许把 validation 内容直接复制进 training data。数据泄漏会让验证损失失真，也违背 assignment 目标。

## 2.17 Problem / TODO 总表

作业总分按 handout 中列出的 Problem 加总为 **71 points**。

| Problem | 分值 | 需要实现 / 运行 | 人工检查与 writeup | 对应测试 | 隐含边界条件 |
|---|---:|---|---|---|---|
| `look_at_cc` | 4 | 不写函数；查看 sample WARC/WET | WARC 第一页 URL、可访问性、HTML 是否可读；WET extraction 质量；25 个 WET records 注释；第几个出现 high-quality | 无 | WET headers 不是正文；原始网页可能有 harmful content；不要只看一个样本就设计 pipeline |
| `extract_text` | 3 | HTML bytes → extracted text；adapter `run_extract_text_from_html_bytes` | 对单个 WARC 运行自己的 extractor，并与 WET 比较 2-3 句 | `tests/test_extract.py` | 输入 bytes 未必 UTF-8；输出要和 `moby_extracted.txt` 完全一致；换行、bullet、标题和 link text 都可能影响测试 |
| `language_identification` | 6 | text → `(language_id, score)`；adapter `run_identify_language` | 说明 langid 出错下游风险；抽检 20 个样本、报告错误、English fraction 和阈值 | `tests/test_langid.py` | fastText label 常带 `__label__` prefix；测试期望 `en` 和 `zh`；score 是 float 且 >0；短文本置信度可能不可靠 |
| `mask_pii` | 3 | adapters `run_mask_emails`、`run_mask_phone_numbers`、`run_mask_ips` | 说明 naive masking 风险；抽检 20 个被替换样本，报告 false positives / false negatives | `tests/test_pii.py` | replacement string 必须完全一致；已有 mask string 不计入替换；phone formats 包括纯 10 位和常见 US 格式；IPv4 不应吞掉句末标点 |
| `harmful_content` | 6 | NSFW 和 toxic classifiers；adapters `run_classify_nsfw`、`run_classify_toxic_speech` | 说明过滤风险；抽检 20 个样本，对比人工判断，报告 harmful fraction 和阈值 | `tests/test_toxicity.py` | 期望 label 是 `nsfw`/`non-nsfw`、`toxic`/`non-toxic`；测试只是 Jigsaw sanity check，不代表真实准确率 |
| `gopher_quality_filters` | 3 | adapter `run_gopher_quality_filter` 返回 bool | 抽检 20 个样本，对比规则与人工判断 | `tests/test_quality.py` 的 Gopher tests | 规则边界要精确；`>30%` ellipsis 才删；3 到 10 字符平均词长通常应视作 inclusive；word/tokenization 选择影响边界 |
| `quality_classifier` | 15 | 训练 quality classifier；adapter `run_classify_quality` | 用 classifier 进入后续 filtering；可说明样本构造和阈值 | `tests/test_quality.py::test_classify_quality` | sanity fixture 期望 low-quality CC label `cc`、high-quality wiki label `wiki`；score float >0；测试不证明分类器真实有效 |
| `exact_deduplication` | 3 | adapter `run_exact_line_deduplication(input_files, output_directory)` | 无单独 written inspection，但理解重复行对训练的影响 | `tests/test_deduplication.py::test_exact_line_deduplication` | 出现次数 >1 的行从所有文件删除，而不是保留一份；输出文件数等于输入；同 basename 写入 output dir；空文件也是合法输出 |
| `minhash_deduplication` | 8 | adapter `run_minhash_deduplication(...)` | 无单独 written inspection，但用于后续 data pipeline | `tests/test_deduplication.py::test_minhash_deduplication_*` | `num_hashes` 可被 `num_bands` 整除；word n-gram；需 normalization；候选对还要算 true Jaccard；duplicates 要聚类，transitive closure；每簇随机/任意保留一个 |
| `filter_data` | 6 | 编写并运行大规模 WET filtering script(s) | 报告每个 filter step 删除比例；报告 2,500 WET runtime 和外推到 entire CC runtime | 无 | 输入已 English p≥0.7，但仍需清洗；streaming 和 multiprocessing 很重要；不要把 validation data 复制进 train；记录 counts |
| `inspect_filtered_data` | 4 | 随机抽样 kept / removed examples | 5 个保留样本质量分析；5 个删除/修改样本原因和是否合理；报告 pipeline iteration | 无 | 需要保留审计日志或样本；只给 metrics 不够；excerpt 即可，不必贴全文 |
| `tokenize_data` | 2 | 编写 tokenization + serialization script | 报告 filtered dataset token count | 无 | GPT-2 tokenizer；每个 doc append `<|endoftext|>`；`np.uint16`；`tofile`；格式必须被 train.py memmap 读取 |
| `train_model` | 8 | 用 fixed `scripts/train.py` 训练 | 报告 best validation loss、learning curve、做了什么 | 无 | 不改训练脚本；8 B200 final run；验证集固定；模型约 430M；数据质量是唯一主要优化变量 |

---

# 3. 代码结构解析

## 3.1 `cs336_data/`

### `cs336_data/common.py`

作用很小但重要：定义 shared data path。

- `MODAL_SHARED_PATH = Path("/shared-data")`
- `get_shared_assets_path()`：
  - 如果 `modal.is_local()`，创建并返回本地 `local-shared-data`；
  - 否则返回 `/shared-data`。

这意味着很多脚本会根据 local / Modal 环境自动决定数据目录。非学生 non-modal 下载时，README 建议修改这里的路径。

### `cs336_data/wet_files.py`

这个文件负责生成 English-filtered WET data。

关键结构：

- `BASE_URL = "https://data.commoncrawl.org/"`
- `_EnglishWetFile(Furu[Path])`：处理一组 WET URLs，输出一个 `data.warc.wet.gz`。
- `_EnglishWetFile._create()`：
  - 加载 English language identifier；
  - 当前 `is_english` 是 TODO；
  - 遍历每个 WET record；
  - 对 `conversion` record 读取 payload，decode UTF-8；
  - English 则写入输出，否则跳过；
  - 非 conversion record 会原样写入。
- `EnglishWetFiles(Furu[list[Path]])`：
  - 默认 `n_files=2500`；
  - `group_size=4`；
  - `shuffle_seed=336`；
  - `crawl_id="CC-MAIN-2026-17"`；
  - 从 `wet.paths.gz` 抽样 WET paths；
  - 本地逐个处理，远端用 Modal `.map()` 并行；
  - 远端会把结果 symlink 到 `/shared-data/english-wet-data`。

工程含义：这个文件展示了“数据集构建也是代码”的思想。English WET files 不是手动下载目录，而是可缓存、可复现的 artifact。

### `cs336_data/modal_utils.py`

Modal 配置文件。

重点：

- `SUNET_ID = "TODO"`：不设置会直接 raise。
- `app = modal.App(f"data-{SUNET_ID}")`
- `data_volume`：个人数据 volume。
- `shared_data_volume`：课程 shared data volume。
- `build_image()`：构建 Modal image，安装依赖并加入 `cs336_basics`、`cs336_data`、`AGENTS.md`、`CLAUDE.md`，可选加入 tests。
- `VOLUME_MOUNTS`：把 `/root/data` 和 `/shared-data` 挂载到 Modal function。
- `MODAL_SECRETS = []`：预留 WandB 或其他 secret。

README 特别说非学生 Modal 路线需要从 shared_data_volume 中移除 `environment_name`。

## 3.2 `cs336_basics/` 在本 assignment 中的用途

这个目录是固定训练基础设施，不是本作业的主要改动区域。

### `data.py`

`get_batch(dataset, batch_size, context_length, device)` 从 `np.memmap` token array 中随机采样起点，构造：

- `x = tokens[i : i + context_length]`
- `y = tokens[i + 1 : i + 1 + context_length]`

这要求 tokenized data 是一个连续 token stream。EOS token 的存在会在这个连续 stream 中标记文档边界。

### `model.py`

包含 optimized Transformer LM：

- `Embedding`
- `RotaryEmbedding`
- `BasicsTransformerLM`
- `TransformerBlock`
- `SwiGLU`
- `CausalMultiHeadSelfAttention`

注释说明它用 PyTorch-native components 替代了一些 Assignment 1 手写组件，例如 `F.softmax`、`F.silu`、`F.scaled_dot_product_attention`、`F.cross_entropy`、fused `AdamW` 等。

### `optimizer.py`

提供 `get_cosine_lr`，包含 warmup 和 cosine decay。

### `train_config.py`

定义固定配置：

- 模型：GPT-2 small-shaped，但参数量更接近 430M；
- context length 512；
- 24 layers、16 heads、d_model 1024；
- 训练步数 16,384；
- bfloat16；
- batch size 128；
- eval interval 2,000。

### `ddp_utils.py`

包含 DDP process group setup/cleanup 辅助函数。`scripts/train.py` 实际也直接使用 PyTorch distributed 初始化。

## 3.3 `scripts/download_data.py` 的学生 / 非学生路径

### offline-only

`download_offline_files(root_path)` 下载：

- `tokenized_paloma_c4_100_domains_validation.bin`
- sample `example.warc.gz`
- sample `example.warc.wet.gz`
- `lid.176.bin`
- Dolma hate speech classifier
- Dolma NSFW classifier

命令：

```sh
uv run scripts/download_data.py --offline-only
```

### 学生路径

README 说明学生可直接访问 `/shared-data`。可以用 Modal shell 查看：

```sh
uv run modal shell ./scripts/download_data.py::main --cmd "ls -l /shared-data"
```

下载全部学生数据：

```sh
uv run modal run scripts/download_data.py
```

### 非学生路径

非学生要自己下载数据：

1. offline-only 可直接运行。
2. full data 前要先实现 `cs336_data/wet_files.py` 里的 `is_english`。
3. Modal 路线：移除 `modal_utils.py` 中 shared data volume 的 `environment_name` 后运行。
4. Non-modal 路线：修改 `common.py` 中共享路径后运行。
5. 可修改 `EnglishWetFiles.n_files` 以少于 2,500 个 WET files 进行实验。

## 3.4 `scripts/train.py` 的固定训练配置

`train.py` 是最终评估入口。重点行为：

- 用 `np.memmap(..., dtype=np.uint16)` 读取 train 和 validation binary。
- 构造 `BasicsTransformerLM`。
- DDP 环境变量存在时初始化 NCCL process group。
- 使用 bfloat16 autocast。
- 使用 fused `torch.optim.AdamW`。
- 每 step 随机从 token stream 采样 batch。
- 每 `eval_interval=2000` steps 在 validation memmap 上估计 loss。
- 最后保存 `model.pt` 和 `model_config.json`。
- Modal function 请求 `B200:8`，并把 train/valid binary 复制到 ephemeral disk 提高 IO 性能。

核心约束：**你只传 `--train-bin`，不要改训练脚本来提升结果。**

## 3.5 `scripts/generate_with_gpt2_tok.py` 的用途

这个脚本用于定性检查训练结果。

它会：

- 加载 GPT-2 tokenizer；
- encode prompt；
- 加载训练输出目录中的 model config 和 weights；
- 调用 `model.generate`；
- 用 temperature、top-k 和 EOS 控制采样；
- 打印生成文本。

它不能替代 validation loss，但可以发现明显问题：乱码、语言混杂、重复模板、PII mask token 过多、过度 toxic style 等。

## 3.6 `tests/adapters.py` 中所有 hooks

测试只调用这些函数：

| Adapter hook | 期望行为 |
|---|---|
| `run_extract_text_from_html_bytes(html_bytes)` | HTML bytes → extracted text string |
| `run_identify_language(text)` | text → `(language_id, score)` |
| `run_mask_emails(text)` | text → `(masked_text, num_masked)` |
| `run_mask_phone_numbers(text)` | text → `(masked_text, num_masked)` |
| `run_mask_ips(text)` | text → `(masked_text, num_masked)` |
| `run_classify_nsfw(text)` | text → `(label, score)` |
| `run_classify_toxic_speech(text)` | text → `(label, score)` |
| `run_classify_quality(text)` | text → `(label, score)` |
| `run_gopher_quality_filter(text)` | text → bool |
| `run_exact_line_deduplication(input_files, output_directory)` | exact line dedup and write files |
| `run_minhash_deduplication(input_files, num_hashes, num_bands, ngrams, jaccard_threshold, output_directory)` | fuzzy dedup and write files |

Adapter 设计的教学意义是：你可以自由组织 `cs336_data` 内部模块，但必须提供稳定外部接口。

## 3.7 每个 test 文件的测试目标

### `tests/test_extract.py`

读取 `fixtures/moby.html`，期望输出完全等于 `fixtures/moby_extracted.txt`。

隐含约束：

- 不是“语义相似”即可，而是 exact string equality。
- 标题、段落、link text、list item、bullet、空行都可能影响。
- extraction 函数必须接受 bytes。

### `tests/test_langid.py`

两个样例：

- `moby_extracted.txt` 应识别为 `en`；
- `欢迎来到我们的网站` 应识别为 `zh`。

隐含约束：

- fastText label 要 remap，例如从 `__label__en` 到 `en`。
- score 必须是 Python `float`，且大于 0。
- 中文测试是短文本，不能依赖长文档平均行为。

### `tests/test_pii.py`

测试：

- 单个 email；
- 多个 email；
- 已存在 `|||EMAIL_ADDRESS|||` 时不应计为新增 mask；
- 四种 US phone formats；
- 一个 IPv4 地址。

隐含约束：

- replacement string 必须逐字符一致。
- `num_masked` 必须准确。
- phone regex 至少覆盖：`2831823829`、`(283)-182-3829`、`(283) 182 3829`、`283-182-3829`。
- IP replacement 要保留句末 period。

### `tests/test_toxicity.py`

用 Jigsaw toxic comments 中的 sanity examples 测：

- NSFW classifier：一个 obscene profanity 样例应为 `nsfw`，一个非 obscene 样例应为 `non-nsfw`。
- Toxic speech classifier：一个 insult/profanity 样例应为 `toxic`，另一个应为 `non-toxic`。

隐含约束：

- label string 要和测试一致，或你必须同步调整 adapter 输出。
- score 只要求 float >0，不代表 calibrated probability。
- 测试注释明确：这只是 sanity check，不代表 classifier 准确。

### `tests/test_quality.py`

分两部分。

第一部分 `test_classify_quality`：

- `low_quality_cc.txt` 应输出 label `cc`；
- `high_quality_wiki_reference.txt` 应输出 label `wiki`；
- score float >0。

第二部分 Gopher tests 覆盖：

- valid input；
- 少于 50 words；
- 多于 100,000 words；
- mean word length <3；
- mean word length >10；
- 超过 30% lines ending with ellipsis；
- 少于 80% words have alphabetic character。

隐含约束：

- 边界条件非常重要。
- classifier test 只看两个 fixtures，不说明真实质量。
- Gopher filter 需要定义一致的 word tokenization，否则平均长度和比例会偏。

### `tests/test_deduplication.py`

包含三类测试。

1. exact line dedup：
   - 输入 5 个 documents；
   - 输出也应有 5 个 files；
   - duplicate lines 从所有文件中移除；
   - doc1/doc2 只有同一行，因此输出为空。

2. MinHash exact duplicates：
   - doc1/doc2 完全相同，只保留一个；
   - 输出 4 个 files。

3. MinHash fuzzy duplicates：
   - `rails_mit_license.txt` 和 `react_mit_license.txt` 是 fuzzy duplicates；
   - 只保留其中一个；
   - `pytorch_license.txt` 应保留；
   - 输出 2 个 files。

隐含约束：

- exact line dedup 和 fuzzy doc dedup 的语义完全不同。
- fuzzy dedup 要在候选之后计算真实 Jaccard，不只是 LSH bucket match。
- duplicates 要聚类，不能只处理孤立 pair。
- 测试接受 rails/react 任意一个被保留。

## 3.8 Fixtures 如何帮助理解边界条件

### HTML fixture

`moby.html` 是一个小 HTML 页面，包含：

- `<h1>` 标题；
- 长段落；
- `<img>`；
- 带 `<a>` 的段落；
- `<ul><li>Novel</li></ul>`。

`moby_extracted.txt` 显示期望 extraction：保留标题、正文、link text 和 list item，但不保留 HTML tags 和 image filename。这说明 extractor 的目标是 visible text，而不是 DOM dump。

### Quality fixtures

`low_quality_cc.txt` 是典型低质量网页导航/论坛模板，包含重复菜单、login、copyright、powered by phpBB 等。它帮助你理解 quality classifier 的负类。

`high_quality_wiki_reference.txt` 是长篇结构化百科/哲学文章，段落连贯、信息密度高、引用清楚。它帮助你理解正类。

### Exact dedup fixtures

输入中 `- home`、`- menu` 等行在多个文件重复。目标输出会把这些重复行从所有文件删除，而不是保留第一次出现。这是最重要的隐藏约束。

### Fuzzy dedup fixtures

Rails 和 React MIT license 不是 exact duplicate，但主体模板高度相同，只是标题、copyright 和换行不同。PyTorch license 更长且内容不同，应保留。它体现了 normalization、n-gram Jaccard、MinHash/LSH candidate generation 的必要性。

## 3.9 `configs/` 解析

### `configs/config.yaml`

它引用 Hydra defaults：

```yaml
defaults:
  - paths: base_paths
  - model: base_model
  - training: base_training
  - _self_
```

但在当前上传包中没有看到对应的 `configs/paths/base_paths.yaml`、`configs/model/base_model.yaml`、`configs/training/base_training.yaml`。因此对这个 repo 的实际训练路径而言，`scripts/train.py` 中的 dataclass config 和 argparse 是更直接的入口。

### `configs/experiment/your_data.yaml`

它设置：

- `paths.train_bin`：你的 GPT-2 tokenized data；
- `paths.valid_bin`：固定 Paloma validation binary，不要改；
- `paths.model_output`：输出目录；
- WandB entity/project。

它更像配置化实验的 scaffold。实际提交和 handout 推荐命令仍然是把 `--train-bin` 传给 `scripts/train.py`。

## 3.10 `pyproject.toml` 解析

依赖体现了本作业技术栈：

- extraction / web archive：`resiliparse`、`fastwarc`、`warcio`、`xopen`；
- classifiers：`fasttext`；
- hashing / dedup：`mmh3`；
- NLP / tokenization：`nltk`、`regex`、`transformers`、`tiktoken`；
- large-scale processing：`polars`、`tqdm`、`submitit`、`modal`；
- training：`torch`、`einops`、`einx`、`jaxtyping`、`wandb`；
- tests：`pytest`、`pytest-timeout`。

这也说明 assignment 的本质是“数据系统 + NLP + 分布式训练评估”的综合项目。

---

# 4. 前置知识

## 4.1 Common Crawl

Common Crawl 是公开网页抓取数据。它提供的是网页抓取快照，而不是高质量语料。真实使用时必须经过解析、过滤、去重、质量控制和法律/隐私/安全审查。

## 4.2 HTML parsing / boilerplate removal

HTML 页面包含正文，也包含大量非正文：导航栏、广告、footer、cookie banner、推荐链接、评论区、脚本残留。boilerplate removal 的目标是尽可能保留 main content，删除模板文本。

难点在于“可见文本不等于训练文本”。菜单和按钮对网页可见，但对 LM 预训练通常价值较低；表格、代码、引用和列表有时又是有价值内容，不能简单删除。

## 4.3 Character encoding detection

HTML bytes 需要 decode 成 Unicode string。多数网页是 UTF-8，但并非全部。错误 decode 会产生乱码或异常。稳健 pipeline 通常先尝试 UTF-8，再根据 meta charset、HTTP header 或库检测结果 fallback。

## 4.4 fastText classifiers

fastText 是高效文本分类工具。它常用 label 格式类似：

```text
__label__en
__label__wiki
__label__cc
```

预测通常返回 label 和 score。工程上要注意：

- label remapping；
- score 是否校准；
- batch prediction 性能；
- 模型加载缓存；
- 多进程中避免反复加载大模型。

## 4.5 Language identification

Language ID 是判断文本主要语言的分类任务。对长文本通常较准，对短文本、code-mixed 文本、专有名词密集文本、低资源语言较难。过滤阈值应通过人工抽检和目标验证集调优。

## 4.6 Regex PII matching

PII regex 是规则匹配任务，适合 email、US phone、IPv4 这类格式明显的内容。它的难点是格式变体太多。regex 往往在 precision 和 recall 间取舍：写得太宽会误伤，写得太窄会漏检。

## 4.7 Toxicity / NSFW classifier

Toxicity 和 NSFW 都是高语境依赖任务。Jigsaw-trained classifier 对维基评论风格有效，但迁移到整个 web 时会遇到 domain shift。它们的输出应该作为过滤信号之一，而不是绝对真理。

## 4.8 Heuristic quality filtering

启发式质量过滤使用可解释规则，例如长度、平均词长、符号比例、stop words、重复行比例等。它速度快、可大规模运行，但只能识别“明显坏”的样本，不能替代语义质量判断。

## 4.9 Classifier training

训练 quality classifier 的关键不是调用 fastText，而是构造数据：

- 正样本是否真的高质量；
- 负样本是否代表真实噪声；
- 是否平衡类别；
- 是否避免 classifier 只学域名或长度；
- 阈值如何根据 validation loss 和人工检查调优。

## 4.10 Line-level dedup

line-level dedup 假设模板文本经常以完全相同的行出现。它能高效删除 navigation/footer，但也可能删除常见有用句子，例如 license、引用、公式或标准免责声明。

## 4.11 MinHash / Jaccard / LSH

- Jaccard：衡量两个 n-gram set 的重合程度。
- MinHash：用固定长度 signature 近似 Jaccard。
- LSH：把 signature 分 band，快速找高相似候选。

三者组合能让 fuzzy dedup 在大规模语料上可行。

## 4.12 Multiprocessing / large file streaming

WET files 很大。正确做法是 streaming：逐 record 读、逐 record 处理、逐 record 写。并行时可以按文件切分任务，使用 `concurrent.futures.ProcessPoolExecutor` 或 Modal `.map()`。不要把 2,500 个 WET files 全读进内存。

## 4.13 Tokenized binary data format

训练脚本期望一个连续 token stream binary file：

- dtype：`np.uint16`；
- tokenizer：GPT-2；
- serialization：`tofile`；
- reading：`np.memmap`。

这个格式没有显式 document offsets，因此 EOS token 是重要边界信号。

## 4.14 Validation loss as data quality signal

Validation loss / perplexity 衡量模型对目标验证分布的拟合程度。在本作业中，Paloma C4 validation loss 是数据 pipeline 的主要反馈信号。

但要注意：

- validation loss 低不一定代表数据全面好；
- 可能过拟合 C4 风格；
- 可能牺牲安全或多样性；
- 可能因数据泄漏而虚假变好；
- 需要结合 generation 和 manual inspection。

---

# 5. 推荐完成顺序

## 5.1 实操路线

### Step 1：先检查 README 和数据路径

先确认：

- 是否在学生环境，有 `/shared-data`；
- 是否需要 `uv run scripts/download_data.py --offline-only`；
- `modal_utils.py` 是否设置 `SUNET_ID`；
- 非学生是否需要修改 `common.py` 和 `wet_files.py`；
- classifiers 是否存在：language ID、NSFW、hate speech；
- Paloma validation binary 是否存在。

不要一开始就跑 full 2,500 WET pipeline。先确保小文件和测试依赖可用。

### Step 2：先跑单元测试，建立失败清单

运行：

```sh
uv run pytest -v ./tests
```

最初 adapter 全是 `NotImplementedError`，失败是正常的。你要记录每个失败对应哪个 adapter，而不是一次性改所有东西。

### Step 3：先实现 extraction

先做 HTML extraction，因为后续 language ID、PII、toxicity、quality 都依赖文本。如果 extraction 输出噪声太多，后面 classifier 的输入会偏离真实正文。

验证顺序：

1. 用 `moby.html` fixture 对齐 exact output。
2. 用 sample WARC 抽取几条，人工看结果。
3. 与 sample WET 对比，观察 WET 是否保留了过多网页结构。

### Step 4：language ID、PII、harmful filters、quality filters 的顺序

推荐逻辑顺序：

1. **Language ID**：先确保目标语言，避免非英文样本进入英文质量判断。
2. **PII masking**：对保留文本做安全替换；通常 masking 是 modification，不一定删除整篇。
3. **Harmful content filtering**：删除或标记 NSFW/toxic 高置信样本。
4. **Gopher quality filters**：快速过滤明显低质文本。
5. **Quality classifier**：更细粒度地区分高质量与普通 CC。

也可以在实验中调整顺序。例如先 Gopher 再 toxicity 能减少 classifier 调用成本；先 toxicity 再 quality 能保留更明确的安全审计统计。关键是每一步记录 kept/removed counts。

### Step 5：exact dedup 先做

exact line dedup 容易验证，先在 fixtures 上通过，再在小样本上跑。

理解重点：重复行从所有文档删除，不是保留一次。这对网页模板很有效，但可能让某些短文档变空。后续 pipeline 应处理空文档或过短文档。

### Step 6：MinHash dedup 逐步验证

推荐分层验证：

1. 用 exact duplicate fixtures 验证可以删掉完全重复文档。
2. 用 MIT license fixtures 验证 fuzzy duplicate。
3. 用少量真实 WET 文档检查 candidate bucket 数量是否合理。
4. 调整 `num_hashes`、`num_bands`、`ngrams`、`jaccard_threshold`。
5. 人工查看被判为 duplicate 的 pairs，估计 false positives。

不要只看输出文件数。要检查是否误删语义不同但共享模板的文档。

### Step 7：在小样本组合 pipeline

先选 5-20 个 WET files 或更少 records，构建端到端 mini pipeline：

```text
read WET → parse records → filters → audit stats → write filtered text → inspect examples
```

记录每一步：

- 输入 documents；
- decode failures；
- non-English removed；
- PII masked count；
- NSFW/toxic removed；
- Gopher removed；
- quality classifier removed；
- exact dedup removed lines；
- fuzzy dedup removed docs；
- final docs/tokens。

### Step 8：扩展到 2,500 WET files

扩展时重点是系统问题：

- streaming IO；
- gzip/WARC 读取；
- 多进程 worker 数；
- 每个 worker 是否重复加载 fastText 模型；
- 输出目录结构；
- failure retry；
- progress bar；
- audit logs；
- disk 空间。

建议每个阶段都能 resume。大规模处理时，最怕跑几个小时后因为一个坏 record 或路径错误全部失败。

### Step 9：tokenization

tokenization 前先确定 filtered data 的文档边界。如果你把每个 document 写成一行，可以按 line tokenize；如果 document 里保留多行，要确保不会错误把段落当成多个文档。

tokenization 检查：

- GPT-2 tokenizer；
- 每个 doc append EOS；
- 输出 dtype 是 `np.uint16`；
- `np.fromfile(output, dtype=np.uint16)` 能读；
- decode 前几千 tokens 看是否正常；
- 统计 token count。

### Step 10：训练并评估

先做小规模 dry run 或单 GPU sanity check，确认 binary 格式、路径和模型加载没有问题。最终用：

```sh
uv run modal run scripts/train.py --train-bin /root/data/your_data.bin
```

记录：

- filtered data 版本；
- token count；
- filter thresholds；
- validation loss curve；
- best validation loss；
- generation samples；
- 是否有异常训练 loss。

### Step 11：manual inspection

每轮 pipeline 都要抽样：

- kept examples：是否像 C4、是否正文丰富、是否有模板残留、是否有 PII、是否安全；
- discarded examples：删除是否合理、是否误删高质量内容；
- modified examples：PII mask 是否正确、是否过度替换。

建议写一个 audit TSV/JSONL，记录 document ID、URL/domain、filter decision、scores、excerpt。注意最终提交时不要把大 `.json`、`.txt`、`.bin` 数据混入 zip。

## 5.2 常见坑

### 1. HTML bytes decode 失败

只用 `html_bytes.decode("utf-8")` 会在非 UTF-8 页面失败或产生乱码。要考虑 fallback 和 encoding detection。即使 tests 只有小 fixture，真实 WARC 会遇到各种编码。

### 2. fastText label 格式不对

fastText 可能返回 `__label__en`，测试期望 `en`。quality classifier 可能返回 `__label__wiki`，测试期望 `wiki`。adapter 层要统一 label。

### 3. PII replacement string 不完全一致

测试要求：

- `|||EMAIL_ADDRESS|||`
- `|||PHONE_NUMBER|||`
- `|||IP_ADDRESS|||`

少一个竖线、多一个空格、大小写不同都会失败。

### 4. PII count 错误

已有的 `|||EMAIL_ADDRESS|||` 不应算作这次 masked email。多次替换要准确计数。IP regex 不能把句末 `.` 一起吞掉。

### 5. Gopher filter 边界条件

规则是：

- less than 50 删除；
- more than 100,000 删除；
- mean word length outside 3 to 10 删除；
- more than 30% ellipsis line 删除；
- less than 80% alphabetic-word 删除。

注意 “less than / more than / outside / more than” 的边界。等于 50、等于 100,000、等于 30%、等于 80% 通常不应按失败处理，除非你有明确测试证据。

### 6. Quality classifier 测试只 sanity，不代表真实质量

`low_quality_cc.txt` 和 `high_quality_wiki_reference.txt` 只是两个样例。通过测试不代表 classifier 能在 Common Crawl 上区分 SEO spam、论坛模板、新闻、百科、博客、产品页等。

### 7. MinHash normalization 不充分

不 lowercase、不去 punctuation、不 normalize whitespace、不去 accents，会漏掉很多模板近重复。MIT license fixture 中换行和 attribution 不同，如果 normalization 太弱，Jaccard 可能低估相似性。

### 8. MinHash 只做 LSH，不算真实 Jaccard

handout 要求 LSH 只是找 candidate duplicates；候选对还要计算真实 n-gram Jaccard，并用 threshold 判断。只要同 bucket 就删，容易 false positive。

### 9. Duplicate clustering 漏掉 transitive closure

如果 A-B 相似、B-C 相似，即使 A-C 不直接相似，也应视作同一个 duplicate cluster。只做 pairwise 删除可能结果不稳定。

### 10. 输出文件名结构不符合测试

dedup 输出必须写到 output directory，并保持输入 basename。比如输入 `a/1.txt`、`a/2.txt`，输出应是 `b/1.txt`、`b/2.txt`。

### 11. exact line dedup 误解为“保留第一次”

本作业 exact line dedup 是只保留 corpus 中唯一出现的行。重复行在所有文件中都删除，不是保留一份。

### 12. 过滤过强导致数据不足

训练脚本会采样约 8.6B token。如果 filtered dataset 太小，训练会反复看到同一批 token，可能过拟合、重复生成、validation loss 不稳。需要在质量和数量之间平衡。

### 13. 误把 validation data 泄漏进 training data

可以用 validation data 观察目标风格或构造过滤策略，但不能复制 validation 内容到 train。否则 validation loss 失去意义。

### 14. WET headers 混入训练文本

WET record headers 如 `Content-Length` 不是正文。如果解析 WET 时简单按 gzip 文本整文件读入，可能把 WARC/WET headers 当训练数据。

### 15. 多进程重复加载大模型

fastText classifiers 如果每个 document 加载一次会极慢。应按 worker 级别缓存加载，或设计批处理。虽然这里不写实现代码，但工程上必须注意。

### 16. 只优化 validation loss，忽略安全和生成质量

低 validation loss 不代表安全、无 PII、无 toxic leakage、无模板复读。最终报告应结合 validation curve、manual inspection 和 generation samples。

---

# 6. 学习路线

## 6.1 知识地图

可以把本作业知识地图分成七层：

```text
A. 原始网页数据层
   Common Crawl, WARC, WAT, WET, URL/domain metadata

B. 文本抽取层
   HTML parsing, encoding detection, boilerplate removal, main-content extraction

C. 安全与合规过滤层
   PII masking, NSFW classification, toxic speech classification, audit logs

D. 质量过滤层
   Gopher heuristic filters, quality classifier, positive/negative sampling, threshold tuning

E. 去重层
   exact line dedup, document-level fuzzy dedup, MinHash, Jaccard, LSH, duplicate clustering

F. 训练数据格式层
   GPT-2 tokenizer, EOS token, np.uint16 binary, memmap, streaming tokenization

G. 评估与迭代层
   fixed training, Paloma C4 validation loss, generation inspection, ablation, manual data inspection
```

这七层串起来，就是一个小型但完整的 LLM pretraining data engineering workflow。

## 6.2 推荐搜索关键词

建议用英文关键词查资料，因为大多数数据集论文和工具文档是英文。

### Common Crawl / Web archive

- `Common Crawl WARC WET WAT difference`
- `WARC file format web archive`
- `FastWARC ArchiveIterator WARC records`
- `warcio WARC WET Python`

### HTML extraction

- `HTML boilerplate removal text extraction`
- `Resiliparse extract_plain_text`
- `trafilatura boilerplate removal`
- `readability main content extraction`
- `character encoding detection HTML Python`

### Language and safety filtering

- `fastText language identification lid.176.bin`
- `PII detection regex email phone IP`
- `Jigsaw toxic comment classification`
- `Dolma fastText jigsaw classifier`
- `data filtering toxicity false positives false negatives`

### Quality filtering

- `Gopher quality filters Appendix A`
- `language model pretraining data quality filtering`
- `WebText Reddit karma filtering`
- `Wikipedia references quality classifier Common Crawl`
- `C4 data cleaning pipeline`

### Deduplication

- `MinHash LSH Jaccard similarity document deduplication`
- `near duplicate detection web documents`
- `RefinedWeb deduplication MinHash`
- `line-level deduplication language model data`
- `duplicate data memorization language models`

### Tokenization / training

- `GPT-2 tokenizer endoftext document boundary`
- `numpy memmap language model training data`
- `pretraining data ablation validation loss`
- `Paloma benchmark language model fit`

## 6.3 推荐深入方向

### 方向 1：Data quality vs data quantity

研究不同过滤强度下 token count 与 validation loss 的关系。过弱会保留噪声，过强会数据不足。真实 LLM 数据工程的核心就是在质量、数量、多样性之间找平衡。

### 方向 2：Classifier calibration and thresholds

fastText score 不一定是可靠概率。可以比较不同 threshold 的 manual precision/recall 和 validation loss，理解 classifier score 的实际意义。

### 方向 3：Dedup 对 memorization 的影响

重复数据会让模型更容易 memorise，也会让训练 token 浪费在模板上。可以比较 no dedup、line dedup、MinHash dedup 的 validation loss 和生成重复率。

### 方向 4：Domain-aware filtering

Paloma C4 100 domains 是目标验证分布。可以基于 domain、URL path、document length、HTML extraction patterns 做更接近 C4 的过滤，但要避免 validation leakage。

### 方向 5：Manual data audit methodology

学习如何抽样、标注、记录 filter decisions。真实团队会维护 data cards、audit dashboards、risk categories，而不是只写一次脚本。

## 6.4 与 Dolma / Common Crawl / C4 / RefinedWeb / FineWeb 的关系

### Common Crawl

Common Crawl 是原始网页数据来源。它提供 WARC/WET，但不保证训练质量。本作业从 Common Crawl 出发，复现了把 raw crawl 变成 LM data 的基本过程。

### C4

C4 是从 Common Crawl 清洗得到的英文文本数据集，强调过滤、去重、启发式清理。本作业用 Paloma C4 100 domains subset 做验证目标，因此你的 pipeline 实际上在学习如何构造更像 C4 且更适合 LM 的数据。

### Dolma

Dolma 是开放预训练语料项目，包含大规模 web data processing 和 safety / quality filtering。作业直接使用 Dolma 发布的 Jigsaw fastText NSFW / hate speech classifiers，说明现代开放数据集会把分类器作为数据过滤组件。

### RefinedWeb

RefinedWeb 强调从 web data only 出发，通过强过滤、dedup、质量控制构造可竞争的预训练语料。本作业的 MinHash normalization 和 fuzzy dedup 思路与这类 web-scale dedup pipeline 高度相关。

### FineWeb

FineWeb 代表较新的 Common Crawl 派生高质量网页语料方向，重点是 extraction、filtering、dedup、版本化实验和下游验证。完成本作业后，你会更容易读懂 FineWeb 这类数据集的处理报告：它们关注的不是“下载网页”，而是每个过滤阶段如何改变 token distribution 和模型表现。

## 6.5 完成后应掌握的能力

完成这个 assignment 后，你应该能：

1. 解释 WARC、WET、HTML extraction 的区别和训练影响。
2. 用人工检查发现 web data 的真实噪声模式。
3. 理解 fastText language ID 和 safety classifiers 的输入输出、label mapping 和阈值风险。
4. 设计 PII masking regex，并分析 false positives / false negatives。
5. 实现并解释 Gopher-style heuristic quality filters。
6. 构造 quality classifier 的正负样本，并理解阈值 trade-off。
7. 区分 exact line dedup 与 fuzzy document dedup。
8. 解释 MinHash、Jaccard、LSH 如何让大规模近重复检测可行。
9. 用 streaming / multiprocessing 思维处理大型 WET files。
10. 把文本转换成 GPT-2 tokenized binary，并理解 EOS 文档边界。
11. 使用固定训练脚本把 validation loss 作为数据质量反馈。
12. 写出有证据的 data pipeline writeup，包括保留/删除样本和 ablation。

## 6.6 可额外做的小实验

### 实验 1：Gopher filter ablation

比较：

- no Gopher filters；
- only length filter；
- length + mean word length；
- all required Gopher filters。

观察每一步删除比例、manual quality 和 validation loss。

### 实验 2：Language threshold sweep

如果你从 raw WET/WARC 开始，可以比较 English probability threshold：

- 0.5；
- 0.7；
- 0.8；
- 0.9；
- 0.95。

观察 English precision、数据量、非英文残留和 validation loss。

### 实验 3：Harmful classifier threshold sweep

对 NSFW/toxic classifier 使用不同 confidence threshold，人工检查高分和边界分数样本。重点不是追求删得最多，而是理解 false positive / false negative。

### 实验 4：PII masking audit

随机抽样被替换文本，统计：

- 正确 email；
- 正确 phone；
- 正确 IP；
- false positive；
- false negative；
- mask 后文本是否仍适合训练。

### 实验 5：MinHash threshold sweep

比较不同 `jaccard_threshold`：

- 0.7；
- 0.8；
- 0.85；
- 0.9。

人工检查 duplicate clusters，观察误删和漏删。

### 实验 6：LSH bands / hashes ablation

固定 threshold，改变：

- `num_hashes`；
- `num_bands`；
- n-gram length。

观察 candidate pair 数量、运行时间、dedup 删除比例和 manual precision。

### 实验 7：Quality classifier positive sample source

比较不同正样本：

- Wikipedia external links；
- manually selected high-quality domains；
- Paloma-like domains but不复制 validation text；
- 混合正样本。

观察 classifier 是否学到域名偏差、长度偏差或真实文本质量。

### 实验 8：Pipeline order ablation

比较：

```text
A: language → PII → harmful → Gopher → quality → dedup
B: language → Gopher → quality → harmful → PII → dedup
C: language → dedup → PII → harmful → quality
```

观察运行时间、filter counts、final token count、validation loss 和 manual quality。

### 实验 9：Domain filtering

用 `tldextract` 统计 final corpus domains，比较是否过度集中在论坛、商品站、SEO 站或少数大站。可以尝试 domain cap 或 blacklist/allowlist，但要用人工检查验证。

### 实验 10：Generation-based qualitative audit

训练后用固定 prompts 生成：

- 普通叙述；
- 技术说明；
- 问答；
- 含 PII 风险的 prompt；
- 容易诱发 toxic 的 prompt。

检查是否出现模板复读、PII mask token、乱码、语言混杂或 unsafe style。

---

## 结论

CS336 Assignment 4 的核心学习点是：**LLM 的能力和风险很大程度上由训练数据 pipeline 决定**。Common Crawl 只是原料；真正的预训练数据来自一连串有取舍的工程决策：抽取什么、删除什么、替换什么、保留什么、如何去重、如何序列化、如何评估。

单元测试能保证基础接口和边界行为，但不能保证数据质量。真正高质量的完成方式应同时具备：

- 正确 adapters；
- 可扩展 pipeline；
- 详细 filter statistics；
- manual inspection；
- ablation；
- token count；
- validation loss curve；
- generation analysis；
- 对 false positives / false negatives 的诚实讨论。

如果把这个 assignment 做扎实，你掌握的就不只是 CS336 的一次作业，而是真实 LLM pretraining data engineering 的缩小版工作流。
