# CS336 作业 4（数据）：过滤语言建模数据

版本 26.0.1  
CS336 Staff  
2026 春季

## 1 作业概述

在本作业中，你将获得一些实践经验：通过过滤 Web 爬取数据来创建用于语言建模的数据。

**你将实现的内容。**

1. 将 Common Crawl HTML 转换为文本。
2. 使用各种方法过滤提取出的文本（例如，有害内容、个人身份信息等）。
3. 对训练数据进行去重。

**你将运行的内容。**

1. 在不同数据集上训练语言模型，以便更好地理解特定处理决策对性能的影响。

**代码的组织方式。**

所有作业代码以及本说明文档均可在 GitHub 上获得：

github.com/stanford-cs336/assignment4-data

请 `git clone` 该仓库。如果有任何更新，我们会通知你，你可以通过 `git pull` 获取最新版本。

1. `cs336_basics/*`：此文件夹包含用于训练我们在作业 1 中构建的模型的代码。该模型经过了轻微优化，用 PyTorch 原生组件替换了作业 1 中一些手工构造的组件（例如，使用 PyTorch 内置的交叉熵内核）。我们还包含了一个支持多 GPU 分布式数据并行训练的训练脚本。你将使用该脚本在你过滤后的数据上训练模型。
2. `cs336_data/*`：这是你为作业 4 编写代码的位置。我们创建了一个名为 `cs336_data` 的模块，其中包含启动代码和本作业的占位符。
3. `tests/*.py`：这里包含你必须通过的所有测试。这些测试会调用 `tests/adapters.py` 中定义的钩子。你将实现这些 adapter，以便将你的代码连接到测试。编写更多测试或修改测试代码可能有助于调试你的代码，但你的实现应当能够通过原始提供的测试套件。
4. `README.md`：该文件包含关于预期目录结构的更多细节，以及一些关于环境设置的基本说明。

**如何提交。**

你需要向 Gradescope 提交以下文件：

- `writeup.pdf`：回答所有书面问题。请对你的回答进行排版。
- `code.zip`：包含你编写的所有代码。

## 2 过滤 Common Crawl

虽然大型语言模型主要在互联网数据上训练，但大多数研究人员并不会构建 Web 爬虫来为其模型获取训练数据。相反，他们使用公开可用的爬取数据。最流行的公开 Web crawl 来自 Common Crawl，这是一个非营利组织，提供免费的网页语料库，并开放了“跨越 17 年、超过 2500 亿个页面”。[^1]

然而，正如我们将看到的，将 Common Crawl（CC）转储数据转换为可用于语言模型训练的数据需要大量工作。例如，网页的原始数据是 HTML，而我们希望从中提取文本。此外，许多页面可能质量低，可能是精确重复或近似重复，可能包含有害内容，或者可能包含敏感信息；我们可能希望过滤掉这些页面，或者从数据集中删除其内容中不理想的部分。在本作业中，我们将搭建一个执行其中若干步骤的管道，把原始互联网数据转换为可供语言模型使用的训练集。

### 2.1 查看数据

在实现任何内容之前，查看原始数据并对其形成直观认识总是有用的。CC 数据以三种格式提供：

**WARC**

（“Web ARChive format”）文件包含原始 CC 数据，其中包括页面 ID 和 URL、元数据和 HTTP 请求细节（例如，请求的日期和时间、服务器 IP 地址），当然也包括页面的原始内容（例如 HTML）。

**WAT**

（“Web Archive Transformation”）文件包含从 WARC 文件中提取的更高层级元数据，并以 JSON 对象形式转储。例如，对于 HTML 页面，这包括该页面中的链接列表以及页面标题。

**WET**

（“Web Extracted Text”）文件包含从原始 HTML 页面中提取出的纯文本。

在下面的问题中，我们将查看一个 WARC 文件及其对应的 WET 文件。这些文件属于一次 2026 年的 Common Crawl 爬取，并按如下方式下载：

```bash
# Download a sample WARC file.
$ wget https://data.commoncrawl.org/crawl-data/CC-MAIN-2026-12/segments/1741613651012.6/warc/CC-MAIN-20260310152245-20260310182245-00000.warc.gz

# Download its corresponding WET file.
$ wget https://data.commoncrawl.org/crawl-data/CC-MAIN-2026-12/segments/1741613651012.6/wet/CC-MAIN-20260310152245-20260310182245-00000.warc.wet.gz
```

这些文件可在以下位置获得：

- `/shared-data/CC/example.warc.gz`
- `/shared-data/CC/example.warc.wet.gz`

**WARNING：**这些文件包含完全未经过滤的互联网页面，其中可能包含大量潜在有害内容。如果你看到某个文档并不想阅读，可以继续向下滚动并跳到另一个文档。

### Problem (look_at_cc): Looking at Common Crawl（查看 Common Crawl）（4 分）

(a) 使用上面路径中提供的 WARC 文件副本。让我们查看该文件中的第一个页面。这是一个 gzip 压缩文件，你可以使用以下命令浏览其内容：

```bash
$ zcat /shared-data/CC/example.warc.gz | less
```

`less` 允许你使用键盘方向键、Page Up、Page Down 浏览该文件。要退出，请按 `q`。

查看第一个网页。它的 URL 是什么？它是否仍然可以访问？仅通过查看原始 HTML，你能否判断该页面大致是什么内容？

**Deliverable：**2-3 句回答。

(b) 现在让我们查看对应的 WET 文件：

```bash
$ zcat /shared-data/CC/example.warc.wet.gz | less
```

注意，WET 文件包含 HTTP 头（例如 `Content-Length`），这些并不是提取出的文本内容的一部分。如果你查看第一个示例，就会看到其中包含了从你刚刚看到的原始 HTML 中提取出的文本。

请注意，提取出的很多文本都令人想起 HTML 结构，并不真正是页面的主要内容。你看到的文本中是否有一些部分你认为本应被提取器过滤掉？请思考这类文本作为训练数据的质量：在看起来像这样的文本上训练模型可能会出现什么问题？相反，模型可能从该页面中提取出哪些有用信息？

**Deliverable：**3-4 句回答。

(c) 什么构成好的训练样例具有高度的上下文依赖性。描述一个该示例可能适合放入训练数据的应用领域，以及一个可能不适合的应用领域。

**Deliverable：**1-2 句回答。

(d) 让我们查看更多示例，以便更好地了解 Common Crawl 中有什么。浏览另外 25 条 WET 记录。对每条记录，非常简要地评论该文档的语言（如果你能识别）、域名、页面类型等。直到看到你认为的“高质量”网页，需要多少个示例？

**Deliverable：**对 25 个文档进行简要注释，包括文档语言、域名、页面类型以及关于该文档的其他杂项说明。并报告直到看到高质量示例所需的示例数量。

### 2.2 HTML 到文本转换

正如你可能已经从上面的 WARC 和 WET 文件中意识到的，从 HTML 中提取文本具有挑战性。通常，任何提取过程都会在 HTML 中寻找可见内容（例如 `<p>` 标签，它们本应包含文本块）。但这仍然可能提取出远多于我们在 Web 浏览器中打开页面时所感知到的主要内容。例如，当打开 StackOverflow 时，主要内容在问题和回答中，但从技术上说，菜单选项、指向其他 StackExchange 无关页面的链接、页脚、注册链接或登录链接——这些都是可见文本，而可靠地区分这些内容与页面主要内容是具有挑战性的。

许多工具实现了文本提取管道。在本作业中，我们将使用 Resiliparse[^2] 库执行文本提取。Resiliparse 还将帮助解决一个更基础的问题：检测包含原始内容的字节的文本编码。尽管 Web 上大多数页面都以 UTF-8 编码（根据 Wikipedia，为 98.2%），我们的文本提取管道也应当对其他编码具有鲁棒性。

**Note：**

我们建议使用 FastWARC 库来遍历每个 WARC 文件中的记录。具体来说，以下类可能会有帮助：

```python
from fastwarc.warc import ArchiveIterator, WarcRecordType
```

### Problem (extract_text): HTML to text conversion（HTML 到文本转换）（3 分）

(a) 编写一个函数，从包含原始 HTML 的字节字符串中提取文本。使用 `resiliparse.extract.html2text.extract_plain_text` 执行提取。该函数需要一个字符串，因此你需要先将字节字符串解码为 Unicode 字符串。请注意，输入字节字符串可能并非以 UTF-8 编码，因此你的函数应当能够在 UTF-8 解码失败时检测编码。Resiliparse 也提供了 `resiliparse.parse.encoding.detect_encoding()`，这可能会有用。

**Deliverable：**一个函数，接收包含 HTML 的字节字符串并返回包含提取文本的字符串。实现 adapter `[run_extract_text_from_html_bytes]`，并确保它通过 `uv run pytest -k test_extract_text_from_html_bytes`。

(b) 在单个 WARC 文件上运行你的文本提取函数。将其输出与对应 WET 文件中的提取文本进行比较。你注意到哪些差异和/或相似性？哪种提取看起来更好？

**Deliverable：**2-3 句回答，比较并对比你自己的函数提取出的文本与 WET 文件中的提取文本。

### 2.3 语言识别

Web 包含用数千种语言编写的页面。但在大多数计算预算下，训练一个能够在规模上有效利用如此多样化数据的多语言模型是具有挑战性的。因此，许多从 Common Crawl 派生出的语言建模训练集只包含有限语言集合的数据。

用于此目的的一个有用库是 fastText（https://fasttext.cc），它提供高效的文本分类器。该库既提供用于在你自己的数据上训练分类器的基础设施，也提供一组预训练模型，包括用于语言识别的模型。你可以从 https://fasttext.cc/docs/en/language-identification.html 下载 fastText 语言识别模型 `lid.176.bin`；它也可在 `/shared-data/classifiers/lid.176.bin` 获得。

通常，语言过滤器会使用分类器给出的分数来决定是否保留给定页面。使用 fastText 语言识别分类器实现一个语言识别过滤器，该过滤器应给出一个非负分数，用于表示它对预测的置信程度。

### Problem (language_identification): Language identification（语言识别）（6 分）

(a) 编写一个函数，接收 Unicode 字符串并识别该字符串中存在的主要语言。你的函数应返回一个二元组，其中包含语言标识符以及一个介于 0 和 1 之间的分数，该分数表示对该预测的置信度。

**Deliverable：**一个执行语言识别的函数，给出其最高语言预测和分数。实现 adapter `[run_identify_language]`，并确保它通过 `uv run pytest -k test_identify_language` 中的两个测试。注意，这些测试假定英语（`"en"`）和中文（`"zh"`）具有特定的字符串标识符，因此如有必要，你的测试 adapter 应执行任何适用的重新映射。

(b) 语言模型在推理时的行为在很大程度上取决于它们训练时所用的数据。因此，数据过滤管道中的问题可能导致下游问题。你认为语言识别过程中的问题可能会引发哪些问题？在更高风险的场景中（例如部署面向用户的产品时），你会如何缓解这些问题？

**Deliverable：**2-5 句回答。

(c) 在从 WARC 文件中提取出的文本上运行你的语言识别系统（通过你先前实现的文本提取函数）。随机手动识别 20 个示例中的语言，并将你的标签与分类器预测进行比较。报告任何分类器错误。英语文档所占比例是多少？基于你的观察，过滤时应使用怎样的分类器置信度阈值？

**Deliverable：**2-5 句回答。

### 2.4 个人身份信息

Web 包含大量可用于联系或识别个人的信息，例如电子邮件地址、电话号码或 IP 地址。我们可能不希望面向用户的语言模型输出关于真实个人的这类信息，因此一个常见步骤是在训练数据集中屏蔽这些信息片段。

你现在将实现三种屏蔽过程，分别用于屏蔽：(a) 电子邮件地址，(b) 电话号码，以及 (c) IP 地址。

### Problem (mask_pii): Personally identifiable information（个人身份信息）（3 分）

(a) 编写一个函数来屏蔽电子邮件。你的函数将接收字符串作为输入，并将所有电子邮件地址实例替换为字符串 `"|||EMAIL_ADDRESS|||"`。为了检测电子邮件地址，你可以查找能够可靠完成此任务的正则表达式。

**Deliverable：**一个函数，将给定字符串中的所有电子邮件地址替换为字符串 `"|||EMAIL_ADDRESS|||"`，并返回一个二元组，其中包含新字符串以及被屏蔽的实例数量。实现 adapter `[run_mask_emails]`，并确保它通过 `uv run pytest -k test_mask_emails` 中的所有测试。

(b) 编写一个函数来屏蔽电话号码。你的函数将接收字符串作为输入，并将所有电话号码实例替换为字符串 `"|||PHONE_NUMBER|||"`。可靠地完成这件事可能极具挑战性，因为电话号码可能以极其多样的格式书写，但你应尝试至少捕获美国常见的电话号码格式，并对轻微的语法偏差具有鲁棒性。

**Deliverable：**一个函数，将给定字符串中的电话号码替换为字符串 `"|||PHONE_NUMBER|||"`，并返回一个二元组，其中包含新字符串以及被屏蔽的实例数量。实现 adapter `[run_mask_phone_numbers]`，并确保它通过 `uv run pytest -k test_mask_phones`。

(c) 编写一个函数来屏蔽 IP 地址。对于此问题，只关注 IPv4 地址（4 个最大为 255 的数字，中间用点分隔）即可。你的函数将接收字符串作为输入，并将所有 IP 地址实例替换为字符串 `"|||IP_ADDRESS|||"`。

**Deliverable：**一个函数，将给定字符串中的 IPv4 地址替换为字符串 `"|||IP_ADDRESS|||"`，并返回一个二元组，其中包含新字符串以及被屏蔽的实例数量。实现 adapter `[run_mask_ips]`，并确保它通过 `uv run pytest -k test_mask_ips`。

(d) 当这些过滤器被朴素地应用于训练集时，你认为下游语言模型中可能会出现哪些问题？你会如何缓解这些问题？

**Deliverable：**2-5 句回答。

(e) 在从 WARC 文件中提取出的文本上运行你的 PII 屏蔽函数（通过你先前实现的文本提取函数）。查看 20 个发生替换的随机示例；给出一些误报和漏报的示例。

**Deliverable：**2-5 句回答。

### 2.5 有害内容

来自 Web 的未过滤转储包含大量我们不希望语言模型在推理期间重复的文本。其中一些训练样例甚至可能出人意料地来自通常无害的网站，例如 Wikipedia——例如，用户在若干页面上留下的评论可能相当有毒。尽管本质上不可能围绕“什么是有害的”建立一条清晰边界，许多数据过滤管道仍会对主要包含有害内容的页面进行一定过滤。

识别这类内容有很多方法，包括统计禁用词列表中的词，或基于人工标注构建简单分类器。在本作业的这一部分中，我们将重点识别两类广义有害内容：“Not safe for work”（NSFW）（包括色情、粗俗语言或其他潜在令人不适的内容），以及有毒言论（“粗鲁、不尊重或不合理、很可能使某人退出讨论的语言”[^3]）。我们将使用 Dolma 项目 [1] 提供的 fasttext 预训练模型，判断输入文本片段是否属于这两类之一。这些分类器在 Jigsaw Toxic Comments 数据集上训练，该数据集包含按多种标签分类的 Wikipedia 评论。[^4]

NSFW 分类器可在以下位置下载：

dolma-artifacts.org/.../jigsaw_fasttext_bigrams_nsfw_final.bin。

仇恨言论分类器可在以下位置获得：

dolma-artifacts.org/.../jigsaw_fasttext_bigrams_hatespeech_final.bin。

我们也已将这两个分类器放在共享数据目录中：

- `/shared-data/classifiers/dolma_fasttext_hatespeech_jigsaw_model.bin` 是用于仇恨和有毒言论的预训练分类器
- `/shared-data/classifiers/dolma_fasttext_nsfw_jigsaw_model.bin` 是用于 NSFW 内容的预训练分类器

使用这些模型实现一个函数，该函数接收包含页面内容的 Unicode 字符串，并返回一个标签（例如，`"toxic"`、`"non-toxic"`）以及一个置信度分数。

### Problem (harmful_content): Harmful content（有害内容）（6 分）

(a) 编写一个函数来检测 NSFW 内容。

**Deliverable：**一个函数，将给定字符串标记为包含 NSFW 内容或不包含 NSFW 内容，并返回一个二元组，其中包含标签和置信度分数。实现 adapter `[run_classify_nsfw]`，并确保它通过 `uv run pytest -k test_classify_nsfw`。注意，该测试只是一个 sanity check，取自 Jigsaw 数据集，但绝不意味着你的分类器是准确的；你应当自行验证其准确性。

(b) 编写一个函数来检测有毒言论。

**Deliverable：**一个函数，将给定字符串标记为由有毒言论构成或不构成有毒言论，并返回一个二元组，其中包含标签和置信度分数。实现 adapter `[run_classify_toxic_speech]`，并确保它通过 `uv run pytest -k test_classify_toxic_speech`。同样，该测试只是一个 sanity check，也取自 Jigsaw。

(c) 当这些过滤器被应用于创建训练集时，你认为下游语言模型中可能会出现哪些问题？你会如何缓解这些问题？

**Deliverable：**2-5 句回答。

(d) 在从 WARC 文件中提取出的文本上运行你的有害内容过滤器（通过你先前实现的文本提取函数）。查看 20 个随机示例，并将分类器预测与自己的判断进行比较。报告任何分类器错误。有害文档所占比例是多少？基于你的观察，过滤时应使用怎样的分类器置信度阈值？

**Deliverable：**2-5 句回答。

### 2.6 质量规则

即使按语言过滤页面并移除有害内容之后，剩余页面中仍有相当大一部分对于语言模型训练而言质量较低。同样，尽管“质量”并不容易定义，查看 Common Crawl 示例有助于识别低质量内容的例子，例如：

- 含有付费墙内容的页面
- 破损链接的占位页面
- 登录、注册或联系表单
- 主要为非文本内容的页面，这些内容会在文本提取过程中丢失（例如照片、视频）

Gopher 论文 [2] 描述了一组简单的质量过滤器，用于从 Web 抓取数据中移除类似的低质量文本简单情况。这些过滤器由易于理解的简单启发式规则组成，并且通常能够覆盖许多显然不适合作为训练样例的情形。Gopher 质量过滤器包括若干基于文档长度、单词长度、符号与单词比例，以及某些英语停用词存在性的准则。在本作业中，你将实现 Gopher 论文 [2] 中描述的过滤器的一个子集。具体而言，你应移除满足以下条件的文档：

- 包含少于 50 个或多于 100,000 个单词。
- 平均单词长度不在 3 到 10 个字符范围内。
- 超过 30% 的行以省略号（`"..."`）结尾。
- 含有至少一个字母字符的单词少于 80%。

关于 Gopher 论文所使用的全部质量过滤器的完整描述，请参见其 Appendix A。

### Problem (gopher_quality_filters): Gopher quality filters（Gopher 质量过滤器）（3 分）

(a) 按照上文描述实现（至少）Gopher 质量过滤器的子集。对于将文本分词为单词，你可能会发现 NLTK 包有用（具体而言是 `nltk.word_tokenize`），不过并不要求你使用它。

**Deliverable：**一个函数，以字符串作为唯一参数，并返回一个布尔值，表示该文本是否通过 Gopher 质量过滤器。实现 adapter `[run_gopher_quality_filter]`。然后，确保你的过滤器通过 `uv run pytest -k test_gopher` 中的测试。

(b) 在从 WARC 文件中提取出的文本上运行你的基于规则的质量过滤器（通过你先前实现的文本提取函数）。查看 20 个随机示例，并将过滤器预测与你自己的判断进行比较。评论任何质量过滤器与你判断不同的情况。

**Deliverable：**2-5 句回答。

### 2.7 质量分类器

现在，让我们超越 Gopher 规则所捕获的简单语法质量准则。语言模型训练并不是第一次出现对内容质量进行排序的需求。特别地，文本质量也是信息检索中的一个基本挑战。搜索引擎利用的经典信号之一是 Web 上的链接结构：高质量页面往往链接到其他高质量页面 [3]。OpenAI 在构建 WebText，即 GPT-2 [4] 训练所用的数据集时，使用了类似的洞见：他们收集了 Reddit 评论中链接的页面，并要求这些评论超过最低 “karma” 阈值。Reddit 的另一种替代来源是 Wikipedia，因为 Wikipedia 页面链接的外部来源往往是受信任的页面 [5]。

尽管使用受控来源通常会带来高质量内容，但最终得到的数据集按照今天的标准会偏小（OpenWebText 有 40GB 文本，而 The Pile 大 20 倍以上）。一种方法是将这些参考页面作为正样本，将来自 Common Crawl 的（随机）页面作为负样本，并训练 fastText 分类器。该分类器给出一个质量分数，你可以使用该分数从整个 Common Crawl 中过滤掉页面。设置质量阈值会在 precision 和 recall 之间进行权衡。

在本作业的这一部分中，你将构建一个质量分类器。为了方便起见，我们已从近期的 Wikipedia 转储中提取了参考页面的 URL，并将它们放置在共享数据目录 `/shared-data/wiki/enwiki-20260501-extracted_urls.txt.gz`。该文件包含截至 2026 年 5 月在英语 Wikipedia 页面中找到的外部链接，但我们希望你对这些 URL 进行子采样，以获得训练分类器所需的“高质量”文本正样本。注意，这些正样本仍可能包含不理想的内容，因此应用你已经构建的其他原语（例如语言识别、过滤规则等）来进一步改善其质量可能会有帮助。给定一个 URL 文件，你可以使用 `wget` 以 WARC 格式抓取其内容：

```bash
wget --timeout=5 \
 -i subsampled_positive_urls.txt \
 --warc-file=subsampled_positive_urls.warc \
 -O /dev/null
```

### Problem (quality_classifier): Quality classifier（质量分类器）（15 分）

(a) 训练一个质量分类器，使其在给定文本时返回一个数值质量分数。

**Deliverable：**一个用于下一小题的质量分类器。

(b) 编写一个函数，将页面标记为高质量或低质量，并给出该标签的置信度分数。

**Deliverable：**一个函数，以字符串作为唯一参数，并返回一个二元组，其中包含标签（高质量或非高质量）以及置信度分数。实现 adapter `[run_classify_quality]`。作为 sanity check，请运行 `uv run pytest -k test_classify_quality`，确保它能正确分类我们提供的两个示例。

## 3 去重

Web 包含大量重复内容。有些页面彼此完全重复——可以想象归档页面，或者由标准工具生成的默认页面，例如流行 Web 服务器生成的 404 页面。但大多数重复发生在更细粒度的层面。例如，考虑 Stack Overflow 上所有的问题页面。虽然每个页面都有独特内容（例如问题、评论、答案本身），但所有页面也都有大量冗余内容，例如 header、菜单选项和 footer；当所有这些页面被渲染时，这些内容会以完全相同的形式重复出现。在本节的第一部分中，我们将处理这种精确重复，稍后再了解如何处理近似重复。

### 3.1 精确行去重

一种对精确重复进行去重的简单方法是：在文档中只保留语料库中唯一的行。事实证明，这足以消除很大一部分冗余，例如我们上面提到的 header 和菜单选项。在较简单的情况下，当移除在其他地方完全重复的行后，我们通常会留下每个页面独特的主要内容（例如 StackOverflow 上的问题和答案）。

为了做到这一点，我们可以对语料库进行一次遍历，统计观察到的每一行出现了多少次。然后，在第二次遍历中，通过仅保留唯一行来重写每个文档。

朴素地说，用于保存计数器的数据结构可能需要与语料库中所有唯一行所需存储空间一样多的空间。一个简单的内存效率技巧是，改为使用该行的哈希作为键，使键具有固定大小（而不是依赖于该行的长度）。现在你将实现这一简单的去重方法。

### Problem (exact_deduplication): Exact line deduplication（精确行去重）（3 分）

编写一个函数，接收输入文件路径列表并对它们执行精确行去重。它应先使用哈希来降低内存占用，统计语料库中每一行的频率，然后通过仅保留唯一行来重写每个文件。

**Deliverable：**一个执行精确行去重的函数。你的函数应接收两个参数：(a) 输入文件路径列表，以及 (b) 输出目录。它应将每个输入文件以相同文件名重写到输出目录中，但通过移除在输入文件集合中出现超过一次的行来对内容进行去重。例如，如果输入路径为 `a/1.txt` 和 `a/2.txt`，输出目录为 `b/`，则你的函数应写入文件 `b/1.txt` 和 `b/2.txt`。实现 adapter `[run_exact_line_deduplication]`，并确保它通过 `uv run pytest -k test_exact_line_deduplication`。

### 3.2 MinHash + LSH 文档去重

精确去重对于移除在多个网页中逐字重复的内容很有用，但无法处理文档内容略有差异的情况。例如，考虑软件许可证文档——许可证文档通常由一个模板生成，该模板要求填写年份和软件作者姓名。因此，一个 MIT 许可项目的许可证文件与另一个 MIT 许可项目的许可证文件包含大量相同内容，但它们并不是精确重复。为了移除这类重复的、主要由模板生成的内容，我们需要模糊去重。为了高效执行模糊文档级去重，我们将使用带有局部敏感哈希（LSH）的 minhash。[^5]

为了执行模糊去重，我们将使用一种特定的文档相似性概念：每个文档的 ngrams 之间的 Jaccard 相似度。集合 $S$ 和 $T$ 的 Jaccard 相似度定义为 $\frac{|S \cap T|}{|S \cup T|}$。若要朴素地执行模糊去重，我们可以将每个文档表示为 n-gram 集合，并计算所有文档对之间的 Jaccard 相似度，在超过特定 Jaccard 相似度阈值时将该对标记为重复。然而，该方法对于大型文档集合（例如 Common Crawl）并不实际。此外，朴素地存储 n-gram 集合会比一个文档本身占用多得多的内存。

**MinHashing**

为了解决内存问题，我们用签名替换文档 n-gram 集合表示。具体而言，我们希望构造这样的签名：如果比较两个文档的签名，我们就能得到两个文档各自 ngrams 集合的 Jaccard 相似度近似值。Minhash 签名满足这些性质。为了计算文档 n-grams 集合 $S = \{s_1, ..., s_n\}$ 的 minhash 签名，我们需要 $k$ 个不同的哈希函数 $h_1, ..., h_k$。[^6] 每个哈希函数将一个 n-gram 映射为一个整数。给定一个哈希函数 $h_i$，文档 ngrams 集合 $S$ 的 minhash 为 $\mathrm{minhash}(h_i, S) = \min(h_i(s_1), h_i(s_2), ..., h_i(s_n))$。文档 n-grams 集合 $S$ 的签名是 $\mathbb{R}^k$ 中的一个向量，其中每个元素 $i$ 包含 $S$ 在随机哈希函数 $h_i$ 下的 minhash，即 $[\mathrm{minhash}(h_1, S), \mathrm{minhash}(h_2, S), ..., \mathrm{minhash}(h_k, S)]$。

事实证明，对于两个文档 n-gram 集合 $S_1$ 和 $S_2$，这两个集合之间的 Jaccard 相似度可由具有相同 minhash 值的列的比例来近似（证明见 [6] 第 3 章第 3.3.3 节）。例如，给定文档签名 `[1, 2, 3, 2]` 和 `[5, 2, 3, 4]`，由于这些签名的第二列和第三列具有相同的 minhash 值，因此 n-gram 集合之间的 Jaccard 相似度可近似为 `2/4`。

**Locality-sensitive hashing (LSH)**

尽管 minhashing 为我们提供了一种内存高效的文档表示，并能保留任意文档对之间的期望相似度，但我们仍然需要比较所有文档对，以找到相似度最高的文档。LSH 提供了一种高效地将可能具有高相似度的文档分桶的方法。为了将 LSH 应用于我们的文档签名（现在是 $\mathbb{R}^k$ 中的向量），我们会将签名划分为 $b$ 个 band，每个 band 包含 $r$ 个 minhash，并且 $k = br$。例如，如果我们有 100 个元素的文档签名（由 100 个随机哈希函数生成），可以将其划分为 2 个 band，每个 band 50 个 minhash；也可以划分为 4 个 band，每个 band 25 个 minhash；或者划分为 50 个 band，每个 band 2 个 minhash，等等。如果两个文档在某个特定 band 上具有相同的哈希值，则它们会被聚集到同一个 bucket 中，并被视为候选重复项。因此，在签名数量固定的情况下，增加 band 数量会提高 recall 并降低 precision。

对于上述内容的一个具体示例，假设我们有一个文档 $D_1$，其 minhash 签名为 `[1, 2, 3, 4, 5, 6]`，另一个文档 $D_2$ 的 minhash 签名为 `[1, 2, 3, 5, 1, 2]`。如果我们使用 3 个 band，每个 band 包含 2 个 minhash，则 $D_1$ 的第一个 band 是 `[1, 2]`，$D_1$ 的第二个 band 是 `[3, 4]`，$D_1$ 的第三个 band 是 `[5, 6]`。类似地，$D_2$ 的第一个 band 是 `[1, 2]`，$D_2$ 的第二个 band 是 `[3, 5]`，$D_2$ 的第三个 band 是 `[1, 2]`。由于第一个 band 中的哈希匹配（两个文档均为 `[1, 2]`），$D_1$ 和 $D_2$ 会在第一个 band 下聚集在一起。它们不会在任何其他 band 下聚集在一起，因为哈希不匹配。然而，请注意，由于这些文档至少在一个 band 下聚集在一起，因此无论其他 band 是否匹配，它们都会被视为候选重复项。

一旦我们识别出候选重复项，就可以用多种方式处理它们。例如，我们可以计算所有候选重复项之间的 Jaccard 相似度，并将超过设定阈值的文档对标记为重复。

最后，我们跨 bucket 聚类重复文档。例如，假设文档 A 和 B 在一个 bucket 中匹配，并且真实 Jaccard 相似度高于我们的阈值；文档 B 和 C 在另一个 bucket 中匹配，并且真实 Jaccard 相似度也高于我们的阈值。那么，我们会将文档 A、B 和 C 视为一个单一簇。我们会随机移除每个簇中除一个文档之外的所有文档。

### Problem (minhash_deduplication): MinHash + LSH document deduplication（MinHash + LSH 文档去重）（8 分）

编写一个函数，接收输入文件路径列表，并使用 minhash 和 LSH 执行模糊文档去重。具体而言，你的函数应为给定路径列表中的每个文档计算 minhash 签名，使用给定 band 数量的 LSH 识别候选重复项，然后计算候选重复项之间真实的 ngram Jaccard 相似度，并移除超过给定阈值的文档。为了提高 recall（遵循 [7]），在计算 minhash 签名和/或比较 Jaccard 相似度之前，应通过小写化、移除标点、规范化空白、移除重音符号以及应用 NFD unicode normalization 对文本进行归一化。

**Deliverable：**一个执行模糊文档去重的函数。你的函数应至少接收以下参数：(a) 输入文件路径列表，(b) 用于计算 minhash 签名的哈希数量，(c) 用于 LSH 的 band 数量，(d) 用于计算 minhash 签名的 n-gram 长度（按单词计），以及 (e) 输出目录。你可以假设用于计算 minhash 签名的哈希数量能够被用于 LSH 的 band 数量整除。

你的函数应将每个输入文件以相同文件名重写到输出目录中，但只写入满足以下条件的文档：(a) 不是候选重复项，和/或 (b) 从聚类后的 bucket 中随机选择并保留的文档。例如，如果输入路径为 `a/1.txt` 和 `a/2.txt`，输出目录为 `b/`，则你的函数应写入文件 `b/1.txt` 和 `b/2.txt`。实现 adapter `[run_minhash_deduplication]`，并确保它通过 `uv run pytest -k test_minhash_deduplication`。

## 4 过滤用于语言建模的数据

现在我们已经实现了多种用于过滤 Web crawl 数据的原语，让我们将它们投入使用，并生成一些语言建模训练数据。

在作业的这一部分中，你的目标是过滤一组 CC WET 文件，以生成语言建模训练数据。你将从已经针对英语过滤过的 WET 文件开始：我们保留了根据 fastText `lid.176.bin` 语言识别模型得到的英语概率至少为 70% 的文档。我们已将这些英语过滤后的 WET 文件放在 `/shared-data/english-wet-data` 下，供你作为起点使用。

具体而言，你的目标是过滤 CC 转储数据来创建语言建模数据，使训练后的 Transformer 语言模型在 Paloma benchmark [8] 的 C4 100 domains 子集上最小化验证困惑度。你不应修改模型架构或训练过程，因为目标是构造最佳数据。该数据集包含来自 C4 语言建模数据集 [9] 中 100 个最常见域名的样本。我们已将该数据的副本放在 `/shared-data/tokenized_paloma_c4_100_domains_validation.bin`（使用 GPT-2 tokenizer 进行 tokenization）——你可能会发现浏览该数据以了解其外观很有用。可以使用以下代码加载它：

```python
import numpy as np

data = np.fromfile(
    "/shared-data/tokenized_paloma_c4_100_domains_validation.bin",
    dtype=np.uint16
)

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
print(tokenizer.decode(data[0:2000]))
```

给定你过滤后的数据集，你将在该数据上训练一个 GPT-2 small 形状的模型，并在 C4 100 上评估其困惑度。

我们注意到，你可以在构造过滤器或分类器以处理 CC WET 文件时使用 Paloma 验证数据，但不允许将任何验证数据逐字复制到你的训练数据中。语言模型绝不应看到来自验证集的任何数据。

即使是这组 WET 文件，也包含相当数量的数据。为了高效处理这些数据，我们建议尽可能使用多进程。特别地，你可能会发现 Python 的 `concurrent.futures` 或 `multiprocessing` API 很有帮助。下面，我们给出一个使用 `concurrent.futures` 在多个进程上并行化函数的最小示例：

```python
import concurrent.futures
import os

from tqdm import tqdm


def process_single_wet_file(input_path: str, output_path: str):
    # TODO: read input path, process the input, and write the output to output_path
    return output_path


# Set up the executor
num_cpus = len(os.sched_getaffinity(0))
executor = concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus)

wet_filepaths = ["a.warc.wet.gz", "b.warc.wet.gz", "c.warc.wet.gz"]
output_directory_path = "/path/to/output_directory/"

futures = []
for wet_filepath in wet_filepaths:
    # For each warc.wet.gz filepath, submit a job to the executor and get a future back
    wet_filename = str(pathlib.Path(wet_filepath).name)
    future = executor.submit(
        process_single_wet_file,
        wet_filepath,
        os.path.join(output_directory_path, wet_filename)
    )
    # Store the futures
    futures.append(future)

# Iterate over the completed futures as they finish, using a progress bar
# to keep track of progress.
for future in tqdm(
    concurrent.futures.as_completed(futures),
    total=len(wet_filepaths),
):
    output_file = future.result()
    print(f"Output file written: {output_file}")
```

对大文本块使用本地并行机制，例如 `concurrent.futures` 或 `multiprocessing`。若要在 modal 上跨 worker 并行化，请对你的输入调用 `.map(...)`。

我们还建议使用 FastWARC 库遍历每个 WET 文件中的记录，并使用 `tldextract` 库提取用于过滤的域名。具体而言，以下类可能会有帮助：

```python
from fastwarc.warc import ArchiveIterator, WarcRecordType
from tldextract import TLDExtract
```

### Problem (filter_data): Filter data for language modeling（过滤语言建模数据）（6 分）

(a) 编写一个脚本，从一组 Common Crawl WET 文件（位于 `/shared-data/english-wet-data` 下）中过滤语言建模数据。你可以自由应用我们在作业前面部分实现的任何原语，也可以自由探索其他过滤器和生成数据的方法（例如，基于 n-gram 语言模型困惑度进行过滤）。你的目标是生成一份数据，使得在其上训练的模型在 Paloma benchmark 的 C4 100 domains 子集上最小化困惑度。

再次提醒，你可以在构造过滤器或分类器以处理 CC WET 文件时使用 Paloma 验证数据，但不允许将任何验证数据逐字复制到你的训练数据中。

你的脚本应报告你使用的每个过滤器保留的示例数量，这样你可以了解各过滤器如何影响最终输出数据。

**Deliverable：**一个脚本（或一系列脚本），以并行方式过滤提供的 CC WET 文件，以生成语言建模数据。书面说明每个过滤步骤移除的被丢弃示例比例。

(b) 过滤提供的 WET 文件（原始为 2,500 个 raw WET 文件）需要多长时间？过滤整个 Common Crawl 转储需要多长时间？

**Deliverable：**数据过滤管道的运行时间。

现在我们已经生成了一些语言建模数据，让我们查看它，以便更好地理解其内容。

### Problem (inspect_filtered_data): Inspect filtered data（检查过滤后的数据）（4 分）

(a) 从你过滤后的数据集中随机抽取五个示例。评论它们的质量，以及它们是否适合用于语言建模，特别是考虑到我们的目标是在 C4 100 domains benchmark 上最小化困惑度。

**Deliverable：**来自最终过滤数据的五个随机示例。由于文档可能很长，只展示相关摘录即可。对于每个示例，给出 1-2 句描述，说明该示例以及它是否值得用于语言建模。

(b) 抽取五个被你的过滤脚本移除和/或修改的 CC WET。你的过滤过程中的哪一部分移除或修改了这些文档？你认为它们的移除和/或修改是否合理？

**Deliverable：**来自原始 WET 的五个随机被丢弃示例。由于文档可能很长，只展示相关摘录即可。对于每个示例，给出 1-2 句描述，说明该示例以及它的移除是否合理。

(c) 如果你上面的分析促使你进一步修改数据管道，可以在训练模型之前进行这些修改。报告你尝试过的任何数据更改和/或数据迭代。

**Deliverable：**描述你尝试过的数据更改和/或数据迭代。

在语言模型训练数据上训练之前，我们需要对它进行 tokenization。通过 `transformers` 使用 GPT-2 tokenizer，将你的过滤数据编码为用于训练语言模型的整数 ID 序列。不要忘记在每个文档之后包含 GPT-2 end-of-sequence token `<|endoftext|>`。下面是一些启动代码：

```python
import multiprocessing
import numpy as np
from tqdm import tqdm

from transformers import AutoTokenizer

input_path = "path/to/your/filtered/data"
output_path = "path/to/your/tokenized/data"

tokenizer = AutoTokenizer.from_pretrained("gpt2")


def tokenize_line_and_add_eos(line):
    return tokenizer.encode(line) + [tokenizer.eos_token_id]


with open(input_path) as f:
    lines = f.readlines()

pool = multiprocessing.Pool(multiprocessing.cpu_count())
chunksize = 100
results = []
for result in tqdm(
    pool.imap(tokenize_line_and_add_eos, lines, chunksize=chunksize),
    total=len(lines),
    desc="Tokenizing lines"
):
    results.append(result)

pool.close()
pool.join()

# Flatten the list of ids and convert to numpy array
all_ids = [token_id for sublist in results for token_id in sublist]
print(f"Tokenized and encoded {input_path} into {len(all_ids)} tokens")
ids_array = np.array(all_ids, dtype=np.uint16)
ids_array.tofile(output_path)
```

### Problem (tokenize_data): Tokenize data（数据 tokenization）（2 分）

编写一个脚本，对你过滤后的数据进行 tokenization 和序列化。确保按照上面的示例代码进行序列化，即使用 `ids_array.tofile(output_path)`，其中 `ids_array` 是一个由整数 ID 构成的 `np.uint16` numpy array。这确保它与提供的训练脚本兼容。

你的过滤数据集中有多少个 token？

**Deliverable：**一个对你过滤后的数据进行 tokenization 和序列化的脚本，以及你生成的数据集中的 token 数量。

现在我们已经对数据集进行了 tokenization，就可以在其上训练模型。我们将在生成的数据上训练一个 GPT-2 small 形状的模型，约 430M 参数，并定期在 C4 100 domains 数据集上测量验证性能。训练运行使用 8 张 B200 GPU、数据并行、每设备 batch size 为 128，以及 16,384 个优化步骤。在 context length 为 512 的情况下，这对应于训练期间采样的大约 $2^{33} = 8.6B$ 个训练 token。我们使用此设置的训练运行大约在 2:00 小时内完成。

使用 `scripts/train.py` 中可用的 `train.py` 脚本启动训练。[^7] 你可以在 `cs336_basics/train_config.py` 中查看将使用的超参数。使用 `--train-bin` 传入你 tokenized training data 的路径：

```bash
uv run modal run scripts/train.py --train-bin /root/data/your_data.bin
```

再次强调，本作业的目标是优化数据以最小化验证损失，而不是通过修改模型和/或优化过程来最小化损失，因此不要修改训练过程或训练脚本。

然后，你可以使用以下命令从保存的模型中抽样：

```bash
uv run modal run scripts/generate_with_gpt2_tok.py --model-path /root/data/output/your_data
```

### Problem (train_model): Train model（训练模型）（8 分）

在你的 tokenized dataset 上训练一个语言模型（GPT-2 small 形状，约 430M 参数）。定期在 C4 100 domains 上测量验证损失。你的模型达到的最佳验证损失是多少？

**Deliverable：**记录到的最佳验证损失、相应的学习曲线，以及对你所做工作的描述。

## Bibliography（参考文献）

[1] L. Soldaini et al., “Dolma: An Open Corpus of Three Trillion Tokens for Language Model Pretraining Research,” arXiv preprint arXiv:2402.00159, 2024.

[2] J. W. Rae et al., “Scaling language models: Methods, analysis & insights from training gopher,” arXiv preprint arXiv:2112.11446, 2021.

[3] L. Page, S. Brin, R. Motwani, and T. Winograd, “The pagerank citation ranking: Bring order to the web,” in Proc. of the 7th International World Wide Web Conf, 1998.

[4] A. Radford et al., “Language models are unsupervised multitask learners,” OpenAI blog, vol. 1, no. 8, p. 9, 2019.

[5] H. Touvron et al., “LLaMA: Open and Efficient Foundation Language Models.” 2023.

[6] J. Leskovec, A. Rajaraman, and J. D. Ullman, Mining of Massive Datasets, 2nd ed. USA: Cambridge University Press, 2014.

[7] G. Penedo et al., “The RefinedWeb Dataset for Falcon LLM: Outperforming Curated Corpora with Web Data, and Web Data Only.” 2023.

[8] I. Magnusson et al., “Paloma: A Benchmark for Evaluating Language Model Fit.” 2023.

[9] C. Raffel et al., “Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer,” Journal of Machine Learning Research, vol. 21, no. 140, pp. 1-67, 2020, [Online]. Available: http://jmlr.org/papers/v21/20-074.html

[^1]: https://commoncrawl.org/
[^2]: https://resiliparse.chatnoir.eu/en/stable/index.html
[^3]: https://current.withgoogle.com/the-current/toxicity/
[^4]: https://paperswithcode.com/dataset/toxic-comment-classification-challenge
[^5]: 如需更深入地了解 LSH 和 minhashing，可参见 J. Leskovec、A. Rajaraman 和 J. D. Ullman [6] 的第 3 章。该章可在线获取：infolab.stanford.edu/~ullman/mmds/ch3n.pdf。
[^6]: 这 $k$ 个不同的哈希函数可以来自同一个族，但使用不同种子。例如，MurmurHash3 是一个哈希函数族，使用特定种子即可实例化该族中的特定哈希函数。
[^7]: https://github.com/stanford-cs336/assignment4-data/blob/master/scripts/train.py
