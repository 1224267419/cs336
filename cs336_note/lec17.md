# Lecture 17 课程报告：从图像编码到 Omni Model 的多模态模型学习讲义

## 0. 资料来源与改写说明

本报告是在用户上传的 Lecture 17 原始中文课程报告基础上进行系统性改写，原始报告主题为 **CS336 Lecture 17: Multimodal Models / 多模态模型**，内容覆盖 CLIP、SigLIP、LLaVA、LLaVA OneVision、Qwen-VL、Qwen2-VL、Qwen3-VL、Chameleon，以及多模态训练中的系统与数据问题。 

本次改写还核对了 Stanford CS336 GitHub 仓库。该仓库说明 Spring 2026 CS336 lectures 包含课程材料，可执行讲义命名为 `lecture_XX.py`，非可执行讲义命名为 `lecture_XX.pdf`；仓库目录中 Lecture 17 对应文件是 `lecture_17.py`。

`lecture_17.py` 的主流程标题为 “Lecture 17: multimodal models”，课程顺序依次调用 `clip()`、`siglip()`、`llava()`、`llava_onevision()`、`qwen_vl()`、`qwen2_vl()`、`qwen3_vl()`、`chameleon()`，这也是本报告保持的讲解顺序。

本报告中的引用说明如下：

- **【字幕†source】**：沿用原报告中的字幕来源标记，表示该内容来自课程讲解或课堂问答。
- **【Python代码†source】**：沿用原报告中的 GitHub `lecture_17.py` 或代码截图来源标记。
- **【代码截图†source】**：表示 `lecture_17.py` 中插入的教学代码截图，例如 `clip-code.png`、`siglip-code.png`。
- 本次新增的解释性内容会用 **补充说明**、**延伸理解**、**工程直觉** 或 **学习提示** 标出，不把它们伪装成课程原文。
- 原报告没有提供逐行字幕编号，因此本报告保留原始引用格式，不强行制造不存在的精确字幕行号。

------

## 1. 学习导读

这一讲要解决的问题是：**当 Transformer 本质上处理 token 序列时，我们如何让它理解图像、视频等非文本模态？进一步地，如何走向能够输入和输出任意模态组合的 omni model？**

### 1.1 本讲要解决的核心问题

此前课程主要围绕语言模型展开，典型形式是：
$$
\text{text} \Rightarrow \text{text}
$$
也就是说，模型输入文本，输出文本。但现实世界不是纯文本的。人类交流和感知还包含图像、视频、音频、空间布局、动作、符号、图表、网页界面等。因此，多模态模型的目标不是简单地“给语言模型加一张图片”，而是要回答：
$$
\text{non-text modality} \Rightarrow \text{tokens} \Rightarrow \text{Transformer}
$$
也就是说，**非文本模态必须被转换成 Transformer 能处理的 token 表示**。`lecture_17.py` 明确提出：Transformer 很有效，因此要使用 Transformer；但 Transformer “说 token”，所以必须把一切转换成 token。

本讲围绕两个问题展开：

1. 如何输入非文本数据，例如理解图像？
2. 如何输出非文本数据，例如生成图像、音频或视频？

课程主要关注第一个问题，即 **图像和视频如何被编码并注入语言模型**；最后通过 Chameleon 讨论一种更接近 omni model 的统一离散 token 路线。【字幕†source】【Python代码†source】

### 1.2 本讲在 CS336 课程中的位置

CS336 的主线是从零理解并构建语言模型，包括数据、tokenization、Transformer、训练、评估、系统优化和后训练等内容。Lecture 17 将这些语言模型基础扩展到多模态场景。

本讲和前面内容的联系可以这样理解：

- **与 tokenization 的联系**：文本要先分词，图像和视频也要“token 化”，只是 token 可以是连续 embedding，也可以是离散 code。
- **与 Transformer 的联系**：多模态模型仍大量依赖 Transformer，只是输入序列中混入了视觉 token。
- **与训练目标的联系**：CLIP/SigLIP 使用对比学习或二分类学习建立图文对齐；LLaVA/Qwen 系列则使用自回归语言建模训练回答。
- **与系统优化的联系**：图像和视频 token 数更多，I/O 更重，训练稳定性更复杂。
- **与后训练的联系**：视觉指令数据、SFT、蒸馏和 RL 在 VLM 中同样重要。

### 1.3 本讲知识结构

| 模块             | 解决的问题                      | 核心概念                                                     | 与后续内容的关系              |
| ---------------- | ------------------------------- | ------------------------------------------------------------ | ----------------------------- |
| 多模态建模总问题 | Transformer 如何处理非文本模态  | token 化、连续 token、离散 token、omni model                 | 奠定整讲主线                  |
| CLIP             | 如何学习图文共享语义空间        | 双编码器、对比学习、双向 softmax                             | 后续 VLM 的视觉编码基础       |
| SigLIP           | 如何缓解 CLIP 对大 batch 的依赖 | sigmoid pairwise loss、二分类目标                            | 提供更易扩展的图文编码器      |
| VLM 标准模板     | 如何把视觉编码接入 LLM          | vision encoder + projector/adapter + LLM                     | LLaVA、Qwen-VL 的共同架构     |
| LLaVA            | 如何用开放组件构建视觉问答助手  | CLIP + linear projector + Vicuna + GPT-4 合成数据            | 展示 VLM 的基本 recipe        |
| LLaVA OneVision  | 如何支持单图、多图和视频        | AnyRes、token budget、任务迁移                               | 扩展 VLM 到更复杂输入         |
| Qwen-VL 系列     | 如何工程化提升 VLM 能力         | cross-attention adapter、dynamic resolution、MRoPE、DeepStack | 展示现代 VLM 的架构与训练细节 |
| Chameleon        | 如何统一理解和生成图文          | VQ-VAE、离散图像 token、自回归建模                           | 走向 omni model 的另一条路线  |
| 系统与训练问题   | 多模态训练为什么更难            | I/O、token 混合、loss 权重、训练稳定性                       | 连接大模型训练工程实践        |

### 1.4 学习本讲前需要知道什么

本讲最相关的前置知识包括：

- **Transformer 基础**：self-attention、位置编码、token embedding。
- **语言模型训练目标**：next-token prediction、自回归分解、交叉熵损失。
- **向量表示与相似度**：embedding、cosine similarity、L2 normalization。
- **softmax 与 sigmoid**：多分类交叉熵和二分类 logistic loss 的区别。
- **Vision Transformer 基础**：图像 patch、patch embedding、位置编码。
- **LLM 微调基础**：SFT、instruction tuning、adapter/projector。
- **RoPE 基础**：旋转位置编码如何表达相对位置。
- **基本系统直觉**：batch size、分布式训练、长上下文、数据加载瓶颈。

------

## 2. 第一部分：多模态建模的根本问题——让非文本模态变成 token

### 2.1 问题引入

课程从语言模型的局限开始：传统语言模型处理的是文本到文本。但现实世界中的信息并不都以文本形式存在。图片、视频、音频、图表、网页截图和屏幕操作都包含大量信息。

因此，多模态模型的第一步不是“设计一个新的神奇模型”，而是回答一个更基础的问题：
$$
\text{图像 / 视频 / 音频} \quad \如何变成 \quad \text{Transformer 可处理的 token 序列？}
$$
`lecture_17.py` 在开头明确列出两个问题：如何输入非文本数据，例如理解图像；如何输出非文本数据，例如生成音频。

### 2.2 概念定义：什么是多模态模型？

**直觉解释**：多模态模型是能同时处理多种信息形式的模型。例如，它不仅能读文字，还能看图、看视频、理解图表，甚至生成图像或音频。

**正式定义**：在本讲语境中，多模态模型可以理解为：
$$
p_\theta(\text{output modalities} \mid \text{input modalities})
$$
其中输入和输出都可以是文本、图像、视频、音频或它们的组合。

课程提出的更远目标是 **omni model / 全模态模型**：

- 输入任意模态组合，用于理解；
- 输出任意模态组合，用于生成。

例如：
$$
(\text{image}, \text{text}) \Rightarrow \text{text}
$$
是视觉问答；
$$
(\text{text}) \Rightarrow \text{image}
$$
是文本生成图像；
$$
(\text{video}, \text{text}) \Rightarrow \text{text}
$$
是视频问答；
$$
(\text{text}, \text{image}) \Rightarrow (\text{text}, \text{image})
$$
则接近图文交错生成。【字幕†source】【Python代码†source】

### 2.3 方法与原理：连续 token 与离散 token

非文本模态转换为 token，大致有两条路线。

#### 路线一：连续 token

连续 token 是向量，例如：
$$
\mathbf{z}_1, \mathbf{z}_2, \dots, \mathbf{z}_m \in \mathbb{R}^d
$$
图像经过 CLIP、SigLIP、ViT 等视觉编码器后，可以得到一组连续 embedding。它们不是词表 ID，而是浮点向量。

这种路线适合理解任务，因为连续向量可以保留较丰富的语义信息。

#### 路线二：离散 token

离散 token 是整数 ID，例如：
$$
s_t \in \{1, 2, \dots, K\}
$$
文本 token 本来就是离散 ID。图像也可以通过 VQ-VAE 等视觉 tokenizer 映射为 codebook index。Chameleon 就属于这种路线。

这种路线的优势是形式统一：文本和图像都变成离散 token 后，可以直接使用自回归语言模型目标。

### 2.4 公式与推导：统一 token 视角

如果模型接收的是文本 token，输入序列是：
$$
x = (x_1, x_2, \dots, x_T)
$$
如果模型接收的是图像连续 token，可以写成：
$$
Z_I = (\mathbf{z}_1, \mathbf{z}_2, \dots, \mathbf{z}_m)
$$
如果再加上文本指令：
$$
X = [\mathbf{z}_1, \dots, \mathbf{z}_m, \mathbf{e}_{x_1}, \dots, \mathbf{e}_{x_T}]
$$
其中：

- `\mathbf{z}_i` 是视觉编码器输出的视觉 token；
- `\mathbf{e}_{x_t}` 是文本 token embedding；
- `X` 是送入 Transformer 的混合序列。

如果使用离散图像 token，则图文可以统一为：
$$
s = (s_1, s_2, \dots, s_T)
$$
训练目标就是标准自回归分解：
$$
p_\theta(s) = \prod_{t=1}^{T} p_\theta(s_t \mid s_{<t})
$$
这正是 Chameleon 一类模型的思想。

### 2.5 课堂例子与解释

课程用 “text ⇒ text” 对比 “world is multimodal” 来引入问题：文本只是世界信息的一种表达形式，而图像、视频、音频同样重要。【字幕†source】

这个例子想说明：**语言模型的成功不是多模态问题的终点，而是起点**。我们已经知道 Transformer 能在 token 序列上扩展得很好；下一步就是把其他模态合理地变成 token。

### 2.6 代码 / PDF 对应说明

`lecture_17.py` 的主函数先说明语言模型已处理文本到文本，然后展示世界是多模态的，再提出 omni model 目标。随后它列出 Transformer 需要 token，因此要把一切转换为 token，并进入 CLIP 与 SigLIP。

### 2.7 学习提示

本节最重要的结论是：**多模态建模的第一性问题是 tokenization / encoding，而不是模型名字。**

容易误解的地方是：不要以为“能输入图片”就是多模态能力的全部。真正的关键是图像如何被编码、视觉 token 如何与语言 token 对齐、训练目标如何让模型学会跨模态关系。

一句话总结：**多模态模型首先要解决“非文本信息如何变成 Transformer 能理解的 token”这个问题。**

------

## 3. 第二部分：CLIP——用对比学习建立图文共享语义空间

### 3.1 问题引入

传统视觉模型通常依赖人工标注图像，例如 ImageNet 分类标签。但人工标注成本高、类别有限、语义表达也比较窄。

CLIP 的问题意识是：互联网上已经存在大量图像和文字的自然配对，例如网页图片、caption、alt text、标题、上下文描述。能否像语言模型利用网页文本那样，利用这些 noisy image-text pairs 训练一个通用视觉表示？

`lecture_17.py` 将 CLIP 的背景概括为：传统计算机视觉模型在 annotated images 上训练，而 CLIP 试图利用更大规模的 image-caption pairs。

CLIP 全称：
$$
\text{CLIP} = \text{Contrastive Language-Image Pretraining}
$$
即 **对比式语言-图像预训练**。

### 3.2 概念定义：什么是对比学习？

**直觉解释**：给模型一批图像和文本，让它学会“哪张图和哪句话是一对”。正确配对要更相似，错误配对要更不相似。

**正式定义**：设一个 batch 中有 `n` 个图文对：
$$
\{(I_1, T_1), (I_2, T_2), \dots, (I_n, T_n)\}
$$
其中：

- `I_i` 是第 `i` 张图像；
- `T_i` 是与它对应的文本；
- `(I_i, T_i)` 是正例；
- `(I_i, T_j), j \neq i` 是负例；
- `(I_j, T_i), j \neq i` 也是负例。

CLIP 要学习两个编码器：
$$
f_I(I_i) = \mathbf{u}_i
$$
其中：

- `f_I` 是图像编码器；
- `f_T` 是文本编码器；
- `\mathbf{u}_i` 是图像 embedding；
- `\mathbf{v}_i` 是文本 embedding。

训练目标是让：
$$
\mathbf{u}_i^\top \mathbf{v}_i
$$
大于：
$$
\mathbf{u}_i^\top \mathbf{v}_j, \quad j \neq i
$$
也就是正确图文对更接近，错误图文对更远离。

### 3.3 方法与原理：CLIP 的双编码器结构

CLIP 包含两个主要模块：

1. **图像编码器 image encoder**
   - 可以是 ResNet；
   - 也可以是 Vision Transformer；
   - 课程中强调最佳模型是 ViT-L/14@336px，即 large ViT、14×14 patch、336×336 输入分辨率。
2. **文本编码器 text encoder**
   - 使用 GPT-2 风格 Transformer；
   - 输入 `[BOS] ... [EOS]`；
   - 取最高层 `[EOS]` activation 作为整段文本表示。

CLIP 的关键不是让图像编码器直接生成文本，而是让图像和文本都投影到一个共享 embedding space。这样，图像分类、检索、zero-shot 任务都可以转化为“比较图文相似度”。

### 3.4 图像预处理：为什么 resize + center crop 会影响能力？

CLIP 需要固定尺寸输入。课程中介绍的处理方式是：

1. 任意尺寸 `W × H` 图像先用 bicubic interpolation 缩放，使短边为 336；
2. 再 center crop 成 `336 × 336`。

这对 ImageNet 分类很合理，因为 ImageNet 中主体物体往往在图像中心。但它也带来一个问题：**边缘信息和小字信息可能被丢掉**。

**工程直觉**：
 如果任务是判断图片中是否有狗，中心裁剪通常没问题；如果任务是读一张论文截图、发票、网页界面或医学影像，缩放和裁剪就可能严重损失信息。因此后续 LLaVA OneVision、Qwen2-VL 等模型会强调 AnyRes 和动态分辨率。

### 3.5 Vision Transformer：把图像 patch 当成视觉 token

ViT 的核心思想是把图像切成 patch。若输入图像为：
$$
336 \times 336
$$
patch 大小为：
$$
14 \times 14
$$
那么每边 patch 数为：
$$
336 / 14 = 24
$$
总 patch 数为：
$$
24 \times 24 = 576
$$
每个 patch 有 RGB 三通道，可以展平成向量，再经过线性层变成 embedding：
$$
\mathbf{x}_p = W_{\text{patch}} \cdot \text{flatten}(P_p) + \mathbf{b}
$$
然后加入位置编码：
$$
\mathbf{h}_p^{(0)} = \mathbf{x}_p + \mathbf{e}_p^{\text{pos}}
$$
最后送入 Transformer encoder。

**直觉解释**：文本 Transformer 把句子切成词或子词；ViT 把图像切成小方块。每个小方块就是一个视觉 token。

**常见误区**：patch token 不等于像素。一个 patch 是局部区域的压缩表示，不是一个单独像素。patch 越大，token 越少，但细节越容易丢；patch 越小，细节更多，但序列更长、计算更贵。

### 3.6 CLIP 损失函数：双向 softmax 对比学习

CLIP 先对图像和文本 embedding 做 L2 normalize：
$$
\hat{\mathbf{u}}_i =
\frac{\mathbf{u}_i}{\|\mathbf{u}_i\|_2},
\quad
\hat{\mathbf{v}}_j =
\frac{\mathbf{v}_j}{\|\mathbf{v}_j\|_2}
$$
然后计算图像 `i` 与文本 `j` 的相似度：
$$
s_{ij} =
\exp(t) \cdot
\hat{\mathbf{u}}_i^\top \hat{\mathbf{v}}_j
$$
其中：

- `\hat{\mathbf{u}}_i` 是归一化后的图像 embedding；
- `\hat{\mathbf{v}}_j` 是归一化后的文本 embedding；
- `\hat{\mathbf{u}}_i^\top \hat{\mathbf{v}}_j` 是 cosine similarity；
- `t` 是可学习的 log temperature；
- `\exp(t)` 控制 logits 的尺度。

对每张图像 `I_i`，正确文本是 `T_i`，所以 image-to-text loss 是：
$$
\mathcal{L}_{I \rightarrow T}
=
-\frac{1}{n}
\sum_{i=1}^{n}
\log
\frac{\exp(s_{ii})}
{\sum_{j=1}^{n}\exp(s_{ij})}
$$
对每段文本 `T_i`，正确图像是 `I_i`，所以 text-to-image loss 是：
$$
\mathcal{L}_{T \rightarrow I}
=
-\frac{1}{n}
\sum_{i=1}^{n}
\log
\frac{\exp(s_{ii})}
{\sum_{j=1}^{n}\exp(s_{ji})}
$$
最终损失为两者平均：
$$
\mathcal{L}_{\text{CLIP}}
=
\frac{1}{2}
\left(
\mathcal{L}_{I \rightarrow T}
+
\mathcal{L}_{T \rightarrow I}
\right)
$$
`lecture_17.py` 中 CLIP 方法部分明确写到：取一个图文 batch，编码每张图和每段文本；每张图要偏好自己的 aligned text，每段文本要偏好自己的 aligned image，并插入了 `clip-code.png` 代码截图。

### 3.7 代码对应：`clip-code.png` 在做什么？

原报告指出，CLIP 代码截图展示了以下步骤：【代码截图†source】

| 代码步骤                    | 数学含义                                       | 学习重点                       |
| --------------------------- | ---------------------------------------------- | ------------------------------ |
| extract image/text features | `I_f = f_I(I), T_f = f_T(T)`                   | 两个编码器分别处理图像和文本   |
| project to joint embedding  | `I_e = normalize(I_f W_i)`                     | 把不同模态映射到同一空间       |
| compute logits              | `logits = I_e T_e^\top \cdot \exp(t)`          | 得到所有图文配对的相似度矩阵   |
| symmetric cross entropy     | `CE(logits, labels)` 与 `CE(logits.T, labels)` | 图像到文本、文本到图像双向匹配 |

这里的 labels 是：
$$
[0, 1, 2, \dots, n-1]
$$
表示第 `i` 张图的正确文本是第 `i` 段文本。

**学习提示**：CLIP loss 的关键是 `n × n` 相似度矩阵。对角线是正例，非对角线是负例。

### 3.8 Zero-shot 分类：为什么 CLIP 能不训练分类头？

CLIP 的代表性结果之一是 zero-shot ImageNet 表现强。课程中提到，zero-shot CLIP 在 ImageNet 上超过了用 1.2M ImageNet 标注图像训练的 ResNet-50。

Zero-shot 分类的做法是：

1. 把每个类别写成 prompt，例如：

$$
\text{``a photo of a dog''}
$$

1. 用文本编码器得到类别文本 embedding：

$$
\hat{\mathbf{v}}_c
$$

1. 用图像编码器得到图像 embedding：

$$
\hat{\mathbf{u}}
$$

1. 选择相似度最大的类别：

$$
\hat{c}
=
\arg\max_c
\hat{\mathbf{u}}^\top \hat{\mathbf{v}}_c
$$

**直觉解释**：CLIP 没有学一个固定的 “dog/cat/car” 分类头，而是学了图文之间的语义对应。因此只要类别名能写成文本，它就能通过相似度比较完成分类。

### 3.9 CLIP 的局限

CLIP 很强，但课程强调它有几个重要限制：

1. **依赖大 batch size**
    负样本来自 batch 内其他图文对。batch 太小，负样本不足。
2. **softmax 绑定整个 batch**
    损失函数对 batch size 敏感。改变 batch size 不只是工程变化，也改变了训练目标的统计性质。
3. **偏高层语义，不够细粒度**
    CLIP 设计选择主要面向图像分类，不擅长保留 OCR、小物体、细粒度空间布局。
4. **训练数据 noisy**
    网页 caption 不一定精确描述图像。

`lecture_17.py` 的 CLIP 小结也写到：图像编码捕获 noisy text 提供的语义，设计决策基于 image classification，不够 fine-grained，并且技术上需要 large batch sizes 和 full-batch softmax。

### 3.10 学习提示

本节复习时要抓住三点：

- CLIP 不是图像生成模型，而是图文对齐模型。
- CLIP 的核心对象是 `n × n` 相似度矩阵。
- CLIP 的能力来自大规模 noisy 图文对和双向对比学习。

一句话总结：**CLIP 用对比学习把图像和文本放到同一个语义空间，使图像理解可以转化为图文相似度匹配。**

------

## 4. 第三部分：SigLIP——把 full-batch softmax 改成 pairwise sigmoid

### 4.1 问题引入

CLIP 的成功依赖大 batch。原因是每个 batch 内的其他图文对都充当负样本，softmax 要在整行或整列上归一化。

这带来两个问题：

- batch size 小时负样本不足；
- batch size 大时分布式通信和矩阵计算成本很高。

SigLIP 的动机是：**能否不用 full-batch softmax，而把每个图文 pair 是否匹配看成独立二分类问题？**

`lecture_17.py` 将差异概括为：CLIP 是 multiclass classification；SigLIP 是 binary classification，判断 text-image pair 是否 aligned。

### 4.2 概念定义：pairwise binary objective

设 batch 中有 `n` 个图文对。对任意组合 `(I_i, T_j)`，定义标签：
$$
y_{ij}
=
\begin{cases}
+1, & i = j \\
-1, & i \neq j
\end{cases}
$$
含义是：

- 对角线 `(I_i, T_i)` 是正例；
- 非对角线 `(I_i, T_j), i \neq j` 是负例。

相似度 logit 定义为：
$$
s_{ij}
=
\exp(t') \cdot
\hat{\mathbf{u}}_i^\top
\hat{\mathbf{v}}_j
+
b
$$
其中：

- `t'` 是可学习 temperature；
- `b` 是可学习 bias；
- `\hat{\mathbf{u}}_i` 和 `\hat{\mathbf{v}}_j` 是归一化后的图像和文本 embedding。

SigLIP 使用 logistic loss：
$$
\mathcal{L}_{\text{SigLIP}}
=
-\frac{1}{n}
\sum_{i=1}^{n}
\sum_{j=1}^{n}
\log
\sigma(y_{ij}s_{ij})
$$
其中：
$$
\sigma(x)
=
\frac{1}{1+\exp(-x)}
$$

### 4.3 直觉解释：SigLIP 到底改了什么？

CLIP 问的是：

> 在这一整批文本里，哪一个文本最匹配这张图？

SigLIP 问的是：

> 这张图和这段文本是不是一对？

这看起来只是损失函数变化，但工程意义很大。CLIP 的 softmax 必须在完整 batch 上归一化；SigLIP 的每个 pair loss 更容易分块计算。

### 4.4 代码对应：`siglip-code.png`

原报告指出，SigLIP 代码截图中的关键变量是：
$$
\text{labels}
=
2I_n - \mathbf{1}
$$
这里：

- `I_n` 是单位矩阵；
- 对角线为 `+1`；
- 非对角线为 `-1`。

loss 是：
$$
-\sum \log \sigma(y_{ij}s_{ij}) / n
$$
这段代码展示的是：先 normalize 图像和文本 embedding，再计算 pairwise logits，最后对每个 pair 做 sigmoid loss。【代码截图†source】

| 课程知识点     | 对应代码函数 / 变量                  | 代码功能               | 理论原理                               | 学习重点                            |
| -------------- | ------------------------------------ | ---------------------- | -------------------------------------- | ----------------------------------- |
| 图文 pair 判断 | `labels = 2 * eye(n) - ones(n)`      | 构造正负样本标签       | 对角线为正例，非对角线为负例           | 理解 pairwise binary classification |
| 相似度计算     | `logits = dot(zimg, ztxt.T) * t + b` | 计算所有图文组合得分   | cosine similarity + temperature + bias | 与 CLIP 相似度矩阵对应              |
| Sigmoid loss   | `log_sigmoid(labels * logits)`       | 判断 pair 是否 aligned | logistic loss                          | 理解为何不需要 full softmax         |

### 4.5 为什么 SigLIP 更易扩展？

SigLIP 的优势不是完全不需要负样本，而是 **负样本的组织方式更灵活**。

CLIP 的 softmax 需要整行或整列一起归一化：
$$
\frac{\exp(s_{ii})}{\sum_j \exp(s_{ij})}
$$
SigLIP 则对每个 pair 做：
$$
\log \sigma(y_{ij}s_{ij})
$$
因此：

- 损失和 batch size 的耦合更弱；
- 分布式训练时可以分块计算；
- 小 batch 时仍然是在估计同一个二分类目标，只是负样本覆盖少一些；
- 大 batch 不再是目标定义本身的必要组成部分。

`lecture_17.py` 也总结 SigLIP decouple batch size from loss，并指出它在小于 16K batch size 时优于 CLIP，batch size 可扩展到 1M，但 32K 通常已经足够。

### 4.6 数据与效率

课程讲义中提到 SigLIP 使用 WebLI 级别的大规模数据，包括 OCR、多语言支持和质量过滤。`lecture_17.py` 写到 WebLI 是十亿级图文对，来自互联网，使用自动 OCR 抽取图中文字，保留最高质量的 10%，支持 100 种语言。

讲义还对比训练效率：

- CLIP：256 个 TPUv3 训练 10 天；
- SigLIP：32 个 TPUv4 训练 5 天。

### 4.7 学习提示

SigLIP 最容易和 CLIP 混淆。复习时请抓住：

- CLIP：多分类 softmax，整批样本相互竞争。
- SigLIP：pairwise sigmoid，每个图文对独立判断是否匹配。
- SigLIP 的核心价值是训练目标更易分布式扩展，并降低 batch size 对目标的影响。

一句话总结：**SigLIP 把“从一批文本里选正确文本”改成“判断每个图文 pair 是否匹配”，从而让图文对齐训练更易扩展。**

------

## 5. 第四部分：VLM 标准模板——Vision Encoder + Projector + LLM

### 5.1 问题引入

CLIP 和 SigLIP 能把图像编码成向量，但它们本身不是完整的聊天式视觉语言模型。用户真正想要的是：
$$
(\text{image}, \text{text instruction})
\Rightarrow
\text{text answer}
$$
例如：

- “这张图里发生了什么？”
- “图中表格的趋势是什么？”
- “这张截图里的按钮在哪里？”
- “这段视频里第二个人做了什么？”

这就需要把视觉编码接入语言模型。

### 5.2 概念定义：VLM 的标准结构

标准 VLM 模板是：
$$
\text{image}
\overset{\text{vision encoder}}{\longrightarrow}
\mathbf{z}_{\text{vision}}
\overset{\text{projector / adapter}}{\longrightarrow}
\mathbf{e}_{\text{LM-space}}
\overset{\text{LLM}}{\longrightarrow}
\text{text}
$$
也就是：

1. **Vision encoder**：把图像变成视觉特征；
2. **Projector / adapter**：把视觉特征映射到语言模型 embedding 空间；
3. **LLM**：把视觉 token 和文本 token 共同作为上下文，生成回答。

`lecture_17.py` 在 CLIP/SigLIP 之后进入 “Injecting image encodings into LLMs”，随后讲解 LLaVA、LLaVA OneVision、Qwen-VL、Qwen2-VL 和 Qwen3-VL。

### 5.3 方法与原理：视觉 token 如何进入 LLM？

设视觉编码器输出：
$$
\mathbf{z}_I = f_I(I)
$$
projector 将其映射到语言模型 embedding 空间：
$$
\mathbf{e}_I = g_\phi(\mathbf{z}_I)
$$
其中：

- `f_I` 是视觉编码器；
- `g_\phi` 是 projector 或 adapter；
- `\mathbf{e}_I` 是可与文本 token embedding 拼接的视觉 token。

若文本输入为：
$$
x = (x_1, x_2, \dots, x_k)
$$
文本 embedding 为：
$$
(\mathbf{e}_{x_1}, \mathbf{e}_{x_2}, \dots, \mathbf{e}_{x_k})
$$
则 LLM 的输入可以写成：
$$
[
\mathbf{e}_{I,1}, \dots, \mathbf{e}_{I,m},
\mathbf{e}_{x_1}, \dots, \mathbf{e}_{x_k}
]
$$
生成目标仍是自回归语言模型目标：
$$
p(y_1, \dots, y_T \mid I, x)
=
\prod_{t=1}^{T}
p(y_t \mid I, x, y_{<t})
$$

### 5.4 工程直觉：为什么需要 projector？

CLIP 的图像 embedding 和 LLM 的 token embedding 虽然都是向量，但它们不在同一个空间中。

可以把这件事类比为：

- CLIP 视觉编码器说的是“视觉语义语言”；
- LLM 说的是“文本 embedding 语言”；
- projector 是翻译器，把视觉编码翻译成 LLM 能读懂的 embedding。

如果没有 projector，LLM 看到的视觉向量可能像乱码。

### 5.5 学习提示

VLM 的核心模板非常重要：
$$
\text{vision encoder}
+
\text{projector / adapter}
+
\text{LLM}
$$
后续 LLaVA 和 Qwen 系列看起来细节很多，但基本都在围绕这三个部件做改进。

一句话总结：**VLM 的常见做法不是从零训练一个全新模型，而是把已有视觉编码器和已有 LLM 通过 projector/adapter 缝合起来。**

------

## 6. 第五部分：LLaVA——开放 VLM recipe 的代表

### 6.1 问题引入

有了 CLIP 视觉编码器和强大的 LLM，下一步是构建能回答图像问题的助手。LLaVA 的意义在于，它展示了一个相对简单、开放、可复现的视觉语言模型 recipe：
$$
\text{CLIP}
+
\text{linear projector}
+
\text{Vicuna}
+
\text{visual instruction data}
$$
`lecture_17.py` 中 LLaVA 部分写到：vision encoder 是 CLIP，text decoder 是 Vicuna，数据来自 MS COCO，并使用 GPT-4 根据 captions 或 detected objects 生成问题或对话，最后形成 158K examples。

### 6.2 概念定义：LLaVA 的组成

LLaVA 全称：
$$
\text{LLaVA}
=
\text{Large Language and Vision Assistant}
$$
其基本组件包括：

| 组件               | 在 LLaVA 中的选择        | 作用                                 |
| ------------------ | ------------------------ | ------------------------------------ |
| Vision encoder     | CLIP ViT-L/14            | 把图像编码成视觉向量                 |
| Projector          | 线性投影矩阵 `W`         | 把视觉向量映射到 LLM embedding space |
| LLM / text decoder | Vicuna                   | 根据图像和文本指令生成回答           |
| 数据               | MS COCO + GPT-4 合成问答 | 训练视觉指令遵循能力                 |

### 6.3 LLaVA 数据生成

LLaVA 使用 MS COCO 图像及其人工 caption、bounding boxes 或 detected objects，再提示 GPT-4 生成：

1. 对话 questions / conversations；
2. 详细描述 detailed descriptions；
3. 复杂推理 complex reasoning。

生成文本再与原始图像配对，得到视觉指令数据。`lecture_17.py` 明确写到 LLaVA 将 GPT-4 生成内容与原图配对，并形成 158K examples。

**补充说明**：
 这里的关键不是 GPT-4 作为最终模型，而是 GPT-4 被用作数据生成器。LLaVA 展示了一个重要策略：用强模型合成指令数据，训练较开放的视觉语言模型。

### 6.4 LLaVA 的两阶段训练

设视觉编码器输出：
$$
\mathbf{z}_I = f_I(I)
$$
线性 projector 为：
$$
\mathbf{e}_I = W\mathbf{z}_I
$$
其中：

- `f_I` 是 CLIP；
- `W` 是可训练投影矩阵；
- `\mathbf{e}_I` 是 LLM 可读的视觉 token embedding。

#### Stage 1：Alignment

冻结：

- vision encoder；
- language model。

只训练：

- projector `W`。

目标是让 `W\mathbf{z}_I` 落入 LLM 可以理解的 embedding 空间。

**直觉解释**：第一阶段像是训练一个翻译器。视觉编码器已经会看图，LLM 已经会说话，但二者语言不通。`W` 的作用是把视觉表示翻译成 LLM 能理解的 token embedding。

#### Stage 2：Fine-tuning

冻结：

- vision encoder。

训练：

- projector `W`；
- language model。

此时模型学习根据图像和用户指令生成回答。`lecture_17.py` 对 LLaVA training 的描述正是：Stage 1 freeze vision encoder and language model, only train W；Stage 2 freeze vision encoder and train W and language model。

### 6.5 LLaVA 的意义

LLaVA 的价值不只是效果，而是展示了一个清晰的开放 VLM 构建流程：
$$
\text{pretrained vision encoder}
+
\text{pretrained LLM}
+
\text{lightweight alignment}
+
\text{instruction tuning}
$$
课程中提到 LLaVA 能展示一定视觉推理能力，例如理解 unusual situation，但整体视觉能力仍不如强闭源模型。【字幕†source】

### 6.6 学习提示

复习 LLaVA 时要抓住：

- LLaVA 没有从零训练视觉语言模型；
- 它复用 CLIP 和 Vicuna；
- 核心桥梁是 linear projector；
- 训练分 alignment 和 fine-tuning；
- GPT-4 在这里主要用于合成视觉指令数据。

一句话总结：**LLaVA 证明了“视觉编码器 + projector + LLM + 合成指令数据”可以构建开放视觉语言助手。**

------

## 7. 第六部分：LLaVA OneVision——从单图扩展到多图和视频

### 7.1 问题引入

原始 LLaVA 主要处理单张图像。但现实任务更复杂：

- 用户可能上传多张图进行比较；
- 任务可能是视频理解；
- 图像可能是高分辨率文档、图表或网页截图；
- OCR 和细节理解需要比 CLIP 更精细的视觉信息。

LLaVA OneVision 因此扩展目标：支持 single image、multiple images 和 video。

`lecture_17.py` 写到 LLaVA OneVision 处理 multiple images 和 video，vision encoder 使用 SigLIP，text decoder 使用 Qwen-2 72B，projector 使用 2-layer MLP。

### 7.2 架构升级

| 组件           | LLaVA            | LLaVA OneVision            |
| -------------- | ---------------- | -------------------------- |
| Vision encoder | CLIP             | SigLIP                     |
| Text decoder   | Vicuna           | Qwen-2 72B                 |
| Projector      | linear projector | 2-layer MLP                |
| 输入类型       | 单图为主         | 单图、多图、视频           |
| 分辨率策略     | 固定 resize/crop | AnyRes / token budget 平衡 |

### 7.3 AnyRes：为什么高分辨率很重要？

CLIP 的固定 `336 × 336` 输入对分类足够，但对 OCR、文档理解、图表阅读可能不够。LLaVA OneVision 采用 AnyRes 思路：

1. 保留全图下采样版本，用于捕捉整体布局；
2. 将原图切成 `a × b` 个 tile；
3. 每个 tile 匹配视觉编码器输入分辨率；
4. 分别编码每个 tile；
5. 将视觉 token 拼接；
6. token 过多时，用 bilinear interpolation 或下采样减少 token 数。

形式化表示为：
$$
I
\rightarrow
\{I_{1,1}, I_{1,2}, \dots, I_{a,b}\}
$$
视觉表示为：
$$
Z_I
=
[
f_I(I_{\text{global}}),
f_I(I_{1,1}),
\dots,
f_I(I_{a,b})
]
$$
`lecture_17.py` 明确说明：为保留高分辨率信息，图像被分成 `a × b` pieces，各自匹配 vision encoder 分辨率，编码后 concatenate；token 太多时用 bilinear interpolation。

### 7.4 单图、多图、视频的 token budget 平衡

LLaVA OneVision 需要处理三类输入：

- single image；
- multiple images；
- video。

如果视频每帧都用高分辨率编码，token 数会爆炸。因此讲义提出目标：**让不同模态产生大致相近长度的 token 序列**。

具体策略是：

| 输入类型 | 分辨率策略                 | 原因                       |
| -------- | -------------------------- | -------------------------- |
| 单图     | 使用更高分辨率             | 可以仔细看细节             |
| 多图     | 每张图使用 base resolution | 避免多图 token 过多        |
| 视频     | 每帧使用更低分辨率         | 帧数多，必须控制上下文长度 |

**工程直觉**：
 多模态训练不是“信息越多越好”。如果视频 token 数远大于文本或图像，它会主导 attention、loss 和训练资源，反而损害整体平衡。

### 7.5 多模态任务之间的迁移

LLaVA OneVision 课程部分还强调 transfer between modalities。`lecture_17.py` 举了几类迁移：单图 diagrams/charts 数据可以泛化到多图；单图 OCR 与多图关系推理可组合成 GUI agent 能力；单图 visual prompting 可以泛化到视频。

这些例子说明：模型不是只在孤立任务上记模板，而是在多样任务中学习可迁移能力。

### 7.6 学习提示

LLaVA OneVision 的重点不是“换了更大模型”这么简单，而是：

- 高分辨率图像需要特殊处理；
- 视频需要控制每帧 token 数；
- 多模态任务之间可能产生迁移；
- 大量工作在数据策划和任务设计上。

一句话总结：**LLaVA OneVision 将 VLM 从单图问答扩展到单图、多图和视频，并通过 AnyRes 与 token budget 平衡解决细节和上下文长度的矛盾。**

------

## 8. 第七部分：Qwen-VL、Qwen2-VL、Qwen3-VL——现代 VLM 的工程化演进

本节要解决的问题是：**当 VLM 从研究原型走向更强的通用模型时，架构、分辨率、位置编码、长上下文和训练流程会如何演进？**

------

### 8.1 Qwen-VL：cross-attention adapter 与三阶段训练

#### 8.1.1 问题引入

LLaVA 使用线性 projector，结构简单。Qwen-VL 则使用更系统的 adapter 和训练流程，希望提升 OCR、中文理解、区域定位、结构化输出等能力。

#### 8.1.2 架构

`lecture_17.py` 中 Qwen-VL 架构包括：OpenCLIP 的 ViT-bigC 视觉编码器、one-layer cross-attention adapter、2D positional encodings、固定长度 256 的映射，以及 `<img>`、`<box>`、`<ref>` 特殊 token。

| 组件                    | Qwen-VL 设计                     | 作用                     |
| ----------------------- | -------------------------------- | ------------------------ |
| Vision encoder          | OpenCLIP ViT-bigC, 14×14 patches | 提取视觉特征             |
| Adapter                 | 一层 cross-attention             | 融合视觉信息并映射到 LLM |
| 2D positional encodings | 高宽空间位置                     | 保留图像空间结构         |
| Special tokens          | `<img>`, `<box>`, `<ref>`        | 支持图像、框、引用区域   |

#### 8.1.3 三阶段训练

Qwen-VL 的训练分三阶段：

1. **Stage 1：大规模低质量数据**
   - 冻结 LM；
   - 训练 vision encoder + adapter；
   - 目标是让视觉模块接入语言模型。
2. **Stage 2：高质量任务特定数据**
   - 提升分辨率；
   - 训练全部参数；
   - 使用 VQA、chart QA 等数据提升能力。
3. **Stage 3：instruction tuning**
   - 冻结 visual encoder；
   - 训练 adapter + LM；
   - 提升指令遵循能力。

`lecture_17.py` 中对应写到：Stage 1 freeze LM, train vision encoder + adaptor；Stage 2 increase resolution and train all parameters；Stage 3 freeze visual encoder, train adaptor + LM。

#### 8.1.4 学习提示

Qwen-VL 和 LLaVA 的重要区别是：Qwen-VL 的 adapter 更复杂，训练流程更工程化，视觉编码器也会在第一阶段参与训练，而不是完全固定。

------

### 8.2 Qwen2-VL：动态分辨率与 MRoPE

#### 8.2.1 问题引入

不同图片的信息密度差异巨大。一张小公式图可能只需要少量视觉 token；一张高分辨率网页截图可能需要大量视觉 token。如果所有图像都用固定分辨率，会造成两类问题：

- 小图被浪费 token；
- 大图丢失细节。

Qwen2-VL 的关键升级是 **dynamic resolution**。`lecture_17.py` 写到 Qwen2-VL 的 key 是 dynamic resolution to handle varying resolutions。

#### 8.2.2 Patch 编码与 token 压缩

Qwen2-VL 的处理流程可以理解为：

1. 每个 `224 × 224` patch 用 ViT/14 编码；
2. 因为 `224 / 14 = 16`，所以原始 grid 是：

$$
16 \times 16 = 256
$$

1. 每 `2 × 2` 个视觉 token 压缩成 1 个；
2. 得到约 `8 × 8 = 64` 个 token；
3. 加上额外特殊 token 后，每个 patch 约 66 tokens。

`lecture_17.py` 对应写到：Each 224 × 224 patch encoded with ViT/14, compress every 2×2 => 66 tokens；视频采样 2 frames/sec，最大 16384 tokens。

#### 8.2.3 MRoPE：多模态旋转位置编码

普通 RoPE 处理一维文本序列。对位置 `m`，RoPE 可抽象为：
$$
\text{RoPE}(\mathbf{x}, m)
=
R(m)\mathbf{x}
$$
它的重要性质是：
$$
(R(m)\mathbf{q})^\top(R(n)\mathbf{k})
=
\mathbf{q}^\top R(n-m)\mathbf{k}
$$
这意味着 attention score 可以感知相对位置 `n - m`。

但图像不是一维，视频也不是一维：

- 图像有高度 `h` 和宽度 `w`；
- 视频还有时间 `t`。

因此 Qwen2-VL 使用 MRoPE，为视觉 token 分配三维坐标：
$$
(t, h, w)
$$
然后让 attention 感知：
$$
(\Delta t, \Delta h, \Delta w)
$$
**直觉解释**：
 文本中 token 的距离是“第几个词到第几个词”。图像中 token 的距离是“上、下、左、右”。视频中还要知道“前一帧、后一帧”。MRoPE 就是把 RoPE 从一维扩展到多维。

#### 8.2.4 学习提示

Qwen2-VL 重点是两件事：

- dynamic resolution：不同大小图像用不同 token 数；
- MRoPE：视觉和视频 token 需要二维/三维位置。

一句话总结：**Qwen2-VL 将 VLM 的输入处理从固定分辨率推进到动态分辨率，并用 MRoPE 处理图像和视频的空间时间位置。**

------

### 8.3 Qwen3-VL：长上下文、Interleaved MRoPE、DeepStack 与训练稳定性

#### 8.3.1 问题引入

Qwen3-VL 不是完全换一种 VLM 范式，而是在多个细节上强化现代 VLM：

- 更强语言模型；
- 更长上下文；
- 更强视觉编码器；
- 更好的多维位置编码；
- 显式视频时间戳；
- 更平衡的 per-token loss；
- 更深的视觉语言融合；
- 更复杂的 pre-training 和 post-training。

`lecture_17.py` 写到 Qwen3-VL 使用 Qwen-3 dense 和 MoE 模型，最长 256K context，使用 SigLIP-2 vision encoder，Interleaved MRoPE，显式 video timestamps，square-root-normalized per-token loss，DeepStack adapter，以及多阶段训练。

#### 8.3.2 Interleaved MRoPE：为什么要交错？

如果把 RoPE 维度简单分成三块：
$$
[t,t,t,t,\; w,w,w,w,\; h,h,h,h]
$$
就可能出现问题：RoPE 不同维度对应不同频率，如果时间轴总是分到低频，宽度或高度总是分到高频，那么不同轴的表达能力不均衡。

Qwen3-VL 改成 interleaved MRoPE：
$$
[t,w,h,\; t,w,h,\; t,w,h,\; t,w,h]
$$
这样 time、width、height 都能接触低频和高频维度。`lecture_17.py` 明确写到 Interleaved MRoPE 会把 temporal、width、height 分散到 low- and high-frequency bands，并给出 `[t w h t w h ...]` 与 `[t t ... w w ... h h ...]` 的对比。

#### 8.3.3 显式 video timestamps

仅靠位置编码，模型知道第几帧在前、第几帧在后，但用户的问题常常是自然语言时间表达，例如：

> “2 秒后发生了什么？”

如果视频 token 附近显式插入 “0 seconds”“2 seconds” 这类时间 token，模型就更容易把语言中的时间表达与视频帧对齐。

`lecture_17.py` 中写到 Qwen3-VL adds explicit video timestamps as separate tokens rather than only in positional embeddings。

#### 8.3.4 Square-root-normalized per-token loss

多模态样本长度差异很大：

- 文本样本可能较短；
- 单图样本可能中等；
- 视频样本可能极长。

如果直接对 token loss 求和，长视频样本会主导训练。一个可理解的归一化形式是：

普通 sum loss：
$$
L_i
=
\sum_{t=1}^{T_i}
\ell_{i,t}
$$
平方根归一化：
$$
\tilde{L}_i
=
\frac{1}{\sqrt{T_i}}
\sum_{t=1}^{T_i}
\ell_{i,t}
$$
其中：

- `T_i` 是样本长度；
- `\ell_{i,t}` 是第 `t` 个 token 的 loss。

这样长样本仍然有更高权重，但权重增长从 `T_i` 降到 `\sqrt{T_i}`。

`lecture_17.py` 将其描述为 square-root-normalized per-token loss，用于平衡 text and multimodal data，因为 video examples are long，不希望它们 dominate。

#### 8.3.5 DeepStack：更深层视觉语言融合

早期 VLM 常把视觉编码器输出看成一串视觉 token，然后在输入层接入 LLM。DeepStack 的思想是：不要只在输入层融合，而是把视觉信息注入多个语言模型层。

`lecture_17.py` 写到 DeepStack 是 cross-layer fusion，用于 inject visual information into multiple layers。

**工程直觉**：
 如果只在输入层接入视觉 token，LLM 后续层需要自己逐层传播视觉信息。DeepStack 则更像在多个层级持续提醒模型“视觉信息在这里”，有利于深层融合。

#### 8.3.6 训练流程

Qwen3-VL 的训练流程包括：

**Pre-training：4 stages**

1. train adapter；
2. train all parameters on 8K length；
3. train all parameters on 32K length；
4. train all parameters on 256K length。

**Post-training：3 stages**

1. SFT on long CoT data；
2. knowledge distillation；
3. RL。

这些阶段在 `lecture_17.py` 中被列为 Qwen3-VL training 部分。

#### 8.3.7 学习提示

Qwen3-VL 的学习重点是：现代 VLM 的进步往往来自很多“看似小但有效”的细节组合，而不是单一神奇模块。

一句话总结：**Qwen3-VL 展示了现代 VLM 的系统工程化方向：长上下文、动态视觉输入、多维位置编码、loss 平衡和深层融合共同决定最终能力。**

------

## 9. 第八部分：Chameleon——把所有模态离散化成 token

### 9.1 问题引入

前面的 VLM 大多是：
$$
\text{image/video input} + \text{text input}
\Rightarrow
\text{text output}
$$
它们可以理解图片和视频，但通常不能直接生成图片。如果要生成图像，往往还要接 diffusion model 或其他生成模块。

Chameleon 提出另一条路线：**把文本和图像都映射成离散 token，然后用同一个 autoregressive Transformer 建模。**

`lecture_17.py` 写到：Chameleon maps everything into discrete tokens，优势是 can analyze and generate images in a uniform way。

### 9.2 概念定义：统一离散 token 建模

Chameleon 的目标可以写成：
$$
\text{text tokens} + \text{image tokens}
\Rightarrow
\text{next token prediction}
$$
一旦文本和图像都被变成离散 token 序列：
$$
s = (s_1, s_2, \dots, s_T)
$$
就可以用标准语言模型目标：
$$
p(s)
=
\prod_{t=1}^{T}
p(s_t \mid s_{<t})
$$
**直觉解释**：
 这相当于把图像也“分词”。如果图像可以被分成视觉词表中的 token，那么 Transformer 就可以像预测下一个词一样预测下一个图像 token。

### 9.3 VQ-VAE：图像如何变成离散 token？

Chameleon 需要视觉 tokenizer。课程讲到 VQ-VAE，即 Vector Quantized Variational Autoencoder。

给定图像 `x`，encoder 先输出连续 latent：
$$
\mathbf{z}_e(x)
$$
有一个 codebook：
$$
\mathcal{E}
=
\{\mathbf{e}_1,\mathbf{e}_2,\dots,\mathbf{e}_K\}
$$
对每个 latent vector，选择最近的 codebook entry：
$$
k^*
=
\arg\min_k
\left\|
\mathbf{z}_e(x)
-
\mathbf{e}_k
\right\|_2^2
$$
量化后得到：
$$
\mathbf{z}_q(x)
=
\mathbf{e}_{k^*}
$$
decoder 再重建图像：
$$
\hat{x}
=
D(\mathbf{z}_q(x))
$$
基本重建损失可以写为：
$$
\mathcal{L}_{\text{recon}}
=
\|x-\hat{x}\|^2
$$
实际 VQ-VAE 通常还会加入 codebook loss 和 commitment loss：
$$
\mathcal{L}
=
\|x-\hat{x}\|^2
+
\|\text{sg}[\mathbf{z}_e(x)]-\mathbf{e}\|_2^2
+
\beta
\|\mathbf{z}_e(x)-\text{sg}[\mathbf{e}]\|_2^2
$$
其中：

- `\text{sg}` 表示 stop-gradient；
- 第二项更新 codebook；
- 第三项鼓励 encoder 输出靠近 codebook entry；
- `\beta` 控制 commitment 强度。

`lecture_17.py` 写到 VQ-VAE 的思想是 map image to a discrete codebook, decode back to image and minimize reconstruction loss；Chameleon 将 `512 × 512` 图像编码成 1024 tokens，codebook size 为 8192。

### 9.4 Chameleon 的训练数据与目标

`lecture_17.py` 写到 Chameleon 训练分两阶段：

- Stage 1，占 80%，大规模 unsupervised，包括 2.9T text tokens、1.5T text/image tokens、400B interleaved text/image tokens；
- Stage 2，占 20%，一半继续用 Stage 1 数据，一半用 high-quality data。

训练目标仍是自回归 next-token prediction：
$$
\mathcal{L}_{\text{AR}}
=
-\sum_{t=1}^{T}
\log p_\theta(s_t \mid s_{<t})
$$
这里 `s_t` 可能是文本 token，也可能是图像 token。

### 9.5 训练稳定性问题

Chameleon 的形式很优雅，但课程强调训练并不简单。原因是文本 token 和图像 token 的统计性质不同：

| 类型       | 特点                       | 后果                                 |
| ---------- | -------------------------- | ------------------------------------ |
| 文本 token | entropy 较低，语义结构强   | 下一个 token 相对更可预测            |
| 图像 token | entropy 较高，局部细节复杂 | 颜色、纹理、边界更难精确预测         |
| 混合训练   | loss 分布差异大            | norm growth、logit drift、训练不稳定 |

`lecture_17.py` 写到：text tokens have low entropy, image tokens have high entropy，会导致 norm growth 和 logit drift；修复包括 QK norm 和 z-loss regularization。

#### QK norm

QK norm 用于控制 attention 中 query/key 的范数，避免 attention logits 过大。

attention logits 通常是：
$$
\frac{QK^\top}{\sqrt{d}}
$$
如果 `Q` 或 `K` 范数增长，attention logits 会变得过大，使 softmax 过于尖锐，训练不稳定。

#### z-loss regularization

z-loss 惩罚 logits scale 或 log-partition 的漂移，使输出分布的尺度更稳定。

**学习提示**：
 不要以为把图像离散化后，它就“和文本一样了”。图像 token 与文本 token 在 entropy、局部相关性和重建难度上仍然很不同。

### 9.6 Chameleon 的优缺点

#### 优点

Chameleon 的优点是统一：
$$
\text{all modalities}
\Rightarrow
\text{discrete token sequence}
\Rightarrow
\text{autoregressive Transformer}
$$
这让图文交错输入和图文交错输出在同一模型框架内变得自然。

#### 缺点

1. **离散化会损失信息**
    OCR、小文字、细纹理、精细布局可能在 quantization 中丢失。
2. **图像生成质量未必优于专用生成模型**
    diffusion models 通常更擅长处理图像高频细节。
3. **多模态训练更不稳定**
    文本 token 和图像 token 的统计性质不同，混合训练容易出问题。

`lecture_17.py` 的 Chameleon summary 写到：它 elegant，因为只是 autoregressive modeling of discrete tokens；但 not as performant，因为 discretization loses information，尤其是 OCR；多模态训练也 tricky。

### 9.7 学习提示

复习 Chameleon 时要抓住：

- 它不是标准 “vision encoder + LLM” 路线；
- 它把图像也离散化成 token；
- 优点是统一理解和生成；
- 缺点是信息损失和训练稳定性。

一句话总结：**Chameleon 是从语言模型视角最统一的多模态路线之一，但图像离散化带来的信息损失和训练不稳定仍是关键挑战。**

------

## 10. Python 代码示例详解

### 10.1 文件整体结构

本讲对应的 GitHub 文件是：

```
lecture_17.py
```

它不是完整训练脚本，而是一个用 `edtrace` 组织的可执行讲义。GitHub README 说明可执行讲义命名为 `lecture_XX.py`，可以编译生成 trace 文件并在本地页面查看。

文件结构大致如下：

```
from edtrace import text, image, link
from lecture_util import article_link, post_link

def main():
    text("## Lecture 17: multimodal models")
    ...
    clip()
    siglip()
    llava()
    llava_onevision()
    qwen_vl()
    qwen2_vl()
    qwen3_vl()
    chameleon()
```

其主函数调用顺序正对应课程讲解顺序：先介绍多模态问题，再讲图像编码 CLIP/SigLIP，再讲视觉编码注入 LLM 的系列模型，最后讲 Chameleon。

### 10.2 主要函数列表

| 函数                | 对应课程模块    | 主要内容                                             | 学习作用              |
| ------------------- | --------------- | ---------------------------------------------------- | --------------------- |
| `main()`            | 全讲主线        | 多模态目标、token 化问题、课程模块顺序               | 理解课程结构          |
| `clip()`            | CLIP            | 图文对比学习、数据、编码器、zero-shot、局限          | 理解图文共享语义空间  |
| `siglip()`          | SigLIP          | sigmoid loss、WebLI、效率、batch size                | 理解 CLIP loss 的替代 |
| `llava()`           | LLaVA           | CLIP + Vicuna + projector + GPT-4 合成数据           | 理解基础 VLM recipe   |
| `llava_onevision()` | LLaVA OneVision | 多图、视频、AnyRes、token budget、迁移               | 理解复杂输入处理      |
| `qwen_vl()`         | Qwen-VL         | cross-attention adapter、special tokens、三阶段训练  | 理解工程化 VLM        |
| `qwen2_vl()`        | Qwen2-VL        | dynamic resolution、MRoPE、视频采样                  | 理解分辨率和位置编码  |
| `qwen3_vl()`        | Qwen3-VL        | long context、SigLIP-2、Interleaved MRoPE、DeepStack | 理解现代 VLM 改进     |
| `chameleon()`       | Chameleon       | 离散图像 token、VQ-VAE、自回归训练、稳定性           | 理解 omni model 方向  |

### 10.3 关键函数逐段解析

#### `clip()`

`clip()` 先引入 CLIP 的背景，再讲方法、数据、图像处理、vision encoder、text encoder、headline result、ablation 和 summary。

它服务的课程概念是：
$$
\text{image-text contrastive learning}
$$
关键教学代码截图是 `clip-code.png`，它将公式：
$$
\mathcal{L}_{\text{CLIP}}
=
\frac{1}{2}
(\mathcal{L}_{I \rightarrow T}
+
\mathcal{L}_{T \rightarrow I})
$$
转化成可执行伪代码流程。

#### `siglip()`

`siglip()` 对比 CLIP 和 SigLIP：

- CLIP：multiclass classification；
- SigLIP：binary classification for aligned or not。

同时讲 WebLI 数据、OCR、多语言、训练效率和 batch size。

它对应的核心公式是：
$$
\mathcal{L}_{\text{SigLIP}}
=
-\frac{1}{n}
\sum_{i=1}^{n}
\sum_{j=1}^{n}
\log \sigma(y_{ij}s_{ij})
$$

#### `llava()`

`llava()` 展示标准 VLM recipe：
$$
\text{CLIP} + \text{linear projection} + \text{Vicuna}
$$
并说明数据如何通过 GPT-4 从 MS COCO captions 或 detected objects 生成问答和对话。

其对应课程概念是：
$$
\text{vision encoder} + \text{projector} + \text{LLM}
$$

#### `llava_onevision()`

`llava_onevision()` 讲多图和视频，尤其强调：

- high resolution 对 OCR 重要；
- CLIP resize/crop 会损失信息；
- AnyRes 将图像分块编码；
- 单图、多图、视频要控制 token budget；
- 不同任务之间存在迁移。

#### `qwen_vl()`、`qwen2_vl()`、`qwen3_vl()`

这三个函数展示 Qwen-VL 系列的演进：

- Qwen-VL：cross-attention adapter、special tokens、三阶段训练；
- Qwen2-VL：dynamic resolution、MRoPE；
- Qwen3-VL：long context、SigLIP-2、Interleaved MRoPE、timestamps、square-root loss、DeepStack、多阶段训练。

#### `chameleon()`

`chameleon()` 展示不同于 VLM 注入路线的统一离散 token 路线：
$$
\text{text tokens} + \text{image tokens}
\Rightarrow
\text{autoregressive modeling}
$$
其中 VQ-VAE 将图像映射到离散 codebook，Chameleon 用自回归 Transformer 建模图文 token 序列。

### 10.4 代码与课程知识点对应表

| 课程知识点        | 对应代码函数 / 变量 / 片段            | 代码功能                                 | 对应理论原理                   | 在课程中的作用       |
| ----------------- | ------------------------------------- | ---------------------------------------- | ------------------------------ | -------------------- |
| 多模态 token 化   | `main()` 中关于 tokens 的 `text(...)` | 引出非文本模态必须 token 化              | Transformer 输入是 token 序列  | 奠定整讲主线         |
| CLIP 对比学习     | `clip()`、`clip-code.png`             | 计算图文相似度矩阵与双向 CE              | contrastive learning           | 建立图文共享空间     |
| SigLIP 二分类目标 | `siglip()`、`siglip-code.png`         | 对每个 pair 做 sigmoid loss              | binary logistic loss           | 缓解 batch size 依赖 |
| VLM 标准模板      | `llava()`                             | CLIP 输出经 `W` 投影到 LM 空间           | representation alignment       | 构建视觉问答模型     |
| AnyRes            | `llava_onevision()`                   | 高分辨率图像分块编码                     | token budget 与细节保留        | 支持 OCR、多图、视频 |
| Qwen-VL adapter   | `qwen_vl()`                           | cross-attention adapter + special tokens | 视觉语言融合                   | 提升结构化视觉能力   |
| MRoPE             | `qwen2_vl()`                          | 为图像/视频引入多维位置编码              | RoPE 多维扩展                  | 支持空间时间理解     |
| DeepStack         | `qwen3_vl()`                          | 跨层注入视觉信息                         | cross-layer fusion             | 加深视觉语言融合     |
| Chameleon         | `chameleon()`                         | 图像离散 token + AR 训练                 | VQ-VAE + next-token prediction | 走向统一生成与理解   |

------

## 11. PDF / 幻灯片 / 图表解析

原报告说明 Lecture 17 对应 GitHub 中的主要资料是 `lecture_17.py`，不是 `lecture_17.pdf`。GitHub 仓库目录也显示 Lecture 17 文件为 `lecture_17.py`。

因此，本节将 `lecture_17.py` 中插入的图片和代码截图视为“图表 / 幻灯片材料”来解析。

### 11.1 主要图表总览

| 图表 / 图片                      | 展示内容                               | 支持的课程结论                 | 对应代码或字幕内容  |
| -------------------------------- | -------------------------------------- | ------------------------------ | ------------------- |
| `multimodality.png`              | 世界包含文本、图像、音频、视频等多模态 | LLM 需要扩展到 omni model      | 课程开场            |
| `clip.png`                       | CLIP 图文对比学习结构                  | 图像和文本可映射到共享语义空间 | `clip()`            |
| `clip-code.png`                  | CLIP loss 伪代码                       | 双向 softmax 对比学习          | `clip()`            |
| `clip-efficiency.png`            | CLIP ranking 目标效率                  | 对比学习比直接预测文本更高效   | `clip()`            |
| `siglip-code.png`                | SigLIP sigmoid loss 伪代码             | pairwise binary objective      | `siglip()`          |
| `siglip-parallelism.png`         | SigLIP 并行训练                        | SigLIP 更易扩展                | `siglip()`          |
| `llava-architecture.png`         | LLaVA 架构                             | CLIP + projector + Vicuna      | `llava()`           |
| `llava-onevision-anyres.png`     | AnyRes 分块处理                        | 高分辨率细节保留               | `llava_onevision()` |
| `llava-onevision-modalities.png` | 单图、多图、视频 token budget          | 不同模态长度要平衡             | `llava_onevision()` |
| `qwen-vl-stages.png`             | Qwen-VL 三阶段训练                     | 训练流程工程化                 | `qwen_vl()`         |
| `qwen2-vl-mrope.png`             | MRoPE                                  | 多维位置编码                   | `qwen2_vl()`        |
| `qwen3-vl-pretraining.png`       | Qwen3-VL 预训练阶段                    | 长上下文分阶段扩展             | `qwen3_vl()`        |
| `chameleon.png`                  | Chameleon 统一 token 建模              | 离散 token 可统一理解与生成    | `chameleon()`       |
| `vq-vae.png`                     | VQ-VAE 图像离散化                      | 图像可编码成 codebook token    | `chameleon()`       |

### 11.2 关键图表详细解释

#### CLIP 图与代码截图

CLIP 图表要回答的问题是：**如何通过图文对训练一个共享语义空间？**

读图重点：

- 左侧通常是图像 encoder；
- 右侧是文本 encoder；
- 中间是相似度矩阵；
- 对角线表示正确图文对；
- 非对角线表示负样本；
- loss 让对角线更高，非对角线更低。

与代码对应：
$$
\text{logits}
=
I_e T_e^\top \cdot \exp(t)
$$
这正是相似度矩阵。

#### SigLIP 代码截图

SigLIP 图表要回答的问题是：**能否不用 full-batch softmax？**

读图重点：

- 对角线 label 是 `+1`；
- 非对角线 label 是 `-1`；
- 每个 pair 做 sigmoid 判断；
- 不再要求每行每列做 softmax 归一化。

#### AnyRes 图表

AnyRes 图表要回答的问题是：**如何在不把整张高分辨率图压缩成低清小图的情况下保留细节？**

读图重点：

- 全图路径保留整体布局；
- tile 路径保留局部细节；
- 拼接后 token 数可能增加；
- token 太多时需要下采样。

容易误解的地方是：AnyRes 不是简单“输入更大图”，而是通过分块控制分辨率和 token 数。

#### MRoPE 图表

MRoPE 图表要回答的问题是：**视觉 token 的位置不是一维，如何让 attention 理解二维和三维结构？**

读图重点：

- 文本位置是一维；
- 图像位置是 `(h, w)`；
- 视频位置是 `(t, h, w)`；
- 多维位置编码帮助模型理解空间和时间关系。

#### Chameleon / VQ-VAE 图表

Chameleon 图表要回答的问题是：**能否把图像也变成离散 token，并像语言一样自回归建模？**

读图重点：

- VQ-VAE encoder 把图像变成 latent；
- codebook 把 latent 量化成离散索引；
- decoder 能从 codebook 表示重建图像；
- Transformer 对文本 token 和图像 token 统一做 next-token prediction。

------

## 12. 关键公式与推导汇总

| 公式                                                         | 符号解释                           | 适用场景            | 直觉解释                       | 代码 / 例子对应   |
| ------------------------------------------------------------ | ---------------------------------- | ------------------- | ------------------------------ | ----------------- |
| `\mathbf{x}_p = W_{\text{patch}}\text{flatten}(P_p)+\mathbf{b}` | `P_p` 为图像 patch，`W` 为线性映射 | ViT patch embedding | 把小图块变成向量 token         | CLIP ViT          |
| `s_{ij} = \exp(t)\hat{\mathbf{u}}_i^\top\hat{\mathbf{v}}_j`  | 图像与文本相似度                   | CLIP / SigLIP       | 正确图文对相似度应更高         | `clip-code.png`   |
| `\mathcal{L}_{I\to T}`                                       | 图像到文本 softmax loss            | CLIP                | 每张图在所有文本中选正确文本   | CLIP 双向 CE      |
| `\mathcal{L}_{T\to I}`                                       | 文本到图像 softmax loss            | CLIP                | 每段文本在所有图像中选正确图像 | CLIP 双向 CE      |
| `y_{ij}=+1/-1`                                               | pair 是否匹配                      | SigLIP              | 对角线为正例，非对角线为负例   | `siglip-code.png` |
| `-\log\sigma(y_{ij}s_{ij})`                                  | logistic pair loss                 | SigLIP              | 每个 pair 独立判断是否 aligned | SigLIP loss       |
| `\mathbf{e}_I = W\mathbf{z}_I`                               | 视觉向量投影到 LM 空间             | LLaVA               | projector 是视觉到语言的翻译器 | LLaVA Stage 1     |
| `p(y_{1:T}\mid I,x)=\prod_t p(y_t\mid I,x,y_{<t})`           | VLM 自回归生成                     | LLaVA/Qwen          | 给定图像和指令逐 token 回答    | VLM 训练          |
| `(R(m)q)^\top(R(n)k)=q^\top R(n-m)k`                         | RoPE 相对位置性质                  | Qwen2-VL / Qwen3-VL | attention 可感知相对距离       | MRoPE             |
| `\tilde{L}_i=\frac{1}{\sqrt{T_i}}\sum_t\ell_{i,t}`           | 按长度平方根归一化                 | Qwen3-VL            | 防止视频长样本主导 loss        | square-root loss  |
| `k^*=\arg\min_k|\mathbf{z}_e(x)-\mathbf{e}_k|^2`             | VQ-VAE 最近 codebook               | Chameleon           | 把连续视觉 latent 变离散 token | VQ-VAE            |
| `p(s)=\prod_t p(s_t\mid s_{<t})`                             | 统一自回归建模                     | Chameleon           | 文本和图像 token 都预测下一个  | Chameleon         |

------

## 13. 概念对比与易混淆点

| 易混淆概念                   | 区别                                                         | 联系                      | 常见误解                           | 正确理解                               |
| ---------------------------- | ------------------------------------------------------------ | ------------------------- | ---------------------------------- | -------------------------------------- |
| CLIP vs SigLIP               | CLIP 用 softmax 多分类；SigLIP 用 sigmoid 二分类             | 都学习图文对齐            | 以为 SigLIP 完全不需要负样本       | SigLIP 仍用负 pair，只是 loss 更可分解 |
| Vision encoder vs LLM        | 视觉编码器看图；LLM 生成语言和推理                           | VLM 需要二者结合          | 以为视觉编码器越大越好             | 很多能力来自 LLM、数据和 adapter       |
| Projector vs Adapter         | projector 可简单线性映射；adapter 可 cross-attention 或深层融合 | 都连接视觉和语言空间      | 以为只是维度变换                   | 它还承担语义对齐作用                   |
| 连续 token vs 离散 token     | 连续 token 是 embedding；离散 token 是 code ID               | 都是 Transformer 输入表示 | 以为 token 必须是整数              | 多模态中 token 可以是连续向量          |
| 图像理解 vs 图像生成         | 理解偏语义；生成要细节                                       | 都需要视觉表示            | 以为同一个 encoder 适合所有任务    | OCR/生成需要更细粒度信息               |
| AnyRes vs dynamic resolution | AnyRes 常通过分块保细节；dynamic resolution 根据输入大小分配 token | 都解决固定分辨率问题      | 以为只是提高分辨率                 | 核心是控制信息保留和 token budget      |
| RoPE vs MRoPE                | RoPE 一维；MRoPE 多维                                        | 都用于位置编码            | 以为图像可以直接当一维序列无损处理 | 图像/视频需要空间和时间位置            |
| Chameleon vs LLaVA/Qwen      | Chameleon 统一离散 token；LLaVA/Qwen 多为连续视觉编码注入 LLM | 都是多模态路线            | 以为 Chameleon 必然更强            | 统一性强，但离散化和训练稳定性困难     |
| VQ-VAE vs diffusion          | VQ-VAE 离散化重建；diffusion 逐步去噪生成                    | 都可用于图像生成相关任务  | 以为 VQ-VAE 已解决高质量生成       | diffusion 通常更擅长高频细节           |

------

## 14. 学习拓展与发散理解

### 14.1 与大模型训练实践的关系

#### 数据准备

多模态训练的数据不只是更多，而是更复杂：

- 图文对可能 noisy；
- OCR 数据需要从图片中抽取文字；
- 视频需要抽帧、解码、采样；
- 多图任务需要构造关系推理样本；
- 指令数据常常需要合成和清洗。

**工程直觉**：VLM 的效果很大程度取决于数据策划，而不是只取决于模型结构。

#### 模型训练

多模态模型训练要考虑：

- vision encoder 是否冻结；
- projector/adapter 如何初始化；
- LLM 是否全量微调；
- 不同模态 token 数如何平衡；
- 长视频样本是否主导 loss；
- 多阶段训练顺序如何安排。

#### 训练效率

图像和视频带来额外成本：

- 图片解码；
- 视频抽帧；
- 高分辨率图像分块；
- 更长上下文；
- 更大 attention 计算。

课程问答中也指出，多模态训练的数据加载和 preprocessing 可能成为瓶颈，需要异步数据加载与计算重叠。【字幕†source】

#### 推理部署

推理时的挑战包括：

- 用户上传图像分辨率不固定；
- 视频帧数不固定；
- 长上下文显存压力高；
- OCR 或 GUI agent 任务需要保留细节；
- 多模态输入的 latency 高于纯文本。

### 14.2 与其他 lecture 的联系

本讲与 CS336 其他主题有自然联系：

- 与 **tokenization**：文本 tokenization 的思想扩展到图像和视频 tokenization。
- 与 **Transformer 架构**：所有多模态输入最终仍要进入 Transformer。
- 与 **训练目标**：CLIP/SigLIP 使用对齐目标，VLM 使用自回归语言目标，Chameleon 使用统一 next-token objective。
- 与 **系统优化**：视频和高分辨率图像导致更长序列、更高 I/O、更复杂 batch 组织。
- 与 **后训练**：视觉指令微调、SFT、蒸馏、RL 等方法在 Qwen3-VL 中继续出现。

### 14.3 面试 / 考试 / 作业可能考察角度

| 问题                                      | 考察点                   | 回答思路                                                     |
| ----------------------------------------- | ------------------------ | ------------------------------------------------------------ |
| 为什么多模态模型首先要解决 token 化问题？ | Transformer 输入形式     | Transformer 处理 token 序列，图像/视频需变成连续或离散 token |
| CLIP loss 如何构造？                      | 对比学习                 | `n×n` 相似度矩阵，对角线正例，行列双向 softmax               |
| SigLIP 相比 CLIP 改了什么？               | 损失函数差异             | CLIP 是 multiclass softmax，SigLIP 是 pairwise sigmoid       |
| 为什么 CLIP 不擅长 OCR？                  | 分辨率与语义粒度         | resize/crop、分类导向、高层语义编码导致细节丢失              |
| VLM 为什么需要 projector？                | 表示空间对齐             | 视觉 embedding 不在 LLM embedding space，需要映射            |
| LLaVA 两阶段训练分别做什么？              | alignment 与 fine-tuning | 第一阶段只训 projector，第二阶段训 projector + LLM           |
| AnyRes 解决什么问题？                     | 高分辨率图像处理         | 分块编码保留局部细节，同时控制 token 数                      |
| MRoPE 为什么需要多维？                    | 图像/视频位置编码        | 图像有高宽，视频有时间；不能只当一维文本                     |
| Qwen3-VL 为什么要 square-root loss？      | 多模态 loss 平衡         | 防止超长视频样本主导训练                                     |
| Chameleon 的优势和问题是什么？            | 统一建模路线             | 优势是统一理解生成，问题是离散化损失和训练不稳定             |

------

## 15. 课程复习总结

### 15.1 本讲核心思想

Lecture 17 的主线是：**前沿模型越来越需要多模态能力，而多模态建模的根本挑战是如何将非文本模态编码成 Transformer 可以处理的 token。**

课程先从 CLIP 讲起，说明如何用对比学习建立图文共享语义空间；再用 SigLIP 说明如何改进 CLIP 的损失函数，使训练更易扩展；随后进入 VLM 标准模板，即把视觉编码器输出通过 projector/adapter 注入 LLM；接着通过 LLaVA、LLaVA OneVision 和 Qwen-VL 系列展示现代视觉语言模型如何处理高分辨率、多图、视频、动态分辨率、位置编码、长上下文和训练稳定性；最后用 Chameleon 讨论把图像也离散化成 token 的统一 autoregressive 路线。

`lecture_17.py` 最后的 summary 也概括了这些主题：frontier models are expected to be multimodal；fundamental challenge 是 encode non-text modalities；comprehension 和 generation 可能需要不同粒度；训练中要平衡 image/video 和 text；当前路线常是 continuous encoders + Transformer + diffusion models for generation。

### 15.2 必须掌握的概念

- **Omni model**：输入和输出都可包含任意模态组合的模型。
- **连续视觉 token**：视觉编码器输出的 embedding。
- **离散视觉 token**：VQ-VAE 等 tokenizer 输出的 codebook ID。
- **CLIP**：双编码器图文对比学习。
- **SigLIP**：pairwise sigmoid 图文匹配损失。
- **Vision encoder + projector + LLM**：VLM 标准架构。
- **AnyRes**：高分辨率图像分块编码。
- **Dynamic resolution**：根据图像大小和复杂度动态分配视觉 token。
- **MRoPE**：用于图像/视频的多维旋转位置编码。
- **DeepStack**：跨层注入视觉信息。
- **VQ-VAE**：将图像量化为离散 codebook token。
- **Chameleon**：统一离散 token 的自回归多模态模型。

### 15.3 必须理解的公式

#### CLIP 相似度

$$
s_{ij}
=
\exp(t)
\hat{\mathbf{u}}_i^\top
\hat{\mathbf{v}}_j
$$

解决的问题：度量图像 `I_i` 和文本 `T_j` 是否匹配。

#### CLIP 双向损失

$$
\mathcal{L}_{\text{CLIP}}
=
\frac{1}{2}
\left(
\mathcal{L}_{I \rightarrow T}
+
\mathcal{L}_{T \rightarrow I}
\right)
$$

解决的问题：同时训练图像检索文本和文本检索图像。

#### SigLIP loss

$$
\mathcal{L}_{\text{SigLIP}}
=
-\frac{1}{n}
\sum_{i=1}^{n}
\sum_{j=1}^{n}
\log
\sigma(y_{ij}s_{ij})
$$

解决的问题：把图文匹配改成 pairwise binary classification。

#### VLM 自回归目标

$$
p(y_1,\dots,y_T \mid I,x)
=
\prod_{t=1}^{T}
p(y_t \mid I,x,y_{<t})
$$

解决的问题：给定图像和文本指令，逐 token 生成回答。

#### RoPE 相对位置性质

$$
(R(m)\mathbf{q})^\top(R(n)\mathbf{k})
=
\mathbf{q}^\top R(n-m)\mathbf{k}
$$

解决的问题：让 attention 感知相对位置。

#### VQ-VAE 最近邻量化

$$
k^*
=
\arg\min_k
\|\mathbf{z}_e(x)-\mathbf{e}_k\|_2^2
$$

解决的问题：把连续图像 latent 映射到离散 codebook token。

### 15.4 必须看懂的代码 / 图表

- `lecture_17.py` 的 `main()`：理解课程讲解顺序。
- `clip-code.png`：理解 CLIP 的双向 cross entropy。
- `siglip-code.png`：理解 SigLIP 的 pairwise sigmoid loss。
- `llava-architecture.png`：理解 vision encoder + projector + LLM。
- `llava-onevision-anyres.png`：理解高分辨率图像分块。
- `qwen2-vl-mrope.png`：理解 MRoPE 的多维位置编码。
- `qwen3-vl-pretraining.png`：理解长上下文多阶段训练。
- `vq-vae.png`：理解图像离散化为 codebook token。

### 15.5 一句话总结

**Lecture 17 的核心是：多模态模型的关键不只是“加入图片”，而是如何把图像、视频等非文本模态编码成合适的 token，并在语义对齐、分辨率、位置编码、训练稳定性和生成能力之间做工程与理论上的权衡。**