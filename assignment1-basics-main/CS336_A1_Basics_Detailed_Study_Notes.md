# CS336 Assignment 1: Basics 中文详细解析报告与学习笔记

> 说明：这份报告按“课程助教 + Transformer/Tokenizer 工程导师 + 代码结构审查者”的方式写成。它不是代码答案，也不直接给可提交实现，而是把 handout、README、AGENTS、`cs336_basics/` 与测试文件组织成一套学习笔记、实现路线、测试地图和实验设计清单。

---

# 0. 如何使用这份笔记

这份笔记建议分三遍读：

第一遍只看 **第 1、3、5 节**，弄清楚作业要交什么、代码怎么接测试、先做什么后做什么。

第二遍看 **第 2、4 节**，把 tokenizer、Transformer、loss、optimizer、training loop 的概念串起来。

第三遍做实验前看 **第 6、7、8、9 节**，重点关注实验日志、debug 方法、hidden test 边界条件和它与真实 LLM pipeline 的关系。

这份报告有一个核心原则：**不要为了过测试而过测试**。CS336 A1 真正训练的是你把一个小型 LLM pipeline 从文本文件一路推进到生成文本的能力：数据 → tokenizer → token IDs → Transformer → loss → optimizer → checkpoint → evaluation → generation → ablation。

---

# 1. Assignment 基本信息

## 1.1 Assignment 名称与主题

- **Assignment 名称**：CS336 Assignment 1 (basics): Building a Transformer LM
- **handout 版本**：PDF 显示为 Version 26.0.3，Spring 2026。
- **README 标题**：仍写着 Spring 2025 Assignment 1: Basics；结合 PDF 和 CHANGELOG，建议以 PDF v26.0.3 的具体要求为准。
- **主题方向**：从零实现一个标准 decoder-only Transformer language model 的基本训练系统。

这个 assignment 不是“实现几个函数”的小练习，而是一个完整的语言模型项目：

1. 训练 byte-level BPE tokenizer。
2. 用 tokenizer 把文本数据编码成 token ID 序列。
3. 手写 Transformer LM 的核心模块。
4. 手写 cross entropy、AdamW、LR schedule、gradient clipping。
5. 写数据加载、checkpoint、training loop。
6. 在 TinyStories 和 OpenWebText 上训练、生成、评估、做 ablation。
7. 最后尝试 leaderboard。

## 1.2 最终要完成什么

你最终要完成两类交付物：

1. **代码交付**：`code.zip`，由 `make_submission.sh` 生成，包含你在仓库中的实现代码。
2. **文字交付**：`writeup.pdf`，回答 handout 里的所有 written questions，并展示实验结果、learning curves、生成文本、资源估算等。

此外还有 leaderboard PR，但它是独立于普通 Gradescope 提交的 leaderboard 流程。

## 1.3 代码重点目录

仓库当前核心结构如下：

```text
assignment1-basics-main/
├── cs336_assignment1_basics.pdf
├── README.md
├── AGENTS.md
├── CHANGELOG.md
├── pyproject.toml
├── make_submission.sh
├── cs336_basics/
│   ├── __init__.py
│   └── pretokenization_example.py
└── tests/
    ├── adapters.py
    ├── common.py
    ├── conftest.py
    ├── test_train_bpe.py
    ├── test_tokenizer.py
    ├── test_model.py
    ├── test_nn_utils.py
    ├── test_optimizer.py
    ├── test_data.py
    ├── test_serialization.py
    ├── fixtures/
    └── _snapshots/
```

其中：

- `cs336_basics/`：你的主要实现区域。当前几乎没有代码，意味着你需要自己组织模块。
- `tests/adapters.py`：测试入口。每个 adapter 应该“调用你的实现”，而不是把核心逻辑写在 adapter 里。
- `tests/test_*.py`：不可改。它们描述了每个模块的输入输出、shape、dtype、数值容忍度和隐藏边界条件。
- `tests/fixtures/`：测试数据与参考模型权重。
- `tests/_snapshots/`：参考输出，用于数值 snapshot comparison。

## 1.4 哪些文件需要修改

主要修改：

- `cs336_basics/` 下你自己创建或补充的模块，例如：
  - `tokenizer.py`
  - `bpe.py` 或 `train_bpe.py`
  - `nn_utils.py`
  - `layers.py`
  - `attention.py`
  - `model.py`
  - `optim.py`
  - `data.py`
  - `serialization.py`
  - `train.py`
  - `generate.py`

必须少量修改：

- `tests/adapters.py`
  - 只做胶水层：接收测试传入的权重、shape、参数，然后调用你在 `cs336_basics/` 中写好的类或函数。
  - 不应在这里写 substantive logic。否则即使本地测试过了，代码结构也会很差，hidden tests 或人工检查也可能出问题。

通常不需要修改，但可以谨慎修改：

- `pyproject.toml`
  - 只有在你确实需要额外依赖时才考虑改。
  - 但作业强调 from-scratch，最好不要引入第三方 tokenizer/model 实现。

## 1.5 哪些文件不应修改

不要改：

- `tests/test_*.py`
- `tests/fixtures/`
- `tests/_snapshots/`
- `tests/common.py`
- `tests/conftest.py`
- `README.md`
- `AGENTS.md`
- `CHANGELOG.md`
- `make_submission.sh`，除非你明确知道自己需要改变 zip 排除规则。

特别注意：`make_submission.sh` 会排除很多大文件类型，包括 `*.txt`、`*.json`、`*.npy`、`*.npz`、`*.pt`、`*.pth`、`tests/fixtures`、`tests/_snapshots` 等。不要把运行所需的核心代码或配置只放在这些被排除的文件里。

## 1.6 README 中的重要命令

环境由 `uv` 管理。

常用命令：

```sh
uv run pytest
```

下载数据：

```sh
mkdir -p data
cd data
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz
cd ..
```

提交打包：

```sh
./make_submission.sh
```

注意：`make_submission.sh` 里 pytest 后有 `|| true`，也就是说测试失败时脚本仍会继续打包。你不能把“生成了 zip”当作“测试通过”。

## 1.7 评分重点

测试和评分重点集中在：

- tokenizer 是否严格匹配 GPT-2/tiktoken 行为。
- BPE training 是否处理特殊 token、tie-breaking、速度约束。
- 模型模块是否数值对齐 reference snapshots。
- shape 是否正确支持 arbitrary leading dimensions。
- dtype/device 是否正确传递。
- 数值稳定性是否足够，比如 softmax、cross entropy、RMSNorm、AdamW。
- optimizer state 和 checkpoint 是否可恢复。
- training loop 是否真的能训练出 reasonable LM。
- 实验是否记录清楚，曲线、loss、生成文本、ablation 是否完整。

---

# 2. PDF / Handout 结构化解析与翻译

## 2.1 Assignment Overview

Handout 开头明确了作业要实现四大组件：

1. **Byte-pair encoding tokenizer**
2. **Transformer language model**
3. **Cross-entropy loss 和 AdamW optimizer**
4. **Training loop，支持保存和加载模型/优化器状态**

并要求实际运行：

1. 在 TinyStories 上训练 BPE tokenizer。
2. 用 tokenizer 编码数据集。
3. 在 TinyStories 上训练 Transformer LM。
4. 生成文本并计算 perplexity。
5. 在 OpenWebText 上训练并提交 leaderboard。

最重要的限制：你需要从零实现核心模块，不能直接使用：

- `torch.nn.Linear`
- `torch.nn.Embedding`
- `torch.nn.functional.softmax`
- `torch.nn.functional.cross_entropy`
- `torch.optim.AdamW`
- 其他会直接替你完成核心作业目标的高阶实现

允许使用：

- `torch.nn.Parameter`
- `torch.nn` 中的容器类，如 `Module`、`ModuleList`、`Sequential`
- `torch.optim.Optimizer` 基类
- 其他不破坏 from-scratch 精神的 PyTorch 张量操作

AGENTS 文件还强调：AI 工具可以帮助解释概念、review、debug、给 sanity checks，但不应该替你写可提交代码。这一点也解释了为什么这份报告会给学习路线、测试地图和设计思路，而不是直接给实现代码。

## 2.2 BPE tokenizer 总体目标

Tokenizer 部分要求你实现 byte-level BPE tokenizer。它分成两步：

1. **Tokenizer training**：从原始文本训练出 `vocab` 和 `merges`。
2. **Tokenizer encode/decode**：用训练好的 `vocab` 和 `merges` 把文本转成 token IDs，并能反向 decode。

核心思想：

- 任何 Unicode 字符串都可以先转成 UTF-8 bytes。
- 初始词表包含 256 个单字节 token。
- BPE 不断合并高频相邻 byte/token pair。
- 合并后得到的 token 可以代表常见子词、单词、空格+词、标点组合等。
- 最终用较大的 vocabulary 换取更短的 token sequence。

## 2.3 Unicode / UTF-8 / byte-level BPE

### Unicode 是什么

Unicode 是字符到整数 code point 的映射。例如：

- `'s'` 的 code point 是 115。
- `'牛'` 的 code point 是 29275。
- Python 中 `ord()` 把字符转 code point，`chr()` 把 code point 转字符。

### 为什么不直接在 Unicode code point 上训练 tokenizer

因为 Unicode 字符空间太大、太稀疏。handout 提到 Unicode 17.0 约 159,801 个字符。如果把每个 code point 当作初始 token：

- 初始 vocab 很大。
- 绝大多数字符极少出现。
- 训练和统计都浪费。
- OOV 虽然减少，但序列和统计复杂度不优雅。

### 为什么选择 UTF-8 bytes，而不是 UTF-16 或 UTF-32

UTF-8 的优点：

- 对 ASCII 兼容，英文和常见网页文本更省空间。
- 互联网主流文本编码是 UTF-8。
- 单位是 bytes，初始 vocab 固定为 256。
- 任意 Unicode 文本都能表示成 byte sequence。
- byte-level tokenizer 天然没有 OOV。

UTF-16/UTF-32 的问题：

- UTF-32 固定 4 bytes，英文文本会膨胀严重。
- UTF-16 在英文上通常也比 UTF-8 冗余。
- UTF-16 还涉及 surrogate pair，处理起来更复杂。
- 对 byte-level BPE 来说，UTF-8 更贴近实际语料存储形式。

### byte-level BPE 为什么重要

Word-level tokenizer 的问题：

- 生僻词、新词、拼写错误、URL、代码、emoji 都容易 OOV。

Pure byte-level tokenizer 的问题：

- 序列过长。
- 模型要学习很多跨 byte 的长程依赖。
- 训练慢，attention 开销随序列长度平方增长。

Byte-level BPE 是折中：

- 初始 token 是 bytes，所以没有 OOV。
- 高频 byte sequence 会被合并成较长 token，所以压缩率更好。
- 常见词、常见前缀、空格+词、标点模式都会被词表捕捉。

## 2.4 BPE tokenizer training

BPE training 的主流程：

1. 初始化 vocabulary：256 个 byte token，加 special tokens。
2. 用 GPT-2 regex 进行 pre-tokenization。
3. 将每个 pre-token 转成 UTF-8 bytes。
4. 统计所有相邻 token pair 的频率。
5. 找出最高频 pair；若频率相同，按 pair 的 bytes 字典序选择 lexicographically greatest pair。
6. 合并该 pair，加入 vocab，记录到 merges。
7. 重复直到达到 `vocab_size`。

### Pre-tokenization 的意义

如果直接在整个 byte 流上合并：

- 可能跨单词、跨标点、跨文档合并。
- 会产生大量只差标点的 token，例如 `dog!` 和 `dog.`。
- 每轮 merge 都扫全语料，成本很高。

Pre-tokenization 让你先得到粗粒度 token，例如 GPT-2 regex 会把 `some text that i'll pre-tokenize` 切成类似：

```text
['some', ' text', ' that', ' i', "'ll", ' pre', '-', 'tokenize']
```

注意很多 token 前面带空格，比如 `' text'`。这很重要，因为 GPT 系 tokenizer 常把空格作为 token 的一部分，这使得 token 能区分句首词和非句首词。

### GPT-2 regex

Handout 给出正则：

```text
'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

应使用 `regex` 包而不是 Python 内置 `re`，因为 `\p{L}` 等 Unicode property 需要更强的 regex 支持。

测试中会拿 GPT-2 vocab/merges 和 tiktoken 对齐，所以 regex 行为必须非常接近 GPT-2。

### special tokens 为什么是硬边界

特殊 token 例如 `<|endoftext|>` 表示文档边界或生成终止符。它们应该：

- 在 vocab 中有单独 ID。
- 编码时永远保持一个 token。
- 训练 BPE merge 时不能被拆开。
- 也不能允许 merge 跨过它们。
- 它们本身不应该贡献普通 merge statistics。

原因很简单：如果允许普通 BPE 把 `<|endoftext|>` 与周围文本合并，模型就无法可靠判断文档边界和停止生成。

测试 `test_train_bpe_special_tokens` 会检查：除 special token 本身外，vocab 中不应出现包含 `<|` 的普通 token。这就是在测 special token 是否被正确移除、分割和隔离。

### BPE merge tie-breaking

Handout 明确：如果多个 pair 频率相同，选择 lexicographically greater pair。

非常容易错的点：

- tie-breaking 应在原始 `bytes` pair 上比较。
- 不要在 GPT-2 remapped unicode 字符串上比较。
- pair 本身是 tuple，例如 `(b's', b't')` 与 `(b'e', b's')` 比较。
- `CHANGELOG` 曾专门修复过 tie-breaking 相关错误，说明这是 hidden tests 的高风险点。

### BPE trainer 性能

测试 `test_train_bpe_speed` 要求在 `corpus.en` 上 vocab size 500、special token `<|endoftext|>` 的训练时间小于 1.5 秒。

对 full TinyStories，handout 提示可以通过 multiprocessing 加速 pre-tokenization，目标在合理机器上做到 2 分钟级别。关键做法是：

- 以 `<|endoftext|>` 为 chunk 边界。
- 每个进程独立 pre-tokenize 和计数。
- 合并局部 counts。
- merge 阶段由于强依赖前一轮结果，Python 中不容易并行。

`cs336_basics/pretokenization_example.py` 提供了 `find_chunk_boundaries` 思路。注意该文件底部有 usage snippet，不适合原样 import 整个模块；更安全做法是理解或提取其中的 chunk boundary 函数思想。

## 2.5 tokenizer encoding / decoding

### Encoding

Encoding 与 training 类似：

1. 先处理 special tokens，保证它们是硬边界和单 token。
2. 对普通文本片段运行 GPT-2 pre-tokenizer。
3. 每个 pre-token 转成 UTF-8 bytes 序列。
4. 按训练时 `merges` 的顺序应用 BPE merges。
5. 把最终 bytes token 映射到 vocab ID。

注意：编码时 merges 的顺序必须是训练时产生的顺序。不能每次动态找最高频 pair，因为 encoding 阶段没有语料频率概念。

### Decoding

Decoding 的流程更简单：

1. 用 vocab 把每个 ID 映射回 bytes。
2. 拼接所有 bytes。
3. 用 UTF-8 decode 成字符串。

关键边界：token ID 序列未必对应合法 UTF-8 bytes。正确行为是 decode 时使用 replacement character `U+FFFD`，也就是 Python 中 `errors='replace'` 的语义。

重要：应该先拼接 bytes，再 decode。不要逐 byte decode，也不要逐 token decode 后拼字符串。多字节字符如 emoji 的 bytes 必须作为整体解码，否则会失败或产生替换符。

## 2.6 `encode_iterable` 为什么需要 lazy streaming

`encode(text)` 可以把整个字符串读进内存，但真实语料可能是几十 GB 到 TB 级。`encode_iterable(iterable)` 要支持对文件句柄或字符串 iterable 进行懒编码：

- 输入可以是文件对象，逐行或逐 chunk 提供字符串。
- 输出应是 generator，逐个 yield token ID。
- 内存复杂度应接近常数，而不是随文件大小线性增长。

测试中有专门的 memory test：`encode_iterable` 在 Linux 上被限制约 1MB 额外内存；而普通 `encode` 的内存测试是 xfail，表示 `encode` 不要求省内存，但 `encode_iterable` 要求省内存。

隐藏风险：懒处理不能改变 tokenization 结果。如果 chunk 边界切在一个可能被 merge 的 pre-token 内部，结果会和整体编码不同。安全策略是保证处理边界落在不会跨越 BPE token 的位置，例如明确的文档边界、特殊 token 边界或经过 carefully buffered 的文本边界。

## 2.7 Transformer LM modules

Handout 的模型是现代 decoder-only Transformer 的简化版：

```text
input token IDs
  -> token embedding
  -> N 个 pre-norm Transformer block
  -> final RMSNorm
  -> LM head linear projection
  -> logits over vocab
```

每个 Transformer block：

```text
x
  -> RMSNorm
  -> causal multi-head self-attention with RoPE
  -> residual add
  -> RMSNorm
  -> SwiGLU feed-forward
  -> residual add
```

## 2.8 Linear

Linear 模块执行无 bias 的线性变换。测试期望权重 shape 是：

```text
weights: (d_out, d_in)
input:   (..., d_in)
output:  (..., d_out)
```

数学上常写 `y = W x`，但 PyTorch 的 row-major/batched 输入通常会用最后一维作为 feature 维。因此 shape 理解比公式本身更重要。

要求：

- subclass `torch.nn.Module`
- 权重用 `torch.nn.Parameter`
- 不用 `nn.Linear` 或 `nn.functional.linear`
- 初始化用 truncated normal
- 不包含 bias

## 2.9 Embedding

Embedding 将 token IDs 转成向量：

```text
input token_ids: (...)
weight:          (vocab_size, d_model)
output:          (..., d_model)
```

测试会直接给 embedding weight，并期待你通过 adapter 加载到实现里。不要使用 `nn.Embedding` 或 `nn.functional.embedding`。

## 2.10 RMSNorm

RMSNorm 对最后一维做归一化：

```text
RMS(x) = sqrt(mean(x_i^2) + eps)
y_i = x_i / RMS(x) * g_i
```

它和 LayerNorm 的区别：

- LayerNorm：减均值，再除以标准差，通常有 gain 和 bias。
- RMSNorm：不减均值，只按 root mean square 缩放，通常只有 gain。

RMSNorm 的优点：

- 更简单，计算略少。
- 在现代 LLM 中常见，比如 LLaMA 系。
- 保留激活均值信息，只控制尺度。

数值重点：handout 明确要求 normalization 前把输入 upcast 到 `float32`，完成后再 cast 回原 dtype。这是为了避免 fp16/bf16 下平方溢出或精度差。

## 2.11 SwiGLU

SwiGLU feed-forward：

```text
FFN(x) = W2( SiLU(W1 x) ⊙ W3 x )
```

其中：

- `W1` 和 `W3`：从 `d_model` 投到 `d_ff`
- `W2`：从 `d_ff` 投回 `d_model`
- `SiLU(x) = x * sigmoid(x)`
- `⊙` 是逐元素乘法

直觉：

- `W3 x` 提供一条近似线性信息路径。
- `SiLU(W1 x)` 像一个连续 gate。
- 二者相乘让 FFN 具有 gated feature selection 能力。

Handout 推荐 `d_ff ≈ 8/3 * d_model`，并 round 到 64 的倍数以更好利用 GPU tensor cores。Ablation 中用普通 SiLU FFN 时，为了参数量相近，使用 `d_ff = 4 * d_model`。

## 2.12 RoPE

RoPE 即 Rotary Position Embedding。它不是给 embedding 加一个位置向量，而是对 query 和 key 的偶数/奇数维 pair 做二维旋转。

核心作用：

- 给 attention 注入位置信息。
- 让 query/key 的点积自然包含相对位置信息。
- 不增加 learnable parameters。
- 可以预计算 sin/cos buffer。

shape 重点：

```text
input x:          (..., seq_len, d_k)
token_positions: (..., seq_len)
output:           (..., seq_len, d_k)
```

在 multi-head attention 中，RoPE 应作用在每个 head 的 `d_head = d_model / num_heads` 维度上，而不是整个 `d_model`。也就是说，常见中间 shape 是：

```text
Q/K: (batch, heads, seq_len, d_head)
```

然后 head 维当作 batch-like dimension，让同一位置旋转规则应用到每个 head。

容易错：

- 只对 Q 和 K 用 RoPE，不对 V 用。
- `d_k` 应为偶数。
- token positions 不能总是假定为 `arange(seq_len)`；adapter 会传显式 `token_positions`。
- truncated input test 会检查较短 sequence 的位置处理。

## 2.13 Scaled dot-product attention

公式：

```text
Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
```

shape：

```text
Q: (..., queries, d_k)
K: (..., keys,    d_k)
V: (..., keys,    d_v)
mask: (..., queries, keys) 或可 broadcast
output: (..., queries, d_v)
```

为什么要除以 `sqrt(d_k)`：

- Q 和 K 维度越大，点积方差越大。
- 不缩放会让 softmax 进入饱和区，梯度变小。
- scaling 让 logits 尺度更稳定。

Mask 语义：

- `True` 表示可以 attend。
- `False` 表示不能 attend。
- mask 应在 softmax 前作用，把不可见位置设成非常小的值或负无穷。
- 被 mask 的位置 softmax probability 应为 0。

## 2.14 Multi-head self-attention

Multi-head attention 做三件事：

1. 用 Q/K/V projection 从输入 `x` 得到 Q、K、V。
2. 将 `d_model` 拆成 `num_heads * d_head`，每个 head 独立 attention。
3. 拼接 head outputs，再用 output projection 投回 `d_model`。

推荐 shape ledger：

```text
x:        (B, T, D)
Q/K/V:    (B, T, D)
reshape:  (B, H, T, d_head)
scores:   (B, H, T, T)
mask:     (T, T) -> broadcast to (1, 1, T, T)
context:  (B, H, T, d_head)
concat:   (B, T, D)
output:   (B, T, D)
```

Causal mask：

- 第 i 个 token 只能 attend 到 `j <= i` 的位置。
- 防止未来 token 泄漏目标答案。
- 对 LM training 至关重要，因为一次 forward 会并行预测所有位置的 next token。

## 2.15 Transformer block

Pre-norm block 公式：

```text
z = x + MultiHeadSelfAttention(RMSNorm(x))
y = z + FFN(RMSNorm(z))
```

Pre-norm 的直觉：

- 残差流从输入到输出更“干净”。
- 梯度更容易通过 residual path 回传。
- 现代 LLM 普遍使用 pre-norm，而原始 Transformer 是 post-norm。

Ablation 会让你比较：

```text
post-norm:
z = RMSNorm(x + MHA(x))
y = RMSNorm(z + FFN(z))
```

## 2.16 Transformer LM

完整 LM：

```text
token_ids: (B, T)
embeddings: (B, T, D)
for each layer: TransformerBlock -> (B, T, D)
final RMSNorm -> (B, T, D)
lm_head -> logits: (B, T, V)
```

Training 时：

- input 是 token sequence 前 T 个位置。
- target 是右移一位的 next tokens。
- logits 每个位置预测下一个 token。

Generation 时：

- 输入 prompt token IDs。
- 取最后一个位置的 logits。
- softmax/sample 得到下一个 token。
- append 回上下文，重复。

## 2.17 Cross entropy

Cross entropy 对每个样本：

```text
loss = -log softmax(logits)[target]
```

数值稳定重点：

- 先减去 logits 最大值。
- 尽量使用 `logsumexp` 思维：`log(sum(exp(logits - max))) + max`。
- 不要先 softmax 再 log，否则容易 underflow/overflow。
- 对所有 batch-like 维度取平均。

Perplexity：

```text
perplexity = exp(mean cross entropy)
```

perplexity 可解释为模型平均每一步“困惑于多少个等可能 token”。loss 越低，perplexity 越低。

## 2.18 AdamW

AdamW 保存每个参数的：

- first moment estimate `m`
- second moment estimate `v`
- step count `t`

核心区别：AdamW 的 weight decay 是 decoupled weight decay，不等同于把 L2 regularization 加进 loss。

普通 L2 regularization：

- 把 `λ ||θ||²` 加进 loss。
- 它影响 gradient。
- 在 Adam 这种自适应优化器中，L2 项会被 moment normalization 处理，效果不等同于简单参数衰减。

AdamW：

- 先或独立对参数做 `θ ← θ - α λ θ` 一类衰减。
- 再用 Adam 的 moment-adjusted gradient 更新。
- 权重衰减与 gradient update 解耦，正则化行为更稳定。

Handout 的 v26.0.0 changelog 提到修正了 AdamW 中 weight decay 与 gradient update 的顺序，因此建议严格按当前 handout 的算法理解。

## 2.19 Learning rate schedule

本作业使用 warmup + cosine decay：

1. `t < warmup_iters`：线性 warmup，从 0 到 max LR。
2. `warmup_iters <= t <= cosine_cycle_iters`：cosine annealing，从 max LR 到 min LR。
3. `t > cosine_cycle_iters`：保持 min LR。

直觉：

- 训练初期参数随机，直接用大 LR 容易不稳定，所以 warmup。
- 中后期逐渐降低 LR，帮助收敛。
- 实验中 learning rate 是最重要的超参数之一。

## 2.20 Gradient clipping

Global gradient clipping 指：把所有参数的 gradient 当作一个大向量，计算整体 L2 norm。

```text
if ||g||_2 > M:
    g <- g * M / (||g||_2 + eps)
```

注意不是每个参数单独 clip，也不是每个 tensor 单独 clip。global norm across all parameter gradients 才是测试和 handout 要求。

它的作用：

- 避免偶发 batch 导致梯度爆炸。
- 让 optimizer step 的最大尺度受控。
- 常用于 Transformer 训练。

## 2.21 Data loading

训练数据会被编码成一个长的一维 token ID 序列：

```text
x = [x_1, x_2, ..., x_n]
```

每个 batch 采样：

```text
input:  [x_i,     ..., x_{i+m-1}]
target: [x_{i+1}, ..., x_{i+m}]
```

shape：

```text
input:  (batch_size, context_length)
target: (batch_size, context_length)
```

要点：

- 起点最大只能到 `len(dataset) - context_length - 1`。
- 输出应是 `torch.LongTensor`。
- 输出应放在用户指定 device 上。
- 大数据应使用 `np.memmap` 或 `np.load(..., mmap_mode='r')`。

## 2.22 Checkpointing

测试最低要求保存：

- `model.state_dict()`
- `optimizer.state_dict()`
- `iteration`

真正训练时建议额外保存：

- config/hyperparameters
- tokenizer vocab/merges 的路径或 hash
- random states：Python、NumPy、PyTorch、CUDA/MPS
- scheduler state 或当前 LR schedule 参数
- gradient scaler state，如果使用 mixed precision
- best validation loss、日志 step、wall-clock time

为什么 optimizer state 必须保存：AdamW 有 m/v moments。如果只保存 model weights，resume 后 optimizer 状态丢失，训练轨迹会突变。

## 2.23 Training loop

标准 loop：

```text
sample batch
forward model
compute loss
zero grad
backward
clip gradients
set learning rate
optimizer step
periodic logging
periodic validation
periodic checkpoint
```

工程重点：

- 命令行或配置文件管理超参数。
- train/valid 数据 memory-map。
- validation 不要每次跑太久，否则会严重拖慢训练。
- checkpoint 不要太频繁，否则 I/O 成瓶颈。
- 记录 step、tokens processed、wall-clock、train loss、valid loss、LR、grad norm。

## 2.24 Generation / decoding

解码每步：

1. 将当前 token 序列输入模型。
2. 取最后一个位置 logits。
3. temperature scaling。
4. top-p/nucleus sampling 可选。
5. sample 下一个 token。
6. 如果生成 `<|endoftext|>` 或达到 max tokens，停止。

Temperature：

- 小于 1：更保守，更像 greedy。
- 大于 1：更多样，但更容易胡言乱语。
- 接近 0：近似 argmax。

Top-p：

- 只保留累计概率达到 p 的最小高概率 token 集。
- 可以过滤掉长尾低质量 token。
- 常与 temperature 一起调。

## 2.25 TinyStories / OpenWebText experiments

TinyStories：

- 简单、儿童故事风格、数据分布窄。
- 小模型容易学到语法和故事模板。
- 适合快速 debug 和 ablation。
- handout 基础配置约 17M non-embedding parameters。

OpenWebText：

- 网页文本，主题复杂、风格多样、噪声更高。
- 同样模型和 compute 下 loss 更难降，生成质量更差。
- 更接近真实 LM pretraining 数据。

推荐基础超参数：

| 超参数 | TinyStories 推荐起点 |
|---|---:|
| vocab size | 10,000 |
| context length | 256 |
| d_model | 512 |
| d_ff | 1344 |
| RoPE theta | 10000 |
| layers | 4 |
| heads | 16 |
| total tokens processed | 327,680,000 |

Low-resource 配置可以把 total tokens processed 降到约 40M，目标 validation loss 可放宽。

## 2.26 Ablations

Ablation 的目的不是“随便改模型”，而是验证架构组件的作用：

- 移除 RMSNorm：观察训练稳定性和 learning rate sensitivity。
- Pre-norm 改 post-norm：观察收敛和稳定性。
- 移除 RoPE（NoPE）：观察 positional information 的贡献。
- SwiGLU 改普通 SiLU FFN：在参数量相近时比较 gating 的价值。

每个 ablation 都应有 baseline 曲线，否则无法解释差异。

## 2.27 Leaderboard

Leaderboard 规则：

- 单次 submission 最多 45 分钟 B200 wall-clock。
- 只能使用提供的 OpenWebText training data。
- 目标是在限制内最小化 validation loss。
- baseline 要至少打过 naive 5.0 loss。

可尝试方向：

- learning rate/batch size schedule
- weight tying
- init 改动
- optimizer betas/weight decay
- context length 调整
- model width/depth tradeoff
- tokenizer/vocab size 影响

但要谨慎：小规模 leaderboard 上有效的 trick 不一定能泛化到大规模预训练。

---

# 3. PDF 中所有 Problem / Exercise / TODO 总表

说明：PDF 中主要是 `Problem`，未发现单独命名的 `Exercise` 或 `TODO` 交付项；`Example` 是讲解样例，不是需要提交的题目。下表覆盖 handout 中所有 Problem。

| # | Problem 名称 | 分值/资源 | 要求完成内容 | 交付物 | 对应测试文件 | 容易遗漏的要求 |
|---:|---|---:|---|---|---|---|
| 1 | `unicode1` Understanding Unicode | 1 pt | 理解 `chr(0)`、repr 和 print 表现 | 简短文字回答 | 无 | NUL 字符不可见但存在于字符串中 |
| 2 | `unicode2` Unicode Encodings | 3 pts | 比较 UTF-8/16/32；说明逐 byte decode 错误；给 invalid bytes | 简短文字回答 | 无 | 多字节字符不能逐 byte decode；invalid UTF-8 要理解 |
| 3 | `train_bpe` BPE Tokenizer Training | 15 pts | 实现 byte-level BPE trainer | `vocab` 和 `merges` 函数 | `test_train_bpe.py` | special tokens 是硬边界；tie-breaking；速度 <1.5s 小测试 |
| 4 | `train_bpe_tinystories` | 2 pts | TinyStories 训练 10K vocab tokenizer | 时间、内存、最长 token、profile 说明 | 无直接单测 | 需要 `<|endoftext|>`；序列化 vocab/merges；profile bottleneck |
| 5 | `train_bpe_expts_owt` | 2 pts | OWT 训练 32K vocab tokenizer | 最长 token；与 TinyStories tokenizer 对比 | 无直接单测 | OWT 更大更杂；资源上限 12h/100GB |
| 6 | `tokenizer` Implementing tokenizer | 15 pts | 实现 Tokenizer encode/decode/from_files/encode_iterable | Tokenizer 类 | `test_tokenizer.py` | special token overlap；lazy streaming；invalid UTF-8 replace |
| 7 | `tokenizer_experiments` | 4 pts | 比较 tokenizer 压缩率、吞吐，编码数据集 | 简短实验回答与 token ID 文件 | 无直接单测 | bytes/token；uint16 原因；Pile 时间估算 |
| 8 | `linear` | 1 pt | 实现无 bias Linear | Linear module | `test_model.py::test_linear` | 权重 shape `(d_out,d_in)`；禁用 `nn.Linear` |
| 9 | `embedding` | 1 pt | 实现 Embedding lookup | Embedding module | `test_model.py::test_embedding` | 参数 shape `(vocab_size,d_model)`；禁用 `nn.Embedding` |
| 10 | `rmsnorm` | 1 pt | 实现 RMSNorm | RMSNorm module | `test_model.py::test_rmsnorm` | upcast fp32；只归一化最后一维；返回原 dtype |
| 11 | `positionwise_feedforward` | 2 pts | 实现 SwiGLU FFN | FFN module | `test_model.py::test_swiglu` | W1/W3/W2 shape；SiLU；d_ff multiple of 64 |
| 12 | `rope` | 2 pts | 实现 RotaryPositionalEmbedding | RoPE module | `test_model.py::test_rope` | token_positions shape；arbitrary batch dims；只对偶/奇维 pair 旋转 |
| 13 | `softmax` | 1 pt | 实现稳定 softmax | softmax function | `test_nn_utils.py` | 减 max 防 overflow；支持任意 dim |
| 14 | `scaled_dot_product_attention` | 5 pts | 实现 SDPA + mask | attention function | `test_model.py` | mask True=可见；支持 3D/4D；缩放 `sqrt(d_k)` |
| 15 | `multihead_self_attention` | 5 pts | 实现 causal MHA | MHA module | `test_model.py` | causal mask；QKV 三次 matmul；RoPE 版本 adapter 也要支持 |
| 16 | `transformer_block` | 3 pts | 实现 pre-norm Transformer block | block module | `test_model.py::test_transformer_block` | residual 顺序；两个 RMSNorm；RoPE in attention |
| 17 | `transformer_lm` | 3 pts | 实现完整 Transformer LM | LM module | `test_model.py::test_transformer_lm` | final RMSNorm；LM head；truncated input |
| 18 | `transformer_accounting` | 5 pts | GPT-2 系列参数/FLOPs 资源估算 | 文字+表格 | 无 | embedding/lm_head、attention、FFN FLOPs 分清 |
| 19 | `cross_entropy` | 1 pt | 实现稳定 CE | loss function | `test_nn_utils.py` | logsumexp 思路；平均 batch-like dims |
| 20 | `learning_rate_tuning` | 1 pt | toy SGD 中测试大 LR | 简短观察 | 无 | 需要比较 decay/slow/diverge |
| 21 | `adamw` | 2 pts | 实现 AdamW Optimizer subclass | AdamW class | `test_optimizer.py` | t 从 1；m/v state；decoupled weight decay；grad None 跳过 |
| 22 | `adamw_accounting` | 2 pts | AdamW memory/FLOPs accounting | 公式与数值估计 | 无 | 参数、激活、梯度、optimizer state 分开算 |
| 23 | `learning_rate_schedule` | 1 pt | warmup+cosine LR function | scheduler function | `test_optimizer.py` | 边界 `t=warmup`、`t>cosine_cycle` |
| 24 | `gradient_clipping` | 1 pt | global gradient clipping | clipping function | `test_nn_utils.py` | 全局 norm；in-place 修改；跳过 grad None；eps=1e-6 |
| 25 | `data_loading` | 2 pts | 从 token array 采样 batch | get_batch function | `test_data.py` | y=x shifted by 1；start index 范围；device 生效 |
| 26 | `checkpointing` | 1 pt | save/load model+optimizer+iteration | serialization functions | `test_serialization.py` | optimizer state 必须一致；支持 path/file-like |
| 27 | `training_together` | 4 pts | 写完整 training script | 可配置训练脚本 | 无直接单测 | memmap；checkpoint；logging；validation |
| 28 | `decoding` | 3 pts | 实现 LM decoding | generation function | 无直接单测 | prompt；max tokens；temperature；top-p；stop token |
| 29 | `experiment_log` | 3 pts | 建立实验跟踪 | logging code + experiment log | 无 | step 与 wall-clock 双坐标；保留失败实验 |
| 30 | `learning_rate` | 3 pts + 2 B200 hrs | TinyStories LR sweep | 多条 learning curves；val loss <=1.45 | 无 | 包含 divergent run；解释 edge of stability |
| 31 | `batch_size_experiment` | 1 pt + 1 B200 hr | batch size sweep | curves + commentary | 无 | batch size 改变时 LR 可能要重调 |
| 32 | `generate` | 1 pt | 用 TinyStories checkpoint 生成 | 至少 256 tokens 文本 + fluency 评论 | 无 | 温度/top-p 会显著影响输出 |
| 33 | `layer_norm_ablation` | 1 pt + 0.5 B200 hr | 移除 RMSNorm 训练 | curves + commentary | 无 | 原最佳 LR 下可能不稳定；尝试更低 LR |
| 34 | `pre_norm_ablation` | 1 pt + 0.5 B200 hr | 改 post-norm 训练 | pre/post-norm curve | 无 | 比较必须有 baseline |
| 35 | `no_pos_emb` | 1 pt + 0.5 B200 hr | 移除 RoPE，NoPE | RoPE vs NoPE curve | 无 | 不要无意保留位置编码 |
| 36 | `swiglu_ablation` | 1 pt + 0.5 B200 hr | SwiGLU vs SiLU FFN | matched-param curves + commentary | 无 | SiLU baseline 用 `d_ff=4*d_model` 近似匹配参数量 |
| 37 | `main_experiment` | 2 pts + 2 B200 hrs | OWT 训练 | learning curve + generated text | 无 | 与 TinyStories loss 不可简单横向比；数据更难 |
| 38 | `leaderboard` | 6 pts + 10 B200 hrs budget | 45min B200 内优化 OWT validation loss | final loss、wall-clock curve、方法说明、leaderboard PR | 无 | 只用提供 OWT；单次 runtime <45min；baseline loss 5.0 |

---

# 4. 代码结构解析

## 4.1 `cs336_basics/` 应该放哪些模块

当前 `cs336_basics/` 只有：

- `__init__.py`
- `pretokenization_example.py`

这意味着课程有意不给你 scaffold。推荐你自己拆分模块，但要保持清晰。一个合理结构如下：

| 建议文件 | 内容 | 在系统中的作用 |
|---|---|---|
| `bpe.py` 或 `train_bpe.py` | BPE training、pre-token counts、merge logic | 从文本训练 vocab/merges |
| `tokenizer.py` | Tokenizer class、from_files、encode/decode、encode_iterable | 文本与 token IDs 互转 |
| `layers.py` | Linear、Embedding、RMSNorm、SwiGLU、RoPE | Transformer 基础层 |
| `attention.py` | softmax、SDPA、MHA | 序列信息聚合 |
| `model.py` | TransformerBlock、TransformerLM | 完整语言模型 |
| `nn_utils.py` | cross entropy、gradient clipping | 训练中非模块工具 |
| `optim.py` | AdamW、LR schedule | 参数更新 |
| `data.py` | get_batch、memmap helpers | 训练数据采样 |
| `serialization.py` | save/load checkpoint | 训练恢复 |
| `train.py` | CLI training loop | 实验运行 |
| `generate.py` | decoding、temperature、top-p | 文本生成 |
| `experiment_utils.py` | logging、config、eval | 实验管理 |

不一定要用这些文件名，但职责应分离。最糟糕的结构是把所有东西塞进 `adapters.py` 或单个大文件里。

## 4.2 `tests/adapters.py` 中每个 adapter 期望什么

`adapters.py` 是测试和你代码之间的桥。测试只调用 adapters，不直接知道你的模块命名。

| Adapter | 期望功能 | 对应实现建议 | 关键输入输出 |
|---|---|---|---|
| `run_linear` | 用给定权重跑 Linear | 构造 Linear 并加载权重 | `(..., d_in) -> (..., d_out)` |
| `run_embedding` | 用给定 embedding weight 查表 | 构造 Embedding 并加载 weight | token IDs `(...) -> (..., d_model)` |
| `run_swiglu` | 用给定 W1/W2/W3 跑 SwiGLU | 构造 FFN 并加载权重 | `(..., d_model) -> (..., d_model)` |
| `run_scaled_dot_product_attention` | SDPA | 调用 attention function | 支持 3D/4D 和 mask |
| `run_multihead_self_attention` | 不带 RoPE 的 causal MHA | 构造 MHA，加载 Q/K/V/O | `(B,T,D)->(B,T,D)` |
| `run_multihead_self_attention_with_rope` | 带 RoPE 的 causal MHA | 构造 MHA + RoPE | token_positions 可选 |
| `run_rope` | 单独跑 RoPE | 构造 RotaryPositionalEmbedding | `(...,T,d_k)->same` |
| `run_transformer_block` | 单层 pre-norm block | 构造 block 并加载 state dict | `(B,T,D)->same` |
| `run_transformer_lm` | 完整 LM forward | 构造 LM 并加载 state dict | `(B,T)->(B,T,V)` |
| `run_rmsnorm` | RMSNorm | 构造 RMSNorm 并加载 gain | arbitrary leading dims |
| `run_silu` | SiLU activation | 调用你自己的 SiLU function | shape 不变 |
| `run_get_batch` | 数据采样 | 调用 data.get_batch | 两个 LongTensor |
| `run_softmax` | 稳定 softmax | 调用 nn_utils.softmax | 任意 dim |
| `run_cross_entropy` | CE loss | 调用 nn_utils.cross_entropy | scalar loss |
| `run_gradient_clipping` | global grad clip | 调用 nn_utils.clip | in-place 修改 grad |
| `get_adamw_cls` | 返回 AdamW 类 | 返回你的 Optimizer subclass | 不是实例，是 class |
| `run_get_lr_cosine_schedule` | LR schedule | 调用 scheduler function | 返回 float |
| `run_save_checkpoint` | 保存 checkpoint | 调用 serialization.save | 支持 path/file-like |
| `run_load_checkpoint` | 加载 checkpoint | 调用 serialization.load | 返回 iteration |
| `get_tokenizer` | 返回 tokenizer 实例 | 构造 Tokenizer | vocab 是 `dict[int, bytes]` |
| `run_train_bpe` | 训练 BPE | 调用 train_bpe function | 返回 vocab, merges |

## 4.3 每个测试文件测什么

### `tests/test_train_bpe.py`

测三件事：

1. `test_train_bpe_speed`
   - 在小语料上训练 vocab size 500。
   - 要求用时 < 1.5s。
   - 这会淘汰极慢 naive 实现。

2. `test_train_bpe`
   - 与 reference vocab/merges 对齐。
   - merges 必须顺序一致。
   - vocab keys/values 的集合必须一致。

3. `test_train_bpe_special_tokens`
   - special token 加入 vocab。
   - 普通 vocab 不应包含 `<|` 片段。
   - snapshot 检查 vocab/merges。

隐含要求：

- BPE merge tie-breaking 要完全一致。
- GPT-2 regex pre-tokenization 要一致。
- special token 训练时不贡献 merge counts。
- special token 是 hard segmentation boundary。

### `tests/test_tokenizer.py`

使用 GPT-2 vocab/merges 和 tiktoken 对比，覆盖：

- 空字符串 roundtrip。
- 单字符 ASCII。
- 单个 emoji。
- ASCII 句子。
- Unicode 字符串。
- special token 保留。
- overlapping special tokens。
- 地址文本、德语文本、TinyStories 样本。
- special token 后 trailing newlines。
- special token 与 double newline/non-whitespace 组合。
- `encode_iterable` 与完整 encode 结果一致。
- `encode_iterable` memory usage。

隐含要求：

- special tokens overlap 时要优先最长匹配。
- decode 要能处理 UTF-8 多字节字符。
- iterable encoding 不能一次性读完整文件。
- 行/块边界不能改变 tokenization。

### `tests/test_model.py`

用 `tests/fixtures/ts_tests/model.pt` 的 reference state dict 和 `_snapshots/*.npz` 做数值对齐，覆盖：

- Linear
- Embedding
- SwiGLU
- scaled dot-product attention，3D 和 4D
- MHA，无 RoPE
- MHA with RoPE
- Transformer block
- Transformer LM
- Transformer LM truncated input
- RMSNorm
- RoPE
- SiLU vs PyTorch reference

隐含要求：

- 不能只对某个固定 shape 写死。
- `...` leading dims 要支持。
- RoPE position 不能写死。
- MHA 权重命名和加载要对齐。
- final RMSNorm 和 LM head 顺序要对齐。

### `tests/test_nn_utils.py`

测：

- softmax 是否匹配 PyTorch，并能处理 `x+100` overflow 情况。
- cross entropy 是否匹配 PyTorch，并能处理 `1000*inputs` 大 logits。
- gradient clipping 是否匹配 `torch.nn.utils.clip_grad_norm_`。

隐含要求：

- stable softmax。
- stable cross entropy。
- gradient clipping 是全局 norm，跳过 `grad is None`。

### `tests/test_optimizer.py`

测：

- AdamW 训练一个小 linear model 1000 步，最终权重要匹配 PyTorch AdamW 或 reference snapshot。
- cosine LR schedule 对 25 个 step 的精确数值。

隐含要求：

- AdamW state 初始化、step 计数、bias correction、weight decay、eps 都要对。
- schedule 边界条件不能 off-by-one。

### `tests/test_data.py`

测：

- `x.shape == y.shape == (batch_size, context_length)`。
- `y == x + 1` 在 `np.arange` 数据上成立。
- start index 覆盖范围正确：最小 0，最大 `len(dataset)-context_length-1`。
- 采样分布大致均匀。
- device 参数确实传给 torch tensor。

隐含要求：

- 随机起点的上界要正确。
- 输出 dtype 应是 integer/long。
- 不要固定 CPU。

### `tests/test_serialization.py`

测：

- 保存后加载到新模型/新 optimizer。
- iteration 一致。
- model state dict key 和 tensor 值一致。
- optimizer state dict 结构和值一致。

隐含要求：

- 保存 optimizer state 不能漏。
- load 后 optimizer param group/state 应能恢复。
- 支持 path 或 file-like object。

## 4.4 fixtures 和 snapshots 的作用

### fixtures

`tests/fixtures/` 提供：

- GPT-2 vocab/merges。
- 小语料 `corpus.en`。
- 地址、德语、TinyStories 样本。
- special token 边界测试文本。
- BPE reference vocab/merges。
- TinyStories 小模型 reference checkpoint 和 config。

它们是 deterministic unit tests 的输入。

### snapshots

`tests/_snapshots/*.npz` 和 `.pkl` 是 reference 输出。

例如：

- `test_linear.npz`
- `test_rmsnorm.npz`
- `test_transformer_lm.npz`
- `test_train_bpe_special_tokens.pkl`

snapshot 测试意味着：你的输出不用和某个公式“看起来差不多”，而要在数值容忍度内和 reference implementation 一致。

## 4.5 shape / dtype / device 隐含要求总表

| 模块 | 输入 shape | 输出 shape | dtype/device 要点 |
|---|---|---|---|
| Linear | `(..., d_in)` | `(..., d_out)` | weight 跟随 device/dtype；输入 dtype 通常保持 |
| Embedding | `(...)` token IDs | `(..., d_model)` | token IDs 应为 long/int；weight device 决定输出 device |
| RMSNorm | `(..., d_model)` | same | 内部 fp32，输出 cast 回原 dtype |
| SwiGLU | `(..., d_model)` | `(..., d_model)` | W1/W3/W2 shape 不可反 |
| RoPE | `(..., seq_len, d_k)` | same | token_positions device 对齐；d_k even |
| SDPA | Q/K/V with leading dims | `(..., queries, d_v)` | mask bool；softmax stable |
| MHA | `(..., T, d_model)` | same | head reshape；causal mask broadcast |
| TransformerBlock | `(B,T,D)` | `(B,T,D)` | residual dtype/device 保持 |
| TransformerLM | `(B,T)` | `(B,T,V)` | truncated T <= context_length |
| get_batch | np array | two tensors `(B,T)` | torch LongTensor；放到 requested device |
| CE | `(..., V)`, targets `(...)` | scalar | stable logsumexp；mean |
| checkpoint | model/optimizer | file | path/file-like 均支持 |

---

# 5. 前置知识学习笔记

## 5.1 Unicode / UTF-8

你需要理解三层概念：

1. **字符**：人看到的符号，如 `你`、`a`、`🙃`。
2. **code point**：Unicode 给字符分配的整数。
3. **encoding bytes**：UTF-8 把 code point 编成 1 到 4 个 byte。

这和 assignment 的关系：byte-level BPE 的初始 token 是 bytes，不是字符。一个 emoji 可能对应 4 个 bytes，BPE 可以把这些 bytes 合并成一个 token，也可能保持多个 token。decode 时必须拼 bytes 后统一 UTF-8 decode。

## 5.2 BPE tokenizer

BPE 的核心不是“分词规则”，而是“数据驱动压缩”：高频相邻 pair 会被合并。训练出来的 token 不是语言学意义上的词，而是能提高压缩率的 byte sequence。

你应掌握：

- vocab 的方向：测试中常用 `dict[int, bytes]`。
- merges 的方向：`list[tuple[bytes, bytes]]`，按创建顺序。
- encoding 时如何用 merges 把 pre-token bytes 合并。
- special tokens 如何不参与普通 merges。
- 为什么 GPT-2 tokenizer 里空格常和后面的词合在一起。

## 5.3 Python 文件流处理

这个作业需要处理大文本文件。你要知道：

- file handle 是 iterable，通常按行 yield string。
- `read()` 会把全部内容读进内存，不适合大文件。
- binary file 可用 seek/tell 找 byte offset。
- multiprocessing chunking 要确保 chunk 边界不会破坏 special token 或 UTF-8 decode。
- `np.memmap` 可以让大 token ID 文件像数组一样访问，但底层懒加载。

这和 tokenizer/training loop 都有关：tokenizer 要 lazy，training data 要 mmap。

## 5.4 PyTorch Module / Parameter / autograd

你要能区分：

- `torch.Tensor`：普通张量。
- `torch.nn.Parameter`：注册为 module 参数，会出现在 `model.parameters()` 和 `state_dict()` 中。
- `torch.nn.Module`：管理子模块、参数、buffers、device/dtype 转换、state dict。
- autograd：forward 中使用张量操作，backward 自动计算梯度。

本作业要求手写模块，但仍要继承 `nn.Module`，因为：

- 参数注册。
- state_dict 保存。
- optimizer 获取参数。
- device/dtype 管理。

## 5.5 Transformer decoder

这个作业实现的是 decoder-only causal Transformer，不是 encoder-decoder Transformer。

特点：

- 输入是一段 token prefix。
- 每个位置只能看自己和过去。
- 输出每个位置的 next-token logits。
- 训练时并行预测整段序列的 next tokens。
- 生成时 autoregressive，一次采一个 token。

## 5.6 Attention mask

Attention mask 决定信息能否流动。causal LM 中：

```text
position i 可以 attend 到 j <= i
position i 不可以 attend 到 j > i
```

测试里 mask 的布尔语义是：

- `True`：允许 attend。
- `False`：禁止 attend。

很多库内部用相反语义，所以不要凭习惯写。

## 5.7 RoPE

RoPE 的关键是“旋转 Q/K，而不是加位置向量”。它对每个位置 i 给定一个角度，并在每个二维子空间内旋转。这样 Q_i 与 K_j 的点积会携带 i-j 的相对位置信息。

你应掌握：

- 只作用于 Q/K。
- 作用维度是 per-head dimension。
- sin/cos 可以注册为 buffer，不是 Parameter。
- token_positions 可能是任意 shape 的 batch-like positions。

## 5.8 RMSNorm

RMSNorm 是现代 LLM 常用 normalization。和 LayerNorm 相比：

- 不减均值。
- 只按均方根缩放。
- 通常没有 bias。
- 计算更简单。

和 assignment 的关系：

- Transformer block 是 pre-norm。
- final layer norm 是 RMSNorm。
- ablation 会让你看到 RMSNorm 对稳定性的影响。

## 5.9 SwiGLU

SwiGLU 是 FFN 里的 gated activation。你要理解它为什么有三组权重：

- `W1` 生成 gate branch。
- `W3` 生成 value branch。
- `W2` 投回模型维度。

它让 FFN 不只是逐位置 MLP，而是带有 feature-wise gating 的 MLP。

## 5.10 Cross entropy / perplexity

Cross entropy 是训练目标。Perplexity 是报告指标。

关系：

```text
perplexity = exp(cross_entropy)
```

如果 loss 是 1.45，则 perplexity 约为 `exp(1.45) ≈ 4.26`。这意味着平均每步模型困惑程度相当于在约 4.26 个等概率 token 中选择。

## 5.11 AdamW

你要理解 AdamW 的三层状态：

- gradient：当前 batch 的一阶信息。
- first moment m：梯度指数滑动平均。
- second moment v：梯度平方指数滑动平均。

AdamW 比 SGD 更稳定，但内存更大，因为 optimizer state 通常至少是参数量的 2 倍额外 tensor。训练大模型时 optimizer state 是显存大头之一。

## 5.12 Cosine LR schedule

LR schedule 是训练稳定性的重要组成：

- warmup 防止初期大步更新损坏参数。
- cosine decay 让训练后期更细致收敛。
- min LR 保证后期不是完全停止学习。

实验中要把 LR schedule 和 total training steps 对齐。若你只训练 5000 steps，cosine_cycle_iters 通常就应接近 5000，而不是照搬全量训练设置。

## 5.13 Checkpointing

Checkpoint 是实验工程的基础。真正可恢复的训练状态至少包括：

- model weights
- optimizer moments
- iteration/step
- scheduler progress
- RNG states
- config

测试只验证前三者，但你的实验系统最好保存更多，否则 resume 后实验曲线会不连续。

---

# 6. 推荐完成顺序

## 6.1 总体路线

推荐顺序：

```text
读文档/跑测试
  -> BPE trainer
  -> Tokenizer encode/decode
  -> data loading
  -> softmax/cross entropy
  -> Linear/Embedding/RMSNorm/SiLU/SwiGLU/RoPE
  -> SDPA/MHA
  -> TransformerBlock/TransformerLM
  -> AdamW/LR schedule/gradient clipping/checkpoint
  -> training loop
  -> decoding/generation
  -> TinyStories experiments
  -> ablations
  -> OpenWebText/leaderboard
```

为什么 tokenizer 先做：

- 后续训练要依赖 tokenized data。
- tokenizer 的边界条件多，早做早发现问题。
- BPE trainer 性能可能需要优化，不能拖到最后。

为什么模型模块按小到大：

- snapshots 是逐层对齐的。
- Linear/Embedding/RMSNorm 错了，后面全部错。
- RoPE/MHA shape 错误最难 debug，最好单独测通。

## 6.2 第 0 步：读哪些文档

先读：

1. `README.md`
   - 知道 uv、pytest、数据下载。
2. `AGENTS.md`
   - 知道 AI 使用边界，避免学术诚信风险。
3. PDF 第 1、2、3、4、5 节
   - 不要先看实验部分，先理解核心模块。
4. `tests/adapters.py`
   - 明确测试入口。
5. 对应 `test_*.py`
   - 每实现一个模块前先读对应测试。

## 6.3 第 1 步：先跑哪些测试

刚开始运行：

```sh
uv run pytest
```

预计会因为 `NotImplementedError` 全部失败。然后按模块跑：

```sh
uv run pytest tests/test_train_bpe.py
uv run pytest tests/test_tokenizer.py
uv run pytest -k test_linear
uv run pytest -k test_embedding
uv run pytest -k test_rmsnorm
uv run pytest -k test_swiglu
uv run pytest -k test_rope
uv run pytest -k test_scaled_dot_product_attention
uv run pytest -k test_multihead_self_attention
uv run pytest -k test_transformer_block
uv run pytest -k test_transformer_lm
uv run pytest tests/test_nn_utils.py
uv run pytest tests/test_optimizer.py
uv run pytest tests/test_data.py
uv run pytest tests/test_serialization.py
```

## 6.4 BPE trainer 如何验证

最小验证集：

- 手写极小 corpus，如 `low lower newest widest` 风格，人工算几轮 merge。
- 检查 vocab size 是否等于目标。
- 检查 merges 数量是否为 `vocab_size - 256 - len(special_tokens)`。
- 检查 special tokens 是否在 vocab。
- 检查普通 token 是否包含 special token 片段。
- 检查 tie-breaking 是否符合 bytes lexicographic greater。

性能验证：

- 先在 `corpus.en` 上过 speed test。
- 再在 TinyStories sample 上 profile。
- full TinyStories 前先做 debug subset。

## 6.5 Transformer modules 实现顺序

推荐按测试依赖顺序：

1. Linear
2. Embedding
3. SiLU
4. SwiGLU
5. RMSNorm
6. RoPE
7. Softmax
8. Scaled dot-product attention
9. Multi-head self-attention without RoPE
10. Multi-head self-attention with RoPE
11. Transformer block
12. Transformer LM

每一步只连接一个 adapter，然后跑对应测试。不要一次写完整模型再跑全部测试，否则 debug 搜索空间太大。

## 6.6 Optimizer 和 training loop 如何整合

先独立完成：

- AdamW
- LR schedule
- gradient clipping
- get_batch
- checkpoint

再整合 training loop。训练 loop 中建议记录：

- `step`
- `tokens_processed = step * batch_size * context_length`
- `lr`
- `train_loss`
- `valid_loss`
- `grad_norm`
- `time_elapsed`
- `tokens_per_sec`
- `checkpoint_path`

## 6.7 TinyStories 实验路线

1. 训练 10K BPE tokenizer。
2. 编码 train/valid 为 `uint16` NumPy array。
3. 用推荐小模型配置跑 smoke test：只训练几十步，确认 loss 下降。
4. overfit single minibatch，确认可接近 0 loss。
5. full run 或 low-resource run。
6. 做 LR sweep。
7. 固定较优 LR 后做 batch size sweep。
8. 训练 checkpoint 后生成文本。
9. 保存 learning curves 和生成样本到 writeup。

## 6.8 OpenWebText 实验路线

1. 训练 32K BPE tokenizer。
2. 编码 OWT train/valid。
3. 先用 TinyStories 验证过的训练脚本跑短 OWT smoke test。
4. 观察 OWT loss 是否合理下降。
5. 同 architecture/iterations 训练 main experiment。
6. 生成文本并比较 TinyStories。
7. 准备 leaderboard 尝试。

## 6.9 出错时优先排查

| 症状 | 优先排查 |
|---|---|
| tokenizer 与 tiktoken 不一致 | GPT-2 regex、special token、merge 顺序、bytes vs unicode tie-break |
| BPE trainer 很慢 | 是否每轮全量重扫；pretokenization 是否一次性存大 list；是否没有用 counts |
| Unicode roundtrip 失败 | 是否逐 byte decode；是否用了错误 errors 策略 |
| attention 输出 NaN | mask 是否导致整行全 false；softmax 是否稳定；是否除以 sqrt(d_k) |
| MHA shape mismatch | head reshape/rearrange；output projection weight shape |
| RoPE snapshot 不对 | 偶奇维旋转方向；token_positions；theta 指数；head dim |
| Transformer LM truncated input 不对 | positions 是否按实际 T；context_length 是否只作为 max buffer |
| AdamW 不对 | step 从 1；bias correction；weight decay 顺序；state 初始化 |
| training loss 不降 | causal mask、target shift、LR、zero_grad、model train mode、CE flatten |
| checkpoint load 后训练异常 | optimizer state、iteration、LR schedule progress 没恢复 |

---

# 7. Hidden tests 与边界条件重点提醒

## 7.1 Tokenizer / BPE 边界

- 空字符串 encode 应为 `[]`，decode 应回 `""`。
- 所有 byte values 0-255 应可表示。
- 多字节 Unicode 字符必须 roundtrip。
- invalid UTF-8 decode 要 replace，而不是 raise。
- special tokens 若 overlap，应优先最长 special token。
- special token 应该允许连续出现，例如 `<|endoftext|><|endoftext|>`。
- special token 后的换行、空白、非空白组合要与 tiktoken 一致。
- `encode_iterable` 不能一次性读完整文件。
- `encode_iterable` 结果应与 `encode(full_text)` 一致。
- BPE training tie-breaking 是 bytes tuple 的 lexicographic greater。
- special tokens 在 training 中是 hard boundaries，不贡献 merge stats。

## 7.2 Model 边界

- 所有模块支持 arbitrary leading batch dimensions。
- Linear 权重是 `(d_out, d_in)`，不要存反。
- Embedding weight 是 `(vocab_size, d_model)`。
- RMSNorm 最后一维归一化，内部 fp32。
- RoPE 的 `token_positions` 不是永远 `arange(T)`。
- RoPE 应只作用于 Q/K，不作用于 V。
- MHA 的 causal mask 应 broadcast 到 batch/head。
- Mask 中 `True` 是允许，不是屏蔽。
- TransformerBlock 是 pre-norm，不是 post-norm。
- TransformerLM 需要 final RMSNorm。
- LM head 输出 logits，不要 softmax。

## 7.3 Optimizer / training 边界

- `run_get_batch` 起点最大是 `len(dataset) - context_length - 1`。
- target 是 input 右移一位。
- 输出 tensor 应在指定 device。
- AdamW 要跳过 `grad is None` 参数。
- Gradient clipping 是 global norm。
- `zero_grad` 在训练 loop 中每步要做。
- Checkpoint 必须保存 optimizer state，不只是 model weights。
- LR schedule 边界 step 容易 off-by-one。

---

# 8. 实验日志模板与学习记录方法

建议每次实验都记录以下字段：

| 字段 | 示例 | 说明 |
|---|---|---|
| run_id | `ts_lr_3e-4_seed1` | 唯一标识 |
| dataset | TinyStories / OWT | 数据集 |
| tokenizer | TS-10K / OWT-32K | tokenizer 来源和 vocab size |
| model config | layers/heads/d_model/d_ff/context | 架构 |
| optimizer | AdamW betas, eps, wd | 优化器 |
| LR schedule | max/min/warmup/cosine steps | 学习率 |
| batch size | 64/128/... | global batch |
| tokens processed | step * B * T | 比 step 更公平 |
| seed | 0/1/2 | 可复现 |
| hardware | CPU/MPS/B200 | 影响吞吐 |
| train loss | scalar over time | 训练曲线 |
| valid loss | scalar over time | 主要比较指标 |
| grad norm | scalar | 稳定性 |
| tokens/sec | throughput | 系统效率 |
| checkpoint | path | 可恢复 |
| generated sample | text | 质量观察 |
| notes | divergence? bugs? | 失败也记录 |

推荐把失败实验也写进 log，因为 writeup 中很多问题要求解释“发散”“稳定性”“组件影响”。

---

# 9. 学习路线：如何通过 A1 学 LLM 基础

## 9.1 知识地图

```text
文本与编码
  Unicode -> UTF-8 -> bytes
  -> byte-level BPE -> vocab/merges -> token IDs

模型结构
  token embedding
  -> residual stream
  -> RMSNorm
  -> QKV projections
  -> RoPE
  -> causal self-attention
  -> SwiGLU FFN
  -> Transformer block
  -> LM head logits

训练目标
  logits -> cross entropy -> perplexity

优化系统
  AdamW -> LR schedule -> gradient clipping
  -> data loader -> checkpoint -> training loop

实验方法
  TinyStories debug
  -> LR/batch sweep
  -> generation
  -> ablations
  -> OWT
  -> leaderboard
```

## 9.2 推荐搜索关键词

Tokenizer：

- Byte-level BPE
- GPT-2 tokenizer regex
- tiktoken BPE
- UTF-8 invalid byte sequence
- special tokens tokenizer boundaries

Transformer：

- Attention Is All You Need scaled dot-product attention
- decoder-only Transformer causal mask
- Rotary Position Embedding RoPE
- RMSNorm vs LayerNorm
- SwiGLU GLU Variants Improve Transformer
- pre-norm vs post-norm Transformer

Optimization：

- AdamW decoupled weight decay
- cosine learning rate warmup
- gradient clipping global norm
- language model perplexity

Systems / experiments：

- np.memmap large array training
- tokens per second language model training
- overfit one batch neural network debugging
- model FLOPs utilization MFU
- NanoGPT training loop

## 9.3 推荐阅读材料或论文关键词

不需要一开始通读所有论文，但应知道它们解决什么问题：

- Sennrich et al. 2016：BPE for rare words。
- GPT-2 tokenizer / tiktoken：现代 regex pre-tokenization 与 byte-level BPE。
- Vaswani et al. 2017：Transformer 与 scaled dot-product attention。
- Ba et al. 2016：LayerNorm。
- Zhang & Sennrich 2019：RMSNorm。
- Su et al. 2021：RoPE / RoFormer。
- Shazeer 2020：GLU variants / SwiGLU。
- Kingma & Ba 2015：Adam。
- Loshchilov & Hutter 2019：AdamW。
- Holtzman et al. 2020：nucleus sampling。
- Kaplan et al. / Hoffmann et al.：scaling laws 与 compute-optimal training。

## 9.4 与真实 LLM training pipeline 的关系

真实 LLM pipeline 通常是：

```text
raw data collection
  -> cleaning/dedup/filtering
  -> tokenizer training or tokenizer selection
  -> tokenization and packing
  -> distributed data loading
  -> Transformer training
  -> checkpointing
  -> evaluation
  -> generation/sampling
  -> ablation and scaling experiments
  -> instruction tuning / alignment
```

A1 虽然规模小，但覆盖了核心骨架：

- byte-level tokenizer 是数据进入模型的入口。
- causal Transformer 是现代 LLM 的主体。
- AdamW + schedule + clipping 是基础训练配方。
- memmap/checkpoint/logging 是训练工程的最低要求。
- ablation 是理解模型组件的科学方法。

## 9.5 完成后应掌握的能力

完成后你应该能：

- 解释 tokenizer 为什么影响模型训练效率和效果。
- 手写并 debug byte-level BPE。
- 读懂 GPT-style Transformer 的 shape 流。
- 从零实现 attention、RoPE、RMSNorm、SwiGLU。
- 写稳定 cross entropy 和 AdamW。
- 搭建可恢复的训练 loop。
- 用 validation loss 和 generation 评估 LM。
- 做有对照组的 ablation study。
- 理解小规模训练结果为什么不能直接代表大规模结论。

## 9.6 可额外尝试的小实验

Tokenizer：

- vocab size 从 1K/5K/10K/32K 比较 bytes/token。
- 比较 TinyStories tokenizer 编码 OWT 与 OWT tokenizer 编码 OWT。
- 找 vocab 中最长 token，观察它是空白、URL、重复符号还是常见词组。

Model：

- 单 batch overfit：loss 是否能接近 0。
- 去掉 causal mask：训练 loss 会不会异常低，生成是否坏掉。
- RoPE theta 改变：观察长 context 行为。
- head 数固定/改变：比较速度和 loss。

Optimizer：

- LR range test：从小 LR 到发散 LR。
- AdamW betas `(0.9,0.999)` vs `(0.9,0.95)`。
- weight decay 0 vs 0.01。
- gradient clipping threshold 改变。

Generation：

- temperature grid：0.2/0.7/1.0/1.3。
- top-p grid：0.8/0.9/0.95/1.0。
- 同一 prompt 多 seed 生成，观察 diversity。

---

# 10. 重点提醒：为什么 A1 是后续 Systems / Scaling / Data / Alignment 的基础

## 10.1 对 Systems 的基础作用

A1 里你会第一次遇到：

- 大文件 tokenization 的吞吐和内存问题。
- `np.memmap` 这类 I/O 技术。
- batch size 与 hardware utilization 的关系。
- checkpoint I/O 对训练效率的影响。
- FLOPs/resource accounting。

这些都是后续 systems assignment 的基础。真实大模型训练时，模型代码只是其中一部分，数据吞吐、显存、通信、checkpoint、profiling 同样关键。

## 10.2 对 Scaling 的基础作用

A1 让你做 GPT-2 small/medium/large/XL 的参数和 FLOPs accounting，还要比较 context length 改变后的 FLOPs。你会看到：

- FFN 和 attention projection 通常是大头。
- context length 变长时，attention score 的 `T^2` 项会变重要。
- 参数量、token 数、compute budget 三者共同决定训练效果。

这正是 scaling laws 的前置直觉。

## 10.3 对 Data 的基础作用

Tokenizer 是 data pipeline 的第一道模型化决策。A1 会让你看到：

- 同一 tokenizer 在不同数据集压缩率不同。
- TinyStories tokenizer 对 OWT 可能压缩差、分词碎。
- special token 决定文档边界能否被模型正确感知。
- 数据集难度会影响 loss 和 generation 质量。

后续 data filtering、dedup、mixture、curriculum 都建立在这些基础上。

## 10.4 对 Alignment 的基础作用

Alignment 看似离 A1 很远，但 RLHF/SFT/DPO 等都依赖：

- tokenizer 一致性。
- causal LM forward。
- cross entropy 或 log-prob 计算。
- checkpoint 与 generation。
- decoding 参数对输出质量和安全性的影响。

如果你不能准确算 token log-probs，后续 preference optimization 就无从谈起。

## 10.5 哪些模块值得手写、复盘、画计算图

强烈建议手写并画图复盘：

1. BPE 一轮 merge 如何改变 pair counts。
2. GPT-2 regex 为什么把空格放进 token。
3. Attention scores `(B,H,T,T)` 的 mask 图。
4. RoPE 对二维 pair 的旋转图。
5. Pre-norm residual stream 的计算图。
6. Cross entropy 的 logsumexp 稳定形式。
7. AdamW 的 m/v/weight decay 更新路径。
8. Checkpoint 中 model state 与 optimizer state 的关系。

---

# 11. 最后一页速查清单

## 11.1 代码实现前检查

- [ ] 已读对应 handout section。
- [ ] 已读对应 `test_*.py`。
- [ ] 知道 adapter 输入输出 shape。
- [ ] 知道是否允许用某个 PyTorch API。
- [ ] 准备了一个 toy sanity check。

## 11.2 Tokenizer 提交前检查

- [ ] `test_train_bpe.py` 通过。
- [ ] `test_tokenizer.py` 通过。
- [ ] special tokens hard boundary。
- [ ] overlapping special tokens 最长优先。
- [ ] invalid UTF-8 decode replace。
- [ ] `encode_iterable` 懒执行。
- [ ] TinyStories 10K tokenizer 已训练并记录时间/内存。
- [ ] OWT 32K tokenizer 已训练并记录对比。

## 11.3 Model 提交前检查

- [ ] Linear/Embedding/RMSNorm/SwiGLU/RoPE 单测通过。
- [ ] SDPA 3D/4D 单测通过。
- [ ] MHA with/without RoPE 通过。
- [ ] TransformerBlock 通过。
- [ ] TransformerLM 和 truncated input 通过。
- [ ] 无未来信息泄漏。
- [ ] logits 未提前 softmax。

## 11.4 Training 提交前检查

- [ ] softmax/CE 数值稳定。
- [ ] AdamW/LR schedule 通过。
- [ ] gradient clipping 通过。
- [ ] get_batch 通过。
- [ ] checkpoint 通过。
- [ ] single minibatch overfit 成功。
- [ ] train/valid loss 曲线记录。
- [ ] checkpoint 可恢复。

## 11.5 Writeup 检查

- [ ] 所有 written Problem 都有回答。
- [ ] 所有实验有 learning curve。
- [ ] TinyStories generation 至少 256 tokens。
- [ ] OWT generation 有对比解释。
- [ ] Ablation 有 baseline 对照。
- [ ] 资源 accounting 有公式和单位。
- [ ] leaderboard 若做，有 wall-clock x-axis <45min。

---

# 12. 你应该带走的核心直觉

1. **Tokenizer 是模型的一部分**。它决定序列长度、压缩率、边界信息和训练效率。
2. **Transformer 的难点主要是 shape 和 mask**。数学公式短，但实现中 batch/head/seq/dim 的排列最容易错。
3. **数值稳定不是细节，是正确性**。softmax、CE、RMSNorm、AdamW 都会因为数值处理不当失败。
4. **训练 loop 是科学实验平台**。没有 logging、checkpoint、validation，就无法做可信实验。
5. **Ablation 是理解架构的最好方式**。不要只相信论文或 handout，要通过 TinyStories 小实验观察模块的作用。
6. **小模型训练是大模型训练的缩影**。规模变大后，问题从“能不能写对”变成“能不能高效、稳定、可复现地训练”。
