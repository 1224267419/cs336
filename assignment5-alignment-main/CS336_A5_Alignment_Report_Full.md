
# CS336 Assignment 5: Alignment 中文详尽报告

本报告结合主 PDF、Supplement PDF、代码和数据目录，全面解析 Assignment 5，适合作业参考和课件使用。

## 1. Assignment 基本信息

- **Assignment 名称**: CS336 Assignment 5: Alignment
- **主 PDF**: `cs336_spring2026_assignment5_alignment.pdf`  
  内容聚焦 Reasoning RL / GRPO，包括数学推理 baseline、prompting 设计、policy gradient、GRPO 训练与 variants。  
- **Supplement PDF**: `cs336_spring2026_assignment5_supplement_safety_rlhf.pdf`  
  内容聚焦安全性 RLHF / SFT / DPO，包括数据处理、DPO loss、alignment tax、安全评测与 red-teaming。  
- **必做主线目标**: 实现 GRPO on-policy loss、off-policy correction、reward shaping、prompting baseline、实验分析。  
- **Optional supplement 目标**: 实现 SFT / DPO 训练与评估，分析安全性与 alignment trade-off。  
- **与 Assignment 1-4 的关系**: A1-A4 熟悉 LLM 微调与简单 RLHF；A5 扩展到复杂 reasoning RL、group normalization、off-policy correction、安全评估。  
- **重点代码目录**:  
  - `cs336_alignment/`: GRPO 训练核心代码、checkpoint 保存、prompt utils
  - `data/`: GSM8K、HH、MMLU、AlpacaEval、SimpleSafetyTests 数据集
  - `tests/`: adapters、GRPO/DPO 测试脚本
  - `scripts/`: evaluation scripts
- **数据集作用**:
  | 数据集 | 用途 |
  |--------|------|
  | GSM8K | 数学推理问题 baseline |
  | HH | Human preference 数据 |
  | MMLU | 多学科知识评测 |
  | AlpacaEval | Instruction-following LLM 评测 |
  | SimpleSafetyTests | 简单安全规则评测 |
- **测试和提交方式**:
  - `pytest` 测试各模块正确性
  - `test_and_make_submission.sh` 自动打包提交

## 2. 主 PDF：Reasoning RL / GRPO 解析

### 2.1 GSM8K Prompting Baselines

- **`question_only.prompt`**: 零-shot baseline，仅包含问题文本。  
- **`r1_zero.prompt`**: reward 1-zero-shot baseline，计算答案 correctness。  
- **`r1_zero_three_shot_gsm8k.prompt`**: 三-shot baseline，引入示例强化模型。  
- **Prompt 使用注意事项**:
  - 示例顺序对结果影响较大
  - response 部分必须结构化，便于 reward 计算

### 2.2 Reward Functions

1. **Answer Correctness Reward**:
\[ r = \mathbf{1}\{ 	ext{model answer} = 	ext{ground truth} \} \]  
2. **Response Format Reward**: 对生成格式正确加分（JSON / 数学公式等）  
3. **Token-level Reward**: 对每个 token 单独打分，用于 sequence-level logprob 加权  
4. **Sequence-level Reward**: 对整个生成序列打分，用于 GSPO

### 2.3 Policy Gradient 基础

- **Loss**: \( L = - \sum_t \log p_	heta(a_t|s_t) \cdot A_t \)  
- **Baseline**: 减少 variance  
- **Advantage**: \( A_t = r_t - b(s_t) \)  
- **Microbatch**: 小批次累积梯度，避免 OOM  
- **Gradient Clipping**: 控制梯度爆炸  
- **Logprob Extraction**: 从 causal LM logits 计算 log softmax -> 选 response token -> mask 对齐

### 2.4 GRPO 训练步骤与 Group Normalization

- **Group normalization**: 对同一 group 的 advantage 做归一化
- **Train Step 伪代码**:
```python
for microbatch in batch.split(microbatch_size):
    logits = model(microbatch.input_ids)
    logprobs = extract_logprobs(logits, microbatch.response_mask)
    rewards = compute_rewards(microbatch.labels)
    advantages = normalize(rewards - baseline)
    loss = -(logprobs * advantages).mean()
    loss.backward()
clip_gradients(model.parameters(), max_norm)
optimizer.step()
optimizer.zero_grad()
```
- **Sequence normalization vs Constant normalization**: sequence 按序列归一化，constant 使用固定值
- **Importance ratio clipping**: \( r = rac{\pi_	heta(a|s)}{\pi_	ext{old}(a|s)} \), clip 防止 off-policy variance

### 2.5 On-policy vs Off-policy

| 属性 | On-policy | Off-policy |
|------|-----------|------------|
| 数据来源 | 当前策略 | 旧策略 / cache |
| Advantage | 当前 batch | importance sampling 校正 |
| Bias | 低 | 可能有 bias |
| Variance | 高 | clip 可控 variance |

- **Importance Sampling**: \( L = - r \cdot \log p_	heta(a|s) \cdot A \)

### 2.6 GRPO Variants & Experiments

- **RFT / Dr.GRPO / MaxRL**: reward shaping 与 estimator 变体
- **Difficulty Reweighting**: 难题加权
- **Advantage Normalization**: 防止大梯度
- **Off-policy Surrogate Objectives**: 利用旧策略数据
- **Token-level Importance Weighting**: 控制每个 token 的梯度贡献
- **自定义 estimator**: 支持用户实验

## 3. Supplement PDF：Safety RLHF / SFT / DPO

### 3.1 Baselines & Prompting

- Zero-shot MMLU / GSM8K / AlpacaEval / SimpleSafetyTests
- Prompt formatting + parser 确保可结构化解析
- Red-teaming: adversarial prompt 测试模型安全性
- Anthropic HH 数据 inspection: human preference 数据分析

### 3.2 SFT (Supervised Fine-Tuning)

- **PackedSFTDataset**: 支持 batch iterator, padding, mask
- **SFT Training Loop**:
```python
for batch in dataloader:
    logits = model(batch.input_ids)
    loss = cross_entropy(logits, batch.labels, mask=batch.mask)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```
- **Evaluation**: MMLU / GSM8K / AlpacaEval / SST

### 3.3 DPO (Direct Preference Optimization)

- **DPO Loss**: contrastive loss
\[ L = - \log \sigma(s_	heta(	ext{chosen}) - s_	heta(	ext{rejected})) \]  
- **Training**: 以 preference 数据训练
- **Alignment Tax**: 安全性带来的性能 trade-off

## 4. 代码结构解析

- `drgrpo_grader.py`: math reward 计算
- `vllm_utils.py`: generation / server / weight sync
- `checkpoint.py`: 模型参数保存与加载
- Prompt 文件格式: 每行一个示例，prompt/response 分隔
- 数据格式: GSM8K / HH / MMLU / AlpacaEval / SST
- `tests/adapters.py`: hooks 实现要求
- `test_grpo.py`: 测试 GRPO loss、logprobs、normalization、train step
- `test_data.py`: 检查 PackedSFTDataset
- `test_dpo.py`: DPO loss correctness
- `test_metrics.py`: parser correctness

## 5. 前置知识解析

- Policy gradient, logprob trick, reward / baseline / advantage
- GRPO / PPO clipping / off-policy correction / importance sampling
- GSPO: sequence-level PPO
- Response-level vs token-level loss
- Microbatch gradient accumulation
- SFT / DPO / preference data
- Evaluation: MMLU/GSM8K/AlpacaEval/SST
- vLLM / HF generation

## 6. 推荐完成顺序与常见坑

### 主线顺序

1. 阅读 README 和主 PDF
2. 阅读 prompts
3. 跑 `uv run pytest tests/test_grpo.py`
4. 实现 tokenization / masks
5. 实现 logprobs
6. 实现 rewards
7. 实现 group normalization
8. 实现 on-policy loss
9. 实现 aggregation
10. 实现 standard GRPO train step
11. 做 prompting baselines
12. 做 GRPO experiments
13. 实现 variants
14. 实现 off-policy losses
15. 做 multi-seed comparisons
16. 做自定义 estimator

### Supplement

1. 实现 parsers
2. 跑 zero-shot eval
3. 实现 PackedSFTDataset
4. 写 SFT loop
5. 评估 SFT
6. 实现 DPO loss
7. 跑 DPO
8. 比较安全性和 alignment tax

### 常见坑

- prompt/output 边界错
- mask shift
- group size 维度错
- reward std 为 0 的处理
- sequence normalization scale 错
- old/new logprob 混淆
- cliprange 用错
- train step 后 gradient 未清空
- DPO 中 prompt logprob 没有正确抵消
- parser 对模型自由文本输出过于脆弱

## 7. 学习路线

- **知识地图**: RLHF -> PPO/GRPO/GSPO -> SFT/DPO -> Safety evaluation
- **推荐搜索关键词**: "policy gradient NLP", "RLHF PPO clipping", "GRPO LLM", "DPO preference learning", "vLLM generation"
- **推荐深入方向**: reward shaping, off-policy correction, alignment tax, multi-task SFT
- **与 RLHF、PPO、DPO、RFT、reasoning RL 的关系**: A5 综合应用这些方法进行 post-training
- **与安全评估的关系**: Supplement PDF 练习模型对有害/偏差内容的对齐
- **完成后应掌握的能力**: GRPO / off-policy / importance sampling / SFT / DPO / reward shaping / alignment evaluation
- **可额外做的小实验**: reward shaping, prompt variation, group size tuning, clipping strategies, DPO beta tuning
