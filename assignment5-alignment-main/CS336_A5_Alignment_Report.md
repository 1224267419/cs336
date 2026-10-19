
# CS336 Assignment 5: Alignment 中文详细报告

## 1. Assignment 基本信息

- **Assignment 名称**: CS336 Assignment 5: Alignment
- **主 PDF**: `cs336_spring2026_assignment5_alignment.pdf`  
  主要讲解 Reasoning RL / GRPO，包括 GSM8K 基线、prompting、policy gradient、GRPO 训练步骤、variants 实验等。
- **Supplement PDF**: `cs336_spring2026_assignment5_supplement_safety_rlhf.pdf`  
  主要讲解 Safety RLHF / SFT / DPO，包括安全评估、SFT 数据集、DPO loss、alignment tax、red-teaming 等。
- **必做主线目标**: 实现 GRPO training、on-policy / off-policy loss、reward shaping、prompting baseline，完成实验曲线与指标分析。
- **Optional supplement 目标**: 实现 SFT training、DPO training，进行安全性评估与 alignment trade-off 分析。
- **与 Assignment 1-4 的关系**: 前几次作业主要熟悉 LLM 微调、RLHF 基础；A5 扩展到复杂 reasoning RL、group normalization、off-policy correction、安全评估。
- **重点代码目录**:
  - `cs336_alignment/` : GRPO training 实现、checkpoint 保存、prompts utils
  - `data/` : GSM8K、HH、MMLU、AlpacaEval 数据集
  - `tests/` : adapters、GRPO/DPO 测试脚本
  - `scripts/` : evaluation scripts
- **数据集作用**:
  - GSM8K: 数学推理问题基线
  - HH: Anthropic Human Preferences 数据
  - MMLU: 多学科知识评测
  - AlpacaEval: Instruction-following LLM 评测
  - SimpleSafetyTests: 简单安全规则评测
- **测试和提交方式**:
  - `pytest` 测试各模块 correctness
  - `test_and_make_submission.sh` 自动打包生成提交文件

## 2. 主 PDF：Reasoning RL / GRPO 结构化解析

### GSM8K Prompting Baselines
- `question_only.prompt`: 仅包含问题文本，无示例；用于零-shot baseline。
- `r1_zero.prompt`: 使用 reward 1-zero-shot baseline，计算 answer correctness。
- `r1_zero_three_shot_gsm8k.prompt`: 三-shot example 强化 baseline，引入示例对比。
- **Response format reward**: 根据生成格式正确性给分，确保模型输出符合指定结构。
- **Answer correctness reward**: 根据生成答案是否正确打分（如数学题答案是否正确）。

### Policy Gradient 基础
- 使用 log-prob trick: loss = -logp(a|s) * advantage
- **Baseline 对 variance 的影响**: 用 baseline 减去 reward 平均值降低 gradient 方差。
- **GRPO group normalization**: 对一组 response 的 advantage 做归一化，控制梯度 scale。
- **Response log probabilities**: 从 causal LM logits 通过 softmax 和 log 提取每个 token 的 logprob。
- **Response masks**: 仅对模型回答部分计算 loss，shift 对齐 labels。
- **Microbatch aggregation**: 分小批次累积梯度，防止 OOM 并保证多样本更新稳定。
- **Standard on-policy GRPO train step**: 输入 batch -> logprobs -> rewards -> baseline -> normalized advantage -> loss -> backward -> gradient clipping -> optimizer step。
- **Learning rate experiments**: 探索不同 lr 对训练稳定性与收敛的影响。
- **Prompt ablation**: 移除示例 / 调整 prompt，分析其对 reward 和训练的影响。
- **RFT / Dr.GRPO / MaxRL variants**: 不同 GRPO 变体实现，更复杂 reward shaping 或 estimator。
- **Difficulty reweighting**: 难题加权，提高模型对难题的学习能力。
- **Advantage normalization**: 归一化 advantage 防止大梯度。
- **Off-policy surrogate objectives**: 利用旧策略 logprobs 做 importance sampling 校正。
- **Token-level importance weighting**: 每个 token 的 logprob 被加权调整，以减少 off-policy bias。
- **Sequence-level GSPO**: 对整个生成序列计算梯度，类似 PPO 但用于 sequence。
- **Clipping**: 控制 importance ratio，避免梯度爆炸。
- **Off-policy experiments**: 对比 on-policy 与 off-policy training 效果。
- **自定义 estimator 实验**: 用户可实现自定义 advantage estimator 并测试效果。

### 主要 Problem / TODO / Deliverable
- 每个问题包含名称、分值、类型（written / implementation / experiment）、adapter、对应测试、实验曲线、metrics、示例。
- **tokenize_prompt_and_output**:
  - `input_ids`: prompt+response token
  - `labels`: 用于计算 loss，response 对应 token 非 mask 部分
  - `response_mask`: 对齐 labels，确保仅计算 response 的 loss
- **从 causal LM logits 得到 response token logprobs**: softmax -> log -> mask -> sum
- **Group normalization 维度**: [group_size, sequence_length] -> normalization over group
- **Sequence vs constant normalization**: sequence 按序列归一化，constant 按固定值归一化
- **On-policy vs Off-policy**: on-policy 使用当前策略生成数据，off-policy 使用旧策略或 cached 数据
- **Importance ratio bias-variance tradeoff**: r = pi(a|s)/mu(a|s)，r clip 控制 variance
- **GRPO vs GSPO clipping**: GRPO 对 group normalization 后的 advantage clip，GSPO 对 sequence-level ratio clip
- **Train step 处理 microbatch & gradient clipping**: 避免 OOM & 保证梯度稳定
- **Tests 检查 grads 清空**: 防止累积梯度影响下一步

## 3. Supplement PDF：Safety RLHF / SFT / DPO 解析

### Zero-shot baselines
- MMLU / GSM8K / AlpacaEval / SimpleSafetyTests: 测试模型 zero-shot 能力
- Prompt formatting 和 parser 实现保证输出可结构化解析
- SFT dataset inspection & PackedSFTDataset: 支持 batch iterator, padding, mask
- SFT training loop & evaluation: 在 MMLU/GSM8K/AlpacaEval/SST 上评估
- Red-teaming: 通过 adversarial prompts 检查模型安全性
- Anthropic HH 数据 inspection: 分析 preference 数据

### DPO Loss & Training
- DPO loss: contrastive loss，用于 preference learning
- DPO training: 利用 prompt-response preference 数据训练模型
- Alignment tax: 安全性/对齐带来的性能 trade-off
- Supplement Problems: 部分 optional，带 tests

## 4. 代码结构解析

- `drgrpo_grader.py`: math reward 计算
- `vllm_utils.py`: generation / server / weight sync
- `checkpoint.py`: 保存和恢复模型参数
- prompt 文件格式约束: 每行一个示例，含 prompt/response 分隔符
- 数据格式: GSM8K / HH / MMLU / AlpacaEval / SST
- `tests/adapters.py`: hook 实现要求
- `test_grpo.py`: 测试 GRPO loss, logprobs, normalization, train step
- `test_data.py`: PackedSFTDataset 检查
- `test_dpo.py`: DPO loss correctness
- `test_metrics.py`: parser 对 output correctness

## 5. 前置知识

- Policy gradient: 基础 RL 算法，loss = -logp * advantage
- Logprob trick: 通过 logp 计算 gradient
- Reward / baseline / advantage: baseline 减少 gradient 方差
- GRPO: group normalization 改进 policy gradient
- PPO clipping: 控制 ratio 避免梯度爆炸
- Off-policy correction: importance sampling
- Importance sampling: r = pi/old_pi
- GSPO: sequence-level PPO
- Response-level vs token-level loss: token 更精细，sequence 更全局
- Microbatch gradient accumulation: 避免 OOM
- SFT: supervised finetuning
- DPO: preference learning
- Preference data: Human preference 或 synthetic reward
- Evaluation: MMLU/GSM8K/AlpacaEval/SST
- vLLM / HF generation: 高效推理 / generation

## 6. 推荐完成顺序

### 主线
1. 先读 README 和主 PDF
2. 先读 prompts
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
