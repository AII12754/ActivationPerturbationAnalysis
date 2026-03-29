# NarrativeQA 失败样本逐 Token Prefill 误差分析

## 实验目的

在此前已确认的 8 个稳定失败样本上，按 token 位置记录 prefill 重建误差，回答两个问题：

1. 在保留下来的 prompt 内，误差是否集中在某个固定区段。
2. 是否存在比“局部误差偏大”更直接的失败触发因素。

本次只使用 GPU4-6，覆盖三条代表性压缩路径：

- pure_int4
- delta_int4_k1_unigram_int4
- pure_int2

失败样本索引仍使用此前三条路径共同命中的 8 个样本：

- 5, 22, 23, 26, 28, 32, 36, 37

结果目录：

- results_narrativeqa_failure_analysis/narrativeqa_pure_int4_shared_int8table_token_failedpos_formal
- results_narrativeqa_failure_analysis/narrativeqa_delta_int4_k1_unigram_int4_shared_int8table_token_failedpos_formal
- results_narrativeqa_failure_analysis/narrativeqa_pure_int2_shared_int8table_token_failedpos_formal

## 方法

在现有 `analyze_narrativeqa_failure.py` 上新增以下能力：

1. 直接拿到 prefill 原始 hidden 与 reconstructed hidden。
2. 对每个 prefill token 计算：
   - recon cosine
   - per-token MSE
3. 将有效 prompt 切成以下区段：
   - instruction_prefix
   - context_body
   - context_q1/q2/q3/q4（context 四等分）
   - question_block
   - answer_cue
4. 额外统计：
   - 全 prompt 归一化 bin 曲线
   - context 内归一化 bin 曲线
   - 256-token 窗口的 worst-cos / worst-MSE 区段

## 核心发现

### 1. 更强的主因不是“某个 token 区段误差突然飙升”，而是压缩路径把 question 截掉了

这是本次分析中最重要的新发现。

对前 40 个 NarrativeQA 样本，按压缩路径的真实执行流程计算 effective prompt 后可见：

- 压缩路径会在 `process_prefill(...)` 中再次把 chat-wrapped prompt 硬截到 `pipeline.max_seq_len = 32768`。
- FP16 官方生成路径没有这个最终硬截断，因此仍能看到完整问题。

在本次 8 个稳定失败样本里：

- 这 8 个样本的 effective prompt 都在 32768 token 处被截断。
- 截断位置恰好落在 `Question:` 之前或其边界上。
- 也就是说，这 8 个失败样本在压缩路径里实际看不到 question block。

直接量化结果：

- `failed_all_question_missing = True`

此外，在前 40 个样本里一共有 12 个样本出现了同样的“question 被截掉”现象：

- 5, 8, 16, 22, 23, 26, 28, 30, 32, 36, 37, 39

其中稳定失败的是其中 8 个。这说明：

- “question 被截断”是强风险条件，至少对当前失败集合是必要条件。
- 但它不是充分条件，因为还有 4 个 question 被截断的样本没有完全塌掉。

这比“高 cosine 但行为不稳定”更直接，也更接近一个可修复的系统性协议问题。

### 2. 在保留下来的 prompt 内，误差最高的区域是最前面的 retained context，而不是尾部

既然 question block 已经被截出窗口，那么在保留下来的 token 范围内，误差模式主要回答的是：

- retained prompt 里哪里误差相对更大。

三条压缩路径的结论一致：

- 最大的 MSE 出现在 retained context 的最前部，尤其是前 256 token 左右。
- context 按四等分后，误差通常从 `context_q1` 向 `context_q4` 轻微下降。
- 没有观察到“越接近尾部 question 越危险”的模式，因为对这些失败样本来说，question 本身已经不在窗口内。

### 3. pure_int4 / delta_int4 / pure_int2 的逐位置模式非常一致，只是误差绝对量不同

三条路径都在同一批失败样本上复现了类似的空间分布：

- pure_int4：误差最小，但 worst 区域仍集中在 retained context 开头。
- delta_int4：整体质量最好之一，空间模式仍与 pure_int4 一致。
- pure_int2：绝对误差明显更高，但“最差区段在 retained context 开头”的结论不变。

因此，逐位置模式并不支持“只有某种 codec 在某个特殊尾部区段崩掉”的说法。

## 量化结果

### Segment 级统计

#### pure_int4

| segment | tokens | mean recon cosine | mean MSE |
| --- | ---: | ---: | ---: |
| instruction_prefix | 568 | 0.998105 | 0.002940 |
| context_body | 261576 | 0.998160 | 0.001344 |
| context_q1 | 65392 | 0.998155 | 0.001394 |
| context_q2 | 65392 | 0.998152 | 0.001348 |
| context_q3 | 65392 | 0.998159 | 0.001331 |
| context_q4 | 65400 | 0.998174 | 0.001302 |

#### delta_int4_k1_unigram_int4

| segment | tokens | mean recon cosine | mean MSE |
| --- | ---: | ---: | ---: |
| instruction_prefix | 568 | 0.999743 | 0.000418 |
| context_body | 261576 | 0.998703 | 0.000924 |
| context_q1 | 65392 | 0.998676 | 0.000976 |
| context_q2 | 65392 | 0.998693 | 0.000931 |
| context_q3 | 65392 | 0.998720 | 0.000898 |
| context_q4 | 65400 | 0.998721 | 0.000889 |

#### pure_int2

| segment | tokens | mean recon cosine | mean MSE |
| --- | ---: | ---: | ---: |
| instruction_prefix | 568 | 0.956136 | 0.074622 |
| context_body | 261576 | 0.956991 | 0.034030 |
| context_q1 | 65392 | 0.956904 | 0.035285 |
| context_q2 | 65392 | 0.956808 | 0.034141 |
| context_q3 | 65392 | 0.956975 | 0.033699 |
| context_q4 | 65400 | 0.957279 | 0.032996 |

观察：

- 三条路径都没有表现出“后 1/4 context 更差”。
- 相反，最前 1/4 retained context 通常最差。
- pure_int2 的绝对误差更大，但空间形状与 int4 路径一致。

### Worst 256-token 窗口

#### pure_int4

- worst MSE 窗口几乎都落在开头：`0-256`
- worst cosine 窗口分布更分散，但仍都在 `context_body`

#### delta_int4

- worst MSE 的最高窗口也是 `0-256`
- worst cosine 窗口大多仍在 retained context 内部分散出现

#### pure_int2

- worst MSE 最高窗口同样是 `0-256`
- worst cosine 窗口与 pure_int4 基本重合，只是数值更差

### 归一化 bin 曲线

三条路径都呈现相同趋势：

- 归一化第 0 个 context bin 的 MSE 最高
- 后续 bin 整体没有单调恶化，反而通常略有改善

也就是说，如果只看 retained prompt 内部，最敏感区段不是尾部，而是最前面的 retained context。

## 与 FP16 的关系

对同一批 `_id` 对照 FP16 rerun 结果可见：

- FP16 在这 8 个样本上能够给出与问题对应的正常答案。
- 压缩路径则在很多样本上输出 prompt 残片、`Question:` 模板片段或无关问句。

这与上面的协议差异一致：

- FP16 看到完整问题。
- 压缩路径在这些样本上没有看到问题尾部。

因此，本次逐 token 分析实际上揭示出一个比“误差敏感区段”更强的结论：

- 目前 NarrativeQA 的主要失败集合，首先是 effective prompt 截断问题。
- 逐 token 重建误差只是叠加因素，而不是当前这批失败样本的首要解释。

## 结论

本次结果可以压缩成三句话：

1. 对 8 个稳定失败样本，压缩路径的 effective prompt 在 32768 token 处把 `Question:` 之后的内容截掉了；这是本批失败样本最强、最直接的共同条件。
2. 在保留下来的 prompt 内部，误差最高的位置不是尾部，而是 retained context 的最开头，尤其是前 256 token 和第一个 context bin。
3. pure_int4、delta_int4、pure_int2 的逐位置误差空间形状高度一致，因此当前证据不支持“某一种压缩路径只在某个特殊尾部区段发生异常”。

## 直接建议

下一步最值得做的不是继续放大 token-level reconstruction metric，而是先修正协议一致性：

1. 让压缩路径与 FP16 使用同一套最终 effective prompt 长度策略。
2. 至少保证 `Question:` 和 `Answer:` cue 永远保留在窗口内。
3. 修完这一点后，再重新看 NarrativeQA 是否还存在真正由 prefill 重建误差引起的剩余掉分。