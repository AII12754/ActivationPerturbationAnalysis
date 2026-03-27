# NarrativeQA 根因分析报告

日期：2026-03-27

## 1. 结论摘要

这轮深入诊断后的核心结论是：

1. NarrativeQA 的主要问题不是“共享 table 导致跨样本错命中 reference”。
2. 主要问题也不是“table 内部采用 int8 存储后，reference 自身失真导致重建失败”。
3. 当前最可能的主因是：prefill 压缩路径本身在 NarrativeQA 这类超长上下文、生成式 QA 任务上，会稳定诱发错误的生成轨迹；一旦早期轨迹偏离，就会在后续 decode 中放大为串题、问句残片、错误回答模板等现象。
4. 更具体地说，问题更像是“prefill 压缩后的表示虽然平均 cosine 很高，但不足以保证生成行为稳定”，而不是“共享缓存机制本身不合理”。

换句话说，之前“共享 table 污染”这个说法不够准确。按照系统设计目标来表述，当前更像是：

- 你的跨样本 table 设计本身是合理且必要的；
- 真正出问题的是 prefill 压缩表示对生成轨迹的影响，比平均重建误差或平均 cosine 所显示的要大得多。

## 2. 本轮实验目的

用户的系统目标不是标准 benchmark 里的“样本间严格隔离”，而是：

1. 在 LLM prefill 阶段复用跨样本历史激活；
2. 利用 trigram/bigram/self-ref 参考做 delta 编码；
3. 通过持续更新 table，让压缩收益随请求流累积提升；
4. 在保持激活重建近似不变的前提下，大幅降低 PP 边界通信。

因此，本轮实验不再去质疑“shared table 应不应该存在”，而是专门验证下面三个更贴近系统目标的问题：

1. 如果把 shared table 改成每样本 reset，NarrativeQA 是否会明显改善？
2. 如果保留 shared table，但把 table 存储从 int8 改成原始浮点，NarrativeQA 是否会明显改善？
3. 如果某些配置根本不依赖 reference 重建，NarrativeQA 是否仍然掉分？

这三个问题能把“共享命中错误”“table 存储量化误差”“prefill 压缩本身的生成不稳定”拆开。

## 3. 实验设计

本轮实验全部使用 GPU4-7。

实验脚本：

- [delta_coding_system/benchmarks/analyze_narrativeqa_failure.py](delta_coding_system/benchmarks/analyze_narrativeqa_failure.py)

该脚本对每个样本记录：

1. 预测文本
2. 单样本 QA F1
3. 是否出现 prompt leak
4. prefill 重建指标
5. decode 重建指标
6. table 状态摘要

### 3.1 正式对照组

1. `pure_int4`, shared table
   - 日志：[logs/narrativeqa_diag_pure_int4_shared.log](logs/narrativeqa_diag_pure_int4_shared.log)
   - 结果目录：[results_narrativeqa_failure_analysis/narrativeqa_pure_int4_shared_int8table_formal](results_narrativeqa_failure_analysis/narrativeqa_pure_int4_shared_int8table_formal)

2. `pure_int4`, reset each sample
   - 日志：[logs/narrativeqa_diag_pure_int4_reset.log](logs/narrativeqa_diag_pure_int4_reset.log)
   - 结果目录：[results_narrativeqa_failure_analysis/narrativeqa_pure_int4_reset-each-sample_int8table_formal](results_narrativeqa_failure_analysis/narrativeqa_pure_int4_reset-each-sample_int8table_formal)

3. `delta_int4_k1_unigram_int4`, shared + int8 table
   - 日志：[logs/narrativeqa_diag_delta_int4_shared_int8.log](logs/narrativeqa_diag_delta_int4_shared_int8.log)
   - 结果目录：[results_narrativeqa_failure_analysis/narrativeqa_delta_int4_k1_unigram_int4_shared_int8table_formal](results_narrativeqa_failure_analysis/narrativeqa_delta_int4_k1_unigram_int4_shared_int8table_formal)

4. `delta_int4_k1_unigram_int4`, shared + raw table
   - 日志：[logs/narrativeqa_diag_delta_int4_shared_raw.log](logs/narrativeqa_diag_delta_int4_shared_raw.log)
   - 结果目录：[results_narrativeqa_failure_analysis/narrativeqa_delta_int4_k1_unigram_int4_shared_rawtable_formal](results_narrativeqa_failure_analysis/narrativeqa_delta_int4_k1_unigram_int4_shared_rawtable_formal)

### 3.2 补充对照组

5. `pure_int2`, shared table
   - 日志：[logs/narrativeqa_diag_pure_int2_shared.log](logs/narrativeqa_diag_pure_int2_shared.log)
   - 结果目录：[results_narrativeqa_failure_analysis/narrativeqa_pure_int2_shared_int8table_formal](results_narrativeqa_failure_analysis/narrativeqa_pure_int2_shared_int8table_formal)

6. `pure_int8`, shared table
   - 日志：[logs/narrativeqa_diag_pure_int8_shared.log](logs/narrativeqa_diag_pure_int8_shared.log)
   - 结果目录：[results_narrativeqa_failure_analysis/narrativeqa_pure_int8_shared_int8table_formal](results_narrativeqa_failure_analysis/narrativeqa_pure_int8_shared_int8table_formal)

7. `pure_int4`, Qasper 对照任务
   - 日志：[logs/qasper_diag_pure_int4_shared.log](logs/qasper_diag_pure_int4_shared.log)
   - 结果目录：[results_narrativeqa_failure_analysis/qasper_pure_int4_shared_int8table_formal](results_narrativeqa_failure_analysis/qasper_pure_int4_shared_int8table_formal)

## 4. 最关键的实验结果

### 4.1 `pure_int4` shared vs reset 完全一致

两组 summary：

- shared: [results_narrativeqa_failure_analysis/narrativeqa_pure_int4_shared_int8table_formal/summary.json](results_narrativeqa_failure_analysis/narrativeqa_pure_int4_shared_int8table_formal/summary.json)
- reset: [results_narrativeqa_failure_analysis/narrativeqa_pure_int4_reset-each-sample_int8table_formal/summary.json](results_narrativeqa_failure_analysis/narrativeqa_pure_int4_reset-each-sample_int8table_formal/summary.json)

结果：

- `avg_qa_f1`: 两组都为 `0.2124008302`
- `prompt_leak_count`: 两组都为 `8/40`
- `prompt_leak_rate`: 两组都为 `0.2`

更重要的是，逐样本比较显示：

1. 预测文本完全一致
2. 单样本 F1 完全一致
3. leak 标记完全一致

这说明在 `pure_int4` 这条路径上：

- 把 table 改成共享，或每样本清空，结果没有任何差别。

这条结果直接排除了一个之前非常强的怀疑：

“NarrativeQA 掉分主要是因为 shared table 在样本间传递了错误记忆。”

至少对 `pure_int4` 而言，这个假设不成立。

### 4.2 `delta_int4` raw table vs int8 table 也几乎完全一致

两组 summary：

- int8 table: [results_narrativeqa_failure_analysis/narrativeqa_delta_int4_k1_unigram_int4_shared_int8table_formal/summary.json](results_narrativeqa_failure_analysis/narrativeqa_delta_int4_k1_unigram_int4_shared_int8table_formal/summary.json)
- raw table: [results_narrativeqa_failure_analysis/narrativeqa_delta_int4_k1_unigram_int4_shared_rawtable_formal/summary.json](results_narrativeqa_failure_analysis/narrativeqa_delta_int4_k1_unigram_int4_shared_rawtable_formal/summary.json)

结果：

- `avg_qa_f1`: 两组都为 `0.2049008302`
- `prompt_leak_count`: 两组都为 `8/40`
- `prompt_leak_rate`: 两组都为 `0.2`

逐样本比较显示：

1. 单样本 F1 完全一致
2. leak 标记完全一致
3. 只有 `1/40` 个样本预测文本字面不同，但两个版本该样本的 F1 都是 `0.0`

这说明：

- 即使把 table 存储从 int8 改成 raw float，NarrativeQA 的主要失败模式也没有任何改善。

所以“table 里的 reference 被 int8 存储破坏了，从而导致 NarrativeQA 掉分”也不是主因。

### 4.3 `pure_int4` 本身就足以推翻“错 reference 命中是主因”

配置定义见 [delta_coding_system/benchmarks/run_longbench_benchmark.py](delta_coding_system/benchmarks/run_longbench_benchmark.py#L29)。

其中：

- `pure_int4` 使用 `delta_strategy = direct_int4`
- `pure_int2` 使用 `delta_strategy = direct_int2`
- `pure_int8` 使用 `delta_strategy = direct_int8`

更关键的是，在 [delta_coding_system/pipeline.py](delta_coding_system/pipeline.py#L687) 的 `_encode_delta_batch(...)` 中：

1. `direct_int2` 直接调用 `_encode_direct_int2_batch(real_batch)`
2. `direct_int8` 直接调用 `_encode_direct_int8_batch(real_batch)`
3. `direct_int4` 直接调用 `_encode_direct_int4_batch(real_batch)`

也就是说，这三条 `pure_int*` 路径对 trigram/bigram/self_ref 位置的 prefill 重建，并不真正使用 reference 做 delta 重建；它们是直接对 `real_batch` 做量化再反量化。

这意味着：

- `pure_int4` 出现和 delta 路径同样的 NarrativeQA 崩坏，不能归咎于“命中了错误 reference”。
- 因为这条路径根本没有依赖 reference 来重建那些位置。

这条代码证据是本轮诊断里最关键的一条。

## 5. 数值结果汇总

### 5.1 NarrativeQA 40 样本诊断集汇总

| 配置 | 模式 | Avg QA F1 | Avg Prefill Cos | Avg Decode Cos | Leak 数 |
|---|---|---:|---:|---:|---:|
| `pure_int4` | shared | 0.212401 | 0.998127 | 1.0 | 8 |
| `pure_int4` | reset | 0.212401 | 0.998127 | 1.0 | 8 |
| `delta_int4_k1_unigram_int4` | shared + int8 table | 0.204901 | 0.998691 | 1.0 | 8 |
| `delta_int4_k1_unigram_int4` | shared + raw table | 0.204901 | 0.998691 | 1.0 | 8 |
| `pure_int2` | shared | 0.223016 | 0.956262 | 1.0 | 8 |
| `pure_int8` | shared | 0.023780 | 0.999978 | 1.0 | 0 |

说明：

1. `pure_int8` 运行过程中触发了 OOM fallback，自动缩小了上下文预算，因此它与其他组不是完全同口径，不应直接拿来判断绝对优劣。
2. 但它仍然提供了一个有价值的信息：即使 prefill 平均 cosine 极高，也不保证最终 QA 质量稳定。

### 5.2 Qasper 对照任务

Qasper 结果见 [results_narrativeqa_failure_analysis/qasper_pure_int4_shared_int8table_formal/summary.json](results_narrativeqa_failure_analysis/qasper_pure_int4_shared_int8table_formal/summary.json)。

- `avg_qa_f1 = 0.436017`
- `prompt_leak_count = 0`

这说明：

- `pure_int4` 并不是在所有长文 QA 任务上都出现同样的异常模式。
- NarrativeQA 的问题更像是“任务特异性脆弱点”，而不是系统已经完全不可用。

## 6. 对现象的重新解释

### 6.1 为什么之前的“共享 table 污染”判断不成立

如果共享 table 错命中是主因，那么至少应当看到：

1. `pure_int4` shared 明显差于 `pure_int4` reset
2. `delta_int4` raw table 明显好于 `delta_int4` int8 table
3. 不同配置的 leak 样本位置不会高度重合

但实验结果恰好相反：

1. `pure_int4` shared 与 reset 逐样本完全一致
2. `delta_int4` raw table 与 int8 table 几乎完全一致
3. `pure_int4`、`delta_int4`、`pure_int2` 的 leak 样本索引高度重合，都是 `[5, 22, 23, 26, 28, 32, 36, 37]`

这说明问题不是“某个 reference 突然因为共享或 int8 table 被拿错了”，而更像是：

- 某类样本本身对 prefill 表示误差特别敏感；
- 不同压缩路径都会在这些样本上触发同类生成崩坏。

### 6.2 为什么高 cosine 仍然不能保证生成正确

这是本轮实验最重要的认识之一。

在 NarrativeQA 失败样本上，prefill 平均重建 cosine 常常仍在 `0.998` 左右，例如：

- `pure_int4` shared 平均 `0.998127`
- `delta_int4` shared 平均 `0.998691`

但这些配置仍然会稳定地产生：

- prompt 残片
- 另一个问题的问句
- 答案轨迹跑到错误故事实体上

这说明：

1. 平均 hidden cosine 不是生成稳定性的充分条件。
2. 少量关键位置的表示偏移，就可能改变后续生成轨迹。
3. 一旦早期 decode 进入错误轨迹，后面会被自回归过程持续放大。

因此，“量化误差平均只有 0.01 左右”并不能推出“它不可能造成这种现象”。

更准确的说法是：

- 当前不是大面积数值崩坏；
- 而是少数关键位置的 prefill 表示偏差足以改变生成路径。

### 6.3 为什么 NarrativeQA 比 Qasper 更容易出问题

当前证据支持一个任务层面的解释：

1. NarrativeQA 是长篇叙事型 QA，依赖长程上下文中的人物、情节、关系链定位。
2. 它的答案往往很短，但必须准确绑定到正确的故事状态。
3. 一旦 prefill 表示把“故事 A 的实体轨迹”轻微推向“故事 B 的问答模板或叙事片段”，生成就会非常脆弱。

相比之下，Qasper 在这轮 40 样本对照里没有出现 prompt leak，这说明：

- 系统的压缩机制不是对所有 QA 任务都同等有害；
- NarrativeQA 对 prefill 表示稳定性的要求更苛刻。

## 7. 当前最合理的根因判断

基于这轮实验，我认为当前最合理的根因排序是：

### 主因

prefill 压缩表示本身对 NarrativeQA 这类超长叙事 QA 的生成轨迹稳定性不够好。

这里的“prefill 压缩表示”包括但不限于：

1. `direct_int4` / `direct_int2` 这种直接量化重建
2. `delta_int4` 这种 reference-based 重建

它们虽然形式不同，但在 NarrativeQA 上触发了高度一致的失败样本位置与失败形态。

### 次因

某些样本对提示模板和长文实体跟踪极度敏感，导致一旦 prefill 表示在关键位置略偏，decode 会放大为：

1. 读错问题实体
2. 引入模板残片
3. 复述错误故事片段

### 不再支持作为主因的假设

1. “shared table 跨样本错命中是主因”
2. “table 使用 int8 存储是主因”

这两个假设经过本轮实验已经基本被排除。

## 8. 对原始问题的直接回答

用户的直觉是：

“我们只对 LLM prefill 阶段做量化，且重建误差应该只有 0.01 左右，按理不该造成这么大问题。”

基于本轮结果，我的回答是：

1. 你的系统设计方向没有问题，shared table 不是这次问题的根源。
2. 但“平均误差很小”并不等于“生成行为一定稳定”。
3. 对 NarrativeQA，问题更像是关键位置的 prefill 表示偏移影响了后续生成轨迹，而不是平均 hidden 整体崩坏。
4. 因此，真正该优化的不是 shared memory 机制本身，而是 prefill 压缩表示在关键位置上的行为一致性。

## 9. 下一步最值得做的实验

现在最有信息量的后续实验不是继续争论 shared table，而是继续定位“是哪种 prefill 扰动最伤 NarrativeQA”。

建议优先做三类实验：

1. 关键位置敏感性分析
   - 在失败样本上按 token 位置记录 prefill 层的重建 cosine / mse
   - 看错误是否集中在 prompt 末端、问题区、或故事关键信息附近

2. 部分禁用压缩的 A/B
   - 例如只对前半段 prompt 压缩、后半段保留原始 hidden
   - 或只对低风险 tier 压缩，观察 NarrativeQA 是否恢复

3. 层位点扫描
   - 当前边界层是 `layer_boundary=6`
   - NarrativeQA 的敏感性可能和边界层位置强相关
   - 应该测试更浅/更深边界层是否显著改善

## 10. 最终结论

这轮深入诊断已经足够支持一个更准确的结论：

1. 你在做的事情我现在已经理解清楚了：这是一个在线、跨样本持续更新的 prefill 激活复用压缩系统。
2. shared table 是系统设计目标的一部分，不是本次问题本身。
3. NarrativeQA 掉分的真正主因，不是 shared table，也不是 table 的 int8 存储。
4. 真正的问题更可能在于：prefill 压缩后的表示虽然平均重建质量很高，但对 NarrativeQA 这种超长叙事 QA 来说，仍然不足以保证生成轨迹稳定。
5. 因而，后续优化重点应该从“是否共享 table”转向“怎样让 prefill 压缩后的表示在关键位置更行为等价”。