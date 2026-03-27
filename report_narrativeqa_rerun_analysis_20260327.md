# NarrativeQA 重跑分析报告

日期：2026-03-27

## 分析范围

这份报告基于以下最新结果，对 NarrativeQA 的异常掉分进行重新分析：

- 一次全新、隔离运行的 FP16 baseline rerun
- 八个压缩配置的独立 rerun
- 样例级输出对比
- 官方 LongBench 路径与压缩运行时实现的代码检查

本次分析使用的主要结果来源：

- `official_longbench/LongBench/pred/qwen14b_fp16_baseline/result.json`
- `official_longbench/LongBench/pred/qwen14b_narrativeqa_rerun_20260327_retry_fp16_baseline/result.json`
- `official_longbench/LongBench/pred/qwen14b_cpuonly_narrativeqa_rerun_20260327_*/result.json`
- `official_longbench/LongBench/pred/qwen14b_narrativeqa_rerun_20260327_retry_fp16_baseline/narrativeqa.jsonl`
- `official_longbench/LongBench/pred/qwen14b_cpuonly_narrativeqa_rerun_20260327_*/narrativeqa.jsonl`

相关实现位置：

- `delta_coding_system/benchmarks/run_longbench_official_benchmark.py`
- `delta_coding_system/pipeline.py`

## 测试设定

- 任务：`narrativeqa`
- 样本数：`200`
- 模型：`Qwen2.5-14B-Instruct`
- 最大上下文长度：`32768`
- 最大生成长度：`128`
- 生成参数：`num_beams=1`、`do_sample=False`、`temperature=1.0`
- 压缩 rerun：全部使用 CPU table only

需要特别说明两点：

1. `32768` 是 `max_seq_len`，表示上下文截断预算，不是生成长度。
2. NarrativeQA 的 `max_new_tokens` 实际是 `128`，来自官方 LongBench 的 `dataset2maxlen.json`。

## 最新结论概览

新的 FP16 rerun 完整跑通，分数仍然是 `28.01`，与原始 FP16 结果完全一致。

八个压缩配置的 NarrativeQA rerun 分数则稳定落在 `15.37` 到 `16.40` 之间，和各自原始结果几乎完全一致。

这说明两件事：

1. FP16 baseline 是稳定可复现的。
2. 压缩路径上的 NarrativeQA 掉分也是稳定可复现的，不是偶发波动。

## 完整分数对比

| 配置 | 原始分数 | rerun 分数 | rerun 相对原始变化 | 相对 FP16 rerun 差值 |
|---|---:|---:|---:|---:|
| `fp16_baseline` | 28.01 | 28.01 | 0.00 | 0.00 |
| `pure_int8` | 16.26 | 16.26 | 0.00 | -11.75 |
| `pure_int4` | 16.40 | 16.40 | 0.00 | -11.61 |
| `pure_int2` | 16.30 | 16.30 | 0.00 | -11.71 |
| `delta_int2_k4_unigram_int4` | 16.24 | 16.25 | +0.01 | -11.76 |
| `delta_int2_k2_unigram_int4` | 15.89 | 15.84 | -0.05 | -12.17 |
| `delta_int4_k1_unigram_int4` | 16.12 | 16.17 | +0.05 | -11.84 |
| `delta_noaffine_int2_k4_unigram_int4` | 15.36 | 15.37 | +0.01 | -12.64 |
| `delta_int2_k4_entropy_unigram_int4` | 16.18 | 16.31 | +0.13 | -11.70 |

从这个表可以直接看到：

- FP16 rerun 和原始结果完全重合。
- 八个压缩配置 rerun 与各自原始结果最大只差 `0.13`。
- 所有压缩配置相对 FP16 都稳定低了大约 `11.6` 到 `12.6` 分。

这不是随机噪声，而是一个高度稳定的系统性偏差。

## 为什么当前结果不太像“单纯压缩误差过大”

你提的判断是对的，而且这正是当前分析里很重要的一点。

如果你的方案只是做了“复用已有中间激活，然后用 delta 和量化尽量修复”，并且单步重建后的 hidden state 与原始 hidden state 差异确实不大，那么按常理说：

1. 单步 hidden 的误差应该是连续、平滑的退化，而不是离散型异常。
2. 由此传导到 KV cache 的差异，通常也应该表现为语义略偏、答案略短、细节略错。
3. 它更像是正常的精度损失，而不应该大量生成“别的问题文本”“prompt 残片”“Answer 模板串入”这种结构性污染。

换句话说，如果根因只是“压缩后 hidden state 和原始 hidden state 仍有一点误差”，那更合理的现象应该是：

- 答案变得更模糊
- 指代更容易错
- 长文定位能力下降
- Rouge/F1 稳定下降

但不应该频繁出现下面这种模式：

- 输出里直接混入 `provide any explanation`
- 输出里混入新的 `Question: ...`
- 输出里变成另一个样本的问题或答案模板

这种现象更像“检索/缓存内容被写脏或串样本”，而不像“单个样本内部的数值误差稍微变大”。

## 这并不代表 KV cache 误差一定完全无关

这里要更严谨一点。

虽然当前证据不支持“仅仅因为 delta/量化误差就导致现在这种掉分规模”，但这不等于 KV cache 偏差完全没有贡献。

更合理的判断是：

1. 压缩重建误差可能会带来一部分正常精度损失。
2. 但它很难独立解释当前 NarrativeQA 上跨所有配置、几乎同幅度、而且伴随 prompt 污染的掉分。
3. 当前更像是“一个共享的系统性问题”为主，数值误差可能只是次要因素。

也就是说，当前最像的结构是：

- 主因：测试时 table 被持续写入，导致跨样本污染
- 次因：压缩重建误差本身可能进一步放大这种污染后的生成漂移

## 样例级证据

目前最强的直接证据仍然来自样例输出本身。

### FP16 rerun 的表现

新的 FP16 rerun 在前几个样本上给出的输出仍然是正常、短答案式、紧贴问题的。例如：

- `_id=341d214...`：`He is staying at the Mulvilles' house.`
- `_id=68fc4dee...`：`To propose a plan for Socrates' escape.`
- `_id=9fbcbb54...`：`Lisa`

这些答案不一定全都与 reference 完全一致，但格式和行为是正常的 QA 生成。

### 压缩 rerun 的异常模式

压缩 rerun 中，部分样本虽然也能正常回答，但一旦出错，错误形态明显不是普通语义误差，而是“文本污染”。

代表性例子如下。

1. `pure_int4`，样本 `_id=36232c9c...`

   预测输出：

   `provide any explanation. Question: Who is the Emperor Rudolph's favorite and why is he known as "Otto of the Silver Hand"?`

   参考答案：

   `Otto was so young.`

2. `delta_int2_k4_unigram_int4`，样本 `_id=36232c9c...`

   预测输出：

   `provide any explanation. Question: Who is the Emperor Rudolph's favorite at the end of the story? Otto of the Silver Hand`

   参考答案：

   `Otto was so young.`

3. `delta_noaffine_int2_k4_unigram_int4`，样本 `_id=36232c9c...`

   预测输出：

   `provide any explanation. Question: Who is the one-eyed Hans and what is his relationship to Baron Conrad? One-eyed Hans is Baron Conrad's trusted servant.`

   参考答案：

   `Otto was so young.`

4. `pure_int4`，样本 `_id=f9cfda8a...`

   预测输出：

   `. What is the name of the painting that was concealed by Vigo's self-portrait?`

   参考答案：

   `Dr. Janosz Poha`

5. `pure_int4`，样本 `_id=67933dc1...`

   预测输出：

   `did the Wyoming Gang decide to attack the Sinsings instead of the Bad Bloods? Answer: To prevent the Sinsings from bartering clues to ultronic secrets to the Hans.`

   参考答案：

   `Because he fought in the first world war.`

这些错误不像是“模型理解错了当前问题”，更像是“当前样本在生成时拿到了别的样本残留的信息”。

## 代码级证据与当前最强根因假设

目前最强的根因假设仍然是：压缩官方评测路径在 `test` 阶段继续写 table，导致 table 在测试中不断被新样本污染。

这个判断来自两个层面的代码事实。

### 1. 官方评测会复用同一个压缩 pipeline

在 `delta_coding_system/benchmarks/run_longbench_official_benchmark.py` 中，一个压缩配置只会构建一次 pipeline，随后在任务循环中重复调用 `_compressed_generate(...)`。

这意味着：同一个配置下，不同样本共享同一个 table 状态。

### 2. `test` 路径中仍然继续写 table

在 `delta_coding_system/pipeline.py` 中：

- `process_prefill(...)` 里虽然有 `is_test = (phase == "test")`
- 但后面的 prefill table update 仍然会异步提交

同样地，在 `process_decode(...)` 中：

- 也计算了 `is_test = (phase == "test")`
- 但 decode 循环里依然调用 `_submit_decode_table_update(...)`

也就是说，当前压缩评测并不是“测试时只读 table”，而是“测试时一边读，一边继续把新样本写回 table”。

对于 NarrativeQA 这种长上下文、开放式生成、并且答案对前文检索较敏感的任务，这种共享状态污染会非常危险。

## 为什么这个假设比“纯数值误差”更能解释现象

当前最关键的现象有三个：

1. 掉分对所有压缩配置都很一致。
2. rerun 后结果几乎完全复现。
3. 错误输出常带有别的问题文本、prompt 残片和答案模板。

如果只是不同压缩配置本身的重建误差不同，那么通常应该看到：

- 不同配置之间差异更大
- 最激进配置显著更差
- 错误形态以语义退化为主

但当前不是这样。

当前看到的是：

- `pure_int8`、`pure_int4`、`pure_int2` 都差不多
- 各种 delta 变体也都差不多
- 错误形态高度相似

这更符合“它们共享了同一个有问题的测试时状态更新机制”。

## 本次 rerun 后可以确认的结论

基于最新结果，现在可以比较确定地说：

1. FP16 NarrativeQA baseline 稳定在 `28.01`。
2. 压缩配置的 NarrativeQA 掉分是真实存在且高度可复现的。
3. 掉分不是某一个压缩配置独有的问题，而是跨配置共享的问题。
4. 从错误形态看，当前主因更像测试时 table 污染，而不是单纯的 delta/量化数值误差。
5. 压缩误差本身可能有贡献，但它目前不像主导因素。

## 下一步最有信息量的验证实验

下一步最值得做的不是继续盲目 rerun 更多压缩配置，而是做一个强对照实验：

1. 把官方压缩评测改成 test 时只读 table、不写 table。
2. 选一个代表配置先重跑 NarrativeQA，例如 `pure_int4` 或 `delta_int2_k4_unigram_int4`。
3. 对比修复前后的 NarrativeQA 分数与异常样例。

如果做完这个 A/B 之后：

- 分数明显回升
- prompt 污染样例明显减少或消失

那就基本可以确认当前主因就是测试时 table 写污染。

## 总结

你提出的直觉是合理的：如果中间激活经过 delta 和量化后已经尽量修复，单纯由此引入的 KV cache 偏差，通常不应该直接造成现在这样统一、稳定、并且带有串样本文本污染特征的大幅掉分。

基于当前 rerun 结果、样例行为和代码路径，最合理的判断是：

- 单纯数值误差不是当前 NarrativeQA 掉分的主要解释。
- 当前主因更可能是测试时共享 table 被持续写入，导致跨样本污染。
- 压缩误差如果有影响，更可能是次级放大因素，而不是决定性根因。