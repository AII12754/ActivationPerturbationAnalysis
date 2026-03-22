# Delta-Coding 激活压缩系统 —— 大样本 Phasewise 实验报告

> 实验日期: 2026-03-21  
> 模型: Qwen2.5-32B-Instruct  
> hidden_dim: 5120  
> layer_boundary: 6  
> 运行方式: 单卡单进程, 多数据集并行, 不使用 TP  
> GPU 分配: GPU 1-6, 避开 GPU 0  
> 数据集: wikitext2, sharegpt, gsm8k, cnn_dm, alpaca, triviaqa  
> 每数据集: 30 warmup + 100 test requests  
> decode 长度: 固定贪心生成 64 token  
> prompt 最大长度: 384

---

## 1. 本轮实验目标

这轮实验专门回答四个问题:

1. 只保留潜在可行路线, 去掉稀疏路线后, delta 路径和 unigram 路径分别谁最优。
2. prefill 阶段和 decode 阶段是否应该使用不同默认策略。
3. 去掉 affine 之后, delta 路径到底能省多少时延, 会损失多少质量。
4. 当前系统里用户感知到的高时延, 是否主要来自 codec 微算子。

本轮与之前综合报告的区别:

1. 样本量从小规模重测扩大到 6 数据集 × 100 test requests。
2. 明确拆分 prefill 与 decode 两阶段。
3. 明确拆分 delta 路径与 unigram 路径。
4. 仅保留可行候选, 不再投入 sparse / top-channel / structured 路线。

---

## 2. 实验实现与产物

新增脚本:

- [delta_coding_system/experiments/compression_experiment_phasewise_large.py](delta_coding_system/experiments/compression_experiment_phasewise_large.py)

结果目录:

- [results_phasewise_large_run](results_phasewise_large_run)
- [results_phasewise_large_merged](results_phasewise_large_merged)

合并后产物:

- [results_phasewise_large_merged/phasewise_request_summary.parquet](results_phasewise_large_merged/phasewise_request_summary.parquet)
- [results_phasewise_large_merged/phasewise_drift.parquet](results_phasewise_large_merged/phasewise_drift.parquet)
- [results_phasewise_large_merged/phasewise_position_detail.parquet](results_phasewise_large_merged/phasewise_position_detail.parquet)
- [results_phasewise_large_merged/phasewise_summary.csv](results_phasewise_large_merged/phasewise_summary.csv)

总记录数:

1. request 级记录: 18000
2. drift 级记录: 18000
3. position 级记录: 83445

说明:

1. `baseline_current` 在本报告中是路径内基线, 不是单一固定编码格式。
2. 对 delta 位置, `baseline_current` 表示当前 Int4 + affine + top1 outlier 路线。
3. 对 unigram 位置, `baseline_current` 表示当前 Int8 unigram 基线。

---

## 3. 样本规模与阶段分布

平均 prompt 长度和 decode 长度如下:

| phase | 平均 prompt 长度 | 平均 decode 长度 | 平均阶段位置数 |
|---|---:|---:|---:|
| prefill | 240.53 | 64.00 | 240.53 |
| decode | 240.53 | 64.00 | 64.00 |

分数据集看, prompt 长度跨度较大:

| 数据集 | 平均 prompt 长度 |
|---|---:|
| alpaca | 74.62 |
| wikitext2 | 114.92 |
| gsm8k | 175.52 |
| triviaqa | 336.73 |
| sharegpt | 361.69 |
| cnn_dm | 379.69 |

这意味着本轮结果不是只在短 prompt 上成立, 也覆盖了长 prompt prefill 场景。

---

## 4. 时延主结论

先给结论:

**当前这套系统里, codec 微算子并不是主要时延瓶颈。**

聚合平均后:

1. prefill 原始模型前向平均: 111.79 ms / request
2. decode 原始模型前向平均: 51.70 ms / token
3. 候选 codec 平均开销:
   - prefill: 0.65 ms 到 1.56 ms / position
   - decode: 0.69 ms 到 1.54 ms / position

对应占比:

1. prefill codec 占比: 0.33% 到 1.72%
2. decode codec 占比: 0.49% 到 2.98%

这说明:

1. 去 affine 或改量化, 确实能进一步抠出 codec 微开销。
2. 但如果用户看到的是更大级别的整体时延, 根因更可能在模型前向、缓存访存、跨阶段调度、或者 pipeline 级数据流, 而不是单个 codec 算子本身。

---

## 5. Delta 路径结果

### 5.1 Prefill 阶段

| 策略 | 压缩率 | codec ms | top1 match | logit cos | KL mean |
|---|---:|---:|---:|---:|---:|
| baseline_current | 2.3004x | 0.7364 | 0.996753 | 0.999927 | 0.000127 |
| delta_noaffine_int4_k1 | 2.3012x | 0.6528 | 0.995288 | 0.999864 | 0.000554 |
| delta_int2_k8_out4 | 2.4736x | 0.9102 | 0.991578 | 0.999555 | 0.000699 |
| delta_int2_k8_out8_entropy | 2.4605x | 0.8730 | 0.991875 | 0.999554 | 0.000700 |
| delta_noaffine_int2_k8_out4 | 2.4747x | 0.8268 | 0.991015 | 0.999446 | 0.001816 |

结论:

1. `delta_noaffine_int4_k1` 是本轮 delta 路径最稳的低时延升级。
2. 相比当前 baseline, 它几乎不改变压缩率, 但 codec 时间从 0.7364 ms 降到 0.6528 ms, 降幅约 11.4%。
3. 质量损失很小, prefill logit cosine 仍有 0.999864。
4. 若追求更高压缩率, `delta_int2_k8_out4` 可以把压缩率提高到 2.4736x, 但会带来更明显的输出漂移。
5. `delta_int2_k8_out8_entropy` 的质量与 `delta_int2_k8_out4` 几乎相同, 但由于这里只是估计熵上界而不是接入真实熵编码器, 当前实际 packet 压缩率并没有反超。

### 5.2 Decode 阶段

| 策略 | 压缩率 | codec ms | top1 match | logit cos | KL mean |
|---|---:|---:|---:|---:|---:|
| baseline_current | 2.4362x | 0.7920 | 0.997917 | 0.999922 | 0.000041 |
| delta_noaffine_int4_k1 | 2.4374x | 0.6907 | 0.997917 | 0.999892 | 0.000128 |
| delta_int2_k8_out8_entropy | 2.6851x | 0.9613 | 0.995807 | 0.999725 | 0.000270 |
| delta_int2_k8_out4 | 2.7131x | 1.0159 | 0.995703 | 0.999724 | 0.000274 |
| delta_noaffine_int2_k8_out4 | 2.7148x | 0.9035 | 0.995964 | 0.999622 | 0.000518 |

结论:

1. decode 阶段也延续同样结论: `delta_noaffine_int4_k1` 是 delta 路径默认首选。
2. 它把 codec 时间从 0.7920 ms 降到 0.6907 ms, 降幅约 12.8%。
3. decode top1 match 与 baseline 持平, 都是 0.997917。
4. 如果目标是把 delta 路径压缩率再推高, `delta_int2_k8_out4` 与 `delta_noaffine_int2_k8_out4` 都可行, 但它们不再是默认安全配置。

### 5.3 Delta 路径最终建议

默认配置:

1. `delta_noaffine_int4_k1`

压缩优先备选:

1. `delta_int2_k8_out4`
2. `delta_noaffine_int2_k8_out4`

仅当后续接入真实熵编码器时再保留观察的路线:

1. `delta_int2_k8_out8_entropy`

---

## 6. Unigram 路径结果

### 6.1 Prefill 阶段

| 策略 | 压缩率 | codec ms | top1 match | logit cos | KL mean |
|---|---:|---:|---:|---:|---:|
| unigram_int4_k4 | 3.3292x | 0.7909 | 0.991184 | 0.999682 | 0.000686 |
| prev2_blend_int4_k2 | 3.4439x | 1.1060 | 0.988559 | 0.999426 | 0.002635 |
| prev_int4_k2 | 3.4599x | 1.2596 | 0.988094 | 0.999344 | 0.003178 |
| zero_affine_int4_k2 | 3.5017x | 1.2610 | 0.986393 | 0.999260 | 0.001755 |
| prev_gs256_k2 | 3.6144x | 1.2500 | 0.984396 | 0.998456 | 0.010334 |
| prev_int2_k8_out4 | 4.2743x | 1.5599 | 0.967035 | 0.996057 | 0.016344 |

结论:

1. 在更大样本的 prefill 阶段, `unigram_int4_k4` 明显成为最平衡方案。
2. 它不仅最快, 而且质量反而优于 `prev_int4_k2` 和 `prev2_blend_int4_k2`。
3. `prev_int4_k2` 的优势主要只剩下一点点额外压缩率, 但要付出更高 codec 时间和更大的 prefill 漂移。
4. `prev_gs256_k2` 在大样本下仍然属于偏激进路线, 不适合作为默认。
5. `prev_int2_k8_out4` 只能视为极限压缩路线, 不适合默认上线。

### 6.2 Decode 阶段

| 策略 | 压缩率 | codec ms | top1 match | logit cos | KL mean |
|---|---:|---:|---:|---:|---:|
| prev_int4_k2 | 3.5124x | 1.2622 | 0.995703 | 0.999784 | 0.000273 |
| unigram_int4_k4 | 3.3601x | 0.8427 | 0.996068 | 0.999783 | 0.000274 |
| prev2_blend_int4_k2 | 3.5150x | 1.1287 | 0.995573 | 0.999773 | 0.000282 |
| zero_affine_int4_k2 | 3.5124x | 1.2595 | 0.994844 | 0.999762 | 0.000314 |
| prev_gs256_k2 | 3.6547x | 1.2555 | 0.993880 | 0.999627 | 0.000507 |
| prev_int2_k8_out4 | 4.2604x | 1.5409 | 0.987578 | 0.998770 | 0.002687 |

结论:

1. decode 阶段里, `prev_int4_k2` 与 `unigram_int4_k4` 已经几乎打平。
2. `prev_int4_k2` 的压缩率略高, 但 `unigram_int4_k4` 的 top1 match 反而略高, logit cosine 几乎相同。
3. 更重要的是 `unigram_int4_k4` 的 codec 时间只有 0.8427 ms, 比 `prev_int4_k2` 的 1.2622 ms 低约 33.2%。
4. 这意味着在 decode 阶段, `prev_int4_k2` 已经失去了此前那种明显的质量优势, 但仍保留了更高的 codec 成本。

### 6.3 Unigram 路径最终建议

新的默认推荐:

1. `unigram_int4_k4`

原因:

1. prefill 阶段它是明确最平衡的方案。
2. decode 阶段它与 `prev_int4_k2` 基本打平, 但更快。
3. 当本轮实验目标包含时延优化时, 它比 `prev_int4_k2` 更合理。

如果更偏向压缩率, 而能接受少量额外漂移与更高 codec 成本:

1. `prev_int4_k2`
2. `prev2_blend_int4_k2`

如果只追求更高压缩率做探索, 但不建议默认上线:

1. `prev_gs256_k2`
2. `prev_int2_k8_out4`

---

## 7. 稳健性观察

按数据集最差值看:

### 7.1 Delta 路径

1. `delta_noaffine_int4_k1`
   - prefill 最差 top1 match: 0.984725
   - decode 最差 top1 match: 0.996562
   - 仍然处于非常稳的范围
2. `delta_int2_k8_out4`
   - prefill 最差 top1 match: 0.978224
   - decode 最差 top1 match: 0.993281
   - 可以作为压缩优先备选, 但不应默认替代 Int4
3. `delta_noaffine_int2_k8_out4`
   - 比 `delta_int2_k8_out4` 更快, 但 prefill 最差 logit cosine 下探更明显

### 7.2 Unigram 路径

1. `unigram_int4_k4`
   - prefill 最差 top1 match: 0.979714
   - decode 最差 top1 match: 0.992031
   - 大样本下表现稳定
2. `prev_int4_k2`
   - prefill 最差 top1 match: 0.964168
   - decode 最差 top1 match: 0.990625
   - 问题主要集中在 prefill, 这也是本轮推荐变更的核心原因
3. `prev_gs256_k2`
   - prefill 最差 top1 match: 0.950601
   - 大样本下波动明显, 不适合作为默认

---

## 8. 本轮结论汇总

### 8.1 默认路线更新

如果只保留一套新的工程默认配置:

1. delta 路径默认: `delta_noaffine_int4_k1`
2. unigram 路径默认: `unigram_int4_k4`

这套组合同时满足:

1. 比当前方案更低的 codec 时延。
2. 维持足够高的 prefill / decode 输出稳定性。
3. 不引入稀疏编码带来的系统复杂度。

### 8.2 关于“高时延”的真正判断

本轮数据不支持“高时延主要由 codec 导致”这个判断。

更准确的说法应该是:

1. codec 还有可优化空间, 但量级只是在 0.1 ms 到 0.7 ms 的微调范围。
2. 真正决定端到端 latency 的主项仍然是模型前向与 pipeline 级调度。
3. 如果下一步继续做 latency 工程, 优先级应转向:
   - prefill / decode 的执行编排
   - KV / hidden 的传输与复用
   - pipeline 中的冗余同步
   - 真正的熵编码实现是否值得引入

### 8.3 与此前综合报告的关系

此前综合报告给出的 unigram 默认建议偏向 `prev_int4_k2`。本轮大样本、phasewise、显式时延约束下, 结论更新为:

1. 如果把 prefill 和 decode 一起看, 且把时延优化纳入目标函数, `unigram_int4_k4` 更适合作为新的默认。
2. `prev_int4_k2` 依然是强备选, 但不再是默认首选。

---

## 9. 最终工程建议

建议直接进入下一轮工程实现的两项改动:

1. 将 delta 路径默认切换为 `delta_noaffine_int4_k1`。
2. 将 unigram 路径默认切换为 `unigram_int4_k4`。

保留为可配置备选:

1. delta: `delta_int2_k8_out4`
2. unigram: `prev_int4_k2`
3. unigram: `prev2_blend_int4_k2`

暂不建议作为默认主路:

1. 所有 sparse 相关路线
2. `prev_gs256_k2`
3. `prev_int2_k8_out4`
4. raw-ref / affine-only 组件消融路线
