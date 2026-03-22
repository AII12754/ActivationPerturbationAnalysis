# Delta-Coding 激活压缩系统综合总报告

> 报告日期: 2026-03-21  
> 模型: Qwen2.5-32B-Instruct  
> 评估数据集: WikiText-2, ShareGPT, GSM8k, CNN/DM, Alpaca, TriviaQA  
> 本报告整合三类结果:  
> 1. 全量策略 / 组件消融  
> 2. delta 路径量化 / outlier / 熵编码专题  
> 3. 新增的综合 drift / latency / decode 策略评估

> 说明: 本报告中的综合实验结果来自修复后的最终重跑。修复点是 unigram direct-Int4 辅助策略曾经会原地修改输入 hidden，已修复并完整重跑，因此本报告不使用那版受污染结果。

---

## 1. 这份报告覆盖什么

这份报告回答的不只是“哪个方案压缩率高”，而是把整个策略空间拆成四个层面一起评估:

1. 压缩率到底能到哪里
2. hidden 重建指标是否真的能转化成输出稳定性
3. 哪些 decode 阶段策略能降低 codec 时延，而不是只会增加复杂度
4. delta 路径、unigram 路径、参考设计、outlier、熵编码、去 affine、结构化稀疏，各自到底贡献了什么

最终目标不是单点最优，而是给出一套工程上可落地的策略分层建议。

---

## 2. 结果文件

历史结果:

- [delta_coding_system/experiments/reports/COMPRESSION_FULL_ABLATION_REPORT.md](delta_coding_system/experiments/reports/COMPRESSION_FULL_ABLATION_REPORT.md)
- [delta_coding_system/experiments/reports/COMPRESSION_DELTA_QUANT_ENTROPY_REPORT.md](delta_coding_system/experiments/reports/COMPRESSION_DELTA_QUANT_ENTROPY_REPORT.md)
- [delta_coding_system/experiments/reports/COMPRESSION_STRATEGY_DRIFT_REPORT.md](delta_coding_system/experiments/reports/COMPRESSION_STRATEGY_DRIFT_REPORT.md)

本轮新增综合实验脚本与结果:

- [delta_coding_system/experiments/compression_experiment_comprehensive.py](delta_coding_system/experiments/compression_experiment_comprehensive.py)
- [results_comprehensive_compression/comprehensive_request_summary.parquet](results_comprehensive_compression/comprehensive_request_summary.parquet)
- [results_comprehensive_compression/comprehensive_drift.parquet](results_comprehensive_compression/comprehensive_drift.parquet)
- [results_comprehensive_compression/comprehensive_position_detail.parquet](results_comprehensive_compression/comprehensive_position_detail.parquet)
- [results_comprehensive_compression.log](results_comprehensive_compression.log)

用于对照的历史结果文件:

- [results_compression_full_ablation/strategies_full.parquet](results_compression_full_ablation/strategies_full.parquet)
- [results_delta_quant_entropy/delta_quant_entropy.parquet](results_delta_quant_entropy/delta_quant_entropy.parquet)
- [results_test_baseline/pipeline_quality_part_000000.parquet](results_test_baseline/pipeline_quality_part_000000.parquet)
- [results_test_baseline/latency_part_000000.parquet](results_test_baseline/latency_part_000000.parquet)
- [results_test_lru/pipeline_quality_part_000000.parquet](results_test_lru/pipeline_quality_part_000000.parquet)
- [results_test_lru/latency_part_000000.parquet](results_test_lru/latency_part_000000.parquet)
- [results_test_fp8/pipeline_quality_part_000000.parquet](results_test_fp8/pipeline_quality_part_000000.parquet)
- [results_test_fp8/latency_part_000000.parquet](results_test_fp8/latency_part_000000.parquet)

---

## 3. 评估设计

### 3.1 历史全量消融

全量消融负责回答:

1. 每个策略在统一口径下的压缩率与 hidden 重建效果
2. raw reference / affine / residual coding 三段各自解释了多少能量
3. 稀疏化、Int2 / Int4、prev-token、mean-pool、EMA、blend、结构化稀疏等组件的边际收益

### 3.2 delta 量化专题

delta 专题负责回答:

1. Int2 / Int4 / Int8 的真实 trade-off
2. outlier top-k 的边际收益
3. 熵编码在 delta 包上到底能带来多少额外收益

### 3.3 新综合实验

新综合实验是这次新增的核心。配置如下:

- 每数据集 20 个 warmup request
- 每数据集 20 个 test request
- 每数据集前 3 个 test request 做真实下游 drift 分析
- 每数据集前 2 个 test request 记录 position-level 细节
- max_seq_len = 384

新增指标分三类:

1. request 级压缩与重建:
   - compression_ratio
   - compression_ratio_zlib_packet
   - compression_ratio_ideal_entropy
   - cosine_mean
   - recon_energy_explained_mean

2. 输出 drift:
   - top1_match_rate
   - first_top1_drift_pos
   - first_logit_cos_below_0_999
   - logit_cosine_mean
   - kl_mean

3. codec 微时延:
   - codec_total_ms
   - pack_ms
   - affine_param_ms
   - affine_apply_ms
   - delta_ms
   - decode_ms
   - entropy_ms

这里的 codec 时延是“每个位置的编码 / 解码微时延均值”，不是整条请求的端到端时延。

---

## 4. 策略空间

### 4.1 delta 路径

本轮重点覆盖了以下 delta 方案:

1. 当前基线: baseline_current
2. 去 affine Int4: delta_noaffine_int4_k1
3. Int2 + out8: delta_int2_k8_out8
4. Int2 + out4: delta_int2_k8_out4
5. 去 affine 的 Int2 + out4: delta_noaffine_int2_k8_out4
6. Int2 + out8 + 熵编码估计: delta_int2_k8_out8_entropy
7. 稀疏组阈值: delta_sparse_thr10
8. Top-1024 通道显式发送: delta_top1024_ch
9. 2-of-8 结构化稀疏: delta_struct2of8
10. delta raw-ref / affine-only 消融

### 4.2 unigram 路径

本轮重点覆盖了以下 unigram 方案:

1. 当前基线: baseline_current
2. 直接 Int4: unigram_int4_k4
3. prev-token Int4: prev_int4_k2
4. prev-token grouped-scale-256: prev_gs256_k2
5. prev-token Int2 + out4: prev_int2_k8_out4
6. zero-reference affine: zero_affine_int4_k2
7. global mean affine: global_mean_int4_k2
8. mean-pool reference: mean_pool_int4_k2
9. EMA reference: ema_prev4_int4_k2
10. prev2 blend: prev2_blend_int4_k2
11. prev raw-ref / affine-only 消融

---

## 5. 全局总结

### 5.1 综合排名

下表是本轮综合实验中最重要的总体结果。

| 策略 | 压缩率 | zlib 后压缩率 | 平均 codec ms | Top-1 一致率 | 首次 Top-1 漂移位置 | 首次 logit cos < 0.999 | 平均 logit cosine | 平均 KL |
|------|-------:|--------------:|--------------:|-------------:|--------------------:|-----------------------:|------------------:|--------:|
| delta_noaffine_int4_k1 | 2.181x | 2.208x | 0.774 | 0.998401 | 197.83 | 155.22 | 0.999895 | 0.000120 |
| baseline_current | 2.180x | 2.208x | 0.837 | 0.998393 | 206.06 | 170.89 | 0.999944 | 0.000117 |
| delta_sparse_thr10 | 2.182x | 2.211x | 0.905 | 0.997945 | 204.22 | 170.89 | 0.999943 | 0.000117 |
| delta_noaffine_int2_k8_out4 | 2.292x | 2.310x | 0.901 | 0.996954 | 162.22 | 64.22 | 0.999461 | 0.000535 |
| delta_int2_k8_out4 | 2.291x | 2.310x | 0.961 | 0.995607 | 117.39 | 71.89 | 0.999646 | 0.000575 |
| delta_int2_k8_out8_entropy | 2.285x | 2.285x | 0.933 | 0.995455 | 123.56 | 70.61 | 0.999657 | 0.000544 |
| unigram_int4_k4 | 3.295x | 3.459x | 0.830 | 0.994609 | 102.11 | 61.83 | 0.999672 | 0.000712 |
| prev2_blend_int4_k2 | 3.429x | 3.647x | 1.282 | 0.993460 | 94.06 | 51.50 | 0.999611 | 0.000718 |
| prev_int4_k2 | 3.451x | 3.673x | 1.463 | 0.992865 | 97.83 | 43.44 | 0.999548 | 0.000833 |
| prev_gs256_k2 | 3.628x | 4.067x | 1.446 | 0.989909 | 92.06 | 39.56 | 0.999277 | 0.001456 |
| zero_affine_int4_k2 | 3.489x | 3.733x | 1.283 | 0.989459 | 105.06 | 0.00 | 0.999003 | 0.002398 |
| global_mean_int4_k2 | 3.489x | 3.730x | 1.281 | 0.980889 | 39.44 | 0.00 | 0.998162 | 0.004676 |
| prev_int2_k8_out4 | 4.405x | 4.647x | 1.787 | 0.975714 | 71.83 | 4.33 | 0.996960 | 0.007032 |

先给综合结论:

1. 默认线上策略仍然应当以 prev_int4_k2 为核心参考点
2. 如果只想优化 delta 路径 decode codec 时延，最值得做的是去 affine 的 Int4 delta
3. 如果希望更高压缩率但仍维持较稳输出，最值得考虑的是 prev_gs256_k2 和 unigram_int4_k4
4. 如果追求极限压缩率，可以上 prev_int2_k8_out4，但它既更慢，也更早开始 drift
5. global mean 参考不成立，zero-affine 可作为无局部参考备选，但不是主路线

---

## 6. Delta 路径结论

### 6.1 decode 阶段去 affine 是这次最值得落地的 delta 改动

baseline_current 对比 delta_noaffine_int4_k1:

- 压缩率: 2.1800x → 2.1805x
- 平均 codec 时延: 0.837ms → 0.774ms
- Top-1 一致率: 0.998393 → 0.998401
- 平均 logit cosine: 0.999944 → 0.999895

真正关键的是 tier 级指标:

- trigram 平均包长: 2852B → 2848B
- trigram codec: 1.261ms → 1.027ms
- bigram 平均包长: 2852B → 2848B
- bigram codec: 1.275ms → 1.021ms

这说明:

1. 去 affine 几乎不影响包长
2. 但能在 delta 位置上把 codec 微时延降低大约 19% 到 20%
3. 输出稳定性几乎不受影响

结论:

- 如果你要优先优化 decode 阶段时延，这是当前最明确、风险最低、收益最稳的一条改动

### 6.2 Int2 delta 是可行的，但应该区分“追求更高 CR”和“追求更低 codec latency”

对比三条核心 Int2 路径:

| 策略 | 压缩率 | codec ms | Top-1 一致率 | 首次 Top-1 漂移位置 | 平均 logit cosine | 平均 KL |
|------|-------:|---------:|-------------:|--------------------:|------------------:|--------:|
| delta_int2_k8_out8 | 2.266x | 0.936 | 0.995455 | 123.56 | 0.999657 | 0.000544 |
| delta_int2_k8_out4 | 2.291x | 0.961 | 0.995607 | 117.39 | 0.999646 | 0.000575 |
| delta_noaffine_int2_k8_out4 | 2.292x | 0.901 | 0.996954 | 162.22 | 0.999461 | 0.000535 |

这里有三个重要点:

1. out4 相比 out8 再多拿到约 0.025 的压缩率
2. 去 affine 的 Int2 + out4 又多拿到约 0.025 的压缩率，同时 codec 还更快
3. 在这轮采样下，delta_noaffine_int2_k8_out4 的输出稳定性没有变差，反而略优于带 affine 的 Int2 版本

这类结果需要谨慎解释，因为增益不大，可能含有采样波动；但至少说明:

- Int2 delta + 去 affine 是值得继续深入验证的方向

### 6.3 熵编码对 Int2 delta 依然有效，但它只是增量优化

delta_int2_k8_out8_entropy 相比 delta_int2_k8_out8:

- 平均压缩率: 2.266x → 2.285x
- zlib 口径下二者相同，说明它本质是在更接近熵上界的字节估计里省出最后一段空间
- 输出指标完全不变

这和 delta 专题实验一致:

- delta_int2_k8: 4.245x
- delta_int2_k8 的 zlib 口径: 4.534x
- delta_int2_k8 的理想熵编码口径: 4.780x

结论:

- Int2 包上做熵编码是对的
- 但它不是决定性跃迁，更像最后 1% 到 5% 的包长优化

### 6.4 稀疏阈值 / Top-1024 / 2-of-8 不值得作为主策略

几个代表性结论:

1. delta_sparse_thr10 的压缩率只比 baseline 高 0.002x，codec 反而更慢
2. delta_top1024_ch 和 delta_struct2of8 的 codec 更快，分别约 0.686ms，但压缩率跌到 2.03x / 2.06x，且 drift 明显更大
3. 它们更像研究型对照组，不是当前最优工程解

### 6.5 delta 消融链路

结合历史 delta 专题和本轮 drift，可以把 delta 路径理解成以下链条:

1. raw reference:
   - delta_ref_only 的重建能量约 0.598
   - 本轮真实输出 Top-1 一致率约 0.956

2. affine:
   - delta_affine_only 的重建能量约 0.830
   - 本轮真实输出 Top-1 一致率约 0.959

3. affine + Int4 / Int2 residual:
   - baseline_current 可到 0.9995 的重建能量和 0.9984 的 Top-1 一致率
   - Int2 版本则在保留高稳定性的同时进一步压缩

这说明:

- delta 路径的主要价值仍然来自 reference + affine
- residual coding 是把剩余误差再补回来，而不是从零开始重建

---

## 7. Unigram 路径结论

### 7.1 prev_int4_k2 仍然是默认策略基线

prev_int4_k2 的总体表现:

- 压缩率 3.451x
- zlib 后 3.673x
- codec 1.463ms
- Top-1 一致率 0.992865
- 首次 Top-1 漂移位置 97.83
- 平均 KL 0.000833

它仍然是最稳、最均衡、最容易解释的默认选择。

### 7.2 prev_gs256_k2 是最强的“轻度激进”版本

相对 prev_int4_k2:

- 压缩率 +0.177x
- zlib 后压缩率 +0.394x
- codec 时延 -0.018ms
- Top-1 一致率 -0.00296
- 首次 Top-1 漂移位置 -5.78

这说明:

- grouped-scale 256 让包结构更可压缩
- 它是一个很干净的 trade-off: 更高 CR，略差稳定性，codec 还更快一点

结论:

- 如果你要比 prev_int4_k2 更激进一点，但不想直接跳到 Int2，首推 prev_gs256_k2

### 7.3 prev_int2_k8_out4 是高压缩上限，但不是默认策略

相对 prev_int4_k2:

- 压缩率 +0.954x
- codec 时延 +0.324ms
- Top-1 一致率 -0.01715
- 首次 Top-1 漂移位置提前 26 个 token
- 平均 KL 增加约 0.0062

结论:

- 它是当前最强高压缩 unigram 路线
- 但它既更慢，也更早 drift
- 只适合明确接受质量下降的激进模式

### 7.4 unigram_int4_k4 是一个被低估的强基线

相对 prev_int4_k2:

- 压缩率低 0.156x
- codec 时延快 0.633ms
- Top-1 一致率反而高 0.00174
- 首次 Top-1 漂移位置更晚 4.28 个 token
- KL 更低

这说明:

- 如果你的第一目标是简化 codec、压低 decode 开销，而不是把压缩率推到极限，直接 Int4 unigram 其实非常强
- 它没有 prev-token 的参考依赖，实现更简单，风险也更低

结论:

- unigram_int4_k4 应当进入正式候选集，而不只是充当对照组

### 7.5 zero-affine 可以保留，global-mean 不建议

zero_affine_int4_k2 相对 prev_int4_k2:

- 压缩率 +0.039x
- codec 时延 -0.180ms
- Top-1 一致率 -0.00341
- 首次 Top-1 漂移位置更晚 7.22 个 token
- 但 first_logit_cos_below_0.999 从 43.44 变成 0，说明从第一个位置开始就有可检测偏差

global_mean_int4_k2 相对 prev_int4_k2:

- 压缩率也只多 0.039x
- codec 也更快
- 但 Top-1 一致率下降更多
- 漂移明显更早

结论:

1. zero-affine 是可留存的无局部参考后备方案
2. global-mean 不是好参考，不建议继续投入

### 7.6 mean-pool / EMA / prev2-blend

这三个策略里最值得注意的是 prev2_blend_int4_k2:

- 压缩率只比 prev_int4_k2 低 0.021x
- codec 时延快 0.182ms
- Top-1 一致率略高 0.0006
- 平均 KL 还略低

但它的首次 Top-1 漂移位置略早 3.78 个 token，而且实现复杂度更高。

结论:

- prev2_blend_int4_k2 是一个很值得保留的“低时延替代方案”
- mean_pool 和 EMA 没有明显超过 prev_int4_k2 的综合收益

---

## 8. 时延分析

### 8.1 现有 pipeline 端到端时延基线

已有 pipeline 结果显示:

| 配置 | 平均压缩率 | 平均余弦 | unigram 占比 | prefill ms | encode_delta ms | encode_unigram ms | total ms |
|------|-----------:|---------:|-------------:|-----------:|----------------:|------------------:|---------:|
| baseline | 2.146x | 0.999813 | 75.845% | 529.9 | 20.76 | 3.56 | 616.66 |
| lru | 2.146x | 0.999813 | 75.845% | 1390.6 | 8.26 | 7.37 | 1431.09 |
| fp8 | 2.146x | 0.999812 | 75.845% | 1377.4 | 8.75 | 8.25 | 1435.49 |

这里最重要的不是 baseline / lru / fp8 谁快，而是两个结构性结论:

1. unigram 位置仍占约 75.8%，它决定了大部分字节与相当一部分编码开销
2. prefill / forward 仍然是端到端大头，单纯压 codec 毫秒数并不会线性转成端到端收益

这也是为什么本报告把 codec 微时延和端到端时延分开讨论。

### 8.2 codec 微时延结论

从综合实验看，几条最值得关注的时延结论是:

1. delta_noaffine_int4_k1:
   - 在几乎不损失质量的情况下，把 delta codec 从 0.837ms 压到 0.774ms

2. unigram_int4_k4:
   - codec 只有 0.830ms，比 prev_int4_k2 快 0.633ms
   - 是最强的简单低时延 unigram 候选

3. prev2_blend_int4_k2:
   - codec 1.282ms，比 prev_int4_k2 快 0.182ms
   - 压缩率只少一点点

4. prev_int2_k8_out4:
   - codec 达到 1.787ms，是本轮主要策略里最慢的
   - 所以它不是“更高压缩率但同等代价”，而是“更高压缩率且更高 codec 代价”

---

## 9. 最终建议

### 9.1 默认线上策略

建议:

1. delta 路径切到 delta_noaffine_int4_k1
2. unigram 路径继续用 prev_int4_k2

原因:

1. delta 去 affine 有稳定且几乎免费的时延收益
2. prev_int4_k2 仍然是 unigram 路径最稳的默认点

### 9.2 轻度激进模式

建议:

1. unigram 路径尝试 prev_gs256_k2

适用场景:

1. 想把压缩率继续推高一点
2. 可以接受非常轻微的输出漂移增加

### 9.3 低时延模式

建议候选顺序:

1. unigram_int4_k4
2. prev2_blend_int4_k2
3. zero_affine_int4_k2

原因:

1. 这三条路线都比 prev_int4_k2 的 codec 更快
2. 其中 unigram_int4_k4 的综合表现尤其好，简单且稳

### 9.4 极限压缩模式

建议:

1. prev_int2_k8_out4 作为单独开关模式保留

不建议:

1. 把它直接升格成默认策略

原因:

1. 它的 drift 更早
2. codec 也更慢

### 9.5 不建议继续投入的路线

1. global_mean_int4_k2
2. delta_sparse_thr10 作为主策略
3. delta_top1024_ch
4. delta_struct2of8

这些方案都存在“额外复杂度大于边际收益”的问题。

---

## 10. 一句话结论

如果把压缩率、真实输出漂移、decode codec 时延、组件消融一起看，最合理的工程路线是:

1. delta 路径先做去 affine
2. unigram 默认继续用 prev_int4_k2
3. 更激进时用 prev_gs256_k2 或 prev_int2_k8_out4
4. 更低时延时优先评估 unigram_int4_k4 和 prev2_blend_int4_k2

这比单纯追求更高压缩率更稳，也更符合 decode 阶段的真实代价结构。
