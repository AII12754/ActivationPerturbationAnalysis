# Real Pipeline 策略接入评测报告

> 实验日期: 2026-03-22  
> 模型: Qwen2.5-32B-Instruct  
> 结构: layer_boundary=6, hidden_dim=5120  
> 数据集: WikiText-2, ShareGPT, GSM8k, CNN/DM, Alpaca, TriviaQA  
> 规模: 每数据集 30 warmup + 100 test requests  
> Decode: 动态生成, 最多 512 tokens, KV Cache 开启  
> 设备: 6 卡并行, 使用 GPU 1-6, 不使用 GPU 0  

## 1. 实验目标

本轮实验不是 phasewise analytical 估算,而是将候选策略直接接入生产 OverlappedPipeline,在真实 overlap 执行路径下评估四类指标:

1. 总压缩率
2. 通信端到端时延
3. Decode 漂移
4. Decode 相似度

本轮直接比较两套真实系统配置:

| 配置 | Delta 路径 | Unigram 路径 |
|------|-----------|-------------|
| baseline_current | baseline_current | baseline_current |
| optimized_default | delta_noaffine_int4_k1 | unigram_int4_k4 |

## 2. 结果口径说明

报告中的 latency 指标使用真实 pipeline 产物中的 request-level 统计字段:

1. pipeline_total_ms: 真实 OverlappedPipeline 的 prefill + decode 运行时间
2. prefill_comm_e2e_Xmbps_ms: prefill 本地编码处理 + 模拟网络传输
3. decode_comm_e2e_Xmbps_per_token_ms: decode 每 token 的本地编码处理 + 模拟网络传输

本节额外补充一个不需要重跑实验的原生 FP16 通信基线:

1. 使用 parquet 中已记录的 prefill_raw_fp16_bytes 和 decode_raw_fp16_bytes
2. 假设发送端直接传输原始 FP16 激活,不做任何编码
3. 因此 FP16 baseline 的通信口径是纯网络传输时延,不包含额外 codec 本地处理开销

需要单独说明:

1. 这轮实验脚本为了拿到精确 decode drift,在每条 request 后额外做了一次原始 full forward 和一次从第 6 层开始的 replay
2. 因此实验任务的实际墙钟耗时明显高于 pipeline_total_ms
3. 但报告中的真实性能结论应以 parquet 中的 pipeline_total_ms 和 comm_e2e 字段为准,而不是以实验脚本总运行时长为准

## 3. 总体结果

### 3.1 平均请求级指标

| 指标 | baseline_current | optimized_default | 相对变化 |
|------|------------------|-------------------|---------|
| prompt_len | 240.53 | 240.53 | 0.00% |
| decode_len | 511.09 | 511.09 | 0.00% |
| prefill_compression_ratio | 2.413 | 3.363 | +39.35% |
| decode_compression_ratio | 2.961 | 3.486 | +17.71% |
| total_compression_ratio | 2.776 | 3.449 | +24.27% |
| prefill_recon_cosine_mean | 0.999652 | 0.997884 | -0.001768 |
| decode_recon_cosine_mean | 0.999639 | 0.998988 | -0.000651 |
| decode_recon_cosine_min | 0.997070 | 0.987538 | -0.009532 |
| prefill_total_ms | 112.19 | 111.41 | -0.69% |
| decode_total_ms | 27406.51 | 27350.02 | -0.21% |
| pipeline_total_ms | 27518.70 | 27461.43 | -0.21% |

### 3.2 通信端到端时延

#### Prefill

| 指标 | FP16 baseline | baseline_current | optimized_default | optimized vs baseline | optimized vs FP16 |
|------|---------------|------------------|-------------------|-----------------------|------------------|
| prefill_comm_200mbps_ms | 98.520 | 43.344 | 31.131 | -28.18% | -68.40% |
| prefill_comm_500mbps_ms | 39.408 | 18.553 | 13.540 | -27.02% | -65.64% |
| prefill_comm_1000mbps_ms | 19.704 | 10.290 | 7.676 | -25.40% | -61.04% |

#### Decode

| 指标 | FP16 baseline | baseline_current | optimized_default | optimized vs baseline | optimized vs FP16 |
|------|---------------|------------------|-------------------|-----------------------|------------------|
| decode_comm_200mbps_per_token_ms | 0.4096 | 0.3459 | 0.2642 | -23.62% | -35.50% |
| decode_comm_500mbps_per_token_ms | 0.1638 | 0.2616 | 0.1937 | -25.99% | +18.23% |
| decode_comm_1000mbps_per_token_ms | 0.0819 | 0.2336 | 0.1701 | -27.15% | +107.70% |

这里需要正确解读:

1. 在 prefill 阶段,压缩后即使加上本地编码处理,总体仍显著优于直接传 FP16
2. 在 decode 阶段,200 Mbps 时 optimized_default 仍优于 FP16 baseline
3. 但在 500/1000 Mbps 下,由于 decode 端本地 classify + encode 开销开始主导,压缩路径的通信端到端时延反而高于直接传 FP16
4. 这说明压缩策略在低带宽链路下收益最明显,而在高带宽链路下需要继续优化 decode 端本地处理开销,否则纯通信口径未必继续占优

### 3.3 Decode 漂移

| 指标 | baseline_current | optimized_default | 差值 |
|------|------------------|-------------------|------|
| top1_match_rate | 0.997735 | 0.995931 | -0.001804 |
| first_top1_drift_pos | 316.73 | 215.92 | -100.81 |
| first_logit_cos_below_0_999 | 258.89 | 152.68 | -106.21 |
| logit_cosine_mean | 0.998987 | 0.998653 | -0.000334 |
| logit_cosine_min | 0.949080 | 0.938272 | -0.010808 |
| kl_mean | 0.000138 | 0.000377 | +0.000239 |
| kl_max | 0.014221 | 0.033865 | +0.019644 |

## 4. 分数据集结果

### 4.1 压缩率与时延收益

| 数据集 | baseline CR | optimized CR | 压缩率变化 | prefill e2e 200Mbps 变化 | decode e2e 200Mbps/token 变化 | pipeline_total 变化 |
|--------|-------------|--------------|-----------|--------------------------|-------------------------------|--------------------|
| WikiText-2 | 2.863 | 3.467 | +21.08% | -29.74% | -24.39% | -0.53% |
| ShareGPT | 2.602 | 3.413 | +31.16% | -28.77% | -23.09% | -0.03% |
| GSM8k | 3.229 | 3.540 | +9.64% | -18.15% | -24.32% | -0.17% |
| CNN/DM | 2.708 | 3.439 | +26.96% | -30.00% | -23.92% | -0.02% |
| Alpaca | 2.670 | 3.429 | +28.41% | -28.32% | -22.67% | -0.39% |
| TriviaQA | 2.580 | 3.408 | +32.07% | -29.42% | -23.36% | -0.13% |

观察:

1. optimized_default 在 6/6 数据集上压缩率全部提升
2. optimized_default 在 6/6 数据集上 prefill 和 decode 的通信端到端时延全部下降
3. optimized_default 在 6/6 数据集上真实 pipeline_total_ms 也全部更低,但降幅很小,说明单机真实运行主要仍由模型 forward 主导,而不是由编码或网络模拟主导

### 4.2 分数据集漂移表现

| 数据集 | baseline top1_match | optimized top1_match | 差值 | baseline logit_cos_mean | optimized logit_cos_mean | 差值 |
|--------|---------------------|----------------------|------|-------------------------|--------------------------|------|
| WikiText-2 | 0.995826 | 0.994043 | -0.001783 | 0.998001 | 0.996813 | -0.001188 |
| ShareGPT | 0.998242 | 0.996621 | -0.001621 | 0.999428 | 0.999217 | -0.000211 |
| GSM8k | 0.999277 | 0.998457 | -0.000820 | 0.999959 | 0.999909 | -0.000050 |
| CNN/DM | 0.998008 | 0.996699 | -0.001309 | 0.996992 | 0.996680 | -0.000312 |
| Alpaca | 0.998086 | 0.996172 | -0.001914 | 0.999840 | 0.999782 | -0.000058 |
| TriviaQA | 0.996973 | 0.993594 | -0.003379 | 0.999701 | 0.999517 | -0.000184 |

最需要注意的退化数据集:

1. TriviaQA: top1_match_rate 降幅最大,下降 0.003379
2. WikiText-2: decode hidden cosine 均值退化最大,decode_recon_cosine_mean 下降 0.001401
3. Alpaca: 首次 top1 漂移位置提前最多之一,平均提前约 135 个 token

## 5. 关键结论

### 5.1 是否值得接入真实系统

结论: 值得。

原因:

1. total_compression_ratio 从 2.776x 提升到 3.449x,总体提升 24.27%
2. 在 200 到 1000 Mbps 三档带宽下,prefill 与 decode 通信端到端时延都稳定下降约 23% 到 28%
3. 真实 pipeline_total_ms 没有变差,反而有轻微下降
4. 这种收益在 6 个数据集上全部一致,不是单一数据集偶然现象
5. 相比原生 FP16 直接传输,prefill 阶段在三档带宽下都明显占优
6. 相比原生 FP16 直接传输,decode 阶段仅在 200 Mbps 这类低带宽场景下明显占优

### 5.2 代价是什么

结论: 代价主要体现在 decode 漂移提前出现,但整体仍处在高相似度区间。

具体表现:

1. top1_match_rate 从 0.997735 降到 0.995931,绝对下降约 0.18 个百分点
2. first_top1_drift_pos 从 316.7 提前到 215.9
3. logit_cosine_mean 从 0.998987 降到 0.998653
4. KL 指标变大,说明尾部请求中会出现更明显的分布偏移

因此:

1. 如果目标是尽可能提高链路压缩率并降低网络敏感阶段的时延,optimized_default 是更优默认配置
2. 如果目标是最保守的长序列稳定性,baseline_current 仍然更稳,尤其在 TriviaQA 和 WikiText-2 上
3. 如果部署链路带宽较高,还需要继续优化 decode 端本地 classify/encode 开销,否则其端到端通信时延不一定优于原生 FP16

### 5.3 为什么真实 pipeline 总时延只小幅改善

因为在当前单机评测中:

1. 512 token decode 的主要成本仍然是后续层 forward
2. 编码路径和模拟网络传输虽然明显下降,但其在 pipeline_total_ms 中占比并不高
3. 所以 optimized_default 的系统级收益主要体现在通信受限场景,而不是单机本地 forward 场景

换句话说:

1. 如果部署环境是真正的跨 stage 网络传输,这轮优化是有实质价值的
2. 如果部署环境几乎没有通信瓶颈,那么它带来的 wall-clock 改善会明显小于压缩率改善

## 6. 建议

### 6.1 默认配置建议

建议将真实 pipeline 默认配置切换为:

1. Delta 路径: delta_noaffine_int4_k1
2. Unigram 路径: unigram_int4_k4

### 6.2 进一步工作建议

建议分两条线继续推进:

1. 线上默认先采用 optimized_default,因为它在真实 pipeline 下已经证明了压缩率和通信时延收益
2. 针对 TriviaQA 和 WikiText-2 单独做保守化回退实验,例如只保留 delta_noaffine_int4_k1,而对 unigram 再尝试更稳的 k 或混合门控

### 6.3 评测方法建议

建议把后续 runner 再拆成两种模式:

1. real_latency_only: 只统计真实 pipeline 指标,用于快速多轮调参
2. real_latency_plus_exact_drift: 保留当前这种精确 drift 回放,用于最终验收

这样可以显著降低实验墙钟时间,避免把离线 drift 计算成本误判成真实 pipeline 本体成本。