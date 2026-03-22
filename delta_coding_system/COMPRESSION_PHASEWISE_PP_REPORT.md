# Delta-Coding 激活压缩系统 —— 动态 Decode 与 PP 通信端到端时延报告

> 实验日期: 2026-03-22
> 模型: Qwen2.5-32B-Instruct
> hidden_dim: 5120
> layer_boundary: 6
> 运行方式: 单卡单进程, 多数据集并行, 不使用 TP
> GPU 分配: GPU 1-6, 避开 GPU 0
> 数据集: wikitext2, sharegpt, gsm8k, cnn_dm, alpaca, triviaqa
> 每数据集: 30 warmup + 100 test requests
> decode 规则: 贪心生成, 遇到 EOS 停止, 否则最多 512 token
> decode 执行方式: 启用 KV Cache

---

## 1. 这轮实验解决什么问题

这轮实验是对上一轮大样本 phasewise 实验的两项关键修正:

1. 不再把 decode 长度固定为 64 token, 而是改成真实动态生成, 直到 EOS 或 512 token 上限。
2. latency 不再只看 codec 微算子, 而是改成更接近 PP 部署的端到端通信口径:
   - 第 6 层完成后的发送侧编码开销
   - 网络传输时延
   - 接收侧解码与重建开销
   - 下一阶段可以开始推理前的总等待时间

这正对应用户提出的三个要求:

1. 长 decode 下重新比较漂移。
2. latency 改成 PP 之间通信端到端时延并附 breakdown。
3. 确认模型推理应该启用 KV Cache。

---

## 2. 实验实现与结果文件

新增脚本:

- [delta_coding_system/compression_experiment_phasewise_pp.py](delta_coding_system/compression_experiment_phasewise_pp.py)

脚本中可直接确认 KV Cache 已启用:

1. prefill 调用在 [delta_coding_system/compression_experiment_phasewise_pp.py](delta_coding_system/compression_experiment_phasewise_pp.py#L251) 使用了 use_cache=True
2. decode 循环在 [delta_coding_system/compression_experiment_phasewise_pp.py](delta_coding_system/compression_experiment_phasewise_pp.py#L266) 读取了 past_key_values
3. decode step 在 [delta_coding_system/compression_experiment_phasewise_pp.py](delta_coding_system/compression_experiment_phasewise_pp.py#L275) 基于 past_key_values 继续推理
4. decode 长度上限在 [delta_coding_system/compression_experiment_phasewise_pp.py](delta_coding_system/compression_experiment_phasewise_pp.py#L585) 设为 512

结果目录:

- [results_phasewise_pp_run](results_phasewise_pp_run)
- [results_phasewise_pp_smoke](results_phasewise_pp_smoke)
- [results_phasewise_pp_merged](results_phasewise_pp_merged)

合并后核心产物:

- [results_phasewise_pp_merged/phasewise_pp_request_summary.parquet](results_phasewise_pp_merged/phasewise_pp_request_summary.parquet)
- [results_phasewise_pp_merged/phasewise_pp_drift.parquet](results_phasewise_pp_merged/phasewise_pp_drift.parquet)
- [results_phasewise_pp_merged/phasewise_pp_position_detail.parquet](results_phasewise_pp_merged/phasewise_pp_position_detail.parquet)
- [results_phasewise_pp_merged/phasewise_pp_summary.csv](results_phasewise_pp_merged/phasewise_pp_summary.csv)
- [results_phasewise_pp_merged/phasewise_pp_vs_64_compare.csv](results_phasewise_pp_merged/phasewise_pp_vs_64_compare.csv)

总记录数:

1. request 级记录: 18000
2. drift 级记录: 18000
3. position 级记录: 133455

---

## 3. 动态 Decode 长度分布

这轮最重要的新事实是: decode 基本不提前结束。

总体统计:

| 指标 | 数值 |
|---|---:|
| 请求数 | 600 |
| 平均 decode 长度 | 511.09 |
| 中位数 | 512 |
| P90 | 512 |
| P95 | 512 |
| P99 | 512 |
| 最大值 | 512 |
| 达到 512 上限的请求数 | 598 / 600 |

分数据集看:

| 数据集 | 平均 decode 长度 |
|---|---:|
| alpaca | 512.00 |
| cnn_dm | 512.00 |
| gsm8k | 512.00 |
| sharegpt | 512.00 |
| triviaqa | 512.00 |
| wikitext2 | 506.52 |

结论:

1. 这轮结果可以视为长 decode 极限口径。
2. 除了极少数 wikitext2 样本因为 EOS 提前结束, 其余几乎全部请求都顶到了 512 token 上限。
3. 因此这轮漂移结果更能反映长生成链条里的累计误差, 比固定 64 token 的口径更严格。

---

## 4. Overlap 实现边界

这里先直接给判断:

1. 当前这份 phasewise PP 实验脚本, 并没有真实执行完整 overlap runtime。
2. 它做的是策略级 phasewise 分析, 然后用 analytical critical-path 模型去估算 PP 通信等待时间。
3. 因此它适合比较不同策略的相对排序, 但不适合直接拿来当生产系统的绝对额外时延。

具体来说:

1. 生产系统里的 overlap 实现在 [delta_coding_system/pipeline.py](delta_coding_system/pipeline.py):
   - prefill 阶段会把 classify 和 forward 并发调度
   - decode 阶段会把 classify 与单步 forward 并发调度
   - table update 走独立线程池, 试图从主 critical path 中剥离
2. 但当前 phasewise PP 脚本 [delta_coding_system/compression_experiment_phasewise_pp.py](delta_coding_system/compression_experiment_phasewise_pp.py) 并没有真实使用这个 pipeline runtime:
   - prefill 的 reference 匹配发生在完整 full_out 之后
   - table update 也是请求末尾串行调用
   - 各 token 的策略评估与 drift 回放也是离线顺序执行

因此, 对“是否正确实现了 overlap”这个问题, 更准确的回答是:

1. 生产 pipeline 有部分 overlap 实现。
2. 当前 phasewise PP 实验没有真实执行完整 overlap, 只是把 token-only 操作视为可被掩盖, 从而不纳入 PP critical path。

---

## 5. PP 通信端到端时延定义

本报告中的 PP 端到端时延 analytical 模型定义为:

$$
T_{pp\_e2e}(b) = T_{sender} + T_{network}(b) + T_{receiver}
$$

其中:

1. 发送侧时延:

$$
T_{sender} = T_{affine\_param} + T_{affine\_apply} + T_{delta} + T_{pack} + T_{outlier}
$$

2. 接收侧时延:

$$
T_{receiver} = T_{decode} + T_{reconstruct}
$$

3. 网络传输时延:

$$
T_{network}(b) = \frac{8 \cdot \text{bytes}}{b \cdot 1000}
$$

这里的 b 分别取 200 Mbps, 500 Mbps, 1000 Mbps。

这里有两个很重要的口径约束:

1. prefill 阶段应看整段 prompt hidden 的总传输时延, 而不是单 token 平均值。
2. decode 阶段才适合看单 token 平均时延, 因为 decode 本身就是 token-by-token 推进。

上一版报告在这一点上写错了: 它把 prefill 网络时延写成了 per-position mean, 这会明显低估真实 prompt 级传输等待时间。

---

## 6. Delta 路径结果

### 5.1 Prefill 阶段

下面的 prefill 表全部改为“单条 request 的总等待时间”。

| 策略 | 压缩率 | sender total ms | 200Mbps 网络 total ms | receiver total ms | 200Mbps e2e total ms | 500Mbps e2e total ms | 1000Mbps e2e total ms | top1 match | logit cos | KL mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_current | 2.3325x | 104.69 | 42.85 | 40.46 | 188.01 | 162.30 | 153.72 | 0.996165 | 0.999903 | 0.000421 |
| delta_noaffine_int4_k1 | 2.3334x | 90.76 | 42.84 | 39.40 | 172.99 | 147.29 | 138.72 | 0.995165 | 0.999861 | 0.000581 |
| delta_int2_k8_out4 | 2.5215x | 146.90 | 40.14 | 46.12 | 233.15 | 209.07 | 201.04 | 0.990364 | 0.999470 | 0.001561 |
| delta_noaffine_int2_k8_out4 | 2.5226x | 133.09 | 40.13 | 43.84 | 217.06 | 192.98 | 184.96 | 0.990326 | 0.999411 | 0.002097 |
| delta_int2_k8_out8_entropy | 2.5058x | 138.87 | 40.37 | 45.14 | 224.38 | 200.16 | 192.09 | 0.990511 | 0.999488 | 0.001262 |

结论:

1. delta 默认仍然应当选 delta_noaffine_int4_k1。
2. 在 prefill 阶段, 它相比 baseline_current:
   - 压缩率几乎不变
   - 500 Mbps 下整段 prompt 的 PP e2e 从 162.30 ms 降到 147.29 ms, 下降约 9.2%
   - 质量只发生很小退化
3. Int2 系列可以继续作为压缩优先备选, 但不应默认替代 Int4。

### 5.2 Decode 阶段

decode 阶段保留单 token 平均值, 因为这是 decode 的自然口径。

| 策略 | 压缩率 | sender ms | 200Mbps 网络 ms | receiver ms | 200Mbps e2e ms | 500Mbps e2e ms | 1000Mbps e2e ms | top1 match | logit cos | KL mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_current | 2.5041x | 0.4631 | 0.1670 | 0.1785 | 0.8085 | 0.7083 | 0.6749 | 0.998364 | 0.999213 | 0.000081 |
| delta_noaffine_int4_k1 | 2.5055x | 0.3875 | 0.1669 | 0.1728 | 0.7272 | 0.6271 | 0.5937 | 0.997777 | 0.998797 | 0.000374 |
| delta_int2_k8_out4 | 2.8155x | 0.6895 | 0.1523 | 0.2039 | 1.0457 | 0.9543 | 0.9239 | 0.995577 | 0.998299 | 0.000546 |
| delta_noaffine_int2_k8_out4 | 2.8174x | 0.6149 | 0.1522 | 0.1966 | 0.9638 | 0.8725 | 0.8420 | 0.996072 | 0.997901 | 0.001596 |
| delta_int2_k8_out8_entropy | 2.7858x | 0.6350 | 0.1535 | 0.2054 | 0.9940 | 0.9019 | 0.8712 | 0.995743 | 0.998301 | 0.000490 |

结论:

1. decode 长度几乎全都扩展到 512 后, delta_noaffine_int4_k1 仍然是最稳默认。
2. 它在 500 Mbps 下的 PP e2e 为 0.6271 ms, 相比 baseline 的 0.7083 ms 下降约 11.5%。
3. 也就是说, 去 affine 的收益不仅体现在 codec 微时延, 放到 PP 端到端口径下仍然成立。

---

## 7. Unigram 路径结果

### 6.1 Prefill 阶段

| 策略 | 压缩率 | sender total ms | 200Mbps 网络 total ms | receiver total ms | 200Mbps e2e total ms | 500Mbps e2e total ms | 1000Mbps e2e total ms | top1 match | logit cos | KL mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| unigram_int4_k4 | 3.3385x | 108.70 | 29.54 | 49.56 | 187.81 | 170.08 | 164.17 | 0.991584 | 0.999662 | 0.000972 |
| prev2_blend_int4_k2 | 3.4473x | 128.62 | 28.25 | 54.43 | 211.30 | 194.35 | 188.70 | 0.988963 | 0.999341 | 0.003177 |
| prev_int4_k2 | 3.4632x | 152.56 | 28.21 | 55.99 | 236.76 | 219.83 | 214.19 | 0.988511 | 0.999331 | 0.003436 |
| prev_gs256_k2 | 3.6116x | 152.14 | 27.01 | 55.84 | 234.99 | 218.78 | 213.38 | 0.984045 | 0.998452 | 0.010661 |
| prev_int2_k8_out4 | 4.2396x | 221.33 | 22.93 | 63.78 | 308.04 | 294.29 | 289.70 | 0.966533 | 0.996073 | 0.016343 |

结论:

1. 在动态长 decode 口径下, unigram 默认依然建议使用 unigram_int4_k4。
2. 它在 prefill 上继续保持三方面同时占优:
   - 最低 PP e2e
   - 足够高的压缩率
   - 最稳的输出质量
3. prev_int4_k2 与 prev2_blend_int4_k2 仍是可保留备选, 但不再适合作为默认。

### 6.2 Decode 阶段

| 策略 | 压缩率 | sender ms | 200Mbps 网络 ms | receiver ms | 200Mbps e2e ms | 500Mbps e2e ms | 1000Mbps e2e ms | top1 match | logit cos | KL mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| unigram_int4_k4 | 3.3789x | 0.4779 | 0.1213 | 0.2093 | 0.8085 | 0.7357 | 0.7115 | 0.996952 | 0.998643 | 0.000237 |
| prev2_blend_int4_k2 | 3.5215x | 0.5459 | 0.1163 | 0.2268 | 0.8890 | 0.8192 | 0.7960 | 0.996622 | 0.998535 | 0.000253 |
| prev_int4_k2 | 3.5190x | 0.6376 | 0.1164 | 0.2320 | 0.9860 | 0.9161 | 0.8928 | 0.996473 | 0.998535 | 0.000234 |
| prev_gs256_k2 | 3.6490x | 0.6264 | 0.1123 | 0.2317 | 0.9703 | 0.9030 | 0.8805 | 0.995640 | 0.998194 | 0.000381 |
| prev_int2_k8_out4 | 4.1919x | 0.8667 | 0.0981 | 0.2597 | 1.2246 | 1.1657 | 1.1461 | 0.990943 | 0.996899 | 0.001753 |

结论:

1. 长 decode 下, unigram_int4_k4 仍然是新的默认首选。
2. 它在 500 Mbps 下的 PP e2e 为 0.7357 ms, 显著低于 prev_int4_k2 的 0.9161 ms, 下降约 19.7%。
3. 同时它的 top1 match 还略高于 prev_int4_k2。
4. 这说明上一轮大样本固定 64 token 里的结论在动态 512 token 长 decode 口径下没有被推翻, 反而更稳了。

---

## 8. 与固定 64 Token 结果相比, 长 Decode 下发生了什么

对比上一轮固定 64 token 结果:

1. decode 长度从平均 64 扩展到平均 511.09。
2. 各策略在 decode 阶段的平均 logit cosine 普遍下降, 大致下降幅度在 0.0007 到 0.0019 之间。
3. 但策略排序基本没有颠覆:
   - delta 默认仍是 delta_noaffine_int4_k1
   - unigram 默认仍是 unigram_int4_k4

一些代表性对比:

### 7.1 Delta 默认候选

delta_noaffine_int4_k1:

1. 固定 64 token decode:
   - top1 match: 0.997917
   - logit cosine: 0.999892
   - KL mean: 0.000128
2. 动态 512 token decode:
   - top1 match: 0.997777
   - logit cosine: 0.998797
   - KL mean: 0.000374

解读:

1. 长 decode 下累计误差确实更明显。
2. 但退化仍然很温和, 仍处在可接受范围。

### 7.2 Unigram 默认候选

unigram_int4_k4:

1. 固定 64 token decode:
   - top1 match: 0.996068
   - logit cosine: 0.999783
   - KL mean: 0.000274
2. 动态 512 token decode:
   - top1 match: 0.996952
   - logit cosine: 0.998643
   - KL mean: 0.000237

解读:

1. top1 match 没有恶化, 甚至均值略升, 说明它在长 decode 上非常稳。
2. 但 logit cosine 的确出现了长期累计下滑, 所以仍应把它理解成“高稳定近似”, 而不是“完全无漂移”。

### 7.3 Prev 参考路线

prev_int4_k2:

1. 固定 64 token decode:
   - top1 match: 0.995703
   - logit cosine: 0.999784
2. 动态 512 token decode:
   - top1 match: 0.996473
   - logit cosine: 0.998535

解读:

1. 它的长期质量仍然不错。
2. 但考虑到 PP e2e 时延明显高于 unigram_int4_k4, 它依然不应该回到默认位。

---

## 9. 为什么它看起来比旧 Pipeline 更慢

你的判断是对的: 当前这份 phasewise PP 报告里的绝对额外开销, 确实明显高于之前 pipeline 系统报告。

这不是模型忽然变慢了, 而是口径不同。

已有 pipeline baseline 结果显示:

1. 平均 prefill forward: 529.86 ms
2. classify: 0.056 ms
3. encode_delta: 20.76 ms
4. encode_self_ref: 1.52 ms
5. encode_unigram: 3.56 ms
6. table_update: 0.007 ms
7. total: 616.66 ms

也就是说, 旧 pipeline 系统里 prefill 额外开销大约只有:

$$
20.76 + 1.52 + 3.56 + 0.056 + 0.007 \approx 25.9\text{ ms}
$$

但这份 phasewise PP 报告里, baseline_current 的 prefill 500 Mbps e2e total 却是 162.30 ms。

差异来自三件事:

1. 旧 pipeline 是实际系统实现, delta 路径按 tier batch 编码, 不是逐 token 单独编码。
2. 旧 pipeline 确实包含 classify/table-update 与 forward 的 overlap 设计。
3. 当前 phasewise PP 脚本是离线 phasewise 策略评估器, 会逐位置重建和统计, 并不等价于真实 batch pipeline。

因此正确解释应该是:

1. 当前 phasewise PP 结果适合比较策略相对优劣。
2. 它不适合直接作为系统绝对额外时延。
3. 如果要回答“真实系统会不会比旧 pipeline 更慢”, 必须把 shortlist 策略真正接入 [delta_coding_system/pipeline.py](delta_coding_system/pipeline.py) 再测一次系统级端到端 latency。

---

## 10. 对“高时延”的重新判断

在 PP 端到端口径下, 三个结论非常明确:

1. 对 prefill, 网络时延绝不能按单 token 均值解释, 必须按整段 request 总传输量解释。
2. 对 decode, 才适合按单 token 平均值解释。
3. 200 Mbps 下, prefill 总网络时延已经明显不可忽略。
4. 500 Mbps 和 1000 Mbps 下, 本地 sender/receiver 负载仍然是大头。

以 decode 阶段两个默认候选为例:

### 8.1 Delta 默认候选

delta_noaffine_int4_k1 在 500 Mbps 下:

1. sender: 0.3875 ms
2. network: 0.0667 ms
3. receiver: 0.1728 ms
4. e2e: 0.6271 ms

baseline_current 在 500 Mbps 下:

1. sender: 0.4631 ms
2. network: 0.0668 ms
3. receiver: 0.1785 ms
4. e2e: 0.7083 ms

这里的主要收益来自 sender 和 receiver 两侧都略降, 而不是网络变快。

### 8.2 Unigram 默认候选

unigram_int4_k4 在 500 Mbps 下:

1. sender: 0.4779 ms
2. network: 0.0485 ms
3. receiver: 0.2093 ms
4. e2e: 0.7357 ms

prev_int4_k2 在 500 Mbps 下:

1. sender: 0.6376 ms
2. network: 0.0466 ms
3. receiver: 0.2320 ms
4. e2e: 0.9161 ms

这里可以清楚看出:

1. 两者网络开销几乎一样。
2. unigram_int4_k4 的优势主要来自更低的 sender 与 receiver 本地开销。

---

## 11. 最终结论

### 9.1 默认策略结论保持不变

在动态长 decode 与 PP 通信端到端口径下, 默认建议仍然是:

1. delta 路径默认: delta_noaffine_int4_k1
2. unigram 路径默认: unigram_int4_k4

### 9.2 长 Decode 不会推翻上一轮推荐, 只会让结论更稳

原因是:

1. 这轮几乎所有请求都生成到了 512 token 上限。
2. 在这样更严格的长期累计误差场景下, 两个默认候选仍然维持了最好的综合平衡。

### 9.3 KV Cache 已确认启用, 而且这也是正确的推理方式

结论是肯定的:

1. 这个模型在 decode 推理时应该启用 KV Cache。
2. 本轮实验已经按这个方式运行。
3. 如果不启用 KV Cache, decode 计算量和端到端时延都会显著失真, 不再代表真实部署。

### 9.4 关于 overlap 的最终判断

最终需要把两句话区分开:

1. “这些 token-only 操作理论上可以在模型推理时被掩盖” 这件事是合理的。
2. “当前 phasewise PP 实验已经真实实现并测量了这种 overlap” 这件事不成立。

当前实验里, overlap 只是 analytical assumption, 不是 runtime fact。

---

## 12. 工程建议

在你明确说“先不要改之前系统”的前提下, 当前最合理的下一步是:

1. 先保持系统默认配置不变。
2. 把这份报告只用作 shortlist 策略排序依据, 不要直接当系统绝对时延结论。
3. 如果下一轮要重做系统级实验, 应该:
   - 先清理旧结果目录, 避免旧数据污染新数据
   - 先检查 GPU 1-7 是否完全空闲, 不使用 GPU 0
   - 将 shortlist 策略真正接进 [delta_coding_system/pipeline.py](delta_coding_system/pipeline.py)
   - 用真实 batch encode + overlap runtime 重测端到端 latency
4. 只有完成这一步, 才能严肃回答“新默认策略会不会让系统比旧 pipeline 更慢”。
