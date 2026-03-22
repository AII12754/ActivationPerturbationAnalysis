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

## 4. PP 通信端到端时延定义

本报告中的 PP 端到端时延定义为:

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

也就是说, 这不是只看编码时间, 而是模拟从第 6 层 hidden 准备好, 到下一阶段拿到重建 hidden 可以继续推理之间的完整等待时间。

---

## 5. Delta 路径结果

### 5.1 Prefill 阶段

| 策略 | 压缩率 | sender ms | 200Mbps 网络 ms | receiver ms | 200Mbps e2e ms | 500Mbps e2e ms | 1000Mbps e2e ms | top1 match | logit cos | KL mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_current | 2.3325x | 0.4380 | 0.1774 | 0.1682 | 0.7837 | 0.6772 | 0.6417 | 0.996165 | 0.999903 | 0.000421 |
| delta_noaffine_int4_k1 | 2.3334x | 0.3796 | 0.1774 | 0.1637 | 0.7207 | 0.6143 | 0.5788 | 0.995165 | 0.999861 | 0.000581 |
| delta_int2_k8_out4 | 2.5215x | 0.6159 | 0.1659 | 0.1901 | 0.9719 | 0.8723 | 0.8391 | 0.990364 | 0.999470 | 0.001561 |
| delta_noaffine_int2_k8_out4 | 2.5226x | 0.5580 | 0.1659 | 0.1826 | 0.9064 | 0.8069 | 0.7737 | 0.990326 | 0.999411 | 0.002097 |
| delta_int2_k8_out8_entropy | 2.5058x | 0.5732 | 0.1668 | 0.1878 | 0.9278 | 0.8277 | 0.7943 | 0.990511 | 0.999488 | 0.001262 |

结论:

1. delta 默认仍然应当选 delta_noaffine_int4_k1。
2. 在 prefill 阶段, 它相比 baseline_current:
   - 压缩率几乎不变
   - 500 Mbps 下 PP e2e 从 0.6772 ms 降到 0.6143 ms, 下降约 9.3%
   - 质量只发生很小退化
3. Int2 系列可以继续作为压缩优先备选, 但不应默认替代 Int4。

### 5.2 Decode 阶段

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

## 6. Unigram 路径结果

### 6.1 Prefill 阶段

| 策略 | 压缩率 | sender ms | 200Mbps 网络 ms | receiver ms | 200Mbps e2e ms | 500Mbps e2e ms | 1000Mbps e2e ms | top1 match | logit cos | KL mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| unigram_int4_k4 | 3.3385x | 0.4529 | 0.1227 | 0.2057 | 0.7814 | 0.7077 | 0.6832 | 0.991584 | 0.999662 | 0.000972 |
| prev2_blend_int4_k2 | 3.4473x | 0.5332 | 0.1191 | 0.2238 | 0.8760 | 0.8046 | 0.7808 | 0.988963 | 0.999341 | 0.003177 |
| prev_int4_k2 | 3.4632x | 0.6299 | 0.1184 | 0.2306 | 0.9789 | 0.9079 | 0.8842 | 0.988511 | 0.999331 | 0.003436 |
| prev_gs256_k2 | 3.6116x | 0.6281 | 0.1136 | 0.2300 | 0.9716 | 0.9035 | 0.8807 | 0.984045 | 0.998452 | 0.010661 |
| prev_int2_k8_out4 | 4.2396x | 0.9059 | 0.0971 | 0.2613 | 1.2643 | 1.2061 | 1.1867 | 0.966533 | 0.996073 | 0.016343 |

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

## 7. 与固定 64 Token 结果相比, 长 Decode 下发生了什么

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

## 8. 对“高时延”的重新判断

在 PP 端到端口径下, 三个结论非常明确:

1. 200 Mbps 下, 网络传输已经与本地 codec 开销同量级, 不能再忽略。
2. 500 Mbps 和 1000 Mbps 下, 发送侧编码与接收侧重建重新成为主要组成部分。
3. 也就是说, 真正的 latency 决策已经从“单纯压 codec 微算子”变成“压缩率、发送端负载、接收端负载、网络带宽”四者共同折中。

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

## 9. 最终结论

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

---

## 10. 工程建议

在你明确说“先不要改之前系统”的前提下, 当前最合理的下一步是:

1. 先保持系统默认配置不变。
2. 以本报告结论作为新默认候选的最终实验依据。
3. 下一轮只做两类工程验证:
   - 把这两个默认候选接进真实 pipeline, 测端到端 latency
   - 在真实链路里替换模拟带宽为实测带宽, 验证 200 / 500 / 1000 Mbps 模型误差
