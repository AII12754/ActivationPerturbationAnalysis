# Strategy Drift / Low-Precision Outlier / Int2 Entropy 专题报告

> 实验日期: 2026-03-21  
> 模型: Qwen2.5-32B-Instruct  
> 数据集: WikiText-2, ShareGPT, GSM8k, CNN/DM, Alpaca, TriviaQA  
> 采样: 每数据集 20 warmup + 20 test requests, 其中前 3 个 test request 额外做下游 drift 分析  
> 目标: 用真实输出漂移而不是仅靠 hidden-state 重建指标，重新评估策略选择

---

## 1. 这轮实验回答的问题

这轮实验专门回答四个问题:

1. Int2 路径里的 outlier 值，能不能继续压成低精度，尤其是 Int4
2. Int2 路径如果再叠加熵编码，真实收益有多大
3. 具体策略应该按什么选: 不只是看 hidden cosine，还要看输出从哪个位置开始偏
4. unigram 参考能否用 zero-reference affine 或全局平均激活替代 prev-token

实验脚本:

- [delta_coding_system/experiments/compression_experiment_strategy_drift.py](delta_coding_system/experiments/compression_experiment_strategy_drift.py)

实验结果:

- [results_strategy_drift/strategy_drift_summary.parquet](results_strategy_drift/strategy_drift_summary.parquet)
- [results_strategy_drift/strategy_drift_logits.parquet](results_strategy_drift/strategy_drift_logits.parquet)
- [results_strategy_drift.log](results_strategy_drift.log)

---

## 2. 评估口径

除了已有的重建指标，这轮新增了真实下游输出漂移指标。做法是:

1. 在 layer_boundary=6 处取原始 hidden state
2. 用各压缩策略重建 hidden state
3. 从第 6 层之后手动精确回放剩余 transformer layers
4. 将重建后的 logits 与原模型 logits 逐位置比较

这里的“回放剩余层”已单独验证与原模型完全一致，误差为 0，因此 drift 指标可以直接解释为压缩引入的真实输出偏差。

本轮重点看四类指标:

- 压缩率: `compression_ratio`, `compression_ratio_zlib_packet`, `compression_ratio_ideal_entropy`
- hidden 重建质量: `cosine_mean`, `recon_energy_explained_mean`
- 输出稳定性: `top1_match_rate`, `logit_cosine_mean`, `kl_mean`
- 漂移起点: `first_top1_drift_pos`, `first_logit_cos_below_0_999`

---

## 3. 总体结果

下表是跨 6 个数据集聚合后的总体结果。

| 策略 | 平均压缩率 | zlib 后压缩率 | 平均 Top-1 一致率 | 首次 Top-1 漂移位置 | 首次 logit cos < 0.999 | 平均 logit cosine | 平均 KL |
|------|-----------:|--------------:|------------------:|--------------------:|-----------------------:|------------------:|--------:|
| baseline | 2.180 | 2.208 | 0.998393 | 206.06 | 170.89 | 0.999944 | 0.000117 |
| delta_int2_k8_prev_int4_k2 | 2.242 | 2.267 | 0.995455 | 123.56 | 70.61 | 0.999656 | 0.000544 |
| delta_int2_k8_out8_prev_int4_k2 | 2.266 | 2.285 | 0.995455 | 123.56 | 70.61 | 0.999657 | 0.000544 |
| delta_int2_k8_out8_entropy_prev_int4_k2 | 2.285 | 2.285 | 0.995455 | 123.56 | 70.61 | 0.999657 | 0.000544 |
| delta_int2_k8_out4_prev_int4_k2 | 2.291 | 2.310 | 0.995607 | 117.39 | 71.89 | 0.999646 | 0.000575 |
| prev_int4_k2 | 3.451 | 3.673 | 0.992865 | 97.83 | 43.44 | 0.999548 | 0.000833 |
| zero_affine_int4_k2 | 3.489 | 3.733 | 0.989459 | 105.06 | 0.00 | 0.999003 | 0.002398 |
| global_mean_int4_k2 | 3.489 | 3.730 | 0.980889 | 39.44 | 0.00 | 0.998162 | 0.004676 |
| prev_gs256_k2 | 3.628 | 4.067 | 0.989909 | 92.06 | 39.56 | 0.999277 | 0.001456 |
| prev_int2_k8 | 4.000 | 4.235 | 0.970881 | 64.67 | 4.33 | 0.996973 | 0.006857 |
| prev_int2_k8_out8 | 4.192 | 4.416 | 0.971025 | 65.61 | 4.33 | 0.996977 | 0.006869 |
| prev_int2_k8_out4 | 4.405 | 4.647 | 0.975714 | 71.83 | 4.33 | 0.996960 | 0.007032 |

先给结论:

1. 如果目标是“尽量不动输出”，最稳的仍然不是 Int2 unigram，而是 `prev_int4_k2`
2. 如果目标是“把压缩率再往上推，同时还能接受一定漂移”，`prev_int2_k8_out4` 是这轮最强的高压缩候选
3. delta 混合 Int2 虽然很稳，但压缩率只比 baseline 高一点，提升不够大
4. `zero_affine_int4_k2` 可以作为 prev-token 的备选，但 `global_mean_int4_k2` 明显不够稳

---

## 4. Int2 outlier 低精度: 可以，尤其 Int4 outlier 值值得保留

这里比较的是 unigram prev-token Int2 路径:

| 策略 | 平均压缩率 | 平均 Top-1 一致率 | 首次 Top-1 漂移位置 | 平均 logit cosine | 平均 KL |
|------|-----------:|------------------:|--------------------:|------------------:|--------:|
| prev_int2_k8 | 4.000 | 0.970881 | 64.67 | 0.996973 | 0.006857 |
| prev_int2_k8_out8 | 4.192 | 0.971025 | 65.61 | 0.996977 | 0.006869 |
| prev_int2_k8_out4 | 4.405 | 0.975714 | 71.83 | 0.996960 | 0.007032 |

关键观察:

1. `out8` 相比原始 `prev_int2_k8`，压缩率从 4.000 提升到 4.192，质量几乎不变
2. `out4` 进一步把压缩率提升到 4.405，而且在本轮采样里 Top-1 一致率反而更高
3. `out4` 的 hidden 重建指标略差于 `out8`，但真实输出指标没有恶化，说明更低精度的 outlier 值并没有成为主导误差源

工程解释:

1. Int2 主干已经决定了误差主量级
2. outlier 值本身继续从更高精度压到 Int4，额外损失较小
3. 但节省的字节是真实可见的，因此收益是正的

结论:

- 对 Int2 unigram 路径，outlier 值继续降到 Int4 是值得的
- 从这轮结果看，`prev_int2_k8_out4` 应该替代 `prev_int2_k8_out8`

---

## 5. Int2 + 熵编码: 有收益，但只值得放在本来就高度可压缩的低比特包上

这里比较的是 delta 混合 Int2 路径:

| 策略 | 平均压缩率 | zlib 后压缩率 | 平均 Top-1 一致率 | 平均 logit cosine | 平均 KL |
|------|-----------:|--------------:|------------------:|------------------:|--------:|
| delta_int2_k8_prev_int4_k2 | 2.242 | 2.267 | 0.995455 | 0.999656 | 0.000544 |
| delta_int2_k8_out8_prev_int4_k2 | 2.266 | 2.285 | 0.995455 | 0.999657 | 0.000544 |
| delta_int2_k8_out8_entropy_prev_int4_k2 | 2.285 | 2.285 | 0.995455 | 0.999657 | 0.000544 |

结论很直接:

1. `out8` 相对不带 outlier 值压缩，多拿到约 0.024 的压缩率增益
2. 再叠加熵编码估计后，平均压缩率再提升约 0.019
3. 所有输出指标完全一致，因为熵编码只改变包大小估计，不改变重建值

但也要看绝对收益:

1. 这条路径最终只有 2.285x，还是明显落后于 unigram Int2 族的 4.0x-4.4x
2. 所以 Int2 + 熵编码是“对低比特包进一步榨干”的办法，不是主导性的压缩率跃迁来源

结论:

- Int2 包上做熵编码是有效的
- 但它解决的是“最后 1-2% 的包长”，不是决定性提升
- 如果系统复杂度敏感，熵编码应作为后处理优化，而不是优先级最高的主策略变更

---

## 6. 用真实输出漂移选策略: 最佳点仍在 `prev_int4_k2`

如果按“压缩率尽量高，但输出尽量晚开始漂”的标准来选，三类候选的 trade-off 非常清楚。

### 6.1 稳健优先: `prev_int4_k2`

`prev_int4_k2` 的总体特征是:

- 平均压缩率 3.451x
- 平均 Top-1 一致率 0.992865
- 平均首次 Top-1 漂移位置 97.83
- 平均首次 logit cos < 0.999 出现在 43.44
- 平均 KL 只有 0.000833

它不是压缩率最高的，但它在所有“真正有明显压缩收益”的 unigram 路径里，输出稳定性最好。

从分数据集结果看，它也没有出现某个数据集完全失控的问题:

- ShareGPT: Top-1 0.998264, 首漂 219.33
- CNN/DM: Top-1 0.980035, 首漂 113.33
- TriviaQA: Top-1 0.986111, 首漂 38.00
- WikiText-2: Top-1 0.994652, 首漂 43.33

结论:

- 如果要选默认策略，仍然首推 `prev_int4_k2`

### 6.2 更激进但仍可用: `prev_gs256_k2`

`prev_gs256_k2` 相比 `prev_int4_k2`:

- 压缩率从 3.451x 提升到 3.628x
- zlib 后压缩率从 3.673x 提升到 4.067x
- Top-1 一致率从 0.992865 下降到 0.989909
- 首次 Top-1 漂移位置从 97.83 提前到 92.06

这说明 grouped-scale 256 的包结构更容易再压缩，但质量也确实退了一步。

结论:

- 如果系统侧更关心二次压缩后的包长，并且能接受轻微额外漂移，`prev_gs256_k2` 是可选项
- 但它不能替代 `prev_int4_k2` 成为新的保守默认

### 6.3 极限压缩优先: `prev_int2_k8_out4`

`prev_int2_k8_out4` 的位置很明确:

- 平均压缩率 4.405x
- zlib 后压缩率 4.647x
- 平均 Top-1 一致率 0.975714
- 平均首次 Top-1 漂移位置 71.83
- 平均首次 logit cos < 0.999 出现在 4.33
- 平均 KL 0.007032

这意味着:

1. 它确实把压缩率再推高了一大截
2. 但 logits 在非常早的位置就已经出现可检测偏差
3. Top-1 真正发生偏移的平均位置虽然没有那么靠前，但稳定性明显不如 Int4 方案

结论:

- 如果任务允许更激进的 lossless-ish 近似，`prev_int2_k8_out4` 是当前最强高压缩候选
- 如果任务要求输出长期稳定，不应该把它设成默认

---

## 7. unigram 新参考: zero affine 可以保留，global mean 不建议继续推

### 7.1 `zero_affine_int4_k2`: 有一定可行性，但仍不如 prev-token

| 策略 | 平均压缩率 | 平均 Top-1 一致率 | 首次 Top-1 漂移位置 | 首次 logit cos < 0.999 | 平均 logit cosine | 平均 KL |
|------|-----------:|------------------:|--------------------:|-----------------------:|------------------:|--------:|
| prev_int4_k2 | 3.451 | 0.992865 | 97.83 | 43.44 | 0.999548 | 0.000833 |
| zero_affine_int4_k2 | 3.489 | 0.989459 | 105.06 | 0.00 | 0.999003 | 0.002398 |

`zero_affine_int4_k2` 的特点是:

1. 压缩率略高一点
2. 平均首个 Top-1 偏移位置并不更差
3. 但 logit cosine 从第一个位置起就能检测到偏差，KL 也明显更高

这说明 zero-reference affine 能学到一个全局偏置式近似，但它缺少 prev-token 那种局部上下文对齐能力。

结论:

- `zero_affine_int4_k2` 可以作为“无需参考 token”的备选基线
- 但如果 prev-token 可用，它仍然不是更优选择

### 7.2 `global_mean_int4_k2`: 不建议继续投入

| 策略 | 平均压缩率 | 平均 Top-1 一致率 | 首次 Top-1 漂移位置 | 平均 logit cosine | 平均 KL |
|------|-----------:|------------------:|--------------------:|------------------:|--------:|
| global_mean_int4_k2 | 3.489 | 0.980889 | 39.44 | 0.998162 | 0.004676 |

它与 zero-affine 的字节数几乎一样，但质量显著更差:

1. Top-1 一致率低于 zero-affine 和 prev-token
2. 首次漂移位置明显更早
3. 在 CNN/DM、TriviaQA、WikiText-2 上都更不稳

结论:

- 全局均值激活不是一个好的 unigram 参考
- 没有理由优先于 zero-affine，更没有理由优先于 prev-token

---

## 8. 分数据集观察

### 8.1 长文本生成集最能暴露 drift

在 ShareGPT 和 CNN/DM 上，drift 指标最有区分度:

- `prev_int4_k2` 在 ShareGPT 仍有 0.998264 的 Top-1 一致率
- `prev_int2_k8_out4` 在 CNN/DM 下降到 0.963542
- `global_mean_int4_k2` 在 CNN/DM 只有 0.966146，说明其误差会沿长上下文持续积累

### 8.2 推理数据集上，Int2 的首漂位置不一定最早，但 logit 偏差出现很早

例如 GSM8k:

- `prev_int2_k8_out4` 的平均首次 Top-1 漂移位置仍有 151.67
- 但首次 logit cos < 0.999 只在 1.00 左右就出现

说明 Int2 方案常常先在 logit 排序边缘引入微小扰动，真正翻转 top-1 可能要更后面才发生。

这也是为什么只看 hidden cosine 或只看最终 Top-1，不足以完整评估策略。

---

## 9. 最终建议

按不同目标，建议如下。

### 9.1 默认线上策略

- 继续使用 `prev_int4_k2`

原因:

1. 它在真实输出 drift 指标上仍然是综合最稳的高压缩 unigram 策略
2. 3.451x 的平均压缩率已经显著高于 baseline 的 2.180x
3. 相比更激进的 Int2 方案，它把 drift 推迟得更靠后

### 9.2 激进高压缩策略

- 增加 `prev_int2_k8_out4` 作为可切换模式

适用条件:

1. 明确接受更高输出漂移
2. 任务对长程生成稳定性要求没那么高
3. 更看重把压缩率提升到 4.4x 左右

### 9.3 不依赖局部参考的后备策略

- 保留 `zero_affine_int4_k2`
- 不建议继续推进 `global_mean_int4_k2`

### 9.4 后处理优化

- 可以在 Int2 包上继续尝试真实熵编码实现
- 但这应是次优先级优化，因为它带来的只是约 0.02 量级的额外压缩率改进

---

## 10. 一句话总结

这轮用真实输出漂移重新评估后，结论比只看 hidden 重建更清楚了:

- 默认最优点仍然是 `prev_int4_k2`
- Int2 路径里，outlier 值降到 Int4 是可行且值得的
- Int2 熵编码有效，但收益属于增量优化
- unigram 新参考里，zero-affine 尚可，global-mean 不成立
