# Delta Coding System 设计说明

## 1. 文档目标

本文档面向 `delta_coding_system` 目录中的系统实现，给出一份完整的方案说明与设计设计稿，覆盖以下内容：

1. 解决的问题与适用场景。
2. 系统边界、核心目标与非目标。
3. 模块划分、关键数据结构与时序流程。
4. 编码算法、表结构设计与重建质量保证机制。
5. Prefill / Decode 两阶段的重叠执行策略。
6. 实验口径、输出产物、容量规划、风险与后续演进方向。

本文档不是论文式结果摘要，而是偏工程落地的设计文档，重点说明该系统为什么这样设计、具体怎样工作、上线后如何扩展与维护。

---

## 2. 背景与问题定义

### 2.1 问题背景

在 Pipeline Parallel LLM 推理中，模型会被切分为多个 stage。位于 stage 边界的中间激活必须在设备之间传输。

如果直接传输 FP16 激活，则通信量与序列长度线性增长：

$$
\text{RawBytes} = \text{seq\_len} \times \text{hidden\_dim} \times 2
$$

当模型隐藏维度较大、prefill 序列较长、跨卡带宽有限时，这部分通信会显著影响端到端延迟，并抬高系统吞吐成本。

### 2.2 本系统解决的问题

本系统针对 PP 边界激活传输，设计了一套可在线运行的 delta-coding 压缩方案，核心目标是：

1. 在不显著损伤激活质量的前提下压缩传输体积。
2. 在工程上保持低额外延迟，尤其避免把 CPU 分类和表更新放到关键路径上。
3. 同时支持 prefill 阶段和 decode 阶段，而不是只优化长序列 prefill。
4. 通过持久化 n-gram 激活表提高命中率，使压缩效果随运行时间提升。

### 2.3 核心思路

对于当前 token 位置的激活 $h_t$，若能找到高相关参考激活 $r_t$，则不直接传输 $h_t$，而传输：

1. 一个仿射变换参数 $(s, b)$，使参考更接近当前激活。
2. 变换后的残差 $\Delta = h_t - (s r_t + b)$。
3. 对残差做低比特量化，仅保留少量异常值的高精度表示。

因此接收端可重建：

$$
\hat{h}_t = s r_t + b + \widehat{\Delta}
$$

参考激活来自四级层次匹配：

1. trigram 表命中。
2. 当前请求内 self-reference。
3. bigram 表命中。
4. 若都失败，退化到 unigram 无参考编码。

---

## 3. 设计目标与非目标

### 3.1 设计目标

1. 近无损重建：重建 cosine 尽量保持在 0.999 级别。
2. 稳定压缩比：prefill 和 decode 都能获得实质性压缩收益。
3. 关键路径可控：编码额外耗时相对模型前向足够小。
4. 在线可持续运行：激活表支持增量更新、内存受控、可淘汰。
5. 工程简洁：依赖 PyTorch 张量原语即可运行，不强依赖专用内核。

### 3.2 非目标

1. 不在本目录中实现真实网络发送与跨进程 RPC，仅模拟传输字节与压缩收益。
2. 不处理训练场景，仅针对推理路径。
3. 不解决多边界联合压缩，目前设计针对单个 PP 边界层。
4. 不包含分布式一致性协议，默认表在发送端本地维护。

---

## 4. 系统范围与运行假设

### 4.1 系统范围

该目录下系统主要由以下文件组成：

| 文件 | 作用 |
|---|---|
| `pipeline.py` | 整体编排，负责 Prefill/Decode 流程、CPU/GPU 重叠、指标汇总 |
| `table.py` | trigram/bigram DAG 表结构、分类查询、在线更新、淘汰 |
| `codec.py` | 仿射参数、delta 计算、Int4/Int8 量化、解码与传输字节统计 |
| `run_experiment.py` | 数据集实验驱动、结果落盘 |
| `analyze.py` | 结果聚合、作图、报告生成 |
| `README.md` | 对外使用入口与概述 |

### 4.2 运行假设

1. 发送端与接收端共享同一套编码协议。
2. 使用固定 PP 边界层的 hidden states 作为压缩对象。
3. GPU 负责模型前向、批量编码与重建，CPU 负责分类与表维护。
4. decode 默认逐 token 推进，但允许未来按请求批处理扩展。
5. 激活表默认驻留发送端设备，并使用低精度 dtype 存储。

---

## 5. 总体架构

### 5.1 逻辑架构

```text
                +-----------------------------+
                |   Activation Producer       |
                |   (PP Stage 0 Forward)      |
                +-------------+---------------+
                              |
                              v
                +-----------------------------+
                |   Tier Classifier           |
                |   trigram/self_ref/bigram   |
                +-------------+---------------+
                              |
                              v
                +-----------------------------+
                |   Encoder                   |
                |   affine + delta + quant    |
                +-------------+---------------+
                              |
                              v
                +-----------------------------+
                |   Transport Payload         |
                |   bytes accounting / send   |
                +-------------+---------------+
                              |
                              v
                +-----------------------------+
                |   Decoder / Reconstructor   |
                +-----------------------------+


                +-----------------------------+
                |   NgramTable                |
                |   build / query / update    |
                +-----------------------------+
```

### 5.2 执行架构

系统刻意把工作拆成三类：

1. GPU 前向主路径。
2. CPU 查询与表更新辅助路径。
3. GPU 编码与重建路径。

其核心设计原则是：能与前向并发的工作，不进入关键路径；只有必须依赖当前 hidden states 的编码步骤保留在关键路径上。

---

## 6. 核心模块设计

## 6.1 OverlappedPipeline

`OverlappedPipeline` 是系统主控制器，负责：

1. 调用上游模型提取指定层 hidden states。
2. 发起 CPU 异步分类任务。
3. 组织各 tier 编码。
4. 收集重建质量、压缩率、耗时指标。
5. 触发表更新。
6. 对 prefill 和 decode 提供统一请求处理入口。

### 6.1.1 初始化参数

关键初始化参数包括：

| 参数 | 含义 |
|---|---|
| `layer_boundary` | PP 切分边界层索引 |
| `table_dtype` | 表存储 dtype，默认可配置为 FP8 |
| `max_table_entries` | 最大表项数，0 表示无限制 |
| `group_size` | Int4 delta 量化 group size |
| `top_k` | 每组保留的高精度异常值数量 |
| `int8_group_size` | unigram Int8 编码组大小 |
| `int8_outlier_top_k` | unigram 异常值数量 |
| `decode_tokens` | 单请求 decode 步数 |
| `max_seq_len` | prefill 最长输入长度 |

### 6.1.2 线程模型

实现中通过单线程 `ThreadPoolExecutor(max_workers=1)` 承载 CPU 侧异步任务。当前方案选择单 worker，原因是：

1. 分类与更新都轻量，不需要复杂并发。
2. 避免多个线程同时读写表结构导致额外锁设计。
3. 保持实现简单，降低工程复杂度。

这意味着当前版本更偏向单请求顺序实验环境；如果未来进入多并发线上环境，需要扩展为多 worker 或 actor 化表服务。

## 6.2 NgramTable

`NgramTable` 是系统的长期记忆结构，用来保存历史 trigram/bigram 对应的激活参考。

### 6.2.1 存储结构

内部不是平铺哈希表，而是二级 DAG / trie：

$$
\_dag[A][B] \rightarrow \text{BigramNode}
$$

其中 `BigramNode` 包含：

1. `bigram_hidden`：前缀 `(A, B)` 对应的 bigram 参考激活。
2. `suffixes[C]`：trigram `(A, B, C)` 对应的 trigram 激活。
3. `last_access`：最近访问时间戳。
4. `hit_count`：历史命中次数。

该结构的好处：

1. trigram 与 bigram 共享 `(A, B)` 前缀，减少重复 key 存储。
2. 查 trigram 时先定位 bigram node，再查 suffix，路径较短。
3. 便于为 bigram/trigram 统一维护命中与淘汰统计。

### 6.2.2 分类逻辑

`classify_and_build_refs` 对输入序列逐位置分类，优先级如下：

1. 若 `(A,B,C)` 在表中，记为 `trigram`。
2. 若当前请求内同一 trigram 已在更早位置出现，记为 `self_ref`。
3. 若 `(B,C)` 的 bigram 前缀在表中，记为 `bigram`。
4. 否则记为 `unigram`。

同时返回：

1. 每个位置的 tier 标签。
2. 对 trigram / bigram 直接可用的参考激活张量。
3. self-ref 源位置索引。
4. 当前序列首次 trigram 出现表，用于后续 self-ref。

### 6.2.3 在线更新逻辑

`update_from_hidden_states` 直接复用 prefill 已经算出的整段 hidden states 写表，而不是重新做 3-token forward。这样避免了额外模型推理，是非常关键的工程优化。

更新规则：

1. 位置 `i >= 1` 时写入 bigram `(token[i-1], token[i])`。
2. 位置 `i >= 2` 时写入 trigram `(token[i-2], token[i-1], token[i])`。
3. 若条目已存在则跳过，避免重复写。

### 6.2.4 容量控制与淘汰

表容量超限后，执行打分式淘汰：

$$
\text{score} = 10 \times \text{hit\_count} + \text{last\_access}
$$

分数越低越先淘汰，并将容量压回到 `0.9 * max_entries` 左右。

这是一种频率与最近性混合的启发式近似 LRU/LFU 策略，优点是实现简单，适合当前离线实验。缺点是：

1. bigram 和 trigram 的价值没有分层建模。
2. 只按 trigram 粒度排序，可能不够精细。
3. 没有基于重建收益或字节收益做价值驱动淘汰。

## 6.3 Codec

`codec.py` 定义所有编码原语。

### 6.3.1 有参考编码路径

对 trigram、bigram、self_ref 三类位置，采用统一路径：

1. 计算仿射参数。
2. 参考激活做仿射变换。
3. 计算 delta。
4. 对 delta 做 group-wise Int4 量化。
5. 每组额外保留 `top_k` 个异常值的 FP16 精度。
6. 接收端反量化并重建。

仿射参数定义：

$$
s = \frac{\langle h, r \rangle}{\langle r, r \rangle + \epsilon}
$$

$$
b = \text{mean}(h - s r)
$$

该设计比单纯直接量化 residual 更稳健，因为参考激活可能存在整体尺度偏差或均值漂移。

### 6.3.2 Int4 量化策略

对于每个 group：

1. 先从绝对值上找出 top-k 异常值。
2. 将异常值暂时置零。
3. 对剩余元素做 min-max 量化到 4 bit，即区间 `[0, 15]`。
4. 两个 4-bit 数打包进一个 byte。

这样做的原因：

1. delta 大部分能量集中在少数异常位置。
2. 将异常值单独保留，能显著改善重建质量。
3. Int4 可以把主体数据量压到原始 FP16 的 1/4 以下。

### 6.3.3 无参考编码路径

对于 unigram，没有可用参考，因此不能走 delta 路线。这里采用：

1. group-wise Int8 量化。
2. 每组 top-k 异常值用 FP16 直传。

这是系统中的保底策略，保证即使命中失败，仍然有稳定、高质量的压缩方式。

### 6.3.4 传输负载定义

编码包统计不仅包含量化主体，还包含：

1. per-group `scale`。
2. per-group `zero_point`。
3. top-k 异常值及其索引。
4. 仿射 `scale` 与 `bias`。
5. 参考索引字段。

因此压缩比不是理论 bit 数推导，而是按真实张量 payload 字节数求和得到，更接近实际工程开销。

---

## 7. Prefill 设计

## 7.1 Prefill 处理目标

Prefill 处理长序列时通信量最大，因此这里是系统的核心收益来源。设计目标是：

1. 尽量提高 tier 覆盖率。
2. 把 CPU 分类隐藏在前向后面。
3. 把表更新移出关键路径。

## 7.2 Prefill 时序

```text
Step 1: tokenization
Step 2: CPU 异步 classify_and_build_refs(token_ids)
Step 3: GPU 执行 prefill forward，拿到 boundary hidden states
Step 4: 收集 classify 结果
Step 5: 按 tier 批量编码
Step 6: 统计传输字节与重建质量
Step 7: 异步 update_from_hidden_states
```

关键点：分类任务只依赖 token ids，不依赖真实 hidden states，因此能与模型 forward 并发。

## 7.3 Prefill 编码组织

Prefill 不是逐 token 编码，而是按 tier 聚合后批量编码：

1. trigram + bigram 合并为 delta batch。
2. unigram 单独做 Int8 batch。
3. self_ref 由于依赖“先前已重建结果”，必须按位置排序后处理。

这个拆分兼顾了两点：

1. 对大多数可并行部分最大化 GPU 批处理吞吐。
2. 对 self_ref 保持引用依赖正确性。

## 7.4 Prefill 中 self-ref 的特殊性

self_ref 使用的是“已重建的 earlier position”作为参考，而不是原始 hidden。这样可以模拟接收端真实可见状态，避免发送端使用接收端不可得的信息，从而保证协议闭环正确。

这是一个重要正确性约束：

1. 如果发送端 self-ref 参考了原始 hidden，接收端无法等价重建。
2. 使用 reconstructed 参考，可以保证编码解码链条一致。

---

## 8. Decode 设计

## 8.1 Decode 的设计特点

decode 阶段是逐 token 运行，单步数据量远小于 prefill，但调用频次高。此阶段的工程重点不是极限压缩比，而是把单步额外编码成本压低到可接受范围。

## 8.2 Decode 时序

每一步 decode 的流程：

```text
Step N:
1. 将 next token 追加到 running_token_ids
2. CPU 异步分类当前 decode 位置 tier
3. GPU 执行单步 forward，获得当前层 hidden
4. 收集分类结果，完成单位置编码与重建
5. 异步写表，为后续 token 提供 trigram/bigram 参考
6. 选择下一个 token
```

## 8.3 Decode 分类逻辑

decode 使用 `_classify_decode_step`，其逻辑与 prefill 类似，但输入是当前增量上下文：

1. 优先查 trigram 表。
2. 若当前 trigram 在本轮 decode 中曾出现，则查 `reconstructed_hiddens` 做 self-ref。
3. 否则查 bigram。
4. 都失败则 unigram。

## 8.4 Decode 表更新逻辑

decode 更新是逐步进行的。当前 trigram 之前缀位置可能在 prefill，也可能在 decode，因此 `_update_table_step` 需要从两类来源取 bigram hidden：

1. 若前缀位置仍在 prefill 区间，则从 `prefill_hidden` 取。
2. 若前缀位置已进入 decode 区间，则从 `decode_hidden_by_pos` 取。
3. 如果都不可得，则退化使用当前 hidden。

这个逻辑保证了跨阶段上下文连续性。

## 8.5 Decode 的收益边界

当前实验结果也揭示了一个设计事实：

1. 单 token decode 的原始传输量本来就很小。
2. 在高带宽下，编码开销可能抵消通信收益。
3. decode 真正更适合的部署方式是多请求批处理。

因此，本系统在 decode 上的工程建议是：

1. 单请求低时延场景谨慎开启。
2. 高并发批处理场景收益更稳定。
3. 未来应优先做 batched decode 编码核融合。

---

## 9. 数据结构与协议设计

## 9.1 DeltaPacket

有参考编码包 `DeltaPacket` 包含：

| 字段 | 含义 |
|---|---|
| `quantized_data` | Int4 打包主体 |
| `scales` | 每组 scale |
| `zero_points` | 每组 zero point |
| `topk_values` | 异常值 FP16 |
| `topk_indices` | 异常值位置 |
| `affine_scale` | 每样本仿射缩放 |
| `affine_bias` | 每样本仿射偏置 |
| `ref_indices` | 参考索引字段，当前实现中占位为 0 |
| `group_size` | 分组大小 |
| `top_k` | 异常值数量 |

## 9.2 Int8OutlierPacket

无参考编码包包含：

| 字段 | 含义 |
|---|---|
| `quantized` | Int8 主体 |
| `scales` | 每组 scale |
| `zero_points` | 每组 zero point |
| `topk_values` | 异常值 FP16 |
| `topk_indices` | 异常位置 |
| `group_size` | 分组大小 |
| `top_k` | 异常值数量 |

## 9.3 当前协议特点与局限

特点：

1. 结构清晰，容易直接映射到张量通信。
2. 字节统计准确，便于实验分析。
3. 接收端重建路径与发送端模拟保持一致。

局限：

1. `ref_indices` 目前未真正承载跨端引用定位信息。
2. 协议更偏研究原型，尚未针对网络序列化做紧凑封包设计。
3. 没有引入版本号、校验字段、兼容性层。

如果进入生产环境，需要在本协议之上加一层 transport schema。

---

## 10. 精度、性能与内存设计权衡

## 10.1 为什么表存 FP8

表是系统中唯一会持续增长的长期状态。若用 FP16 存储，内存增长更快；用 FP8 可以近似减半：

$$
\text{TableBytes} \approx (N_{tri} + N_{bi}) \times hidden\_dim \times \text{bytes\_per\_element}
$$

实验表明，FP8 上采样回 FP16 后对参考质量影响很小，因此这是一个非常划算的工程权衡。

## 10.2 为什么 delta 用 Int4，unigram 用 Int8

1. 有参考时，delta 动态范围明显更小，可以安全压到 Int4。
2. 无参考时，直接压 Int4 风险更高，Int8 更稳。
3. 这种分 tier 设计把高压缩比与质量稳定性结合起来。

## 10.3 为什么保留 top-k 异常值

量化误差往往由少数高幅值通道主导。单纯 min-max 量化会被极值拉宽范围，导致大多数正常值分辨率下降。把异常值抽离后：

1. 主体分布更集中。
2. 量化分辨率提升。
3. 小额元数据换来显著质量改进。

## 10.4 为什么要做仿射适配

直接使用参考激活做 residual 时，常会存在整体 scale mismatch。仿射变换可以先把参考校准到更接近当前激活，再对残差量化，通常能同时改善：

1. raw cosine。
2. delta 的动态范围。
3. 最终重建精度。

---

## 11. 指标与输出设计

系统输出分为三层：

## 11.1 在线阶段指标

Prefill 关注：

1. 各 tier 数量。
2. 总传输字节与压缩比。
3. 重建 cosine、MSE。
4. 各环节耗时。

Decode 关注：

1. 每步 tier。
2. 每步重建质量。
3. 每步传输字节。
4. forward / classify / encode 时间。

## 11.2 落盘结果

`run_experiment.py` 将结果保存为多个 parquet 表：

| 文件 | 作用 |
|---|---|
| `prefill_quality.parquet` | 每请求 prefill 聚合指标 |
| `tier_detail.parquet` | 按 tier 的细粒度质量与字节分解 |
| `decode_step_detail.parquet` | 每 decode step 明细 |
| `decode_aggregate.parquet` | 每请求 decode 聚合结果 |
| `table_growth.parquet` | 表项数与内存增长轨迹 |

## 11.3 离线分析

`analyze.py` 负责生成：

1. 压缩比图。
2. prefill/decode tier 分布图。
3. decode 质量曲线。
4. 带宽速度提升分析。
5. 表增长与内存图。
6. markdown 实验报告。

这套产物使系统既能作为算法原型，也能作为工程评估工具。

---

## 12. 典型请求生命周期

### 12.1 Warmup 请求

warmup 的目标是填表，不关注压缩收益。处理逻辑：

1. 跑 prefill。
2. 直接写表。
3. decode 也可运行，但重点是扩大覆盖率。

### 12.2 Test 请求

test 阶段的请求：

1. 在已有表上进行 tier 分类。
2. 计算压缩、质量和时延指标。
3. 更新表，为后续请求继续改进命中率。

这种 warmup + test 两阶段口径让实验更贴近线上服务的“冷启动后进入稳态”过程。

---

## 13. 正确性约束与潜在风险

## 13.1 正确性约束

1. self-ref 必须引用 reconstructed activation，而不是原始 activation。
2. decode 的表更新必须区分前缀 hidden 来自 prefill 还是 decode。
3. 编码和解码必须使用一致的 `group_size`、`top_k` 与仿射定义。
4. 表存储 dtype 改变时，查询侧必须保证正确 upcast。

## 13.2 当前实现风险

1. `ThreadPoolExecutor(max_workers=1)` 更适合单流实验，不适合高并发服务。
2. 表更新和查询都在本地对象上进行，没有显式锁，未来并发化需重构。
3. `ref_indices` 尚未落地为真实传输协议字段，跨端引用仍是模拟语义。
4. decode 单步场景下，压缩未必总能带来时延收益。
5. 当前 eviction 更偏启发式，不一定最优。

## 13.3 失效场景

以下场景会降低收益：

1. 请求文本分布变化大，n-gram 命中率下降。
2. 业务高度随机，重复模式少。
3. 单 token decode 且链路带宽很高。
4. hidden distribution 变化过快，历史参考不再稳定。

---

## 14. 上线视角的扩展建议

## 14.1 协议层扩展

建议新增：

1. 协议版本号。
2. tier 编码枚举。
3. 真实 `ref_index` 或哈希 key。
4. payload 校验和。
5. 向后兼容字段。

## 14.2 多并发服务化

建议把 `NgramTable` 从进程内对象演化为独立服务或分片服务：

1. 支持读写锁或 RCU 模式。
2. 支持按模型、层、租户分片。
3. 支持热更新与冷加载。
4. 支持命中率、收益率监控。

## 14.3 核函数优化

当前编码已较轻量，但仍有进一步优化空间：

1. 将 top-k、量化、pack 融合为单个 Triton/CUDA kernel。
2. 减少 `clone` 和中间张量分配。
3. 降低 GPU 同步点。
4. 在 decode 中引入 batch 编码。

## 14.4 淘汰策略优化

建议从简单 hit/access 打分升级为收益驱动：

1. 按历史节省字节数打分。
2. 按重建质量收益打分。
3. 按数据集域或业务域独立预算。
4. 对 bigram 与 trigram 分别建预算池。

---

## 15. 推荐的后续演进路线

### 阶段一：工程化收口

1. 明确真实传输协议。
2. 补齐接收端独立实现。
3. 将 `ref_indices` 变成真实可路由引用。
4. 增加单元测试和一致性测试。

### 阶段二：性能优化

1. 融合量化 kernel。
2. batched decode。
3. 异步流水线进一步减少同步。
4. 更精细的内存池管理。

### 阶段三：线上部署能力

1. 多 worker / 多请求并行。
2. 表服务化与分片。
3. 监控与回滚机制。
4. 基于业务流量动态调参。

---

## 16. 结论

`delta_coding_system` 的本质是一套围绕 PP 边界激活压缩而设计的工程化近无损通信系统。它并不是单一的量化模块，而是由以下四部分共同构成：

1. 持久化 n-gram 参考记忆。
2. 分层匹配与分 tier 编码策略。
3. CPU/GPU 重叠执行的低关键路径编排。
4. 可复现实验与分析链路。

从设计上看，这个系统最有价值的地方不是单点压缩算法，而是把“参考命中、压缩编码、在线更新、容量控制、分析评估”整合成了一个闭环。

如果继续沿当前方向演进，最合理的路线是：

1. 把研究原型协议补全为真实生产协议。
2. 把单机单流表结构演进为并发可服务化组件。
3. 把当前 PyTorch 原语实现下沉为融合 kernel。

完成这三步后，该系统就能从实验性验证方案进一步逼近生产级跨卡激活压缩方案。