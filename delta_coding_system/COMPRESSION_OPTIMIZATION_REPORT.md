# Delta-Coding 激活压缩系统 —— 压缩率优化实验报告

> **实验日期**: 2026-03-20  
> **模型**: Qwen2.5-32B-Instruct (hidden_dim=5120, layer_boundary=6)  
> **测试数据集**: WikiText-2, ShareGPT, GSM8k, CNN/DM, Alpaca, TriviaQA  
> **每数据集**: 30 warmup + 30 test requests (seq_len ≤ 512)  

---

## 1. 现有系统基线分析

### 1.1 当前压缩方案

当前 Delta-Coding 系统采用分层压缩策略:

| 位置分类 | 参考来源 | 量化方式 | 每位置传输字节 | 压缩比 |
|----------|---------|----------|--------------|--------|
| **Trigram** | DAG 表三元组查找 | Affine + Int4 + top-1 outlier | ~2,852 B | **3.59×** |
| **Bigram** | DAG 表二元组查找 | Affine + Int4 + top-1 outlier | ~2,852 B | **3.59×** |
| **Self-Ref** | 序列内重复三元组 | Affine + Int4 + top-1 outlier | ~2,852 B | **3.59×** |
| **Unigram** | 无参考 | Int8 + top-1 outlier | ~5,400 B | **1.90×** |
| *FP16 原始* | — | — | 10,240 B | 1.00× |

**Delta 路径 (Affine + Int4) 字节分解:**

| 组件 | 大小 | 说明 |
|------|------|------|
| quantized_data | 2,560 B | Int4 packed (5120 / 2) |
| scales | 80 B | 40 groups × FP16 |
| zero_points | 80 B | 40 groups × FP16 |
| topk_values | 80 B | 40 groups × 1 × FP16 |
| topk_indices | 40 B | 40 groups × 1 × uint8 |
| affine_scale | 2 B | FP16 |
| affine_bias | 2 B | FP16 |
| ref_indices | 8 B | int64 |
| **合计** | **2,852 B** | |

### 1.2 基线性能 (原始 Pipeline 系统, 1400 requests)

| 数据集 | 压缩率 | 余弦相似度 | Trigram% | Bigram% | SelfRef% | **Unigram%** |
|--------|--------|-----------|---------|---------|---------|------------|
| GSM8k | 2.690× | 0.999714 | 27.8% | 25.6% | 8.0% | **38.6%** |
| Alpaca | 2.675× | 0.999684 | 11.9% | 15.1% | 33.9% | **39.1%** |
| TriviaQA | 2.608× | 0.999721 | 30.7% | 17.6% | 7.4% | **44.3%** |
| ShareGPT | 2.527× | 0.999733 | 11.6% | 11.6% | 27.0% | **49.8%** |
| WikiText-2 | 2.456× | 0.999678 | 14.6% | 23.5% | 9.2% | **52.7%** |
| CNN/DM | 2.335× | 0.999714 | 7.9% | 22.8% | 8.3% | **60.9%** |
| **平均** | **2.521×** | **0.999711** | 16.2% | 20.0% | 14.5% | **49.3%** |

### 1.3 瓶颈分析

```
字节组成分析 (原始系统):
  Trigram 字节:  12.8%  ─── 已高效压缩 (3.59×)
  Bigram 字节:   14.6%  ─── 已高效压缩 (3.59×)
  SelfRef 字节:  12.5%  ─── 已高效压缩 (3.59×)
  ██████████████████████████████████████████████████████████████
  Unigram 字节:  60.1%  ─── 低效压缩 (1.90×)  ← 瓶颈所在
```

**核心发现: Unigram 路径仅占 49.3% 的位置,却贡献了 60.1% 的传输字节。** 这是因为 Int8 量化 (5,400 B/token) 与 Delta 路径 (2,852 B/token) 差距近 2 倍。优化 Unigram 路径是提升压缩率的关键。

---

## 2. 实验设计

### 2.1 探索的压缩策略

我们设计并实验了 12 种压缩策略,覆盖四个技术方向:

#### 方向一: 稀疏传输 (Sparse Group Transmission)

| 策略 | 原理 | 适用路径 |
|------|------|---------|
| **sparse_010** | Delta 分组中能量 < 10% max 的组跳过不传,用 5 字节 bitmask 标记 | Delta |
| **sparse_005** | 同上,阈值 5% | Delta |

#### 方向二: 低精度量化 (Lower Precision Quantization)

| 策略 | 原理 | 适用路径 |
|------|------|---------|
| **int2_topk4** | Int2 (4 levels) + top-4 outlier 替代 Int4 + top-1 | Delta |
| **int2_topk8** | Int2 (4 levels) + top-8 outlier | Delta |
| **adaptive_bw** | Trigram 用 Int2+top4, Bigram/SelfRef 保持 Int4 | Delta |
| **unigram_int4_k4** | Unigram 用 Int4+top4 替代 Int8+top1 | Unigram |
| **unigram_int4_k8** | Unigram 用 Int4+top8 | Unigram |

#### 方向三: 相邻 Token 参考 (Previous-Token Reference)

| 策略 | 原理 | 适用路径 |
|------|------|---------|
| **prev_token (prev_int4_k1)** | Unigram 位置使用前一 token 的 hidden 作为参考,Affine+Int4+top1 | Unigram→Delta |
| **prev_int4_k2** | 同上,top_k=2 | Unigram→Delta |
| **prev_int4_k4** | 同上,top_k=4 | Unigram→Delta |
| **prev_int2_k4** | 前一 token 参考 + Int2+top4 | Unigram→Delta |
| **prev_int2_k8** | 前一 token 参考 + Int2+top8 | Unigram→Delta |
| **mean_pool** | 使用前 4 token 均值作为参考 | Unigram→Delta |
| **prev_gs256** | 前一 token 参考,group_size=256 (减少 metadata) | Unigram→Delta |

#### 方向四: 学习式变换 (Learned Transformation)

| 策略 | 原理 | 适用路径 |
|------|------|---------|
| **linear_pred** | 用逐通道线性回归 ($w_i \cdot ref_i + b_i$) 替代逐样本 Affine | Delta |

#### 方向五: 组合策略 (Combined)

| 策略 | 原理 |
|------|------|
| **sparse_prev** | Delta 路径用 sparse group + Unigram 用 prev-token |
| **combined** | Trigram 用 sparse+Int2, Bigram 用 sparse+Int4, Unigram 用 prev-token |
| **best_combined** | Delta 保持 Int4+top1, Unigram 用 prev-token+Int4+top2 |
| **best_sparse** | Delta 用 sparse+Int4, Unigram 用 prev-token+Int4+top2 |
| **ultra_compress** | 全路径 Int2+top8 |

---

## 3. 实验结果

### 3.1 Round 1: 全策略对比

| 排名 | 策略 | 压缩率 | vs 基线 | 平均余弦 | 最低余弦 | 可行性 |
|------|------|--------|---------|---------|---------|--------|
| 1 | **combined** | **3.736×** | **+67.9%** | 0.99464 | 0.897 | ⚠️ 质量偏低 |
| 2 | **sparse_prev** | **3.560×** | **+60.0%** | 0.99726 | 0.992 | ✅ 推荐 |
| 3 | **prev_token** | **3.552×** | **+59.7%** | 0.99728 | 0.993 | ✅ 推荐 |
| 4 | unigram_int4_k4 | 3.309× | +48.7% | 0.71653 | 0.028 | ❌ 质量不可接受 |
| 5 | unigram_int4_k8 | 2.988× | +34.3% | 0.86140 | 0.700 | ❌ 质量不可接受 |
| 6 | int2_topk4 | 2.387× | +7.3% | 0.99579 | 0.946 | ⚠️ 质量有下降 |
| 7 | int2_topk8 | 2.298× | +3.3% | 0.99735 | 0.969 | ⚠️ |
| 8 | adaptive_bw | 2.286× | +2.7% | 0.99869 | 0.966 | ⚠️ |
| 9 | sparse_010 | 2.229× | +0.2% | 0.99958 | 0.992 | ✅ 微弱提升 |
| 10 | linear_pred | 2.229× | +0.2% | 0.99957 | 0.994 | ✅ 微弱提升 |
| 11 | sparse_005 | 2.229× | +0.2% | 0.99959 | 0.994 | ✅ |
| 12 | **baseline** | **2.225×** | — | 0.99960 | 0.994 | ✅ |

### 3.2 Round 2: Prev-Token 变体精细对比

| 排名 | 策略 | 压缩率 | vs 基线 | 余弦 (pos>0) | 余弦 min (pos>0) |
|------|------|--------|---------|-------------|-----------------|
| 1 | prev_int2_k4 | 4.567× | +105% | 0.958 | — |
| 2 | ultra_compress | 4.255× | +91% | 0.964 | — |
| 3 | prev_int2_k8 | 3.971× | +79% | 0.970 | — |
| 4 | prev_gs256 | 3.621× | +63% | 0.985 | — |
| 5 | **prev_int4_k1** | **3.552×** | **+60%** | **0.996** | **0.992** |
| 6 | **best_combined** | **3.454×** | **+55%** | **0.998** | **0.995** |
| 7 | **prev_int4_k2** | **3.454×** | **+55%** | **0.998** | **0.995** |
| 8 | mean_pool | 3.454× | +55% | 0.998 | 0.995 |
| 9 | prev_int4_k4 | 3.273× | +47% | 0.998 | — |
| 10 | baseline | 2.225× | — | 1.000 | 0.996 |

### 3.3 逐 Tier 字节与质量分析 (Position 级别)

#### Unigram 位置 (瓶颈路径, pos > 0):

| 策略 | 字节/位置 | 余弦均值 | 余弦最低 | vs Int8 节省 |
|------|----------|---------|---------|-------------|
| prev_int2_k4 | 1,939 B | 0.954 | — | -64.1% |
| prev_int2_k8 | 2,417 B | 0.970 | — | -55.2% |
| prev_gs256 | 2,776 B | 0.990 | — | -48.6% |
| **prev_int4_k1** | **2,855 B** | **0.995** | **0.992** | **-47.1%** |
| **prev_int4_k2** | **2,975 B** | **0.998** | **0.995** | **-44.9%** |
| mean_pool | 2,975 B | 0.993 | — | -44.9% |
| prev_int4_k4 | 3,214 B | 0.994 | — | -40.5% |
| **baseline (Int8)** | **5,400 B** | **0.996** | — | — |

#### Trigram/Bigram 位置 (已优化路径):

| 策略 | 字节/位置 | 余弦均值 | 变化 |
|------|----------|---------|------|
| ultra_compress | 2,412 B | 0.991-0.994 | 字节减少但质量下降 |
| **baseline (Int4)** | **2,852 B** | **0.999** | 已接近最优 |
| sparse (5%) | 2,857 B | 0.999 | 几乎无差异 |

### 3.4 按数据集细分结果 (Round 1, prev_token)

| 数据集 | 基线 CR | prev_token CR | 提升 | prev_token 余弦 |
|--------|---------|--------------|------|----------------|
| GSM8k | 2.574× | 3.575× | +38.9% | 0.99800 |
| CNN/DM | 2.197× | 3.591× | +63.4% | 0.99716 |
| TriviaQA | 2.201× | 3.566× | +62.0% | 0.99722 |
| WikiText-2 | 2.205× | 3.519× | +59.6% | 0.99722 |
| ShareGPT | 2.119× | 3.532× | +66.7% | 0.99709 |
| Alpaca | 2.054× | 3.531× | +71.9% | 0.99698 |

### 3.5 投射到完整 Pipeline 系统

将 prev-token 策略应用于原始 Pipeline 系统 (含 trigram/自引用等完整分类):

| 配置 | 压缩率 | vs 原始 2.521× | 余弦 (pos>0) |
|------|--------|---------------|-------------|
| 原始系统 | 2.521× | — | 0.99971 |
| **+ prev_int4_k1** | **3.586×** | **+42.3%** | ~0.996 |
| **+ prev_int4_k2** | **3.521×** | **+39.7%** | ~0.998 |

---

## 4. 各策略详细分析

### 4.1 ✅ 前一 Token 参考 (Previous-Token Reference) — **强烈推荐**

**原理:**  
对于 Unigram 位置 (无 n-gram 表匹配), 使用前一个 token 的 hidden state 作为参考,计算 affine 变换后的 delta,再用 Int4 量化。

$$
\text{pred} = \alpha \cdot h_{t-1} + \beta, \quad \delta = h_t - \text{pred}, \quad \text{transmit:}\ Q_{\text{Int4}}(\delta)
$$

**为什么有效:**  
相邻 token 的 hidden state 具有高度相关性。即使 token 类型不同,layer-6 的特征表示包含丰富的上下文信息,相邻位置共享大量特征。Affine 变换可以捕获尺度差异,Int4 delta 量化捕获残差细节。

**结果:**

| 变体 | 字节/位置 | 压缩率 | 余弦 | 推荐度 |
|------|----------|--------|------|--------|
| prev_int4_k1 | 2,855 B | 3.59× | 0.996 | ⭐⭐⭐⭐ |
| prev_int4_k2 | 2,975 B | 3.52× | 0.998 | ⭐⭐⭐⭐⭐ |
| prev_int4_k4 | 3,214 B | 3.27× | 0.998 | ⭐⭐⭐ |

**推荐配置: `prev_int4_k2`** — 压缩率 3.52×, 余弦 0.998, 与原始 Delta 路径质量持平。

**实现代价:**
- 发送端: 额外一次 affine 计算 (~10μs) + Int4 量化 (已有)
- 接收端: 需保留前一 token 的重建 hidden (仅 10KB 额外内存)
- 无需修改传输协议 (复用 DeltaPacket)
- 不需要 ref_indices (隐式指向 prev, 节省 8 bytes)

### 4.2 ⚠️ Int2 Delta 量化 — **有条件推荐**

**原理:**  
将 delta 路径的量化从 Int4 (16 levels) 降为 Int2 (4 levels), 用更多 top-k outlier 补偿精度损失。

$$
\text{data bytes}: 5120 / 4 = 1280\ B \quad (\text{vs Int4:}\ 5120 / 2 = 2560\ B)
$$

**结果:**

| 变体 | 数据字节 | 总字节/位置 | 余弦 |
|------|---------|-----------|------|
| Int4 top1 (基线) | 2,560 B | 2,852 B | 0.999 |
| Int2 top4 | 1,280 B | 2,280 B | 0.996 |
| Int2 top8 | 1,280 B | 2,600 B | 0.997 |

**分析:**  
- Int2 top4 节省了 20% 的 delta 路径字节, 但余弦下降 0.003
- 对于 trigram 路径 (delta 本就很小), Int2 可能足够
- **适用场景:** 带宽极度受限, 可容忍 cos < 0.997

### 4.3 ❌ 裸 Int4 Unigram — **不推荐**

**原理:**  
直接用 Int4 量化原始激活 (无参考), 用更多 outlier 补偿。

**结果:**  
- `unigram_int4_k4`: 余弦仅 0.716 — **完全不可用**
- `unigram_int4_k8`: 余弦仅 0.861 — 仍不可接受

**结论:** 5120 维 hidden state 的值域范围太大, Int4 的 16 个量化级别远远不够。**必须有参考才能用 Int4。**

### 4.4 ❌ 稀疏组传输 — **收益极小**

**原理:**  
用 5 字节 bitmask 标记哪些 group 的 delta 能量足够大需要传输, 跳过能量小的组。

**结果:** 仅 +0.2% 压缩提升。

**原因:** Affine 变换已经非常有效地对齐了参考和目标, delta 的能量分布相对均匀, 很少有组可以完全跳过。

### 4.5 ❌ 学习式线性预测器 — **收益极小**

**原理:**  
用逐通道线性回归 $\hat{h}_i = w_i \cdot r_i + b_i$ 替代逐样本的最小二乘 affine。节省 per-sample 的 affine_scale 和 affine_bias (4 bytes), 但需要预训练。

**结果:** 仅 +0.2% 压缩提升, 且需要额外的训练流程。

**结论:** 逐样本 Affine 已经是近似最优的线性变换, 全局线性模型无法显著减小 delta。

### 4.6 ⚠️ 均值池化参考 — **不如 prev-token**

**原理:**  
用最近 4 个 token 的均值作为 unigram 位置的参考。

**结果:** 与 `prev_int4_k2` 相同压缩率 (3.45×), 余弦略低 (0.993 vs 0.998)。

**结论:** 均值模糊了特征细节, 单个前一 token 反而是更好的参考。

---

## 5. 推荐方案

### 5.1 推荐方案 A: prev_int4_k2 (高质量优先)

```
分类流程:
  pos 0: Int8 + top1 (无参考, 5400 B)
  pos 1+:
    ├── Trigram match → Affine + Int4 + top1 (2852 B)   ← 不变
    ├── Bigram match  → Affine + Int4 + top1 (2852 B)   ← 不变
    ├── Self-Ref      → Affine + Int4 + top1 (2852 B)   ← 不变
    └── Unigram       → prev-token Affine + Int4 + top2 (2975 B) ← 新增!
```

| 指标 | 原始系统 | 推荐方案 A | 变化 |
|------|---------|-----------|------|
| 压缩率 | 2.521× | **3.521×** | **+39.7%** |
| 余弦 (pos>0) | 0.9997 | ~0.998 | -0.002 |
| 每 token 字节 | ~4,660 B | ~2,970 B | -36.3% |

### 5.2 推荐方案 B: prev_int4_k1 (最高压缩)

```
  └── Unigram → prev-token Affine + Int4 + top1 (2855 B)
```

| 指标 | 原始系统 | 推荐方案 B | 变化 |
|------|---------|-----------|------|
| 压缩率 | 2.521× | **3.586×** | **+42.3%** |
| 余弦 (pos>0) | 0.9997 | ~0.996 | -0.004 |
| 每 token 字节 | ~4,660 B | ~2,855 B | -38.7% |

### 5.3 可选增强: 自适应 Bitwidth

在方案 A/B 基础上, 对 Trigram 路径 (delta 最小) 使用 Int2+top4:

```
  ├── Trigram match → Affine + Int2 + top4 (2280 B)   ← 节省 20%
  ├── Bigram/Self   → Affine + Int4 + top1 (2852 B)   ← 不变
  └── Unigram       → prev-token + Int4 + top2 (2975 B)
```

预计压缩率: ~3.7×, 余弦略有下降 (~0.996)。

---

## 6. Pareto 前沿分析

```
余弦相似度
  1.000 ┤ ● baseline (2.23×)
        │   ● linear_pred (2.23×)
  0.999 ┤     ● sparse_005 (2.23×)
        │           ● adaptive_bw (2.29×)
  0.998 ┤              ★ prev_int4_k2 (3.45×)  ← Pareto 最优
        │               ★ best_combined (3.45×)
  0.997 ┤                  ★ prev_int4_k1 (3.55×)  ← Pareto 最优
        │                      ● int2_topk8 (2.30×)
  0.996 ┤                        ● int2_topk4 (2.39×)
        │
  0.995 ┤                              ● combined (3.74×)
        │
  0.990 ┤                                ● prev_gs256 (3.62×)
        │
  0.970 ┤                                    ● prev_int2_k8 (3.97×)
        │
  0.960 ┤                                       ● prev_int2_k4 (4.57×)
        │
        └─────────────────────────────────────────────── 压缩率 →
          2.0    2.5    3.0    3.5    4.0    4.5    5.0
```

**Pareto 前沿上的策略:**
1. `baseline` (2.23×, cos=0.9996) — 最高质量
2. `prev_int4_k2` (3.45×, cos=0.998) — **最佳质量-压缩平衡** ★
3. `prev_int4_k1` (3.55×, cos=0.996) — **最高可用压缩** ★
4. `prev_int2_k8` (3.97×, cos=0.970) — 激进但可能可用
5. `prev_int2_k4` (4.57×, cos=0.958) — 极端压缩

---

## 7. 实现建议

### 7.1 最小改动实现 (prev_int4_k2)

需修改文件:
1. **`pipeline.py`** — 在编码阶段:
   - Unigram 位置不再直接 Int8 编码
   - 改为使用前一位置的 hidden state 计算 affine + Int4 delta
   - 保留前一位置的重建结果用于下一位置

2. **`codec.py`** — 无需修改 (复用现有 Int4 函数)

3. **`table.py`** — 无需修改

**关键代码变更 (伪代码):**

```python
# pipeline.py 编码阶段
prev_recon_h = None
for pos in range(seq_len):
    if tier[pos] in ("trigram", "bigram", "self_ref"):
        # 现有 delta 路径, 不变
        recon, pkt = encode_delta(real_h[pos], ref_h[pos])
    elif prev_recon_h is not None:
        # 新增: 使用前一 token 的重建 hidden 作为参考
        recon, pkt = encode_delta(real_h[pos], prev_recon_h, top_k=2)
    else:
        # pos=0, 无参考
        recon, pkt = encode_int8(real_h[pos])
    prev_recon_h = recon
```

### 7.2 接收端变更

接收端需要维护前一 token 的重建结果:

```python
prev_recon = None
for pos in range(seq_len):
    if pkt.has_ref:
        ref = table.lookup(pkt.ref_indices) if pkt.ref_indices >= 0 else prev_recon
        recon = decode_delta(pkt, ref)
    else:
        recon = decode_int8(pkt)
    prev_recon = recon
```

### 7.3 Decode 阶段的特殊处理

Decode 阶段每步只处理一个 token, prev_recon 自然就是上一步的重建结果, 无需额外缓存。

---

## 8. 总结

| 维度 | 原始系统 | + Prev-Token (推荐) | 变化 |
|------|---------|-------------------|------|
| 平均压缩率 | 2.52× | **3.52×** | **+39.7%** |
| 每 token 字节 | ~4,660 B | ~2,970 B | **-36.3%** |
| 余弦相似度 | 0.9997 | ~0.998 | -0.002 |
| 实现复杂度 | — | 低 (仅改 pipeline.py) | — |
| 额外延迟 | — | ~10μs/token | 可忽略 |
| 额外内存 | — | 10 KB (一个 hidden) | 可忽略 |

**关键发现:**

1. **前一 Token 参考 (prev-token reference) 是最有效的优化**, 将 Unigram 路径从 5,400 B 降至 2,975 B, 减少 44.9%
2. **稀疏组传输和学习式预测器收益极小** (<0.2%), 不值得引入复杂度
3. **裸 Int4 量化 (无参考) 不可行**, 激活向量值域太大, 必须有参考做 delta
4. **Int2 量化可作为进阶优化**, 在 Trigram 路径上使用可再提升 ~5%, 但质量有所下降
5. 前一 token 参考之所以有效, 是因为 **LLM 中间层的 hidden state 具有极高的时序连续性** — 这是 Delta-Coding 系统与 LLM 特性的深度耦合

---

## 附录 A: 实验数据文件

| 文件 | 内容 |
|------|------|
| `results_compression_exp/compression_strategies.parquet` | Round 1: 12 种策略 × 6 数据集 × 30 requests |
| `results_compression_exp_v2/strategies_v2.parquet` | Round 2: 11 种变体 × 6 数据集 × 30 requests |
| `results_compression_exp_v2/position_detail.parquet` | 逐位置质量分析 (前 3 requests/数据集) |

## 附录 B: 字节计算公式

### Int4 Delta 路径

$$
\text{bytes} = \underbrace{\frac{H}{2}}_{\text{packed}} + \underbrace{G \times 2}_{\text{scales}} + \underbrace{G \times 2}_{\text{zeros}} + \underbrace{G \times k \times 2}_{\text{outlier vals}} + \underbrace{G \times k \times 1}_{\text{outlier idx}} + \underbrace{2 + 2 + 8}_{\text{affine + ref}}
$$

其中 $H = 5120$, $G = 40$ (groups), $k = \text{top\_k}$。

| top_k | Delta 路径字节 | Prev-token 路径字节 (无 ref_idx) |
|-------|--------------|-------------------------------|
| 1 | 2,852 B | 2,844 B |
| 2 | 2,972 B | 2,964 B |
| 4 | 3,212 B | 3,204 B |

### 压缩率计算

$$
\text{CR} = \frac{S \times H \times 2}{\sum_{i} \text{bytes}(i)}
$$

其中 $S$ 为序列长度, 分母对每个位置按分类路径累加字节。
