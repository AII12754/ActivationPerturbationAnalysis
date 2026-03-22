# Delta-Coding System

用于 pipeline-parallel LLM 中间激活压缩的核心系统目录。

当前 root 目录只保留生产运行和基准评测所需的核心代码。历史探索脚本、实验报告和设计草稿已整理到 experiments 子目录。

## 当前默认策略

真实 pipeline 默认固定为 optimized_default:

1. Delta 路径: delta_noaffine_int4_k1
2. Unigram 路径: unigram_int4_k4

`OverlappedPipeline` 默认构造参数已经切换到这套配置,后续主流程不再默认使用 baseline_current。

## 核心目录

```text
delta_coding_system/
├── __init__.py
├── README.md
├── table.py
├── codec.py
├── pipeline.py
├── evaluation.py
├── run_experiment.py
├── run_pipeline_strategy_real.py
├── analyze.py
├── benchmarks/
├── experiments/
└── report/
```

说明:

1. table.py: n-gram DAG 表与缓存管理
2. codec.py: 激活编码与重建逻辑
3. pipeline.py: 真实 OverlappedPipeline 运行时
4. evaluation.py: 评测辅助函数,包括 logit drift 与后半层 replay
5. run_experiment.py: 核心系统实验入口,默认走 optimized_default
6. run_pipeline_strategy_real.py: 真实 pipeline 基准评测入口,输出压缩率、通信时延、相似度和相对 FP16 的 drift
7. benchmarks/: 后续任务基准说明,包括 SWE-Bench Verified 之类任务评测的建议流程
8. experiments/: 历史探索性实验与报告归档

## 架构概览

```text
Sender (PP Stage 0)                          Receiver (PP Stage 1)
┌─────────────────────────────────┐          ┌───────────────────────┐
│ ① Prefill forward (GPU)        │          │                       │
│ ② Classify (CPU, overlapped)   │ ───────→ │ ⑤ Decode + reconstruct│
│ ③ Tiered encode (GPU)          │          │                       │
│ ④ Table update (CPU, async)    │          │                       │
└─────────────────────────────────┘          └───────────────────────┘
```

Tier fallback:

1. Trigram
2. Self-ref
3. Bigram
4. Unigram

当前实现重点:

1. prefill 和 decode 均支持真实 overlap
2. decode 使用 KV Cache
3. 默认压缩路径已经切到 optimized_default
4. benchmark 输出内置 FP16 通信基线字段,便于后续直接对照

## 快速开始

### 1. 运行核心系统实验

```bash
python -m delta_coding_system.run_experiment \
  --gpu 1 \
  --datasets gsm8k triviaqa \
  --warmup-requests 10 \
  --test-requests 10
```

### 2. 在新数据集上运行

`run_experiment.py` 和 `run_pipeline_strategy_real.py` 都支持两类数据源:

1. 内置数据集名字,例如 `gsm8k`、`wikitext2`
2. 自定义路径,例如 parquet/json/jsonl/txt 文件或目录

示例:

```bash
python -m delta_coding_system.run_experiment \
  --gpu 1 \
  --datasets /path/to/my_dataset.parquet

python delta_coding_system/run_pipeline_strategy_real.py \
  --gpu 1 \
  --datasets /path/to/my_dataset.jsonl \
  --warmup-requests 20 \
  --test-requests 100 \
  --max-decode-tokens 512
```

通用路径加载的文本提取启发式:

1. 优先使用 `text`、`prompt`、`content` 等单字段文本
2. 若存在 `instruction/input/output`、`question/answer`、`prompt/completion` 等组合字段,则自动拼接
3. txt 文件按空行切分样本

## 真实基准评测

主入口:

[delta_coding_system/run_pipeline_strategy_real.py](delta_coding_system/run_pipeline_strategy_real.py)

这个脚本默认评测固定生产策略 optimized_default,并输出:

1. request 级压缩率
2. request 级真实 pipeline 时延
3. 200/500/1000 Mbps 下的通信端到端时延
4. 原生 FP16 通信基线
5. decode 相似度
6. 相对原模型 FP16 logits 的 drift 指标

示例:

```bash
python delta_coding_system/run_pipeline_strategy_real.py \
  --gpu 1 \
  --datasets /path/to/my_dataset.parquet \
  --warmup-requests 30 \
  --test-requests 100 \
  --max-decode-tokens 512 \
  --output-dir results_pipeline_strategy_real_run
```

输出文件:

1. `request_summary.parquet`
2. `decode_drift.parquet`

## 与原模型 FP16 的对比方式

当前系统内置两类对比:

1. 通信侧对比: 直接使用 `raw_fp16_bytes` 推导出 FP16 传输基线
2. 模型侧对比: 使用原模型 FP16 logits 作为 drift 参考

这意味着你在新数据集上测试时,不用再额外重跑一套单独的通信基线实验。

## SWE-Bench Verified 等任务评测

对于 SWE-Bench Verified、代码修复、问答评分等任务级基准,建议采用两层评测:

1. 模型内部指标:
   使用 `run_pipeline_strategy_real.py` 导出压缩率、通信时延、decode similarity、drift
2. 任务分数指标:
   使用同一批 prompt 对 compressed pipeline 和原模型 FP16 分别生成输出,再交给任务自己的评分 harness

原因:

1. 内部 drift 只能说明表示和 logits 的偏移程度
2. 它不能替代 SWE-Bench Verified 这类 benchmark 的最终任务得分

具体建议见:

[delta_coding_system/benchmarks/README.md](delta_coding_system/benchmarks/README.md)

## 历史探索归档

历史探索代码与报告已移动到:

1. [delta_coding_system/experiments](delta_coding_system/experiments)
2. [delta_coding_system/experiments/reports](delta_coding_system/experiments/reports)

这些文件用于追溯策略选择过程,但不再是默认生产流程的一部分。
