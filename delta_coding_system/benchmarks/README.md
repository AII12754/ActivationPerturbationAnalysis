# Benchmarking Notes

## 当前可直接跑分的 benchmark

现阶段最适合直接接入并跑正式分数的是 HumanEval。

原因:
1. 数据集中自带 prompt、unit tests 和 entry point
2. 评分是明确的 pass@1,不需要额外 judge model
3. 不需要像 SWE-Bench Verified 那样为每题准备独立仓库环境和测试重放框架

现成脚本:

1. [delta_coding_system/benchmarks/run_humaneval_benchmark.py](delta_coding_system/benchmarks/run_humaneval_benchmark.py)

这个脚本只输出你关心的指标,并且现在可以一次对比多组压缩配置,包括 delta 路径和 unigram 路径策略:

1. compressed system 最终分数
2. FP16 baseline 最终分数
3. compressed system 的 prefill/decode 总通信 e2e 时延
4. FP16 baseline 的 prefill/decode 总通信 e2e 时延
5. compressed system 的 prefill/decode/overall 通信量压缩百分比

当前默认行为:
1. 默认不 warmup
2. 默认只输出最终分数
3. 默认不跑 FP16 baseline
4. 只有显式传 `--no-score-only` 或 `--include-fp16-baseline` 时才会回到更慢的完整统计模式

示例:

```bash
python delta_coding_system/benchmarks/run_humaneval_benchmark.py \
	--gpu 4 \
	--bandwidth-mbps 200 \
	--config optimized_default delta_noaffine_int2_k8_unigram_int4 prev_int4_k2 \
	--max-samples 164 \
	--output results_humaneval_benchmark/summary.json
```

## 其他 benchmark 的接入难度

### HumanEval

状态: 现成可跑

输出:
1. pass@1
2. 与 FP16 baseline 的分数对比
3. prefill/decode 总通信 e2e 时延对比

### MT-Bench

状态: 不建议作为第一优先级

原因:
1. 需要 judge model 或现成评审协议
2. 分数不是数据集内置单元测试直接算出来的
3. judge 选择会显著影响分数

### SWE-Bench Verified

状态: 不是现成可跑

原因:
1. 需要为每道题恢复指定 repo 和 base commit
2. 需要应用候选 patch 或生成代码修改
3. 需要运行题目对应测试集
4. 需要完整任务 harness,不是只读 prompt 后生成文本就能评分

### LongBench / InfiniteBench / LooGLE / L-eval

状态: LongBench 英文可自动判分子集现已接入,其他长文本 benchmark 仍未统一接入

原因:
1. 各子任务 metric 不同,包括 EM、F1、ROUGE、检索准确率等
2. 当前已优先接入最容易稳定自动评分的英文子任务:
	- `hotpotqa_e`
	- `2wikimqa_e`
	- `musique`
	- `qasper`
	- `triviaqa_e`
	- `passage_retrieval_en_e`
	- `passage_count_e`
3. 现已补充可自动判分的长上下文任务:
	- `narrativeqa` (`qa_f1`)
	- `qmsum` (`rouge_l`)
	- `gov_report` (`rouge_l`)
	- `passage_count` (`count_exact`)
4. 现在也支持把 `fp16_baseline` 当作 LongBench 对比配置直接一起跑

现成脚本:

1. [delta_coding_system/benchmarks/run_longbench_benchmark.py](delta_coding_system/benchmarks/run_longbench_benchmark.py)

默认特点:
1. 默认不 warmup
2. 默认只输出最终分数
3. 默认跑英文可自动判分子集
4. 默认输出 overall 平均分和分 task 分数

示例:

```bash
python delta_coding_system/benchmarks/run_longbench_benchmark.py \
	--gpu 5 \
	--config fp16_baseline delta_noaffine_int2_k2_unigram_int4 \
	--tasks narrativeqa musique qmsum gov_report passage_count \
	--samples-per-task 20 \
	--output results_longbench/summary_gpu5.json
```

## 核心 benchmark 入口

模型内部指标入口:

1. [delta_coding_system/run_pipeline_strategy_real.py](delta_coding_system/run_pipeline_strategy_real.py)

任务分数入口:

1. HumanEval: [delta_coding_system/benchmarks/run_humaneval_benchmark.py](delta_coding_system/benchmarks/run_humaneval_benchmark.py)

## 原则

1. 对评分类 benchmark,最终结论以任务分数为主
2. 通信 e2e 时延只保留 request 级总量即可
3. similarity、drift 等内部指标在这类 benchmark 上不是必须项
4. 后续如果接 SWE-Bench Verified,应单独做 task harness,而不是强行复用文本生成评测脚本
