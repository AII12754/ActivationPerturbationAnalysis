# Benchmarking Notes

## 当前可直接跑分的 benchmark

现阶段最适合直接接入并跑正式分数的是 HumanEval。

原因:
1. 数据集中自带 prompt、unit tests 和 entry point
2. 评分是明确的 pass@1,不需要额外 judge model
3. 不需要像 SWE-Bench Verified 那样为每题准备独立仓库环境和测试重放框架

现成脚本:

1. [delta_coding_system/benchmarks/run_humaneval_benchmark.py](delta_coding_system/benchmarks/run_humaneval_benchmark.py)

这个脚本只输出你关心的指标:

1. compressed system 最终分数
2. FP16 baseline 最终分数
3. compressed system 的 prefill/decode 总通信 e2e 时延
4. FP16 baseline 的 prefill/decode 总通信 e2e 时延
5. compressed system 的 prefill/decode/overall 通信量压缩百分比

示例:

```bash
python delta_coding_system/benchmarks/run_humaneval_benchmark.py \
	--gpu 1 \
	--bandwidth-mbps 200 \
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

状态: 数据现成,但还没有接入统一评分脚本

原因:
1. 各子任务 metric 不同,包括 EM、F1、ROUGE、检索准确率等
2. 更适合在 HumanEval 跑通后,再做统一 task-specific scorer

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
