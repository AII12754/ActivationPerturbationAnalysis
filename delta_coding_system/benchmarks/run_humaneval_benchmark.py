#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, cast

import pyarrow.parquet as pq
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from delta_coding_system.pipeline import OverlappedPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("humaneval_benchmark")

DATASET_PATH = Path("/root/share/dataset/humaneval/openai_humaneval")
FP16_CONFIG_NAME = "fp16_baseline"

CONFIGS = {
    "optimized_default": {
        "delta_strategy": "delta_noaffine_int4_k1",
        "unigram_strategy": "unigram_int4_k4",
    },
    "delta_baseline_unigram_int4": {
        "delta_strategy": "baseline_current",
        "unigram_strategy": "unigram_int4_k4",
    },
    "delta_int2_k8_unigram_int4": {
        "delta_strategy": "delta_int2_k8_out4",
        "unigram_strategy": "unigram_int4_k4",
    },
    "delta_noaffine_int2_k8_unigram_int4": {
        "delta_strategy": "delta_noaffine_int2_k8_out4",
        "unigram_strategy": "unigram_int4_k4",
    },
    "delta_int2_k4_unigram_int4": {
        "delta_strategy": "delta_int2_k4_out4",
        "unigram_strategy": "unigram_int4_k4",
    },
    "delta_noaffine_int2_k4_unigram_int4": {
        "delta_strategy": "delta_noaffine_int2_k4_out4",
        "unigram_strategy": "unigram_int4_k4",
    },
    "delta_int2_k2_unigram_int4": {
        "delta_strategy": "delta_int2_k2_out4",
        "unigram_strategy": "unigram_int4_k4",
    },
    "delta_noaffine_int2_k2_unigram_int4": {
        "delta_strategy": "delta_noaffine_int2_k2_out4",
        "unigram_strategy": "unigram_int4_k4",
    },
    "delta_int2_k8_entropy_unigram_int4": {
        "delta_strategy": "delta_int2_k8_out4_entropy",
        "unigram_strategy": "unigram_int4_k4",
    },
    "delta_int2_k4_entropy_unigram_int4": {
        "delta_strategy": "delta_int2_k4_out4_entropy",
        "unigram_strategy": "unigram_int4_k4",
    },
    "legacy_prev_int4_k2": {
        "delta_strategy": "delta_noaffine_int4_k1",
        "unigram_strategy": "prev_int4_k2",
    },
    "legacy_prev_gs256_k2": {
        "delta_strategy": "delta_noaffine_int4_k1",
        "unigram_strategy": "prev_gs256_k2",
    },
}

LEGACY_CONFIG_NAMES = (
    "legacy_prev_int4_k2",
    "legacy_prev_gs256_k2",
)


def _network_ms(total_bytes: float, bandwidth_mbps: int) -> float:
    return float(total_bytes) * 8.0 / (float(bandwidth_mbps) * 1000.0)


def _load_humaneval_records() -> List[Dict[str, str]]:
    files = sorted(DATASET_PATH.glob("test-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No HumanEval parquet files found under {DATASET_PATH}")
    records: List[Dict[str, str]] = []
    for file_path in files:
        records.extend(pq.read_table(file_path).to_pylist())
    return records


def _clean_completion(text: str) -> str:
    sentinels = ["\nclass ", "\ndef ", "\nif __name__", "\n```", "\n# Example"]
    end = len(text)
    for marker in sentinels:
        idx = text.find(marker)
        if idx != -1:
            end = min(end, idx)
    return text[:end].rstrip()


def _decode_generated(tokenizer, generated_ids: List[int]) -> str:
    if not generated_ids:
        return ""
    return _clean_completion(tokenizer.decode(generated_ids, skip_special_tokens=True))


def _score_candidate(prompt: str, completion: str, test_code: str, entry_point: str, timeout_sec: int) -> bool:
    program = "\n".join([
        prompt.rstrip(),
        completion.rstrip(),
        "",
        test_code.rstrip(),
        "",
        f"check({entry_point})",
        "",
    ])
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as handle:
        tmp_path = Path(handle.name)
        handle.write(program)
    try:
        completed = subprocess.run(
            [sys.executable, str(tmp_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
            check=False,
            text=True,
        )
        return completed.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        tmp_path.unlink(missing_ok=True)


def _build_pipeline(model, tokenizer, device: torch.device, args, config_name: str) -> OverlappedPipeline:
    cfg = CONFIGS[config_name]
    return OverlappedPipeline(
        model=model,
        tokenizer=tokenizer,
        layer_boundary=args.layer_boundary,
        table_dtype=torch.float16,
        max_table_entries=args.max_table_entries,
        group_size=args.group_size,
        top_k=1,
        int8_group_size=args.group_size,
        int8_outlier_top_k=1,
        decode_tokens=args.max_new_tokens,
        max_seq_len=args.max_seq_len,
        device=device,
        domain_aware=False,
        delta_strategy=cfg["delta_strategy"],
        unigram_strategy=cfg["unigram_strategy"],
        track_transfer_bytes=not args.score_only,
    )


def _reduction_pct(compressed_bytes: int, raw_bytes: int) -> float:
    if raw_bytes <= 0:
        return 0.0
    return (1.0 - (compressed_bytes / raw_bytes)) * 100.0


def _write_summary(output_path: str, summary: Dict[str, object]) -> None:
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))


def _load_existing_summary(output_path: str) -> Dict[str, object] | None:
    out_path = Path(output_path)
    if not out_path.exists():
        return None
    return json.loads(out_path.read_text())


def _effective_strategy_payload(config_name: str) -> Dict[str, object]:
    if config_name == FP16_CONFIG_NAME:
        return {
            "mode": "fp16",
            "delta_strategy": None,
            "unigram_strategy": None,
        }
    cfg = CONFIGS[config_name]
    return {
        "mode": "compressed",
        "delta_strategy": cfg["delta_strategy"],
        "unigram_strategy": cfg["unigram_strategy"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HumanEval benchmark for compressed pipeline vs FP16 baseline")
    parser.add_argument("--model", default="/root/share/models/Qwen2.5-32B-Instruct")
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--config", choices=sorted([*CONFIGS.keys(), FP16_CONFIG_NAME]), nargs="+", default=["optimized_default"])
    parser.add_argument("--max-samples", type=int, default=164)
    parser.add_argument("--warmup-samples", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--layer-boundary", type=int, default=6)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--max-table-entries", type=int, default=100000)
    parser.add_argument("--bandwidth-mbps", type=int, default=200)
    parser.add_argument("--exec-timeout-sec", type=int, default=10)
    parser.add_argument("--output", default="results_humaneval_benchmark/summary.json")
    parser.add_argument("--save-per-sample", action="store_true")
    parser.add_argument("--score-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-fp16-baseline", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    from transformers.models.auto.modeling_auto import AutoModelForCausalLM
    from transformers.models.auto.tokenization_auto import AutoTokenizer

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map={"": device},
        trust_remote_code=True,
    )
    model.eval()

    all_records = _load_humaneval_records()
    records = all_records[: args.max_samples]
    warmup_records = records[: min(args.warmup_samples, len(records))]
    test_records = records[min(args.warmup_samples, len(records)):]

    hidden_dim = model.config.hidden_size
    fp16_pass = 0
    fp16_prefill_comm_ms = 0.0
    fp16_decode_comm_ms = 0.0
    fp16_prefill_bytes = 0
    fp16_decode_bytes = 0
    results_by_config: Dict[str, Dict[str, object]] = {}
    fp16_initialized = False
    existing_summary = _load_existing_summary(args.output) if args.resume else None
    existing_configs: Dict[str, Any] = {}
    if existing_summary is not None:
        maybe_configs = existing_summary.get("configs", {})
        if isinstance(maybe_configs, dict):
            existing_configs = maybe_configs

    logger.info("Scoring %d HumanEval samples for %d configs...", len(test_records), len(args.config))

    for config_name in args.config:
        existing_result = existing_configs.get(config_name)
        if isinstance(existing_result, dict) and existing_result.get("completed"):
            results_by_config[config_name] = existing_result
            logger.info("Skipping config=%s because completed result already exists in %s", config_name, args.output)
            continue

        pipeline = None if config_name == FP16_CONFIG_NAME else _build_pipeline(model, tokenizer, device, args, config_name)
        logger.info("Warmup %d HumanEval samples for config=%s...", len(warmup_records), config_name)
        if pipeline is not None:
            for row in warmup_records:
                pipeline.process_request(row["prompt"], phase="warmup")

        compressed_pass = 0
        compressed_prefill_comm_ms = 0.0
        compressed_decode_comm_ms = 0.0
        compressed_prefill_bytes = 0
        compressed_decode_bytes = 0
        compressed_prefill_raw_bytes = 0
        compressed_decode_raw_bytes = 0
        per_sample: List[Dict[str, object]] = []
        start_index = 0

        if isinstance(existing_result, dict) and not existing_result.get("completed"):
            compressed_pass = int(existing_result.get("passed", 0))
            compressed_prefill_comm_ms = float(existing_result.get("prefill_comm_e2e_total_ms", 0.0))
            compressed_decode_comm_ms = float(existing_result.get("decode_comm_e2e_total_ms", 0.0))
            compressed_prefill_bytes = int(existing_result.get("prefill_transfer_bytes", 0))
            compressed_decode_bytes = int(existing_result.get("decode_transfer_bytes", 0))
            compressed_prefill_raw_bytes = int(existing_result.get("prefill_fp16_raw_bytes", 0))
            compressed_decode_raw_bytes = int(existing_result.get("decode_fp16_raw_bytes", 0))
            if args.save_per_sample:
                per_sample = list(existing_result.get("per_sample", []))
            start_index = len(per_sample) if args.save_per_sample else int(existing_result.get("num_samples", 0))
            logger.info("Resuming config=%s from sample %d", config_name, start_index)

        for idx, row in enumerate(test_records[start_index:], start=start_index + 1):
            prompt = row["prompt"]
            test_code = row["test"]
            entry_point = row["entry_point"]

            if pipeline is None:
                encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_seq_len).to(device)
                with torch.no_grad():
                    generated = model.generate(
                        **encoded,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                generated_ids = generated[0, encoded.input_ids.shape[1]:].tolist()
                compressed_completion = _decode_generated(tokenizer, generated_ids)
                compressed_ok = _score_candidate(prompt, compressed_completion, test_code, entry_point, args.exec_timeout_sec)
            else:
                prefill_res, decode_res, _ = pipeline.process_request(prompt, phase="test")
                compressed_completion = _decode_generated(tokenizer, decode_res.generated_token_ids)
                compressed_ok = _score_candidate(prompt, compressed_completion, test_code, entry_point, args.exec_timeout_sec)
            compressed_pass += int(compressed_ok)

            sample_metrics: Dict[str, object] | None = None
            if args.save_per_sample:
                sample_metrics = {
                    "task_id": row["task_id"],
                    "compressed_pass": compressed_ok,
                }
            if not args.score_only:
                if pipeline is None:
                    prompt_len = int(encoded.input_ids.shape[1])
                    decode_len = len(generated_ids)
                    fp16_prefill_bytes_local = int(prompt_len * hidden_dim * 2)
                    fp16_decode_bytes_local = int(decode_len * hidden_dim * 2)
                    comp_prefill_comm = _network_ms(fp16_prefill_bytes_local, args.bandwidth_mbps)
                    comp_decode_comm = _network_ms(fp16_decode_bytes_local, args.bandwidth_mbps)
                    compressed_prefill_comm_ms += comp_prefill_comm
                    compressed_decode_comm_ms += comp_decode_comm
                    compressed_prefill_bytes += fp16_prefill_bytes_local
                    compressed_decode_bytes += fp16_decode_bytes_local
                    compressed_prefill_raw_bytes += fp16_prefill_bytes_local
                    compressed_decode_raw_bytes += fp16_decode_bytes_local
                else:
                    comp_prefill_local = prefill_res.classify_ms + prefill_res.encode_delta_ms + prefill_res.encode_self_ref_ms + prefill_res.encode_unigram_ms
                    comp_decode_local = decode_res.total_classify_ms + decode_res.total_encode_ms
                    comp_prefill_comm = comp_prefill_local + _network_ms(prefill_res.total_transfer_bytes, args.bandwidth_mbps)
                    comp_decode_comm = comp_decode_local + _network_ms(decode_res.total_transfer_bytes, args.bandwidth_mbps)
                    compressed_prefill_comm_ms += comp_prefill_comm
                    compressed_decode_comm_ms += comp_decode_comm
                    compressed_prefill_bytes += int(prefill_res.total_transfer_bytes)
                    compressed_decode_bytes += int(decode_res.total_transfer_bytes)
                    compressed_prefill_raw_bytes += int(prefill_res.raw_fp16_bytes)
                    compressed_decode_raw_bytes += int(decode_res.raw_fp16_bytes)
                if sample_metrics is not None:
                    sample_metrics.update({
                        "compressed_prefill_comm_ms": comp_prefill_comm,
                        "compressed_decode_comm_ms": comp_decode_comm,
                    })

            if args.include_fp16_baseline:
                encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_seq_len).to(device)
                with torch.no_grad():
                    generated = model.generate(
                        **encoded,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                fp16_generated_ids = generated[0, encoded.input_ids.shape[1]:].tolist()
                fp16_completion = _decode_generated(tokenizer, fp16_generated_ids)
                fp16_ok = _score_candidate(prompt, fp16_completion, test_code, entry_point, args.exec_timeout_sec)

                if not fp16_initialized:
                    fp16_pass += int(fp16_ok)
                    if not args.score_only:
                        prompt_len = int(encoded.input_ids.shape[1])
                        fp16_decode_len = len(fp16_generated_ids)
                        fp16_prefill_comm = _network_ms(prompt_len * hidden_dim * 2, args.bandwidth_mbps)
                        fp16_decode_comm = _network_ms(fp16_decode_len * hidden_dim * 2, args.bandwidth_mbps)
                        fp16_prefill_comm_ms += fp16_prefill_comm
                        fp16_decode_comm_ms += fp16_decode_comm
                        fp16_prefill_bytes += int(prompt_len * hidden_dim * 2)
                        fp16_decode_bytes += int(fp16_decode_len * hidden_dim * 2)

                if sample_metrics is not None:
                    sample_metrics["fp16_pass"] = fp16_ok

            if sample_metrics is not None:
                per_sample.append(sample_metrics)

            if idx % 10 == 0 or idx == len(test_records):
                partial_result: Dict[str, object] = {
                    "score_name": "pass@1",
                    "score": compressed_pass / max(idx, 1),
                    "passed": compressed_pass,
                    "num_samples": idx,
                    "effective_strategy": _effective_strategy_payload(config_name),
                    "completed": idx == len(test_records),
                }
                if not args.score_only:
                    compressed_total_bytes = compressed_prefill_bytes + compressed_decode_bytes
                    compressed_total_raw_bytes = compressed_prefill_raw_bytes + compressed_decode_raw_bytes
                    partial_result.update({
                        "prefill_comm_e2e_total_ms": compressed_prefill_comm_ms,
                        "decode_comm_e2e_total_ms": compressed_decode_comm_ms,
                        "prefill_transfer_bytes": compressed_prefill_bytes,
                        "decode_transfer_bytes": compressed_decode_bytes,
                        "total_transfer_bytes": compressed_total_bytes,
                        "prefill_fp16_raw_bytes": compressed_prefill_raw_bytes,
                        "decode_fp16_raw_bytes": compressed_decode_raw_bytes,
                        "total_fp16_raw_bytes": compressed_total_raw_bytes,
                        "prefill_comm_reduction_pct": _reduction_pct(compressed_prefill_bytes, compressed_prefill_raw_bytes),
                        "decode_comm_reduction_pct": _reduction_pct(compressed_decode_bytes, compressed_decode_raw_bytes),
                        "total_comm_reduction_pct": _reduction_pct(compressed_total_bytes, compressed_total_raw_bytes),
                    })
                if args.save_per_sample:
                    partial_result["per_sample"] = per_sample
                results_by_config[config_name] = partial_result
                partial_summary = {
                    "benchmark": "HumanEval",
                    "num_samples": len(test_records),
                    "score_only": args.score_only,
                    "include_fp16_baseline": args.include_fp16_baseline,
                    "configs": results_by_config,
                }
                if not args.score_only:
                    partial_summary["bandwidth_mbps"] = args.bandwidth_mbps
                _write_summary(args.output, partial_summary)
                logger.info(
                    "Config=%s processed %d/%d samples compressed_pass=%d",
                    config_name,
                    idx,
                    len(test_records),
                    compressed_pass,
                )

        fp16_initialized = True
        result: Dict[str, object] = {
            "score_name": "pass@1",
            "score": compressed_pass / max(len(test_records), 1),
            "passed": compressed_pass,
            "num_samples": len(test_records),
            "effective_strategy": _effective_strategy_payload(config_name),
            "completed": True,
        }
        if not args.score_only:
            compressed_total_bytes = compressed_prefill_bytes + compressed_decode_bytes
            compressed_total_raw_bytes = compressed_prefill_raw_bytes + compressed_decode_raw_bytes
            result.update({
                "prefill_comm_e2e_total_ms": compressed_prefill_comm_ms,
                "decode_comm_e2e_total_ms": compressed_decode_comm_ms,
                "prefill_transfer_bytes": compressed_prefill_bytes,
                "decode_transfer_bytes": compressed_decode_bytes,
                "total_transfer_bytes": compressed_total_bytes,
                "prefill_fp16_raw_bytes": compressed_prefill_raw_bytes,
                "decode_fp16_raw_bytes": compressed_decode_raw_bytes,
                "total_fp16_raw_bytes": compressed_total_raw_bytes,
                "prefill_comm_reduction_pct": _reduction_pct(compressed_prefill_bytes, compressed_prefill_raw_bytes),
                "decode_comm_reduction_pct": _reduction_pct(compressed_decode_bytes, compressed_decode_raw_bytes),
                "total_comm_reduction_pct": _reduction_pct(compressed_total_bytes, compressed_total_raw_bytes),
            })
        if args.save_per_sample:
            result["per_sample"] = per_sample
        results_by_config[config_name] = result
        if pipeline is not None:
            pipeline.shutdown()

    summary = {
        "benchmark": "HumanEval",
        "num_samples": len(test_records),
        "score_only": args.score_only,
        "include_fp16_baseline": args.include_fp16_baseline,
        "configs": results_by_config,
    }
    if not args.score_only:
        summary["bandwidth_mbps"] = args.bandwidth_mbps
    if args.include_fp16_baseline:
        total = max(len(test_records), 1)
        fp16_score = fp16_pass / total
        fp16_baseline: Dict[str, object] = {
            "score_name": "pass@1",
            "score": fp16_score,
            "passed": fp16_pass,
        }
        if not args.score_only:
            fp16_total_bytes = fp16_prefill_bytes + fp16_decode_bytes
            fp16_baseline.update({
                "prefill_comm_e2e_total_ms": fp16_prefill_comm_ms,
                "decode_comm_e2e_total_ms": fp16_decode_comm_ms,
                "prefill_transfer_bytes": fp16_prefill_bytes,
                "decode_transfer_bytes": fp16_decode_bytes,
                "total_transfer_bytes": fp16_total_bytes,
            })
        summary["fp16_baseline"] = fp16_baseline
        summary["score_delta_vs_fp16"] = {
            name: cast(float, metrics["score"]) - fp16_score for name, metrics in results_by_config.items()
        }

    _write_summary(args.output, summary)
    logger.info("Saved summary to %s", Path(args.output))


if __name__ == "__main__":
    main()
