#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch
import torch.distributed as dist

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from delta_coding_system.benchmarks.run_longbench_benchmark import (
    CONFIGS,
    FP16_CONFIG_NAME,
    _build_pipeline,
    _decode_generated,
    _network_ms,
    _reduction_pct,
    _run_fp16_request_with_retry,
    _run_request_with_retry,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("longbench_v2_benchmark")

_DIST_RANK = 0

DATASET_PATH = Path("/root/share/dataset/LongbenchV2/data.json")


def _load_records() -> List[Dict[str, object]]:
    return json.loads(DATASET_PATH.read_text())


def _domain_order(records: List[Dict[str, object]]) -> List[str]:
    seen: List[str] = []
    for row in records:
        domain = str(row["domain"])
        if domain not in seen:
            seen.append(domain)
    return seen


def _build_prompt(row: Dict[str, object]) -> str:
    return (
        "Read the long context carefully and answer the multiple-choice question. "
        "Reply with exactly one capital letter: A, B, C, or D.\n\n"
        f"Context:\n{str(row['context']).strip()}\n\n"
        f"Question:\n{str(row['question']).strip()}\n\n"
        f"A. {str(row['choice_A']).strip()}\n"
        f"B. {str(row['choice_B']).strip()}\n"
        f"C. {str(row['choice_C']).strip()}\n"
        f"D. {str(row['choice_D']).strip()}\n\n"
        "Answer:"
    )


def _extract_choice(prediction: str) -> str:
    match = re.search(r"\b([ABCD])\b", prediction.upper())
    return match.group(1) if match else ""


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


def _write_summary(output_path: str, summary: Dict[str, object]) -> None:
    if _DIST_RANK != 0:
        return
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))


def _setup_tp(tp_size: int) -> tuple[int, int, bool]:
    global _DIST_RANK

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    distributed = tp_size > 1 or world_size > 1
    if distributed:
        if world_size <= 1:
            raise RuntimeError("Tensor parallel requires torchrun with WORLD_SIZE > 1")
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group("nccl")
        _DIST_RANK = rank
        if rank != 0:
            logger.setLevel(logging.WARNING)
        logger.info("Initialized TP rank=%d local_rank=%d world_size=%d", rank, local_rank, world_size)
    else:
        _DIST_RANK = 0
    return rank, local_rank, distributed


def _cleanup_tp(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LongBench v2 benchmark for compressed pipeline")
    parser.add_argument("--model", default="/root/share/models/Qwen2.5-32B-Instruct")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config", choices=sorted([*CONFIGS.keys(), FP16_CONFIG_NAME]), nargs="+", default=["optimized_default"])
    parser.add_argument("--domains", nargs="+", default=None)
    parser.add_argument("--samples-per-domain", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--layer-boundary", type=int, default=6)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--max-table-entries", type=int, default=100000)
    parser.add_argument("--bandwidth-mbps", type=int, default=200)
    parser.add_argument("--output", default="results_longbench_v2/summary.json")
    parser.add_argument("--save-per-sample", action="store_true")
    parser.add_argument("--score-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tp-size", type=int, default=1)
    args = parser.parse_args()

    rank, local_rank, distributed = _setup_tp(args.tp_size)

    from transformers.models.auto.modeling_auto import AutoModelForCausalLM
    from transformers.models.auto.tokenization_auto import AutoTokenizer

    all_records = _load_records()
    domain_names = _domain_order(all_records)
    selected_domains = args.domains or domain_names
    grouped_records: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in all_records:
        domain = str(row["domain"])
        if domain in selected_domains:
            grouped_records[domain].append(row)

    device_index = local_rank if distributed else args.gpu
    device = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model_kwargs = {
        "torch_dtype": torch.float16,
        "trust_remote_code": True,
    }
    if distributed:
        model_kwargs["tp_plan"] = "auto"
    else:
        model_kwargs["device_map"] = {"": device}
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.eval()
    hidden_dim = model.config.hidden_size

    existing_summary = None
    if args.resume:
        out_path = Path(args.output)
        if out_path.exists():
            existing_summary = json.loads(out_path.read_text())
    existing_configs = existing_summary.get("configs", {}) if existing_summary is not None else {}

    total_samples = 0
    for domain in selected_domains:
        records = grouped_records[domain]
        if args.samples_per_domain is not None:
            records = records[: args.samples_per_domain]
        total_samples += len(records)
    logger.info("Scoring %d LongBench v2 samples across %d domains for %d configs...", total_samples, len(selected_domains), len(args.config))

    results_by_config: Dict[str, Dict[str, object]] = {}
    for config_name in args.config:
        existing_result = existing_configs.get(config_name)
        if isinstance(existing_result, dict) and existing_result.get("completed"):
            results_by_config[config_name] = existing_result
            logger.info("Skipping config=%s because completed result already exists in %s", config_name, args.output)
            continue

        pipeline = None if config_name == FP16_CONFIG_NAME else _build_pipeline(model, tokenizer, device, args, config_name)
        overall_correct = 0
        overall_count = 0
        prefill_comm_ms = 0.0
        decode_comm_ms = 0.0
        prefill_bytes = 0
        decode_bytes = 0
        prefill_raw_bytes = 0
        decode_raw_bytes = 0
        domain_results: Dict[str, Dict[str, object]] = {}

        for domain in selected_domains:
            records = grouped_records[domain]
            if args.samples_per_domain is not None:
                records = records[: args.samples_per_domain]

            domain_correct = 0
            per_sample: List[Dict[str, object]] = []
            for idx, row in enumerate(records, start=1):
                prompt = _build_prompt(row)
                if pipeline is None:
                    prediction_text, prompt_len, decode_len = _run_fp16_request_with_retry(
                        model,
                        tokenizer,
                        device,
                        prompt,
                        args.max_seq_len,
                        args.max_new_tokens,
                        domain,
                        row["_id"],
                    )
                else:
                    prefill_res, decode_res, _ = _run_request_with_retry(
                        pipeline,
                        prompt,
                        phase="test",
                        config_name=config_name,
                        task_name=domain,
                        sample_id=row["_id"],
                    )
                    prediction_text = _decode_generated(tokenizer, decode_res.generated_token_ids)

                predicted_choice = _extract_choice(prediction_text)
                correct_choice = str(row["answer"]).strip().upper()
                is_correct = int(predicted_choice == correct_choice)
                domain_correct += is_correct
                overall_correct += is_correct
                overall_count += 1

                if not args.score_only:
                    if pipeline is None:
                        current_prefill_bytes = int(prompt_len * hidden_dim * 2)
                        current_decode_bytes = int(decode_len * hidden_dim * 2)
                        prefill_comm_ms += _network_ms(current_prefill_bytes, args.bandwidth_mbps)
                        decode_comm_ms += _network_ms(current_decode_bytes, args.bandwidth_mbps)
                        prefill_bytes += current_prefill_bytes
                        decode_bytes += current_decode_bytes
                        prefill_raw_bytes += current_prefill_bytes
                        decode_raw_bytes += current_decode_bytes
                    else:
                        prefill_local = prefill_res.classify_ms + prefill_res.encode_delta_ms + prefill_res.encode_self_ref_ms + prefill_res.encode_unigram_ms
                        decode_local = decode_res.total_classify_ms + decode_res.total_encode_ms
                        prefill_comm_ms += prefill_local + _network_ms(prefill_res.total_transfer_bytes, args.bandwidth_mbps)
                        decode_comm_ms += decode_local + _network_ms(decode_res.total_transfer_bytes, args.bandwidth_mbps)
                        prefill_bytes += int(prefill_res.total_transfer_bytes)
                        decode_bytes += int(decode_res.total_transfer_bytes)
                        prefill_raw_bytes += int(prefill_res.raw_fp16_bytes)
                        decode_raw_bytes += int(decode_res.raw_fp16_bytes)

                if args.save_per_sample:
                    per_sample.append({
                        "task_id": row["_id"],
                        "prediction": prediction_text,
                        "predicted_choice": predicted_choice,
                        "answer": correct_choice,
                        "correct": bool(is_correct),
                    })

                if idx % 10 == 0 or idx == len(records):
                    domain_result: Dict[str, object] = {
                        "score_name": "accuracy",
                        "score": domain_correct / max(idx, 1),
                        "correct": domain_correct,
                        "num_samples": idx,
                        "completed": idx == len(records),
                    }
                    if args.save_per_sample:
                        domain_result["per_sample"] = per_sample
                    domain_results[domain] = domain_result

                    config_result: Dict[str, object] = {
                        "score_name": "accuracy",
                        "score": overall_correct / max(overall_count, 1),
                        "correct": overall_correct,
                        "num_samples": overall_count,
                        "domains": domain_results,
                        "effective_strategy": _effective_strategy_payload(config_name),
                        "completed": False,
                    }
                    if not args.score_only:
                        total_transfer_bytes = prefill_bytes + decode_bytes
                        total_fp16_raw_bytes = prefill_raw_bytes + decode_raw_bytes
                        config_result.update({
                            "prefill_comm_e2e_total_ms": prefill_comm_ms,
                            "decode_comm_e2e_total_ms": decode_comm_ms,
                            "prefill_transfer_bytes": prefill_bytes,
                            "decode_transfer_bytes": decode_bytes,
                            "total_transfer_bytes": total_transfer_bytes,
                            "prefill_fp16_raw_bytes": prefill_raw_bytes,
                            "decode_fp16_raw_bytes": decode_raw_bytes,
                            "total_fp16_raw_bytes": total_fp16_raw_bytes,
                            "prefill_comm_reduction_pct": _reduction_pct(prefill_bytes, prefill_raw_bytes),
                            "decode_comm_reduction_pct": _reduction_pct(decode_bytes, decode_raw_bytes),
                            "total_comm_reduction_pct": _reduction_pct(total_transfer_bytes, total_fp16_raw_bytes),
                        })
                    results_by_config[config_name] = config_result
                    _write_summary(args.output, {
                        "benchmark": "LongBenchV2",
                        "score_name": "accuracy",
                        "score_only": args.score_only,
                        "domains": selected_domains,
                        "samples_per_domain": args.samples_per_domain,
                        "configs": results_by_config,
                    })
                    logger.info(
                        "Config=%s domain=%s processed %d/%d samples acc=%.4f",
                        config_name,
                        domain,
                        idx,
                        len(records),
                        domain_correct / max(idx, 1),
                    )

        config_result = {
            "score_name": "accuracy",
            "score": overall_correct / max(overall_count, 1),
            "correct": overall_correct,
            "num_samples": overall_count,
            "domains": domain_results,
            "effective_strategy": _effective_strategy_payload(config_name),
            "completed": True,
        }
        if not args.score_only:
            total_transfer_bytes = prefill_bytes + decode_bytes
            total_fp16_raw_bytes = prefill_raw_bytes + decode_raw_bytes
            config_result.update({
                "prefill_comm_e2e_total_ms": prefill_comm_ms,
                "decode_comm_e2e_total_ms": decode_comm_ms,
                "prefill_transfer_bytes": prefill_bytes,
                "decode_transfer_bytes": decode_bytes,
                "total_transfer_bytes": total_transfer_bytes,
                "prefill_fp16_raw_bytes": prefill_raw_bytes,
                "decode_fp16_raw_bytes": decode_raw_bytes,
                "total_fp16_raw_bytes": total_fp16_raw_bytes,
                "prefill_comm_reduction_pct": _reduction_pct(prefill_bytes, prefill_raw_bytes),
                "decode_comm_reduction_pct": _reduction_pct(decode_bytes, decode_raw_bytes),
                "total_comm_reduction_pct": _reduction_pct(total_transfer_bytes, total_fp16_raw_bytes),
            })
        results_by_config[config_name] = config_result
        _write_summary(args.output, {
            "benchmark": "LongBenchV2",
            "score_name": "accuracy",
            "score_only": args.score_only,
            "domains": selected_domains,
            "samples_per_domain": args.samples_per_domain,
            "configs": results_by_config,
        })
        if pipeline is not None:
            pipeline.shutdown()

    _write_summary(args.output, {
        "benchmark": "LongBenchV2",
        "score_name": "accuracy",
        "score_only": args.score_only,
        "domains": selected_domains,
        "samples_per_domain": args.samples_per_domain,
        "configs": results_by_config,
    })
    if rank == 0:
        logger.info("Saved summary to %s", Path(args.output))
    _cleanup_tp(distributed)


if __name__ == "__main__":
    main()