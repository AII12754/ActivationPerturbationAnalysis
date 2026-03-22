#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import pyarrow.parquet as pq
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from delta_coding_system.pipeline import OverlappedPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("humaneval_benchmark")

DATASET_PATH = Path("/root/share/dataset/humaneval/openai_humaneval")


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


def _build_pipeline(model, tokenizer, device: torch.device, args) -> OverlappedPipeline:
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
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HumanEval benchmark for compressed pipeline vs FP16 baseline")
    parser.add_argument("--model", default="/root/share/models/Qwen2.5-32B-Instruct")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=164)
    parser.add_argument("--warmup-samples", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--layer-boundary", type=int, default=6)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--max-table-entries", type=int, default=100000)
    parser.add_argument("--bandwidth-mbps", type=int, default=200)
    parser.add_argument("--exec-timeout-sec", type=int, default=10)
    parser.add_argument("--output", default="results_humaneval_benchmark/summary.json")
    parser.add_argument("--save-per-sample", action="store_true")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

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

    pipeline = _build_pipeline(model, tokenizer, device, args)
    logger.info("Warmup %d HumanEval samples...", len(warmup_records))
    for row in warmup_records:
        pipeline.process_request(row["prompt"], phase="warmup")

    hidden_dim = model.config.hidden_size
    compressed_pass = 0
    fp16_pass = 0
    compressed_prefill_comm_ms = 0.0
    compressed_decode_comm_ms = 0.0
    fp16_prefill_comm_ms = 0.0
    fp16_decode_comm_ms = 0.0
    compressed_prefill_bytes = 0
    compressed_decode_bytes = 0
    compressed_prefill_raw_bytes = 0
    compressed_decode_raw_bytes = 0
    fp16_prefill_bytes = 0
    fp16_decode_bytes = 0
    per_sample: List[Dict[str, object]] = []

    logger.info("Scoring %d HumanEval samples...", len(test_records))
    for idx, row in enumerate(test_records, start=1):
        prompt = row["prompt"]
        test_code = row["test"]
        entry_point = row["entry_point"]

        prefill_res, decode_res, _ = pipeline.process_request(prompt, phase="test")
        compressed_completion = _decode_generated(tokenizer, decode_res.generated_token_ids)
        compressed_ok = _score_candidate(prompt, compressed_completion, test_code, entry_point, args.exec_timeout_sec)
        compressed_pass += int(compressed_ok)

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
        fp16_pass += int(fp16_ok)

        prompt_len = int(encoded.input_ids.shape[1])
        fp16_decode_len = len(fp16_generated_ids)
        fp16_prefill_comm = _network_ms(prompt_len * hidden_dim * 2, args.bandwidth_mbps)
        fp16_decode_comm = _network_ms(fp16_decode_len * hidden_dim * 2, args.bandwidth_mbps)
        fp16_prefill_comm_ms += fp16_prefill_comm
        fp16_decode_comm_ms += fp16_decode_comm
        fp16_prefill_bytes += int(prompt_len * hidden_dim * 2)
        fp16_decode_bytes += int(fp16_decode_len * hidden_dim * 2)

        per_sample.append({
            "task_id": row["task_id"],
            "compressed_pass": compressed_ok,
            "fp16_pass": fp16_ok,
            "compressed_prefill_comm_ms": comp_prefill_comm,
            "compressed_decode_comm_ms": comp_decode_comm,
            "fp16_prefill_comm_ms": fp16_prefill_comm,
            "fp16_decode_comm_ms": fp16_decode_comm,
        })

        if idx % 10 == 0 or idx == len(test_records):
            logger.info(
                "Processed %d/%d samples compressed_pass=%d fp16_pass=%d",
                idx,
                len(test_records),
                compressed_pass,
                fp16_pass,
            )

    total = max(len(test_records), 1)
    compressed_total_bytes = compressed_prefill_bytes + compressed_decode_bytes
    compressed_total_raw_bytes = compressed_prefill_raw_bytes + compressed_decode_raw_bytes
    fp16_total_bytes = fp16_prefill_bytes + fp16_decode_bytes

    def _reduction_pct(compressed_bytes: int, raw_bytes: int) -> float:
        if raw_bytes <= 0:
            return 0.0
        return (1.0 - (compressed_bytes / raw_bytes)) * 100.0

    summary = {
        "benchmark": "HumanEval",
        "bandwidth_mbps": args.bandwidth_mbps,
        "num_samples": len(test_records),
        "compressed_system": {
            "score_name": "pass@1",
            "score": compressed_pass / total,
            "passed": compressed_pass,
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
        },
        "fp16_baseline": {
            "score_name": "pass@1",
            "score": fp16_pass / total,
            "passed": fp16_pass,
            "prefill_comm_e2e_total_ms": fp16_prefill_comm_ms,
            "decode_comm_e2e_total_ms": fp16_decode_comm_ms,
            "prefill_transfer_bytes": fp16_prefill_bytes,
            "decode_transfer_bytes": fp16_decode_bytes,
            "total_transfer_bytes": fp16_total_bytes,
        },
        "score_delta": (compressed_pass - fp16_pass) / total,
    }
    if args.save_per_sample:
        summary["per_sample"] = per_sample

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    logger.info("Saved summary to %s", out_path)

    pipeline.shutdown()


if __name__ == "__main__":
    main()