from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from delta_coding_system.benchmarks.run_longbench_benchmark import CONFIGS, FP16_CONFIG_NAME, _build_pipeline
from delta_coding_system.pipeline import OverlappedPipeline
from delta_coding_system.run_experiment import load_dataset_texts

LOGGER = logging.getLogger("ntp_loss_benchmark")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teacher-forced NTP loss benchmark for cache-table compression strategies")
    parser.add_argument("--model", default="/root/share/models/Qwen2.5-14B-Instruct")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config", choices=sorted([FP16_CONFIG_NAME, *CONFIGS.keys()]), required=True)
    parser.add_argument("--dataset", default="wikitext2")
    parser.add_argument("--warmup-requests", type=int, default=30)
    parser.add_argument("--test-requests", type=int, default=100)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--continuation-tokens", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--layer-boundary", type=int, default=6)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--max-table-entries", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split-manifest", default=None)
    return parser.parse_args()


def _load_model_and_tokenizer(model_path: str, device: torch.device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map={"": device},
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def _build_pipeline_for_ntp(model, tokenizer, device: torch.device, args: argparse.Namespace) -> OverlappedPipeline:
    pipeline_args = SimpleNamespace(
        layer_boundary=args.layer_boundary,
        max_table_entries=args.max_table_entries,
        group_size=args.group_size,
        max_new_tokens=args.continuation_tokens,
        max_seq_len=args.max_seq_len,
        score_only=True,
    )
    return _build_pipeline(model, tokenizer, device, pipeline_args, args.config)


def _prepare_token_sequences(tokenizer, dataset_name: str, needed: int, min_tokens: int, max_seq_len: int, seed: int) -> List[List[int]]:
    texts = load_dataset_texts(dataset_name)
    rng = random.Random(seed)
    rng.shuffle(texts)

    sequences: List[List[int]] = []
    buffer_parts: List[str] = []
    idx = 0
    max_iterations = max(len(texts) * 3, needed * 20)
    iterations = 0

    while len(sequences) < needed and iterations < max_iterations:
        text = texts[idx % len(texts)]
        idx += 1
        iterations += 1
        if not text or not text.strip():
            continue
        buffer_parts.append(text.strip())
        joined = "\n\n".join(buffer_parts)
        token_ids = tokenizer.encode(joined, add_special_tokens=False)
        if len(token_ids) >= min_tokens:
            sequences.append(token_ids[:max_seq_len])
            buffer_parts = []
        elif len(token_ids) > max_seq_len:
            sequences.append(token_ids[:max_seq_len])
            buffer_parts = []

    if len(sequences) < needed:
        raise RuntimeError(f"Could not prepare enough token sequences from dataset={dataset_name}; got {len(sequences)} need {needed}")
    return sequences[:needed]


def _sequence_digest(token_ids: List[int]) -> str:
    payload = ",".join(str(token_id) for token_id in token_ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _default_split_manifest_path(args: argparse.Namespace) -> Path:
    stem = (
        f"{args.dataset}_w{args.warmup_requests}_t{args.test_requests}"
        f"_p{args.prompt_tokens}_c{args.continuation_tokens}"
        f"_m{args.max_seq_len}_s{args.seed}.json"
    )
    return PROJECT_ROOT / "results_ntp_loss_qwen14b" / "splits" / stem


def _load_or_create_split_manifest(
    tokenizer,
    args: argparse.Namespace,
) -> Tuple[List[List[int]], List[List[int]], Path]:
    manifest_path = Path(args.split_manifest) if args.split_manifest else _default_split_manifest_path(args)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        warmup_sequences = [list(map(int, seq)) for seq in payload["warmup_sequences"]]
        test_sequences = [list(map(int, seq)) for seq in payload["test_sequences"]]
        return warmup_sequences, test_sequences, manifest_path

    total_needed = args.warmup_requests + args.test_requests
    min_tokens = args.prompt_tokens + args.continuation_tokens
    sequences = _prepare_token_sequences(
        tokenizer,
        args.dataset,
        total_needed,
        min_tokens,
        args.max_seq_len,
        args.seed,
    )
    warmup_sequences = sequences[:args.warmup_requests]
    test_sequences = sequences[args.warmup_requests:args.warmup_requests + args.test_requests]
    payload = {
        "dataset": args.dataset,
        "seed": args.seed,
        "warmup_requests": args.warmup_requests,
        "test_requests": args.test_requests,
        "prompt_tokens": args.prompt_tokens,
        "continuation_tokens": args.continuation_tokens,
        "max_seq_len": args.max_seq_len,
        "warmup_hashes": [_sequence_digest(seq) for seq in warmup_sequences],
        "test_hashes": [_sequence_digest(seq) for seq in test_sequences],
        "warmup_sequences": warmup_sequences,
        "test_sequences": test_sequences,
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return warmup_sequences, test_sequences, manifest_path


def _wait_pending_update(pipeline: OverlappedPipeline) -> None:
    if pipeline._pending_prefill_update is not None:
        pipeline._pending_prefill_update.result()
        pipeline._pending_prefill_update = None


def _run_prefill_from_ids(
    pipeline: OverlappedPipeline,
    token_ids: List[int],
) -> Tuple[torch.Tensor, object, torch.Tensor, Dict[Tuple[int, int, int], torch.Tensor], Dict[Tuple[int, int], torch.Tensor], torch.Tensor]:
    _wait_pending_update(pipeline)
    torch.cuda.set_device(pipeline.device)

    classify_future = pipeline.classify_executor.submit(
        pipeline.table.classify_and_build_refs,
        token_ids,
        pipeline.hidden_dim,
        pipeline.device,
    )
    input_tensor = torch.tensor([token_ids], dtype=torch.long, device=pipeline.device)
    prefill_hidden, prefix_cache = pipeline._run_prefix_prefill(input_tensor)

    tiers, sparse_refs, self_ref_sources, _first_occurrence_map = classify_future.result()
    trigram_indices = [i for i, tier in enumerate(tiers) if tier == "trigram"]
    bigram_indices = [i for i, tier in enumerate(tiers) if tier == "bigram"]
    self_ref_indices = [i for i, tier in enumerate(tiers) if tier == "self_ref"]
    unigram_indices = [i for i, tier in enumerate(tiers) if tier == "unigram"]

    real_acts = prefill_hidden
    reconstructed = torch.zeros_like(real_acts)

    delta_indices = trigram_indices + bigram_indices
    if delta_indices:
        idx_t = torch.tensor(delta_indices, dtype=torch.long, device=pipeline.device)
        real_batch = real_acts[idx_t]
        ref_batch = sparse_refs.gather(delta_indices)
        recon_batch, _ = pipeline._encode_delta_batch(real_batch, ref_batch, include_ref_idx=True)
        reconstructed[idx_t] = recon_batch

    if unigram_indices:
        if pipeline._unigram_uses_prev_ref():
            for pos in sorted(unigram_indices):
                real_uni = real_acts[pos].unsqueeze(0)
                if pos > 0:
                    prev_ref = reconstructed[pos - 1].unsqueeze(0)
                    if torch.count_nonzero(prev_ref).item() == 0:
                        prev_ref = real_acts[pos - 1].unsqueeze(0)
                    recon_uni, _ = pipeline._encode_prev_unigram_batch(real_uni, prev_ref)
                else:
                    recon_uni, _ = pipeline._encode_unigram_batch(real_uni)
                reconstructed[pos] = recon_uni.squeeze(0)
        else:
            idx_u = torch.tensor(unigram_indices, dtype=torch.long, device=pipeline.device)
            real_uni = real_acts[idx_u]
            recon_uni, _ = pipeline._encode_unigram_batch(real_uni)
            reconstructed[idx_u] = recon_uni

    if self_ref_indices:
        sorted_self_ref = sorted(self_ref_indices)
        idx_sr = torch.tensor(sorted_self_ref, dtype=torch.long, device=pipeline.device)
        real_sr = real_acts[idx_sr]
        source_positions = [self_ref_sources[pos] for pos in sorted_self_ref]
        src_t = torch.tensor(source_positions, dtype=torch.long, device=pipeline.device)
        ref_sr = reconstructed[src_t]
        recon_sr, _ = pipeline._encode_delta_batch(real_sr, ref_sr, include_ref_idx=True)
        reconstructed[idx_sr] = recon_sr

    pipeline.table.update_from_hidden_states(token_ids, prefill_hidden)
    local_prompt_trigrams, local_prompt_bigrams = pipeline._build_local_prompt_refs(token_ids, prefill_hidden)
    next_logits, suffix_cache = pipeline._run_suffix_prefill(reconstructed)
    return prefix_cache, suffix_cache, next_logits, local_prompt_trigrams, local_prompt_bigrams, prefill_hidden


@torch.inference_mode()
def _teacher_forced_compressed_loss(
    pipeline: OverlappedPipeline,
    prompt_ids: List[int],
    continuation_ids: List[int],
) -> Tuple[float, Dict[str, int]]:
    prefix_cache, suffix_cache, next_logits, local_prompt_trigrams, local_prompt_bigrams, prefill_hidden = _run_prefill_from_ids(
        pipeline,
        prompt_ids,
    )

    running_token_ids = list(prompt_ids)
    decode_hidden_by_pos: Dict[int, torch.Tensor] = {}
    reconstructed_hiddens: Dict[int, torch.Tensor] = {}
    first_occ_map: Dict[Tuple[int, int, int], int] = {}
    total_loss = 0.0
    tier_counts = {"trigram": 0, "bigram": 0, "self_ref": 0, "unigram": 0}

    for target_id in continuation_ids:
        target_tensor = torch.tensor([target_id], dtype=torch.long, device=pipeline.device)
        total_loss += float(F.cross_entropy(next_logits.float(), target_tensor, reduction="sum").item())

        next_tok = torch.tensor([[target_id]], dtype=torch.long, device=pipeline.device)
        running_token_ids.append(target_id)
        decode_pos = len(running_token_ids) - 1

        h = pipeline._run_prefix_decode_step(next_tok, prefix_cache)
        decode_hidden_by_pos[decode_pos] = h

        tier, ref_h, _raw_cos = pipeline._classify_decode_step(
            running_token_ids,
            decode_pos,
            first_occ_map,
            reconstructed_hiddens,
            local_prompt_trigrams,
            local_prompt_bigrams,
        )

        real_h_2d = h.unsqueeze(0)
        if tier == "unigram" and pipeline._unigram_uses_prev_ref():
            if decode_pos - 1 < len(prompt_ids):
                prev_ref = prefill_hidden[decode_pos - 1].unsqueeze(0)
            elif (decode_pos - 1) in reconstructed_hiddens:
                prev_ref = reconstructed_hiddens[decode_pos - 1].unsqueeze(0)
            else:
                prev_ref = None
        else:
            prev_ref = None

        if tier == "unigram" and prev_ref is not None and pipeline._unigram_uses_prev_ref():
            recon, _ = pipeline._encode_prev_unigram_batch(real_h_2d, prev_ref)
        else:
            recon, _ = pipeline._encode_decode_step(real_h_2d, ref_h, tier)

        reconstructed_hiddens[decode_pos] = recon.squeeze(0)
        pipeline._update_table_step(
            running_token_ids,
            decode_pos,
            h,
            prefill_hidden,
            prompt_ids,
            decode_hidden_by_pos,
        )
        next_logits = pipeline._run_suffix_decode_step(recon.squeeze(0), suffix_cache)
        tier_counts[tier] += 1

    mean_loss = total_loss / max(len(continuation_ids), 1)
    return mean_loss, tier_counts


@torch.inference_mode()
def _teacher_forced_fp16_loss(model, device: torch.device, prompt_ids: List[int], continuation_ids: List[int]) -> float:
    input_ids = prompt_ids + continuation_ids[:-1]
    model_input = torch.tensor([input_ids], dtype=torch.long, device=device)
    outputs = model(model_input, use_cache=False)
    start = len(prompt_ids) - 1
    end = start + len(continuation_ids)
    logits = outputs.logits[:, start:end, :].contiguous()
    labels = torch.tensor([continuation_ids], dtype=torch.long, device=device)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), labels.view(-1), reduction="mean")
    return float(loss.item())


def _save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    total_tokens_needed = args.prompt_tokens + args.continuation_tokens
    if total_tokens_needed > args.max_seq_len:
        raise ValueError("prompt_tokens + continuation_tokens must be <= max_seq_len")

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    model, tokenizer = _load_model_and_tokenizer(args.model, device)
    warmup_sequences, test_sequences, manifest_path = _load_or_create_split_manifest(tokenizer, args)
    warmup_hashes = [_sequence_digest(seq) for seq in warmup_sequences]
    test_hashes = [_sequence_digest(seq) for seq in test_sequences]

    overlap = set(warmup_hashes) & set(test_hashes)
    if overlap:
        raise RuntimeError(f"Warmup/test split overlap detected: {len(overlap)} duplicated sequences")

    pipeline = None if args.config == FP16_CONFIG_NAME else _build_pipeline_for_ntp(model, tokenizer, device, args)

    try:
        if pipeline is not None:
            LOGGER.info("Warmup %d sequences for config=%s", len(warmup_sequences), args.config)
            for idx, seq in enumerate(warmup_sequences, start=1):
                warmup_ids = seq[:args.prompt_tokens]
                _run_prefill_from_ids(pipeline, warmup_ids)
                if idx % 10 == 0 or idx == len(warmup_sequences):
                    stats = pipeline.table.stats
                    LOGGER.info(
                        "Warmup %d/%d table_trigrams=%d table_bigrams=%d",
                        idx,
                        len(warmup_sequences),
                        stats["num_trigrams"],
                        stats["num_bigrams"],
                    )

        per_sample: List[Dict] = []
        total_loss = 0.0
        total_count = 0
        agg_tiers = {"trigram": 0, "bigram": 0, "self_ref": 0, "unigram": 0}

        LOGGER.info("Test %d sequences for config=%s dataset=%s", len(test_sequences), args.config, args.dataset)
        for idx, seq in enumerate(test_sequences, start=1):
            prompt_ids = seq[:args.prompt_tokens]
            continuation_ids = seq[args.prompt_tokens:args.prompt_tokens + args.continuation_tokens]

            if args.config == FP16_CONFIG_NAME:
                mean_loss = _teacher_forced_fp16_loss(model, device, prompt_ids, continuation_ids)
                tier_counts = None
            else:
                mean_loss, tier_counts = _teacher_forced_compressed_loss(pipeline, prompt_ids, continuation_ids)
                for key in agg_tiers:
                    agg_tiers[key] += int(tier_counts[key])

            total_loss += mean_loss * len(continuation_ids)
            total_count += len(continuation_ids)
            sample_row = {
                "sample_index": idx - 1,
                "mean_ntp_loss": mean_loss,
                "num_eval_tokens": len(continuation_ids),
            }
            if tier_counts is not None:
                sample_row.update({f"num_{key}": int(value) for key, value in tier_counts.items()})
            per_sample.append(sample_row)

            if idx % 10 == 0 or idx == len(test_sequences):
                LOGGER.info(
                    "Config=%s progress=%d/%d running_mean_ntp_loss=%.6f",
                    args.config,
                    idx,
                    len(test_sequences),
                    total_loss / max(total_count, 1),
                )

        mean_ntp_loss = total_loss / max(total_count, 1)
        summary = {
            "model": args.model,
            "dataset": args.dataset,
            "config": args.config,
            "warmup_requests": args.warmup_requests,
            "test_requests": args.test_requests,
            "prompt_tokens": args.prompt_tokens,
            "continuation_tokens": args.continuation_tokens,
            "max_seq_len": args.max_seq_len,
            "mean_ntp_loss": mean_ntp_loss,
            "perplexity": math.exp(mean_ntp_loss) if mean_ntp_loss < 20 else float("inf"),
            "num_eval_tokens": total_count,
            "split_manifest": str(manifest_path),
            "warmup_hashes": warmup_hashes,
            "test_hashes": test_hashes,
            "per_sample": per_sample,
        }
        if pipeline is not None:
            summary["aggregate_tier_counts"] = agg_tiers
            summary["table_stats"] = pipeline.table.stats

        _save_json(Path(args.output), summary)
        LOGGER.info("Saved NTP loss summary to %s", args.output)
    finally:
        if pipeline is not None:
            pipeline.shutdown()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()