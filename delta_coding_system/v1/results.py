from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


@dataclass
class PrefillResult:
    seq_len: int
    num_trigram: int = 0
    num_bigram: int = 0
    num_self_ref: int = 0
    num_unigram: int = 0
    raw_cosine_mean: float = 0.0
    raw_cosine_min: float = 0.0
    recon_cosine_mean: float = 0.0
    recon_cosine_min: float = 0.0
    mse_mean: float = 0.0
    mse_max: float = 0.0
    total_transfer_bytes: int = 0
    raw_fp16_bytes: int = 0
    compression_ratio: float = 1.0
    transfer_bytes_by_tier: Dict[str, int] = field(default_factory=dict)
    tier_detail: List[Dict[str, Any]] = field(default_factory=list)
    prefill_fwd_ms: float = 0.0
    classify_ms: float = 0.0
    encode_delta_ms: float = 0.0
    encode_self_ref_ms: float = 0.0
    encode_unigram_ms: float = 0.0
    table_update_ms: float = 0.0
    prev_update_wait_ms: float = 0.0
    total_ms: float = 0.0
    reconstructed_hidden: Optional[torch.Tensor] = None


@dataclass
class DecodeStepRecord:
    step: int
    tier: str
    raw_cosine: float
    recon_cosine: float
    transfer_bytes: int
    raw_fp16_bytes: int
    fwd_ms: float
    classify_ms: float
    encode_ms: float
    table_update_ms: float


@dataclass
class DecodeResult:
    decode_tokens: int
    step_records: List[DecodeStepRecord] = field(default_factory=list)
    num_trigram: int = 0
    num_bigram: int = 0
    num_self_ref: int = 0
    num_unigram: int = 0
    recon_cosine_mean: float = 0.0
    recon_cosine_min: float = 0.0
    total_transfer_bytes: int = 0
    raw_fp16_bytes: int = 0
    compression_ratio: float = 1.0
    total_fwd_ms: float = 0.0
    total_classify_ms: float = 0.0
    total_encode_ms: float = 0.0
    total_table_update_ms: float = 0.0
    total_ms: float = 0.0
    generated_token_ids: List[int] = field(default_factory=list)
    reconstructed_hidden: Optional[torch.Tensor] = None


@dataclass
class DecodeClassifyResult:
    tier: str
    stored_ref: Optional[Any] = None
    ref_is_materialized: bool = False