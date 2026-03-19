"""Delta-coding system for pipeline-parallel activation compression.

Provides overlapped CPU/GPU pipeline for encoding/decoding activations
using tiered n-gram reference matching with FP8 table storage and LRU eviction.
"""

from delta_coding_system.table import NgramTable
from delta_coding_system.codec import encode_decode_single
from delta_coding_system.pipeline import OverlappedPipeline

__all__ = ["NgramTable", "encode_decode_single", "OverlappedPipeline"]
