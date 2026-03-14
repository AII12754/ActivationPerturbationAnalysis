"""E4: Cross-sequence alignment experiment.

For a pair of prompts, runs both through the model, extracts hidden states,
then compares per layer using CKA, Procrustes distance, subspace overlap,
centroid cosine, and shared-token cosine.
"""

from __future__ import annotations

import gc
import logging
from itertools import product
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from ..core.types import ExperimentJob, make_experiment_id, resolve_dataset_list, resolve_dtype
from ..metrics.similarity import linear_cka, procrustes_distance, subspace_overlap
from .base import BaseExperiment

logger = logging.getLogger(__name__)

ALIGNMENT_COLUMNS = [
    "experiment_id", "model_id",
    "pair_type", "dataset_a", "dataset_b",
    "prompt_id_a", "prompt_id_b", "context_length",
    "layer", "cka", "procrustes_distance", "subspace_overlap",
    "centroid_cosine", "shared_token_cosine",
    "num_shared_tokens", "seq_len_a", "seq_len_b",
]


class CrossSequenceAlignmentExperiment(BaseExperiment):
    experiment_id = "e4"
    experiment_name = "cross_sequence"

    @property
    def is_two_prompt(self) -> bool:
        return True

    @classmethod
    def table_schemas(cls) -> Dict[str, List[str]]:
        return {"alignment": ALIGNMENT_COLUMNS}

    @classmethod
    def default_config_section(cls) -> str:
        return "cross_sequence"

    @classmethod
    def build_sweep_jobs(cls, config: Dict[str, Any]) -> List[ExperimentJob]:
        sweep = config["sweep"]
        context_lengths = sweep.get("context_lengths", [512])
        if isinstance(context_lengths, int):
            context_lengths = [context_lengths]
        num_prompts = sweep.get("num_prompts_per_length", 3)
        dataset_specs = resolve_dataset_list(config)

        jobs: List[ExperimentJob] = []

        # Within-dataset pairs
        for ds_spec in dataset_specs:
            for ctx_len in context_lengths:
                for i in range(num_prompts):
                    for j in range(i + 1, num_prompts):
                        params = {
                            "dataset_name": ds_spec["name"],
                            "dataset_config": ds_spec["config"],
                            "dataset_split": ds_spec["split"],
                            "dataset_name_b": ds_spec["name"],
                            "dataset_config_b": ds_spec["config"],
                            "dataset_split_b": ds_spec["split"],
                            "context_length": ctx_len,
                            "prompt_index": i,
                            "prompt_index_b": j,
                            "pair_type": "within_dataset",
                        }
                        jobs.append(ExperimentJob(
                            job_id=make_experiment_id(params), params=params,
                        ))

        # Cross-dataset pairs (first prompt from each)
        for idx_a in range(len(dataset_specs)):
            for idx_b in range(idx_a + 1, len(dataset_specs)):
                for ctx_len in context_lengths:
                    params = {
                        "dataset_name": dataset_specs[idx_a]["name"],
                        "dataset_config": dataset_specs[idx_a]["config"],
                        "dataset_split": dataset_specs[idx_a]["split"],
                        "dataset_name_b": dataset_specs[idx_b]["name"],
                        "dataset_config_b": dataset_specs[idx_b]["config"],
                        "dataset_split_b": dataset_specs[idx_b]["split"],
                        "context_length": ctx_len,
                        "prompt_index": 0,
                        "prompt_index_b": 0,
                        "pair_type": "cross_dataset",
                    }
                    jobs.append(ExperimentJob(
                        job_id=make_experiment_id(params), params=params,
                    ))

        logger.info("Built %d cross-sequence sweep jobs.", len(jobs))
        return jobs

    @torch.inference_mode()
    def run_two_prompt(
        self, model, tokenizer,
        prompt_text_a: str, prompt_text_b: str,
        config: Dict[str, Any], **kwargs,
    ) -> Dict[str, List[Dict[str, Any]]]:
        cross_cfg = config.get("cross_sequence", {})
        act_dtype = resolve_dtype(cross_cfg.get("activation_dtype", "float32"))
        subspace_k = cross_cfg.get("subspace_k", 10)

        device = next(model.parameters()).device

        # Forward pass A
        ids_a = tokenizer.encode(prompt_text_a, add_special_tokens=False)
        tensor_a = torch.tensor([ids_a], dtype=torch.long, device=device)
        outputs_a = model(input_ids=tensor_a, output_hidden_states=True, use_cache=False)
        hidden_a = outputs_a.hidden_states
        num_all_layers = len(hidden_a) - 1
        hidden_a_cpu = [h.squeeze(0).to(act_dtype).cpu() for h in hidden_a]
        del outputs_a
        gc.collect()
        torch.cuda.empty_cache()

        # Forward pass B
        ids_b = tokenizer.encode(prompt_text_b, add_special_tokens=False)
        tensor_b = torch.tensor([ids_b], dtype=torch.long, device=device)
        outputs_b = model(input_ids=tensor_b, output_hidden_states=True, use_cache=False)
        hidden_b = outputs_b.hidden_states
        hidden_b_cpu = [h.squeeze(0).to(act_dtype).cpu() for h in hidden_b]
        del outputs_b
        gc.collect()
        torch.cuda.empty_cache()

        records = []
        for layer_idx in range(num_all_layers + 1):
            ha = hidden_a_cpu[layer_idx]
            hb = hidden_b_cpu[layer_idx]

            cka = linear_cka(ha, hb)
            proc_dist = procrustes_distance(ha, hb)
            sub_overlap = subspace_overlap(ha, hb, k=subspace_k)

            centroid_a = ha.mean(dim=0)
            centroid_b = hb.mean(dim=0)
            centroid_cos = F.cosine_similarity(
                centroid_a.unsqueeze(0), centroid_b.unsqueeze(0)
            ).item()

            set_a = set(ids_a)
            set_b = set(ids_b)
            shared_ids = set_a & set_b
            shared_cosines = []
            if shared_ids:
                for tid in list(shared_ids)[:50]:
                    pos_a = [i for i, t in enumerate(ids_a) if t == tid]
                    pos_b = [i for i, t in enumerate(ids_b) if t == tid]
                    if pos_a and pos_b and pos_a[0] < ha.shape[0] and pos_b[0] < hb.shape[0]:
                        cos = F.cosine_similarity(
                            ha[pos_a[0]].unsqueeze(0), hb[pos_b[0]].unsqueeze(0)
                        ).item()
                        shared_cosines.append(cos)
            shared_token_cosine = (
                sum(shared_cosines) / len(shared_cosines)
                if shared_cosines else float("nan")
            )

            records.append({
                "layer": layer_idx,
                "cka": cka,
                "procrustes_distance": proc_dist,
                "subspace_overlap": sub_overlap,
                "centroid_cosine": centroid_cos,
                "shared_token_cosine": shared_token_cosine,
                "num_shared_tokens": len(shared_ids),
                "seq_len_a": ha.shape[0],
                "seq_len_b": hb.shape[0],
            })

        del hidden_a_cpu, hidden_b_cpu
        gc.collect()
        return {"alignment": records}

    def run(self, model, tokenizer, prompt_text: str, config: Dict[str, Any], **kwargs):
        raise NotImplementedError("E4 requires two prompts. Use run_two_prompt().")
