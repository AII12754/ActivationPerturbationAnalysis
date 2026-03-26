#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="Hello world")
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        tp_plan="auto",
    )

    inputs = tokenizer(args.prompt, return_tensors="pt")
    inputs = {key: value.to(torch.device(f"cuda:{local_rank}")) for key, value in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True, use_cache=True)

    hidden_devices = sorted({str(t.device) for t in outputs.hidden_states})
    print({
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "first_param_device": str(next(model.parameters()).device),
        "hidden_devices": hidden_devices,
        "num_hidden_states": len(outputs.hidden_states),
        "last_logits_device": str(outputs.logits.device),
    }, flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()