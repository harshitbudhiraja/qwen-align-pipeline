"""
Merge LoRA adapters into the base model for deployment.

After training, you have a base model + LoRA adapter weights.
For production inference, merging them into a single model is faster
and avoids the PEFT overhead.

Usage:
    python src/merge_lora.py \
        --base-model  Qwen/Qwen2.5-Coder-7B-Instruct \
        --lora-checkpoint outputs/grpo/final \
        --output-dir  outputs/merged
"""

import argparse
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_and_save(
    base_model: str,
    lora_checkpoint: str,
    output_dir: str,
    push_to_hub: bool = False,
    hub_repo: str | None = None,
):
    print(f"\nMerging LoRA into base model")
    print(f"  Base      : {base_model}")
    print(f"  Adapter   : {lora_checkpoint}")
    print(f"  Output    : {output_dir}\n")

    # Load base in fp16 (no quantization — we need full precision for merge)
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="cpu",           # merge on CPU to avoid GPU OOM
        trust_remote_code=True,
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, lora_checkpoint)

    print("Merging weights...")
    model = model.merge_and_unload()
    model.eval()

    print(f"Saving merged model to {output_dir}...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)

    print("Merge complete.")
    print(f"\nTo run inference:")
    print(f"  from transformers import pipeline")
    print(f"  pipe = pipeline('text-generation', model='{output_dir}', device_map='auto')")

    if push_to_hub and hub_repo:
        print(f"\nPushing to Hugging Face Hub: {hub_repo}")
        model.push_to_hub(hub_repo)
        tokenizer.push_to_hub(hub_repo)
        print("Upload complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model",      required=True)
    parser.add_argument("--lora-checkpoint", required=True)
    parser.add_argument("--output-dir",      required=True)
    parser.add_argument("--push-to-hub",     action="store_true")
    parser.add_argument("--hub-repo",        default=None)
    args = parser.parse_args()

    merge_and_save(
        base_model=args.base_model,
        lora_checkpoint=args.lora_checkpoint,
        output_dir=args.output_dir,
        push_to_hub=args.push_to_hub,
        hub_repo=args.hub_repo,
    )
