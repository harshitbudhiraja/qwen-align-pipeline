"""
Stage 4a: RL fine-tuning with GRPO (Group Relative Policy Optimization).

GRPO is ideal here because:
  - No learned reward model needed (rule-based reward is enough)
  - Works well with small batch sizes (important for 7B models)
  - Supported natively by TRL

Run AFTER SFT:
    python src/train_grpo.py --sft-checkpoint outputs/sft/final

References:
  - TRL GRPOTrainer docs
  - DeepSeek-R1 paper (GRPO originates there)
"""

import os
import re
import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, LoraConfig
from trl import GRPOTrainer, GRPOConfig

# Import our reward functions
import sys
sys.path.insert(0, str(Path(__file__).parent))
from reward_model import compute_reward, batch_rewards


# ── Generation prompt for GRPO rollouts ───────────────────────────────────

SYSTEM_PROMPT = (
    "You are a coding assistant. "
    "Always wrap your expected info output in <info>...</info> tags, "
    "and always wrap your output Python code in <python>...</python> tags. "
    "Never output raw code without these tags."
)


def format_prompt(example: dict) -> dict:
    """
    Convert a raw dataset row into the Qwen chat format
    for GRPO rollout generation.
    """
    prompt = example.get("prompt", example.get("text", ""))
    # Strip any existing assistant turn so model generates fresh
    if "<|im_start|>assistant" in prompt:
        prompt = prompt.split("<|im_start|>assistant")[0] + "<|im_start|>assistant\n"
    else:
        prompt = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    return {"prompt": prompt}


# ── Reward wrapper for GRPOTrainer ────────────────────────────────────────

def grpo_reward_fn(completions: list[str], **kwargs) -> list[float]:
    """
    GRPOTrainer calls this with a list of generated completions.
    Returns a scalar reward per completion.
    """
    return batch_rewards(completions)


# ── Model loading ─────────────────────────────────────────────────────────

def load_for_grpo(sft_checkpoint: str, base_model: str | None = None):
    """
    Load the SFT-fine-tuned model (LoRA checkpoint) for GRPO training.
    """
    # Detect base model from adapter config if not supplied
    if base_model is None:
        adapter_cfg_path = Path(sft_checkpoint) / "adapter_config.json"
        if adapter_cfg_path.exists():
            import json
            with open(adapter_cfg_path) as f:
                base_model = json.load(f).get("base_model_name_or_path", "Qwen/Qwen2.5-Coder-7B-Instruct")
        else:
            base_model = "Qwen/Qwen2.5-Coder-0.5B"

    print(f"Base model     : {base_model}")
    print(f"SFT checkpoint : {sft_checkpoint}")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    
    sft_path = Path(sft_checkpoint)
    if sft_path.exists() and sft_path.is_dir():
        model = PeftModel.from_pretrained(base, sft_checkpoint, is_trainable=True)
    else:
        print(f"SFT checkpoint '{sft_checkpoint}' not found locally. Initializing a new LoRA adapter.")
        from peft import get_peft_model
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base, peft_config)
        
    model.enable_input_require_grads()

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# ── Main GRPO training ────────────────────────────────────────────────────

def run_grpo(
    sft_checkpoint: str,
    data_path: str     = "data/sft_train.jsonl",
    output_dir: str    = "outputs/grpo",
    num_epochs: int    = 1,
    lr: float          = 5e-6,
    batch_size: int    = 4,
    grad_accum: int    = 4,
    num_generations: int = 4,    # G in GRPO — samples per prompt
    # max_new_tokens: int  = 512,
    kl_coef: float       = 0.1,   # KL penalty coefficient
    base_model: str | None = None,
):
    print(f"\n{'='*60}")
    print(f"  GRPO Training")
    print(f"  SFT checkpoint : {sft_checkpoint}")
    print(f"  Data           : {data_path}")
    print(f"  Output         : {output_dir}")
    print(f"  Generations/prompt (G) : {num_generations}")
    print(f"{'='*60}\n")

    model, tokenizer = load_for_grpo(sft_checkpoint, base_model)

    # Load and format dataset
    raw_dataset = load_dataset("json", data_files=data_path, split="train")
    dataset = raw_dataset.map(format_prompt, remove_columns=raw_dataset.column_names)
    print(f"GRPO dataset: {len(dataset)} prompts")

    grpo_config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        bf16=True,
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        report_to="none",

        # GRPO-specific
        num_generations=num_generations,       # G: how many completions per prompt
        max_completion_length=512,
        beta=kl_coef,                          # KL penalty vs reference model
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[grpo_reward_fn],
        args=grpo_config,
        train_dataset=dataset,
    )

    print("Starting GRPO training...")
    trainer.train()

    print(f"\nSaving GRPO model to {output_dir}/final")
    trainer.save_model(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")
    print("GRPO complete.")

    return trainer


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-checkpoint", required=True)
    parser.add_argument("--data",           default="data/sft_train.jsonl")
    parser.add_argument("--output-dir",     default="outputs/grpo")
    parser.add_argument("--epochs",         type=int,   default=1)
    parser.add_argument("--lr",             type=float, default=5e-6)
    parser.add_argument("--batch-size",     type=int,   default=4)
    parser.add_argument("--generations",    type=int,   default=4)
    parser.add_argument("--kl-coef",        type=float, default=0.1)
    parser.add_argument("--base-model",     default=None)
    args = parser.parse_args()

    run_grpo(
        sft_checkpoint=args.sft_checkpoint,
        data_path=args.data,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        num_generations=args.generations,
        kl_coef=args.kl_coef,
        base_model=args.base_model,
    )
