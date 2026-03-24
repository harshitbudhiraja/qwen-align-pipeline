"""
Stage 4b: DPO (Direct Preference Optimization) — offline alternative to GRPO.

Use DPO when you already have (or can generate) preferred/rejected pairs.
DPO is simpler to tune than GRPO but requires pre-computed pairs.

Dataset format required (dpo_train.jsonl):
  {"prompt": "...", "chosen": "...```python...```...", "rejected": "...raw code..."}

Run AFTER SFT:
    python src/train_dpo.py --sft-checkpoint outputs/sft/final

When to use DPO vs GRPO:
  - DPO  : You have quality (chosen, rejected) pairs already.
            Faster training, more stable loss curve.
  - GRPO : You only have prompts. Model generates rollouts online
            and rewards are computed on the fly. More data-efficient.
"""

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from trl import DPOTrainer, DPOConfig


# ── Model loading ─────────────────────────────────────────────────────────

def load_for_dpo(sft_checkpoint: str, base_model: str | None = None):
    if base_model is None:
        adapter_cfg = Path(sft_checkpoint) / "adapter_config.json"
        if adapter_cfg.exists():
            import json
            with open(adapter_cfg) as f:
                base_model = json.load(f).get("base_model_name_or_path",
                                               "Qwen/Qwen2.5-Coder-7B-Instruct")
        else:
            base_model = "Qwen/Qwen2.5-Coder-7B-Instruct"

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Policy model (trainable)
    policy = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=bnb, device_map="auto", trust_remote_code=True,
    )
    policy = PeftModel.from_pretrained(policy, sft_checkpoint, is_trainable=True)
    policy.enable_input_require_grads()

    # Reference model (frozen copy — DPO needs it for KL constraint)
    ref_model = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=bnb, device_map="auto", trust_remote_code=True,
    )
    ref_model = PeftModel.from_pretrained(ref_model, sft_checkpoint, is_trainable=False)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return policy, ref_model, tokenizer


# ── Dataset formatting ────────────────────────────────────────────────────

def prepare_dpo_dataset(data_path: str):
    """
    Load and validate the DPO dataset.
    Expected columns: prompt, chosen, rejected.
    """
    dataset = load_dataset("json", data_files=data_path, split="train")

    required = {"prompt", "chosen", "rejected"}
    assert required.issubset(set(dataset.column_names)), (
        f"DPO dataset must have columns: {required}. Found: {dataset.column_names}"
    )

    # Filter out rows where chosen == rejected (no signal)
    before = len(dataset)
    dataset = dataset.filter(lambda x: x["chosen"] != x["rejected"])
    print(f"DPO dataset: {len(dataset)} pairs (filtered {before - len(dataset)} identical pairs)")

    return dataset


# ── Main DPO training ─────────────────────────────────────────────────────

def run_dpo(
    sft_checkpoint: str,
    data_path: str  = "data/dpo_train.jsonl",
    output_dir: str = "outputs/dpo",
    num_epochs: int = 1,
    lr: float       = 1e-6,
    batch_size: int = 2,
    grad_accum: int = 8,
    beta: float     = 0.1,          # KL penalty strength (DPO β)
    max_length: int = 2048,
    max_prompt_length: int = 512,
    base_model: str | None = None,
    loss_type: str  = "sigmoid",    # "sigmoid" | "ipo" | "hinge" | "kto_pair"
):
    print(f"\n{'='*60}")
    print(f"  DPO Training (β={beta}, loss={loss_type})")
    print(f"  SFT checkpoint : {sft_checkpoint}")
    print(f"  Data           : {data_path}")
    print(f"  Output         : {output_dir}")
    print(f"{'='*60}\n")

    policy, ref_model, tokenizer = load_for_dpo(sft_checkpoint, base_model)
    dataset = prepare_dpo_dataset(data_path)

    dpo_config = DPOConfig(
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

        # DPO-specific
        beta=beta,
        loss_type=loss_type,
        max_length=max_length,
        max_prompt_length=max_prompt_length,

        # Optional: use reference-free DPO variant if ref_model is None
        # reference_free=True,
    )

    trainer = DPOTrainer(
        model=policy,
        ref_model=ref_model,
        tokenizer=tokenizer,
        args=dpo_config,
        train_dataset=dataset,
    )

    print("Starting DPO training...")
    trainer.train()

    print(f"\nSaving DPO model to {output_dir}/final")
    trainer.save_model(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")
    print("DPO complete.")

    return trainer


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-checkpoint", required=True)
    parser.add_argument("--data",           default="data/dpo_train.jsonl")
    parser.add_argument("--output-dir",     default="outputs/dpo")
    parser.add_argument("--epochs",         type=int,   default=1)
    parser.add_argument("--lr",             type=float, default=1e-6)
    parser.add_argument("--batch-size",     type=int,   default=2)
    parser.add_argument("--beta",           type=float, default=0.1)
    parser.add_argument("--loss-type",      default="sigmoid",
                        choices=["sigmoid", "ipo", "hinge", "kto_pair"])
    parser.add_argument("--base-model",     default=None)
    args = parser.parse_args()

    run_dpo(
        sft_checkpoint=args.sft_checkpoint,
        data_path=args.data,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        beta=args.beta,
        loss_type=args.loss_type,
        base_model=args.base_model,
    )
