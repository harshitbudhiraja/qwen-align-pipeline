"""
Stage 2: Supervised Fine-Tuning (SFT) with QLoRA.

Trains Qwen2.5-Coder on your prepared dataset using 4-bit quantization
and LoRA adapters.  Run this BEFORE the RL stage.

Usage:
    python src/train_sft.py --config configs/sft_config.yaml

Or programmatically:
    from src.train_sft import run_sft
    run_sft(model_name="Qwen/Qwen2.5-Coder-7B-Instruct", data_path="data/sft_train.jsonl")
"""

import os
import yaml
import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig


# ── Default configuration ─────────────────────────────────────────────────

DEFAULT_CONFIG = {
    # Model
    "model_name":    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "output_dir":    "outputs/sft",
    "data_path":     "data/sft_train.jsonl",

    # QLoRA
    "load_in_4bit":      True,
    "bnb_4bit_quant":    "nf4",          # nf4 | fp4
    "bnb_4bit_compute":  "bfloat16",
    "lora_r":            16,
    "lora_alpha":        32,
    "lora_dropout":      0.05,
    "lora_target_modules": [             # Qwen2.5 attention + MLP modules
        "q_proj", "k_proj", "v_proj",
        "o_proj", "gate_proj",
        "up_proj", "down_proj",
    ],

    # Training
    "num_train_epochs":       3,
    "per_device_train_batch": 2,
    "gradient_accumulation":  8,
    "learning_rate":          2e-4,
    "warmup_ratio":           0.05,
    "lr_scheduler":           "cosine",
    "max_seq_length":         2048,
    "bf16":                   True,
    "logging_steps":          10,
    "save_steps":             100,
    "save_total_limit":       3,
    "dataloader_workers":     4,
}


# ── Model loading ─────────────────────────────────────────────────────────

def load_model_and_tokenizer(cfg: dict):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg["load_in_4bit"],
        bnb_4bit_quant_type=cfg["bnb_4bit_quant"],
        bnb_4bit_compute_dtype=getattr(torch, cfg["bnb_4bit_compute"].replace("bfloat16", "bfloat16")),
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model.enable_input_require_grads()

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_name"],
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# ── LoRA configuration ────────────────────────────────────────────────────

def apply_lora(model, cfg: dict):
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["lora_target_modules"],
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


# ── Main training function ────────────────────────────────────────────────

def run_sft(cfg: dict | None = None):
    if cfg is None:
        cfg = DEFAULT_CONFIG

    print(f"\n{'='*60}")
    print(f"  SFT Training: {cfg['model_name']}")
    print(f"  Data        : {cfg['data_path']}")
    print(f"  Output      : {cfg['output_dir']}")
    print(f"{'='*60}\n")

    # Load dataset
    dataset = load_dataset("json", data_files=cfg["data_path"], split="train")
    if "text" in dataset.column_names:
        dataset = dataset.select_columns(["text"])
    print(f"Loaded {len(dataset)} training examples.")

    # Load model + tokenizer
    model, tokenizer = load_model_and_tokenizer(cfg)

    # Apply LoRA
    model = apply_lora(model, cfg)

    # TRL SFTConfig
    training_args = SFTConfig(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch"],
        gradient_accumulation_steps=cfg["gradient_accumulation"],
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg["warmup_ratio"],
        lr_scheduler_type=cfg["lr_scheduler"],
        bf16=cfg["bf16"],
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        save_total_limit=cfg["save_total_limit"],
        dataloader_num_workers=cfg["dataloader_workers"],
        report_to="none",
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset,
    )

    print("Starting SFT training...")
    trainer.train()

    print(f"\nSaving final model to {cfg['output_dir']}/final")
    trainer.save_model(f"{cfg['output_dir']}/final")
    tokenizer.save_pretrained(f"{cfg['output_dir']}/final")
    print("SFT complete.")

    return trainer


# ── CLI entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config (optional; falls back to defaults)")
    parser.add_argument("--model",  type=str, default=None)
    parser.add_argument("--data",   type=str, default=None)
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG.copy()
    if args.config:
        with open(args.config) as f:
            cfg.update(yaml.safe_load(f))
    if args.model:
        cfg["model_name"] = args.model
    if args.data:
        cfg["data_path"] = args.data

    run_sft(cfg)
