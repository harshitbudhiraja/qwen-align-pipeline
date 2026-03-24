#!/usr/bin/env python3
"""
End-to-end pipeline runner for Qwen code-output alignment.

Runs all stages in sequence:
  1. Dataset preparation
  2. SFT (Supervised Fine-Tuning)
  3. RL alignment (GRPO or DPO)
  4. Evaluation
  5. LoRA merge (optional)

Usage:
    # Full pipeline with GRPO (recommended)
    python run_pipeline.py \
        --dataset path/to/your_synthetic_data.jsonl \
        --model   Qwen/Qwen2.5-Coder-7B-Instruct \
        --rl-method grpo

    # DPO instead of GRPO
    python run_pipeline.py --dataset data.jsonl --rl-method dpo

    # Skip SFT (if you already have a checkpoint)
    python run_pipeline.py \
        --dataset data.jsonl \
        --skip-sft \
        --sft-checkpoint outputs/sft/final

    # Only run eval on an existing checkpoint
    python run_pipeline.py --eval-only --sft-checkpoint outputs/grpo/final
"""

import argparse
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def run_pipeline(args):
    print("\n" + "="*65)
    print("  Qwen Code-Output Alignment Pipeline")
    print("="*65 + "\n")

    # ── Stage 1: Data preparation ──────────────────────────────────────
    if not args.skip_data_prep:
        print("\n[Stage 1] Preparing dataset...")
        from prepare_dataset import prepare_datasets
        datasets = prepare_datasets(
            input_path=args.dataset,
            output_dir=args.data_dir,
            dpo_rejection_ratio=0.5,
        )
        sft_data = str(Path(args.data_dir) / "sft_train.jsonl")
        dpo_data = str(Path(args.data_dir) / "dpo_train.jsonl")
    else:
        sft_data = str(Path(args.data_dir) / "sft_train.jsonl")
        dpo_data = str(Path(args.data_dir) / "dpo_train.jsonl")
        print(f"[Stage 1] Skipped. Using existing data in {args.data_dir}/")

    # ── Stage 2: SFT ──────────────────────────────────────────────────
    if not args.skip_sft and not args.eval_only:
        print("\n[Stage 2] Supervised Fine-Tuning (SFT)...")
        from train_sft import run_sft, DEFAULT_CONFIG
        sft_cfg = DEFAULT_CONFIG.copy()
        sft_cfg.update({
            "model_name": args.model,
            "data_path":  sft_data,
            "output_dir": str(Path(args.output_dir) / "sft"),
        })
        run_sft(sft_cfg)
        sft_checkpoint = str(Path(args.output_dir) / "sft" / "final")
    else:
        sft_checkpoint = args.sft_checkpoint
        print(f"[Stage 2] Skipped. Using checkpoint: {sft_checkpoint}")

    # ── Stage 3: RL Alignment ──────────────────────────────────────────
    if not args.eval_only:
        if args.rl_method == "grpo":
            print(f"\n[Stage 3] RL Alignment — GRPO...")
            from train_grpo import run_grpo
            run_grpo(
                sft_checkpoint=sft_checkpoint,
                data_path=sft_data,
                output_dir=str(Path(args.output_dir) / "grpo"),
            )
            final_checkpoint = str(Path(args.output_dir) / "grpo" / "final")

        elif args.rl_method == "dpo":
            print(f"\n[Stage 3] RL Alignment — DPO...")
            from train_dpo import run_dpo
            run_dpo(
                sft_checkpoint=sft_checkpoint,
                data_path=dpo_data,
                output_dir=str(Path(args.output_dir) / "dpo"),
            )
            final_checkpoint = str(Path(args.output_dir) / "dpo" / "final")

        elif args.rl_method == "sft_only":
            print("[Stage 3] Skipped (sft_only mode).")
            final_checkpoint = sft_checkpoint

        else:
            raise ValueError(f"Unknown rl_method: {args.rl_method}")
    else:
        final_checkpoint = sft_checkpoint

    # ── Stage 4: Evaluation ────────────────────────────────────────────
    print(f"\n[Stage 4] Evaluating {final_checkpoint}...")
    from evaluate import run_eval
    result = run_eval(
        checkpoint=final_checkpoint,
        data_path=sft_data,
        n_samples=args.eval_samples,
        output_json=str(Path(args.output_dir) / "eval_results.json"),
    )

    # ── Stage 5: Merge (optional) ──────────────────────────────────────
    if args.merge:
        print(f"\n[Stage 5] Merging LoRA into base model...")
        from merge_lora import merge_and_save
        merge_and_save(
            base_model=args.model,
            lora_checkpoint=final_checkpoint,
            output_dir=str(Path(args.output_dir) / "merged"),
            push_to_hub=args.push_to_hub,
            hub_repo=args.hub_repo,
        )

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  Pipeline Complete")
    print("="*65)
    print(result.pretty())
    print(f"\n  Final checkpoint: {final_checkpoint}")
    print(f"  Eval results   : {args.output_dir}/eval_results.json")
    if args.merge:
        print(f"  Merged model   : {args.output_dir}/merged")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen code-output alignment pipeline")

    # Data
    parser.add_argument("--dataset",   default=None,
                        help="Path to synthetic dataset JSONL")
    parser.add_argument("--data-dir",  default="data",
                        help="Directory for processed datasets")
    parser.add_argument("--output-dir", default="outputs",
                        help="Root directory for all checkpoints")

    # Model
    parser.add_argument("--model",   default="Qwen/Qwen2.5-Coder-7B-Instruct",
                        help="Base model name or path")

    # Pipeline control
    parser.add_argument("--skip-data-prep", action="store_true")
    parser.add_argument("--skip-sft",       action="store_true")
    parser.add_argument("--sft-checkpoint", default=None,
                        help="Existing SFT checkpoint to start RL from")
    parser.add_argument("--rl-method",  default="grpo",
                        choices=["grpo", "dpo", "sft_only"])
    parser.add_argument("--eval-only",  action="store_true",
                        help="Only run evaluation on --sft-checkpoint")
    parser.add_argument("--eval-samples", type=int, default=200)

    # Post-processing
    parser.add_argument("--merge",       action="store_true",
                        help="Merge LoRA into base model after training")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-repo",    default=None)

    args = parser.parse_args()

    if args.dataset is None and not args.skip_data_prep and not args.eval_only:
        parser.error("--dataset is required unless --skip-data-prep or --eval-only")

    run_pipeline(args)
