"""
Stage 5: Evaluation of the aligned model.

Measures:
  - Format compliance rate (% outputs with correct fences)
  - Python syntax pass rate
  - HTML validity rate
  - Average reward score
  - (Optional) HumanEval / pass@k for code correctness

Usage:
    python src/evaluate.py --checkpoint outputs/grpo/final --data data/sft_train.jsonl
"""

import re
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))
from reward_model import explain_reward, is_compliant


# ── Evaluation result dataclass ───────────────────────────────────────────

@dataclass
class EvalResult:
    n_samples:           int
    format_compliance:   float   # % with correct tags
    python_syntax_rate:  float   # % of Python blocks that parse
    avg_reward:          float
    avg_code_score:      float
    avg_info_score:      float
    avg_len_score:       float

    def pretty(self) -> str:
        lines = [
            f"  Samples evaluated     : {self.n_samples}",
            f"  Format compliance     : {self.format_compliance*100:.1f}%",
            f"  Python syntax rate    : {self.python_syntax_rate*100:.1f}%",
            f"  Avg total reward      : {self.avg_reward:.3f}",
            f"    ↳ code tag presence : {self.avg_code_score:.3f}",
            f"    ↳ info tag presence : {self.avg_info_score:.3f}",
            f"    ↳ info length ok    : {self.avg_len_score:.3f}",
        ]
        return "\n".join(lines)


# ── Generation ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a coding assistant. "
    "Always wrap your expected info output in <info>...</info> tags, "
    "and always wrap your output Python code in <python>...</python> tags. "
    "Never output raw code without these tags."
)


def generate_responses(
    checkpoint: str,
    prompts: list[str],
    batch_size: int = 4,
    max_new_tokens: int = 512,
    temperature: float = 0.0,   # greedy by default for deterministic eval
) -> list[str]:
    """Generate responses from the checkpoint for each prompt."""

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()

    responses = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch = prompts[i : i + batch_size]
        formatted = []
        for p in batch:
            # Build chat template
            formatted.append(
                f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{p}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

        inputs = tokenizer(formatted, return_tensors="pt", padding=True,
                           truncation=True, max_length=1024).to(model.device)

        with torch.no_grad():
            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            if temperature > 0:
                gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
            else:
                gen_kwargs["do_sample"] = False

            outputs = model.generate(**inputs, **gen_kwargs)

        # Decode only the generated tokens (not the prompt)
        input_len = inputs["input_ids"].shape[1]
        for out in outputs:
            decoded = tokenizer.decode(out[input_len:], skip_special_tokens=True)
            responses.append(decoded.strip())

    return responses


# ── Metric computation ────────────────────────────────────────────────────

def compute_metrics(responses: list[str]) -> EvalResult:
    import ast

    n = len(responses)
    compliant  = 0
    py_total, py_pass = 0, 0

    reward_sums = defaultdict(float)

    for r in responses:
        if is_compliant(r):
            compliant += 1

        # Python syntax
        py_blocks = re.findall(r"<python>([\s\S]*?)</python>", r, re.IGNORECASE)
        for blk in py_blocks:
            py_total += 1
            try:
                ast.parse(blk)
                py_pass += 1
            except SyntaxError:
                pass

        bd = explain_reward(r)
        reward_sums["total"]    += bd.total
        reward_sums["has_code"] += float(bd.has_code)
        reward_sums["has_info"] += float(bd.has_info)
        reward_sums["info_len"] += float(bd.info_len)

    return EvalResult(
        n_samples          = n,
        format_compliance  = compliant / n if n else 0,
        python_syntax_rate = py_pass / py_total if py_total else 0,
        avg_reward         = reward_sums["total"]    / n,
        avg_code_score     = reward_sums["has_code"] / n,
        avg_info_score     = reward_sums["has_info"] / n,
        avg_len_score      = reward_sums["info_len"] / n,
    )


# ── Main eval loop ────────────────────────────────────────────────────────

def run_eval(
    checkpoint: str,
    data_path: str,
    n_samples: int  = 200,
    batch_size: int = 4,
    output_json: str | None = None,
):
    print(f"\nEvaluating {checkpoint}")
    print(f"Data: {data_path} (first {n_samples} examples)\n")

    dataset = load_dataset("json", data_files=data_path, split="train")
    dataset = dataset.select(range(min(n_samples, len(dataset))))

    # Extract raw prompts (strip any existing assistant prefix)
    prompts = []
    for row in dataset:
        p = row.get("prompt", row.get("text", ""))
        if "<|im_start|>user\n" in p:
            p = p.split("<|im_start|>user\n")[-1].split("<|im_end|>")[0].strip()
        prompts.append(p)

    responses = generate_responses(checkpoint, prompts, batch_size=batch_size)
    result    = compute_metrics(responses)

    print("\nResults:")
    print(result.pretty())

    if output_json:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(asdict(result), f, indent=2)
        print(f"\nResults saved to {output_json}")

    return result


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data",       default="data/sft_train.jsonl")
    parser.add_argument("--n-samples",  type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output",     default=None, help="Save JSON results to this path")
    args = parser.parse_args()

    run_eval(
        checkpoint=args.checkpoint,
        data_path=args.data,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        output_json=args.output,
    )
