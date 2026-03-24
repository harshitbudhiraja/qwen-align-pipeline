"""
Dataset preparation for Qwen code-output alignment.

Loads your synthetic dataset, filters/validates format compliance,
and converts to the SFT and DPO formats required by TRL.

Expected input format (JSONL):
  {"prompt": "...", "response": "...", "label": "python|html|both|other"}

Outputs:
  - data/sft_train.jsonl  — (prompt, chosen) pairs for SFT
  - data/dpo_train.jsonl  — (prompt, chosen, rejected) pairs for DPO
"""

import json
import re
import ast
import random
from pathlib import Path
from typing import Optional
from datasets import Dataset


# ── Format detection ────────────────────────────────────────────────────────

def has_python_block(text: str) -> bool:
    """Check if text contains a fenced Python code block."""
    return bool(re.search(r"```python\s[\s\S]*?```", text, re.IGNORECASE))


def has_html_block(text: str) -> bool:
    """Check if text contains a fenced HTML code block."""
    return bool(re.search(r"```html\s[\s\S]*?```", text, re.IGNORECASE))


def is_valid_python(text: str) -> bool:
    """Extract and attempt to parse all Python blocks."""
    blocks = re.findall(r"```python\s([\s\S]*?)```", text, re.IGNORECASE)
    if not blocks:
        return False
    for block in blocks:
        try:
            ast.parse(block)
        except SyntaxError:
            return False
    return True


def is_valid_html(text: str) -> bool:
    """Minimal HTML validity — checks for at least one opening tag."""
    blocks = re.findall(r"```html\s([\s\S]*?)```", text, re.IGNORECASE)
    if not blocks:
        return False
    for block in blocks:
        if not re.search(r"<[a-zA-Z][^>]*>", block):
            return False
    return True


def is_compliant(response: str) -> bool:
    """
    Bypass compliance for now, so we can extract user prompts for GRPO.
    """
    return True


# ── Rejection sampling to build non-compliant variants ───────────────────────

CORRUPT_TEMPLATES = [
    # Plain text dump with no code fences
    lambda code: code.strip().replace("```python", "").replace("```html", "").replace("```", ""),
    # Wrong fence language tag
    lambda code: re.sub(r"```python", "```js", code),
    lambda code: re.sub(r"```html", "```xml", code),
    # Missing closing fence
    lambda code: re.sub(r"```\n(?!python|html)", "\n", code, count=1),
]


def make_rejected(response: str) -> str:
    """Create a non-compliant variant for DPO rejected column."""
    fn = random.choice(CORRUPT_TEMPLATES)
    return fn(response)


# ── Main processing ────────────────────────────────────────────────────────

def build_chat_prompt(prompt: str) -> str:
    """
    Wrap a plain prompt in the Qwen chat template format.
    Adjust if your base model uses a different template.
    """
    system = (
        "You are a coding assistant. "
        "Always wrap Python code in ```python\\n...\\n``` fences "
        "and HTML in ```html\\n...\\n``` fences. "
        "Never output raw code without fences."
    )
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def prepare_datasets(
    input_path: str,
    output_dir: str = "data",
    dpo_rejection_ratio: float = 0.5,
    seed: int = 42,
) -> dict:
    """
    Load, filter, and split the synthetic dataset.

    Args:
        input_path:           Path to your JSONL synthetic dataset.
        output_dir:           Where to write sft_train.jsonl / dpo_train.jsonl.
        dpo_rejection_ratio:  Fraction of SFT examples to also include in DPO
                              (with synthetically generated rejected responses).
        seed:                 Random seed for reproducibility.

    Returns:
        dict with "sft" and "dpo" HuggingFace Dataset objects.
    """
    random.seed(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sft_rows, dpo_rows, skipped = [], [], 0

    with open(input_path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[warn] Line {i}: JSON parse error — {e}")
                skipped += 1
                continue

            prompt   = row.get("prompt", "")
            response = row.get("response", "")
            
            if not prompt and "conversation" in row:
                conv = row["conversation"]
                student_msgs = [m for m in conv if m["role"] == "student"]
                if student_msgs:
                    prompt = student_msgs[0]["content"].strip()
                tutor_msgs = [m for m in conv if m["role"] == "tutor"]
                if tutor_msgs:
                    response = tutor_msgs[-1]["content"].strip()
            
            prompt = prompt.strip()
            response = response.strip()

            if not prompt or not response:
                skipped += 1
                continue

            if not is_compliant(response):
                skipped += 1
                continue

            chat_prompt = build_chat_prompt(prompt)

            # SFT row: flat text for causal-LM training
            sft_rows.append({
                "text": chat_prompt + response + "<|im_end|>",
                "prompt": chat_prompt,
                "chosen": response,
            })

            # DPO row: only a fraction (avoids 100% synthetic rejections)
            if random.random() < dpo_rejection_ratio:
                rejected = make_rejected(response)
                dpo_rows.append({
                    "prompt": chat_prompt,
                    "chosen": response,
                    "rejected": rejected,
                })

    print(f"\nDataset stats:")
    print(f"  SFT examples  : {len(sft_rows)}")
    print(f"  DPO pairs     : {len(dpo_rows)}")
    print(f"  Skipped       : {skipped}")

    sft_dataset = Dataset.from_list(sft_rows)
    dpo_dataset = Dataset.from_list(dpo_rows)

    sft_dataset.to_json(str(out / "sft_train.jsonl"))
    dpo_dataset.to_json(str(out / "dpo_train.jsonl"))

    return {"sft": sft_dataset, "dpo": dpo_dataset}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True, help="Path to your synthetic dataset JSONL")
    parser.add_argument("--output", default="data", help="Output directory")
    parser.add_argument("--dpo-ratio", type=float, default=0.5)
    args = parser.parse_args()

    prepare_datasets(args.input, args.output, args.dpo_ratio)
