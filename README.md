# Qwen Code-Output Alignment Pipeline

Fine-tune and align a Qwen2.5-Coder model to **always output code in
`\`\`\`python` or `\`\`\`html` fenced blocks** using SFT + GRPO/DPO.

---

## Pipeline overview

```
Synthetic dataset
      │
      ▼
[1] Data prep       → filter compliant examples, build DPO pairs
      │
      ▼
[2] SFT (QLoRA)     → teach the model the format from examples
      │
      ▼
[3a] GRPO           → online RL: model generates, reward penalises wrong format
 or
[3b] DPO            → offline RL: preferred vs rejected pairs
      │
      ▼
[4] Evaluation      → format compliance %, syntax rate, avg reward
      │
      ▼
[5] Merge + deploy  → merge LoRA into base for fast inference
```

---

## Hardware requirements

| Model size | Min VRAM | Recommended |
|------------|----------|-------------|
| 7B (4-bit) | 10 GB    | 24 GB (A10G, 3090) |
| 14B (4-bit)| 18 GB    | 40 GB (A100) |

---

## Quick start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare your dataset
Your synthetic dataset should be a JSONL file with one example per line:
```json
{"prompt": "Write a function that reverses a string", "response": "```python\ndef reverse(s):\n    return s[::-1]\n```"}
{"prompt": "Create a red button", "response": "```html\n<button style='background:red;color:white'>Click me</button>\n```"}
```

### 3. Run the full pipeline
```bash
python run_pipeline.py \
    --dataset your_data.jsonl \
    --model   Qwen/Qwen2.5-Coder-7B-Instruct \
    --rl-method grpo
```

### 4. Run only SFT (no RL)
```bash
python run_pipeline.py \
    --dataset your_data.jsonl \
    --rl-method sft_only
```

### 5. Run individual stages
```bash
# Stage 1: Prepare data
python src/prepare_dataset.py --input your_data.jsonl --output data/

# Stage 2: SFT
python src/train_sft.py --data data/sft_train.jsonl --model Qwen/Qwen2.5-Coder-7B-Instruct

# Stage 3a: GRPO
python src/train_grpo.py --sft-checkpoint outputs/sft/final

# Stage 3b: DPO (alternative)
python src/train_dpo.py --sft-checkpoint outputs/sft/final

# Stage 4: Evaluate
python src/evaluate.py --checkpoint outputs/grpo/final

# Stage 5: Merge LoRA
python src/merge_lora.py \
    --base-model  Qwen/Qwen2.5-Coder-7B-Instruct \
    --lora-checkpoint outputs/grpo/final \
    --output-dir  outputs/merged
```

---

## Reward model

The reward function (`src/reward_model.py`) is **rule-based** — no neural judge needed:

| Component     | Weight | Condition |
|---------------|--------|-----------|
| Fence present | 0.4    | `\`\`\`python` or `\`\`\`html` block exists |
| Syntax valid  | 0.4    | All code blocks parse without errors |
| Content score | 0.2    | Response has reasonable length |

Test the reward on any text:
```python
from src.reward_model import explain_reward
print(explain_reward("```python\ndef foo(): return 1\n```"))
# RewardBreakdown(fence=0.40, syntax=0.40, content=0.16, total=0.96)
```

---

## GRPO vs DPO — which to use?

| | GRPO | DPO |
|---|---|---|
| Data needed | Prompts only | (prompt, chosen, rejected) pairs |
| Training mode | Online (generates rollouts) | Offline |
| GPU memory | Higher (generates during train) | Lower |
| Tuning | Fewer hyperparameters | Need good β |
| **Recommendation** | ✅ Start here | Use if you have quality pairs |

---

## Output structure

```
outputs/
  sft/
    final/          ← SFT LoRA checkpoint
  grpo/
    final/          ← GRPO-aligned LoRA checkpoint
  merged/           ← Merged model (ready for deployment)
  eval_results.json ← Evaluation metrics
data/
  sft_train.jsonl
  dpo_train.jsonl
```

---

## Tips

- **Dataset size**: 1k–5k high-quality examples is usually enough for SFT. GRPO can improve on fewer.
- **Format diversity**: Include a mix of Python-only, HTML-only, and mixed (Python + HTML) examples.
- **Longer training**: If compliance is still below 90% after SFT, run another GRPO epoch before giving up.
- **KL coefficient**: Start with `kl_coef=0.1`. Increase if the model starts generating gibberish; decrease if compliance doesn't improve.
