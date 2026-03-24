"""
Rule-based reward function for format-constrained RL training.

This module defines the reward signals used during GRPO training
for generating a specific serialized format with XML-style tags.

Reward breakdown:
  r_code  : 1.0 — <python>...</python> tags present
  r_info  : 1.0 — <info>...</info> tags present
  r_len   : 1.0 — <info> content length is ok

Total reward is in [0, 3].
"""

import re
from dataclasses import dataclass

# ── Individual reward components ───────────────────────────────────────────

def has_code_block(output: str) -> bool:
    """True if response contains a <python> tag pair."""
    return bool(re.search(r"<python>[\s\S]*?</python>", output, re.IGNORECASE))

def has_info_block(output: str) -> bool:
    """True if response contains a <info> tag pair."""
    return bool(re.search(r"<info>[\s\S]*?</info>", output, re.IGNORECASE))

def info_length_ok(output: str, min_tokens: int = 10, max_tokens: int = 500) -> bool:
    """True if info block contains an appropriate number of tokens."""
    matches = re.findall(r"<info>([\s\S]*?)</info>", output, re.IGNORECASE)
    if not matches:
        return False
    # Check all info blocks if multiple exist, but usually we just want at least one healthy block
    text = " ".join(matches)
    tokens = len(text.split())
    return min_tokens <= tokens <= max_tokens


# ── Composite reward ───────────────────────────────────────────────────────

def compute_reward(output: str) -> float:
    """
    Combined reward. This is the function you pass to GRPO.
    """
    score = 0
    if has_code_block(output): score += 1
    if has_info_block(output): score += 1
    if info_length_ok(output): score += 1
    return float(score)

# ── Reward for a batch (used in training loop) ─────────────────────────────

def batch_rewards(responses: list[str]) -> list[float]:
    """Compute rewards for a batch of generated responses."""
    return [compute_reward(r) for r in responses]


# ── Diagnostic breakdown ───────────────────────────────────────────────────

@dataclass
class RewardBreakdown:
    has_code: bool
    has_info: bool
    info_len: bool
    total:    float

    def __repr__(self):
        return (
            f"RewardBreakdown("
            f"has_code={self.has_code}, "
            f"has_info={self.has_info}, "
            f"info_len={self.info_len}, "
            f"total={self.total:.1f})"
        )


def explain_reward(output: str) -> RewardBreakdown:
    """Return a full breakdown for debugging / logging."""
    hc = has_code_block(output)
    hi = has_info_block(output)
    il = info_length_ok(output)
    total = sum([hc, hi, il])
    return RewardBreakdown(has_code=hc, has_info=hi, info_len=il, total=float(total))


def is_compliant(output: str) -> bool:
    """Returns True if it achieves a perfect score."""
    return compute_reward(output) == 3.0

# ── Quick test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    samples = [
        ("Perfect", "<info> Here is the answer which is definitely long enough to hit ten tokens for sure! </info>\n<python>\ndef add(a, b):\n    return a + b\n</python>"),
        ("No info", "<python>\ndef add(a, b):\n    return a + b\n</python>"),
        ("No code", "<info> I can help with that. Here is what we will do exactly right now. </info>"),
        ("Bad length", "<info> short </info>\n<python>\npass\n</python>"),
        ("Nothing", "Sure, here's how you would do that conceptually."),
    ]

    print(f"{'Sample':<20} {'Code':>6} {'Info':>6} {'Len':>6} {'Total':>6}")
    print("-" * 48)
    for name, text in samples:
        bd = explain_reward(text)
        print(f"{name:<20} {str(bd.has_code):>6} {str(bd.has_info):>6} {str(bd.info_len):>6} {bd.total:>6.1f}")
