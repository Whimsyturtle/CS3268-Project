"""Build ReFT-r1's neutral examples: ordinary prompts paired with Gemma's own unsteered replies.

AxBench trains ReFT-r1 on these alongside the positive examples (`train_on_negative: true` in
axbench/sweep/wuzhengx/2b/l10/lsreft.yaml). Gemma's replies are what it already says, so the
vector can only fit them by leaving ordinary text unchanged. The prompt sources and the 70/15/15
mix follow AxBench Appendix I and footnote 6.

Run with `uv run python -m steering.reft_r1.neutral`.
"""

import json
import random
import re

from datasets import load_dataset

from evaluation import alpaca_eval
from evaluation.runner import ROOT, load_model
from steering.reft_r1.main import (
    MAX_NEW_TOKENS,
    NEUTRAL_EXAMPLES,
    config,
    run_unsteered,
)

# name: (dataset, subset, revision, prompt column, column that must be empty, share).
# Rows with extra context or input are skipped, since the prompt alone would not make sense.
SOURCES = {
    "dolly": ("databricks/databricks-dolly-15k", None, "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a",
              "instruction", "context", 0.70),
    "python_code": ("iamtarun/python_code_instructions_18k_alpaca", None,
                    "7cae181e29701a8663a07a3ea43c8e105b663ba1", "instruction", "input", 0.15),
    "gsm8k": ("openai/gsm8k", "main", "740312add88f781978c0658806c59bc2815b9866",
              "question", None, 0.15),
}
# A reply this close to the length limit may have been cut off. Leaving it open only skips
# grading its end-of-turn token, so the margin errs on the safe side.
CUT_OFF_MARGIN = 8


def normalize(text: str) -> str:
    """Lowercase and keep only letters and digits, so trivial differences do not hide a match."""
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def sample_prompts(total: int, seed: int) -> list[dict[str, str]]:
    """Pick `total` prompts in AxBench's mix, skipping any that appear in AlpacaEval.

    Args:
        total: How many prompts to return.
        seed: Seed for the random choice.

    Returns:
        Records with `id`, `source` and `prompt`, grouped by source.
    """
    alpaca_config = config["benchmarks"]["alpaca_eval"]
    excluded = {normalize(alpaca_eval.format_prompt(example, alpaca_config))
                for example in alpaca_eval.load_examples(alpaca_config)}
    counts = {name: round(source[-1] * total) for name, source in SOURCES.items()}
    counts["dolly"] += total - sum(counts.values())
    rng = random.Random(seed)

    records = []
    for name, (dataset, subset, revision, prompt_column, empty_column, _share) in SOURCES.items():
        rows = load_dataset(dataset, subset, split="train", revision=revision)
        candidates, seen = [], set(excluded)
        for index, row in enumerate(rows):
            key = normalize(row[prompt_column])
            if key and key not in seen and not (empty_column and row[empty_column].strip()):
                seen.add(key)
                candidates.append(index)
        for index in sorted(rng.sample(candidates, counts[name])):
            records.append({"id": f"{name}-{index}", "source": name,
                            "prompt": rows[index][prompt_column].strip()})
    return records


if __name__ == "__main__":
    construction = (ROOT / "shared_examples" / "construction.jsonl").read_text().splitlines()
    records = sample_prompts(len(construction), config["seed"])
    model, tokenizer = load_model(config["model"])
    replies = run_unsteered(model, tokenizer, [record["prompt"] for record in records])
    with NEUTRAL_EXAMPLES.open("w") as file:
        for record, reply in zip(records, replies, strict=True):
            n_tokens = len(tokenizer(reply, add_special_tokens=False).input_ids)
            complete = n_tokens < MAX_NEW_TOKENS - CUT_OFF_MARGIN
            file.write(json.dumps({**record, "response": reply, "complete": complete}) + "\n")
    print(f"Wrote {len(records)} neutral examples to {NEUTRAL_EXAMPLES}")
