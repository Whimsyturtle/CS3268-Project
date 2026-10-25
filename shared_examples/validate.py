"""Validate shared examples and optionally check exact held-out prompt overlap."""

import argparse
import importlib
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FIELDS = {"id", "category", "prompt", "biased_response", "unbiased_response"}
BENCHMARKS = ("bbq", "holistic_bias", "mmlu_pro", "alpaca_eval")
EXPECTED_CATEGORY_COUNT = 13
EXPECTED_PER_CATEGORY = {"construction": 20, "development": 5}


def load_jsonl(path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def normalized(text):
    return " ".join(text.casefold().split())


def validate_local(construction_path, development_path):
    splits = {
        "construction": load_jsonl(construction_path),
        "development": load_jsonl(development_path),
    }
    all_ids = set()
    prompts = {}
    for split_name, rows in splits.items():
        split_ids = set()
        split_prompts = set()
        for line_number, row in enumerate(rows, start=1):
            missing = REQUIRED_FIELDS - row.keys()
            if missing:
                raise ValueError(
                    f"{split_name} line {line_number} is missing {sorted(missing)}"
                )
            valid_fields = all(
                isinstance(row[field], str) and row[field].strip()
                for field in REQUIRED_FIELDS
            )
            if not valid_fields:
                raise ValueError(f"{split_name} line {line_number} has an empty field")
            if row["id"] in split_ids:
                raise ValueError(f"Duplicate ID in {split_name}: {row['id']}")
            split_ids.add(row["id"])
            split_prompts.add(normalized(row["prompt"]))

        if all_ids & split_ids:
            raise ValueError("Construction and development IDs overlap")
        all_ids.update(split_ids)
        prompts[split_name] = split_prompts

        category_counts = Counter(row["category"] for row in rows)
        if len(category_counts) != EXPECTED_CATEGORY_COUNT:
            raise ValueError(
                f"{split_name} has {len(category_counts)} categories, "
                f"expected {EXPECTED_CATEGORY_COUNT}"
            )
        unexpected_counts = {
            category: count
            for category, count in category_counts.items()
            if count != EXPECTED_PER_CATEGORY[split_name]
        }
        if unexpected_counts:
            raise ValueError(
                f"{split_name} categories with unexpected counts: {unexpected_counts}"
            )

    if prompts["construction"] & prompts["development"]:
        raise ValueError("Construction and development prompts overlap")
    return splits


def check_evaluation_overlap(splits, config_path):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    shared_texts = {
        normalized(row[field])
        for rows in splits.values()
        for row in rows
        for field in ("prompt", "biased_response", "unbiased_response")
    }
    for benchmark in BENCHMARKS:
        module = importlib.import_module(f"evaluation.{benchmark}")
        benchmark_config = {**config["benchmarks"][benchmark], "seed": config["seed"]}
        examples = module.load_examples(benchmark_config)
        held_out = {
            normalized(module.format_prompt(example, benchmark_config))
            for example in examples
        }
        overlap = shared_texts & held_out
        if overlap:
            raise ValueError(
                f"{len(overlap)} shared texts overlap exactly with {benchmark} prompts"
            )
        print(f"{benchmark}: checked {len(held_out)} held-out prompts")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-evaluation-overlap", action="store_true")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "main.json")
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT))
    splits = validate_local(
        ROOT / "shared_examples" / "construction.jsonl",
        ROOT / "shared_examples" / "development.jsonl",
    )
    for split_name, rows in splits.items():
        counts = Counter(row["category"] for row in rows)
        print(f"{split_name}: {len(rows)} examples across {len(counts)} categories")
    if args.check_evaluation_overlap:
        check_evaluation_overlap(splits, args.config)


if __name__ == "__main__":
    main()
