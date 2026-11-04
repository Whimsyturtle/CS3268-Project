"""Score and generate blinded review outputs for ICV strength selection."""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F

from evaluation.runner import generate, load_model, source_commit
from steering.icv import apply, load_intervention
from steering.icv.build import BBQ_DIRECT


STRENGTHS = (0.0, 0.1, 0.25, 0.5)


def _read_records(path):
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len({record["id"] for record in records}) != len(records):
        raise ValueError("development example IDs must be unique")
    return records


def _plans(records):
    """Return variant plans as lists of (artifact name, records) groups."""
    mapped = set(BBQ_DIRECT.values())
    direct_artifact = {category: name for name, category in BBQ_DIRECT.items()}
    mapped_records = [record for record in records if record["category"] in mapped]
    routed = defaultdict(list)
    for record in mapped_records:
        routed[direct_artifact[record["category"]]].append(record)
    if set(routed) != set(BBQ_DIRECT):
        raise ValueError("development examples do not cover every routed BBQ category")
    return {
        "pooled_all": [("pooled_all", mapped_records)],
        "pooled_bbq_axes": [("pooled_bbq_axes", mapped_records)],
        "category_routed": sorted(routed.items()),
    }


def _response_inputs(tokenizer, prompt, response):
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )["input_ids"][0]
    response_ids = tokenizer(response, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    if response_ids.numel() == 0:
        raise ValueError("development responses must not be empty")
    input_ids = torch.cat((prompt_ids, response_ids))
    response_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    response_mask[prompt_ids.numel():] = True
    return input_ids, response_mask


@torch.no_grad()
def _mean_log_likelihoods(model, tokenizer, prompts, responses, batch_size):
    """Return mean log likelihood over response tokens only."""
    encoded = [_response_inputs(tokenizer, prompt, response) for prompt, response in zip(prompts, responses, strict=True)]
    scores = []
    for start in range(0, len(encoded), batch_size):
        batch = encoded[start:start + batch_size]
        length = max(input_ids.numel() for input_ids, _ in batch)
        input_ids = torch.full(
            (len(batch), length), tokenizer.pad_token_id, dtype=torch.long, device=model.device
        )
        attention_mask = torch.zeros_like(input_ids)
        response_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for row, (tokens, mask) in enumerate(batch):
            input_ids[row, :tokens.numel()] = tokens.to(model.device)
            attention_mask[row, :tokens.numel()] = 1
            response_mask[row, :tokens.numel()] = mask.to(model.device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits[:, :-1]
        targets = input_ids[:, 1:]
        target_mask = response_mask[:, 1:]
        token_scores = F.log_softmax(logits.float(), dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        scores.extend(((token_scores * target_mask).sum(1) / target_mask.sum(1)).cpu().tolist())
    return scores


def _summary_rows(variant, strength, margins):
    rows = []
    groups = {"all": margins}
    groups.update({
        category: [row for row in margins if row["category"] == category]
        for category in sorted({row["category"] for row in margins})
    })
    for subgroup, values in groups.items():
        rows.append({
            "variant": variant,
            "strength": strength,
            "subgroup": subgroup,
            "n": len(values),
            "preference_accuracy": sum(row["margin"] > 0 for row in values) / len(values),
            "mean_margin": sum(row["margin"] for row in values) / len(values),
        })
    return rows


def _blind_generations(generations, seed):
    generations = list(generations)
    random.Random(seed).shuffle(generations)
    public, key = [], []
    for index, row in enumerate(generations):
        review_id = f"icv-{index:04d}"
        public.append({
            "review_id": review_id,
            **{name: value for name, value in row.items() if name not in {"variant", "strength"}},
        })
        key.append({
            "review_id": review_id,
            "variant": row["variant"],
            "strength": row["strength"],
        })
    return public, key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/main.json"))
    parser.add_argument("--vectors", type=Path, default=Path("steering_vectors/icv/bbq_variants"))
    parser.add_argument("--examples", type=Path, default=Path("shared_examples/development.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/icv_tuning"))
    parser.add_argument("--strengths", type=float, nargs="+", default=STRENGTHS)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_new_tokens < 1:
        parser.error("batch size and max new tokens must be positive")
    if len(set(args.strengths)) != len(args.strengths) or any(value < 0 for value in args.strengths):
        parser.error("strengths must be unique and nonnegative")
    args.output.mkdir(parents=True)
    source = source_commit()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    records = _read_records(args.examples)
    plans = _plans(records)
    model, tokenizer = load_model(config["model"])
    artifact_names = {artifact for groups in plans.values() for artifact, _ in groups}
    interventions = {
        name: load_intervention(args.vectors / name, model=model)
        for name in artifact_names
    }

    score_rows = []
    generations = []
    decoding = {
        "batch_size": args.batch_size,
        "do_sample": False,
        "max_new_tokens": args.max_new_tokens,
    }
    manifest = {
        "source": source,
        "config_path": str(args.config),
        "vectors": str(args.vectors),
        "examples": str(args.examples),
        "development_example_ids": [record["id"] for record in plans["pooled_all"][0][1]],
        "model": config["model"],
        "seed": config["seed"],
        "strengths": args.strengths,
        "decoding": decoding,
        "interventions": {
            name: json.loads((args.vectors / name / "config.json").read_text(encoding="utf-8"))
            for name in sorted(artifact_names)
        },
    }
    (args.output / "config.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    for variant, groups in plans.items():
        for strength in args.strengths:
            margins = []
            for artifact, group in groups:
                prompts = [record["prompt"] for record in group]
                with apply(model, interventions[artifact], strength=strength):
                    biased = _mean_log_likelihoods(
                        model, tokenizer, prompts, [record["biased_response"] for record in group], args.batch_size
                    )
                    unbiased = _mean_log_likelihoods(
                        model, tokenizer, prompts, [record["unbiased_response"] for record in group], args.batch_size
                    )
                    outputs = [] if strength == 0 else generate(
                        model, tokenizer, prompts, decoding=decoding, seed=config["seed"]
                    )
                for record, biased_score, unbiased_score in zip(group, biased, unbiased, strict=True):
                    margins.append({"category": record["category"], "margin": unbiased_score - biased_score})
                if outputs:
                    for record, output in zip(group, outputs, strict=True):
                        generations.append({
                            "variant": variant,
                            "strength": strength,
                            "example_id": record["id"],
                            "category": record["category"],
                            "prompt": record["prompt"],
                            "output": output,
                        })
            score_rows.extend(_summary_rows(variant, strength, margins))
            print(f"completed {variant} at strength {strength}", flush=True)

    if 0 in args.strengths:
        baseline_records = plans["pooled_all"][0][1]
        baseline_prompts = [record["prompt"] for record in baseline_records]
        baseline_outputs = generate(
            model, tokenizer, baseline_prompts, decoding=decoding, seed=config["seed"]
        )
        generations.extend({
            "variant": "baseline",
            "strength": 0,
            "example_id": record["id"],
            "category": record["category"],
            "prompt": record["prompt"],
            "output": output,
        } for record, output in zip(baseline_records, baseline_outputs, strict=True))

    with (args.output / "scores.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=score_rows[0].keys())
        writer.writeheader()
        writer.writerows(score_rows)

    public_rows, key_rows = _blind_generations(generations, config["seed"])
    for name, rows in (("generations.jsonl", public_rows), ("generation_key.jsonl", key_rows)):
        with (args.output / name).open("w", encoding="utf-8") as file:
            file.writelines(json.dumps(row) + "\n" for row in rows)


if __name__ == "__main__":
    main()
