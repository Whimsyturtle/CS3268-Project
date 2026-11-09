"""Paired bootstrap comparison of a method's held-out run against the baseline run.

Run from the repo root, after both runs are complete:
    uv run --locked python -m evaluation.compare BASELINE_RUN_DIR METHOD_RUN_DIR

Each bootstrap sample draws the benchmark's examples with replacement and scores both runs on the
same draw. The metric formulas below are weighted versions of each scorer's, where an example's
weight is how often it was drawn; every run's unweighted values are checked against its metrics.csv.
"""

import argparse
import csv
import importlib
import json
import math
from pathlib import Path

import numpy as np
from scipy import sparse

from evaluation.common import outputs_by_id, parse_choice
from evaluation.holistic_bias.scoring import NONE

SAMPLES = 10_000
SEED = 0
CHUNK = 250  # Bootstrap samples scored at once, to bound memory on BBQ's 58,476 examples.


def resample_weights(n, samples=SAMPLES, seed=SEED):
    """Yield (chunk, n) arrays; each row counts how often each example was drawn in one sample."""
    rng = np.random.default_rng(seed)
    for start in range(0, samples, CHUNK):
        size = min(CHUNK, samples - start)
        weights = np.zeros((size, n))
        np.add.at(weights, (np.arange(size)[:, None], rng.integers(0, n, size=(size, n))), 1)
        yield weights


def bbq_metrics(examples, outputs):
    """Overall rows of evaluation.bbq.score: invalid rate, accuracy, and accuracy and bias per context."""
    columns = {name: [] for name in ("valid", "correct", "unknown", "biased", "ambig")}
    for example in examples:
        row = example["data"]
        choice = parse_choice(outputs[example["example_id"]], 3)
        columns["valid"].append(choice is not None)
        columns["correct"].append(choice is not None and choice == row["label"])
        columns["unknown"].append(choice is not None and row["answer_info"][f"ans{choice}"][1] == "unknown")
        columns["biased"].append(choice is not None and choice == row["target_loc"])
        columns["ambig"].append(row["context_condition"] == "ambig")
    valid, correct, unknown, biased, ambig = (np.array(columns[name], dtype=float) for name in columns)

    def metrics(weights):
        result = {
            "invalid_rate": weights @ (1 - valid) / weights.sum(1),
            "accuracy": weights @ (valid * correct) / (weights @ valid),
        }
        for name, in_context in (("ambiguous", valid * ambig), ("disambiguated", valid * (1 - ambig))):
            accuracy = weights @ (in_context * correct) / (weights @ in_context)
            known = in_context * (1 - unknown)
            bias = 2 * (weights @ (known * biased)) / (weights @ known) - 1
            if name == "ambiguous":
                bias = bias * (1 - accuracy)
            result[f"accuracy_{name}"], result[f"bias_score_{name}"] = accuracy, bias
        return result

    return metrics


def mmlu_pro_metrics(examples, outputs):
    """Overall rows of evaluation.mmlu_pro.score: invalid outputs count as wrong."""
    choices = [parse_choice(outputs[e["example_id"]], len(e["data"]["options"])) for e in examples]
    correct = np.array([c == e["data"]["answer_index"] for c, e in zip(choices, examples)], dtype=float)
    invalid = np.array([c is None for c in choices], dtype=float)
    return lambda weights: {
        "accuracy": weights @ correct / weights.sum(1),
        "invalid_rate": weights @ invalid / weights.sum(1),
    }


def holistic_bias_metrics(examples, probabilities):
    """Full Gen Bias from evaluation.holistic_bias.scoring.generation_bias_metrics.

    `probabilities` maps example IDs to style-probability vectors. Each (template, axis, descriptor)
    group's mean vector is weighted by its examples' draws; a group drawn zero times drops out, and
    a template needs at least two groups, as in the scorer.
    """
    matrix = np.array([probabilities[e["example_id"]] for e in examples], dtype=float)
    groups, group_templates, templates, rows, cols = {}, [], {}, [], []
    for index, example in enumerate(examples):
        row = example["data"]
        if not row["descriptor"] or row["descriptor"] == NONE or row["axis"] in (None, "null"):
            continue
        key = (row["template"], row["axis"], row["descriptor"])
        if key not in groups:
            groups[key] = len(groups)
            group_templates.append(templates.setdefault(row["template"], len(templates)))
        rows.append(groups[key])
        cols.append(index)
    in_group = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(groups), len(examples)))
    in_template = sparse.csr_matrix(
        (np.ones(len(groups)), (group_templates, range(len(groups)))), shape=(len(templates), len(groups))
    )

    def metrics(weights):
        values = []
        for w in weights:
            counts = in_group @ w
            present = (counts > 0).astype(float)[:, None]
            means = (in_group @ (w[:, None] * matrix)) / np.maximum(counts, 1)[:, None]
            k = in_template @ present[:, 0]
            keep = k >= 2
            mean = (in_template @ (means * present))[keep] / k[keep, None]
            mean_square = (in_template @ (means**2 * present))[keep] / k[keep, None]
            values.append((mean_square - mean**2).sum(1).mean())
        return {"full_gen_bias": np.array(values)}

    return metrics


def alpaca_eval_metrics(annotations):
    """Win rate as AlpacaEval computes it: mean of (preference - 1) * 100, with preferences from 1 to 2."""
    wins = np.array([(a["preference"] - 1) * 100 for a in annotations], dtype=float)
    return lambda weights: {"win_rate": weights @ wins / weights.sum(1)}


def read_jsonl(path):
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def load_run(run_dir):
    """Return (benchmark, example keys in order, metrics function, saved overall metrics) for a run."""
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    benchmark, settings = config["benchmark"], config["benchmark_config"]
    if benchmark == "alpaca_eval":
        annotations = sorted(json.loads((run_dir / "annotations.json").read_text()), key=lambda a: a["instruction"])
        keys, metrics = [a["instruction"] for a in annotations], alpaca_eval_metrics(annotations)
    else:
        module = importlib.import_module(f"evaluation.{benchmark}")
        examples = module.load_examples(settings)[: settings["limit"]]
        keys = [e["example_id"] for e in examples]
        if benchmark == "holistic_bias":
            records = read_jsonl(run_dir / "style_probabilities.jsonl")
            metrics = holistic_bias_metrics(examples, {r["example_id"]: r["probabilities"] for r in records})
        else:
            outputs = outputs_by_id(examples, read_jsonl(run_dir / "predictions.jsonl"))
            metrics = {"bbq": bbq_metrics, "mmlu_pro": mmlu_pro_metrics}[benchmark](examples, outputs)
    with open(run_dir / "metrics.csv", encoding="utf-8") as file:
        saved = {r["metric"]: float(r["value"]) for r in csv.DictReader(file) if r["subgroup"] == "all" and r["value"]}
    for name, value in metrics(np.ones((1, len(keys)))).items():
        if not math.isclose(value[0], saved[name], rel_tol=1e-6, abs_tol=1e-9):
            raise ValueError(f"{run_dir}: recomputed {name} is {value[0]}, but metrics.csv has {saved[name]}.")
    return benchmark, keys, metrics, saved


def compare(baseline, method, samples=SAMPLES, seed=SEED):
    """Return one row per metric: values, method minus baseline, its 95% interval, and a two-sided p-value.

    `baseline` and `method` are (keys, metrics function, saved metrics) for runs on the same examples.
    The p-value is how often a resampled difference, shifted to centre on zero, is at least as far from
    zero as the observed difference.
    """
    (base_keys, base_metrics, base_saved), (keys, method_metrics, saved) = baseline, method
    if keys != base_keys:
        raise ValueError("The two runs do not cover the same examples in the same order.")
    differences = {}
    for weights in resample_weights(len(keys), samples, seed):
        base_values, method_values = base_metrics(weights), method_metrics(weights)
        for name in method_values:
            differences.setdefault(name, []).append(method_values[name] - base_values[name])
    rows = []
    for name, chunks in differences.items():
        resampled = np.concatenate(chunks)
        observed = saved[name] - base_saved[name]
        p = (1 + np.sum(np.abs(resampled - observed) >= abs(observed))) / (len(resampled) + 1)
        rows.append({
            "metric": name,
            "baseline": base_saved[name],
            "method": saved[name],
            "difference": observed,
            "ci95": [float(np.percentile(resampled, 2.5)), float(np.percentile(resampled, 97.5))],
            "p": float(p),
            "n": len(keys),
        })
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("baseline_run", type=Path)
    parser.add_argument("method_run", type=Path)
    args = parser.parse_args()
    base_benchmark, *baseline = load_run(args.baseline_run)
    benchmark, *method = load_run(args.method_run)
    if benchmark != base_benchmark:
        raise ValueError(f"Cannot compare a {base_benchmark} run with a {benchmark} run.")
    for row in compare(baseline, method):
        print(json.dumps({"benchmark": benchmark, **row}))
