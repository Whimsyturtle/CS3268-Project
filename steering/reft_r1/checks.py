"""Side checks on the trained ReFT-r1 vector. These are not main results.

Run with `uv run python -m steering.reft_r1.checks`. Finished runs are skipped.
Results are saved under results/bbq/ and results/reft_r1_checks/.
"""

import csv
import json
import random
import tempfile
from collections import Counter
from typing import Any, NamedTuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from evaluation import bbq, runner
from evaluation.common import parse_choice
from steering.reft_r1 import SavedVector, apply, load_intervention
from steering.reft_r1.main import Example, config, make_batch

CHECKS_DIR = runner.ROOT / "results" / "reft_r1_checks"
RANDOM_DIR = "steering_vectors/reft_r1_random_control"
PER_GROUP = 25  # Examples per BBQ category and context condition.
SUBSET_DESCRIPTION = "25 per category and context condition, from balanced_subset."
DEV_STRENGTHS = [-1.0, -0.5, 0.0, 0.5, 1.0]


class Check(NamedTuple):
    """One BBQ run on the shared 550-question subset."""

    method: str
    run_id: str
    strength: float | None  # Multiple of max_act. None for the baseline.
    intervention_dir: str | None  # None uses the trained vector.


CHECKS = [
    Check("baseline", "pilot-550", None, None),
    Check("reft_r1", "pilot-550-s0", 0.0, None),
    Check("reft_r1", "pilot-550-s05", 0.5, None),
    Check("reft_r1", "pilot-550-s1", 1.0, None),
    Check("reft_r1", "pilot-550-s2", 2.0, None),
    Check("reft_r1", "check-550-subtract05", -0.5, None),
    Check("reft_r1", "check-550-subtract1", -1.0, None),
    Check("reft_r1", "check-550-subtract2", -2.0, None),
    Check("reft_r1", "check-550-random05", 0.5, RANDOM_DIR),
    Check("reft_r1", "check-550-random1", 1.0, RANDOM_DIR),
]


def balanced_subset(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sample the same number of examples from each category and context condition.

    Args:
        examples: All BBQ examples.

    Returns:
        The subset, in a fixed order.
    """
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for example in examples:
        row = example["data"]
        groups.setdefault((row["category"], row["context_condition"]), []).append(
            example
        )
    rng = random.Random(0)
    return [e for key in sorted(groups) for e in rng.sample(groups[key], PER_GROUP)]


def save_random_control() -> None:
    """Save a random unit vector with the trained vector's scale and layer."""
    trained_dir = runner.ROOT / config["methods"]["reft_r1"]["intervention_dir"]
    trained = torch.load(trained_dir / "vector.pt", weights_only=True)
    generator = torch.Generator().manual_seed(config["seed"])
    vector = torch.randn(trained["vector"].shape, generator=generator)
    control = SavedVector(
        vector=vector / vector.norm(),
        max_act=trained["max_act"],
        layer=trained["layer"],
    )
    directory = runner.ROOT / RANDOM_DIR
    directory.mkdir(exist_ok=True)
    torch.save(control, directory / "vector.pt")
    settings = {
        "model": config["model"],
        "layer": trained["layer"],
        "max_act": trained["max_act"],
        "seed": config["seed"],
        "control": "Random direction, not trained. Not a main result.",
    }
    (directory / "config.json").write_text(json.dumps(settings, indent=2) + "\n")


def run_check(check: Check) -> None:
    """Run one BBQ check through the shared runner, unless it already exists.

    Args:
        check: The run to do.
    """
    if (runner.ROOT / "results" / "bbq" / check.method / check.run_id).exists():
        return
    run_config = json.loads((runner.ROOT / "configs" / "main.json").read_text())
    run_config["benchmarks"]["bbq"]["subset"] = SUBSET_DESCRIPTION
    if check.strength is not None:
        run_config["methods"]["reft_r1"]["strength"] = check.strength
    if check.intervention_dir is not None:
        run_config["methods"]["reft_r1"]["intervention_dir"] = check.intervention_dir
    with tempfile.NamedTemporaryFile("w", suffix=".json") as file:
        json.dump(run_config, file)
        file.flush()
        print(
            runner.run(
                file.name, method=check.method, benchmark="bbq", run_id=check.run_id
            )
        )


@torch.no_grad()
def log_prob_per_token(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    example: Example,
    response: str,
) -> float:
    """Measure how likely the model finds a response, per token.

    Args:
        model: The model, steered or not.
        tokenizer: The model's tokenizer.
        example: The example giving the prompt.
        response: The response to measure.

    Returns:
        The mean log-probability of the response tokens.
    """
    batch = make_batch(
        tokenizer, [{**example, "unbiased_response": response}], model.device
    )
    hidden = model.model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    ).last_hidden_state
    targets = batch["labels"][:, 1:]
    graded = targets != -100
    logits = model.lm_head(hidden[:, :-1][graded]).float()
    log_probs = torch.log_softmax(logits, dim=-1)
    return log_probs.gather(1, targets[graded][:, None]).mean().item()


def development_rows() -> list[dict[str, float]]:
    """Compare biased and unbiased response likelihoods on the development examples.

    Returns:
        One row per strength: mean log-probabilities and how many pairs prefer unbiased.
    """
    path = runner.ROOT / "shared_examples" / "development.jsonl"
    examples: list[Example] = [
        json.loads(line) for line in path.read_text().splitlines()
    ]
    model, tokenizer = runner.load_model(config["model"])
    directory = runner.ROOT / config["methods"]["reft_r1"]["intervention_dir"]
    intervention = load_intervention(directory, model=model)
    rows = []
    for strength in DEV_STRENGTHS:
        with apply(model, intervention, strength=strength):
            unbiased = [
                log_prob_per_token(model, tokenizer, e, e["unbiased_response"])
                for e in examples
            ]
            biased = [
                log_prob_per_token(model, tokenizer, e, e["biased_response"])
                for e in examples
            ]
        rows.append(
            {
                "strength": strength,
                "unbiased": sum(unbiased) / len(unbiased),
                "biased": sum(biased) / len(biased),
                "prefers_unbiased": sum(u > b for u, b in zip(unbiased, biased)),
            }
        )
    return rows


def bbq_row(check: Check, rows_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Summarize one BBQ run.

    Args:
        check: The run to summarize.
        rows_by_id: BBQ rows by example ID.

    Returns:
        The run's main metrics, plus how often it chose "unknown" and each letter.
    """
    run_dir = runner.ROOT / "results" / "bbq" / check.method / check.run_id
    with open(run_dir / "metrics.csv") as file:
        metrics = {
            r["metric"]: r["value"]
            for r in csv.DictReader(file)
            if r["subgroup"] == "all"
        }
    unknown = Counter()
    valid = Counter()
    letters = Counter()
    for line in (run_dir / "predictions.jsonl").read_text().splitlines():
        prediction = json.loads(line)
        row = rows_by_id[prediction["example_id"]]
        choice = parse_choice(prediction["output"], 3)
        if choice is not None:
            valid[row["context_condition"]] += 1
            letters["ABC"[choice]] += 1
            unknown[row["context_condition"]] += (
                row["answer_info"][f"ans{choice}"][1] == "unknown"
            )
    return {
        "run": check.run_id,
        "metrics": metrics,
        "unknown": {
            c: unknown[c] / valid[c] if valid[c] else None
            for c in ("ambig", "disambig")
        },
        "letters": dict(letters),
    }


def number(value: str | float | None) -> str:
    """Format a metric for the summary table.

    Args:
        value: A number, or a blank or None when undefined.

    Returns:
        The value to 3 decimal places, or "n/a".
    """
    return "n/a" if value in ("", None) else f"{float(value):.3f}"


def write_summary(
    bbq_rows: list[dict[str, Any]], dev_rows: list[dict[str, float]]
) -> None:
    """Write the results of all checks to summary.md.

    Args:
        bbq_rows: One summary per BBQ run.
        dev_rows: One row per development strength.
    """
    lines = [
        "# ReFT-r1 side checks",
        "",
        "These checks are not main results. They use 550 BBQ test questions (25 per",
        "category and context condition), so do not tune anything on them.",
        "Strength is a multiple of `max_act`. Reproduce with `steering/reft_r1/checks.py`.",
        "The runs' saved configs show `limit: null`. Earlier runs do not record the subset,",
        "so it is defined by their `predictions.jsonl` and `balanced_subset`.",
        "`checks.log` keeps these runs' earlier names, `probe-550-neg*` and `probe-550-random*`.",
        "",
        (
            "| Run | Invalid | Acc. ambig. | Acc. disambig. | Bias ambig. "
            "| Bias disambig. | Unknown ambig. | Unknown disambig. | A / B / C |"
        ),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in bbq_rows:
        metrics = row["metrics"]
        letters = " / ".join(str(row["letters"].get(letter, 0)) for letter in "ABC")
        cells = [
            row["run"],
            number(metrics["invalid_rate"]),
            number(metrics["accuracy_ambiguous"]),
            number(metrics["accuracy_disambiguated"]),
            number(metrics["bias_score_ambiguous"]),
            number(metrics["bias_score_disambiguated"]),
            number(row["unknown"]["ambig"]),
            number(row["unknown"]["disambig"]),
            letters,
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "Development examples, mean log-probability per response token:",
        "",
        "| Strength | Unbiased | Biased | Prefers unbiased (of 65) |",
        "|---|---|---|---|",
    ]
    for row in dev_rows:
        lines.append(
            f"| {row['strength']} | {row['unbiased']:.3f} | {row['biased']:.3f} "
            f"| {row['prefers_unbiased']} |"
        )
    (CHECKS_DIR / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    all_examples = bbq.load_examples(
        {**config["benchmarks"]["bbq"], "seed": config["seed"]}
    )
    subset = balanced_subset(all_examples)
    bbq.load_examples = lambda _config: subset  # The runner then uses only the subset.

    save_random_control()
    CHECKS_DIR.mkdir(exist_ok=True)
    for check in CHECKS:
        run_check(check)

    development = development_rows()
    (CHECKS_DIR / "development.json").write_text(
        json.dumps(development, indent=2) + "\n"
    )
    rows_by_id = {example["example_id"]: example["data"] for example in subset}
    write_summary([bbq_row(check, rows_by_id) for check in CHECKS], development)
    print((CHECKS_DIR / "summary.md").read_text())
