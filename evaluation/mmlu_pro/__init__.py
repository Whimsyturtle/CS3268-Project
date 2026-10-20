"""MMLU-Pro: general capability through multiple-choice reasoning questions."""

from datasets import load_dataset

from evaluation.common import format_options, metric, outputs_by_id, parse_choice


def load_examples(config):
    rows = load_dataset(config["dataset"], split=config["split"], revision=config["revision"])
    return [{"example_id": str(row["question_id"]), "data": row} for row in rows]


def format_prompt(example, config):
    return format_options(example["data"]["question"], example["data"]["options"])


def score(examples, predictions, config):
    """Accuracy per subject. Outputs that are not a valid letter count as wrong and are reported as `invalid_rate`."""
    outputs = outputs_by_id(examples, predictions)
    choices_by_subgroup = {"all": []}
    for example in examples:
        row = example["data"]
        choice = parse_choice(outputs[example["example_id"]], len(row["options"]))
        result = {"valid": choice is not None, "correct": choice == row["answer_index"]}
        choices_by_subgroup["all"].append(result)
        choices_by_subgroup.setdefault(row["category"], []).append(result)

    metrics = []
    for subgroup, results in choices_by_subgroup.items():
        n = len(results)
        metrics.append(metric(subgroup, "accuracy", sum(r["correct"] for r in results) / n, n))
        metrics.append(metric(subgroup, "invalid_rate", sum(not r["valid"] for r in results) / n, n))
    return metrics
