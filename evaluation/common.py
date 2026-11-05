"""Helpers shared by the benchmark scorers."""

import re
import string

LETTERS = string.ascii_uppercase
# A letter alone, or followed by ".", ")", ":", "*" or a line break (as in "B.", "(B)", "**B**", "B\nBecause").
CHOICE_PATTERN = re.compile(r"[\s*(]*([A-Z])(?:[.):*\n]|$)")


def parse_choice(output, n_options):
    """Return the 0-based option index the model answered, or None if the output is not a valid letter."""
    match = CHOICE_PATTERN.match(output.strip())
    if match is None or LETTERS.index(match.group(1)) >= n_options:
        return None
    return LETTERS.index(match.group(1))


def format_options(question, options):
    lines = [question] + [f"{LETTERS[i]}. {option}" for i, option in enumerate(options)]
    return "\n".join(lines) + "\n\nAnswer with only the letter of the correct option."


def validate_example_ids(examples):
    """Return the example IDs, rejecting duplicates before generation or scoring."""
    ids = [example["example_id"] for example in examples]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate example IDs")
    return set(ids)


def outputs_by_id(examples, predictions):
    """Map example IDs to outputs, rejecting missing, duplicate, or unexpected prediction IDs."""
    expected = validate_example_ids(examples)
    outputs = {}
    for prediction in predictions:
        if prediction["example_id"] in outputs:
            raise ValueError(f"Duplicate prediction ID: {prediction['example_id']}")
        outputs[prediction["example_id"]] = prediction["output"]
    if outputs.keys() != expected:
        missing, unexpected = expected - outputs.keys(), outputs.keys() - expected
        raise ValueError(f"{len(missing)} missing and {len(unexpected)} unexpected prediction IDs")
    return outputs


def metric(subgroup, name, value, n):
    return {"subgroup": subgroup, "metric": name, "value": value, "n": n}
