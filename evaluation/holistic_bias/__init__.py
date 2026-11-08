"""HolisticBias generation bias evaluation (Smith et al., 2022)."""

import os
import random
import re
from pathlib import Path

from datasets import load_dataset

from evaluation.holistic_bias.scoring import classify_and_score

NONE = "(none)"


def _cell_key(row):
    """Return the paper's descriptor-template cell, excluding noun-only controls."""
    axis = row["axis"]
    descriptor = row["descriptor"]
    if axis in (None, "null") or descriptor in (None, NONE):
        return None
    return axis, descriptor, row["template"]


def _noun_phrase_rank(row, row_index):
    """Prefer a consistent human noun, with a stable fallback for fixed phrases."""
    noun = row["noun"]
    if noun == "person":
        noun_priority = 0
    elif noun in {"woman", "man"}:
        noun_priority = 1
    elif noun not in (None, NONE):
        noun_priority = 2
    else:
        noun_priority = 3
    return (
        noun_priority,
        str(noun).casefold(),
        row["noun_phrase"].casefold(),
        row["plural_noun_phrase"].casefold(),
        row_index,
    )


def _select_descriptor_template_grid(rows):
    """Select one canonical noun phrase for every compatible descriptor-template cell."""
    selected = {}
    for row_index, row in enumerate(rows):
        key = _cell_key(row)
        if key is None:
            continue
        candidate = (_noun_phrase_rank(row, row_index), row_index, dict(row))
        if key not in selected or candidate[0] < selected[key][0]:
            selected[key] = candidate
    if not selected:
        raise ValueError("HolisticBias contains no descriptor-template cells.")
    return [(selected[key][1], selected[key][2]) for key in sorted(selected)]


def _apply_variations(row, rng):
    """Apply the four random stylistic variations from the released sentence generator."""
    row = dict(row)
    text = row["text"]
    noun = row["noun"]
    noun_phrase = (
        row["plural_noun_phrase"] if "{plural_noun_phrase}" in row["template"] else row["noun_phrase"]
    )
    variations = {
        "remove_im_contraction": rng.random() < 0.5,
        "remove_descriptor_hyphens": rng.random() < 0.5,
        "lowercase_descriptor": rng.random() < 0.5,
        "remove_final_period": rng.random() < 0.5,
    }
    variant_noun_phrase = noun_phrase
    if variations["remove_im_contraction"]:
        text = re.sub(r"\bI'm\b", "I am", text)
    if variations["remove_descriptor_hyphens"]:
        if "-" in noun:
            raise ValueError("A noun contains a hyphen, so the released variation would also alter the noun.")
        changed = variant_noun_phrase.replace("-", " ")
        text = text.replace(variant_noun_phrase, changed)
        variant_noun_phrase = changed
    if variations["lowercase_descriptor"]:
        if noun != noun.lower():
            raise ValueError("A noun is not lowercase, so the released variation would also lowercase the noun.")
        changed = variant_noun_phrase.lower()
        text = text.replace(variant_noun_phrase, changed)
        variant_noun_phrase = changed
    if variations["remove_final_period"]:
        text = text.rstrip(".")
    if variant_noun_phrase not in text:
        raise ValueError("The varied noun phrase is not present in its sentence.")
    return {**row, **variations, "variant_noun_phrase": variant_noun_phrase, "text": text}


def load_examples(config):
    """Load one canonical noun phrase for every compatible descriptor-template cell."""
    rows = load_dataset(config["dataset"], "sentences", split=config["split"], revision=config["revision"])
    rng = random.Random(config["seed"])
    examples = []
    for prompt_index, (row_index, row) in enumerate(_select_descriptor_template_grid(rows)):
        row = _apply_variations(row, rng)
        examples.append({"example_id": f"{prompt_index}:{row_index}", "data": row})
    return examples


def validate(config):
    if config["noun_phrase_selection"] != "canonical":
        raise ValueError("HolisticBias noun_phrase_selection must be 'canonical'.")
    if config["limit"] is not None and (not isinstance(config["limit"], int) or config["limit"] <= 0):
        raise ValueError("HolisticBias limit must be a positive integer or null.")
    if not isinstance(config["classifier"]["batch_size"], int) or config["classifier"]["batch_size"] <= 0:
        raise ValueError("HolisticBias classifier batch_size must be a positive integer.")
    python_env = config["classifier"]["python_env"]
    python = os.getenv(python_env)
    if not python:
        raise ValueError(f"Set {python_env} before running HolisticBias.")
    if not Path(python).is_file():
        raise ValueError(f"{python_env} does not identify an existing classifier Python executable.")
    # ParlAI writes .built only after a complete download and extraction.
    model_file = config["classifier"]["model_file"].removeprefix("zoo:")
    model_dir = Path(config["classifier"]["datapath"]).expanduser() / "models" / Path(model_file).parent
    if not (model_dir / ".built").is_file():
        raise ValueError(
            f"Download the HolisticBias classifier to {model_dir} first; see evaluation/holistic_bias/README.md."
        )


def format_prompt(example, config):
    return example["data"]["text"]


def score(examples, predictions, config):
    return classify_and_score(examples, predictions, config)
