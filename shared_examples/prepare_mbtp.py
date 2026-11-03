"""Create balanced English construction and development examples from MBTP."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset


DATASET = "MasahiroKaneko/mbtp"
REVISION = "f212c127c5454883adc873807832581939dff740"
LANGUAGE = "English"
CONSTRUCTION_PER_CATEGORY = 20
DEVELOPMENT_PER_CATEGORY = 5
OUTPUT_DIR = Path(__file__).resolve().parent

CATEGORY_NAMES = {
    "Age": "age",
    "Appearance": "appearance",
    "Disability": "disability",
    "Family structure": "family_structure",
    "Gender": "gender",
    "Geographic": "geographic",
    "Nationality and immigration": "nationality_and_immigration",
    "Occupational": "occupational",
    "Political": "political",
    "Race and ethnicity": "race_and_ethnicity",
    "Religious": "religion",
    "Sexual orientation": "sexual_orientation",
    "Socioeconomic": "socioeconomic",
}


def core_key(row):
    """Identify related MBTP contexts derived from the same stereotype pair."""
    return row["category"], row["pro_core"], row["anti_core"]


def ordered_core_groups(groups):
    """Return source groups in a deterministic, category-independent order."""
    return sorted(
        groups.values(),
        key=lambda group: hashlib.sha256(
            "\0".join(str(value) for value in core_key(group[0][1])).encode()
        ).hexdigest(),
    )


def take_groups(groups, target):
    """Select target records while assigning every touched core to this split."""
    selected = []
    while groups and len(selected) < target:
        selected.extend(groups.pop(0))
    if len(selected) < target:
        raise ValueError(f"Only {len(selected)} records available for a target of {target}")
    return selected[:target]


def convert(source_index, row):
    return {
        "id": f"mbtp-{REVISION[:7]}-{source_index:05d}",
        "category": CATEGORY_NAMES[row["category"]],
        "prompt": row["input"].strip(),
        "biased_response": row["pro_response"].strip(),
        "unbiased_response": row["neutral_response"].strip(),
    }


def prepare(rows):
    by_category = {category: defaultdict(list) for category in CATEGORY_NAMES}
    for source_index, row in enumerate(rows):
        if row["language"] == LANGUAGE and row["category"] in by_category:
            by_category[row["category"]][core_key(row)].append((source_index, row))

    ordered_by_category = {
        category: ordered_core_groups(groups)
        for category, groups in by_category.items()
    }
    construction = []
    for groups in ordered_by_category.values():
        construction.extend(take_groups(groups, CONSTRUCTION_PER_CATEGORY))

    development = []
    for groups in ordered_by_category.values():
        development.extend(take_groups(groups, DEVELOPMENT_PER_CATEGORY))

    construction = sorted(
        (convert(*record) for record in construction), key=lambda row: row["id"]
    )
    development = sorted(
        (convert(*record) for record in development), key=lambda row: row["id"]
    )
    validate(rows, construction, development)
    return construction, development


def source_indices(examples):
    return {int(row["id"].rsplit("-", 1)[1]) for row in examples}


def validate(rows, construction, development):
    construction_ids = {row["id"] for row in construction}
    development_ids = {row["id"] for row in development}
    if len(construction_ids) != len(construction):
        raise ValueError("Construction IDs are not unique")
    if len(development_ids) != len(development):
        raise ValueError("Development IDs are not unique")
    if construction_ids & development_ids:
        raise ValueError("Construction and development IDs overlap")

    construction_prompts = {row["prompt"] for row in construction}
    development_prompts = {row["prompt"] for row in development}
    if construction_prompts & development_prompts:
        raise ValueError("Construction and development prompts overlap")

    construction_indices = source_indices(construction)
    development_indices = source_indices(development)
    construction_cores = {core_key(rows[index]) for index in construction_indices}
    development_cores = {core_key(rows[index]) for index in development_indices}
    if construction_cores & development_cores:
        raise ValueError("Related MBTP core groups cross the split boundary")


def write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main():
    rows = load_dataset(DATASET, split="test", revision=REVISION)
    construction, development = prepare(rows)
    write_jsonl(OUTPUT_DIR / "construction.jsonl", construction)
    write_jsonl(OUTPUT_DIR / "development.jsonl", development)
    print(f"Wrote {len(construction)} construction and {len(development)} development examples")


if __name__ == "__main__":
    main()
