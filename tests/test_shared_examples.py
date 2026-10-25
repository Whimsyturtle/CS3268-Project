from collections import Counter
from pathlib import Path

from shared_examples.validate import validate_local


ROOT = Path(__file__).resolve().parents[1]


def test_shared_examples_are_valid_and_disjoint():
    splits = validate_local(
        ROOT / "shared_examples" / "construction.jsonl",
        ROOT / "shared_examples" / "development.jsonl",
    )

    construction_counts = Counter(
        row["category"] for row in splits["construction"]
    )
    development_counts = Counter(row["category"] for row in splits["development"])

    assert len(construction_counts) == 13
    assert set(construction_counts.values()) == {20}
    assert set(development_counts.values()) == {5}
    assert {row["category"] for row in splits["construction"]} == {
        row["category"] for row in splits["development"]
    }
