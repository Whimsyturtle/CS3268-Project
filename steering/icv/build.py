"""Build the ICV artifact from the shared construction examples."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from steering.icv import (
    METHOD,
    UPSTREAM_REVISION,
    construct_from_differences,
    extract_differences,
)


BBQ_DIRECT = {
    "age": "age",
    "disability_status": "disability",
    "gender_identity": "gender",
    "nationality": "nationality_and_immigration",
    "physical_appearance": "appearance",
    "race_ethnicity": "race_and_ethnicity",
    "religion": "religion",
    "ses": "socioeconomic",
    "sexual_orientation": "sexual_orientation",
}

# family_structure: BBQ has no family-structure category.
# geographic: excluded from the primary mapping; used only as a Nationality transfer diagnostic.
# occupational: profession stereotypes are not equivalent to BBQ socioeconomic-status bias.
# political: BBQ has no political category.
BBQ_VARIANTS = {
    "pooled_bbq_axes": tuple(BBQ_DIRECT.values()),
    **{name: (category,) for name, category in BBQ_DIRECT.items()},
    "race_x_gender_components": ("race_and_ethnicity", "gender"),
    "race_x_ses_components": ("race_and_ethnicity", "socioeconomic"),
    "geographic_transfer_to_nationality": ("geographic",),
}


def _read_records(path):
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) < 2:
        raise ValueError("construction examples need at least two records")
    if len({record["id"] for record in records}) != len(records):
        raise ValueError("construction example IDs must be unique")
    return records


def _save_artifact(output, vector, metadata):
    output.mkdir(parents=True, exist_ok=True)
    torch.save(vector, output / "vector.pt")
    (output / "config.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def _metadata(experiment, config, records, vector, *, variant=None):
    metadata = {
        "model": config["name"],
        "revision": config["revision"],
        "layers": list(range(vector.shape[0])),
        "source_example_ids": [record["id"] for record in records],
        "seed": experiment["seed"],
        "method": METHOD,
        "upstream_revision": UPSTREAM_REVISION,
        "source_field": "biased_response",
        "target_field": "unbiased_response",
    }
    if variant is not None:
        metadata["variant"] = variant
        metadata["source_categories"] = sorted({record["category"] for record in records})
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/main.json"))
    parser.add_argument("--examples", type=Path, default=Path("shared_examples/construction.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("steering_vectors/icv"))
    parser.add_argument(
        "--bbq-variants",
        action="store_true",
        help="build pooled, mapped-category, intersection-component, and transfer variants",
    )
    args = parser.parse_args()

    experiment = json.loads(args.config.read_text(encoding="utf-8"))
    config = experiment["model"]
    records = _read_records(args.examples)

    variant_indices = None
    if args.bbq_variants:
        variant_indices = {}
        for name, categories in BBQ_VARIANTS.items():
            indices = [
                index
                for index, record in enumerate(records)
                if categories is None or record["category"] in categories
            ]
            if len(indices) < 2:
                raise ValueError(f"BBQ variant {name!r} needs at least two examples")
            variant_indices[name] = indices

    tokenizer = AutoTokenizer.from_pretrained(config["name"], revision=config["revision"])
    model = AutoModelForCausalLM.from_pretrained(
        config["name"], revision=config["revision"], dtype=getattr(torch, config["dtype"])
    ).to("cuda" if torch.cuda.is_available() else "cpu").eval()

    differences = extract_differences(model, tokenizer, [
        (record["biased_response"], record["unbiased_response"])
        for record in records
    ])

    if not args.bbq_variants:
        vector = construct_from_differences(differences)
        _save_artifact(args.output, vector, _metadata(experiment, config, records, vector))
        return

    for name, indices in variant_indices.items():
        selected = [records[index] for index in indices]
        vector = construct_from_differences(differences[indices])
        metadata = _metadata(experiment, config, selected, vector, variant=name)
        _save_artifact(args.output / "bbq_variants" / name, vector, metadata)


if __name__ == "__main__":
    main()
