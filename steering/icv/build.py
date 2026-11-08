"""Build the ICV artifact from the shared construction examples."""

import argparse
import json
from pathlib import Path

import torch

from cluster_runtime import save_cluster_runtime
from evaluation.runner import load_model
from steering.icv import METHOD, UPSTREAM_REVISION, construct


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/main.json"))
    parser.add_argument("--examples", type=Path, default=Path("shared_examples/construction.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("steering_vectors/icv"))
    args = parser.parse_args()

    experiment = json.loads(args.config.read_text(encoding="utf-8"))
    config = experiment["model"]
    records = _read_records(args.examples)

    model, tokenizer = load_model(config)
    save_cluster_runtime(model, args.output)

    vector = construct(model, tokenizer, [
        (record["biased_response"], record["unbiased_response"])
        for record in records
    ])
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
    _save_artifact(args.output, vector, metadata)


if __name__ == "__main__":
    main()
