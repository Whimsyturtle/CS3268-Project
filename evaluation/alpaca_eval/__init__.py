"""AlpacaEval 2.0: general response quality on open-ended instructions."""

import json
from pathlib import Path

from huggingface_hub import hf_hub_download

from evaluation.common import outputs_by_id


def load_examples(config):
    path = hf_hub_download(config["dataset"], "alpaca_eval.json", repo_type="dataset", revision=config["revision"])
    rows = json.loads(Path(path).read_text())
    return [{"example_id": str(index), "data": row} for index, row in enumerate(rows)]


def format_prompt(example, config):
    return example["data"]["instruction"]


def score(examples, predictions, config):
    outputs_by_id(examples, predictions)
    raise NotImplementedError("AlpacaEval scoring needs an LLM judge, which the team has not chosen yet.")
