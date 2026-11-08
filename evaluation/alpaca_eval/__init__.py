"""AlpacaEval prompts scored by the configured pairwise LLM judge."""

import json
import os
import tempfile
from pathlib import Path

from huggingface_hub import hf_hub_download

from evaluation.common import metric, outputs_by_id

PROMPT = Path(__file__).with_name("prompt.txt")


def load_examples(config):
    path = hf_hub_download(
        config["dataset"], config["reference_file"], repo_type="dataset", revision=config["revision"]
    )
    rows = json.loads(Path(path).read_text())
    return [{"example_id": str(index), "data": row} for index, row in enumerate(rows)]


def format_prompt(example, config):
    return example["data"]["instruction"]


def validate(config):
    judge = config["judge"]
    if not os.getenv(judge["api_key_env"]):
        raise ValueError(f"Set {judge['api_key_env']} before scoring AlpacaEval.")


def score(examples, predictions, config):
    outputs = outputs_by_id(examples, predictions)
    judge = config["judge"]
    validate(config)
    api_key = os.environ[judge["api_key_env"]]
    import alpaca_eval

    model_outputs = [
        {
            "instruction": example["data"]["instruction"],
            "output": outputs[example["example_id"]],
            "generator": config["generator"],
        }
        for example in examples
    ]
    reference_outputs = [
        {
            "instruction": example["data"]["instruction"],
            "output": example["data"]["output"],
            "generator": example["data"].get("generator", "reference"),
        }
        for example in examples
    ]
    # AlpacaEval 0.6.6 loads named annotator and client configurations from files.
    with tempfile.TemporaryDirectory(dir=config["output_dir"]) as directory:
        client_path = Path(directory) / "clients.yaml"
        client = {"default": [{"api_key": api_key, "base_url": judge["base_url"]}]}
        client_path.write_text(json.dumps(client))
        annotator = {
            "soclaas_weighted": {
                "prompt_template": str(PROMPT),
                "fn_completions": "openai_completions",
                "completions_kwargs": {
                    "model_name": judge["model"],
                    "requires_chatml": True,
                    "max_tokens": 1,
                    "temperature": 1,
                    "logprobs": True,
                    "top_logprobs": 5,
                    "reasoning_effort": judge["reasoning_effort"],
                    "client_config_path": str(client_path),
                },
                "fn_completion_parser": "logprob_parser",
                "completion_parser_kwargs": {
                    "numerator_token": "m",
                    "denominator_tokens": ["m", "M"],
                    "is_binarize": False,
                },
                "completion_key": "completions_all",
                "batch_size": 1,
            }
        }
        annotator_path = Path(directory) / "configs.yaml"
        annotator_path.write_text(json.dumps(annotator))
        leaderboard, _ = alpaca_eval.evaluate(
            model_outputs=model_outputs,
            reference_outputs=reference_outputs,
            annotators_config=annotator_path,
            name=config["generator"],
            output_path=config["output_dir"],
            precomputed_leaderboard=None,
            is_cache_leaderboard=False,
            is_return_instead_of_print=True,
            fn_metric="get_winrate",
            sort_by="win_rate",
        )
    row = leaderboard.loc[config["generator"]]
    names = ["win_rate", "standard_error", "avg_length"]
    return [metric("all", name, float(row[name]), int(row["n_total"])) for name in names]
