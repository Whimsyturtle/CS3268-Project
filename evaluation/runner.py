"""Shared runner: loads the model, applies steering, generates, saves predictions, then scores. See README.md for usage."""

import argparse
import contextlib
import csv
import importlib
import json
import subprocess
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]


def load_model(config):
    tokenizer = AutoTokenizer.from_pretrained(config["name"], revision=config["revision"])
    model = AutoModelForCausalLM.from_pretrained(
        config["name"], revision=config["revision"], dtype=getattr(torch, config["dtype"])
    )
    # Transformers does not consistently retain the requested revision on nested model configs.
    model._cs3268_source_revision = config["revision"]
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return model.to(device).eval(), tokenizer


@torch.no_grad()
def generate(model, tokenizer, prompts, *, decoding, seed):
    """Return one chat response per prompt, in order. `decoding` holds `batch_size` and `model.generate` settings."""
    torch.manual_seed(seed)
    settings = dict(decoding)
    batch_size = settings.pop("batch_size")
    tokenizer.padding_side = "left"
    outputs = []
    for start in range(0, len(prompts), batch_size):
        chats = [[{"role": "user", "content": prompt}] for prompt in prompts[start : start + batch_size]]
        inputs = tokenizer.apply_chat_template(
            chats, add_generation_prompt=True, padding=True, return_tensors="pt", return_dict=True
        ).to(model.device)
        tokens = model.generate(**inputs, **settings)
        outputs += tokenizer.batch_decode(tokens[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return outputs


def steering_context(model, method, config):
    if method == "baseline":
        return contextlib.nullcontext()
    steering = importlib.import_module(f"steering.{method}")
    intervention = steering.load_intervention(ROOT / config["intervention_dir"], model=model)
    return steering.apply(model, intervention, strength=config["strength"])


def source_commit():
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()

    return {"commit": git("rev-parse", "HEAD"), "status": git("status", "--porcelain"), "uncommitted_diff": git("diff", "HEAD")}


def run(config_path, *, method, benchmark, run_id):
    config = json.loads(Path(config_path).read_text())
    benchmark_config = {**config["benchmarks"][benchmark], "seed": config["seed"]}
    decoding = {**config["decoding"], "max_new_tokens": benchmark_config["max_new_tokens"]}
    method_config = config["methods"].get(method, {})
    if method != "baseline" and method_config["strength"] is None:
        raise ValueError(f"Set the {method} strength in the config after tuning it on development data.")
    if Path(run_id).name != run_id:
        raise ValueError(f"run_id must be a single directory name, not {run_id!r}.")
    intervention_config = None
    if method != "baseline":
        intervention_config = json.loads((ROOT / method_config["intervention_dir"] / "config.json").read_text())
    run_dir = ROOT / "results" / benchmark / method / run_id
    run_dir.mkdir(parents=True)  # Fails if the run already exists, so runs are never overwritten.

    resolved = {
        "config_path": str(config_path),
        "method": method,
        "benchmark": benchmark,
        "run_id": run_id,
        "source": source_commit(),
        "model": config["model"],
        "method_config": method_config,
        "intervention_config": intervention_config,
        "benchmark_config": benchmark_config,
        "decoding": decoding,
    }
    (run_dir / "config.json").write_text(json.dumps(resolved, indent=2))

    module = importlib.import_module(f"evaluation.{benchmark}")
    examples = module.load_examples(benchmark_config)[: benchmark_config["limit"]]
    prompts = [module.format_prompt(example, benchmark_config) for example in examples]
    model, tokenizer = load_model(config["model"])
    with steering_context(model, method, method_config):
        outputs = generate(model, tokenizer, prompts, decoding=decoding, seed=config["seed"])

    predictions = [{"example_id": e["example_id"], "output": o} for e, o in zip(examples, outputs)]
    with open(run_dir / "predictions.jsonl", "w") as file:
        file.writelines(json.dumps(prediction) + "\n" for prediction in predictions)

    try:
        metrics = module.score(examples, predictions, benchmark_config)
    except NotImplementedError as error:
        print(f"Saved predictions to {run_dir}, but did not score them: {error}")
        return run_dir
    with open(run_dir / "metrics.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["subgroup", "metric", "value", "n"])
        writer.writeheader()
        writer.writerows({**row, "value": "" if row["value"] is None else row["value"]} for row in metrics)
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--method", required=True, choices=["baseline", "caa", "icv", "sae", "reft_r1"])
    parser.add_argument("--benchmark", required=True, choices=["bbq", "holistic_bias", "mmlu_pro", "alpaca_eval"])
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(run(args.config, method=args.method, benchmark=args.benchmark, run_id=args.run_id))
