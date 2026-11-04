"""Check a constructed SAE intervention on the pinned pretrained Gemma model."""

import argparse
import importlib.metadata
import json
import time
from pathlib import Path

import torch

from evaluation.runner import ROOT, generate, load_model, source_commit
from shared_examples.validate import load_jsonl
from steering.sae import apply, load_intervention


@torch.inference_mode()
def check(config_path, intervention_dir, run_id, strength):
    if Path(run_id).name != run_id or run_id in (".", ".."):
        raise ValueError("run_id must be a single directory name.")
    if strength == 0:
        raise ValueError("The application check needs a nonzero strength.")
    run_dir = ROOT / "results" / "sae_checks" / run_id
    run_dir.mkdir(parents=True)
    start = time.perf_counter()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rows = load_jsonl(ROOT / "shared_examples" / "development.jsonl")[:2]
    prompts = [row["prompt"] for row in rows]
    decoding = {**config["decoding"], "max_new_tokens": config["benchmarks"]["bbq"]["max_new_tokens"]}
    model, tokenizer = load_model(config["model"])
    intervention = load_intervention(intervention_dir, model=model)
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompts[0]}], add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    ).to(model.device)
    baseline_logits = model(**inputs).logits[:, -1].clone()
    baseline = generate(model, tokenizer, prompts, decoding=decoding, seed=config["seed"])
    with apply(model, intervention, strength=0):
        zero_logits = model(**inputs).logits[:, -1].clone()
        zero = generate(model, tokenizer, prompts, decoding=decoding, seed=config["seed"])
    with apply(model, intervention, strength=strength):
        steered_logits = model(**inputs).logits[:, -1].clone()
        steered = generate(model, tokenizer, prompts, decoding=decoding, seed=config["seed"])
    restored_logits = model(**inputs).logits[:, -1].clone()
    restored = generate(model, tokenizer, prompts, decoding=decoding, seed=config["seed"])
    try:
        with apply(model, intervention, strength=strength):
            raise RuntimeError("SAE cleanup check")
    except RuntimeError as error:
        if str(error) != "SAE cleanup check":
            raise
    exception_logits = model(**inputs).logits[:, -1].clone()
    checks = {
        "zero_logits_equal": torch.equal(baseline_logits, zero_logits),
        "zero_generations_equal": baseline == zero,
        "steered_logits_finite": bool(torch.isfinite(steered_logits).all()),
        "steered_logits_changed": not torch.equal(baseline_logits, steered_logits),
        "normal_exit_restored": torch.equal(baseline_logits, restored_logits) and baseline == restored,
        "exception_exit_restored": torch.equal(baseline_logits, exception_logits),
    }
    report = {
        "purpose": "application check only; no tuning or benchmark conclusions",
        "source": source_commit(), "model": config["model"], "intervention": intervention["metadata"],
        "strength": strength, "decoding": decoding, "seed": config["seed"],
        "example_ids": [row["id"] for row in rows], "checks": checks,
        "environment": {name: importlib.metadata.version(name)
                        for name in ["torch", "transformers", "sae-lens", "scikit-learn"]},
        "device": str(model.device),
        "gpu": torch.cuda.get_device_name() if model.device.type == "cuda" else None,
        "elapsed_seconds": time.perf_counter() - start,
        "outputs": {"baseline": baseline, "zero": zero, "steered": steered, "restored": restored},
    }
    (run_dir / "checks.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not all(checks.values()):
        raise RuntimeError(f"SAE application checks failed; inspect {run_dir / 'checks.json'}")
    print(f"SAE application checks passed: {run_dir / 'checks.json'}")
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "main.json")
    parser.add_argument("--intervention-dir", type=Path, default=ROOT / "steering_vectors" / "sae")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    args = parser.parse_args()
    check(args.config, args.intervention_dir, args.run_id, args.strength)
