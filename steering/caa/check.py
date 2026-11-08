"""Check CAA application mechanics on development prompts; this does not measure bias."""

import argparse
import hashlib
import json
import math
import platform
import re
from pathlib import Path
from unittest.mock import patch

import torch
import transformers

from cluster_runtime import save_cluster_runtime
from evaluation.runner import ROOT, generate, load_model, source_commit
from shared_examples.validate import validate_local
from steering.caa import apply, load_intervention


def parameter_state(model):
    """Cheap mutation indicators, without copying the base model's parameters."""
    return {name: (id(value), value.data_ptr(), value._version)
            for name, value in model.named_parameters()}


@torch.inference_mode()
def observe(model, tokenizer, prompts, decoding, seed):
    tokenizer.padding_side = "left"
    inputs = tokenizer.apply_chat_template(
        [[{"role": "user", "content": prompt}] for prompt in prompts],
        tokenize=True, add_generation_prompt=True, padding=True,
        return_tensors="pt", return_dict=True,
    ).to(model.device)
    logits = model(**inputs, use_cache=True, logits_to_keep=1).logits[:, -1].float().cpu()
    generated = []
    original_generate = model.generate

    def capture(**kwargs):
        tokens = original_generate(**kwargs)
        generated.extend(tokens[:, kwargs["input_ids"].shape[1]:].cpu().tolist())
        return tokens

    with patch.object(model, "generate", capture):
        texts = generate(model, tokenizer, prompts, decoding=decoding, seed=seed)
    return {"logits": logits, "token_ids": generated, "texts": texts}


def identical(left, right):
    return (torch.equal(left["logits"], right["logits"])
            and left["token_ids"] == right["token_ids"] and left["texts"] == right["texts"])


def run(config_path, intervention_dir, run_id, *, strength=1.0, max_new_tokens=8, prompts=2):
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
        raise ValueError("run_id must be a single directory name")
    if not math.isfinite(strength) or strength == 0:
        raise ValueError("the nonzero smoke-check strength must be finite")
    if type(prompts) is not int or not 1 <= prompts <= 16:
        raise ValueError("select between 1 and 16 development prompts")
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 32:
        raise ValueError("max_new_tokens must be between 1 and 32 for this smoke check")
    config_path, intervention_dir = Path(config_path), Path(intervention_dir)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    decoding = {**config["decoding"], "max_new_tokens": max_new_tokens}
    if decoding.get("do_sample") is not False or decoding.get("use_cache") is not True:
        raise ValueError("CAA checks require the shared greedy decoding and cache settings")
    if decoding.get("batch_size") != 16:
        raise ValueError("CAA checks preserve the shared batch size of 16")
    run_dir = ROOT / "results/caa_checks" / run_id
    run_dir.mkdir(parents=True)  # Never replace previous evidence, including a failed check.
    report = {
        "run_id": run_id, "purpose": "application_smoke_check_not_bias_evaluation",
        "config_path": str(config_path), "model": config["model"],
        "intervention_dir": str(intervention_dir), "strength": strength,
        "decoding": decoding, "seed": config["seed"], "checks": {}, "outputs": {},
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "torch": torch.__version__, "transformers": transformers.__version__,
                        "cuda_runtime": torch.version.cuda},
    }
    try:
        report["source"] = source_commit()
        development = ROOT / "shared_examples/development.jsonl"
        rows = validate_local(ROOT / "shared_examples/construction.jsonl", development)["development"][:prompts]
        report["data"] = {"path": "shared_examples/development.jsonl",
                          "sha256": hashlib.sha256(development.read_bytes()).hexdigest(),
                          "examples": [{"id": row["id"], "prompt": row["prompt"]} for row in rows]}
        model, tokenizer = load_model(config["model"])
        save_cluster_runtime(model, run_dir)
        intervention = load_intervention(intervention_dir, model=model)
        report["intervention"] = intervention["metadata"]
        report["checks"]["tokenizer_chat_template_matches"] = (
            hashlib.sha256(json.dumps(tokenizer.chat_template, sort_keys=True).encode("utf-8")).hexdigest()
            == intervention["metadata"]["tokenizer"]["chat_template_sha256"])
        report["environment"].update({
            "device": str(model.device), "dtype": str(model.dtype),
            "gpu": torch.cuda.get_device_name(model.device) if model.device.type == "cuda" else None,
            "generation_config": model.generation_config.to_dict(),
        })
        weights = parameter_state(model)
        hooks = {name: (tuple(module._forward_hooks), tuple(module._forward_pre_hooks))
                 for name, module in model.named_modules()}
        selected_prompts = [row["prompt"] for row in rows]

        def snapshot(label):
            result = observe(model, tokenizer, selected_prompts, decoding, config["seed"])
            logits = result["logits"]
            report["outputs"][label] = {
                "token_ids": result["token_ids"], "texts": result["texts"],
                "logits_sha256": hashlib.sha256(logits.numpy().tobytes()).hexdigest(),
                "logits_shape": list(logits.shape), "top_token_ids": logits.argmax(-1).tolist(),
            }
            return result

        baseline = snapshot("baseline")
        with apply(model, intervention, strength=0):
            zero = snapshot("zero_strength")
        report["checks"]["zero_logits_identical"] = torch.equal(baseline["logits"], zero["logits"])
        report["checks"]["zero_tokens_identical"] = baseline["token_ids"] == zero["token_ids"]
        report["checks"]["zero_texts_identical"] = baseline["texts"] == zero["texts"]
        with apply(model, intervention, strength=strength):
            nonzero = snapshot("nonzero_strength")
        report["checks"]["baseline_logits_finite"] = bool(torch.isfinite(baseline["logits"]).all())
        report["checks"]["nonzero_logits_finite"] = bool(torch.isfinite(nonzero["logits"]).all())
        report["checks"]["nonzero_logits_changed"] = not torch.equal(baseline["logits"], nonzero["logits"])
        delta = (baseline["logits"] - nonzero["logits"]).abs().max().item()
        report["nonzero_max_abs_logit_change"] = delta if math.isfinite(delta) else None
        report["checks"]["normal_exit_restores_baseline"] = identical(baseline, snapshot("after_normal_exit"))
        cleanup_probe = InterruptedError("CAA cleanup check")
        try:
            with apply(model, intervention, strength=strength):
                snapshot("before_forced_exception")
                raise cleanup_probe
        except InterruptedError as error:
            if error is not cleanup_probe:
                raise
        report["checks"]["exception_exit_restores_baseline"] = identical(baseline, snapshot("after_exception_exit"))
        report["checks"]["parameter_versions_and_storage_unchanged"] = weights == parameter_state(model)
        report["checks"]["hooks_restored"] = hooks == {
            name: (tuple(module._forward_hooks), tuple(module._forward_pre_hooks))
            for name, module in model.named_modules()}
        report["passed"] = all(report["checks"].values())
        if not report["passed"]:
            raise RuntimeError("CAA application checks failed; inspect checks.json")
    except Exception as error:
        report["passed"] = False
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        (run_dir / "checks.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/main.json")
    parser.add_argument("--intervention-dir", type=Path, default=ROOT / "steering_vectors/caa")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--prompts", type=int, default=2)
    args = parser.parse_args()
    print(run(args.config, args.intervention_dir, args.run_id, strength=args.strength,
              max_new_tokens=args.max_new_tokens, prompts=args.prompts))
