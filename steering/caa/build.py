"""Construct a raw mean-difference CAA vector from shared paired responses."""

import argparse
import hashlib
import json
from pathlib import Path
import re

import torch

from evaluation.runner import ROOT, load_model, source_commit
from shared_examples.validate import validate_local
from steering.caa import METHOD, model_layer


def response_tokens(tokenizer, prompt, response):
    """Find assistant content without prompt, padding, or closing chat markers."""
    user = [{"role": "user", "content": prompt}]
    prefix = tokenizer.apply_chat_template(user, tokenize=True, return_dict=False, add_generation_prompt=True)
    empty = tokenizer.apply_chat_template(
        user + [{"role": "assistant", "content": ""}], tokenize=True, return_dict=False)
    full = tokenizer.apply_chat_template(
        user + [{"role": "assistant", "content": response}], tokenize=True, return_dict=False)
    suffix = empty[len(prefix):]
    if (not suffix or empty[:len(prefix)] != prefix or full[:len(prefix)] != prefix
            or full[-len(suffix):] != suffix):
        raise ValueError("Chat template does not expose a stable assistant response boundary")
    special = set(tokenizer.all_special_ids)
    positions = [index for index in range(len(prefix), len(full) - len(suffix)) if full[index] not in special]
    if not positions:
        raise ValueError("Construction response has no content tokens")
    return full, positions


@torch.inference_mode()
def construct(model, tokenizer, rows, *, layer):
    """Average unbiased-minus-biased block outputs at the last response content token."""
    block = model_layer(model, layer)
    if model.training:
        raise ValueError("Construct CAA with the model in evaluation mode")
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Construction requires nonempty pairs with unique IDs")
    total = torch.zeros(model.config.hidden_size, dtype=torch.float32)
    position = None
    captured = None

    def capture(_module, _inputs, output):
        nonlocal captured
        hidden = output[0] if isinstance(output, tuple) else output
        captured = hidden[0, position].detach().to(device="cpu", dtype=torch.float32, copy=True)

    handle = block.register_forward_hook(capture)
    try:
        for index, row in enumerate(rows):
            states = {}
            for label in ("unbiased", "biased"):
                tokens, positions = response_tokens(tokenizer, row["prompt"], row[f"{label}_response"])
                if len(tokens) > model.config.max_position_embeddings:
                    raise ValueError(f"Construction example {row['id']} exceeds the model context length")
                position = positions[-1]
                captured = None
                model.model(input_ids=torch.tensor([tokens], device=model.device), use_cache=False)
                if captured is None or not torch.isfinite(captured).all():
                    raise ValueError("Construction produced missing or non-finite activations")
                states[label] = captured
            total += states["unbiased"] - states["biased"]
            if (index + 1) % 20 == 0:
                print(f"Collected {index + 1}/{len(rows)} construction pairs", flush=True)
    finally:
        handle.remove()
    vector = total / len(rows)
    if not torch.isfinite(vector).all() or vector.norm() == 0:
        raise ValueError("CAA mean difference must be finite and nonzero")
    return vector


def build(config_path, caa_config_path, output_dir):
    """Build once and save the vector with its construction provenance."""
    output_dir = Path(output_dir)
    metadata_path = output_dir / "config.json"
    if ((metadata_path.exists() and json.loads(metadata_path.read_text(encoding="utf-8")))
            or (output_dir / "vector.pt").exists()):
        raise FileExistsError("CAA output already contains an intervention; choose a new --output-dir")
    construction = ROOT / "shared_examples/construction.jsonl"
    rows = validate_local(construction, ROOT / "shared_examples/development.jsonl")["construction"]
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    settings = json.loads(Path(caa_config_path).read_text(encoding="utf-8"))
    revision = config["model"]["revision"]
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("CAA construction requires a full 40-character model revision")
    torch.manual_seed(config["seed"])
    model, tokenizer = load_model(config["model"])
    if (model.config.name_or_path != config["model"]["name"]
            or getattr(model.config, "revision", None) != config["model"]["revision"]
            or model.dtype != getattr(torch, config["model"]["dtype"])):
        raise ValueError("Loaded model does not match the configured name, revision, and precision")
    layer = settings["layer"]
    vector = construct(model, tokenizer, rows, layer=layer)
    output_dir.mkdir(parents=True, exist_ok=True)
    vector_path = output_dir / "vector.pt"
    torch.save(vector, vector_path)
    metadata = {
        "method": METHOD, "model": config["model"],
        "tokenizer": {"name": config["model"]["name"], "revision": config["model"]["revision"],
                      "chat_template_sha256": hashlib.sha256(
                          json.dumps(tokenizer.chat_template, sort_keys=True).encode("utf-8")).hexdigest()},
        "layer": layer, "location": f"model.layers.{layer}.output",
        "extraction": "last_assistant_content_token", "sign": "unbiased_minus_biased",
        "normalization": "none", "generation_scope": "last_prefill_and_generated_tokens",
        "source_example_ids": [row["id"] for row in rows],
        "construction": {"path": "shared_examples/construction.jsonl",
                         "sha256": hashlib.sha256(construction.read_bytes()).hexdigest()},
        "seed": config["seed"], "source": source_commit(),
        "vector_sha256": hashlib.sha256(vector_path.read_bytes()).hexdigest(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved CAA layer {layer} vector from {len(rows)} pairs to {output_dir}")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/main.json")
    parser.add_argument("--caa-config", type=Path, default=ROOT / "configs/caa.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "steering_vectors/caa")
    args = parser.parse_args()
    build(args.config, args.caa_config, args.output_dir)
