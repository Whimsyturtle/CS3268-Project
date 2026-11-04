"""Select one pretrained SAE feature using shared construction examples."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from evaluation.runner import ROOT, load_model, source_commit
from shared_examples.validate import validate_local


def load_sae(settings, *, model):
    """Load pinned Google weights through SAELens, preserving their native scale."""
    from sae_lens import JumpReLUSAE, JumpReLUSAEConfig

    def download(filename):
        return hf_hub_download(repo_id=settings["repo_id"], revision=settings["revision"],
                               filename=f"{settings['path']}/{filename}")

    config = json.loads(Path(download("config.json")).read_text(encoding="utf-8"))
    hook = config["hf_hook_point_in"]
    parts = hook.split(".")
    if (len(parts) != 4 or parts[:2] != ["model", "layers"] or parts[3] != "output"
            or not parts[2].isdigit() or config["hf_hook_point_out"] != hook
            or config["architecture"] != "jump_relu" or config["type"] != "sae"
            or config["affine_connection"] or config["model_name"] != model.config.name_or_path):
        raise ValueError("Expected a single-layer Gemma residual-stream JumpReLU SAE for this model.")
    layer = int(parts[2])
    if not 0 <= layer < len(model.model.layers):
        raise ValueError("SAE checkpoint layer is outside the model.")
    weights = load_file(download("params.safetensors"), device=str(model.device))
    d_in, width = weights["w_enc"].shape
    if d_in != model.config.hidden_size or width != config["width"]:
        raise ValueError("SAE checkpoint dimensions do not match the model/configuration.")
    sae = JumpReLUSAE(JumpReLUSAEConfig(
        d_in=d_in, d_sae=width, dtype="float32", device=str(model.device),
        apply_b_dec_to_input=False, normalize_activations="none", reshape_activations="none",
    ))
    sae.load_state_dict({"W_enc": weights["w_enc"], "W_dec": weights["w_dec"],
                         "b_enc": weights["b_enc"], "b_dec": weights["b_dec"],
                         "threshold": weights["threshold"]}, strict=True)
    sae.eval().requires_grad_(False)
    return sae, {**settings, **config, "layer": layer, "d_in": d_in}


def response_tokens(tokenizer, prompt, response):
    """Identify response content without prompt, padding, or chat boundary tokens."""
    user = [{"role": "user", "content": prompt}]
    prefix = tokenizer.apply_chat_template(user, tokenize=True, add_generation_prompt=True)
    empty = tokenizer.apply_chat_template(user + [{"role": "assistant", "content": ""}], tokenize=True)
    full = tokenizer.apply_chat_template(user + [{"role": "assistant", "content": response}], tokenize=True)
    suffix = empty[len(prefix):]
    if (not suffix or empty[:len(prefix)] != prefix or full[:len(prefix)] != prefix
            or full[-len(suffix):] != suffix):
        raise ValueError("Chat template does not expose a stable assistant response boundary.")
    special = set(tokenizer.all_special_ids)
    positions = [index for index in range(len(prefix), len(full) - len(suffix))
                 if full[index] not in special]
    if not positions:
        raise ValueError("Construction response has no content tokens.")
    return full, positions


@torch.inference_mode()
def collect_activations(model, tokenizer, sae, layer, rows):
    """Max-pool each feature over response tokens, one unpadded response at a time."""
    collected = {"unbiased": [], "biased": []}
    for index, row in enumerate(rows):
        for label in collected:
            tokens, positions = response_tokens(tokenizer, row["prompt"], row[f"{label}_response"])
            if len(tokens) > model.config.max_position_embeddings:
                raise ValueError(f"Construction example {row['id']} exceeds the model context length.")
            pooled = []

            def capture(module, inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                pooled.append(sae.encode(hidden[0, positions]).amax(dim=0).cpu())

            handle = model.model.layers[layer].register_forward_hook(capture)
            try:
                model.model(input_ids=torch.tensor([tokens], device=model.device), use_cache=False)
            finally:
                handle.remove()
            collected[label].append(pooled[0])
        if (index + 1) % 20 == 0:
            print(f"Collected {index + 1}/{len(rows)} construction pairs", flush=True)
    return {label: torch.stack(scores) for label, scores in collected.items()}


def select_feature(activations):
    """Rank features by unbiased-positive AUROC; resolve ties by lowest feature ID."""
    from sklearn.metrics import roc_auc_score

    positive, negative = activations["unbiased"], activations["biased"]
    scores = torch.cat([positive, negative]).float()
    if not torch.isfinite(scores).all():
        raise ValueError("SAE activations contain non-finite values.")
    labels = [1] * len(positive) + [0] * len(negative)
    aucs = [float(roc_auc_score(labels, scores[:, feature].numpy()))
            for feature in range(scores.shape[1])]
    feature = max(range(len(aucs)), key=aucs.__getitem__)
    if aucs[feature] <= 0.5:
        raise ValueError("No SAE feature distinguishes unbiased construction responses above chance.")
    scale = float(scores[:, feature].max())
    return feature, aucs, scale


def build(config_path, sae_config_path, output_dir):
    metadata_path = output_dir / "config.json"
    if ((metadata_path.exists() and json.loads(metadata_path.read_text(encoding="utf-8")))
            or any((output_dir / name).exists() for name in ["direction.pt", "activations.pt", "selection.json"])):
        raise ValueError("Output already contains an intervention; choose a new --output-dir.")
    construction = ROOT / "shared_examples" / "construction.jsonl"
    rows = validate_local(construction, ROOT / "shared_examples" / "development.jsonl")["construction"]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    settings = json.loads(sae_config_path.read_text(encoding="utf-8"))
    torch.manual_seed(config["seed"])
    model, tokenizer = load_model(config["model"])
    if (model.config.name_or_path != config["model"]["name"]
            or getattr(model.config, "_commit_hash", None) != config["model"]["revision"]):
        raise ValueError("Loaded model does not report the configured name and revision.")
    sae, checkpoint = load_sae(settings, model=model)
    activations = collect_activations(model, tokenizer, sae, checkpoint["layer"], rows)
    feature, aucs, scale = select_feature(activations)
    direction = sae.W_dec[feature].detach().float().cpu()
    if not torch.isfinite(direction).all() or direction.norm() == 0:
        raise ValueError("Selected SAE decoder direction must be finite and nonzero.")
    metadata = {
        "model": config["model"], "layer": checkpoint["layer"], "sae": checkpoint,
        "seed": config["seed"], "source": source_commit(),
        "construction": {"path": "shared_examples/construction.jsonl",
                         "sha256": hashlib.sha256(construction.read_bytes()).hexdigest(),
                         "example_ids": [row["id"] for row in rows]},
        "selection": {"method": "max_response_activation_auroc", "positive": "unbiased",
                      "feature_id": feature, "auroc": aucs[feature], "activation_scale": scale,
                      "tie_break": "lowest_feature_id"},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(direction, output_dir / "direction.pt")
    torch.save(activations, output_dir / "activations.pt")
    (output_dir / "selection.json").write_text(json.dumps({"auroc": aucs}, indent=2) + "\n", encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Selected feature {feature}, construction AUROC {aucs[feature]:.4f}; saved to {output_dir}")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "main.json")
    parser.add_argument("--sae-config", type=Path, default=ROOT / "configs" / "sae.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "steering_vectors" / "sae")
    args = parser.parse_args()
    build(args.config, args.sae_config, args.output_dir)
