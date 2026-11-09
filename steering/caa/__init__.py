"""Contrastive Activation Addition for text-only Gemma 3. See README.md."""

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path

import torch

METHOD = "caa_last_response_token_v1"


def model_layer(model, layer):
    """Resolve the configured decoder-block output location."""
    if model.config.model_type != "gemma3_text":
        raise ValueError("CAA requires a text-only Gemma 3 model")
    layers = model.model.layers
    if len(layers) != model.config.num_hidden_layers:
        raise ValueError("Gemma layer count differs from its configuration")
    if type(layer) is not int or not 0 <= layer < len(layers):
        raise ValueError("CAA layer is outside the model")
    return layers[layer]


def load_intervention(intervention_dir, *, model):
    """Load a CAA vector and bind it to a compatible pinned model."""
    directory = Path(intervention_dir)
    metadata = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if not metadata:
        raise ValueError("Build the CAA intervention with python -m steering.caa.build first")
    conventions = {
        "method": METHOD, "extraction": "last_assistant_content_token",
        "sign": "unbiased_minus_biased", "normalization": "none",
        "generation_scope": "last_prefill_and_generated_tokens",
    }
    if any(metadata.get(key) != value for key, value in conventions.items()):
        raise ValueError("CAA construction or application convention does not match this implementation")
    if (model.config.name_or_path != metadata.get("model")
            or getattr(model.config, "revision", None) != metadata.get("revision")
            or str(model.dtype).removeprefix("torch.") != metadata.get("dtype")):
        raise ValueError("CAA model name, revision, or precision does not match the loaded model")
    tokenizer = metadata["tokenizer"]
    if (tokenizer["name"] != metadata["model"] or tokenizer["revision"] != metadata["revision"]):
        raise ValueError("CAA tokenizer must use the same pinned model revision")
    layer = metadata["layer"]
    block = model_layer(model, layer)
    if metadata["location"] != f"model.layers.{layer}.output":
        raise ValueError("CAA activation location does not match its layer")
    vector_path = directory / "vector.pt"
    if hashlib.sha256(vector_path.read_bytes()).hexdigest() != metadata["vector_sha256"]:
        raise ValueError("CAA vector hash does not match the saved metadata")
    vector = torch.load(vector_path, map_location="cpu", weights_only=True)
    if (not isinstance(vector, torch.Tensor) or vector.dtype != torch.float32
            or vector.shape != (model.config.hidden_size,)
            or not torch.isfinite(vector).all() or vector.norm() == 0):
        raise ValueError("CAA vector must be a finite nonzero float32 hidden-size vector")
    return {"model": model, "layer": block, "vector": vector, "metadata": metadata}


@contextmanager
def apply(model, intervention, *, strength):
    """Steer the last prefill token and cached continuation tokens, then remove hooks."""
    if intervention["model"] is not model:
        raise ValueError("Load the CAA intervention against the model receiving it")
    if not math.isfinite(strength):
        raise ValueError("CAA strength must be finite")
    if strength == 0:
        yield
        return
    if model.training:
        raise ValueError("Apply CAA with the model in evaluation mode")
    delta = (intervention["vector"] * strength).to(device=model.device, dtype=model.dtype)
    if not torch.isfinite(delta).all():
        raise ValueError("CAA strength overflows the model precision")
    mask = None

    def prepare(_module, args, kwargs):
        nonlocal mask
        if not kwargs.get("use_cache", model.config.use_cache):
            raise ValueError("CAA application requires use_cache=True")
        tokens = kwargs.get("input_ids", args[0] if args else None)
        if tokens is None or tokens.ndim != 2 or tokens.shape[1] == 0:
            raise ValueError("CAA application requires a nonempty batch of input token IDs")
        cache = kwargs.get("past_key_values")
        past_length = 0 if cache is None else cache.get_seq_length()
        if past_length:
            mask = torch.ones_like(tokens, dtype=torch.bool)
            return
        attention = kwargs.get("attention_mask")
        if attention is None:
            attention = tokens.ne(model.config.pad_token_id)
        if attention.ndim != 2 or attention.shape != tokens.shape or not attention.bool().any(dim=1).all():
            raise ValueError("CAA prefill requires a nonempty 2D attention mask matching the input IDs")
        positions = torch.arange(tokens.shape[1], device=tokens.device).expand_as(tokens)
        last = positions.masked_fill(~attention.bool(), -1).amax(dim=1)
        mask = positions.eq(last.unsqueeze(1))

    def steer(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if mask is None or mask.shape != hidden.shape[:2]:
            raise ValueError("CAA block output does not match the current model input")
        changed = hidden + mask.unsqueeze(-1) * delta
        return (changed, *output[1:]) if isinstance(output, tuple) else changed

    handles = []
    try:
        handles.append(model.register_forward_pre_hook(prepare, with_kwargs=True))
        handles.append(intervention["layer"].register_forward_hook(steer))
        yield
    finally:
        for handle in handles:
            handle.remove()
