"""Paired-demonstration ICV for the text-only Gemma 3 model."""

from contextlib import contextmanager
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F


UPSTREAM_REVISION = "b187c6387ae41097f5e38182d5a8df7d06ca9ec4"
METHOD = "upstream_joint_pca_v1"


def _layers(model):
    if model.config.model_type != "gemma3_text":
        raise ValueError("ICV requires a text-only Gemma 3 model")
    layers = model.model.layers
    if len(layers) != model.config.num_hidden_layers:
        raise ValueError("Gemma 3 layer count differs from its configuration")
    return layers


def _hidden(output):
    return output[0] if isinstance(output, tuple) else output


def _principal_direction(differences):
    """Match the authors' centered PCA, SVD sign, and mean addition."""
    if differences.ndim != 2 or differences.shape[0] < 2:
        raise ValueError("ICV needs at least two difference vectors")
    if not torch.isfinite(differences).all():
        raise ValueError("demonstrations produced non-finite activations")
    mean = differences.mean(0)
    centered = differences - mean
    u, _, vh = torch.linalg.svd(centered, full_matrices=False)
    sign = u[u[:, 0].abs().argmax(), 0].sign()
    # The authors' code centers the differences and adds their mean to PC1.
    # This differs from the uncentered objective written in paper Equation (2).
    # Source: https://github.com/shengliu66/ICV/blob/b187c6387ae41097f5e38182d5a8df7d06ca9ec4/tasks/base.py#L254-L273
    return sign * vh[0] + mean


def construct(model, tokenizer, pairs):
    """Return an (layers, hidden_size) ICV from (biased, unbiased) responses."""
    pairs = list(pairs)
    if len(pairs) < 2:
        raise ValueError("ICV needs at least two demonstration pairs")
    layers = _layers(model)
    device = next(model.parameters()).device
    modules = [model.get_input_embeddings(), *layers]
    captured = []

    def capture(_module, _inputs, output):
        captured.append(_hidden(output)[:, -1, :].detach().float().cpu())

    def states(text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("demonstration responses must be nonempty strings")
        captured.clear()
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.inference_mode():
            model(**inputs, use_cache=False)
        if len(captured) != len(modules):
            raise ValueError("model did not execute every decoder layer")
        return torch.stack([state[0] for state in captured]).flatten()

    # The authors use the final token from the embedding and every decoder layer.
    # Source: https://github.com/shengliu66/ICV/tree/b187c6387ae41097f5e38182d5a8df7d06ca9ec4
    handles = [module.register_forward_hook(capture) for module in modules]
    try:
        differences = torch.stack([
            states(target) - states(source)
            for source, target in pairs
        ])
    finally:
        for handle in handles:
            handle.remove()

    # The authors fit one PCA to the concatenated states. The embedding segment
    # is removed because steering is applied at decoder MLP outputs.
    # Source: https://github.com/shengliu66/ICV/blob/b187c6387ae41097f5e38182d5a8df7d06ca9ec4/tasks/base.py#L254-L273
    direction = _principal_direction(differences)
    return direction.reshape(len(layers) + 1, -1)[1:]


def load_intervention(intervention_dir: Path, *, model):
    """Load a saved ICV and check model revision and layer dimensions."""
    metadata = json.loads((intervention_dir / "config.json").read_text(encoding="utf-8"))
    if metadata.get("method") != METHOD:
        raise ValueError("ICV construction method changed; rebuild the vector")
    layers = _layers(model)
    revision = getattr(model.config, "revision", None)
    legacy_revision = getattr(model, "_cs3268_source_revision", None)
    if revision is not None and legacy_revision is not None and revision != legacy_revision:
        raise ValueError("ICV model revision conflicts with the legacy runtime revision")
    if revision is None:
        revision = legacy_revision
    if (metadata["model"] != model.config._name_or_path
            or metadata["revision"] != revision):
        raise ValueError("ICV model or revision does not match the loaded model")
    if metadata["layers"] != list(range(len(layers))):
        raise ValueError("ICV layer locations do not match the loaded model")
    vector = torch.load(intervention_dir / "vector.pt", map_location="cpu", weights_only=True)
    if not isinstance(vector, torch.Tensor) or vector.shape != (len(layers), model.config.hidden_size):
        raise ValueError("ICV shape does not match the loaded model")
    if not torch.isfinite(vector).all():
        raise ValueError("ICV contains non-finite values")
    return vector


def _shift(hidden, vector, strength):
    if strength == 0:
        return hidden
    original = hidden.float()
    vector = vector.to(device=hidden.device, dtype=torch.float32)
    # The authors normalize both terms and increase strength when the activation
    # points against the ICV.
    # Source: https://github.com/shengliu66/ICV/blob/b187c6387ae41097f5e38182d5a8df7d06ca9ec4/utils/llm_layers.py#L8-L27
    weight = 1 + F.cosine_similarity(original, -vector, dim=-1).clamp_min(0).unsqueeze(-1)
    changed = F.normalize(original, dim=-1) + strength * weight * F.normalize(vector, dim=-1)
    return (F.normalize(changed, dim=-1) * original.norm(dim=-1, keepdim=True)).to(hidden.dtype)


@contextmanager
def apply(model, intervention, *, strength: float):
    """Steer every token at each MLP output and remove hooks on exit."""
    layers = _layers(model)
    if not isinstance(strength, (int, float)) or not math.isfinite(strength):
        raise ValueError("ICV strength must be finite")
    if intervention.shape != (len(layers), model.config.hidden_size):
        raise ValueError("ICV shape does not match the loaded model")
    if not torch.isfinite(intervention).all():
        raise ValueError("ICV contains non-finite values")

    def hook(vector):
        def shift(_module, _inputs, output):
            hidden = _hidden(output)
            restored = _shift(hidden, vector, strength)
            return (restored, *output[1:]) if isinstance(output, tuple) else restored
        return shift

    handles = []
    try:
        for layer, vector in zip(layers, intervention, strict=True):
            handles.append(layer.mlp.register_forward_hook(hook(vector)))
        yield
    finally:
        for handle in handles:
            handle.remove()
