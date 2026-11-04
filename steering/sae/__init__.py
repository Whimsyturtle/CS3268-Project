"""Apply a selected pretrained SAE decoder direction. See README.md."""

import contextlib
import json
import math

import torch


def load_intervention(intervention_dir, *, model):
    """Load the saved direction and bind it to a compatible model layer."""
    metadata = json.loads((intervention_dir / "config.json").read_text(encoding="utf-8"))
    if not metadata:
        raise ValueError("Build the SAE intervention with python -m steering.sae.build first.")
    expected = metadata["model"]
    if (model.config.name_or_path != expected["name"]
            or getattr(model.config, "_commit_hash", None) != expected["revision"]):
        raise ValueError("SAE intervention model name/revision does not match the loaded model.")
    if model.dtype != getattr(torch, expected["dtype"]):
        raise ValueError("SAE intervention precision does not match the loaded model.")
    layer = metadata["layer"]
    if not isinstance(layer, int) or not 0 <= layer < len(model.model.layers):
        raise ValueError("SAE intervention layer is outside the model.")
    hook_point = f"model.layers.{layer}.output"
    checkpoint = metadata["sae"]
    if (checkpoint["model_name"] != expected["name"]
            or checkpoint["hf_hook_point_in"] != hook_point
            or checkpoint["hf_hook_point_out"] != hook_point):
        raise ValueError("SAE checkpoint does not match the model and residual-stream location.")
    direction = torch.load(intervention_dir / "direction.pt", map_location="cpu", weights_only=True)
    if direction.shape != (model.config.hidden_size,) or checkpoint["d_in"] != model.config.hidden_size:
        raise ValueError("SAE direction dimension does not match the model hidden size.")
    if not direction.is_floating_point() or not torch.isfinite(direction).all() or direction.norm() == 0:
        raise ValueError("SAE direction must be finite and nonzero.")
    scale = metadata["selection"]["activation_scale"]
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("SAE activation scale must be finite and positive.")
    return {"model": model, "layer": model.model.layers[layer],
            "direction": direction, "scale": scale, "metadata": metadata}


@contextlib.contextmanager
def apply(model, intervention, *, strength):
    """Add strength * construction activation scale * decoder direction at each token."""
    if intervention["model"] is not model:
        raise ValueError("Load the SAE intervention against the model receiving it.")
    if not math.isfinite(strength):
        raise ValueError("SAE strength must be finite.")
    if strength == 0:
        yield
        return
    delta = (intervention["direction"] * (strength * intervention["scale"])).to(
        device=model.device, dtype=model.dtype)
    if not torch.isfinite(delta).all():
        raise ValueError("SAE strength overflows the model precision.")

    def steer(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        steered = hidden + delta
        return (steered, *output[1:]) if isinstance(output, tuple) else steered

    handle = intervention["layer"].register_forward_hook(steer)
    try:
        yield
    finally:
        handle.remove()
