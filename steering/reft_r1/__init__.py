"""ReFT-r1 inference: load a trained vector and add it to one layer while generating."""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TypedDict

import torch
from transformers import PreTrainedModel


class SavedVector(TypedDict):
    """Contents of vector.pt, and the intervention that `apply` uses."""

    vector: torch.Tensor  # (hidden_size,) with length 1.
    max_act: float  # Largest token score on the construction examples.
    layer: int  # Index of the layer whose output is steered.


def load_intervention(intervention_dir: Path, *, model: PreTrainedModel) -> SavedVector:
    """Load a trained vector and check that it fits the model.

    Args:
        intervention_dir: Folder holding vector.pt and config.json.
        model: The model the vector will steer.

    Returns:
        The saved vector, its scale, and its layer.

    Raises:
        ValueError: If the vector was trained for a different model, does not fit, or
            disagrees with its config.json.
    """
    settings = json.loads((intervention_dir / "config.json").read_text())
    saved = torch.load(intervention_dir / "vector.pt", weights_only=True)
    loaded_model = {
        "name": model.config.name_or_path,
        "revision": model.config.revision,  # Set by evaluation.runner.load_model.
        "dtype": str(model.dtype).removeprefix("torch."),
    }

    if settings["model"] != loaded_model:
        raise ValueError(
            f"Vector was trained on {settings['model']}, not {loaded_model}"
        )
    if (settings["layer"], settings["max_act"]) != (saved["layer"], saved["max_act"]):
        raise ValueError("vector.pt and config.json are from different training runs")
    if saved["vector"].shape != (model.config.hidden_size,):
        raise ValueError(
            f"Vector shape {tuple(saved['vector'].shape)} does not fit the model"
        )
    if not 0 <= saved["layer"] < len(model.model.layers):
        raise ValueError(f"Model has no layer {saved['layer']}")
    return saved


@contextmanager
def apply(
    model: PreTrainedModel, intervention: SavedVector, *, strength: float
) -> Iterator[None]:
    """Add `strength * max_act * vector` to one layer's output, at every token position.

    Args:
        model: The model to steer.
        intervention: The vector from `load_intervention`.
        strength: Multiple of `max_act` to add. 0 leaves outputs unchanged.

    Yields:
        Nothing. The model is steered inside the `with` block and restored on exit.
    """
    steering = strength * intervention["max_act"] * intervention["vector"]
    hook = model.model.layers[intervention["layer"]].register_forward_hook(
        lambda _module, _inputs, output: output + steering.to(output)
    )
    try:
        yield
    finally:
        hook.remove()
