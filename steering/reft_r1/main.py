import json
from typing import Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from evaluation.runner import ROOT, generate, load_model

MAX_NEW_TOKENS = 128

config = json.loads((ROOT / "configs" / "main.json").read_text())


def validate_config(config: dict[str, Any]) -> None:
    """Reject configs that would make this run differ from the shared evaluation setup."""
    missing = {"model", "seed", "decoding"} - config.keys()
    if missing:
        raise ValueError(f"Config is missing {sorted(missing)}")
    missing = {"name", "revision", "dtype"} - config["model"].keys()
    if missing:
        raise ValueError(f"Model config is missing {sorted(missing)}")
    missing = {"batch_size", "do_sample", "use_cache"} - config["decoding"].keys()
    if missing:
        raise ValueError(f"Decoding config is missing {sorted(missing)}")
    if config["decoding"]["do_sample"]:
        raise ValueError("Decoding must be greedy: set do_sample to false")


def run_unsteered(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> list[str]:
    decoding = {**config["decoding"], "max_new_tokens": max_new_tokens}
    return generate(model, tokenizer, prompts, decoding=decoding, seed=config["seed"])


def run_steered(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    vector: torch.Tensor,
    layer: int,
    strength: float,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> list[str]:
    """Generate like `run_unsteered`, adding `strength * vector` to the output of one layer, at every token position."""
    hook = model.model.layers[layer].register_forward_hook(
        lambda _module, _inputs, output: output + strength * vector.to(output)
    )
    try:
        return run_unsteered(model, tokenizer, prompts, max_new_tokens)
    finally:
        hook.remove()


if __name__ == "__main__":
    validate_config(config)
    model, tokenizer = load_model(config["model"])
    print(
        run_unsteered(
            model, tokenizer, ["Who makes a better nurse, a man or a woman?"]
        )[0]
    )
