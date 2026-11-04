"""Small real-library checks; these do not load pretrained Gemma weights."""

import json

import pytest
import torch
from safetensors.torch import save_file
from transformers import Gemma3ForCausalLM, Gemma3TextConfig

from steering.sae import apply, load_intervention
from steering.sae import build as builder


def tiny_gemma():
    torch.manual_seed(0)
    config = Gemma3TextConfig(
        vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=64, sliding_window=16, pad_token_id=0,
        bos_token_id=2, eos_token_id=1,
    )
    config.name_or_path = "test/tiny-gemma"
    config._commit_hash = "synthetic-revision"
    return Gemma3ForCausalLM(config).eval()


def test_pinned_weight_loader_uses_library_jumprelu_without_bias_subtraction(tmp_path, monkeypatch):
    model = tiny_gemma()
    config = {"hf_hook_point_in": "model.layers.0.output",
              "hf_hook_point_out": "model.layers.0.output", "width": 3,
              "model_name": model.config.name_or_path, "architecture": "jump_relu",
              "l0": 2, "affine_connection": False, "type": "sae"}
    (tmp_path / "config.json").write_text(json.dumps(config))
    weights = {"w_enc": torch.ones(32, 3), "w_dec": torch.arange(96).reshape(3, 32).float(),
               "b_enc": torch.tensor([0., -1., 0.]), "b_dec": torch.full((32,), 7.),
               "threshold": torch.tensor([1., 1., 40.])}
    save_file(weights, tmp_path / "params.safetensors")
    settings = {"repo_id": "test/sae", "revision": "pinned", "path": "resid_post/test"}
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path / kwargs["filename"].split("/")[-1])

    monkeypatch.setattr(builder, "hf_hub_download", download)
    sae, metadata = builder.load_sae(settings, model=model)
    assert [call["revision"] for call in calls] == ["pinned", "pinned"]
    assert metadata["d_in"] == 32
    assert torch.equal(sae.encode(torch.ones(1, 32)), torch.tensor([[32., 31., 0.]]))
    assert torch.equal(sae.W_dec, weights["w_dec"])
    assert not sae.training and not any(param.requires_grad for param in sae.parameters())
    config["hf_hook_point_out"] = "model.layers.1.output"
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="single-layer"):
        builder.load_sae(settings, model=model)


@torch.inference_mode()
def test_native_gemma_logits_cached_generation_and_state_restoration(tmp_path):
    model = tiny_gemma()
    metadata = {
        "model": {"name": model.config.name_or_path, "revision": model.config._commit_hash,
                  "dtype": "float32"}, "layer": 0,
        "sae": {"model_name": model.config.name_or_path, "d_in": 32,
                "hf_hook_point_in": "model.layers.0.output",
                "hf_hook_point_out": "model.layers.0.output"},
        "selection": {"activation_scale": 2.0},
    }
    (tmp_path / "config.json").write_text(json.dumps(metadata))
    torch.save(torch.linspace(-1, 1, 32), tmp_path / "direction.pt")
    intervention = load_intervention(tmp_path, model=model)
    tokens = torch.tensor([[2, 4, 5]])
    baseline_logits = model(tokens).logits.clone()
    generation = dict(max_new_tokens=3, do_sample=False, use_cache=True)
    baseline_tokens = model.generate(tokens, **generation)
    original_parameters = {name: value.clone() for name, value in model.named_parameters()}
    with apply(model, intervention, strength=0):
        assert torch.equal(model(tokens).logits, baseline_logits)
        assert torch.equal(model.generate(tokens, **generation), baseline_tokens)
    with apply(model, intervention, strength=0.5):
        steered = model(tokens).logits
        assert torch.isfinite(steered).all()
        assert not torch.equal(steered, baseline_logits)
        assert model.generate(tokens, **generation).shape[0] == 1
    assert torch.equal(model(tokens).logits, baseline_logits)
    with pytest.raises(RuntimeError, match="exit check"):
        with apply(model, intervention, strength=0.5):
            model(tokens)
            raise RuntimeError("exit check")
    assert torch.equal(model.generate(tokens, **generation), baseline_tokens)
    assert all(torch.equal(value, original_parameters[name]) for name, value in model.named_parameters())
