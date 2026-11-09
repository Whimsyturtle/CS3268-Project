"""Focused ICV math and hook checks using a small real Gemma 3 model."""

import json

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers

from steering.icv import (
    METHOD,
    _principal_direction,
    _shift,
    apply,
    construct,
    load_intervention,
)


def test_principal_direction_adds_mean_using_upstream_svd_sign():
    differences = torch.tensor([[1.0, 0.0], [2.0, 0.0], [4.0, 0.0]])
    torch.testing.assert_close(_principal_direction(differences), torch.tensor([10 / 3, 0.0]))
    torch.testing.assert_close(_principal_direction(-differences), torch.tensor([-10 / 3, 0.0]))


def test_pca_is_joint_across_segments():
    differences = torch.tensor([[1., 0., 2., 0.], [2., 0., 4., 0.], [4., 0., 8., 0.]])
    expected = differences.mean(0) + torch.tensor([1., 0., 2., 0.]) / 5**0.5
    torch.testing.assert_close(_principal_direction(differences), expected)


def test_shift_uses_cosine_weight_and_normalized_addition():
    hidden = torch.tensor([[[3., 4.], [-3., 4.]]], dtype=torch.bfloat16)
    # Opposing token gets multiplier 1.6; aligned token gets multiplier 1.
    expected = torch.tensor([[[1.1, 0.8], [0.2, 0.8]]])
    expected = (expected / expected.norm(dim=-1, keepdim=True) * 5).to(hidden.dtype)
    torch.testing.assert_close(_shift(hidden, torch.tensor([2., 0.]), 0.5), expected)
    assert _shift(hidden, torch.tensor([2., 0.]), 0) is hidden


def test_shift_preserves_each_token_norm():
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    shifted = _shift(hidden, torch.tensor([0.5, -0.5]), 0.5)
    assert not torch.equal(hidden, shifted)
    torch.testing.assert_close(hidden.norm(dim=-1), shifted.norm(dim=-1))


def test_load_checks_layer_locations(tmp_path):
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig

    config = Gemma3TextConfig(
        vocab_size=32, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8,
    )
    config._name_or_path = "test/model"
    model = Gemma3ForCausalLM(config)
    model._cs3268_source_revision = "test-revision"
    torch.save(torch.zeros(2, 32), tmp_path / "vector.pt")
    metadata = {"model": "test/model", "revision": "test-revision", "layers": [1, 0], "method": METHOD}
    (tmp_path / "config.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="layer locations"):
        load_intervention(tmp_path, model=model)
    metadata["layers"] = [0, 1]
    (tmp_path / "config.json").write_text(json.dumps(metadata), encoding="utf-8")
    torch.testing.assert_close(load_intervention(tmp_path, model=model), torch.zeros(2, 32))
    del model._cs3268_source_revision
    with pytest.raises(ValueError, match="model or revision"):
        load_intervention(tmp_path, model=model)
    model._cs3268_source_revision = "test-revision"
    del metadata["method"]
    (tmp_path / "config.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="rebuild"):
        load_intervention(tmp_path, model=model)


def test_construct_extracts_each_layer_and_removes_hooks():
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig, PreTrainedTokenizerFast

    config = Gemma3TextConfig(
        vocab_size=32, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8,
    )
    model = Gemma3ForCausalLM(config).eval()
    raw = Tokenizer(models.WordLevel({
        "[UNK]": 0, "[PAD]": 1, "bad": 2, "good": 3,
        "sad": 4, "kind": 5, "mean": 6, "fair": 7,
    }, unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw, unk_token="[UNK]", pad_token="[PAD]"
    )
    pairs = [
        ("bad", "good"), ("sad", "kind"), ("mean", "fair")
    ]
    # Independent extraction includes the embedding segment in the PCA input.
    def states(text):
        with torch.inference_mode():
            output = model(**tokenizer(text, return_tensors="pt"), output_hidden_states=True)
        # The final hidden_states entry is post-final-norm; capture the raw layer instead.
        return torch.cat([state[0, -1].float() for state in output.hidden_states[:-1]])

    final_states = []
    handle = model.model.layers[-1].register_forward_hook(
        lambda _m, _a, out: final_states.append((out[0] if isinstance(out, tuple) else out)[0, -1].detach().float())
    )
    try:
        differences = []
        for source, target in pairs:
            source_state = states(source)
            source_state = torch.cat([source_state, final_states.pop()])
            target_state = states(target)
            target_state = torch.cat([target_state, final_states.pop()])
            differences.append(target_state - source_state)
    finally:
        handle.remove()
    expected = _principal_direction(torch.stack(differences)).reshape(3, 32)[1:]

    # Transformers may install its own hooks; only require construct to restore
    # the hook registries it receives.
    modules = [model.get_input_embeddings(), *model.model.layers]
    hooks_before = [dict(module._forward_hooks) for module in modules]
    vector = construct(model, tokenizer, pairs)
    assert vector.shape == (2, 32)
    torch.testing.assert_close(vector, expected)
    assert [dict(module._forward_hooks) for module in modules] == hooks_before


def test_apply_changes_model_and_removes_hooks():
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig

    config = Gemma3TextConfig(
        vocab_size=32, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8,
    )
    model = Gemma3ForCausalLM(config).eval()
    input_ids = torch.tensor([[1, 5, 6]])
    vector = torch.zeros(2, 32)
    vector[:, 0] = 1
    captured = []
    hooks_before = [dict(layer.mlp._forward_hooks) for layer in model.model.layers]
    handles = [layer.mlp.register_forward_hook(
        lambda _layer, _args, output: captured.append(output.detach().clone())
    ) for layer in model.model.layers]
    observer_hooks = [dict(layer.mlp._forward_hooks) for layer in model.model.layers]
    try:
        with torch.inference_mode():
            model(input_ids, use_cache=False)
            original = [state.clone() for state in captured]
            captured.clear()
            with apply(model, vector, strength=0.5):
                model(input_ids, use_cache=False)
            changed = [state.clone() for state in captured]
        # Observer hooks run before steering; layer 2 still sees the changed layer 1 input.
        assert any(not torch.equal(a, b) for a, b in zip(original, changed))
        assert [dict(layer.mlp._forward_hooks) for layer in model.model.layers] == observer_hooks
    finally:
        for handle in handles:
            handle.remove()

    with pytest.raises(RuntimeError):
        with apply(model, vector, strength=0.5):
            raise RuntimeError("check cleanup")
    assert [dict(layer.mlp._forward_hooks) for layer in model.model.layers] == hooks_before
