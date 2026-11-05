"""Offline checks for SAE feature construction and reversible steering."""

import json
from types import SimpleNamespace

import pytest
import torch
from transformers import BatchEncoding

from steering.sae import apply, load_intervention
from steering.sae import build as builder


class Layer(torch.nn.Module):
    def __init__(self, tuple_output=False):
        super().__init__()
        self.tuple_output = tuple_output

    def forward(self, hidden):
        return (hidden, "cache") if self.tuple_output else hidden


class Model(torch.nn.Module):
    def __init__(self, tuple_output=False):
        super().__init__()
        self.layers = torch.nn.ModuleList([Layer(tuple_output)])
        self.config = SimpleNamespace(name_or_path="test/gemma", revision="revision",
                                      hidden_size=2, max_position_embeddings=100)
        self.dtype = torch.float32
        self.device = torch.device("cpu")

    @property
    def model(self):
        return self

    def forward(self, input_ids, use_cache=False):
        hidden = input_ids.float().unsqueeze(-1).repeat(1, 1, 2)
        return self.layers[0](hidden)


@pytest.fixture
def saved(tmp_path):
    metadata = {"model": {"name": "test/gemma", "revision": "revision", "dtype": "float32"},
                "layer": 0, "sae": {"model_name": "test/gemma", "d_in": 2,
                    "hf_hook_point_in": "model.layers.0.output",
                    "hf_hook_point_out": "model.layers.0.output"},
                "selection": {"activation_scale": 3.0}}
    (tmp_path / "config.json").write_text(json.dumps(metadata))
    torch.save(torch.tensor([1.0, -2.0]), tmp_path / "direction.pt")
    return tmp_path, metadata


@pytest.mark.parametrize("tuple_output", [False, True])
@pytest.mark.parametrize("strength", [0, 2, -2])
def test_apply_preserves_output_and_restores_hooks(saved, tuple_output, strength):
    path, _ = saved
    model = Model(tuple_output)
    calls = []
    existing = model.layers[0].register_forward_hook(lambda *args: calls.append(1))
    intervention = load_intervention(path, model=model)
    tokens = torch.tensor([[2, 4]])
    baseline = model(tokens)
    with apply(model, intervention, strength=strength):
        assert len(model.layers[0]._forward_hooks) == (1 if strength == 0 else 2)
        result = model(tokens)
    hidden = lambda output: output[0] if tuple_output else output
    assert torch.equal(hidden(result), hidden(baseline) + strength * 3 * torch.tensor([1., -2.]))
    if tuple_output:
        assert result[1:] == baseline[1:]
    assert torch.equal(hidden(model(tokens)), hidden(baseline))
    with pytest.raises(RuntimeError, match="failure"):
        with apply(model, intervention, strength=1):
            raise RuntimeError("failure")
    assert list(model.layers[0]._forward_hooks) == [existing.id]
    assert len(calls) == 3
    existing.remove()
    with pytest.raises(ValueError, match="receiving"):
        with apply(Model(), intervention, strength=1):
            pass


@pytest.mark.parametrize("section,key,value", [
    ("model", "name", "other"), ("model", "revision", "other"),
    ("model", "dtype", "float64"), ("sae", "hf_hook_point_out", "model.layers.1.output"),
    ("sae", "d_in", 3), ("selection", "activation_scale", 0),
    ("selection", "activation_scale", float("nan")),
])
def test_load_rejects_incompatible_metadata(saved, section, key, value):
    path, metadata = saved
    metadata[section][key] = value
    (path / "config.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        load_intervention(path, model=Model())


def test_load_rejects_missing_recorded_revision(saved):
    path, _ = saved
    model = Model()
    del model.config.revision
    with pytest.raises(ValueError, match="name/revision"):
        load_intervention(path, model=model)


@pytest.mark.parametrize("direction", [torch.zeros(2), torch.ones(3), torch.tensor([float("inf"), 1.])])
def test_load_rejects_invalid_direction(saved, direction):
    path, _ = saved
    torch.save(direction, path / "direction.pt")
    with pytest.raises(ValueError):
        load_intervention(path, model=Model())


class Tokenizer:
    all_special_ids = [99]

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, return_dict=True):
        prefix = [90, 91]
        if add_generation_prompt:
            tokens = prefix
        else:
            content = messages[-1]["content"]
            tokens = prefix + ([2, 99, 5] if content else []) + [80, 81]
        return BatchEncoding({"input_ids": tokens}) if return_dict else tokens


def test_response_positions_exclude_prompt_special_and_multitoken_suffix():
    assert builder.response_tokens(Tokenizer(), "prompt", "response") == (
        [90, 91, 2, 99, 5, 80, 81], [2, 4])
    with pytest.raises(ValueError, match="no content"):
        builder.response_tokens(Tokenizer(), "prompt", "")


@pytest.mark.parametrize("tuple_output", [False, True])
def test_collect_max_pools_only_response_and_cleans_up(tuple_output):
    model = Model(tuple_output)
    rows = [{"id": "pair", "prompt": "prompt", "unbiased_response": "yes", "biased_response": "no"}]
    sae = SimpleNamespace(encode=lambda hidden: hidden)
    result = builder.collect_activations(model, Tokenizer(), sae, 0, rows)
    for values in result.values():
        assert torch.equal(values, torch.tensor([[5., 5.]]))
    assert not model.layers[0]._forward_hooks
    def fail(hidden):
        raise RuntimeError("encoding failed")
    with pytest.raises(RuntimeError, match="encoding failed"):
        builder.collect_activations(model, Tokenizer(), SimpleNamespace(encode=fail), 0, rows)
    assert not model.layers[0]._forward_hooks


def test_selection_uses_unbiased_positive_auroc_and_lowest_id_tie():
    feature, aucs, scale = builder.select_feature({
        "unbiased": torch.tensor([[2., 2., 1., 0.], [3., 3., 1., 1.]]),
        "biased": torch.tensor([[0., 0., 1., 2.], [2., 2., 1., 3.]])})
    assert feature == 0
    assert aucs == [0.875, 0.875, 0.5, 0.0]
    assert scale == 3
    with pytest.raises(ValueError, match="above chance"):
        builder.select_feature({"unbiased": torch.ones(2, 1), "biased": torch.ones(2, 1)})


def test_build_uses_only_validated_construction_rows_and_refuses_overwrite(tmp_path, monkeypatch):
    root = tmp_path / "root"
    examples = root / "shared_examples"
    examples.mkdir(parents=True)
    (examples / "construction.jsonl").write_text("construction bytes")
    config = tmp_path / "main.json"
    config.write_text(json.dumps({"seed": 0, "model": {
        "name": "test/gemma", "revision": "revision", "dtype": "float32"}}))
    settings = tmp_path / "sae.json"
    settings.write_text("{}")
    construction = [{"id": "construction-only"}]
    development = [{"id": "must-not-be-collected"}]
    monkeypatch.setattr(builder, "ROOT", root)
    monkeypatch.setattr(builder, "validate_local", lambda *args: {
        "construction": construction, "development": development})
    monkeypatch.setattr(builder, "load_model", lambda config: (Model(), Tokenizer()))
    monkeypatch.setattr(builder, "load_sae", lambda settings, model: (
        SimpleNamespace(W_dec=torch.tensor([[1., -2.]])), {"layer": 0}))
    def collect(model, tokenizer, sae, layer, rows):
        assert rows is construction
        return {"unbiased": torch.tensor([[3.]]), "biased": torch.tensor([[1.]])}
    monkeypatch.setattr(builder, "collect_activations", collect)
    monkeypatch.setattr(builder, "source_commit", lambda: "test-source")
    output = tmp_path / "output"
    assert builder.build(config, settings, output) == output
    metadata = json.loads((output / "config.json").read_text())
    assert metadata["construction"]["example_ids"] == ["construction-only"]
    assert metadata["selection"]["positive"] == "unbiased"
    assert torch.equal(torch.load(output / "direction.pt", weights_only=True), torch.tensor([1., -2.]))
    monkeypatch.setattr(builder, "load_model", lambda config: pytest.fail("overwrite loaded model"))
    with pytest.raises(ValueError, match="already contains"):
        builder.build(config, settings, output)
