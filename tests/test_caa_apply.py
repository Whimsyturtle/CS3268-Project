import hashlib
import json

import pytest
import torch
from transformers import DynamicCache

from steering.caa import apply, load_intervention


def test_zero_strength_and_exit_restore_native_gemma(caa_artifact, caa_model, caa_tokenizer):
    intervention = load_intervention(caa_artifact, model=caa_model)
    inputs = caa_tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hi?"}], tokenize=True,
        add_generation_prompt=True, return_tensors="pt", return_dict=True)
    decoding = {"do_sample": False, "use_cache": True, "max_new_tokens": 3}
    with torch.inference_mode():
        baseline_logits = caa_model(**inputs).logits.clone()
        baseline_tokens = caa_model.generate(**inputs, **decoding)
        parameters = {name: value.clone() for name, value in caa_model.named_parameters()}
        with apply(caa_model, intervention, strength=0):
            assert not caa_model._forward_pre_hooks
            assert torch.equal(caa_model(**inputs).logits, baseline_logits)
            assert torch.equal(caa_model.generate(**inputs, **decoding), baseline_tokens)
        for _ in range(2):
            with apply(caa_model, intervention, strength=1):
                changed = caa_model(**inputs).logits
                assert torch.isfinite(changed).all() and not torch.equal(changed, baseline_logits)
                generated = caa_model.generate(**inputs, **decoding)
                assert generated.shape[0] == 1
                assert torch.equal(generated[:, :inputs["input_ids"].shape[1]], inputs["input_ids"])
            assert torch.equal(caa_model(**inputs).logits, baseline_logits)
        with pytest.raises(RuntimeError, match="exit test"):
            with apply(caa_model, intervention, strength=1):
                caa_model(**inputs)
                raise RuntimeError("exit test")
        assert torch.equal(caa_model.generate(**inputs, **decoding), baseline_tokens)
        assert all(torch.equal(value, parameters[name]) for name, value in caa_model.named_parameters())
    assert not caa_model._forward_pre_hooks and not caa_model.model.layers[0]._forward_hooks


def test_exact_prefill_and_cached_deltas_with_left_padding_and_batch_reset(caa_artifact, caa_model):
    intervention = load_intervention(caa_artifact, model=caa_model)
    block = caa_model.model.layers[0]
    captured_before, captured_after = [], []
    before = block.register_forward_hook(lambda module, inputs, output: captured_before.append(output.clone()))
    try:
        with apply(caa_model, intervention, strength=0.5):
            after = block.register_forward_hook(lambda module, inputs, output: captured_after.append(output.clone()))
            try:
                with torch.inference_mode():
                    inputs = torch.tensor([[0, 2, 10, 11], [2, 12, 13, 14]])
                    cache = DynamicCache(config=caa_model.config)
                    output = caa_model(input_ids=inputs, attention_mask=inputs.ne(0).long(), past_key_values=cache, use_cache=True)
                    caa_model(input_ids=torch.tensor([[15], [16]]), attention_mask=torch.ones(2, 5),
                              past_key_values=output.past_key_values, use_cache=True)
                    # A one-token fresh prefill after a different-size batch must reset the mask.
                    caa_model(input_ids=torch.tensor([[2]]), attention_mask=torch.ones(1, 1), use_cache=True)
            finally:
                after.remove()
        expected = torch.zeros_like(captured_before[0])
        expected[:, -1] = intervention["vector"] * 0.5
        torch.testing.assert_close(captured_after[0] - captured_before[0], expected, atol=1e-7, rtol=1e-5)
        for index in (1, 2):
            torch.testing.assert_close(captured_after[index] - captured_before[index],
                                       (intervention["vector"] * 0.5).expand_as(captured_before[index]), atol=1e-7, rtol=1e-5)
        assert list(block._forward_hooks.values())  # The unrelated observer survives CAA cleanup.
    finally:
        before.remove()


@pytest.mark.parametrize("field,value", [
    ("revision", "b" * 40), ("model", "other/model"), ("dtype", "bfloat16"), ("dtype", None),
])
def test_loader_rejects_model_mismatch(caa_artifact, caa_model, field, value):
    path = caa_artifact / "config.json"
    metadata = json.loads(path.read_text())
    if value is None:
        del metadata[field]
    else:
        metadata[field] = value
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="model name, revision, or precision"):
        load_intervention(caa_artifact, model=caa_model)


@pytest.mark.parametrize("problem", ["hash", "shape", "nonfinite", "zero", "layer", "location", "convention"])
def test_loader_rejects_incompatible_artifacts(caa_artifact, caa_model, problem):
    path = caa_artifact / "config.json"
    metadata = json.loads(path.read_text())
    vector_path = caa_artifact / "vector.pt"
    if problem in {"hash", "shape", "nonfinite", "zero"}:
        vector = {"shape": torch.ones(17), "nonfinite": torch.full((16,), float("nan")),
                  "zero": torch.zeros(16), "hash": torch.ones(16)}[problem]
        torch.save(vector, vector_path)
        if problem != "hash":
            metadata["vector_sha256"] = hashlib.sha256(vector_path.read_bytes()).hexdigest()
    elif problem == "layer":
        metadata["layer"] = 2
    elif problem == "location":
        metadata["location"] = "model.norm"
    else:
        metadata["normalization"] = "unit"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        load_intervention(caa_artifact, model=caa_model)


def test_partial_setup_failure_removes_pre_hook(caa_artifact, caa_model, monkeypatch):
    intervention = load_intervention(caa_artifact, model=caa_model)

    def fail(*args, **kwargs):
        raise RuntimeError("registration failed")

    monkeypatch.setattr(intervention["layer"], "register_forward_hook", fail)
    with pytest.raises(RuntimeError, match="registration failed"):
        with apply(caa_model, intervention, strength=1):
            pass
    assert not caa_model._forward_pre_hooks


@pytest.mark.parametrize("strength", [float("nan"), float("inf")])
def test_application_rejects_nonfinite_strength(caa_artifact, caa_model, strength):
    intervention = load_intervention(caa_artifact, model=caa_model)
    with pytest.raises(ValueError, match="finite"):
        with apply(caa_model, intervention, strength=strength):
            pass


def test_application_rejects_uncached_mode_and_cleans_up(caa_artifact, caa_model):
    intervention = load_intervention(caa_artifact, model=caa_model)
    with pytest.raises(ValueError, match="use_cache=True"):
        with apply(caa_model, intervention, strength=1):
            caa_model(input_ids=torch.tensor([[2, 5]]), use_cache=False)
    assert not caa_model._forward_pre_hooks and not caa_model.model.layers[0]._forward_hooks


def test_native_bfloat16_application(caa_artifact, caa_model, caa_tokenizer, caa_pairs):
    from steering.caa.build import construct

    caa_model.to(torch.bfloat16)
    vector = construct(caa_model, caa_tokenizer, caa_pairs, layer=0)
    vector_path = caa_artifact / "vector.pt"
    torch.save(vector, vector_path)
    path = caa_artifact / "config.json"
    metadata = json.loads(path.read_text())
    metadata["dtype"] = "bfloat16"
    metadata["vector_sha256"] = hashlib.sha256(vector_path.read_bytes()).hexdigest()
    path.write_text(json.dumps(metadata))
    intervention = load_intervention(caa_artifact, model=caa_model)
    tokens = torch.tensor([[2, 8, 9]])
    with torch.inference_mode():
        baseline = caa_model(tokens).logits.clone()
        with apply(caa_model, intervention, strength=1):
            changed = caa_model(tokens).logits
            assert torch.isfinite(changed).all() and not torch.equal(changed, baseline)
        assert torch.equal(caa_model(tokens).logits, baseline)
