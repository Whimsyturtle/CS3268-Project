import hashlib
import json

import pytest
import torch

from steering.caa import build as builder


@pytest.mark.parametrize("response", ["answer", "  answer\n", "answer\nnext", "answer <end_of_turn> next"])
def test_response_content_mask_excludes_prompt_and_chat_markers(caa_tokenizer, response):
    tokens, positions = builder.response_tokens(caa_tokenizer, "prompt", response)
    expected = caa_tokenizer(response.strip(), add_special_tokens=False)["input_ids"]
    expected = [token for token in expected if token not in caa_tokenizer.all_special_ids]
    assert [tokens[p] for p in positions] == expected
    assert positions[-1] < len(tokens) - 2
    with pytest.raises(ValueError, match="no content"):
        builder.response_tokens(caa_tokenizer, "prompt", "  \n")


def test_construct_matches_independent_block_output_mean_and_sign(caa_model, caa_tokenizer, caa_pairs):
    differences = []
    with torch.inference_mode():
        for row in caa_pairs:
            states = {}
            for label in ("unbiased", "biased"):
                tokens, positions = builder.response_tokens(caa_tokenizer, row["prompt"], row[f"{label}_response"])
                captured = []
                handle = caa_model.model.layers[0].register_forward_hook(
                    lambda module, inputs, output: captured.append(output.detach().clone()))
                try:
                    caa_model.model(input_ids=torch.tensor([tokens]), use_cache=False)
                finally:
                    handle.remove()
                states[label] = captured[0][0, positions[-1]].float()
            differences.append(states["unbiased"] - states["biased"])
    expected = torch.stack(differences).mean(0)
    parameters = {name: value.detach().clone() for name, value in caa_model.named_parameters()}
    actual = builder.construct(caa_model, caa_tokenizer, caa_pairs, layer=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    swapped = [{**row, "unbiased_response": row["biased_response"], "biased_response": row["unbiased_response"]} for row in caa_pairs]
    torch.testing.assert_close(builder.construct(caa_model, caa_tokenizer, swapped, layer=0), -actual, rtol=0, atol=0)
    assert actual.dtype == torch.float32 and actual.shape == (16,)
    assert all(torch.equal(value, parameters[name]) for name, value in caa_model.named_parameters())
    assert not caa_model.model.layers[0]._forward_hooks


@pytest.mark.parametrize("case", ["empty", "duplicate", "too_long", "bad_layer"])
def test_construction_rejects_invalid_inputs_and_removes_hook(caa_model, caa_tokenizer, caa_pairs, case):
    rows, layer = caa_pairs, 0
    if case == "empty":
        rows = []
    elif case == "duplicate":
        rows = [caa_pairs[0]] * 2
    elif case == "too_long":
        rows = [{**caa_pairs[0], "unbiased_response": "x" * 300}]
    else:
        layer = 2
    with pytest.raises(ValueError):
        builder.construct(caa_model, caa_tokenizer, rows, layer=layer)
    assert not caa_model.model.layers[0]._forward_hooks


@pytest.mark.parametrize("cluster", [False, True])
def test_build_records_exact_provenance_and_refuses_overwrite(tmp_path, monkeypatch, caa_model, caa_tokenizer, caa_pairs, cluster):
    (tmp_path / "shared_examples").mkdir()
    source = tmp_path / "shared_examples/construction.jsonl"
    source.write_text("source data", encoding="utf-8")
    config = {"model": {"name": caa_model.config.name_or_path, "revision": caa_model.config.revision, "dtype": "float32"}, "seed": 0}
    config_path, settings_path = tmp_path / "experiment.json", tmp_path / "caa.json"
    config_path.write_text(json.dumps(config))
    settings_path.write_text(json.dumps({"layer": 0}))
    if cluster:
        caa_model._cs3268_cluster_runtime = {
            "runtime": {"python": "3.13.18", "torch": "2.11.0+cu126", "cuda": "12.6",
                        "gpu": "NVIDIA H100 NVL MIG 3g.47gb", "gpu_multiprocessors": 60,
                        "gpu_total_memory": 47_000_000_000, "uv_lock_sha256": "b" * 64,
                        "packages": {"torch": "2.11.0+cu126", "transformers": "5.5.4"}},
            "allocation": {"host": "xgpi16", "slurm_job_id": "1234", "driver": "580.126.09"},
        }

    def load_model(model_config):
        assert model_config == config["model"]
        return caa_model, caa_tokenizer

    monkeypatch.setattr(builder, "ROOT", tmp_path)
    monkeypatch.setattr(builder, "validate_local", lambda *args: {"construction": caa_pairs})
    monkeypatch.setattr(builder, "load_model", load_model)
    monkeypatch.setattr(builder, "source_commit", lambda: {"commit": "fixture"})
    output = builder.build(config_path, settings_path, tmp_path / "artifact")
    metadata = json.loads((output / "config.json").read_text())
    assert metadata["model"] == config["model"]["name"]
    assert metadata["revision"] == config["model"]["revision"]
    assert metadata["dtype"] == config["model"]["dtype"]
    assert metadata["tokenizer"] == {
        "name": config["model"]["name"], "revision": config["model"]["revision"],
        "chat_template_sha256": hashlib.sha256(
            json.dumps(caa_tokenizer.chat_template, sort_keys=True).encode("utf-8")).hexdigest(),
    }
    assert metadata["source_example_ids"] == [row["id"] for row in caa_pairs]
    assert metadata["extraction"] == "last_assistant_content_token"
    assert metadata["sign"] == "unbiased_minus_biased" and metadata["normalization"] == "none"
    assert len(metadata["vector_sha256"]) == len(metadata["construction"]["sha256"]) == 64
    assert torch.load(output / "vector.pt", weights_only=True).shape == (16,)
    runtime_path = output / "cluster_runtime.json"
    if cluster:
        assert json.loads(runtime_path.read_text()) == {
            "runtime": caa_model._cs3268_cluster_runtime["runtime"],
            "allocations": [caa_model._cs3268_cluster_runtime["allocation"]],
        }
    else:
        assert not runtime_path.exists()
    with pytest.raises(FileExistsError):
        builder.build(config_path, settings_path, output)
    config["model"]["revision"] = "main"
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="40-character"):
        builder.build(config_path, settings_path, tmp_path / "unpinned")


def test_build_rejects_unverified_runtime_before_construction(tmp_path, monkeypatch, caa_artifact):
    output = tmp_path / "unverified"
    output.mkdir()
    (output / "cluster_runtime.json").write_text(json.dumps({"runtime": {}, "allocations": []}))
    monkeypatch.setattr(builder, "construct", lambda *args, **kwargs: pytest.fail("construction started"))
    with pytest.raises(ValueError, match="verified cluster runtime"):
        builder.build(tmp_path / "experiment.json", tmp_path / "settings.json", output)
    assert not (output / "vector.pt").exists()
