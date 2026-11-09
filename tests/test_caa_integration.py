import json
import sys
from types import SimpleNamespace

from evaluation import runner
from steering.caa import build as builder
import tuning


def test_real_caa_artifact_runs_through_shared_runner(tmp_path, monkeypatch, caa_artifact, caa_model, caa_tokenizer):
    config = json.loads((runner.ROOT / "configs/main.json").read_text(encoding="utf-8"))
    original_decoding = dict(config["decoding"])
    config["model"] = {"name": caa_model.config.name_or_path, "revision": caa_model.config.revision, "dtype": "float32"}
    config["methods"]["caa"]["intervention_dir"] = str(caa_artifact)
    config["benchmarks"]["bbq"]["limit"] = 2
    config_path = tmp_path / "runner.json"
    examples = [{"example_id": str(index), "data": {"prompt": prompt}}
                for index, prompt in enumerate(["Question?", "Another question?"])]
    benchmark = SimpleNamespace(
        load_examples=lambda config: examples,
        format_prompt=lambda example, config: example["data"]["prompt"],
        score=lambda examples, predictions, config: [],
    )
    monkeypatch.setitem(sys.modules, "evaluation.bbq", benchmark)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "source_commit", lambda: {"commit": "fixture"})
    monkeypatch.setattr(runner, "load_model", lambda config: (caa_model, caa_tokenizer))
    saved = []
    for method, strength in [("baseline", 0), ("caa", 0), ("caa", 0.5)]:
        config["methods"]["caa"]["strength"] = strength
        config_path.write_text(json.dumps(config), encoding="utf-8")
        directory = runner.run(config_path, method=method, benchmark="bbq", run_id=f"{method}-{strength}")
        predictions = [json.loads(line) for line in (directory / "predictions.jsonl").read_text().splitlines()]
        assert [p["example_id"] for p in predictions] == ["0", "1"]
        assert all(isinstance(p["output"], str) for p in predictions)
        resolved = json.loads((directory / "config.json").read_text())
        assert resolved["decoding"] == {**original_decoding, "max_new_tokens": 8}
        if method == "caa":
            metadata = resolved["intervention_config"]
            assert metadata["model"] == config["model"]["name"]
            assert metadata["revision"] == config["model"]["revision"]
            assert metadata["dtype"] == config["model"]["dtype"]
        assert (directory / "metrics.csv").exists()
        saved.append(predictions)
    assert saved[0] == saved[1]
    assert not caa_model._forward_pre_hooks and not caa_model.model.layers[0]._forward_hooks


def test_built_caa_artifact_passes_tuner_validation_and_resumes_generation(
        tmp_path, monkeypatch, caa_model, caa_tokenizer, caa_pairs):
    construction = [json.loads(line) for line in tuning.CONSTRUCTION_PATH.read_text(encoding="utf-8").splitlines()]
    # Use the approved IDs with short synthetic pairs on the native tiny Gemma.
    pairs = [{**caa_pairs[index % len(caa_pairs)], "id": row["id"]}
             for index, row in enumerate(construction)]
    source = tmp_path / "shared_examples/construction.jsonl"
    source.parent.mkdir()
    source.write_text("".join(json.dumps(row) + "\n" for row in pairs), encoding="utf-8")
    model_config = {"name": caa_model.config.name_or_path, "revision": caa_model.config.revision, "dtype": "float32"}
    experiment_path, settings_path = tmp_path / "experiment.json", tmp_path / "caa.json"
    experiment_path.write_text(json.dumps({"model": model_config, "seed": 0}), encoding="utf-8")
    settings_path.write_text(json.dumps({"layer": 0}), encoding="utf-8")
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    monkeypatch.setattr(builder, "validate_local", lambda *args: {"construction": pairs})
    monkeypatch.setattr(builder, "load_model", lambda config: (caa_model, caa_tokenizer))
    monkeypatch.setattr(builder, "source_commit", lambda: {"commit": "fixture"})
    artifact = builder.build(experiment_path, settings_path, tmp_path / "artifact")
    metadata = json.loads((artifact / "config.json").read_text(encoding="utf-8"))
    assert metadata["source_example_ids"] == [row["id"] for row in construction]
    development = tuning.read_development(tuning.ROOT / "shared_examples/development.jsonl")
    records = [{**row, "prompt": "Question?"} for row in development]
    monkeypatch.setattr(tuning, "read_development", lambda path: records)
    monkeypatch.setattr(tuning, "load_model", lambda config: (caa_model, caa_tokenizer))
    config = json.loads((tuning.ROOT / "configs/tuning.json").read_text(encoding="utf-8"))
    config["model"] = model_config
    config["generation"]["max_new_tokens"] = 2
    config["methods"] = {"caa": {"intervention_dir": str(artifact), "strengths": [0, 0.5]}}
    config_path = tmp_path / "tuning.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    config = tuning.read_config(config_path, method="caa")
    run_dir = tmp_path / "tuning-run"
    generated = tuning.generate_records(config, "caa", run_dir)
    assert len(generated) == 130
    expected = {row["id"] for row in development}
    for method in ("baseline", "caa"):
        subset = [row for row in generated if row["method"] == method]
        assert len(subset) == 65 and {row["example_id"] for row in subset} == expected
    assert len({row["review_id"] for row in generated}) == 130
    assert (tmp_path / "tuning-run/generations.jsonl").exists()
    assert tuning.generate_records(config, "caa", run_dir) == generated
    # Resume after baseline completion; preserve its rows and regenerate only CAA.
    baseline = [row for row in generated if row["method"] == "baseline"]
    tuning.write_jsonl_atomic(run_dir / "generations.jsonl", baseline)
    assert tuning.generate_records(config, "caa", run_dir) == generated
    assert not caa_model._forward_pre_hooks and not caa_model.model.layers[0]._forward_hooks
