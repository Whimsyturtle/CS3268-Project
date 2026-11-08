import json
import sys
from types import SimpleNamespace

from evaluation import runner
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
            assert resolved["intervention_config"]["model"] == config["model"]
        assert (directory / "metrics.csv").exists()
        saved.append(predictions)
    assert saved[0] == saved[1]
    assert not caa_model._forward_pre_hooks and not caa_model.model.layers[0]._forward_hooks


def test_real_caa_artifact_runs_in_shared_development_generation(tmp_path, monkeypatch, caa_artifact, caa_model, caa_tokenizer):
    development = tuning.read_development(tuning.ROOT / "shared_examples/development.jsonl")
    records = [{**row, "prompt": "Question?"} for row in development]
    monkeypatch.setattr(tuning, "read_development", lambda path: records)
    monkeypatch.setattr(tuning, "load_model", lambda config: (caa_model, caa_tokenizer))
    config = {
        "examples": "fixture", "seed": 0,
        "model": {"name": caa_model.config.name_or_path, "revision": caa_model.config.revision, "dtype": "float32"},
        "generation": {"batch_size": 4, "do_sample": False, "max_new_tokens": 2},
        "methods": {"caa": {"intervention_dir": str(caa_artifact), "strengths": [0, 0.5]}},
    }
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
    assert not caa_model._forward_pre_hooks and not caa_model.model.layers[0]._forward_hooks
