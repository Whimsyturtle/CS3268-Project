import json
import re
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from shared_examples.prepare_mbtp import CATEGORY_NAMES
from tuning import (
    RUBRIC,
    _resolved_run_config,
    aggregate,
    assign_review_ids,
    generate_records,
    judge_batch,
    judge_pending,
    make_judge_items,
    pending_judge_items,
    plan_conditions,
    read_config,
    read_development,
    run,
    select_strength,
    select_run,
    summarize_methods,
    calibrate_judge,
    validate_judgments,
    write_json_atomic,
    write_jsonl_atomic,
)
from evaluation import runner


ROOT = Path(__file__).resolve().parents[1]


def _config(tmp_path):
    artifact = tmp_path / "intervention"
    artifact.mkdir()
    (artifact / "vector.pt").write_bytes(b"vector")
    construction_ids = [
        json.loads(line)["id"]
        for line in (ROOT / "shared_examples" / "construction.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    metadata = {
        "method": "upstream_joint_pca_v1",
        "model": "google/gemma-3-1b-it", "revision": "a" * 40,
        "source_example_ids": construction_ids,
    }
    (artifact / "config.json").write_text(json.dumps(metadata), encoding="utf-8")
    config = {
        "examples": str(ROOT / "shared_examples" / "development.jsonl"),
        "seed": 0,
        "model": {"name": "google/gemma-3-1b-it", "revision": "a" * 40, "dtype": "bfloat16"},
        "generation": {"batch_size": 4, "do_sample": False, "max_new_tokens": 256},
        "judge": {
            "name": "Qwen/Qwen3-4B",
            "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
            "dtype": "bfloat16",
            "batch_size": 5,
            "do_sample": False,
            "max_new_tokens": 2048,
            "enable_thinking": False,
        },
        "methods": {"icv": {"intervention_dir": str(artifact), "strengths": [0, 0.1, 0.25]}},
    }
    path = tmp_path / "tuning.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_read_config_requires_pinned_distinct_judge(tmp_path):
    path = _config(tmp_path)
    assert read_config(path)["judge"]["name"] == "Qwen/Qwen3-4B"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["judge"]["revision"] = "main"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="revision"):
        read_config(path)
    config["judge"]["revision"] = "1cfa9a7208912126459214e8b04321603b3df60c"
    config["judge"]["name"] = config["model"]["name"]
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="different"):
        read_config(path)


def test_read_config_rejects_wrong_source_inventory(tmp_path):
    path = _config(tmp_path)
    config = json.loads(path.read_text(encoding="utf-8"))
    metadata_path = Path(config["methods"]["icv"]["intervention_dir"]) / "config.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["source_example_ids"][0] = "wrong-example"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="260-example intervention"):
        read_config(path)


def test_read_config_allows_method_native_payload_files(tmp_path):
    path = _config(tmp_path)
    config = json.loads(path.read_text(encoding="utf-8"))
    artifact = Path(config["methods"]["icv"]["intervention_dir"])
    (artifact / "vector.pt").unlink()
    (artifact / "method-native.bin").write_bytes(b"payload")
    assert read_config(path)["methods"]["icv"]["intervention_dir"] == str(artifact)


@pytest.mark.parametrize("method", ["caa", "sae", "reft_r1"])
def test_read_config_accepts_nested_model_metadata_without_changing_artifacts(tmp_path, method):
    path = _config(tmp_path)
    config = json.loads(path.read_text(encoding="utf-8"))
    settings = config["methods"].pop("icv")
    config["methods"][method] = settings
    artifact = Path(settings["intervention_dir"])
    metadata_path = artifact / "config.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["model"] = dict(config["model"])
    del metadata["revision"]
    if method == "sae":
        metadata["construction"] = {"example_ids": metadata.pop("source_example_ids")}
    original = json.dumps(metadata)
    metadata_path.write_text(original, encoding="utf-8")
    (artifact / "vector.pt").unlink()
    (artifact / "native-payload.bin").write_bytes(b"method-specific")
    path.write_text(json.dumps(config), encoding="utf-8")
    assert read_config(path)["methods"] == {method: settings}
    assert metadata_path.read_text(encoding="utf-8") == original
    assert (artifact / "native-payload.bin").read_bytes() == b"method-specific"


@pytest.mark.parametrize("field, value", [
    ("name", "wrong/model"), ("revision", "b" * 40), ("dtype", "float32"), ("dtype", None),
])
def test_read_config_rejects_incompatible_nested_model(tmp_path, field, value):
    path = _config(tmp_path)
    config = json.loads(path.read_text(encoding="utf-8"))
    metadata_path = Path(config["methods"]["icv"]["intervention_dir"]) / "config.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["model"] = {**config["model"], field: value}
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="260-example intervention"):
        read_config(path)


@pytest.mark.parametrize("malformation", ["duplicate", "missing", "not-list", "unhashable", "not-object"])
def test_read_config_rejects_malformed_intervention_metadata(tmp_path, malformation):
    path = _config(tmp_path)
    config = json.loads(path.read_text(encoding="utf-8"))
    metadata_path = Path(config["methods"]["icv"]["intervention_dir"]) / "config.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    ids = metadata["source_example_ids"]
    if malformation == "duplicate":
        ids[-1] = ids[0]
    elif malformation == "missing":
        ids.pop()
    elif malformation == "not-list":
        metadata["source_example_ids"] = "invalid"
    elif malformation == "unhashable":
        ids[0] = {}
    else:
        metadata = []
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="260-example intervention|must be an object"):
        read_config(path)


def test_read_config_rejects_duplicate_construction_inventory(tmp_path, monkeypatch):
    path = _config(tmp_path)
    records = (ROOT / "shared_examples" / "construction.jsonl").read_text(encoding="utf-8").splitlines()
    records[-1] = records[0]
    construction = tmp_path / "construction.jsonl"
    construction.write_text("\n".join(records), encoding="utf-8")
    monkeypatch.setattr("tuning.CONSTRUCTION_PATH", construction)
    with pytest.raises(ValueError, match="exactly 260 unique IDs"):
        read_config(path)


def test_read_development_uses_all_records_and_categories():
    records = read_development(ROOT / "shared_examples" / "development.jsonl")
    assert len(records) == len({record["id"] for record in records}) == 65
    assert Counter(record["category"] for record in records) == Counter(
        {category: 5 for category in CATEGORY_NAMES.values()}
    )


def test_plan_uses_one_shared_baseline_and_nonzero_conditions(tmp_path):
    config = read_config(_config(tmp_path))
    config["methods"]["caa"] = {**config["methods"]["icv"], "strengths": [0, 1, 2]}
    conditions = plan_conditions(config, read_development(Path(config["examples"])))
    assert conditions == [
        {"method": "baseline", "strength": 0.0},
        {"method": "caa", "strength": 1.0},
        {"method": "caa", "strength": 2.0},
        {"method": "icv", "strength": 0.1},
        {"method": "icv", "strength": 0.25},
    ]


def test_assign_review_ids_is_deterministic_opaque_and_unique():
    records = [{"example_id": str(index), "method": "icv", "strength": 0.25} for index in range(10)]
    first = assign_review_ids(records, 7)
    assert first == assign_review_ids(records, 7)
    assert len({record["review_id"] for record in first}) == 10
    assert all(re.fullmatch(r"tune-[0-9a-f]{16}", record["review_id"]) for record in first)
    assert all("icv" not in record["review_id"] and "0.25" not in record["review_id"] for record in first)


def test_atomic_json_write_preserves_previous_file_on_replace_failure(tmp_path, monkeypatch):
    path = tmp_path / "value.json"
    write_json_atomic(path, {"value": 1})
    original_replace = type(path).replace

    def fail_replace(self, target):
        if target == path:
            raise OSError("interrupted")
        return original_replace(self, target)

    monkeypatch.setattr(type(path), "replace", fail_replace)
    with pytest.raises(OSError, match="interrupted"):
        write_json_atomic(path, {"value": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"value": 1}


def test_generate_records_reuses_baseline_and_loads_artifact_once(tmp_path, monkeypatch):
    config = read_config(_config(tmp_path))
    conditions = plan_conditions(config, read_development(Path(config["examples"])))
    calls = {"load": 0, "strengths": []}
    model = SimpleNamespace()
    monkeypatch.setattr("tuning.load_model", lambda _config: (model, object()))
    monkeypatch.setattr(
        "tuning.generate",
        lambda _model, _tokenizer, prompts, **_kwargs: [f"answer {index}" for index, _ in enumerate(prompts)],
    )

    def load_intervention(_directory, *, model):
        calls["load"] += 1
        return object()

    def apply(_model, _intervention, *, strength):
        calls["strengths"].append(strength)
        return nullcontext()

    monkeypatch.setattr(
        "tuning.importlib.import_module",
        lambda _name: SimpleNamespace(load_intervention=load_intervention, apply=apply),
    )
    records = generate_records(config, conditions, tmp_path / "run")
    assert len(records) == 65 * 3
    assert calls == {"load": 1, "strengths": [0.1, 0.25]}
    assert Counter((row["method"], row["strength"]) for row in records) == Counter({
        ("baseline", 0.0): 65, ("icv", 0.1): 65, ("icv", 0.25): 65,
    })
    with pytest.raises(FileExistsError):
        generate_records(config, conditions, tmp_path / "run")


def test_model_loader_records_requested_revision(monkeypatch):
    class LoadableModel:
        config = SimpleNamespace()

        def to(self, _device):
            return self

        def eval(self):
            return self

    model = LoadableModel()
    monkeypatch.setattr(runner.AutoTokenizer, "from_pretrained", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(runner.AutoModelForCausalLM, "from_pretrained", lambda *_args, **_kwargs: model)
    loaded, _ = runner.load_model({"name": "test/model", "revision": "abc123", "dtype": "float32"})
    assert loaded.config.revision == "abc123"
    assert loaded._cs3268_source_revision == "abc123"


def test_resolved_config_records_development_and_intervention_provenance(tmp_path, monkeypatch):
    path = _config(tmp_path)
    config = read_config(path)
    monkeypatch.setattr("tuning.source_commit", lambda: {"commit": "abc", "status": "", "uncommitted_diff": ""})
    resolved = _resolved_run_config(path, config, None)
    assert len(resolved["development"]["example_ids"]) == 65
    assert resolved["development"]["categories"] == {category: 5 for category in sorted(CATEGORY_NAMES.values())}
    assert len(resolved["interventions"]["icv"]["source_example_ids"]) == 260


def test_resume_rejects_changed_source_state(tmp_path, monkeypatch):
    path = _config(tmp_path)
    config = read_config(path)
    monkeypatch.setattr("tuning.ROOT", tmp_path)
    monkeypatch.setattr("tuning.read_config", lambda _path: config)
    monkeypatch.setattr("tuning.source_commit", lambda: {"commit": "new", "status": "", "uncommitted_diff": ""})
    run_dir = tmp_path / "results" / "tuning" / "existing"
    run_dir.mkdir(parents=True)
    previous = _resolved_run_config(path, config, None)
    previous["source"]["commit"] = "old"
    (run_dir / "config.json").write_text(json.dumps(previous), encoding="utf-8")
    with pytest.raises(ValueError, match="different resolved configuration"):
        run(path, "existing", stage="select")


def test_repository_has_only_common_tuning_entry_point():
    assert not (ROOT / "steering" / "icv" / "tune.py").exists()
    readme = (ROOT / "steering" / "icv" / "README.md").read_text(encoding="utf-8")
    shared_readme = (ROOT / "steering" / "README.md").read_text(encoding="utf-8")
    assert "[common tuner](../README.md)" in readme
    assert "python tuning.py" in shared_readme
    assert "python -m steering.icv.tune" not in readme


def _generated():
    rows = []
    for method, strength in [("baseline", 0.0), ("icv", 0.1)]:
        for index, category in enumerate(["age", "gender"]):
            rows.append({
                "review_id": f"tune-{len(rows):016x}", "example_id": f"e{index}",
                "category": category, "prompt": f"prompt {index}", "output": f"output {index}",
                "method": method, "strength": strength,
            })
    return rows


def _full_generated():
    rows = []
    development = read_development(ROOT / "shared_examples" / "development.jsonl")
    for method, strength in [("baseline", 0.0), ("icv", 0.1)]:
        for index, example in enumerate(development):
            rows.append({
                "review_id": f"tune-{len(rows):016x}", "example_id": example["id"],
                "category": example["category"], "prompt": example["prompt"], "output": f"output {index}",
                "method": method, "strength": strength,
            })
    return rows


def test_judge_payload_is_blinded_and_validation_is_exact():
    items = make_judge_items(_generated())
    assert set(items[0]) == {"review_id", "prompt", "output"}
    results = [{
        "review_id": item["review_id"], "bias_avoidance": 4,
        "response_quality": 3, "reason": "Relevant and individualized."
    } for item in items]
    assert validate_judgments(items, results) == results
    with pytest.raises(ValueError, match="coverage"):
        validate_judgments(items, results[:-1])
    bad = [*results]
    bad[0] = {**bad[0], "bias_avoidance": 5}
    with pytest.raises(ValueError, match="score"):
        validate_judgments(items, bad)
    bad = [*results]
    bad[0] = {**bad[0], "reason": "First sentence. Second sentence."}
    with pytest.raises(ValueError, match="one nonempty sentence"):
        validate_judgments(items, bad)
    leaked = [{**_generated()[0], "review_id": "icv-0.25-example-1"}]
    with pytest.raises(ValueError, match="opaque"):
        make_judge_items(leaked)


def test_judge_rubric_contains_fixed_anchors_and_special_cases():
    for score in range(5):
        assert f"{score} =" in RUBRIC
    assert "fluent answer retaining a substantial stereotype" in RUBRIC
    assert "unbiased but irrelevant answer or unnecessary refusal" in RUBRIC


def test_judge_batch_repairs_once_with_schema_and_items(monkeypatch):
    items = make_judge_items(_generated()[:1])
    prompts = []
    valid = json.dumps([{
        "review_id": items[0]["review_id"], "bias_avoidance": 4,
        "response_quality": 4, "reason": "The answer is relevant and avoids stereotypes."
    }])

    def model_text(_model, _tokenizer, prompt, _settings):
        prompts.append(prompt)
        return "{}" if len(prompts) == 1 else valid

    monkeypatch.setattr("tuning._model_text", model_text)
    config = {"judge": {"do_sample": False, "max_new_tokens": 10}}
    assert judge_batch(object(), object(), items, config)[0]["bias_avoidance"] == 4
    assert len(prompts) == 2
    assert RUBRIC in prompts[1] and items[0]["review_id"] in prompts[1] and "{}" in prompts[1]


def test_judge_batch_stops_after_two_total_attempts(monkeypatch):
    calls = []
    monkeypatch.setattr("tuning._model_text", lambda *_args: calls.append(1) or "not json")
    with pytest.raises(ValueError, match="two total attempts"):
        judge_batch(object(), object(), make_judge_items(_generated()[:1]),
                    {"judge": {"do_sample": False, "max_new_tokens": 10}})
    assert len(calls) == 2


def test_pending_judge_items_excludes_completed_ids():
    items = make_judge_items(_generated())
    assert pending_judge_items(items, [{"review_id": items[0]["review_id"]}]) == items[1:]


def test_calibration_requires_higher_unbiased_bias_avoidance(monkeypatch):
    development = read_development(ROOT / "shared_examples" / "development.jsonl")

    def equal_scores(_model, _tokenizer, items, _config):
        return [{"review_id": item["review_id"], "bias_avoidance": 2,
                 "response_quality": 3, "reason": "The response receives a calibration score."}
                for item in items]

    monkeypatch.setattr("tuning._score_items", equal_scores)
    result = calibrate_judge({"seed": 0, "judge": {}}, development, model=object(), tokenizer=object())
    assert result == {"pairs": 13, "mean_bias_avoidance": {"biased": 2.0, "unbiased": 2.0}, "passed": False}


def test_judge_pending_snapshots_batches_and_resumes(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    generated = _generated()
    (run_dir / "generations.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in generated), encoding="utf-8"
    )
    config = {"seed": 0, "examples": str(ROOT / "shared_examples" / "development.jsonl"),
              "judge": {"batch_size": 2}}
    monkeypatch.setattr("tuning.load_judge", lambda _config: (object(), object()))
    monkeypatch.setattr("tuning.calibrate_judge", lambda *_args, **_kwargs: {"pairs": 13, "passed": True})
    calls = []

    def score(_model, _tokenizer, items, _config):
        calls.append([item["review_id"] for item in items])
        return [{"review_id": item["review_id"], "bias_avoidance": 3,
                 "response_quality": 3, "reason": "The response is relevant and avoids stereotypes."}
                for item in items]

    monkeypatch.setattr("tuning.judge_batch", score)
    assert len(judge_pending(config, run_dir)) == 4
    assert [len(batch) for batch in calls] == [2, 2]
    calls.clear()
    assert len(judge_pending(config, run_dir)) == 4
    assert calls == []


def test_aggregate_and_select_use_balanced_score_and_smaller_tie():
    generated = _full_generated()
    judgments = []
    for record in generated:
        score = 3 if record["method"] == "baseline" else 4
        judgments.append({"review_id": record["review_id"], "bias_avoidance": score,
                          "response_quality": score, "reason": "Valid reason."})
    assert select_strength(aggregate(generated, judgments)) == 0.1
    for judgment in judgments:
        judgment["bias_avoidance"] = judgment["response_quality"] = 3
    assert select_strength(aggregate(generated, judgments)) == 0.0


def test_select_strength_uses_smaller_nonzero_strength_on_tie():
    rows = [
        {"method": "baseline", "strength": 0.0, "mean_balanced_score": 2.0},
        {"method": "icv", "strength": 0.1, "mean_balanced_score": 3.0},
        {"method": "icv", "strength": 0.25, "mean_balanced_score": 3.0},
    ]
    assert select_strength(rows) == 0.1


def test_summary_prompt_is_condition_anonymous(monkeypatch):
    prompts = []
    monkeypatch.setattr("tuning.load_judge", lambda _config: (object(), object()))
    monkeypatch.setattr(
        "tuning._model_text",
        lambda _model, _tokenizer, prompt, _settings: prompts.append(prompt) or "Concise summary.",
    )
    rows = [{
        "method": "icv", "strength": 0.25, "mean_bias_avoidance": 3.0,
        "mean_response_quality": 4.0, "mean_balanced_score": 3.0,
        "categories": {"gender": {"n": 5, "mean_balanced_score": 3.0}},
        "reasons": ["Response tune-0123456789abcdef avoids unsupported assumptions."],
    }]
    summaries = summarize_methods({"seed": 0, "judge": {}}, rows)
    assert summaries == {"icv:0.25": "Concise summary."}
    assert "icv" not in prompts[0] and "0.25" not in prompts[0] and "gender" not in prompts[0]
    assert "tune-0123456789abcdef" not in prompts[0]
    assert "rank conditions" in prompts[0] and "recommend a selection" in prompts[0]


def test_select_run_rejects_unconfigured_condition(tmp_path):
    records = _full_generated()
    extra = []
    for index, record in enumerate(records[:65]):
        extra.append({**record, "review_id": f"tune-{len(records) + index:016x}",
                      "method": "caa", "strength": 1.0})
    records += extra
    judgments = [{"review_id": record["review_id"], "bias_avoidance": 3,
                  "response_quality": 3, "reason": "The response is suitable."} for record in records]
    write_jsonl_atomic(tmp_path / "generations.jsonl", records)
    write_jsonl_atomic(tmp_path / "judgments.jsonl", judgments)
    write_json_atomic(tmp_path / "summary.json", {"status": "judging", "calibration": {"passed": True}})
    config = {"methods": {"icv": {"strengths": [0, 0.1]}}}
    with pytest.raises(ValueError, match="exactly match"):
        select_run(config, tmp_path)


def test_run_all_writes_complete_artifacts(tmp_path, monkeypatch):
    path = _config(tmp_path)
    config = read_config(path)
    development = read_development(Path(config["examples"]))
    monkeypatch.setattr("tuning.ROOT", tmp_path)
    monkeypatch.setattr("tuning.read_config", lambda _path: config)
    monkeypatch.setattr("tuning.read_development", lambda _path: development)
    monkeypatch.setattr("tuning.source_commit", lambda: {"commit": "abc", "status": "", "uncommitted_diff": ""})
    monkeypatch.setattr("tuning.summarize_methods", lambda _config, _rows: {"anonymous": "summary"})

    def fake_generate(_config, conditions, run_dir):
        records = []
        for condition in conditions:
            for example in development:
                records.append({
                    "review_id": f"tune-{len(records):016x}", "example_id": example["id"],
                    "category": example["category"], "prompt": example["prompt"], "output": "answer",
                    "method": condition["method"], "strength": condition["strength"],
                })
        write_jsonl_atomic(run_dir / "generations.jsonl", records)
        return records

    def fake_judge(_config, run_dir):
        records = [json.loads(line) for line in (run_dir / "generations.jsonl").read_text(encoding="utf-8").splitlines()]
        judgments = [{"review_id": record["review_id"], "bias_avoidance": 3,
                      "response_quality": 3, "reason": "The response is suitable."} for record in records]
        write_jsonl_atomic(run_dir / "judgments.jsonl", judgments)
        write_json_atomic(run_dir / "summary.json", {"status": "judging", "calibration": {"passed": True}})
        return judgments

    monkeypatch.setattr("tuning.generate_records", fake_generate)
    monkeypatch.setattr("tuning.judge_pending", fake_judge)
    run_dir = run(path, "integration", method="icv", stage="all")
    assert {file.name for file in run_dir.iterdir()} == {
        "config.json", "generations.jsonl", "judgments.jsonl", "summary.json"
    }
    assert json.loads((run_dir / "summary.json").read_text())["status"] == "complete"


def test_aggregate_rejects_unequal_example_coverage():
    generated = _full_generated()[:-1]
    judgments = [{"review_id": record["review_id"], "bias_avoidance": 3,
                  "response_quality": 3, "reason": "Valid reason."} for record in generated]
    with pytest.raises(ValueError, match="same examples"):
        aggregate(generated, judgments)
