import json
import re
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from shared_examples.prepare_mbtp import CATEGORY_NAMES
from tuning import (
    PROTOCOL,
    RUBRIC,
    _shard_path,
    _resolved_run_config,
    aggregate,
    assign_review_ids,
    generate_condition,
    judge_batch,
    judge_condition,
    make_judge_items,
    pending_judge_items,
    read_config,
    read_development,
    run,
    select_strength,
    select_run,
    summarize_methods,
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
            "batch_size": 1,
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


def test_read_config_requires_independent_judging(tmp_path):
    path = _config(tmp_path)
    config = json.loads(path.read_text(encoding="utf-8"))
    config["judge"]["batch_size"] = 2
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="evaluated independently"):
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


def test_read_development_uses_all_records_and_categories():
    records = read_development(ROOT / "shared_examples" / "development.jsonl")
    assert len(records) == len({record["id"] for record in records}) == 65
    assert Counter(record["category"] for record in records) == Counter(
        {category: 5 for category in CATEGORY_NAMES.values()}
    )


def test_read_development_allows_balanced_expansion(tmp_path):
    records = read_development(ROOT / "shared_examples" / "development.jsonl")
    expanded = records + [{**record, "id": record["id"] + "-second"} for record in records]
    path = tmp_path / "development.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in expanded), encoding="utf-8")
    assert len(read_development(path)) == 130


def test_read_development_rejects_other_balanced_sizes(tmp_path):
    records = read_development(ROOT / "shared_examples" / "development.jsonl")
    extra = []
    for category in CATEGORY_NAMES.values():
        record = next(row for row in records if row["category"] == category)
        extra.append({**record, "id": record["id"] + "-sixth"})
    path = tmp_path / "development.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records + extra), encoding="utf-8")
    with pytest.raises(ValueError, match="five or ten"):
        read_development(path)


def test_shard_paths_use_method_and_strength(tmp_path):
    assert _shard_path(tmp_path, "generations", "icv", 0) == tmp_path / "generations" / "icv" / "0.jsonl"
    assert _shard_path(tmp_path, "judgments", "icv", 0.05) == tmp_path / "judgments" / "icv" / "0.05.jsonl"
    assert _shard_path(tmp_path, "generations", "icv", 1.0000000000000002) != _shard_path(
        tmp_path, "generations", "icv", 1.0000000000000004
    )


def test_assign_review_ids_is_deterministic_opaque_and_unique():
    records = [{"example_id": str(index), "method": "icv", "strength": 0.25} for index in range(10)]
    first = assign_review_ids(records, 7)
    assert first == assign_review_ids(records, 7)
    assert len({record["review_id"] for record in first}) == 10
    assert all(re.fullmatch(r"tune-[0-9a-f]{16}", record["review_id"]) for record in first)
    assert all("icv" not in record["review_id"] and "0.25" not in record["review_id"] for record in first)
    expanded = assign_review_ids(records + [{"example_id": "new", "method": "icv", "strength": 0.1}], 7)
    original_ids = {record["example_id"]: record["review_id"] for record in first}
    assert all(original_ids[record["example_id"]] == record["review_id"]
               for record in expanded if record["example_id"] in original_ids)


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


def test_generate_condition_writes_independent_baseline_and_strength_shards(tmp_path, monkeypatch):
    config = read_config(_config(tmp_path))
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
    run_dir = tmp_path / "run"
    baseline = generate_condition(config, "icv", 0, run_dir)
    steered = generate_condition(config, "icv", 0.1, run_dir)
    assert len(baseline) == len(steered) == 65
    assert calls == {"load": 1, "strengths": [0.1]}
    assert {(row["method"], row["strength"]) for row in baseline} == {("baseline", 0.0)}
    assert {(row["method"], row["strength"]) for row in steered} == {("icv", 0.1)}
    assert _shard_path(run_dir, "generations", "icv", 0).is_file()
    assert _shard_path(run_dir, "generations", "icv", 0.1).is_file()
    with pytest.raises(FileExistsError):
        generate_condition(config, "icv", 0.1, run_dir)


def test_model_loader_records_requested_revision(monkeypatch):
    class LoadableModel:
        config = SimpleNamespace(name_or_path="test/model")

        def to(self, _device):
            return self

        def eval(self):
            return self

    model = LoadableModel()
    monkeypatch.setattr(runner.AutoTokenizer, "from_pretrained", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(runner.AutoModelForCausalLM, "from_pretrained", lambda *_args, **_kwargs: model)
    revision = "a" * 40
    loaded, _ = runner.load_model({"name": "test/model", "revision": revision, "dtype": "float32"})
    assert loaded._cs3268_source_revision == revision


def test_resolved_config_records_development_and_intervention_provenance(tmp_path, monkeypatch):
    path = _config(tmp_path)
    config = read_config(path)
    monkeypatch.setattr("tuning.source_commit", lambda: {"commit": "abc", "status": "", "uncommitted_diff": ""})
    resolved = _resolved_run_config(path, config, "icv")
    assert resolved["protocol"] == PROTOCOL
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
    previous = _resolved_run_config(path, config, "icv")
    previous["source"]["commit"] = "old"
    (run_dir / "config.json").write_text(json.dumps(previous), encoding="utf-8")
    with pytest.raises(ValueError, match="different resolved configuration"):
        run(path, "existing", method="icv", stage="select")


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


def _write_sharded_run(run_dir, records, judgments):
    judgments_by_id = {row["review_id"]: row for row in judgments}
    for strength in sorted({record["strength"] for record in records}):
        shard = [record for record in records if record["strength"] == strength]
        write_jsonl_atomic(_shard_path(run_dir, "generations", "icv", strength), shard)
        write_jsonl_atomic(
            _shard_path(run_dir, "judgments", "icv", strength),
            [judgments_by_id[record["review_id"]] for record in shard],
        )


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
    bad[0] = {**bad[0], "bias_avoidance": 0}
    with pytest.raises(ValueError, match="score"):
        validate_judgments(items, bad)
    bad = [*results]
    bad[0] = {**bad[0], "reason": "First sentence. Second sentence."}
    with pytest.raises(ValueError, match="one nonempty sentence"):
        validate_judgments(items, bad)
    leaked = [{**_generated()[0], "review_id": "icv-0.25-example-1"}]
    with pytest.raises(ValueError, match="opaque"):
        make_judge_items(leaked)


def test_judge_rubric_contains_fixed_anchors():
    for score in range(1, 6):
        assert f"{score} =" in RUBRIC


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


def test_judge_condition_snapshots_items_and_resumes(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    generated = _generated()[:2]
    write_jsonl_atomic(_shard_path(run_dir, "generations", "icv", 0), generated)
    config = {"seed": 0, "examples": str(ROOT / "shared_examples" / "development.jsonl"),
              "judge": {"batch_size": 1}}
    monkeypatch.setattr("tuning.load_judge", lambda _config: (object(), object()))
    calls = []

    def score(_model, _tokenizer, items, _config):
        calls.append([item["review_id"] for item in items])
        return [{"review_id": item["review_id"], "bias_avoidance": 3,
                 "response_quality": 3, "reason": "The response is relevant and avoids stereotypes."}
                for item in items]

    monkeypatch.setattr("tuning.judge_batch", score)
    assert len(judge_condition(config, "icv", 0, run_dir)) == 2
    assert [len(batch) for batch in calls] == [1, 1]
    calls.clear()
    assert len(judge_condition(config, "icv", 0, run_dir)) == 2
    assert calls == []


def test_aggregate_and_select_use_balanced_score_and_smaller_tie():
    generated = _full_generated()
    judgments = []
    for record in generated:
        score = 3 if record["method"] == "baseline" else 4
        judgments.append({"review_id": record["review_id"], "bias_avoidance": score,
                          "response_quality": score, "reason": "Valid reason."})
    rows = aggregate(generated, judgments)
    assert select_strength(rows) == 0.1
    paired = next(row["paired_vs_baseline"] for row in rows if row["method"] == "icv")
    assert paired["mean_difference"] == 1
    assert (paired["improved"], paired["tied"], paired["worsened"]) == (65, 0, 0)
    assert paired["confidence_interval_95"] == [1.0, 1.0]
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


def test_select_run_rejects_missing_shard(tmp_path):
    records = _full_generated()
    judgments = [{"review_id": record["review_id"], "bias_avoidance": 3,
                  "response_quality": 3, "reason": "The response is suitable."} for record in records]
    _write_sharded_run(tmp_path, records, judgments)
    _shard_path(tmp_path, "judgments", "icv", 0.1).unlink()
    config = {
        "seed": 0, "examples": str(ROOT / "shared_examples" / "development.jsonl"),
        "methods": {"icv": {"strengths": [0, 0.1]}},
    }
    with pytest.raises(FileNotFoundError, match="missing completed shard"):
        select_run(config, tmp_path)


def test_select_run_accepts_ten_development_records_per_category(tmp_path, monkeypatch):
    original = read_development(ROOT / "shared_examples" / "development.jsonl")
    development = original + [{**record, "id": record["id"] + "-second"} for record in original]
    examples = tmp_path / "development.jsonl"
    examples.write_text("".join(json.dumps(record) + "\n" for record in development), encoding="utf-8")
    records = []
    for method, strength in [("baseline", 0.0), ("icv", 0.1)]:
        for example in development:
            records.append({
                "review_id": f"tune-{len(records):016x}", "example_id": example["id"],
                "category": example["category"], "prompt": example["prompt"], "output": "answer",
                "method": method, "strength": strength,
            })
    judgments = [{
        "review_id": record["review_id"],
        "bias_avoidance": 4 if record["method"] == "baseline" else 5,
        "response_quality": 4 if record["method"] == "baseline" else 5,
        "reason": "The response is suitable.",
    } for record in records]
    _write_sharded_run(tmp_path, records, judgments)
    monkeypatch.setattr("tuning.summarize_methods", lambda _config, _rows: {})
    summary = select_run({
        "seed": 0, "examples": str(examples), "judge": {},
        "methods": {"icv": {"strengths": [0, 0.1]}},
    }, tmp_path)
    assert summary["selections"] == {"icv": 0.1}
    assert summary["rankings"]["icv"][0]["paired_vs_baseline"]["improved"] == 130


def test_sharded_stages_require_prepare_and_configured_strength(tmp_path, monkeypatch):
    path = _config(tmp_path)
    config = read_config(path)
    monkeypatch.setattr("tuning.ROOT", tmp_path)
    monkeypatch.setattr("tuning.read_config", lambda _path: config)
    monkeypatch.setattr("tuning.source_commit", lambda: {"commit": "abc", "status": "", "uncommitted_diff": ""})
    with pytest.raises(FileNotFoundError, match="prepare"):
        run(path, "sharded", method="icv", stage="generate", strength=0.1)
    run_dir = run(path, "sharded", method="icv", stage="prepare")
    assert (run_dir / "generations" / "icv").is_dir()
    assert (run_dir / "judgments" / "icv").is_dir()
    with pytest.raises(ValueError, match="not configured"):
        run(path, "sharded", method="icv", stage="generate", strength=0.5)


def test_aggregate_rejects_unequal_example_coverage():
    generated = _full_generated()[:-1]
    judgments = [{"review_id": record["review_id"], "bias_avoidance": 3,
                  "response_quality": 3, "reason": "Valid reason."} for record in generated]
    with pytest.raises(ValueError, match="same examples"):
        aggregate(generated, judgments)
