import csv
import json

import numpy as np
import pytest

from explorer.analysis import control_summary, load_capture_bundle, load_control_file
from explorer.results import ExplorerDataError, discover_runs, join_generations, load_run


def write_run(root, benchmark="bbq", method="baseline", run_id="run-1", *, seed=0, ids=("a", "b")):
    path = root / benchmark / method / run_id
    path.mkdir(parents=True)
    config = {
        "benchmark": benchmark,
        "method": method,
        "run_id": run_id,
        "source": {"commit": "a" * 40, "status": "", "uncommitted_diff": ""},
        "model": {"name": "google/gemma-3-1b-it", "revision": "b" * 40},
        "method_config": {} if method == "baseline" else {"strength": 0.1},
        "benchmark_config": {"dataset": "dataset", "revision": "c" * 40, "split": "test", "seed": seed},
        "decoding": {"do_sample": False, "max_new_tokens": 4},
    }
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (path / "predictions.jsonl").write_text("".join(json.dumps({"example_id": value, "output": f"output {value}"}) + "\n" for value in ids), encoding="utf-8")
    with (path / "metrics.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["subgroup", "metric", "value", "n"])
        writer.writeheader()
        writer.writerow({"subgroup": "all", "metric": "accuracy", "value": 0.5, "n": 2})
    return path


def test_discover_and_join_runs(tmp_path):
    baseline = write_run(tmp_path, method="baseline", run_id="base")
    steered = write_run(tmp_path, method="icv", run_id="steered")
    runs = discover_runs(tmp_path)
    assert [run.identity.method for run in runs] == ["baseline", "icv"]
    comparison = join_generations([load_run(baseline), load_run(steered)])
    assert comparison["examples"][0]["outputs"] == {"baseline": "output a", "icv": "output a"}
    assert comparison["prompt_status"] == "unavailable"


def test_duplicate_prediction_ids_are_rejected(tmp_path):
    path = write_run(tmp_path, ids=("a", "a"))
    with pytest.raises(ExplorerDataError, match="Duplicate example_id"):
        load_run(path)


def test_mismatched_example_sets_are_rejected(tmp_path):
    baseline = load_run(write_run(tmp_path, method="baseline", run_id="base"))
    steered = load_run(write_run(tmp_path, method="icv", run_id="steered", ids=("a", "c")))
    with pytest.raises(ExplorerDataError, match="example-ID sets"):
        join_generations([baseline, steered])


def test_run_child_file_symlink_cannot_escape_results_root(tmp_path):
    root = tmp_path / "results"
    path = write_run(root)
    outside = tmp_path / "outside.jsonl"
    outside.write_text(json.dumps({"example_id": "a", "output": "outside"}) + "\n", encoding="utf-8")
    predictions = path / "predictions.jsonl"
    predictions.unlink()
    try:
        predictions.symlink_to(outside)
    except OSError:
        pytest.skip("File symlinks are unavailable on this Windows configuration")
    with pytest.raises(ExplorerDataError, match="outside configured root"):
        load_run(path, results_root=root)


def test_control_summary_uses_midrank_percentile():
    summary = control_summary(2.0, [0, 1, 2, 2, 3], family="norm-matched random directions")
    assert summary["status"] == "available"
    assert summary["percentile"] == 60.0


def test_control_file_requires_provenance(tmp_path):
    path = tmp_path / "controls.json"
    path.write_text(json.dumps({"family": "norm-matched random directions", "matching": {}, "metric": {}, "observed": 2, "controls": [0, 1, 2, 2, 3]}), encoding="utf-8")
    with pytest.raises(ExplorerDataError, match="matching criteria"):
        load_control_file(path)


def test_control_file_retains_validated_provenance(tmp_path):
    path = tmp_path / "controls.json"
    record = {
        "family": "norm-matched random directions",
        "matching": {"norm_tolerance": 0.01},
        "metric": {"benchmark": "bbq", "subgroup": "all", "name": "bias_score_ambiguous"},
        "observed": {"run_id": "icv-observed", "method": "icv", "effect": 2.0},
        "controls": [{"run_id": f"control-{index}", "effect": value} for index, value in enumerate([0, 1, 2, 2, 3])],
    }
    path.write_text(json.dumps(record), encoding="utf-8")
    loaded = load_control_file(path)
    assert loaded["observed"] == record["observed"]
    assert loaded["controls"] == record["controls"]
    assert loaded["summary"]["percentile"] == 60.0


def test_capture_bundle_fits_shared_pca(tmp_path):
    bundle = tmp_path / "captures" / "one"
    bundle.mkdir(parents=True)
    (bundle / "capture_config.json").write_text(json.dumps({"model": {"name": "gemma", "revision": "abc"}, "split_role": "development", "source_split": "dev", "baseline_run_id": "base"}), encoding="utf-8")
    rows = [
        {"run_id": "base", "example_id": "a", "method": "baseline", "strength": None, "state": "baseline", "category": "x", "layer": 4, "hook": "residual", "token_selection": "last", "source_split": "dev"},
        {"run_id": "base", "example_id": "b", "method": "baseline", "strength": None, "state": "baseline", "category": "x", "layer": 4, "hook": "residual", "token_selection": "last", "source_split": "dev"},
        {"run_id": "icv", "example_id": "a", "method": "icv", "strength": 0.1, "state": "intervention", "category": "x", "layer": 4, "hook": "residual", "token_selection": "last", "source_split": "dev"},
    ]
    (bundle / "capture_index.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    np.save(bundle / "activations.npy", np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.5, 1.0, 1.0]]))
    result = load_capture_bundle(bundle, capture_root=tmp_path / "captures")
    assert len(result["records"]) == 3
    assert len(result["explained_variance_ratio"]) == 2


def test_capture_path_must_stay_under_root(tmp_path):
    with pytest.raises(ExplorerDataError, match="outside configured root"):
        load_capture_bundle(tmp_path / "outside", capture_root=tmp_path / "captures")


def test_capture_requires_development_identity(tmp_path):
    bundle = tmp_path / "captures" / "one"
    bundle.mkdir(parents=True)
    (bundle / "capture_config.json").write_text(json.dumps({"model": {"name": "gemma", "revision": "abc"}}), encoding="utf-8")
    (bundle / "capture_index.jsonl").write_text("", encoding="utf-8")
    np.save(bundle / "activations.npy", np.ones((2, 2)))
    with pytest.raises(ExplorerDataError, match="development capture"):
        load_capture_bundle(bundle, capture_root=tmp_path / "captures")
