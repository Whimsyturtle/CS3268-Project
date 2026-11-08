import csv
import json
import subprocess

from fastapi.testclient import TestClient

from explorer.app import ExplorerSettings, create_app, formatter_source_commit


def write_run(root, method, run_id, benchmark="bbq", max_new_tokens=4):
    path = root / benchmark / method / run_id
    path.mkdir(parents=True)
    config = {
        "benchmark": benchmark, "method": method, "run_id": run_id,
        "source": {"commit": "a" * 40, "status": "", "uncommitted_diff": ""},
        "model": {"name": "google/gemma-3-1b-it", "revision": "b" * 40},
        "method_config": {} if method == "baseline" else {"strength": 0.1},
        "benchmark_config": {"dataset": benchmark, "revision": "c" * 40, "split": "test", "seed": 0},
        "decoding": {"do_sample": False, "max_new_tokens": max_new_tokens},
    }
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (path / "predictions.jsonl").write_text(json.dumps({"example_id": "one", "output": "<script>alert(1)</script>"}) + "\n", encoding="utf-8")
    with (path / "metrics.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["subgroup", "metric", "value", "n"])
        writer.writeheader(); writer.writerow({"subgroup": "all", "metric": "accuracy", "value": 1, "n": 1})


def client(tmp_path):
    write_run(tmp_path, "baseline", "base")
    write_run(tmp_path, "icv", "steered")
    return TestClient(create_app(ExplorerSettings(results_root=tmp_path)))


def test_app_serves_five_view_interface(tmp_path):
    response = client(tmp_path).get("/")
    assert response.status_code == 200
    for label in ["Overview", "Strength", "Generations", "Controls", "Representation"]:
        assert label in response.text
    assert "default-src 'self'" in response.headers["content-security-policy"]


def test_api_lists_runs_metrics_and_compares_exact_outputs(tmp_path):
    api = client(tmp_path)
    assert api.get("/api/status").json()["gpu_required"] is False
    assert len(api.get("/api/runs").json()) == 2
    assert len(api.get("/api/metrics").json()) == 2
    response = api.post("/api/generations", json={"run_keys": ["bbq/baseline/base", "bbq/icv/steered"]})
    assert response.status_code == 200
    assert response.json()["examples"][0]["outputs"]["icv"] == "<script>alert(1)</script>"


def test_frontend_renders_outputs_with_text_content(tmp_path):
    source = client(tmp_path).get("/assets/app.js").text
    assert ".textContent" in source
    assert "innerHTML" not in source
    assert "Math.abs(row.value)" not in source
    assert "Development-selected" in source


def test_unconfigured_optional_views_are_explicit(tmp_path):
    api = client(tmp_path)
    assert api.get("/api/controls").json()["status"] == "unavailable"
    assert api.get("/api/representation").json()["status"] == "unavailable"


def test_local_access_headers_are_enforced(tmp_path):
    api = client(tmp_path)
    assert api.get("/api/status", headers={"host": "example.com"}).status_code == 403
    assert api.get("/api/status", headers={"origin": "https://example.com"}).status_code == 403


def test_formatter_revision_requires_clean_relevant_files(tmp_path):
    (tmp_path / "evaluation" / "bbq").mkdir(parents=True)
    common = tmp_path / "evaluation" / "common.py"
    common.write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "evaluation" / "bbq" / "__init__.py").write_text("", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "evaluation"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "fixture"], cwd=tmp_path, check=True)
    assert formatter_source_commit(tmp_path, "bbq") is not None
    common.write_text("VALUE = 2\n", encoding="utf-8")
    assert formatter_source_commit(tmp_path, "bbq") is None


def test_explicit_experiment_group_pairs_bias_and_capability(tmp_path):
    for method, run_id in [("baseline", "base"), ("icv", "steered")]:
        write_run(tmp_path, method, run_id)
        write_run(tmp_path, method, run_id, benchmark="mmlu_pro")
    group = {
        "name": "main-test", "split_role": "test", "points": [
            {"method": method, "strength": None if method == "baseline" else 0.1,
             "bias": {"run": f"bbq/{method}/{run_id}", "metric": "accuracy", "subgroup": "all"},
             "capability": {"run": f"mmlu_pro/{method}/{run_id}", "metric": "accuracy", "subgroup": "all"}}
            for method, run_id in [("baseline", "base"), ("icv", "steered")]
        ]
    }
    api = TestClient(create_app(ExplorerSettings(results_root=tmp_path, experiment_groups=(group,))))
    response = api.get("/api/groups")
    assert response.status_code == 200
    result = response.json()[0]
    assert result["status"] == "available"
    assert result["split_role"] == "test"
    assert [point["method"] for point in result["points"]] == ["baseline", "icv"]


def test_invalid_experiment_group_is_unavailable_without_breaking_app(tmp_path):
    write_run(tmp_path, "baseline", "base")
    group = {
        "name": "missing", "split_role": "test", "points": [
            {"method": "baseline", "strength": None,
             "bias": {"run": "bbq/baseline/base", "metric": "accuracy", "subgroup": "all"},
             "capability": {"run": "mmlu_pro/baseline/missing", "metric": "accuracy", "subgroup": "all"}}
        ]
    }
    api = TestClient(create_app(ExplorerSettings(results_root=tmp_path, experiment_groups=(group,))))
    result = api.get("/api/groups").json()[0]
    assert result["status"] == "unavailable"
    assert api.get("/api/metrics").status_code == 200


def test_experiment_group_rejects_mixed_axis_identities(tmp_path):
    for method, run_id in [("baseline", "base"), ("icv", "steered")]:
        write_run(tmp_path, method, run_id)
        write_run(tmp_path, method, run_id, benchmark="mmlu_pro")
    group = {
        "name": "mixed", "split_role": "test", "points": [
            {"method": "baseline", "strength": None,
             "bias": {"run": "bbq/baseline/base", "metric": "accuracy", "subgroup": "all"},
             "capability": {"run": "mmlu_pro/baseline/base", "metric": "accuracy", "subgroup": "all"}},
            {"method": "icv", "strength": 0.1,
             "bias": {"run": "bbq/icv/steered", "metric": "different_metric", "subgroup": "all"},
             "capability": {"run": "mmlu_pro/icv/steered", "metric": "accuracy", "subgroup": "all"}}
        ]
    }
    api = TestClient(create_app(ExplorerSettings(results_root=tmp_path, experiment_groups=(group,))))
    result = api.get("/api/groups").json()[0]
    assert result["status"] == "unavailable"
    assert "mixes metric axes" in result["reason"]


def test_experiment_group_rejects_per_axis_token_budget_mismatch(tmp_path):
    write_run(tmp_path, "baseline", "base")
    write_run(tmp_path, "baseline", "base", benchmark="mmlu_pro", max_new_tokens=4)
    write_run(tmp_path, "icv", "steered")
    write_run(tmp_path, "icv", "steered", benchmark="mmlu_pro", max_new_tokens=512)
    group = {
        "name": "budget-mismatch", "split_role": "test", "points": [
            {"method": method, "strength": None if method == "baseline" else 0.1,
             "bias": {"run": f"bbq/{method}/{run_id}", "metric": "accuracy", "subgroup": "all"},
             "capability": {"run": f"mmlu_pro/{method}/{run_id}", "metric": "accuracy", "subgroup": "all"}}
            for method, run_id in [("baseline", "base"), ("icv", "steered")]
        ]
    }
    api = TestClient(create_app(ExplorerSettings(results_root=tmp_path, experiment_groups=(group,))))
    result = api.get("/api/groups").json()[0]
    assert result["status"] == "unavailable"
    assert "generation budgets" in result["reason"]
