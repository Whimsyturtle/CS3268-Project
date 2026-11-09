import contextlib
import json
from pathlib import Path

import pytest
import torch

from steering.caa import check as checker


@pytest.fixture
def check_setup(tmp_path, monkeypatch, caa_model, caa_tokenizer, caa_artifact, caa_pairs):
    development = tmp_path / "shared_examples/development.jsonl"
    development.write_text("development fixture", encoding="utf-8")
    config = {"model": {"name": caa_model.config.name_or_path,
                        "revision": caa_model.config.revision, "dtype": "float32"},
              "seed": 0, "decoding": {"batch_size": 16, "do_sample": False, "use_cache": True}}
    config_path = tmp_path / "check.json"
    config_path.write_text(json.dumps(config))

    def load_model(model_config):
        assert model_config == config["model"]
        return caa_model, caa_tokenizer

    monkeypatch.setattr(checker, "ROOT", tmp_path)
    monkeypatch.setattr(checker, "load_model", load_model)
    monkeypatch.setattr(checker, "validate_local", lambda *args: {"development": caa_pairs})
    monkeypatch.setattr(checker, "source_commit", lambda: {"commit": "fixture"})
    return config_path, caa_artifact


@pytest.mark.parametrize("cluster", [False, True])
def test_check_records_identity_restoration_provenance_and_raw_outputs(check_setup, caa_model, cluster):
    if cluster:
        caa_model._cs3268_cluster_runtime = {
            "runtime": {"python": "3.13.18", "torch": "2.11.0+cu126", "cuda": "12.6",
                        "gpu": "NVIDIA H100 NVL MIG 3g.47gb", "gpu_multiprocessors": 60,
                        "gpu_total_memory": 47_000_000_000, "uv_lock_sha256": "b" * 64,
                        "packages": {"torch": "2.11.0+cu126", "transformers": "5.5.4"}},
            "allocation": {"host": "xgpi16", "slurm_job_id": "1234", "driver": "580.126.09"},
        }
    parameters = {name: value.detach().clone() for name, value in caa_model.named_parameters()}
    run_dir = checker.run(*check_setup, "verified", max_new_tokens=2)
    report = json.loads((run_dir / "checks.json").read_text())
    assert report["passed"] and all(report["checks"].values())
    assert report["data"]["examples"][0]["id"] == "one"
    assert report["source"] == {"commit": "fixture"}
    assert report["intervention"]["sign"] == "unbiased_minus_biased"
    assert report["environment"]["device"] == "cpu"
    assert report["outputs"]["baseline"]["token_ids"]
    assert len(report["outputs"]["baseline"]["texts"]) == 2
    assert report["nonzero_max_abs_logit_change"] > 0
    runtime_path = run_dir / "cluster_runtime.json"
    if cluster:
        assert json.loads(runtime_path.read_text()) == {
            "runtime": caa_model._cs3268_cluster_runtime["runtime"],
            "allocations": [caa_model._cs3268_cluster_runtime["allocation"]],
        }
    else:
        assert not runtime_path.exists()
    assert all(torch.equal(value, parameters[name]) for name, value in caa_model.named_parameters())
    with pytest.raises(FileExistsError):
        checker.run(*check_setup, "verified", max_new_tokens=2)


def test_check_persists_boolean_failures_before_raising(check_setup, monkeypatch):
    monkeypatch.setattr(checker, "apply", lambda *args, **kwargs: contextlib.nullcontext())
    with pytest.raises(RuntimeError, match="checks failed"):
        checker.run(*check_setup, "ineffective", max_new_tokens=1)
    report = json.loads((checker.ROOT / "results/caa_checks/ineffective/checks.json").read_text())
    assert report["passed"] is False
    assert report["checks"]["nonzero_logits_changed"] is False
    assert report["checks"]["zero_logits_identical"] is True
    assert report["error"]["type"] == "RuntimeError"


def test_check_persists_runtime_error(check_setup, monkeypatch):
    def broken_load(*args, **kwargs):
        raise ValueError("artifact mismatch")

    monkeypatch.setattr(checker, "load_intervention", broken_load)
    with pytest.raises(ValueError, match="artifact mismatch"):
        checker.run(*check_setup, "bad-artifact")
    report = json.loads((checker.ROOT / "results/caa_checks/bad-artifact/checks.json").read_text())
    assert report["passed"] is False
    assert report["error"]["message"] == "artifact mismatch"


def test_observe_restores_generate_after_generation_error(caa_model, caa_tokenizer, monkeypatch):
    def failed_generate(**kwargs):
        raise RuntimeError("generation failed")

    monkeypatch.setattr(caa_model, "generate", failed_generate)
    with pytest.raises(RuntimeError, match="generation failed"):
        checker.observe(caa_model, caa_tokenizer, ["prompt"],
                        {"batch_size": 16, "do_sample": False, "use_cache": True, "max_new_tokens": 1}, 0)
    assert caa_model.generate is failed_generate


@pytest.mark.parametrize("options", [{"strength": 0}, {"strength": float("nan")},
                                     {"prompts": 0}, {"prompts": 17},
                                     {"max_new_tokens": 0}, {"max_new_tokens": 33}])
def test_check_rejects_invalid_smoke_settings(check_setup, options):
    with pytest.raises(ValueError):
        checker.run(*check_setup, "invalid", **options)


def test_caa_strength_grid_uses_the_shared_tuning_config():
    root = Path(__file__).resolve().parents[1]
    shared = json.loads((root / "configs/tuning.json").read_text())
    assert shared["methods"]["caa"] == {
        "intervention_dir": "steering_vectors/caa", "strengths": [0, 0.5, 1, 2]}
