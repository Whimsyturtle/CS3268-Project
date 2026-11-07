import copy
import contextlib
import hashlib
import json
from types import SimpleNamespace

import pytest

import cluster_runtime as runtime
import tuning
from evaluation import runner


@pytest.fixture
def cluster(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    (tmp_path / ".python-version").write_text("3.13.18\n")
    (tmp_path / "uv.lock").write_bytes(b"test locked dependencies")
    monkeypatch.delenv("SLURM_JOBID", raising=False)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("VIRTUAL_ENV", "another-environment")
    monkeypatch.setattr(runtime.socket, "gethostname", lambda: "xgpi11")
    monkeypatch.setattr(runtime.platform, "python_implementation", lambda: "CPython")
    monkeypatch.setattr(runtime.platform, "python_version", lambda: "3.13.18")
    monkeypatch.setattr(runtime.sys, "prefix", "active-environment")
    monkeypatch.setattr(runtime.sys, "base_prefix", "base-interpreter")
    monkeypatch.setattr(runtime.sys, "executable", "active-environment/python")
    monkeypatch.setattr(runtime.torch, "__version__", "2.14.1+cu126")
    monkeypatch.setattr(runtime.torch.version, "cuda", "12.6")
    monkeypatch.setattr(runtime.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runtime.torch.cuda, "device_count", lambda: 1)
    device = SimpleNamespace(name=runtime.GPU_NAME, multi_processor_count=60,
                             total_memory=49794777088)
    monkeypatch.setattr(runtime.torch.cuda, "get_device_properties", lambda index: device)
    monkeypatch.setattr(runtime.importlib.metadata, "distributions", lambda: [
        SimpleNamespace(metadata={"Name": "torch"}, version="2.14.1+cu126")])
    calls = []

    def subprocess_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="580.173.02\n" if command[0] == "nvidia-smi" else "", stderr="")

    monkeypatch.setattr(runtime.subprocess, "run", subprocess_run)
    return SimpleNamespace(device=device, calls=calls, root=tmp_path)


@pytest.mark.parametrize("host,job,expected", [
    ("laptop", None, False), ("laptop", "17", True),
    ("xlogin", None, True), ("xlogin2.comp.nus.edu.sg", None, True),
    ("xgpi11", None, True), ("xgph3", None, True),
])
def test_cluster_detection(monkeypatch, host, job, expected):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_JOBID", raising=False)
    if job:
        monkeypatch.setenv("SLURM_JOBID", job)
    monkeypatch.setattr(runtime.socket, "gethostname", lambda: host)
    assert runtime.on_cluster() is expected


def test_local_runs_need_no_environment_or_gpu_checks(monkeypatch):
    monkeypatch.setattr(runtime, "on_cluster", lambda: False)
    monkeypatch.setattr(runtime.torch.cuda, "is_available", lambda: pytest.fail("local GPU check"))
    assert runtime.check_cluster_runtime() is None


def test_cluster_node_requires_allocation(cluster, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID")
    with pytest.raises(RuntimeError, match="Slurm"):
        runtime.check_cluster_runtime()
    assert not cluster.calls


@pytest.mark.parametrize("implementation,version", [("CPython", "3.12.0"), ("PyPy", "3.13.18")])
def test_wrong_python_fails_before_dependency_check(cluster, monkeypatch, implementation, version):
    monkeypatch.setattr(runtime.platform, "python_implementation", lambda: implementation)
    monkeypatch.setattr(runtime.platform, "python_version", lambda: version)
    with pytest.raises(RuntimeError, match="CPython 3.13.18"):
        runtime.check_cluster_runtime()
    assert not cluster.calls


def test_requires_active_virtual_environment(cluster, monkeypatch):
    monkeypatch.setattr(runtime.sys, "prefix", runtime.sys.base_prefix)
    with pytest.raises(RuntimeError, match="project environment"):
        runtime.check_cluster_runtime()


@pytest.mark.parametrize("detail", ["stale lock", "installed packages differ"])
def test_uv_check_failure_stops_run(cluster, monkeypatch, detail):
    monkeypatch.setattr(runtime.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=1, stdout="", stderr=detail))
    with pytest.raises(RuntimeError, match=detail):
        runtime.check_cluster_runtime()


def test_missing_uv_has_actionable_error(cluster, monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("uv")
    monkeypatch.setattr(runtime.subprocess, "run", missing)
    with pytest.raises(RuntimeError, match="uv must be on PATH"):
        runtime.check_cluster_runtime()


def test_cuda_build_mismatch(cluster, monkeypatch):
    monkeypatch.setattr(runtime.torch.version, "cuda", "12.8")
    with pytest.raises(RuntimeError, match="CUDA 12.6"):
        runtime.check_cluster_runtime()


@pytest.mark.parametrize("available,count", [(False, 0), (True, 0), (True, 2)])
def test_requires_one_visible_gpu(cluster, monkeypatch, available, count):
    monkeypatch.setattr(runtime.torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(runtime.torch.cuda, "device_count", lambda: count)
    with pytest.raises(RuntimeError, match="exactly one visible"):
        runtime.check_cluster_runtime()


@pytest.mark.parametrize("name,sms", [("NVIDIA H100 NVL", 132), (runtime.GPU_NAME, 59),
                                     ("NVIDIA A100 MIG 3g.47gb", 60)])
def test_rejects_wrong_actual_gpu_profile(cluster, name, sms):
    cluster.device.name, cluster.device.multi_processor_count = name, sms
    with pytest.raises(RuntimeError, match="Expected.*60 SMs"):
        runtime.check_cluster_runtime()


def test_verified_evidence_and_uv_target_active_interpreter(cluster):
    evidence = runtime.check_cluster_runtime()
    assert evidence == {
        "runtime": {"python": "3.13.18", "torch": "2.14.1+cu126", "cuda": "12.6",
                    "gpu": runtime.GPU_NAME, "gpu_multiprocessors": 60,
                    "gpu_total_memory": 49794777088,
                    "uv_lock_sha256": hashlib.sha256(b"test locked dependencies").hexdigest(),
                    "packages": {"torch": "2.14.1+cu126"}},
        "allocation": {"host": "xgpi11", "slurm_job_id": "123", "driver": "580.173.02"}}
    command, options = cluster.calls[0]
    assert command == ["uv", "sync", "--locked", "--check", "--active", "--offline",
                       "--python", "active-environment/python"]
    assert options["cwd"] == cluster.root
    assert options["env"]["VIRTUAL_ENV"] == "active-environment"


def test_save_and_resume_records_changed_allocation(cluster):
    evidence = runtime.check_cluster_runtime()
    model = SimpleNamespace(_cs3268_cluster_runtime=evidence)
    output = cluster.root / "run"
    runtime.save_cluster_runtime(model, output)
    runtime.save_cluster_runtime(model, output, resuming=True)
    second = copy.deepcopy(evidence)
    second["allocation"].update(host="xgpi12", driver="580.178.04", slurm_job_id="456")
    runtime.save_cluster_runtime(SimpleNamespace(_cs3268_cluster_runtime=second), output, resuming=True)
    saved = json.loads((output / "cluster_runtime.json").read_text())
    assert saved == {"runtime": evidence["runtime"], "allocations": [evidence["allocation"], second["allocation"]]}


def test_resume_rejects_changed_runtime(cluster):
    evidence = runtime.check_cluster_runtime()
    runtime.save_cluster_runtime(SimpleNamespace(_cs3268_cluster_runtime=evidence), cluster.root)
    changed = copy.deepcopy(evidence)
    changed["runtime"]["packages"]["torch"] = "different"
    with pytest.raises(ValueError, match="runtime changed"):
        runtime.save_cluster_runtime(SimpleNamespace(_cs3268_cluster_runtime=changed), cluster.root, resuming=True)


def test_resume_rejects_legacy_partial_run(cluster):
    model = SimpleNamespace(_cs3268_cluster_runtime=runtime.check_cluster_runtime())
    with pytest.raises(ValueError, match="no cluster runtime evidence"):
        runtime.save_cluster_runtime(model, cluster.root, resuming=True)
    assert not (cluster.root / "cluster_runtime.json").exists()


def test_local_append_to_verified_run_rejected(cluster):
    model = SimpleNamespace(_cs3268_cluster_runtime=runtime.check_cluster_runtime())
    runtime.save_cluster_runtime(model, cluster.root)
    with pytest.raises(ValueError, match="verified cluster runtime"):
        runtime.save_cluster_runtime(SimpleNamespace(), cluster.root, resuming=True)


def test_local_save_needs_no_cluster_evidence(tmp_path):
    runtime.save_cluster_runtime(SimpleNamespace(), tmp_path)
    assert not (tmp_path / "cluster_runtime.json").exists()


def test_shared_loader_checks_runtime_before_any_download(monkeypatch):
    def reject():
        raise RuntimeError("unexpected runtime")
    def download(*args, **kwargs):
        pytest.fail("download began before runtime validation")
    monkeypatch.setattr(runner, "check_cluster_runtime", reject)
    monkeypatch.setattr(runner.AutoTokenizer, "from_pretrained", download)
    monkeypatch.setattr(runner.AutoModelForCausalLM, "from_pretrained", download)
    with pytest.raises(RuntimeError, match="unexpected runtime"):
        runner.load_model({"name": "test/gemma", "revision": "a" * 40, "dtype": "float32"})


@pytest.mark.parametrize("method", ["caa", "icv", "sae", "reft_r1"])
def test_tuning_saves_runtime_before_generating_and_completed_run_needs_no_gpu(cluster, monkeypatch, method):
    evidence = runtime.check_cluster_runtime()
    model = SimpleNamespace(_cs3268_cluster_runtime=evidence)
    monkeypatch.setattr(tuning, "read_development", lambda path: [
        {"id": "example-1", "category": "age", "prompt": "A development prompt"}])
    monkeypatch.setattr(tuning, "load_model", lambda config: (model, object()))
    monkeypatch.setattr(runtime.torch.cuda, "is_available", lambda: False)
    monkeypatch.setitem(runtime.sys.modules, f"steering.{method}", SimpleNamespace(
        load_intervention=lambda directory, **kwargs: object(),
        apply=lambda *args, **kwargs: contextlib.nullcontext()))
    calls = []

    def generate(*args, **kwargs):
        saved = json.loads((cluster.root / "cluster_runtime.json").read_text())
        assert saved["runtime"] == evidence["runtime"]
        calls.append(args[2])
        return ["A generated response"]

    monkeypatch.setattr(tuning, "generate", generate)
    config = {"model": {}, "examples": "unused.jsonl", "seed": 7, "generation": {},
              "methods": {method: {"strengths": [0, 1], "intervention_dir": "unused"}}}
    records = tuning.generate_records(config, method, cluster.root)
    assert {(row["method"], row["strength"]) for row in records} == {("baseline", 0), (method, 1)}
    assert calls == [["A development prompt"], ["A development prompt"]]

    def unexpected_load(config):
        pytest.fail("completed generation must allow CPU-only judging")

    monkeypatch.setattr(tuning, "load_model", unexpected_load)
    assert tuning.generate_records(config, method, cluster.root) == records


def test_icv_builder_saves_runtime_before_construction(cluster, monkeypatch):
    from steering.icv import build

    evidence = runtime.check_cluster_runtime()
    model = SimpleNamespace(_cs3268_cluster_runtime=evidence)
    config = {"model": {"name": "test/gemma", "revision": "a" * 40}, "seed": 7}
    config_path = cluster.root / "experiment.json"
    config_path.write_text(json.dumps(config))
    rows = [{"id": str(index), "biased_response": f"biased {index}",
             "unbiased_response": f"unbiased {index}"} for index in range(2)]
    examples = cluster.root / "examples.jsonl"
    examples.write_text("\n".join(json.dumps(row) for row in rows))
    output = cluster.root / "icv"
    monkeypatch.setattr(runtime.sys, "argv", ["build", "--config", str(config_path),
                                               "--examples", str(examples), "--output", str(output)])
    loaded = []

    def load_model(settings):
        loaded.append(settings)
        return model, "tokenizer"

    vector = runtime.torch.tensor([[1.0, 2.0]])

    def construct(actual_model, tokenizer, pairs):
        assert actual_model is model and tokenizer == "tokenizer"
        assert pairs == [(row["biased_response"], row["unbiased_response"]) for row in rows]
        assert json.loads((output / "cluster_runtime.json").read_text())["runtime"] == evidence["runtime"]
        return vector

    monkeypatch.setattr(build, "load_model", load_model)
    monkeypatch.setattr(build, "construct", construct)
    build.main()
    assert loaded == [config["model"]]
    assert runtime.torch.equal(runtime.torch.load(output / "vector.pt", weights_only=True), vector)
    assert json.loads((output / "config.json").read_text())["source_example_ids"] == ["0", "1"]
