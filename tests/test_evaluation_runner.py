import json
import subprocess
import sys
import types

import pytest

from evaluation import runner


@pytest.mark.parametrize("revision", ["main", "short", "g" * 40, None])
def test_load_model_rejects_unpinned_revision_before_loading(revision, monkeypatch):
    def unexpected_load(*args, **kwargs):
        pytest.fail("Unpinned model settings must fail before downloading anything")

    monkeypatch.setattr(runner.AutoTokenizer, "from_pretrained", unexpected_load)
    monkeypatch.setattr(runner.AutoModelForCausalLM, "from_pretrained", unexpected_load)
    with pytest.raises(ValueError, match="commit hash"):
        runner.load_model({"name": "test/gemma", "revision": revision, "dtype": "float32"})


def test_load_model_rejects_unexpected_model_name(monkeypatch):
    monkeypatch.setattr(runner.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object())
    model = types.SimpleNamespace(config=types.SimpleNamespace(name_or_path="other/model"))
    monkeypatch.setattr(runner.AutoModelForCausalLM, "from_pretrained", lambda *args, **kwargs: model)
    with pytest.raises(ValueError, match="model name"):
        runner.load_model({"name": "test/gemma", "revision": "a" * 40, "dtype": "float32"})

def test_run_uses_shared_decoding_with_benchmark_max_new_tokens(tmp_path, monkeypatch):
    config = json.loads((runner.ROOT / "configs" / "main.json").read_text())
    config["benchmarks"]["bbq"]["limit"] = 1
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    validated = []
    benchmark = types.SimpleNamespace(
        validate=lambda config: validated.append(config),
        load_examples=lambda config: [{"example_id": "1", "data": {}}],
        format_prompt=lambda example, config: "prompt",
        score=lambda examples, predictions, config: [],
    )
    calls = []

    def fake_generate(model, tokenizer, prompts, *, decoding, seed):
        calls.append(decoding)
        return ["A"]

    monkeypatch.setitem(sys.modules, "evaluation.bbq", benchmark)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "source_commit", lambda: {})
    monkeypatch.setattr(runner, "load_model", lambda config: (None, None))
    monkeypatch.setattr(runner, "generate", fake_generate)

    run_dir = runner.run(config_path, method="baseline", benchmark="bbq", run_id="test")

    expected = {**config["decoding"], "max_new_tokens": config["benchmarks"]["bbq"]["max_new_tokens"]}
    assert validated == [{**config["benchmarks"]["bbq"], "seed": config["seed"]}]
    assert calls == [expected]
    assert json.loads((run_dir / "config.json").read_text())["decoding"] == expected


def test_holistic_bias_uses_shared_greedy_decoding():
    config = json.loads((runner.ROOT / "configs" / "main.json").read_text())

    benchmark = config["benchmarks"]["holistic_bias"]
    decoding = {**config["decoding"], "max_new_tokens": benchmark["max_new_tokens"]}

    assert benchmark["noun_phrase_selection"] == "canonical"
    assert decoding["do_sample"] is False
    assert "top_p" not in decoding


def test_source_commit_preserves_non_ascii_working_diff(tmp_path, monkeypatch):
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, capture_output=True, check=True)

    git("init")
    notes = tmp_path / "notes.txt"
    notes.write_text("initial\n", encoding="utf-8")
    git("add", "notes.txt")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid",
        "commit", "-m", "fixture")
    changed = "\u66f4\u65b0 \U0001f422"
    notes.write_text(changed + "\n", encoding="utf-8")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    source = runner.source_commit()
    assert len(source["commit"]) == 40
    assert "notes.txt" in source["status"]
    assert "+" + changed in source["uncommitted_diff"]
