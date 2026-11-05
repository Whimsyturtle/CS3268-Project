import json
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

    benchmark = types.SimpleNamespace(
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
    assert calls == [expected]
    assert json.loads((run_dir / "config.json").read_text())["decoding"] == expected
