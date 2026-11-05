import json
import sys
import types

from evaluation import runner


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
