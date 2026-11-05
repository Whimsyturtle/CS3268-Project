import json
import sys
import types

import pytest
import torch

from evaluation import runner
from evaluation.common import outputs_by_id


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


@pytest.mark.parametrize("ids, message", [([], "at least one"), (["1", "1"], "Duplicate example")])
def test_run_rejects_invalid_selection_before_loading_model(tmp_path, monkeypatch, ids, message):
    config = json.loads((runner.ROOT / "configs" / "main.json").read_text())
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    monkeypatch.setitem(sys.modules, "evaluation.bbq", types.SimpleNamespace(
        load_examples=lambda _config: [{"example_id": value, "data": {}} for value in ids],
        format_prompt=lambda *_args: pytest.fail("invalid examples must not be formatted"),
    ))
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "source_commit", lambda: {})
    monkeypatch.setattr(runner, "load_model", lambda _config: pytest.fail("model must not load"))
    with pytest.raises(ValueError, match=message):
        runner.run(config_path, method="baseline", benchmark="bbq", run_id="invalid")


def test_scoring_rejects_duplicate_example_ids():
    with pytest.raises(ValueError, match="Duplicate example"):
        outputs_by_id([{"example_id": "1"}, {"example_id": "1"}],
                      [{"example_id": "1", "output": "A"}])


@pytest.mark.parametrize("outputs", [[], ["A", "B"]])
def test_run_rejects_output_count_mismatch_before_saving_predictions(tmp_path, monkeypatch, outputs):
    config = json.loads((runner.ROOT / "configs" / "main.json").read_text())
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    monkeypatch.setitem(sys.modules, "evaluation.bbq", types.SimpleNamespace(
        load_examples=lambda _config: [{"example_id": "1", "data": {}}],
        format_prompt=lambda *_args: "prompt",
        score=lambda *_args: pytest.fail("incomplete predictions must not be scored"),
    ))
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "source_commit", lambda: {})
    monkeypatch.setattr(runner, "load_model", lambda _config: (None, None))
    monkeypatch.setattr(runner, "generate", lambda *_args, **_kwargs: outputs)
    with pytest.raises(ValueError, match="zip"):
        runner.run(config_path, method="baseline", benchmark="bbq", run_id="invalid")
    assert not (tmp_path / "results" / "bbq" / "baseline" / "invalid" / "predictions.jsonl").exists()


@pytest.mark.parametrize("output_counts", [[1], [3], [3, 1]])
def test_generate_rejects_wrong_output_count_per_batch(output_counts):
    class Inputs(dict):
        def to(self, _device):
            return self

    class Tokenizer:
        def apply_chat_template(self, chats, **_kwargs):
            return Inputs(input_ids=torch.ones(len(chats), 2, dtype=torch.long))

        def batch_decode(self, tokens, **_kwargs):
            return ["A"] * tokens.shape[0]

    counts = iter(output_counts)
    model = types.SimpleNamespace(device="cpu", generate=lambda **_kwargs: torch.ones(next(counts), 3))
    with pytest.raises(ValueError, match="one output per prompt in each batch"):
        runner.generate(model, Tokenizer(), ["prompt"] * 4, decoding={"batch_size": 2}, seed=0)
