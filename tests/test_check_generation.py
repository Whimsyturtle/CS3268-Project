import json
from types import SimpleNamespace

import pytest
import torch

from steering.sae import check_generation as diagnostic


@pytest.fixture
def saved_runs(tmp_path, monkeypatch):
    rows = [{"id": str(i), "prompt": f"prompt {i}"} for i in range(5)]
    config = {
        "examples": "development.jsonl", "seed": 0,
        "model": {"name": "gemma", "revision": "a" * 40, "dtype": "bfloat16"},
        "generation": {"batch_size": 4, "do_sample": False, "max_new_tokens": 2048},
        "development": {"example_ids": [row["id"] for row in rows]},
    }
    directories = [tmp_path / "new", tmp_path / "old"]
    for directory in directories:
        directory.mkdir()
        (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
        records = [{"example_id": row["id"], "prompt": row["prompt"],
                    "method": "baseline", "strength": 0, "output": f"answer {row['id']}"}
                   for row in rows]
        (directory / "generations.jsonl").write_text(
            "\n".join(json.dumps(row) for row in records), encoding="utf-8",
        )
    monkeypatch.setattr(diagnostic, "read_development", lambda path: rows)
    return directories, rows, config


@pytest.mark.parametrize("changed_repeat", [False, True])
def test_repeats_original_batch_and_compares_saved_text(saved_runs, tmp_path, monkeypatch, changed_repeat):
    directories, rows, config = saved_runs
    original_files = {path: path.read_bytes() for directory in directories for path in directory.iterdir()}
    tokenization = []

    def tokenize(chats, **kwargs):
        tokenization.append((chats, kwargs))
        return {"input_ids": torch.tensor([[1, 2]] * 4), "attention_mask": torch.ones((4, 2), dtype=torch.int64)}

    tokenizer = SimpleNamespace(apply_chat_template=tokenize, get_chat_template=lambda: "template")
    model = SimpleNamespace(
        dtype=torch.bfloat16,
        config=SimpleNamespace(_attn_implementation="sdpa", to_dict=lambda: {"model_type": "gemma3"}),
        generation_config=SimpleNamespace(to_dict=lambda: {"num_beams": 1}),
    )
    monkeypatch.setattr(diagnostic.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(diagnostic, "load_model", lambda settings: (model, tokenizer))
    monkeypatch.setattr(diagnostic, "environment", lambda model: {"driver": "test"})
    monkeypatch.setattr(diagnostic, "source_commit", lambda: {"commit": "test"})
    calls = []

    def generate(actual_model, actual_tokenizer, prompts, *, decoding, seed):
        assert actual_model is model and actual_tokenizer is tokenizer
        calls.append((prompts, decoding.copy(), seed))
        outputs = [f"answer {i}" for i in range(4)]
        if changed_repeat and len(calls) == 2:
            outputs[0] = "different answer"
        return outputs

    monkeypatch.setattr(diagnostic, "generate", generate)
    output = tmp_path / "reports" / "check.json"
    report = diagnostic.check(directories[0], directories[1:], output)
    assert calls == [([row["prompt"] for row in rows[:4]], config["generation"], 0)] * 2
    assert tokenizer.padding_side == "left"
    assert tokenization[0][0] == [[{"role": "user", "content": row["prompt"]}] for row in rows[:4]]
    assert tokenization[0][1] == dict(add_generation_prompt=True, padding=True, return_tensors="pt", return_dict=True)
    assert report["checks"]["same_process_equal"] is not changed_repeat
    assert report["checks"]["matches_saved"] == {str(path): [4, 3 if changed_repeat else 4] for path in directories}
    assert (report["checks"]["output_sha256"][0] == report["checks"]["output_sha256"][1]) is not changed_repeat
    assert report["inputs"]["input_ids"] == [[1, 2]] * 4
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert all(path.read_bytes() == data for path, data in original_files.items())


@pytest.mark.parametrize("mismatch", ["settings", "prompt", "order"])
def test_mismatched_reference_fails_before_loading(saved_runs, tmp_path, monkeypatch, mismatch):
    directories, rows, config = saved_runs
    if mismatch == "settings":
        config["generation"]["batch_size"] = 2
        (directories[1] / "config.json").write_text(json.dumps(config), encoding="utf-8")
    elif mismatch == "prompt":
        path = directories[1] / "generations.jsonl"
        path.write_text(path.read_text(encoding="utf-8").replace("prompt 0", "changed prompt"), encoding="utf-8")
    else:
        rows.reverse()
    monkeypatch.setattr(diagnostic, "load_model", lambda config: pytest.fail("Must validate before loading"))
    with pytest.raises(ValueError, match="differ"):
        diagnostic.check(directories[0], directories[1:], tmp_path / "check.json")


def test_existing_report_is_preserved(tmp_path):
    output = tmp_path / "check.json"
    output.write_text("existing report", encoding="utf-8")
    with pytest.raises(FileExistsError):
        diagnostic.check(tmp_path / "missing", [], output)
    assert output.read_text(encoding="utf-8") == "existing report"
