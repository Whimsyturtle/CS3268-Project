import contextlib
import copy
import json
from pathlib import Path

import pytest

from steering.sae import audit_coherence as audit
import tuning


def test_audit_paths_still_resolve_from_repository_root():
    assert audit.ROOT == tuning.ROOT
    config = json.loads((audit.ROOT / "configs/sae_coherence.json").read_text(encoding="utf-8"))
    assert (audit.ROOT / config["examples"]).is_file()


def fixture_data():
    development = tuning.read_development(tuning.ROOT / "shared_examples/development.jsonl")
    config = {"seed": 7, "examples": str(tuning.ROOT / "shared_examples/development.jsonl"),
              "methods": {"sae": {"strengths": audit.STRENGTHS}}, "judge": {}}
    records = tuning.assign_review_ids([
        {**condition, "example_id": d["id"], "category": d["category"],
         "prompt": d["prompt"], "output": "An understandable answer."}
        for condition in tuning.plan_conditions(config, "sae") for d in development], config["seed"])
    return config, records


def result(severity=0):
    return {"severity": severity, "reason": "No conflict." if not severity else "Conflicting answer.",
            "evidence": [] if not severity else [{"source": "response", "quote": "answer"}]}


def test_schema_exact_quotes_empty_and_bool_rejection():
    item = {"prompt": "Free event.", "output": "Ticket answer"}
    assert audit.validate_result(result(1), item)["severity"] == 1
    for bad in ({**result(), "severity": True}, {**result(1), "evidence": []},
                {**result(), "reason": " "},
                {**result(1), "evidence": [{"source": "response", "quote": "invented"}]}):
        with pytest.raises(ValueError):
            audit.validate_result(bad, item)
    assert audit.validate_result({**result(3), "evidence": []}, {**item, "output": ""})["severity"] == 3
    with pytest.raises(ValueError, match="empty response"):
        audit.validate_result(result(), {**item, "output": ""})


def test_blind_payload():
    prompt = audit.judge_prompt({"review_id": "hidden-id", "method": "sae",
                                 "strength": 123456, "prompt": "Question", "output": "Answer"})
    assert '"prompt": "Question", "response": "Answer"' in prompt
    for secret in ("hidden-id", "sae", "123456"):
        assert secret not in prompt
    assert "untrusted" in prompt and "external knowledge" in prompt


def test_inventory_requires_exact_conditions_ids_and_prompts():
    config, records = fixture_data()
    assert len(audit.validate_inventory(config, records)) == 520
    variants = [records[:-1], records + records[:1]]
    for field, value in (("prompt", "different"), ("example_id", "unknown"),
                         ("strength", 999), ("review_id", records[1]["review_id"])):
        changed = copy.deepcopy(records)
        changed[0][field] = value
        variants.append(changed)
    for variant in variants:
        with pytest.raises(ValueError):
            audit.validate_inventory(config, variant)


def test_paired_counts_and_zero_interval():
    row = audit.paired_summary([True, False, True, False], [False, True, True, False], 9)
    assert row["rate_difference"] == 0
    assert [row[k] for k in ("newly_affected", "resolved", "both", "neither")] == [1, 1, 1, 1]
    assert row["confidence_interval_95"][0] < 0 < row["confidence_interval_95"][1]
    low, high = audit.wilson(0, 65)
    assert low == pytest.approx(0, abs=1e-15) and 0.05 < high < 0.06


def write_judgments(path, records, positive=None):
    rows = [{"pass": p, "review_id": r["review_id"], **result(int((p, r["review_id"]) == positive))}
            for p in (1, 2) for r in records]
    tuning.write_jsonl_atomic(path / "judgments.jsonl", rows)
    return rows


def test_summary_separate_passes_blind_queue_and_changed_input(tmp_path):
    config, records = fixture_data()
    positive = (1, records[0]["review_id"])
    write_judgments(tmp_path, records, positive)
    summary = audit.summarize(config, tmp_path, records)
    assert sum(r["count"] for r in summary["pass_disagreements"]) == 1
    assert sum(r["thresholds"]["any"]["count"] for r in summary["aggregates"]) == 1
    queue = tuning._read_jsonl(tmp_path / "review_queue.jsonl")
    assert len(queue) == 105  # 1 positive + ceil(.2*64) + seven times ceil(.2*65)
    assert all(set(row) == {"review_id", "prompt", "response"} for row in queue)
    assert any(row["review_id"] == records[0]["review_id"] for row in queue)
    assert audit.summarize(config, tmp_path, records) == json.loads((tmp_path / "summary.json").read_text())
    rows = tuning._read_jsonl(tmp_path / "judgments.jsonl")
    rows[0]["reason"] = "Changed reason"
    tuning.write_jsonl_atomic(tmp_path / "judgments.jsonl", rows)
    with pytest.raises(ValueError, match="changed inputs"):
        audit.summarize(config, tmp_path, records)


def test_judge_resume_persists_raw_and_never_repeats_valid_calls(tmp_path, monkeypatch):
    config, records = fixture_data()
    records = records[:2]
    calls = []
    monkeypatch.setattr(tuning, "_judge_client", lambda _: contextlib.nullcontext(object()))
    monkeypatch.setattr(audit, "request_raw", lambda *args: calls.append(args[1]) or json.dumps({"choices": [{"message": {"content": json.dumps(result())}}]}))
    assert len(audit.judge_pending(config, tmp_path, records)) == 4
    assert len(calls) == 4
    calls.clear()
    audit.judge_pending(config, tmp_path, records)
    assert calls == []
    assert len(list((tmp_path / "raw_judgments").glob("*.json"))) == 4
    records[0]["output"] = "Changed answer."
    with pytest.raises(ValueError, match="changed inputs"):
        audit.judge_pending(config, tmp_path, records)


def test_failed_invocations_append_attempts_and_recover_without_rejudging(tmp_path, monkeypatch):
    config, records = fixture_data()
    records = records[:1]
    calls = []
    monkeypatch.setattr(tuning, "_judge_client", lambda _: contextlib.nullcontext(object()))
    monkeypatch.setattr(audit, "request_raw", lambda *args: calls.append(args[1]) or "malformed")
    originals = {}
    for invocation in range(2):
        with pytest.raises(ValueError, match="three attempts"):
            audit.judge_pending(config, tmp_path, records)
        for path, content in originals.items():
            assert path.read_bytes() == content
        originals = {p: p.read_bytes() for p in (tmp_path / "raw_judgments").glob("*.json")}
        assert len(originals) == 3 * (invocation + 1)
    assert len(calls) == 6 and "Previous response invalid:" in calls[1]
    valid = json.dumps({"choices": [{"message": {"content": json.dumps(result())}}]})
    monkeypatch.setattr(audit, "request_raw", lambda *args: calls.append(args[1]) or valid)
    completed = audit.judge_pending(config, tmp_path, records)
    assert len(completed) == 2 and len(calls) == 8
    for path, content in originals.items():
        assert path.read_bytes() == content
    # Simulate a crash after raw success but before the completed JSONL write.
    tuning.write_jsonl_atomic(tmp_path / "judgments.jsonl", completed[:1])
    monkeypatch.setattr(audit, "request_raw", lambda *args: pytest.fail("saved valid calls must not repeat"))
    assert audit.judge_pending(config, tmp_path, records) == completed


def test_finite_strength_validation_precedes_tuner_load(tmp_path, monkeypatch):
    config = {"methods": {"sae": {"strengths": [0, float("nan")]}}}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    monkeypatch.setattr(tuning, "read_config", lambda _: pytest.fail("must validate before loading"))
    with pytest.raises(ValueError, match="finite"):
        audit.read_config(path)


def test_judge_and_summary_stages_do_not_load_model_or_require_cuda(tmp_path, monkeypatch):
    config, records = fixture_data()
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "read_config", lambda _: config)
    monkeypatch.setattr(tuning, "source_commit", lambda: {"commit": "abc", "status": "", "uncommitted_diff": ""})
    config["methods"]["sae"]["intervention_dir"] = "unused"
    monkeypatch.setattr(tuning, "_intervention_provenance", lambda _: {})
    monkeypatch.setattr(audit, "generation_environment", lambda: pytest.fail("no GPU access"))
    monkeypatch.setattr(tuning, "load_model", lambda _: pytest.fail("no model loading"))
    run_dir = tmp_path / "results/coherence/test"
    tuning.write_jsonl_atomic(run_dir / "generations.jsonl", records)
    tuning.write_json_atomic(run_dir / "generation_inputs.json", {"sha256": audit.digest(records)})
    monkeypatch.setattr(audit, "judge_pending", lambda *args: write_judgments(run_dir, records))
    audit.run(Path("config.json"), "test", "judge")
    audit.run(Path("config.json"), "test", "summarize")
    assert (run_dir / "summary.json").exists()

def test_dirty_generation_does_not_reserve_run_id(tmp_path, monkeypatch):
    config, _ = fixture_data()
    config["methods"]["sae"]["intervention_dir"] = "unused"
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "read_config", lambda _: config)
    monkeypatch.setattr(tuning, "source_commit", lambda: {"commit": "abc", "status": "M file", "uncommitted_diff": "diff"})
    monkeypatch.setattr(tuning, "_intervention_provenance", lambda _: {})
    with pytest.raises(ValueError, match="clean source"):
        audit.run(Path("config.json"), "dirty", "generate")
    assert not (tmp_path / "results/coherence/dirty").exists()


def test_generation_seal_and_completed_all_skip_gpu(tmp_path, monkeypatch):
    config, records = fixture_data()
    config["methods"]["sae"]["intervention_dir"] = "unused"
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "read_config", lambda _: config)
    monkeypatch.setattr(tuning, "source_commit", lambda: {"commit": "abc", "status": "", "uncommitted_diff": ""})
    monkeypatch.setattr(tuning, "_intervention_provenance", lambda _: {})
    monkeypatch.setattr(audit, "generation_environment", lambda: {"gpu": "test"})
    def generate(_config, _method, run_dir):
        tuning.write_jsonl_atomic(run_dir / "generations.jsonl", records)
    monkeypatch.setattr(tuning, "generate_records", generate)
    run_dir = audit.run(Path("config.json"), "generated", "generate")
    assert json.loads((run_dir / "generation_inputs.json").read_text())["sha256"] == audit.digest(records)
    monkeypatch.setattr(audit, "generation_environment", lambda: pytest.fail("completed generation must skip GPU"))
    monkeypatch.setattr(tuning, "generate_records", lambda *args: pytest.fail("no model loading"))
    monkeypatch.setattr(audit, "judge_pending", lambda *args: None)
    monkeypatch.setattr(audit, "summarize", lambda *args: None)
    audit.run(Path("config.json"), "generated", "all")
    records[0]["output"] = "Modified answer"
    tuning.write_jsonl_atomic(run_dir / "generations.jsonl", records)
    with pytest.raises(ValueError, match="changed inputs"):
        audit.run(Path("config.json"), "generated", "judge")


def test_repetition_screen_uses_normalized_word_tokens():
    assert audit.max_repeated_eight_gram("one two three four five six seven eight") == 1
    assert audit.max_repeated_eight_gram(("One, two three four five six seven EIGHT! " * 8)) == 8
