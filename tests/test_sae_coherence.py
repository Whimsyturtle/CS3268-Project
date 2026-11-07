import contextlib
import copy
import hashlib
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
        with pytest.raises(ValueError, match="unresolved judgments"):
            audit.judge_pending(config, tmp_path, records)
        for path, content in originals.items():
            assert path.read_bytes() == content
        originals = {p: p.read_bytes() for p in (tmp_path / "raw_judgments").glob("*.json")}
        assert len(originals) == 6 * (invocation + 1)
    assert len(calls) == 12 and "Previous response invalid:" in calls[1]
    valid = json.dumps({"choices": [{"message": {"content": json.dumps(result())}}]})
    monkeypatch.setattr(audit, "request_raw", lambda *args: calls.append(args[1]) or valid)
    completed = audit.judge_pending(config, tmp_path, records)
    assert len(completed) == 2 and len(calls) == 14
    for path, content in originals.items():
        assert path.read_bytes() == content
    # Simulate a crash after raw success but before the completed JSONL write.
    tuning.write_jsonl_atomic(tmp_path / "judgments.jsonl", completed[:1])
    monkeypatch.setattr(audit, "request_raw", lambda *args: pytest.fail("saved valid calls must not repeat"))
    assert audit.judge_pending(config, tmp_path, records) == completed


def test_markdown_quote_retry_keeps_strict_evidence_and_raw_history(tmp_path, monkeypatch):
    config, records = fixture_data()
    item = records[0]
    item["output"] = "**Visual Aids:**  Use diagrams, sketches, and models to explain concepts."
    bad = {**result(1), "evidence": [{"source": "response", "quote": item["output"].replace("**", "")}]}
    good = {**result(1), "evidence": [{"source": "response", "quote": item["output"]}]}
    with pytest.raises(ValueError, match="exact source substring"):
        audit.validate_result(bad, item)
    audit.validate_result(good, item)
    # The other pass already succeeded; the failed pass has three durable replies.
    prior = [{"pass": 2, "review_id": item["review_id"], **result()}]
    tuning.write_jsonl_atomic(tmp_path / "judgments.jsonl", prior)
    originals = {}
    for attempt in range(3):
        path = tmp_path / "raw_judgments" / f"1-{item['review_id']}-{attempt}.json"
        tuning.write_json_atomic(path, {"http_body": json.dumps({"choices": [{"message": {"content": json.dumps(bad)}}]})})
        originals[path] = path.read_bytes()
    calls = []
    monkeypatch.setattr(tuning, "_judge_client", lambda _: contextlib.nullcontext(object()))
    def request(_client, prompt, _settings):
        calls.append(prompt)
        return json.dumps({"choices": [{"message": {"content": json.dumps(good)}}]})
    monkeypatch.setattr(audit, "request_raw", request)
    completed = audit.judge_pending(config, tmp_path, [item])
    assert completed[:1] == prior and completed[1]["evidence"] == good["evidence"]
    assert len(calls) == 1 and "Do not remove ** or paraphrase" in calls[0]
    assert "Apply the same severity rubric" in calls[0]
    saved = json.loads((tmp_path / "raw_judgments" / f"1-{item['review_id']}-3.json").read_text(encoding="utf-8"))
    assert saved["request_prompt"] == calls[0]
    assert all(path.read_bytes() == content for path, content in originals.items())
    audit.judge_pending(config, tmp_path, [item])
    assert len(calls) == 1


def test_saved_markdown_label_failure_recovers_without_api_call(tmp_path, monkeypatch):
    config, records = fixture_data()
    item = records[0]
    lines = ["* **Avoid Complex Visuals:**  Minimize the use of complex animations, gradients, or color palettes.",
             "* **Visual Aids:**  Use diagrams, sketches, and models to explain concepts."]
    item["output"] = "\n\n".join(lines)
    original = {"severity": 1, "reason": "Judge's original claim, still requiring manual review.",
                "evidence": [{"source": "response", "quote": line[2:].replace("**", "")} for line in lines]}
    raw = json.dumps({"choices": [{"message": {"content": json.dumps(original)}}]})
    raw_path = tmp_path / "raw_judgments" / f"1-{item['review_id']}-0.json"
    tuning.write_json_atomic(raw_path, {"http_body": raw})
    before = raw_path.read_bytes()
    prior = [{"pass": 2, "review_id": item["review_id"], **result()}]
    tuning.write_jsonl_atomic(tmp_path / "judgments.jsonl", prior)
    monkeypatch.setattr(tuning, "_judge_client", lambda _: contextlib.nullcontext(object()))
    monkeypatch.setattr(audit, "request_raw", lambda *args: pytest.fail("recover saved reply without API cost"))
    rows = audit.judge_pending(config, tmp_path, [item])
    recovered = rows[1]
    assert rows[:1] == prior and raw_path.read_bytes() == before
    assert recovered["severity"] == original["severity"] and recovered["reason"] == original["reason"]
    assert [e["quote"] for e in recovered["evidence"]] == [line[2:] for line in lines]
    trace = recovered["evidence_recovery"]
    assert trace["raw_file"] == raw_path.name and trace["raw_sha256"] == hashlib.sha256(before).hexdigest()
    for repair in trace["repairs"]:
        assert item["output"][repair["start"]:repair["end"]] == repair["exact_quote"]
        assert repair["original_quote"] == original["evidence"][repair["evidence_index"]]["quote"]
    assert audit.judge_pending(config, tmp_path, [item]) == rows


@pytest.mark.parametrize("source,quote", [
    ("* **Label:** text\n* **Label:** text", "Label: text"),
    ("* **Label:**  text", "Label: text"),
    ("* **Label:** text", "Label: invented"),
    ("* **Label:** text", "label: text"),
    ("* **Label:** text", "bel: text"),
    ("```\n* **Label:** text\n```", "Label: text"),
    ("`\n* **Label:** text\n`", "Label: text"),
    ("~~~\n* **Label:** text\n~~~", "Label: text"),
    ("    * **Label:** text", "Label: text"),
    ("\t* **Label:** text", "Label: text"),
    (" \t* **Label:** text", "Label: text"),
    ("* ** Label:** text", " Label: text"),
    ("* \\**Label:** text", "Label: text"),
    ("* **Label:** `code`", "Label: `code`"),
    ("2**3 + 4**5", "23 + 45"),
])
def test_evidence_recovery_rejects_text_changes_ambiguity_partial_labels_and_code(source, quote):
    assert audit.restore_list_label_quote(source, quote) is None


def test_invalid_judgments_do_not_block_other_items_or_become_zeroes(tmp_path, monkeypatch):
    config, records = fixture_data()
    records = records[:2]
    records[0]["output"] = "UNRESOLVED"
    monkeypatch.setattr(tuning, "_judge_client", lambda _: contextlib.nullcontext(object()))
    def request(_client, prompt, _settings):
        response = ({**result(1), "evidence": [{"source": "response", "quote": "invented"}]}
                    if "UNRESOLVED" in prompt else result())
        return json.dumps({"choices": [{"message": {"content": json.dumps(response)}}]})
    monkeypatch.setattr(audit, "request_raw", request)
    with pytest.raises(ValueError, match="2 unresolved judgments"):
        audit.judge_pending(config, tmp_path, records)
    completed = tuning._read_jsonl(tmp_path / "judgments.jsonl")
    assert len(completed) == 2 and all(r["review_id"] == records[1]["review_id"] for r in completed)
    pending = tuning._read_jsonl(tmp_path / "pending_judgments.jsonl")
    assert len(pending) == 2 and all("severity" not in r for r in pending)
    assert len(list((tmp_path / "raw_judgments").glob("*.json"))) == 8
    with pytest.raises(ValueError, match="incomplete judgment inventory"):
        audit.summarize(config, tmp_path, records)


def test_recovered_zero_labels_are_counted_and_always_manually_reviewed(tmp_path):
    config, records = fixture_data()
    # Pick a zero-rated response outside the ordinary sampled review queue.
    write_judgments(tmp_path, records)
    audit.summarize(config, tmp_path, records)
    sampled = {r["review_id"] for r in tuning._read_jsonl(tmp_path / "review_queue.jsonl")}
    item = next(r for r in records if r["review_id"] not in sampled)
    separate = tmp_path / "recovered"
    judgments = write_judgments(separate, records)
    row = next(r for r in judgments if r["review_id"] == item["review_id"] and r["pass"] == 1)
    row["evidence_recovery"] = {"repairs": [{"original_quote": "format-recovery fixture"}]}
    tuning.write_jsonl_atomic(separate / "judgments.jsonl", judgments)
    summary = audit.summarize(config, separate, records)
    assert sum(r["recovered_evidence_count"] for r in summary["aggregates"]) == 1
    assert item["review_id"] in {r["review_id"] for r in tuning._read_jsonl(separate / "review_queue.jsonl")}


@pytest.fixture
def continuation_run(tmp_path, monkeypatch):
    config, records = fixture_data()
    development = tmp_path / "development.jsonl"
    development.write_bytes(Path(config["examples"]).read_bytes())
    config["examples"] = str(development)
    config["methods"]["sae"]["intervention_dir"] = "unused"
    source = {"commit": "old", "status": "", "uncommitted_diff": ""}
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "read_config", lambda _: copy.deepcopy(config))
    monkeypatch.setattr(tuning, "source_commit", lambda: dict(source))
    monkeypatch.setattr(tuning, "_intervention_provenance", lambda _: {"hash": "original"})
    monkeypatch.setattr(tuning, "generate_records", lambda *args: pytest.fail("no regeneration"))
    monkeypatch.setattr(audit, "generation_environment", lambda: pytest.fail("no GPU required"))
    monkeypatch.setattr(audit, "judge_pending", lambda *args: pytest.fail("no judging before validation"))
    run_dir = tmp_path / "results/coherence/continued"
    tuning.write_jsonl_atomic(run_dir / "generations.jsonl", records)
    audit.run(Path("unused"), "continued", "generate")
    tuning.write_jsonl_atomic(run_dir / "judgments.jsonl", [
        {"pass": 1, "review_id": records[0]["review_id"], **result()}])
    tuning.write_json_atomic(run_dir / "raw_judgments/old.json", {"http_body": "retained"})
    source["commit"] = "new"
    return config, records, run_dir, source


def test_explicit_judging_continuation_preserves_inputs_and_pins_new_source(continuation_run, monkeypatch):
    config, records, run_dir, source = continuation_run
    original = (run_dir / "config.json").read_bytes()
    with pytest.raises(ValueError, match="--resume-judging"):
        audit.run(Path("unused"), "continued", "judge")
    for stage in ("generate", "all", "summarize"):
        with pytest.raises(ValueError, match="only valid"):
            audit.run(Path("unused"), "continued", stage, resume_judging=True)
    for stage in ("generate", "all"):
        with pytest.raises(ValueError, match="changed inputs"):
            audit.run(Path("unused"), "continued", stage)
    monkeypatch.setattr(audit, "judge_pending", lambda *args: None)
    audit.run(Path("unused"), "continued", "judge", resume_judging=True)
    manifest = (run_dir / "judging_continuation.json").read_bytes()
    saved = json.loads(manifest)
    assert saved["original_source"]["commit"] == "old"
    assert saved["source"]["commit"] == "new" and saved["retained_judgments_count"] == 1
    # Resume after a stop immediately following manifest creation, then append work.
    audit.run(Path("unused"), "continued", "judge")
    write_judgments(run_dir, records)
    tuning.write_json_atomic(run_dir / "raw_judgments/new.json", {"http_body": "new"})
    audit.run(Path("unused"), "continued", "summarize")
    assert (run_dir / "summary.json").exists()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["inputs"]["judging_continuation"] == audit.digest(saved)
    assert (run_dir / "config.json").read_bytes() == original
    assert (run_dir / "judging_continuation.json").read_bytes() == manifest
    with pytest.raises(ValueError, match="clean source"):
        source["status"] = "M file"
        audit.run(Path("unused"), "continued", "judge")
    source["status"] = ""
    source["commit"] = "another"
    with pytest.raises(ValueError, match="cannot continue judging a completed summary"):
        audit.run(Path("unused"), "continued", "judge", resume_judging=True)


def test_second_continuation_preserves_and_checks_every_previous_record(continuation_run, monkeypatch):
    _, records, run_dir, source = continuation_run
    monkeypatch.setattr(audit, "judge_pending", lambda *args: None)
    audit.run(Path("unused"), "continued", "judge", resume_judging=True)
    first_path = run_dir / "judging_continuation.json"
    first = first_path.read_bytes()
    # The user already tried the first fix and saved additional failed replies.
    raw_path = run_dir / "raw_judgments/second-attempt.json"
    tuning.write_json_atomic(raw_path, {"http_body": "new failure"})
    source["commit"] = "second-fix"
    with pytest.raises(ValueError, match="--resume-judging"):
        audit.run(Path("unused"), "continued", "judge")
    audit.run(Path("unused"), "continued", "judge", resume_judging=True)
    second_path = run_dir / "judging_continuation-2.json"
    second = second_path.read_bytes()
    saved = json.loads(second)
    assert saved["previous_manifest_sha256"] == hashlib.sha256(first).hexdigest()
    assert "second-attempt.json" in saved["retained_raw_sha256"]
    assert first_path.read_bytes() == first
    audit.run(Path("unused"), "continued", "judge", resume_judging=True)
    assert second_path.read_bytes() == second and len(audit.continuation_paths(run_dir)) == 2
    raw_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed retained raw"):
        audit.run(Path("unused"), "continued", "judge")
    tuning.write_json_atomic(raw_path, {"http_body": "new failure"})
    first_path.write_bytes(first + b"\n")
    with pytest.raises(ValueError, match="changed judging continuation history"):
        audit.run(Path("unused"), "continued", "judge")
    first_path.write_bytes(first)
    write_judgments(run_dir, records)
    audit.run(Path("unused"), "continued", "summarize")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert "judging_continuations" in summary["inputs"]


@pytest.mark.parametrize("change", ["judge", "decoding", "rubric", "data", "intervention", "generation",
                                    "incomplete", "seal", "dirty", "summary"])
def test_continuation_rejects_changed_or_incomplete_inputs(continuation_run, monkeypatch, change):
    config, records, run_dir, source = continuation_run
    if change == "judge":
        config["judge"]["model"] = "different"
    elif change == "decoding":
        config["generation"] = {"max_new_tokens": 1}
    elif change == "rubric":
        monkeypatch.setattr(audit, "RUBRIC", "different")
    elif change == "data":
        path = Path(config["examples"])
        path.write_bytes(path.read_bytes() + b"\n")
    elif change == "intervention":
        monkeypatch.setattr(tuning, "_intervention_provenance", lambda _: {"hash": "changed"})
    elif change == "generation":
        records[0]["output"] = "changed"
        tuning.write_jsonl_atomic(run_dir / "generations.jsonl", records)
    elif change == "incomplete":
        tuning.write_jsonl_atomic(run_dir / "generations.jsonl", records[:-1])
    elif change == "seal":
        (run_dir / "generation_inputs.json").unlink()
    elif change == "dirty":
        source["status"] = "M file"
    elif change == "summary":
        tuning.write_json_atomic(run_dir / "summary.json", {})
    with pytest.raises(ValueError):
        audit.run(Path("unused"), "continued", "judge", resume_judging=True)
    assert not (run_dir / "judging_continuation.json").exists()


@pytest.mark.parametrize("change", ["judgment", "raw", "missing_raw"])
def test_continuation_rejects_changes_to_retained_evidence(continuation_run, monkeypatch, change):
    _, _, run_dir, _ = continuation_run
    monkeypatch.setattr(audit, "judge_pending", lambda *args: None)
    audit.run(Path("unused"), "continued", "judge", resume_judging=True)
    monkeypatch.setattr(audit, "judge_pending", lambda *args: pytest.fail("changed evidence must stop"))
    if change == "judgment":
        rows = tuning._read_jsonl(run_dir / "judgments.jsonl")
        rows[0]["reason"] = "changed"
        tuning.write_jsonl_atomic(run_dir / "judgments.jsonl", rows)
    elif change == "raw":
        tuning.write_json_atomic(run_dir / "raw_judgments/old.json", {"http_body": "changed"})
    else:
        (run_dir / "raw_judgments/old.json").unlink()
    with pytest.raises(ValueError, match="changed retained"):
        audit.run(Path("unused"), "continued", "judge")


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
