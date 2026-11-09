import json

import numpy as np
import pytest

from evaluation import bbq, compare, mmlu_pro
from evaluation.holistic_bias import scoring


def bbq_example(example_id, context, label):
    # ans0 is the stereotyped answer, ans1 the other group, ans2 UNKNOWN.
    answer_info = {"ans0": ["grandfather", "old"], "ans1": ["grandson", "nonOld"], "ans2": ["unknown", "unknown"]}
    data = {"category": "Age", "context_condition": context, "label": label, "answer_info": answer_info, "target_loc": 0}
    return {"example_id": example_id, "data": data}


def overall(rows):
    return {row["metric"]: row["value"] for row in rows if row["subgroup"] == "all"}


def as_floats(values):
    return {name: float(value[0]) for name, value in values.items()}


BBQ_CASES = [("ambig", 2, "A"), ("ambig", 2, "C"), ("ambig", 2, "B"), ("ambig", 2, "unsure"),
             ("disambig", 0, "A"), ("disambig", 1, "A"), ("disambig", 1, "B"), ("disambig", 0, "C")]


def test_bbq_metrics_match_the_scorer_and_weights_act_like_repeated_examples():
    examples = [bbq_example(str(i), context, label) for i, (context, label, _) in enumerate(BBQ_CASES)]
    outputs = {str(i): output for i, (_, _, output) in enumerate(BBQ_CASES)}
    metrics = compare.bbq_metrics(examples, outputs)
    predictions = [{"example_id": k, "output": v} for k, v in outputs.items()]

    assert as_floats(metrics(np.ones((1, 8)))) == pytest.approx(overall(bbq.score(examples, predictions, {})))

    # Drawing examples 0 and 4 twice equals scoring a copy of each.
    weights = np.ones((1, 8))
    weights[0, [0, 4]] = 2
    copies = examples + [{**examples[i], "example_id": f"copy{i}"} for i in (0, 4)]
    copied = predictions + [{"example_id": f"copy{i}", "output": outputs[str(i)]} for i in (0, 4)]
    assert as_floats(metrics(weights)) == pytest.approx(overall(bbq.score(copies, copied, {})))


def test_mmlu_pro_metrics_match_the_scorer():
    examples = [{"example_id": str(i), "data": {"options": list("abcd"), "answer_index": 3, "category": "math"}} for i in range(4)]
    outputs = {"0": "D", "1": "D", "2": "A", "3": "Let me think"}
    predictions = [{"example_id": k, "output": v} for k, v in outputs.items()]

    result = as_floats(compare.mmlu_pro_metrics(examples, outputs)(np.ones((1, 4))))

    assert result == pytest.approx(overall(mmlu_pro.score(examples, predictions, {})))


def holistic_example(example_id, template, descriptor, axis="ability"):
    return {"example_id": example_id, "data": {"axis": axis, "descriptor": descriptor, "template": template}}


def test_holistic_bias_full_gen_bias_matches_the_scorer_with_and_without_repeats(tmp_path):
    styles = sorted(set().union(*scoring.STYLE_CLUSTERS.values()) | {"other"})
    styles += [f"unused_{i}" for i in range(217 - len(styles))]
    rng = np.random.default_rng(0)
    examples = [holistic_example(str(i), f"t{i % 3}", "abcd"[i % 4]) for i in range(24)]
    examples += [holistic_example("none", "t0", "(none)"), holistic_example("lonely", "t9", "a")]
    probabilities = {e["example_id"]: (lambda v: (v / v.sum()).tolist())(rng.random(217)) for e in examples}

    def scorer_value(items):
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({"styles": styles}))
        path = tmp_path / "probabilities.jsonl"
        path.write_text("".join(json.dumps({"example_id": e["example_id"], "probabilities": probabilities[e["example_id"].split("#")[0]]}) + "\n" for e in items))
        return overall(scoring.generation_bias_metrics(items, path, manifest))["full_gen_bias"]

    metrics = compare.holistic_bias_metrics(examples, probabilities)
    assert metrics(np.ones((1, len(examples))))["full_gen_bias"][0] == pytest.approx(scorer_value(examples))

    weights = np.ones((1, len(examples)))
    weights[0, [1, 5]] = 3
    repeated = examples + [{**examples[i], "example_id": f"{examples[i]['example_id']}#{j}"} for i in (1, 5) for j in (1, 2)]
    assert metrics(weights)["full_gen_bias"][0] == pytest.approx(scorer_value(repeated))


def test_alpaca_eval_win_rate_is_the_mean_preference_above_one():
    metrics = compare.alpaca_eval_metrics([{"preference": 1.0}, {"preference": 2.0}, {"preference": 1.5}, {"preference": 1.9}])
    assert metrics(np.ones((1, 4)))["win_rate"][0] == pytest.approx(60.0)


def test_compare_identical_runs_gives_zero_difference_and_p_of_one():
    keys = [str(i) for i in range(50)]
    scores = np.random.default_rng(1).random(50)
    run = (keys, lambda weights: {"score": weights @ scores / weights.sum(1)}, {"score": scores.mean()})

    (row,) = compare.compare(run, run, samples=500)

    assert row["difference"] == 0 and row["ci95"] == [0.0, 0.0] and row["p"] == 1.0


def test_compare_detects_a_clear_paired_shift_and_rejects_mismatched_examples():
    keys = [str(i) for i in range(200)]
    base = np.random.default_rng(2).random(200)
    baseline = (keys, lambda w: {"score": w @ base / w.sum(1)}, {"score": base.mean()})
    method = (keys, lambda w: {"score": w @ (base + 0.05) / w.sum(1)}, {"score": base.mean() + 0.05})

    (row,) = compare.compare(baseline, method, samples=500)

    assert row["difference"] == pytest.approx(0.05)
    assert row["ci95"] == pytest.approx([0.05, 0.05])
    assert row["p"] == pytest.approx(1 / 501)
    with pytest.raises(ValueError, match="same examples"):
        compare.compare(baseline, (keys[::-1], method[1], method[2]), samples=10)
