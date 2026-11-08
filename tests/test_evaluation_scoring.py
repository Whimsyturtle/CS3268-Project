import json
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from evaluation import alpaca_eval, bbq, mmlu_pro
from evaluation.common import outputs_by_id, parse_choice


def test_parse_choice_accepts_letters_and_rejects_other_text():
    assert parse_choice("B\n", 3) == 1
    assert parse_choice("**C**", 3) == 2
    assert parse_choice("(A) The grandson", 3) == 0
    assert parse_choice("C\n\nExplanation", 10) == 2
    assert parse_choice("A man was there", 3) is None
    assert parse_choice("D", 3) is None
    assert parse_choice("Here's how to solve it", 10) is None


def test_outputs_by_id_rejects_bad_coverage():
    examples = [{"example_id": "a", "data": {}}, {"example_id": "b", "data": {}}]
    with pytest.raises(ValueError, match="missing"):
        outputs_by_id(examples, [{"example_id": "a", "output": "A"}])
    with pytest.raises(ValueError, match="Duplicate"):
        outputs_by_id(examples, [{"example_id": "a", "output": "A"}] * 2)
    with pytest.raises(ValueError, match="unexpected"):
        outputs_by_id(examples, [{"example_id": i, "output": "A"} for i in "abc"])


def bbq_example(example_id, context, label):
    # ans0 is the stereotyped answer, ans1 the other group, ans2 UNKNOWN.
    data = {
        "category": "Age",
        "context_condition": context,
        "label": label,
        "ans0": "The grandfather",
        "ans1": "The grandson",
        "ans2": "Can't be determined",
        "answer_info": {"ans0": ["grandfather", "old"], "ans1": ["grandson", "nonOld"], "ans2": ["unknown", "unknown"]},
        "target_loc": 0,
    }
    return {"example_id": example_id, "data": data}


def values(metrics, subgroup="all"):
    return {row["metric"]: row["value"] for row in metrics if row["subgroup"] == subgroup}


def test_bbq_scores_counter_stereotypical_answers_as_minus_one():
    examples = [bbq_example("1", "disambig", 1), bbq_example("2", "disambig", 0)]
    result = values(bbq.score(examples, [{"example_id": "1", "output": "B"}, {"example_id": "2", "output": "B"}], {}))
    assert result["bias_score_disambiguated"] == -1
    assert result["accuracy_disambiguated"] == 0.5


def test_bbq_ambiguous_bias_score_is_scaled_and_invalid_outputs_excluded():
    examples = [bbq_example(str(i), "ambig", 2) for i in range(5)]
    outputs = ["A", "A", "A", "C", "I think"]
    predictions = [{"example_id": str(i), "output": output} for i, output in enumerate(outputs)]
    result = values(bbq.score(examples, predictions, {}))
    # 4 valid answers: 1 correct UNKNOWN, 3 biased. Bias = (1 - 1/4) * (2 * 3/3 - 1).
    assert result["accuracy_ambiguous"] == 0.25
    assert result["bias_score_ambiguous"] == pytest.approx(0.75)
    assert result["invalid_rate"] == 0.2
    assert result["bias_score_disambiguated"] is None


def test_mmlu_pro_counts_invalid_outputs_as_wrong():
    examples = [
        {"example_id": str(i), "data": {"options": list("abcd"), "answer_index": 3, "category": "math"}}
        for i in range(4)
    ]
    outputs = ["D", "D", "A", "Let me think"]
    predictions = [{"example_id": str(i), "output": output} for i, output in enumerate(outputs)]
    result = values(mmlu_pro.score(examples, predictions, {}), "math")
    assert result == {"accuracy": 0.5, "invalid_rate": 0.25}


def test_alpaca_eval_uses_configured_judge_and_reference_outputs(monkeypatch):
    monkeypatch.setenv("TEST_SOCLAAS_KEY", "secret")
    captured = {}

    def evaluate(**kwargs):
        captured.update(kwargs)
        captured["annotator"] = json.loads(Path(kwargs["annotators_config"]).read_text())
        completions = captured["annotator"]["soclaas_weighted"]["completions_kwargs"]
        captured["client"] = json.loads(Path(completions["client_config_path"]).read_text())
        leaderboard = pd.DataFrame(
            {
                "win_rate": [52.0],
                "standard_error": [1.5],
                "n_total": [2],
                "avg_length": [20],
            },
            index=["icv"],
        )
        return leaderboard, []

    monkeypatch.setitem(sys.modules, "alpaca_eval", types.SimpleNamespace(evaluate=evaluate))
    examples = [
        {"example_id": "a", "data": {"instruction": "A", "output": "Reference A"}},
        {"example_id": "b", "data": {"instruction": "B", "output": "Reference B"}},
    ]
    predictions = [
        {"example_id": "b", "output": "Candidate B"},
        {"example_id": "a", "output": "Candidate A"},
    ]
    config = {
        "generator": "icv",
        "output_dir": ".",
        "judge": {
            "base_url": "https://example.test/v1",
            "model": "judge-model",
            "reasoning_effort": "none",
            "api_key_env": "TEST_SOCLAAS_KEY",
        },
    }

    result = values(alpaca_eval.score(examples, predictions, config))

    assert captured["model_outputs"] == [
        {"instruction": "A", "output": "Candidate A", "generator": "icv"},
        {"instruction": "B", "output": "Candidate B", "generator": "icv"},
    ]
    assert captured["reference_outputs"][0]["output"] == "Reference A"
    assert Path(captured["annotators_config"]).name == "configs.yaml"
    completions = captured["annotator"]["soclaas_weighted"]["completions_kwargs"]
    assert completions["model_name"] == "judge-model"
    assert completions["requires_chatml"] is True
    assert completions["reasoning_effort"] == "none"
    assert "secret" not in json.dumps(captured["annotator"])
    assert captured["client"] == {"default": [{"api_key": "secret", "base_url": "https://example.test/v1"}]}
    assert captured["fn_metric"] == "get_winrate"
    assert result == {
        "win_rate": 52.0,
        "standard_error": 1.5,
        "avg_length": 20.0,
    }


def test_alpaca_eval_requires_api_key(monkeypatch):
    monkeypatch.delenv("TEST_SOCLAAS_KEY", raising=False)
    config = {
        "generator": "baseline",
        "output_dir": ".",
        "judge": {"api_key_env": "TEST_SOCLAAS_KEY"},
    }
    with pytest.raises(ValueError, match="TEST_SOCLAAS_KEY"):
        alpaca_eval.validate(config)

