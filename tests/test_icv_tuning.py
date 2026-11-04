from types import SimpleNamespace

import pytest
import torch

from evaluation import runner
from steering.icv.tune import _blind_generations, _mean_log_likelihoods, _plans, _summary_rows


class TinyTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, _chat, **_kwargs):
        assert _kwargs["return_dict"] is True
        return {"input_ids": torch.tensor([[8, 9]])}

    def __call__(self, text, **_kwargs):
        tokens = {"short": [4], "long": [2, 3]}[text]
        return {"input_ids": torch.tensor([tokens])}


class TargetAwareModel:
    device = torch.device("cpu")

    def __call__(self, input_ids, **_kwargs):
        logits = torch.zeros(*input_ids.shape, 10)
        logits[:, :-1].scatter_(-1, input_ids[:, 1:].unsqueeze(-1), input_ids[:, 1:].unsqueeze(-1).float())
        return SimpleNamespace(logits=logits)


class LoadableModel:
    def to(self, _device):
        return self

    def eval(self):
        return self


def test_plans_route_each_mapped_category_and_keep_all_pool():
    categories = [
        "age", "appearance", "disability", "family_structure", "gender",
        "nationality_and_immigration", "race_and_ethnicity", "religion",
        "sexual_orientation", "socioeconomic",
    ]
    records = [{"id": str(i), "category": category} for i, category in enumerate(categories)]

    plans = _plans(records)

    assert len(plans["pooled_all"][0][1]) == 9
    assert len(plans["pooled_bbq_axes"][0][1]) == 9
    assert len(plans["category_routed"]) == 9
    assert all(len(group) == 1 for _, group in plans["category_routed"])


def test_plans_require_every_routed_category():
    with pytest.raises(ValueError, match="every routed"):
        _plans([{"id": "one", "category": "age"}])


def test_summary_rows_report_aggregate_and_categories():
    margins = [
        {"category": "age", "margin": 0.5},
        {"category": "age", "margin": -0.1},
        {"category": "gender", "margin": 0.2},
    ]

    rows = _summary_rows("routed", 0.25, margins)
    aggregate = next(row for row in rows if row["subgroup"] == "all")

    assert aggregate["n"] == 3
    assert aggregate["preference_accuracy"] == pytest.approx(2 / 3)
    assert aggregate["mean_margin"] == pytest.approx(0.2)
    assert {row["subgroup"] for row in rows} == {"all", "age", "gender"}


def test_likelihood_averages_response_tokens_and_ignores_padding():
    scores = _mean_log_likelihoods(
        TargetAwareModel(), TinyTokenizer(), ["one", "two"], ["short", "long"], batch_size=2
    )

    token_score = lambda token: token - torch.log(torch.exp(torch.tensor(float(token))) + 9).item()
    assert scores[0] == pytest.approx(token_score(4), abs=1e-6)
    assert scores[1] == pytest.approx((token_score(2) + token_score(3)) / 2, abs=1e-6)


def test_blind_generations_keep_conditions_only_in_key():
    public, key = _blind_generations([
        {"variant": "pooled", "strength": 0.25, "example_id": "one", "output": "answer"}
    ], seed=0)

    assert public == [{"review_id": "icv-0000", "example_id": "one", "output": "answer"}]
    assert key == [{"review_id": "icv-0000", "variant": "pooled", "strength": 0.25}]


def test_model_loader_records_requested_revision(monkeypatch):
    model = LoadableModel()
    monkeypatch.setattr(runner.AutoTokenizer, "from_pretrained", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(runner.AutoModelForCausalLM, "from_pretrained", lambda *_args, **_kwargs: model)

    loaded, _ = runner.load_model({"name": "test/model", "revision": "abc123", "dtype": "float32"})

    assert loaded._cs3268_source_revision == "abc123"
