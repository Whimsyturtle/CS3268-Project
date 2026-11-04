import json
import types

import pytest
import torch
from transformers import Gemma3ForCausalLM, Gemma3TextConfig

from steering import reft_r1
from steering.reft_r1 import main

PINNED_MODEL = main.config["model"]
HIDDEN_SIZE = 16


class FakeTokenizer:
    """Tokenizes one character per token, with a simple chat format."""

    pad_token_id = 0

    def apply_chat_template(self, chat, add_generation_prompt=False, tokenize=False):
        text = "".join(f"<{turn['role']}>{turn['content']}" for turn in chat)
        return text + "<assistant>" if add_generation_prompt else text

    def __call__(self, text, add_special_tokens=False):
        return types.SimpleNamespace(input_ids=[1 + ord(c) % 60 for c in text])


EXAMPLE = {"id": "1", "prompt": "Who is best?", "unbiased_response": "Nobody is."}


@pytest.fixture
def model():
    torch.manual_seed(0)
    config = Gemma3TextConfig(
        vocab_size=64,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        name_or_path=PINNED_MODEL["name"],
    )
    return Gemma3ForCausalLM(config).eval()


def intervention():
    vector = torch.zeros(HIDDEN_SIZE)
    vector[0] = 1.0
    return reft_r1.SavedVector(vector=vector, max_act=5.0, layer=1)


def logits(model):
    return model(input_ids=torch.tensor([[1, 2, 3, 4]])).logits


def save(directory, saved, model_config=PINNED_MODEL):
    torch.save(saved, directory / "vector.pt")
    (directory / "config.json").write_text(json.dumps({"model": model_config}))


def test_strength_zero_matches_unsteered_and_strength_changes_output(model):
    unsteered = logits(model)
    with reft_r1.apply(model, intervention(), strength=0.0):
        assert torch.equal(logits(model), unsteered)
    with reft_r1.apply(model, intervention(), strength=1.0):
        assert not torch.equal(logits(model), unsteered)
    assert torch.equal(logits(model), unsteered)


def test_apply_restores_model_after_an_exception(model):
    unsteered = logits(model)
    with (
        pytest.raises(RuntimeError),
        reft_r1.apply(model, intervention(), strength=1.0),
    ):
        raise RuntimeError
    assert not model.model.layers[1]._forward_hooks
    assert torch.equal(logits(model), unsteered)


def test_load_intervention_returns_saved_vector(model, tmp_path):
    save(tmp_path, intervention())
    loaded = reft_r1.load_intervention(tmp_path, model=model)
    assert torch.equal(loaded["vector"], intervention()["vector"])
    assert (loaded["max_act"], loaded["layer"]) == (5.0, 1)


@pytest.mark.parametrize(
    "change",
    [
        {"model_config": {**PINNED_MODEL, "revision": "other"}},
        {"saved": {**intervention(), "vector": torch.zeros(HIDDEN_SIZE + 1)}},
        {"saved": {**intervention(), "layer": 2}},
    ],
)
def test_load_intervention_rejects_a_vector_that_does_not_fit(model, tmp_path, change):
    save(
        tmp_path,
        change.get("saved", intervention()),
        change.get("model_config", PINNED_MODEL),
    )
    with pytest.raises(ValueError):
        reft_r1.load_intervention(tmp_path, model=model)


def test_load_intervention_rejects_a_different_model(model, tmp_path):
    save(tmp_path, intervention())
    model.config.name_or_path = "other/model"
    with pytest.raises(ValueError):
        reft_r1.load_intervention(tmp_path, model=model)


def test_tokenize_example_grades_only_the_response():
    tokenizer = FakeTokenizer()
    tokenized = main.tokenize_example(tokenizer, EXAMPLE)
    prompt = "<user>Who is best?<assistant>"
    n_prompt = len(prompt)
    assert tokenized.labels[:n_prompt] == [-100] * n_prompt
    assert tokenized.labels[n_prompt:] == tokenized.input_ids[n_prompt:]
    assert len(tokenized.labels) == len(tokenized.input_ids) > n_prompt


def test_tokenize_example_rejects_misaligned_prompt_tokens():
    class ShiftedTokenizer(FakeTokenizer):
        def apply_chat_template(
            self, chat, add_generation_prompt=False, tokenize=False
        ):
            text = super().apply_chat_template(chat, add_generation_prompt)
            return text if add_generation_prompt else "x" + text

    with pytest.raises(ValueError):
        main.tokenize_example(ShiftedTokenizer(), EXAMPLE)


def test_padding_is_not_graded_and_bos_and_padding_are_not_scored():
    short = {**EXAMPLE, "unbiased_response": "No."}
    batch = main.make_batch(FakeTokenizer(), [EXAMPLE, short], torch.device("cpu"))
    n_padding = (batch["attention_mask"][1] == 0).sum()
    assert n_padding > 0
    assert (batch["labels"][1, -n_padding:] == -100).all()

    token_mask = main.real_tokens(batch["attention_mask"])
    assert (token_mask[:, 0] == 0).all()
    assert torch.equal(token_mask[:, 1:], batch["attention_mask"][:, 1:])

    hidden = torch.ones(*batch["input_ids"].shape, HIDDEN_SIZE)
    scores = main.score_tokens(hidden, torch.ones(HIDDEN_SIZE), token_mask)
    assert (scores[token_mask == 0] == 0).all()
    assert (scores[token_mask == 1] == HIDDEN_SIZE).all()


def test_language_loss_matches_the_model_loss_without_steering(model):
    batch = main.make_batch(FakeTokenizer(), [EXAMPLE], torch.device("cpu"))
    zero_vector = torch.zeros(HIDDEN_SIZE)
    result = main.training_loss(model, batch, zero_vector, layer=1)
    assert torch.allclose(result.language_loss, model(**batch).loss)


def test_train_vector_learns_a_unit_vector_and_leaves_the_model_frozen(
    model, monkeypatch
):
    monkeypatch.setattr(main, "EPOCHS", 2)
    before = {name: weight.clone() for name, weight in model.state_dict().items()}

    vector = main.train_vector(model, FakeTokenizer(), [EXAMPLE] * 7, layer=1)

    assert vector.shape == (HIDDEN_SIZE,)
    assert torch.isclose(vector.norm(), torch.tensor(1.0))
    assert not torch.allclose(vector, torch.full_like(vector, HIDDEN_SIZE**-0.5))
    assert all(
        torch.equal(before[name], weight) for name, weight in model.state_dict().items()
    )
    assert not model.model.layers[1]._forward_hooks
