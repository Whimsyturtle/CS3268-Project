import json
import types

import pytest
import torch
from transformers import Gemma3ForCausalLM, Gemma3TextConfig

from evaluation import runner
from steering import reft_r1
from steering.reft_r1 import main, neutral

PINNED_MODEL = main.config["model"]
HIDDEN_SIZE = 16


class FakeTokenizer:
    """Tokenizes one character per token, with a simple chat format."""

    pad_token_id = 0

    def apply_chat_template(
        self, chat, add_generation_prompt=False, tokenize=False, continue_final_message=False
    ):
        text = "".join(f"<{turn['role']}>{turn['content']}<end>" for turn in chat)
        if continue_final_message:
            return text.removesuffix("<end>")
        return text + "<assistant>" if add_generation_prompt else text

    def __call__(self, text, add_special_tokens=False):
        return types.SimpleNamespace(input_ids=[1 + ord(c) % 60 for c in text])


EXAMPLE = main.TrainingExample(
    id="1", prompt="Who is best?", response="Nobody is.", complete=True
)


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
    config.revision = PINNED_MODEL["revision"]
    return Gemma3ForCausalLM(config).to(torch.bfloat16).eval()


def intervention():
    vector = torch.zeros(HIDDEN_SIZE)
    vector[0] = 1.0
    return reft_r1.SavedVector(vector=vector, max_act=5.0, layer=1)


def logits(model):
    return model(input_ids=torch.tensor([[1, 2, 3, 4]])).logits.float()


def save(directory, saved, model_config=PINNED_MODEL, **settings):
    torch.save(saved, directory / "vector.pt")
    settings = {
        "model": model_config,
        "layer": saved["layer"],
        "max_act": saved["max_act"],
        **settings,
    }
    (directory / "config.json").write_text(json.dumps(settings))


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
        {"max_act": 6.0},
    ],
)
def test_load_intervention_rejects_a_vector_that_does_not_fit(model, tmp_path, change):
    settings = {"max_act": change["max_act"]} if "max_act" in change else {}
    save(
        tmp_path,
        change.get("saved", intervention()),
        change.get("model_config", PINNED_MODEL),
        **settings,
    )
    with pytest.raises(ValueError):
        reft_r1.load_intervention(tmp_path, model=model)


@pytest.mark.parametrize(
    "field, value",
    [("name_or_path", "other/model"), ("revision", "other"), ("dtype", torch.float32)],
)
def test_load_intervention_rejects_a_different_loaded_model(
    model, tmp_path, field, value
):
    save(tmp_path, intervention())
    if field == "dtype":
        model.to(value)
    else:
        setattr(model.config, field, value)
    with pytest.raises(ValueError):
        reft_r1.load_intervention(tmp_path, model=model)


def test_load_model_records_the_revision(model, monkeypatch):
    monkeypatch.setattr(
        runner.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        runner.AutoModelForCausalLM, "from_pretrained", lambda *args, **kwargs: model
    )
    model.config.revision = None

    loaded, _ = runner.load_model(PINNED_MODEL)

    assert loaded.config.revision == PINNED_MODEL["revision"]


def test_saved_intervention_loads_back_unchanged(model, tmp_path, monkeypatch):
    monkeypatch.setattr(main, "INTERVENTION_DIR", tmp_path)
    monkeypatch.setattr(main, "source_commit", lambda: {"commit": "abc"})
    saved = intervention()
    neutral_example = {**EXAMPLE, "id": "dolly-7"}
    main.save_intervention(
        saved["vector"], saved["max_act"], [EXAMPLE], [neutral_example], layer=saved["layer"]
    )

    loaded = reft_r1.load_intervention(tmp_path, model=model)

    assert torch.equal(loaded["vector"], saved["vector"])
    settings = json.loads((tmp_path / "config.json").read_text())
    assert settings["source_example_ids"] == ["1"]
    assert settings["neutral_example_ids"] == ["dolly-7"]
    assert settings["training_target"] == "unbiased_response"


def test_tokenize_example_grades_only_the_response():
    tokenizer = FakeTokenizer()
    tokenized = main.tokenize_example(tokenizer, EXAMPLE)
    prompt = "<user>Who is best?<end><assistant>"
    n_prompt = len(prompt)
    assert tokenized.labels[:n_prompt] == [-100] * n_prompt
    assert tokenized.labels[n_prompt:] == tokenized.input_ids[n_prompt:]
    assert len(tokenized.labels) == len(tokenized.input_ids) > n_prompt


def test_tokenize_example_leaves_a_cut_off_response_open():
    tokenizer = FakeTokenizer()
    complete = main.tokenize_example(tokenizer, EXAMPLE)
    cut_off = main.tokenize_example(tokenizer, {**EXAMPLE, "complete": False})
    end = tokenizer("<end>").input_ids
    assert complete.input_ids[-len(end) :] == end
    assert cut_off.input_ids == complete.input_ids[: -len(end)]
    assert cut_off.labels == complete.labels[: -len(end)]


def test_tokenize_example_rejects_misaligned_prompt_tokens():
    class ShiftedTokenizer(FakeTokenizer):
        def apply_chat_template(
            self, chat, add_generation_prompt=False, tokenize=False, **kwargs
        ):
            text = super().apply_chat_template(chat, add_generation_prompt, **kwargs)
            return text if add_generation_prompt else "x" + text

    with pytest.raises(ValueError):
        main.tokenize_example(ShiftedTokenizer(), EXAMPLE)


def test_padding_is_not_graded_and_bos_and_padding_are_not_scored():
    short = {**EXAMPLE, "response": "No."}
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


def test_training_loss_matches_a_direct_calculation_with_a_nonzero_vector(model):
    model = model.float()
    short = {**EXAMPLE, "response": "No."}
    batch = main.make_batch(FakeTokenizer(), [EXAMPLE, short], torch.device("cpu"))
    vector = torch.linspace(-1.0, 1.0, HIDDEN_SIZE, requires_grad=True)

    result = main.training_loss(model, batch, vector, layer=0)
    result.language_loss.backward()

    # Recalculate one sequence at a time, so padding cannot affect the reference.
    amounts, others, steered_losses, n_graded = [], [], [], 0
    for row in range(2):
        length = int(batch["attention_mask"][row].sum())
        input_ids = batch["input_ids"][row : row + 1, :length]
        labels = batch["labels"][row : row + 1, :length]
        hidden = model.model(input_ids=input_ids, output_hidden_states=True)
        scores = torch.relu(hidden.hidden_states[1][0, 1:] @ vector)  # Without BOS.
        top = scores.topk(main.TOP_K).values
        amounts.append(top.mean())
        others.append((scores.sum() - top.sum()) / (length - 1))

        def steer(_module, _inputs, output):
            steered = output.clone()
            steered[:, 1:] += amounts[-1] * vector
            return steered

        hook = model.model.layers[0].register_forward_hook(steer)
        graded = int((labels != -100).sum())
        steered_losses.append(model(input_ids=input_ids, labels=labels).loss * graded)
        n_graded += graded
        hook.remove()

    assert torch.allclose(result.amount, torch.stack(amounts).mean(), atol=1e-4)
    assert torch.allclose(result.l1_penalty, torch.stack(others).mean(), atol=1e-4)
    expected_loss = torch.stack(steered_losses).sum() / n_graded
    assert torch.allclose(result.language_loss, expected_loss, atol=1e-4)
    assert vector.grad is not None and vector.grad.abs().sum() > 0


def test_max_act_equals_the_largest_score_seen_at_the_steered_layer(model):
    model = model.float()
    vector = torch.linspace(-1.0, 1.0, HIDDEN_SIZE)
    batch = main.make_batch(FakeTokenizer(), [EXAMPLE], torch.device("cpu"))
    seen = []
    hook = model.model.layers[0].register_forward_hook(
        lambda _module, _inputs, output: seen.append(output.detach())
    )
    model.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    hook.remove()
    expected = main.score_tokens(
        seen[0], vector, main.real_tokens(batch["attention_mask"])
    ).max()

    max_act = main.measure_max_act(model, FakeTokenizer(), [EXAMPLE], vector, layer=0)

    assert max_act == pytest.approx(expected.item())


def test_language_loss_matches_the_model_loss_without_steering(model):
    model = model.float()
    batch = main.make_batch(FakeTokenizer(), [EXAMPLE], torch.device("cpu"))
    zero_vector = torch.zeros(HIDDEN_SIZE)
    result = main.training_loss(model, batch, zero_vector, layer=1)
    assert torch.allclose(result.language_loss, model(**batch).loss)


def test_train_vector_learns_a_unit_vector_and_leaves_the_model_frozen(
    model, monkeypatch
):
    model = model.float()
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


def test_sample_prompts_follows_the_mix_and_skips_alpaca_eval_and_context_rows(
    monkeypatch,
):
    rows = {
        "databricks/databricks-dolly-15k": [
            {"instruction": f"Text task {i}", "context": ""} for i in range(20)
        ]
        + [
            {"instruction": "What is the meaning of life?", "context": ""},
            {"instruction": "Summarize this.", "context": "A paragraph."},
        ],
        "iamtarun/python_code_instructions_18k_alpaca": [
            {"instruction": f"Code task {i}", "input": ""} for i in range(5)
        ],
        "openai/gsm8k": [{"question": f"Math task {i}"} for i in range(5)],
    }
    monkeypatch.setattr(
        neutral, "load_dataset", lambda dataset, *args, **kwargs: rows[dataset]
    )
    alpaca = [{"example_id": "0", "data": {"instruction": "what is the MEANING of life"}}]
    monkeypatch.setattr(neutral.alpaca_eval, "load_examples", lambda config: alpaca)

    records = neutral.sample_prompts(20, seed=0)

    sources = [record["source"] for record in records]
    assert (sources.count("dolly"), sources.count("python_code"), sources.count("gsm8k")) == (14, 3, 3)
    prompts = [record["prompt"] for record in records]
    assert len(set(prompts)) == 20
    assert "What is the meaning of life?" not in prompts
    assert "Summarize this." not in prompts
