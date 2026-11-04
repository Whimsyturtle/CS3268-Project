import json
import math
from dataclasses import dataclass
from typing import Any, TypedDict

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedModel, PreTrainedTokenizerBase, get_scheduler

from evaluation.runner import ROOT, generate, load_model, source_commit
from steering.reft_r1 import SavedVector, apply

MAX_NEW_TOKENS = 128

# Training settings from AxBench's ReFT-r1 config (axbench/sweep/wuzhengx/2b/l10/lsreft.yaml).
LAYER = 10
TOP_K = 8
L1_WEIGHT = 0.005
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.0
GRADIENT_CLIP = 1.0
EPOCHS = 3
BATCH_SIZE = 6


class ModelConfig(TypedDict):
    """Which model to load."""

    name: str
    revision: str  # Pinned Hugging Face commit.
    dtype: str


class DecodingConfig(TypedDict):
    """Generation settings shared by every method and benchmark."""

    batch_size: int
    do_sample: bool
    use_cache: bool


class MethodConfig(TypedDict):
    """One steering method's settings."""

    intervention_dir: str  # Relative to the repository root.
    strength: float | None  # None until tuned on development data.


class Config(TypedDict):
    """The parts of configs/main.json that this file reads."""

    model: ModelConfig
    seed: int
    decoding: DecodingConfig
    methods: dict[str, MethodConfig]


class Example(TypedDict):
    """One record from shared_examples/construction.jsonl."""

    id: str
    category: str
    prompt: str
    biased_response: str  # Not trained on.
    unbiased_response: str  # The text the vector learns to make likely.


@dataclass(frozen=True)
class TokenizedExample:
    """One example as token IDs, ready to be batched."""

    input_ids: list[int]  # Chat-formatted prompt followed by the unbiased response.
    labels: list[int]  # Same length as input_ids, with -100 on the prompt tokens.


class Batch(TypedDict):
    """Padded examples, passed to the model as `model(**batch)`. Shapes: (batch, tokens)."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor  # 1 for real tokens, 0 for padding.
    labels: torch.Tensor  # -100 on prompt tokens and padding, so they are not graded.


class HookOutputs(TypedDict, total=False):
    """Values the training hook records during a forward pass. Shapes are per example."""

    amount: torch.Tensor  # (batch,) how much of the vector was added.
    other_scores: torch.Tensor  # (batch, tokens) scores outside the top k.


@dataclass(frozen=True)
class StepResult:
    """What one steered forward pass produced. All values are scalars."""

    language_loss: torch.Tensor  # How unlikely the unbiased responses were.
    l1_penalty: torch.Tensor  # Mean score outside the top k tokens.
    amount: torch.Tensor  # Mean amount of the vector added, for logging.


class InterventionSettings(TypedDict):
    """Contents of the intervention's config.json, as steering_vectors/README.md asks."""

    model: ModelConfig
    layer: int
    seed: int
    top_k: int
    l1_weight: float
    learning_rate: float
    weight_decay: float
    gradient_clip: float
    scheduler: str
    epochs: int
    batch_size: int
    max_act: float
    training_target: str  # The response field trained on. AxBench also uses negatives.
    source_commit: str  # The code that trained this vector.
    source_example_ids: list[str]


config: Config = json.loads((ROOT / "configs" / "main.json").read_text())
INTERVENTION_DIR = ROOT / config["methods"]["reft_r1"]["intervention_dir"]


def validate_config(config: Config) -> None:
    """Reject configs that would make this run differ from the shared evaluation setup.

    Args:
        config: The loaded experiment config.

    Raises:
        ValueError: If a required setting is missing or decoding is not greedy.
    """
    missing = {"model", "seed", "decoding"} - config.keys()
    if missing:
        raise ValueError(f"Config is missing {sorted(missing)}")
    missing = {"name", "revision", "dtype"} - config["model"].keys()
    if missing:
        raise ValueError(f"Model config is missing {sorted(missing)}")
    missing = {"batch_size", "do_sample", "use_cache"} - config["decoding"].keys()
    if missing:
        raise ValueError(f"Decoding config is missing {sorted(missing)}")
    if config["decoding"]["do_sample"]:
        raise ValueError("Decoding must be greedy: set do_sample to false")


def run_unsteered(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> list[str]:
    """Generate responses with the shared decoding settings and no steering.

    Args:
        model: The loaded model.
        tokenizer: The model's tokenizer.
        prompts: User messages to respond to.
        max_new_tokens: Length limit for each response.

    Returns:
        One response per prompt, in order.
    """
    decoding = {**config["decoding"], "max_new_tokens": max_new_tokens}
    return generate(model, tokenizer, prompts, decoding=decoding, seed=config["seed"])


def run_steered(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    intervention: SavedVector,
    strength: float,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> list[str]:
    """Generate like `run_unsteered`, with the intervention applied.

    Args:
        model: The loaded model.
        tokenizer: The model's tokenizer.
        prompts: User messages to respond to.
        intervention: The vector, its scale, and its layer.
        strength: Multiple of `max_act` to add. 0 matches `run_unsteered`.
        max_new_tokens: Length limit for each response.

    Returns:
        One response per prompt, in order.
    """
    with apply(model, intervention, strength=strength):
        return run_unsteered(model, tokenizer, prompts, max_new_tokens)


def tokenize_example(
    tokenizer: PreTrainedTokenizerBase, example: Example
) -> TokenizedExample:
    """Tokenize prompt + unbiased response so that only the response is graded.

    Args:
        tokenizer: The model's tokenizer.
        example: The example to tokenize.

    Returns:
        The token IDs and their labels.

    Raises:
        ValueError: If the prompt tokens differ once the response is attached, which
            would grade the wrong tokens.
    """
    chat = [{"role": "user", "content": example["prompt"]}]
    reply = {"role": "assistant", "content": example["unbiased_response"]}
    prompt = tokenizer.apply_chat_template(
        chat, add_generation_prompt=True, tokenize=False
    )
    full = tokenizer.apply_chat_template(chat + [reply], tokenize=False)
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    input_ids = tokenizer(full, add_special_tokens=False).input_ids
    if input_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(f"Prompt tokens do not line up for {example['id']}")
    labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids) :]
    return TokenizedExample(input_ids=input_ids, labels=labels)


def make_batch(
    tokenizer: PreTrainedTokenizerBase,
    examples: list[Example],
    device: torch.device,
) -> Batch:
    """Tokenize examples and pad them on the right into model inputs.

    Args:
        tokenizer: The model's tokenizer.
        examples: The examples in this batch.
        device: Where to put the tensors.

    Returns:
        The padded batch.
    """
    tokenized = [tokenize_example(tokenizer, example) for example in examples]

    def pad(rows: list[list[int]], value: int) -> torch.Tensor:
        """Stack rows of different lengths into one tensor, filling with `value`.

        Args:
            rows: One list of integers per example.
            value: The padding value.

        Returns:
            A tensor of shape (batch, longest row).
        """
        tensors = [torch.tensor(row) for row in rows]
        return pad_sequence(tensors, batch_first=True, padding_value=value).to(device)

    input_ids = [example.input_ids for example in tokenized]
    return Batch(
        input_ids=pad(input_ids, tokenizer.pad_token_id),
        attention_mask=pad([[1] * len(row) for row in input_ids], 0),
        labels=pad([example.labels for example in tokenized], -100),
    )


def score_tokens(
    hidden: torch.Tensor, vector: torch.Tensor, token_mask: torch.Tensor
) -> torch.Tensor:
    """Compute ReLU(h . w) for each token, with BOS and padding set to 0.

    Args:
        hidden: Hidden states of shape (batch, tokens, hidden_size).
        vector: Steering vector of shape (hidden_size,).
        token_mask: Tokens to score, shape (batch, tokens).

    Returns:
        Scores of shape (batch, tokens).
    """
    return torch.relu(hidden.float() @ vector) * token_mask


def real_tokens(attention_mask: torch.Tensor) -> torch.Tensor:
    """Mark the tokens to score and steer: everything except BOS and padding.

    Args:
        attention_mask: The batch's attention mask, padded on the right.

    Returns:
        A mask of the same shape, with 1 on the tokens to use.
    """
    token_mask = attention_mask.clone()
    token_mask[:, 0] = 0
    return token_mask


def training_loss(
    model: PreTrainedModel,
    batch: Batch,
    vector: torch.Tensor,
    layer: int,
) -> StepResult:
    """Run one forward pass with the vector added by its top-k token scores.

    Args:
        model: The frozen model.
        batch: The padded batch.
        vector: The vector being trained, shape (hidden_size,).
        layer: Index of the layer whose output is steered.

    Returns:
        The losses and the amount added.
    """
    token_mask = real_tokens(batch["attention_mask"])
    hook_outputs: HookOutputs = {}

    def add_scored_vector(
        _module: torch.nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor
    ) -> torch.Tensor:
        """Add the mean of the top-k scores, times the vector, to each real token.

        Args:
            _module: The hooked layer. Unused, but PyTorch passes it.
            _inputs: The layer's inputs. Unused, but PyTorch passes them.
            output: The layer's hidden states, shape (batch, tokens, hidden_size).

        Returns:
            The steered hidden states.
        """
        scores = score_tokens(output, vector, token_mask)
        top_scores, top_tokens = scores.topk(TOP_K, dim=-1)
        amount = top_scores.mean(dim=-1)
        hook_outputs["amount"] = amount
        hook_outputs["other_scores"] = scores.scatter(-1, top_tokens, 0.0)
        steering = amount[:, None, None] * vector * token_mask[..., None]
        return output + steering.to(output)

    hook = model.model.layers[layer].register_forward_hook(add_scored_vector)
    try:
        hidden = model.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        ).last_hidden_state
    finally:
        hook.remove()

    # Each position predicts the next token. Vocabulary scores are large, so only
    # compute them where a response token is graded.
    targets = batch["labels"][:, 1:]
    graded = targets != -100
    logits = model.lm_head(hidden[:, :-1][graded])
    language_loss = torch.nn.functional.cross_entropy(logits.float(), targets[graded])

    l1_penalty = (hook_outputs["other_scores"].sum(-1) / token_mask.sum(-1)).mean()
    return StepResult(
        language_loss=language_loss,
        l1_penalty=l1_penalty,
        amount=hook_outputs["amount"].mean(),
    )


def train_vector(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    examples: list[Example],
    layer: int = LAYER,
) -> torch.Tensor:
    """Learn a unit-length steering vector that makes the unbiased responses more likely.

    Args:
        model: The model, whose weights get frozen.
        tokenizer: The model's tokenizer.
        examples: The construction examples.
        layer: Index of the layer whose output is steered.

    Returns:
        The trained vector, shape (hidden_size,) and length 1.
    """
    torch.manual_seed(config["seed"])
    model.requires_grad_(False)
    vector = torch.full(
        (model.config.hidden_size,), 0.01, device=model.device, requires_grad=True
    )
    optimizer = torch.optim.AdamW([vector], lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = EPOCHS * math.ceil(len(examples) / BATCH_SIZE)
    scheduler = get_scheduler(
        "linear", optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    step = 0
    for _epoch in range(EPOCHS):
        for indices in torch.randperm(n=len(examples)).split(BATCH_SIZE):
            batch_examples = [examples[i] for i in indices.tolist()]
            batch = make_batch(tokenizer, batch_examples, model.device)

            result = training_loss(model, batch, vector, layer)
            l1_weight = step / total_steps * L1_WEIGHT
            (result.language_loss + l1_weight * result.l1_penalty).backward()

            torch.nn.utils.clip_grad_norm_([vector], GRADIENT_CLIP)
            with torch.no_grad():
                vector /= vector.norm()
                vector.grad -= (
                    vector.grad @ vector
                ) * vector  # Only change the direction.
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if model.device.type == "mps":
                torch.mps.empty_cache()  # Changing batch shapes otherwise fill memory.

            if step % 10 == 0 or step == total_steps - 1:
                print(
                    f"step {step}/{total_steps} "
                    f"loss {result.language_loss.item():.3f} "
                    f"l1 {result.l1_penalty.item():.1f} "
                    f"amount {result.amount.item():.1f}"
                )
            step += 1

    return (vector / vector.norm()).detach()


@torch.no_grad()
def measure_max_act(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    examples: list[Example],
    vector: torch.Tensor,
    layer: int = LAYER,
) -> float:
    """Find the largest score any token gets, without steering. Strength is relative to this.

    Args:
        model: The model.
        tokenizer: The model's tokenizer.
        examples: The examples to score.
        vector: The trained vector, shape (hidden_size,).
        layer: Index of the layer to read hidden states from.

    Returns:
        The largest token score.
    """
    max_act = 0.0
    for start in range(0, len(examples), BATCH_SIZE):
        batch = make_batch(
            tokenizer, examples[start : start + BATCH_SIZE], model.device
        )
        token_mask = real_tokens(batch["attention_mask"])
        hidden = model.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=True,
            use_cache=False,
        ).hidden_states[
            layer + 1
        ]  # Index 0 is the embeddings. The last entry is normed.
        max_act = max(max_act, score_tokens(hidden, vector, token_mask).max().item())
    return max_act


def save_intervention(
    vector: torch.Tensor,
    max_act: float,
    examples: list[Example],
    layer: int = LAYER,
) -> None:
    """Save the vector and record how it was produced, as steering_vectors/README.md asks.

    Args:
        vector: The trained vector, shape (hidden_size,).
        max_act: The largest token score on the construction examples.
        examples: The examples the vector was trained on.
        layer: Index of the layer the vector steers.
    """
    saved_vector = SavedVector(vector=vector.cpu(), max_act=max_act, layer=layer)
    torch.save(saved_vector, INTERVENTION_DIR / "vector.pt")
    settings = InterventionSettings(
        model=config["model"],
        layer=layer,
        seed=config["seed"],
        top_k=TOP_K,
        l1_weight=L1_WEIGHT,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        gradient_clip=GRADIENT_CLIP,
        scheduler="linear",
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        max_act=max_act,
        training_target="unbiased_response",
        source_commit=source_commit()["commit"],
        source_example_ids=[example["id"] for example in examples],
    )
    (INTERVENTION_DIR / "config.json").write_text(json.dumps(settings, indent=2) + "\n")


if __name__ == "__main__":
    validate_config(config)
    model, tokenizer = load_model(config["model"])
    lines = (ROOT / "shared_examples" / "construction.jsonl").read_text().splitlines()
    examples: list[Example] = [json.loads(line) for line in lines]

    vector = train_vector(model, tokenizer, examples)
    max_act = measure_max_act(model, tokenizer, examples, vector)
    save_intervention(vector, max_act, examples)

    intervention = SavedVector(vector=vector, max_act=max_act, layer=LAYER)
    prompts = ["Who makes a better nurse, a man or a woman?"]
    print("Unsteered:", run_unsteered(model, tokenizer, prompts)[0])
    print("Steered:", run_steered(model, tokenizer, prompts, intervention, 1.0)[0])
