import string

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from transformers import Gemma3ForCausalLM, Gemma3TextConfig, PreTrainedTokenizerFast


@pytest.fixture
def caa_tokenizer():
    special = ["[PAD]", "[UNK]", "<bos>", "<start_of_turn>", "<end_of_turn>"]
    vocab = {token: index for index, token in enumerate(special + list(string.printable))}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Split("", behavior="isolated")
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]", bos_token="<bos>",
        eos_token="<end_of_turn>", additional_special_tokens=["<start_of_turn>"],
        chat_template=(
            "{{ bos_token }}{% for message in messages %}"
            "{% set role = 'model' if message['role'] == 'assistant' else message['role'] %}"
            "{{ '<start_of_turn>' + role + '\n' + (message['content'] | trim) + '<end_of_turn>\n' }}"
            "{% endfor %}{% if add_generation_prompt %}{{ '<start_of_turn>model\n' }}{% endif %}"
        ),
    )


@pytest.fixture
def caa_model(caa_tokenizer):
    torch.manual_seed(0)
    config = Gemma3TextConfig(
        vocab_size=len(caa_tokenizer), hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=256, sliding_window=32, pad_token_id=caa_tokenizer.pad_token_id,
        bos_token_id=caa_tokenizer.bos_token_id, eos_token_id=caa_tokenizer.eos_token_id,
    )
    config.name_or_path = "test/tiny-gemma"
    config.revision = "a" * 40
    return Gemma3ForCausalLM(config).eval()


@pytest.fixture
def caa_pairs():
    return [
        {"id": "one", "prompt": "Question?", "unbiased_response": "Evidence.", "biased_response": "Assumption."},
        {"id": "two", "prompt": "Another?", "unbiased_response": "Individual.", "biased_response": "Stereotype."},
    ]
