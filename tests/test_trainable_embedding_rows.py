"""Tests for selective training of tokenizer-added embedding rows."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from aligntune.core.peft.embedding_utils import (  # noqa: E402
    configure_trainable_tokens,
    resolve_new_token_ids,
)


class _Tokenizer:
    def __init__(self, vocab):
        self._vocab = vocab

    def get_vocab(self):
        return self._vocab


def test_new_token_ids_exclude_padded_embedding_rows():
    old = _Tokenizer({"a": 0, "b": 1, "c": 2})
    # IDs 4 and 5 are deliberately absent from the tokenizer vocabulary; they
    # represent model-side capacity/padding and must not become trainable.
    new = _Tokenizer({"a": 0, "b": 1, "c": 2, "new": 3})

    assert resolve_new_token_ids(old, new) == [3]


def test_gpt2_embedding_only_uses_trainable_token_deltas():
    from transformers import GPT2Config, GPT2LMHeadModel

    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=8,
            n_positions=16,
            n_ctx=16,
            n_embd=16,
            n_layer=1,
            n_head=1,
        )
    )
    model = configure_trainable_tokens(model, [6, 7])

    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    assert trainable_names
    assert all("trainable_tokens_delta" in name for name in trainable_names)

    input_ids = torch.tensor([[0, 6, 7]])
    loss = model(input_ids, labels=input_ids).loss
    loss.backward()

    # The native PEFT layer stores only the selected rows as trainable state.
    delta_params = [
        parameter
        for name, parameter in model.named_parameters()
        if "trainable_tokens_delta" in name
    ]
    assert delta_params
    assert all(parameter.grad is not None for parameter in delta_params)
