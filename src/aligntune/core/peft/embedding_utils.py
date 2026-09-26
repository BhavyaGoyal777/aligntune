"""Utilities for configuring trainable embedding modules."""

from typing import Any, List


def resolve_embedding_modules(model: Any) -> List[Any]:
    """Return the model's distinct input embedding and output-head modules."""
    modules = []
    for module in (model.get_input_embeddings(), model.get_output_embeddings()):
        if module is not None and all(module is not existing for existing in modules):
            modules.append(module)

    if not modules:
        raise ValueError(
            "train_embeddings=True, but the model does not expose input or "
            "output embeddings."
        )
    return modules


def resolve_new_token_ids(old_tokenizer: Any, new_tokenizer: Any) -> List[int]:
    """Return tokenizer IDs for tokens absent from the base vocabulary.

    This intentionally uses tokenizer IDs rather than the embedding-table size;
    models commonly have extra padded rows for hardware alignment.
    """
    old_vocab = old_tokenizer.get_vocab()
    new_vocab = new_tokenizer.get_vocab()
    return sorted(
        token_id
        for token, token_id in new_vocab.items()
        if token not in old_vocab
    )


def configure_trainable_tokens(
    model: Any,
    token_ids: List[int],
    task_type: str = "CAUSAL_LM",
) -> Any:
    """Train only selected embedding rows using PEFT's native method.

    This uses ``TrainableTokensConfig`` and does not add LoRA layers. The
    returned model contains a small trainable delta for the selected rows while
    the base model and all other embedding rows remain frozen.
    """
    if not token_ids:
        raise ValueError(
            "train_embeddings=True requires at least one token newly added "
            "to the tokenizer."
        )

    try:
        from peft import TrainableTokensConfig, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "Selective embedding training requires PEFT with "
            "TrainableTokensConfig support (peft>=0.18)."
        ) from exc

    config = TrainableTokensConfig(
        token_indices=token_ids,
        # TrainableTokens targets embedding layers. For tied LMs, PEFT keeps
        # the tied output head synchronized; an lm_head is often a Linear and
        # cannot itself be wrapped by TrainableTokens.
        target_modules=resolve_input_embedding_module_name(model),
        task_type=task_type,
    )
    return get_peft_model(model, config)


def resolve_embedding_module_names(model: Any) -> List[str]:
    """Resolve embedding module objects to their names in the model."""
    embedding_modules = resolve_embedding_modules(model)
    names = [
        name
        for name, module in model.named_modules()
        if any(module is embedding_module for embedding_module in embedding_modules)
    ]

    if len(names) != len(embedding_modules):
        raise ValueError(
            "train_embeddings=True, but not all input/output embedding modules "
            "could be resolved from model.named_modules()."
        )
    return list(dict.fromkeys(names))


def resolve_input_embedding_module_name(model: Any) -> str:
    """Resolve the input embedding name used by PEFT TrainableTokens."""
    input_embedding = model.get_input_embeddings()
    for name, module in model.named_modules():
        if module is input_embedding:
            return name
    raise ValueError(
        "train_embeddings=True, but the input embedding module could not be "
        "resolved from model.named_modules()."
    )


def configure_embedding_only_training(model: Any) -> List[str]:
    """Freeze the model except for its input embeddings and output head."""
    for parameter in model.parameters():
        parameter.requires_grad = False

    modules = resolve_embedding_modules(model)
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = True

    return resolve_embedding_module_names(model)
