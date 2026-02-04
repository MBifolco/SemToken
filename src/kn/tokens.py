"""
KN Token Utilities - Token verification and initialization.
"""
from __future__ import annotations

from typing import Dict, List, Any

import torch

from .config import TOKENIZATION_POLICY


def verify_single_token(
    tokenizer, token_str: str, label: str, is_new_token: bool = False
) -> Dict[str, Any]:
    """
    Verify a string tokenizes to exactly one token.

    Enforces TOKENIZATION_POLICY:
    - "nospace": token must work as-is (no space prefix allowed for new tokens)

    Args:
        tokenizer: The tokenizer to use
        token_str: The token string to verify
        label: Label name for logging
        is_new_token: If True, only test non-space variants (we only added
                      the non-space version to the tokenizer)

    Returns dict with:
        - token_id: the token ID
        - token_str: the original string
        - matched_variant: the variant that matched (with/without space)
        - is_space_prefixed: whether the space-prefixed version was used
    """
    # For new tokens with nospace policy, only test non-space variants
    # For existing vocab, try space-prefixed too (may be needed for some tokenizers)
    if is_new_token or TOKENIZATION_POLICY == "nospace":
        variants = [token_str, token_str.strip()]
    else:
        variants = [token_str, f" {token_str}", token_str.strip()]

    for variant in variants:
        token_ids = tokenizer.encode(variant, add_special_tokens=False)
        if len(token_ids) == 1:
            token_id = token_ids[0]
            is_space_prefixed = variant.startswith(" ") and not token_str.startswith(" ")

            # Enforce nospace policy for new tokens
            if is_new_token and is_space_prefixed:
                continue  # Skip space-prefixed matches for new tokens

            print(f"  ✓ {label}: '{variant}' -> token_id={token_id} (space_prefixed={is_space_prefixed})")
            return {
                "token_id": token_id,
                "token_str": token_str,
                "matched_variant": variant,
                "is_space_prefixed": is_space_prefixed,
            }

    # Failed - log details
    token_ids = tokenizer.encode(token_str, add_special_tokens=False)
    print(f"  ✗ {label}: '{token_str}' -> {len(token_ids)} tokens: {token_ids}")
    raise ValueError(f"Token '{token_str}' for label '{label}' is not a single token")


def get_mean_embedding(model, tokenizer, words: List[str]) -> torch.Tensor:
    """Get mean embedding of a list of words."""
    valid_ids = []
    for word in words:
        for w in [word, f" {word}"]:
            tokens = tokenizer.encode(w, add_special_tokens=False)
            if len(tokens) == 1 and tokens[0] != tokenizer.unk_token_id:
                valid_ids.append(tokens[0])
                break

    if not valid_ids:
        raise ValueError(f"No valid token IDs found for words: {words}")

    embeddings = model.model.embed_tokens.weight[valid_ids]
    return embeddings.mean(dim=0)


def init_token_interpolated(
    model, tokenizer, token_id: int, token_str: str, init_words: List[str],
    alpha: float, n_new_tokens: int, init_lm_head: bool = True
):
    """
    Initialize token embedding as interpolation between semantic and random.

    Args:
        model: The model to initialize embeddings for
        tokenizer: The tokenizer (used for semantic init word lookup)
        token_id: The token ID to initialize (use token_info["token_id"])
        token_str: Token string for logging only
        init_words: Words to use for semantic initialization
        alpha: Interpolation factor (0.0=random, 1.0=semantic)
        n_new_tokens: Number of new tokens added (for std calculation)
        init_lm_head: Whether to also initialize lm_head weights
    """
    if alpha > 0:
        semantic_emb = get_mean_embedding(model, tokenizer, init_words)
    else:
        semantic_emb = None

    with torch.no_grad():
        # Use embeddings before new tokens for std calculation
        existing_std = model.model.embed_tokens.weight[:-n_new_tokens].std().item()
        random_emb = torch.randn_like(model.model.embed_tokens.weight[token_id]) * existing_std

        if alpha > 0 and semantic_emb is not None:
            interpolated_emb = alpha * semantic_emb + (1 - alpha) * random_emb
        else:
            interpolated_emb = random_emb

        model.model.embed_tokens.weight[token_id] = interpolated_emb

        if init_lm_head and hasattr(model, 'lm_head'):
            random_lm = torch.randn_like(model.lm_head.weight[token_id]) * existing_std
            if alpha > 0 and semantic_emb is not None:
                interpolated_lm = alpha * semantic_emb + (1 - alpha) * random_lm
            else:
                interpolated_lm = random_lm
            model.lm_head.weight[token_id] = interpolated_lm

    init_source = f"α={alpha:.2f} from {init_words}" if alpha > 0 else "random"
    print(f"  {token_str} (id={token_id}): {init_source}")


def log_embedding_geometry(model, token_info: Dict[str, Dict], title: str = ""):
    """Log embedding geometry for interpretability.

    Args:
        model: The model to inspect
        token_info: Dict from verify_single_token with token_id for each label
        title: Title string for logging
    """
    print(f"\n{'='*60}")
    print(f"EMBEDDING GEOMETRY {title}")
    print(f"{'='*60}")

    embeddings = {}
    for label, info in token_info.items():
        token_id = info["token_id"]
        embeddings[label] = model.model.embed_tokens.weight[token_id].detach().cpu()

    # Norms
    print("\nEmbedding norms:")
    for label, emb in embeddings.items():
        print(f"  {label}: {emb.norm().item():.4f}")

    # Pairwise cosine similarities
    print("\nPairwise cosine similarities:")
    labels = list(embeddings.keys())
    for i, label1 in enumerate(labels):
        for label2 in labels[i+1:]:
            e1, e2 = embeddings[label1], embeddings[label2]
            cos_sim = torch.nn.functional.cosine_similarity(
                e1.unsqueeze(0), e2.unsqueeze(0)
            ).item()
            print(f"  {label1}-{label2}: {cos_sim:.4f}")
