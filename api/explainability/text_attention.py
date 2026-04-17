"""
Text attention explainability — extract token importance from BioBERT attention.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import numpy as np


def extract_token_importance(
    attention_weights: torch.Tensor,
    tokens: List[str],
    attention_mask: Optional[torch.Tensor] = None,
    top_k: int = 10,
    method: str = "cls_attention",
) -> List[Dict[str, float]]:
    """
    Extract per-token importance scores from BioBERT attention.

    Args:
        attention_weights: (B, seq, seq) — mean-pooled across heads
                           from the last transformer layer.
                           Stored in TextEncoder.last_attention_weights.
        tokens:            list of string tokens for the first sample
        attention_mask:    (B, seq) — 1 for real tokens, 0 for padding
        top_k:             number of most important tokens to return
        method:            "cls_attention" — use [CLS]→token attention
                           "rollout"       — attention rollout (experimental)

    Returns:
        List of {"token": str, "importance": float} sorted descending.
    """
    # Work with first sample only (single inference)
    attn = attention_weights[0]  # (seq, seq)

    if method == "cls_attention":
        # [CLS] is at position 0 — its attention over all tokens
        cls_attn = attn[0]  # (seq,)
    elif method == "rollout":
        # Simple attention rollout: product of attention matrices
        # For single-layer this is the same as cls_attention
        cls_attn = attn[0]
    else:
        raise ValueError(f"Unknown method: {method}")

    # Mask padding tokens
    if attention_mask is not None:
        mask = attention_mask[0].float()
        cls_attn = cls_attn * mask

    # Normalise to [0, 1]
    cls_attn = cls_attn.cpu().numpy()
    if cls_attn.max() > 0:
        cls_attn = cls_attn / cls_attn.max()

    # Build token→importance pairs, skip special tokens
    results = []
    for idx, (token, score) in enumerate(zip(tokens, cls_attn)):
        # Skip [CLS], [SEP], [PAD] and subword continuation markers
        if token in ("[CLS]", "[SEP]", "[PAD]"):
            continue
        results.append({
            "token": token,
            "importance": round(float(score), 4),
            "index": idx,
        })

    # Sort by importance
    results.sort(key=lambda x: x["importance"], reverse=True)

    return results[:top_k]


def merge_subword_importance(
    token_importance: List[Dict],
) -> List[Dict[str, float]]:
    """
    Merge WordPiece subword tokens (e.g. '##gia', '##nop') back into
    full words and sum their importance scores.

    Args:
        token_importance: output of extract_token_importance

    Returns:
        list of {"token": str, "importance": float} with merged subwords
    """
    merged: List[Dict] = []
    current_word = ""
    current_score = 0.0

    for item in sorted(token_importance, key=lambda x: x.get("index", 0)):
        token = item["token"]
        score = item["importance"]

        if token.startswith("##"):
            # Continuation of previous word
            current_word += token[2:]
            current_score += score
        else:
            # Save previous word
            if current_word:
                merged.append({
                    "token": current_word,
                    "importance": round(current_score, 4),
                })
            current_word = token
            current_score = score

    # Don't forget the last word
    if current_word:
        merged.append({
            "token": current_word,
            "importance": round(current_score, 4),
        })

    merged.sort(key=lambda x: x["importance"], reverse=True)
    return merged


def format_explanation(
    token_importance: List[Dict],
    original_text: str,
    top_k: int = 10,
) -> Dict:
    """
    Create a structured explanation combining token importance with
    the original text.

    Returns:
        {
            "original_text": str,
            "important_tokens": [{"token": str, "importance": float}, ...],
            "highlighted_text": str  (tokens wrapped in **bold**)
        }
    """
    merged = merge_subword_importance(token_importance)[:top_k]
    important_words = {item["token"].lower() for item in merged}

    # Build highlighted text
    words = original_text.split()
    highlighted = []
    for word in words:
        clean_word = word.lower().strip(".,;:!?()[]\"'")
        if clean_word in important_words:
            highlighted.append(f"**{word}**")
        else:
            highlighted.append(word)

    return {
        "original_text": original_text,
        "important_tokens": merged,
        "highlighted_text": " ".join(highlighted),
    }
