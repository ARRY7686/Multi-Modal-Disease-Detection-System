"""
Text attention explainability — extract token importance from BioBERT attention.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

SPECIAL_TOKENS = ("[CLS]", "[SEP]", "[PAD]")


def extract_token_importance(
    attention_weights: torch.Tensor,
    tokens: List[str],
    attention_mask: Optional[torch.Tensor] = None,
    top_k: Optional[int] = None,
) -> List[Dict]:
    """
    Per-token importance from [CLS] attention in BioBERT's last layer.

    Args:
        attention_weights: (B, seq, seq) — mean-pooled across heads
                           from the last transformer layer.
                           Stored in TextEncoder.last_attention_weights.
        tokens:            list of string tokens for the first sample
        attention_mask:    (B, seq) — 1 for real tokens, 0 for padding
        top_k:             if given, keep only the k most important tokens.
                           Leave as None when the result will be passed to
                           merge_subword_importance(), which needs every
                           token to reassemble WordPiece words correctly.

    Returns:
        List of {"token": str, "importance": float, "index": int} in
        sequence order (or sorted by importance when top_k is given).
    """
    cls_attn = attention_weights[0, 0].float()  # [CLS] row: (seq,)

    if attention_mask is not None:
        cls_attn = cls_attn * attention_mask[0].float()

    cls_attn = cls_attn.cpu().numpy()
    if cls_attn.max() > 0:
        cls_attn = cls_attn / cls_attn.max()

    results = [
        {"token": token, "importance": float(score), "index": idx}
        for idx, (token, score) in enumerate(zip(tokens, cls_attn, strict=True))
        if token not in SPECIAL_TOKENS
    ]

    if top_k is not None:
        results.sort(key=lambda x: x["importance"], reverse=True)
        results = results[:top_k]
    return results


def merge_subword_importance(token_importance: List[Dict]) -> List[Dict]:
    """
    Merge WordPiece subword tokens (e.g. 'dys', '##pnea') back into full
    words, scoring each word by the mean of its pieces, then re-normalise
    to [0, 1].

    A '##' piece is only attached to the token immediately before it in the
    sequence, so passing a filtered list cannot glue pieces onto unrelated
    words.

    Returns:
        list of {"token": str, "importance": float} sorted by importance
    """
    words: List[Dict] = []
    prev_index: Optional[int] = None

    for item in sorted(token_importance, key=lambda x: x["index"]):
        token, score, index = item["token"], item["importance"], item["index"]
        contiguous = prev_index is not None and index == prev_index + 1
        if token.startswith("##") and words and contiguous:
            words[-1]["token"] += token[2:]
            words[-1]["scores"].append(score)
        else:
            words.append({"token": token.removeprefix("##"), "scores": [score]})
        prev_index = index

    merged = [
        {"token": w["token"], "importance": sum(w["scores"]) / len(w["scores"])}
        for w in words
    ]
    peak = max((m["importance"] for m in merged), default=0.0)
    for m in merged:
        m["importance"] = round(m["importance"] / peak, 4) if peak > 0 else 0.0

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

    Args:
        token_importance: output of extract_token_importance(top_k=None)

    Returns:
        {
            "original_text": str,
            "important_tokens": [{"token": str, "importance": float}, ...],
            "highlighted_text": str  (important words wrapped in **bold**)
        }
    """
    # Drop punctuation before merging so it cannot set the normalisation peak.
    words_only = [t for t in token_importance if t["token"].removeprefix("##").isalnum()]
    merged = merge_subword_importance(words_only)[:top_k]
    important_words = {item["token"].lower() for item in merged}

    highlighted = []
    for word in original_text.split():
        clean_word = word.lower().strip(".,;:!?()[]\"'")
        highlighted.append(f"**{word}**" if clean_word in important_words else word)

    return {
        "original_text": original_text,
        "important_tokens": merged,
        "highlighted_text": " ".join(highlighted),
    }
