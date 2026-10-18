"""
Text preprocessing — BioBERT tokenization wrapper and symptom text cleaning.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Union

import torch
from transformers import AutoTokenizer


class TextPreprocessor:
    """
    Wraps the BioBERT tokenizer with medical-text cleaning.
    """

    def __init__(
        self,
        model_name: str = "dmis-lab/biobert-v1.1",
        max_length: int = 128,
        tokenizer_path: Optional[str] = None,
    ):
        self.max_length = max_length
        # Load from saved directory (local inference) or HF hub (training)
        source = tokenizer_path if tokenizer_path else model_name
        self.tokenizer = AutoTokenizer.from_pretrained(source)

    # ──────────────────────────────────────────────────────────────────
    # Text cleaning
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Normalise clinical symptom text.
          - lowercase
          - collapse whitespace
          - remove non-ASCII (keep basic punctuation)
          - normalise common medical abbreviations
        """
        text = text.lower().strip()
        # Collapse multi-space / newlines
        text = re.sub(r"\s+", " ", text)
        # Remove odd characters but keep letters, digits, basic punctuation
        text = re.sub(r"[^\w\s.,;:'\-/%()+]", "", text)
        # Common abbreviation normalisation
        replacements = {
            r"\by/o\b": "year old",
            r"\byr\b": "year",
            r"\byrs\b": "years",
            r"\bhx\b": "history",
            r"\bdx\b": "diagnosis",
            r"\brx\b": "prescription",
            r"\btx\b": "treatment",
            r"\bsx\b": "symptoms",
            r"\bpt\b": "patient",
            r"\bc/o\b": "complaining of",
            r"\bs/p\b": "status post",
            r"\bw/\b": "with",
            r"\bw/o\b": "without",
            r"\bb/l\b": "bilateral",
        }
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)

        return text.strip()

    # ──────────────────────────────────────────────────────────────────
    # Tokenization
    # ──────────────────────────────────────────────────────────────────

    def tokenize(
        self,
        text: Union[str, List[str]],
        clean: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize one or more symptom texts.

        Args:
            text:  single string or list of strings
            clean: whether to apply clean_text() first

        Returns:
            {"input_ids": (B, seq_len), "attention_mask": (B, seq_len)}
        """
        if isinstance(text, str):
            text = [text]

        if clean:
            text = [self.clean_text(t) for t in text]

        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }

    def decode(self, token_ids: torch.Tensor) -> List[str]:
        """Decode token IDs back to text (for debugging / explainability)."""
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)

    def tokens_to_words(self, token_ids: torch.Tensor) -> List[List[str]]:
        """
        Convert token IDs → list of string tokens per sample.
        Useful for attention visualisation.
        """
        results = []
        for ids in token_ids:
            tokens = self.tokenizer.convert_ids_to_tokens(ids.tolist())
            results.append(tokens)
        return results

    def save(self, path: str) -> None:
        """Save tokenizer to a directory (for model export)."""
        self.tokenizer.save_pretrained(path)
