"""
Text Encoder — BioBERT backbone with projection head.

Architecture matches the Kaggle training notebook exactly.

Returns:
  - forward()             → [CLS] embedding     (B, projection_dim)
  - get_token_features()  → per-token features   (B, seq_len, projection_dim)

Also exposes `last_attention_weights` after each forward pass for explainability.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class TextEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "dmis-lab/biobert-v1.1",
        hidden_dim: int = 768,
        projection_dim: int = 512,
        dropout: float = 0.3,
        max_length: int = 128,
        freeze_bert: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim
        self.max_length = max_length

        self.bert = AutoModel.from_pretrained(model_name, output_attentions=True)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.last_attention_weights: Optional[torch.Tensor] = None

    @staticmethod
    def load_tokenizer(model_name: str = "dmis-lab/biobert-v1.1"):
        return AutoTokenizer.from_pretrained(model_name)

    def _encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )
        self.last_attention_weights = outputs.attentions[-1].mean(dim=1).detach()
        return outputs.last_hidden_state

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self._encode(input_ids, attention_mask)
        return self.projection(hidden[:, 0, :])

    def get_token_features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self._encode(input_ids, attention_mask)
        return self.projection(hidden)

    def freeze_bert(self) -> None:
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert(self, last_n_layers: int = 2) -> None:
        layers = self.bert.encoder.layer
        for layer in layers[-last_n_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
