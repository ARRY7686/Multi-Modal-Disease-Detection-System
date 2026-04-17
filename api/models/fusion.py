"""
Fusion Layers — architecture matches Kaggle training notebook exactly.

  1. ConcatFusion        – simple concatenation baseline
  2. CrossAttentionFusion – bidirectional cross-attention (preferred)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


# =========================================================================
# 1. Concat Fusion (baseline)
# =========================================================================
class ConcatFusion(nn.Module):
    def __init__(self, dim: int = 512, num_classes: int = 25, dropout: float = 0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, num_classes),
        )

    def forward(self, img_emb: torch.Tensor, txt_emb: torch.Tensor) -> torch.Tensor:
        return self.classifier(torch.cat([img_emb, txt_emb], dim=-1))


# =========================================================================
# 2. Cross-Attention Fusion (advanced)
# =========================================================================
class CrossAttentionBlock(nn.Module):
    """Bidirectional cross-attention — attribute names match notebook."""

    def __init__(self, dim: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.cross_attn_img2txt = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.cross_attn_txt2img = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm1_img = nn.LayerNorm(dim)
        self.norm1_txt = nn.LayerNorm(dim)
        self.norm2_img = nn.LayerNorm(dim)
        self.norm2_txt = nn.LayerNorm(dim)
        self.ffn_img = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 4, dim), nn.Dropout(dropout),
        )
        self.ffn_txt = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 4, dim), nn.Dropout(dropout),
        )

    def forward(
        self,
        img_feats: torch.Tensor,
        txt_feats: torch.Tensor,
        txt_mask: Optional[torch.Tensor] = None,
    ):
        key_padding_mask = ~txt_mask.bool() if txt_mask is not None else None
        h_img, _ = self.cross_attn_img2txt(
            img_feats, txt_feats, txt_feats, key_padding_mask=key_padding_mask,
        )
        img_feats = self.norm1_img(img_feats + h_img)
        h_txt, _ = self.cross_attn_txt2img(txt_feats, img_feats, img_feats)
        txt_feats = self.norm1_txt(txt_feats + h_txt)
        img_feats = self.norm2_img(img_feats + self.ffn_img(img_feats))
        txt_feats = self.norm2_txt(txt_feats + self.ffn_txt(txt_feats))
        return img_feats, txt_feats


class CrossAttentionFusion(nn.Module):
    def __init__(
        self,
        dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        num_classes: int = 25,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttentionBlock(dim, num_heads, dropout=0.1)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, num_classes),
        )

    def forward(
        self,
        img_spatial: torch.Tensor,
        txt_tokens: torch.Tensor,
        txt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            img_spatial, txt_tokens = layer(img_spatial, txt_tokens, txt_mask)
        img_pool = img_spatial.mean(dim=1)
        if txt_mask is not None:
            mask_f = txt_mask.unsqueeze(-1).float()
            txt_pool = (txt_tokens * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        else:
            txt_pool = txt_tokens.mean(dim=1)
        fused = torch.cat([img_pool, txt_pool], dim=-1)
        return self.classifier(fused)
