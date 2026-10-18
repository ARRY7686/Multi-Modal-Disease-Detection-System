"""
MultiModalDetector — architecture matches Kaggle training notebook exactly.

Supports three inference modes:
  - "multimodal"  – both image + text (full pipeline)
  - "image_only"  – image encoder → small head → logits
  - "text_only"   – text encoder  → small head → logits
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .fusion import CrossAttentionFusion
from .image_encoder import ImageEncoder
from .text_encoder import TextEncoder


class MultiModalDetector(nn.Module):
    def __init__(
        self,
        image_encoder: ImageEncoder,
        text_encoder: TextEncoder,
        fusion: nn.Module,
        num_classes: int = 25,
        projection_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.fusion = fusion
        # Direct nn.Sequential heads — matches notebook state dict keys
        self.image_head = nn.Sequential(
            nn.Linear(projection_dim, projection_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim // 2, num_classes),
        )
        self.text_head = nn.Sequential(
            nn.Linear(projection_dim, projection_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim // 2, num_classes),
        )

    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        mode: str = "multimodal",
    ) -> Dict[str, torch.Tensor]:
        image_pooled = None
        text_pooled = None

        if mode == "multimodal" and image is not None and input_ids is not None:
            img_spatial = self.image_encoder.get_spatial_features(image)
            txt_tokens = self.text_encoder.get_token_features(input_ids, attention_mask)
            # Notebook passes attention_mask directly; fusion inverts internally
            logits = self.fusion(img_spatial, txt_tokens, attention_mask)

        elif mode == "image_only" and image is not None:
            image_pooled = self.image_encoder(image)
            logits = self.image_head(image_pooled)

        elif mode == "text_only" and input_ids is not None:
            text_pooled = self.text_encoder(input_ids, attention_mask)
            logits = self.text_head(text_pooled)

        else:
            raise ValueError(f"Invalid mode={mode} or missing inputs")

        return {
            "logits": logits,
            "image_pooled": image_pooled,
            "text_pooled": text_pooled,
        }

    @classmethod
    def from_config(cls, cfg, pretrained: bool = True) -> "MultiModalDetector":
        """
        Construct from a Config object.

        pretrained=False skips downloading ImageNet / BioBERT weights; use it
        when a trained checkpoint is about to be loaded over the whole model.
        """
        image_encoder = ImageEncoder(
            backbone_name=cfg.image_backbone,
            pretrained=pretrained and cfg.image_pretrained,
            feature_dim=cfg.image_feature_dim,
            projection_dim=cfg.projection_dim,
            dropout=cfg.dropout,
        )
        text_encoder = TextEncoder(
            model_name=cfg.text_model_name,
            hidden_dim=cfg.text_hidden_dim,
            projection_dim=cfg.projection_dim,
            dropout=cfg.dropout,
            max_length=cfg.text_max_length,
            pretrained=pretrained,
        )
        fusion = CrossAttentionFusion(
            dim=cfg.projection_dim,
            num_heads=cfg.fusion_num_heads,
            num_layers=cfg.fusion_num_cross_layers,
            num_classes=cfg.num_classes,
            dropout=cfg.dropout,
        )
        return cls(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            fusion=fusion,
            num_classes=cfg.num_classes,
            projection_dim=cfg.projection_dim,
            dropout=cfg.dropout,
        )
