"""
Image Encoder — EfficientNet-B4 backbone with projection head.

Architecture matches the Kaggle training notebook exactly so that
pre-trained weights load without key mismatches.

Returns:
  - forward()              → pooled embedding   (B, projection_dim)
  - get_spatial_features() → spatial feature map (B, H*W, projection_dim)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tv_models


class ImageEncoder(nn.Module):
    def __init__(
        self,
        backbone_name: str = "efficientnet_b4",
        pretrained: bool = True,
        feature_dim: int = 1792,
        projection_dim: int = 512,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.projection_dim = projection_dim

        # --- backbone (matches notebook: nn.Sequential(*list(base.children())[:-2])) ---
        weights = "IMAGENET1K_V1" if pretrained else None
        backbone = getattr(tv_models, backbone_name)(weights=weights)
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)

        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

        # --- projection head (for pooled [CLS]-style embedding) ---
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # --- spatial projection (for cross-attention token features) ---
        self.spatial_projection = nn.Sequential(
            nn.Linear(feature_dim, projection_dim),
            nn.LayerNorm(projection_dim),
        )

    # -----------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        pooled = self.pool(feat).flatten(1)
        return self.projection(pooled)

    # -----------------------------------------------------------------
    def get_spatial_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)                     # (B, C, H, W)
        b, c, h, w = feat.shape
        spatial = feat.permute(0, 2, 3, 1).reshape(b, h * w, c)  # (B, HW, C)
        return self.spatial_projection(spatial)

    # -----------------------------------------------------------------
    def get_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)

    # -----------------------------------------------------------------
    def freeze_backbone(self) -> None:
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, last_n_blocks: int = 2) -> None:
        blocks = list(self.features.children())
        for block in blocks[-last_n_blocks:]:
            for param in block.parameters():
                param.requires_grad = True
