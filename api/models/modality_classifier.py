"""
Modality Classifier — MobileNetV3-Small backbone.

Auto-detects whether an input image is a chest X-ray, skin lesion,
or brain MRI scan.  Returns class index and confidence.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torchvision.models as tv_models


class ModalityClassifier(nn.Module):
    MODALITY_NAMES = ["chest", "skin", "brain"]

    def __init__(
        self,
        num_classes: int = 3,
        pretrained: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        weights = "IMAGENET1K_V1" if pretrained else None
        backbone = tv_models.mobilenet_v3_small(weights=weights)

        # Keep feature extractor, replace classifier
        self.features = backbone.features
        self.avgpool = backbone.avgpool

        in_features = 576  # MobileNetV3-Small last channel count
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) — resized to 224×224 expected
        Returns:
            logits: (B, num_classes)
        """
        feat = self.features(x)
        feat = self.avgpool(feat).flatten(1)
        return self.classifier(feat)

    @torch.no_grad()
    def predict_modality(self, x: torch.Tensor) -> Dict[str, any]:
        """
        Convenience method for inference.

        Returns:
            {"modality": str, "confidence": float, "all_probs": dict}
        """
        self.eval()
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=-1).squeeze(0)  # (num_classes,)
        idx = probs.argmax().item()
        return {
            "modality": self.MODALITY_NAMES[idx],
            "confidence": probs[idx].item(),
            "all_probs": {
                name: probs[i].item()
                for i, name in enumerate(self.MODALITY_NAMES)
            },
        }
