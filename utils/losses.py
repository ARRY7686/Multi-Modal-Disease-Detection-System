"""
Loss functions for multi-label disease classification.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Sigmoid Focal Loss for multi-label classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Applied independently per label (binary).
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            inputs:  (B, C) raw logits (before sigmoid)
            targets: (B, C) binary labels {0, 1}
        Returns:
            scalar loss
        """
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none"
        )
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross-Entropy for multi-label classification.
    Accepts per-class positive weights to handle class imbalance.
    """

    def __init__(
        self,
        pos_weights: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.reduction = reduction
        self.register_buffer(
            "pos_weights",
            pos_weights if pos_weights is not None else None,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            inputs:  (B, C) raw logits
            targets: (B, C) binary labels
        """
        return F.binary_cross_entropy_with_logits(
            inputs,
            targets,
            pos_weight=self.pos_weights,
            reduction=self.reduction,
        )


def get_loss_fn(
    loss_type: str = "focal",
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    pos_weights: Optional[torch.Tensor] = None,
) -> nn.Module:
    """Factory: build the configured loss function."""
    if loss_type == "focal":
        return FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    elif loss_type == "bce_weighted":
        return WeightedBCELoss(pos_weights=pos_weights)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
