"""Loss functions for old photo crack segmentation training.

Crack segmentation in historical photographs suffers from severe class imbalance
because crack pixels typically constitute only 1-5% of the total image area.
Combining Binary Cross-Entropy with smooth Dice Loss balances pixel-wise accuracy
with overall region overlap.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    """Combined Binary Cross-Entropy with Logits and Dice Loss."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1e-6) -> None:
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute combined BCE + Dice loss.

        Args:
            logits: Predicted logits from model of shape (B, 1, H, W).
            targets: Binary target mask of shape (B, 1, H, W) with values in [0, 1].

        Returns:
            Scalar loss tensor.
        """
        if logits.shape != targets.shape:
            raise ValueError(f"Shape mismatch in BCEDiceLoss: logits {logits.shape} vs targets {targets.shape}")

        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")

        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = (1.0 - dice_score).mean()

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class FocalLoss(nn.Module):
    """Focal Loss for addressing extreme class imbalance in thin crack structures."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss."""
        bce_raw = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        modulating_factor = (1.0 - p_t) ** self.gamma

        loss = alpha_t * modulating_factor * bce_raw
        return loss.mean()
