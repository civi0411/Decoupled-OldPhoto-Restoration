from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. FUNCTIONAL API (Toán học thuần túy)
# ==========================================

def _validate_shapes(logits: torch.Tensor, targets: torch.Tensor) -> None:
    if logits.shape != targets.shape:
        raise ValueError(f"logits shape {tuple(logits.shape)} phải khớp targets shape {tuple(targets.shape)}")

def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    _validate_shapes(logits, targets)

    probabilities = torch.sigmoid(logits)
    targets = targets.float()

    # Flatten theo từng batch (giữ nguyên chiều batch_size)
    probabilities = probabilities.flatten(start_dim=1)
    targets = targets.flatten(start_dim=1)

    intersection = (probabilities * targets).sum(dim=1)
    denominator = probabilities.sum(dim=1) + targets.sum(dim=1)
    
    dice_score = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice_score.mean()

def tversky_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.5,
    beta: float = 0.5,
    smooth: float = 1e-6,
) -> torch.Tensor:
    _validate_shapes(logits, targets)
    if alpha < 0.0 or beta < 0.0:
        raise ValueError("tversky alpha và beta phải không âm.")

    probabilities = torch.sigmoid(logits)
    targets = targets.float()

    probabilities = probabilities.flatten(start_dim=1)
    targets = targets.flatten(start_dim=1)

    true_positive = (probabilities * targets).sum(dim=1)
    false_positive = (probabilities * (1.0 - targets)).sum(dim=1)
    false_negative = ((1.0 - probabilities) * targets).sum(dim=1)

    tversky_index = (true_positive + smooth) / (
        true_positive + alpha * false_positive + beta * false_negative + smooth
    )
    return 1.0 - tversky_index.mean()


# ==========================================
# 2. CLASS API (Dành cho vòng lặp Training)
# ==========================================

class BCEDiceLoss(nn.Module):
    """
    Class quản lý hàm Loss BCE + Dice cho Module 1 (Crack Segmentation).
    Kế thừa nn.Module giúp dễ dàng đẩy lên GPU (model.to(device)).
    """
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, eps: float = 1e-6):
        super(BCEDiceLoss, self).__init__()
        if bce_weight < 0.0 or dice_weight < 0.0:
            raise ValueError("bce_weight và dice_weight phải không âm.")
        if bce_weight == 0.0 and dice_weight == 0.0:
            raise ValueError("Không thể đặt cả bce_weight và dice_weight bằng 0.")
            
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        _validate_shapes(logits, targets)
        targets_float = targets.float()
        
        # 1. Tính toán từng thành phần
        bce = F.binary_cross_entropy_with_logits(logits, targets_float)
        dice = dice_loss(logits, targets_float, eps=self.eps)
        
        # 2. Tính tổng Loss
        total_loss = (self.bce_weight * bce) + (self.dice_weight * dice)
        
        # 3. Đóng gói vào Dictionary để Tracker/Logger sử dụng
        loss_dict = {
            'bce_dice_total': total_loss.item(),
            'bce': bce.item(),
            'dice': dice.item()
        }
        
        return total_loss, loss_dict


class BCETverskyLoss(nn.Module):
    """
    Class quản lý hàm Loss BCE + Tversky (Dùng khi cực kỳ mất cân bằng dữ liệu)
    """
    def __init__(
        self, 
        bce_weight: float = 0.5, 
        tversky_weight: float = 0.5, 
        alpha: float = 0.7,  # Phạt nặng False Negative (Bỏ sót vết nứt)
        beta: float = 0.3,   # Phạt nhẹ False Positive (Nhận nhầm vết nứt)
        smooth: float = 1e-6
    ):
        super(BCETverskyLoss, self).__init__()
        self.bce_weight = bce_weight
        self.tversky_weight = tversky_weight
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        _validate_shapes(logits, targets)
        targets_float = targets.float()
        
        bce = F.binary_cross_entropy_with_logits(logits, targets_float)
        tversky = tversky_loss(logits, targets_float, alpha=self.alpha, beta=self.beta, smooth=self.smooth)
        
        total_loss = (self.bce_weight * bce) + (self.tversky_weight * tversky)
        
        loss_dict = {
            'bce_tversky_total': total_loss.item(),
            'bce': bce.item(),
            'tversky': tversky.item()
        }
        
        return total_loss, loss_dict