from __future__ import annotations

import numpy as np
import warnings
from skimage.metrics import structural_similarity as ssim_metric

# ==============================================================================
# HÀM HỖ TRỢ DÙNG CHUNG (UTILS)
# ==============================================================================

def _to_binary(mask: np.ndarray) -> np.ndarray:
    """Chuyển đổi mask thành nhị phân (0 và 1)."""
    return (mask > 0).astype(np.uint8)

def _get_confusion_elements(y_true: np.ndarray, y_pred: np.ndarray):
    """Tính toán nhanh TP, FP, FN, TN bằng phép toán vector."""
    gt = _to_binary(y_true).astype(bool)
    pred = _to_binary(y_pred).astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    tn = np.logical_and(~pred, ~gt).sum()
    
    return tp, fp, fn, tn


# ==============================================================================
# NHÓM 1: ĐÁNH GIÁ MODULE SEGMENTATION (PHÂN ĐOẠN VẾT NỨT)
# Mục tiêu: Đánh giá độ chính xác của mặt nạ (mask) tìm được so với ground-truth.
# ==============================================================================

def compute_iou_binary(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Tính Intersection over Union (IoU) - Chỉ số quan trọng nhất cho mask."""
    tp, fp, fn, _ = _get_confusion_elements(y_true, y_pred)
    union = tp + fp + fn
    if union == 0:
        return 1.0  # Cả mask thật và dự đoán đều trống (đúng hoàn toàn)
    return float(tp / union)

def compute_dice_score(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Tính Dice/F1 Score - Đánh giá tổng hòa độ trùng khớp."""
    tp, fp, fn, _ = _get_confusion_elements(y_true, y_pred)
    total = (2.0 * tp) + fp + fn
    if total == 0:
        return 1.0
    return float((2.0 * tp) / total)

def compute_precision(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Độ chính xác: Tỷ lệ nhận đúng vết nứt trên tổng số pixel dự đoán là nứt."""
    tp, fp, _, _ = _get_confusion_elements(y_true, y_pred)
    if (tp + fp) == 0:
        return 1.0 if tp > 0 else 0.0
    return float(tp / (tp + fp))

def compute_recall(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Độ bao phủ: Tỷ lệ tìm được vết nứt trên tổng số vết nứt thực tế (Rất quan trọng để không sót nứt mảnh)."""
    tp, _, fn, _ = _get_confusion_elements(y_true, y_pred)
    if (tp + fn) == 0:
        return 1.0
    return float(tp / (tp + fn))

def compute_fpr(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """False Positive Rate (FPR): Tỷ lệ nhận nhầm nền ảnh thành vết nứt."""
    _, fp, _, tn = _get_confusion_elements(y_true, y_pred)
    if (fp + tn) == 0:
        return 0.0
    return float(fp / (fp + tn))


# ==============================================================================
# NHÓM 2: ĐÁNH GIÁ MODULE INPAINTING (PHỤC HỒI ẢNH)
# Mục tiêu: Đo lường chất lượng ảnh sau khi lấp vết nứt (Toàn ảnh & Cục bộ).
# ==============================================================================

# --- ĐÁNH GIÁ TOÀN ẢNH (GLOBAL) ---

def compute_psnr(pred_image: np.ndarray, gt_image: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio toàn ảnh."""
    if pred_image.shape != gt_image.shape:
        raise ValueError(f"Shape mismatch PSNR: {pred_image.shape} vs {gt_image.shape}")
    mse = float(np.mean(np.square(pred_image.astype(np.float32) - gt_image.astype(np.float32))))
    return float("inf") if mse == 0.0 else float(20.0 * np.log10(255.0) - 10.0 * np.log10(mse))

def compute_ssim(pred_image: np.ndarray, gt_image: np.ndarray) -> float:
    """Structural Similarity Index toàn ảnh."""
    if pred_image.shape != gt_image.shape:
        raise ValueError(f"Shape mismatch SSIM: {pred_image.shape} vs {gt_image.shape}")
    channel_axis = -1 if pred_image.ndim == 3 else None
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ssim_metric(gt_image, pred_image, data_range=255, channel_axis=channel_axis)

# --- ĐÁNH GIÁ CỤC BỘ TRONG VÙNG NỨT (MASKED-REGION) - QUAN TRỌNG NHẤT ---

def compute_masked_psnr(pred_image: np.ndarray, gt_image: np.ndarray, mask: np.ndarray) -> float:
    """Chỉ tính PSNR tại những pixel nằm trong vùng vết nứt (tránh ảo giác điểm cao do nền ảnh lớn)."""
    binary_mask = _to_binary(mask).astype(bool)
    if not np.any(binary_mask):
        return float("inf")
    
    pred_masked = pred_image[binary_mask].astype(np.float32)
    gt_masked = gt_image[binary_mask].astype(np.float32)
    
    mse = float(np.mean(np.square(pred_masked - gt_masked)))
    return float("inf") if mse == 0.0 else float(20.0 * np.log10(255.0) - 10.0 * np.log10(mse))

def compute_masked_ssim(pred_image: np.ndarray, gt_image: np.ndarray, mask: np.ndarray) -> float:
    """
    Tính SSIM cục bộ: Lấy bản đồ SSIM toàn ảnh, sau đó chỉ cắt ra tính trung bình ở vùng bị nứt.
    Phương pháp này chuẩn học thuật để tránh lỗi 'window size' khi mask quá nhỏ.
    """
    binary_mask = _to_binary(mask).astype(bool)
    if not np.any(binary_mask):
        return 1.0

    channel_axis = -1 if pred_image.ndim == 3 else None
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, ssim_map = ssim_metric(gt_image, pred_image, data_range=255, channel_axis=channel_axis, full=True)
    
    if ssim_map.ndim == 3:
        ssim_map = ssim_map.mean(axis=2)
        
    return float(ssim_map[binary_mask].mean())


# ==============================================================================
# NHÓM MỞ RỘNG (FUTURE WORK PLACEHOLDERS)
# Đặt sẵn hàm ném lỗi để kiểm soát pipeline không gọi nhầm khi chưa setup thư viện.
# ==============================================================================

def compute_lpips(*_args: object, **_kwargs: object) -> float:
    raise NotImplementedError("LPIPS là optional dependency, sẽ triển khai trong Future Work.")

def compute_fid(*_args: object, **_kwargs: object) -> float:
    raise NotImplementedError("FID là optional dependency, sẽ triển khai trong Future Work.")