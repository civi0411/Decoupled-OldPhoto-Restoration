from __future__ import annotations

import numpy as np


def _to_binary(mask: np.ndarray) -> np.ndarray:
    return (mask > 0).astype(np.uint8)


def compute_iou_binary(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = _to_binary(pred_mask)
    gt = _to_binary(gt_mask)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def compute_mae_image(pred_image: np.ndarray, gt_image: np.ndarray) -> float:
    if pred_image.shape != gt_image.shape:
        raise ValueError(f"Images must share the same shape to compute MAE: {pred_image.shape} vs {gt_image.shape}")
    diff = np.abs(pred_image.astype(np.float32) - gt_image.astype(np.float32))
    return float(diff.mean())


def compute_psnr(pred_image: np.ndarray, gt_image: np.ndarray) -> float:
    if pred_image.shape != gt_image.shape:
        raise ValueError(f"Images must share the same shape to compute PSNR: {pred_image.shape} vs {gt_image.shape}")
    mse = float(np.mean(np.square(pred_image.astype(np.float32) - gt_image.astype(np.float32))))
    return float("inf") if mse == 0.0 else float(20.0 * np.log10(255.0) - 10.0 * np.log10(mse))


def intersection_over_union(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_mask = _to_binary(y_true)
    pred_mask = _to_binary(y_pred)
    intersection = np.logical_and(true_mask, pred_mask).sum()
    union = np.logical_or(true_mask, pred_mask).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def dice_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_mask = _to_binary(y_true)
    pred_mask = _to_binary(y_pred)
    intersection = np.logical_and(true_mask, pred_mask).sum()
    total = true_mask.sum() + pred_mask.sum()
    if total == 0:
        return 1.0
    return float((2.0 * intersection) / total)


def lpips_score(*_args: object, **_kwargs: object) -> float:
    raise NotImplementedError("LPIPS is unavailable for this artifact set because the optional dependency is not configured.")


def fid_score(*_args: object, **_kwargs: object) -> float:
    raise NotImplementedError("FID is unavailable for this artifact set because the optional dependency is not configured.")
