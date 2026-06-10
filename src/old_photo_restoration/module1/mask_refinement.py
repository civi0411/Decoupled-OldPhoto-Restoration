from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from old_photo_restoration.module1.predictor import SegmentationPredictor


def ensure_binary_mask(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim == 3 and array.shape[2] == 3:
        array = cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    elif array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    elif array.ndim != 2:
        raise ValueError(f"Mask phải có shape HxW hoặc HxWx1/HxWx3, nhận được {array.shape}")

    if array.dtype == np.bool_:
        binary = array.astype(np.uint8) * 255
    elif np.issubdtype(array.dtype, np.floating):
        threshold = 0.5 if float(np.nanmax(array)) <= 1.0 else 127.0
        binary = (array > threshold).astype(np.uint8) * 255
    else:
        binary = (array > 127).astype(np.uint8) * 255
    return np.ascontiguousarray(binary.astype(np.uint8))


def mask_ratio(mask: np.ndarray) -> float:
    binary = ensure_binary_mask(mask)
    return float((binary > 0).mean())


def remove_small_components_keep_long(mask: np.ndarray, min_area: int, min_span: int) -> np.ndarray:
    binary = (ensure_binary_mask(mask) > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    kept = np.zeros_like(binary)
    for label_index in range(1, num_labels):
        area = int(stats[label_index, cv2.CC_STAT_AREA])
        width = int(stats[label_index, cv2.CC_STAT_WIDTH])
        height = int(stats[label_index, cv2.CC_STAT_HEIGHT])
        if area >= min_area or max(width, height) >= min_span:
            kept[labels == label_index] = 1
    return kept.astype(np.uint8) * 255


def filter_components_by_area_or_span(binary_mask: np.ndarray, min_area: int, min_span: int) -> tuple[np.ndarray, int, int]:
    mask = (ensure_binary_mask(binary_mask) > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered = np.zeros_like(mask)
    kept = 0
    total = max(0, num_labels - 1)
    for label_index in range(1, num_labels):
        area = int(stats[label_index, cv2.CC_STAT_AREA])
        width = int(stats[label_index, cv2.CC_STAT_WIDTH])
        height = int(stats[label_index, cv2.CC_STAT_HEIGHT])
        if area >= min_area or max(width, height) >= min_span:
            filtered[labels == label_index] = 1
            kept += 1
    return filtered * 255, total, kept


def _line_kernels(kernel_len: int) -> list[np.ndarray]:
    if kernel_len < 3:
        kernel_len = 3
    if kernel_len % 2 == 0:
        kernel_len += 1

    horizontal = np.zeros((kernel_len, kernel_len), dtype=np.uint8)
    vertical = np.zeros((kernel_len, kernel_len), dtype=np.uint8)
    diag_down = np.zeros((kernel_len, kernel_len), dtype=np.uint8)
    diag_up = np.zeros((kernel_len, kernel_len), dtype=np.uint8)
    center = kernel_len // 2
    horizontal[center, :] = 1
    vertical[:, center] = 1
    np.fill_diagonal(diag_down, 1)
    np.fill_diagonal(np.fliplr(diag_up), 1)
    return [horizontal, vertical, diag_down, diag_up]


def bridge_line_gaps(mask: np.ndarray, kernel_len: int = 7, iterations: int = 1) -> np.ndarray:
    binary = ensure_binary_mask(mask)
    bridged = binary.copy()
    for kernel in _line_kernels(kernel_len):
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        bridged = np.maximum(bridged, closed)
    return ensure_binary_mask(bridged)


def refine_mask_repair_wide(mask: np.ndarray) -> np.ndarray:
    binary = ensure_binary_mask(mask)
    square3 = np.ones((3, 3), dtype=np.uint8)
    ellipse3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    repaired = remove_small_components_keep_long(binary, min_area=8, min_span=24)
    repaired = bridge_line_gaps(repaired, kernel_len=9, iterations=1)
    repaired = cv2.morphologyEx(repaired, cv2.MORPH_CLOSE, square3, iterations=1)
    repaired = cv2.dilate(repaired, ellipse3, iterations=1)
    repaired = cv2.morphologyEx(repaired, cv2.MORPH_CLOSE, square3, iterations=1)
    return ensure_binary_mask(repaired)


def build_cv_crack_mask(
    image_rgb: np.ndarray,
    profile: str = "notebook_v7_candidate",
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    if profile != "notebook_v7_candidate":
        raise ValueError(f"Chưa hỗ trợ cv profile: {profile}")

    gray = cv2.cvtColor(image_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    blackhat_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    tophat_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    close_kernel = np.ones((3, 3), dtype=np.uint8)
    dilate_kernel = np.ones((3, 3), dtype=np.uint8)

    blackhat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, blackhat_kernel)
    tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, tophat_kernel)
    edges = cv2.Canny(enhanced, threshold1=60, threshold2=160)

    score = np.maximum(blackhat, (0.65 * tophat).astype(np.uint8))
    score = np.maximum(score, (0.25 * edges).astype(np.uint8))
    score = cv2.GaussianBlur(score, (3, 3), 0)

    pctl = 97.2
    threshold = max(18.0, float(np.percentile(score, pctl)))
    raw_mask = (score >= threshold).astype(np.uint8) * 255
    after_close = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    after_component_filter, num_components, num_components_kept = filter_components_by_area_or_span(
        after_close,
        min_area=22,
        min_span=18,
    )
    final_mask = cv2.dilate(after_component_filter, dilate_kernel, iterations=1)

    area_pct_before_fallback = mask_ratio(final_mask) * 100.0
    fallback_used = False
    if area_pct_before_fallback < 1.0:
        fallback_used = True
        pctl = 96.0
        threshold = max(14.0, float(np.percentile(score, pctl)))
        raw_mask = (score >= threshold).astype(np.uint8) * 255
        after_close = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
        after_component_filter, num_components, num_components_kept = filter_components_by_area_or_span(
            after_close,
            min_area=22,
            min_span=18,
        )
        final_mask = cv2.dilate(after_component_filter, dilate_kernel, iterations=1)

    final_mask = ensure_binary_mask(final_mask)
    area_pct_final = mask_ratio(final_mask) * 100.0
    debug_images = {
        "gray": gray,
        "clahe": enhanced,
        "blackhat_response": blackhat,
        "tophat_response": tophat,
        "canny": edges,
        "score": score,
        "cv_raw_mask": raw_mask,
        "cv_after_close": after_close,
        "cv_after_component_filter": after_component_filter,
        "cv_final_mask": final_mask,
    }
    info = {
        "cv_mask_ratio_before_invert_check": mask_ratio(final_mask),
        "cv_mask_ratio_after_invert_check": mask_ratio(final_mask),
        "cv_auto_inverted": False,
        "cv_warning": None,
        "cv_pctl": float(pctl),
        "cv_threshold": float(threshold),
        "cv_area_pct_before_fallback": float(area_pct_before_fallback),
        "cv_area_pct_final": float(area_pct_final),
        "cv_fallback_used": bool(fallback_used),
        "cv_num_components": int(num_components),
        "cv_num_components_kept": int(num_components_kept),
    }
    return final_mask, info, debug_images


def union_masks(dl_mask: np.ndarray, cv_mask: np.ndarray) -> np.ndarray:
    return np.maximum(ensure_binary_mask(dl_mask), ensure_binary_mask(cv_mask))


def build_hybrid_mask(
    image_path: Path,
    predictor: SegmentationPredictor,
    threshold: float = 0.5,
) -> dict[str, Any]:
    image_rgb = predictor.read_image_rgb(image_path)
    dl_mask = predictor.predict_dl_mask(image_path, threshold=threshold)
    cv_mask, cv_info, cv_debug_images = build_cv_crack_mask(image_rgb, profile="notebook_v7_candidate")
    union_mask = union_masks(dl_mask, cv_mask)
    final_mask = refine_mask_repair_wide(union_mask)

    dl_cnt = int(np.count_nonzero(dl_mask))
    cv_cnt = int(np.count_nonzero(cv_mask))
    union_cnt = int(np.count_nonzero(union_mask))
    final_cnt = int(np.count_nonzero(final_mask))
    rejected_cv = cv2.bitwise_and(cv_mask, cv2.bitwise_not(final_mask))

    total_pixels = int(dl_mask.size)
    stats = {
        "dl_mask_ratio": float(dl_cnt / total_pixels),
        "cv_mask_ratio": float(cv_cnt / total_pixels),
        "union_before_refine_ratio": float(union_cnt / total_pixels),
        "final_mask_ratio": float(final_cnt / total_pixels),
        "rejected_cv_over_cv_ratio": float(np.count_nonzero(rejected_cv) / cv_cnt) if cv_cnt > 0 else 0.0,
    }
    return {
        "image_rgb": image_rgb,
        "dl_mask": dl_mask,
        "cv_mask": cv_mask,
        "union_mask": union_mask,
        "final_mask": final_mask,
        "stats": stats,
        "cv_info": cv_info,
        "cv_debug_images": cv_debug_images,
    }
