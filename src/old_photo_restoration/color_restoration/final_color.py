from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .config import FinalColorConfig
from .processing import validate_rgb_uint8


def srgb_to_linear(image_rgb: np.ndarray) -> np.ndarray:
    image = validate_rgb_uint8(image_rgb).astype(np.float32) / 255.0
    return np.where(image <= 0.04045, image / 12.92, ((image + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(image_linear: np.ndarray) -> np.ndarray:
    image = np.clip(np.asarray(image_linear, dtype=np.float32), 0.0, 1.0)
    srgb = np.where(
        image <= 0.0031308,
        image * 12.92,
        1.055 * (image ** (1.0 / 2.4)) - 0.055,
    )
    return validate_rgb_uint8(np.clip(srgb * 255.0, 0, 255).astype(np.uint8))


def estimate_conservative_ccm(
    image_rgb: np.ndarray,
    config: FinalColorConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    linear = srgb_to_linear(image_rgb)
    luminance = (
        0.2126 * linear[:, :, 0]
        + 0.7152 * linear[:, :, 1]
        + 0.0722 * linear[:, :, 2]
    )
    channel_max = linear.max(axis=2)
    channel_min = linear.min(axis=2)
    saturation = (channel_max - channel_min) / np.maximum(channel_max, 1e-6)
    low, high = np.percentile(luminance, config.ccm_luminance_percentiles)
    sample_mask = (
        (luminance >= low)
        & (luminance <= high)
        & (saturation <= config.ccm_saturation_max)
    )
    fallback_used = False
    if int(sample_mask.sum()) < config.ccm_min_sample_pixels:
        sample_mask = (luminance >= low) & (luminance <= high)
        fallback_used = True

    channel_means = linear[sample_mask].mean(axis=0)
    neutral_level = float(np.exp(np.log(np.maximum(channel_means, 1e-6)).mean()))
    raw_gains = neutral_level / np.maximum(channel_means, 1e-6)
    limited_gains = np.clip(
        raw_gains,
        1.0 - config.ccm_max_gain_delta,
        1.0 + config.ccm_max_gain_delta,
    )
    gains = 1.0 + config.ccm_strength * (limited_gains - 1.0)
    matrix = np.diag(gains.astype(np.float32))
    return matrix, {
        "method": "robust_gray_world_diagonal_ccm",
        "strength": config.ccm_strength,
        "max_gain_delta": config.ccm_max_gain_delta,
        "sample_ratio": float(sample_mask.mean()),
        "fallback_sample_mask_used": fallback_used,
        "channel_means_linear_rgb": channel_means.tolist(),
        "raw_gains": raw_gains.tolist(),
        "limited_gains": limited_gains.tolist(),
        "applied_gains": gains.tolist(),
        "matrix": matrix.tolist(),
    }


def apply_ccm(image_rgb: np.ndarray, matrix: np.ndarray) -> tuple[np.ndarray, float]:
    matrix_array = np.asarray(matrix, dtype=np.float32)
    if matrix_array.shape != (3, 3):
        raise ValueError(f"CCM must have shape 3x3, got {matrix_array.shape}")
    linear = srgb_to_linear(image_rgb)
    corrected = np.einsum("...c,dc->...d", linear, matrix_array)
    clipped_ratio = float(np.mean((corrected < 0.0) | (corrected > 1.0)))
    return linear_to_srgb(corrected), clipped_ratio


def apply_final_chroma_matching(
    image_rgb: np.ndarray,
    reference_rgb: np.ndarray,
    *,
    strength: float,
    max_shift: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    image = validate_rgb_uint8(image_rgb)
    reference = validate_rgb_uint8(reference_rgb)
    if reference.shape[:2] != image.shape[:2]:
        reference = cv2.resize(
            reference,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    reference_lab = cv2.cvtColor(reference, cv2.COLOR_RGB2LAB).astype(np.float32)
    raw_shift_a = float(reference_lab[:, :, 1].mean() - image_lab[:, :, 1].mean())
    raw_shift_b = float(reference_lab[:, :, 2].mean() - image_lab[:, :, 2].mean())
    shift_a = float(np.clip(raw_shift_a * strength, -max_shift, max_shift))
    shift_b = float(np.clip(raw_shift_b * strength, -max_shift, max_shift))
    image_lab[:, :, 1] = np.clip(image_lab[:, :, 1] + shift_a, 0, 255)
    image_lab[:, :, 2] = np.clip(image_lab[:, :, 2] + shift_b, 0, 255)
    matched = cv2.cvtColor(image_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    return validate_rgb_uint8(matched), {
        "method": "opencv_lab_chroma_match",
        "strength": float(strength),
        "max_shift": float(max_shift),
        "shift_a": shift_a,
        "shift_b": shift_b,
    }


def apply_final_color_correction(
    image_rgb: np.ndarray,
    reference_rgb: np.ndarray,
    config: FinalColorConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    image = validate_rgb_uint8(image_rgb)
    if not config.enabled:
        return image.copy(), {
            "name": "ccm_color_correction",
            "status": "skipped",
            "backend": "none",
            "reason": "disabled",
        }

    current = image.copy()
    details: dict[str, Any] = {}
    if config.method in {"ccm", "ccm_then_chroma_match"}:
        matrix, ccm_metadata = estimate_conservative_ccm(current, config)
        current, clipped_ratio = apply_ccm(current, matrix)
        ccm_metadata["clipped_ratio"] = clipped_ratio
        details["ccm"] = ccm_metadata
    if config.method in {"lab_chroma_match", "ccm_then_chroma_match"}:
        current, chroma_metadata = apply_final_chroma_matching(
            current,
            reference_rgb,
            strength=config.chroma_match_strength,
            max_shift=config.chroma_match_max_shift,
        )
        details["chroma_match"] = chroma_metadata
    return current, {
        "name": "ccm_color_correction",
        "status": "applied",
        "backend": config.method,
        "reference": "quality_restored",
        **details,
    }
