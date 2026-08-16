from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .config import ColorRestorationConfig


def validate_rgb_uint8(image_rgb: np.ndarray) -> np.ndarray:
    """Validate the strict in-memory boundary used by the core module."""
    if not isinstance(image_rgb, np.ndarray):
        raise TypeError("image_rgb must be a numpy.ndarray")
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"image_rgb must have shape HxWx3, got {image_rgb.shape}")
    if image_rgb.dtype != np.uint8:
        raise TypeError(f"image_rgb must use dtype uint8, got {image_rgb.dtype}")
    if image_rgb.shape[0] <= 0 or image_rgb.shape[1] <= 0:
        raise ValueError("image_rgb height and width must be positive")
    return np.ascontiguousarray(image_rgb)


def _lab_change_metrics(reference_rgb: np.ndarray, candidate_rgb: np.ndarray) -> dict[str, float]:
    reference_lab = cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    candidate_lab = cv2.cvtColor(candidate_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    luminance_change = np.abs(candidate_lab[:, :, 0] - reference_lab[:, :, 0])
    chroma_change = np.linalg.norm(candidate_lab[:, :, 1:] - reference_lab[:, :, 1:], axis=2)
    pixel_change = np.abs(candidate_rgb.astype(np.float32) - reference_rgb.astype(np.float32))
    return {
        "mean_pixel_change": float(pixel_change.mean()),
        "mean_luminance_change": float(luminance_change.mean()),
        "mean_chroma_change": float(chroma_change.mean()),
    }


def preprocess_after_lama(
    image_rgb: np.ndarray,
    config: ColorRestorationConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    image = validate_rgb_uint8(image_rgb).copy()
    metadata: dict[str, Any] = {
        "name": "quality_restoration",
        "status": "applied",
        "backend": "opencv_conservative",
        "denoise_applied": False,
        "contrast_applied": False,
        "sharpen_applied": False,
    }

    if config.denoise.enabled and config.denoise.strength > 0:
        image = cv2.fastNlMeansDenoisingColored(
            image,
            None,
            config.denoise.strength,
            config.denoise.strength,
            7,
            21,
        )
        metadata["denoise_applied"] = True
        metadata["denoise_strength"] = config.denoise.strength

    if config.contrast.enabled:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=config.contrast.clip_limit,
            tileGridSize=config.contrast.tile_grid_size,
        )
        enhanced_l = clahe.apply(l_channel)
        image = cv2.cvtColor(cv2.merge([enhanced_l, a_channel, b_channel]), cv2.COLOR_LAB2RGB)
        metadata["contrast_applied"] = True
        metadata["clahe_clip_limit"] = config.contrast.clip_limit
        metadata["clahe_tile_grid_size"] = list(config.contrast.tile_grid_size)

    if config.sharpen.enabled and config.sharpen.amount > 0:
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=config.sharpen.sigma)
        image = cv2.addWeighted(
            image,
            1.0 + config.sharpen.amount,
            blurred,
            -config.sharpen.amount,
            0,
        )
        metadata["sharpen_applied"] = True
        metadata["sharpen_amount"] = config.sharpen.amount
        metadata["sharpen_sigma"] = config.sharpen.sigma

    return validate_rgb_uint8(image), metadata


def apply_conservative_color_correction(
    image_rgb: np.ndarray,
    config: ColorRestorationConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    image = validate_rgb_uint8(image_rgb)
    correction = config.color_correction
    metadata: dict[str, Any] = {
        "name": "color_correction",
        "backend": "opencv_lab_neutral_shift",
        "status": "skipped",
        "reason": "disabled",
    }
    if not correction.enabled or correction.strength == 0:
        return image.copy(), metadata

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    neutral_mask = (
        (hsv[:, :, 1] <= correction.neutral_saturation_max)
        & (lab[:, :, 0] >= 40)
        & (lab[:, :, 0] <= 220)
    )
    neutral_count = int(neutral_mask.sum())
    metadata.update(
        {
            "neutral_pixel_count": neutral_count,
            "neutral_pixel_ratio": float(neutral_mask.mean()),
            "strength": correction.strength,
            "max_chroma_shift": correction.max_chroma_shift,
        }
    )
    if neutral_count < correction.min_neutral_pixels:
        metadata["reason"] = "insufficient_neutral_pixels"
        return image.copy(), metadata

    mean_a = float(lab[:, :, 1][neutral_mask].mean())
    mean_b = float(lab[:, :, 2][neutral_mask].mean())
    shift_a = float(
        np.clip(
            (128.0 - mean_a) * correction.strength,
            -correction.max_chroma_shift,
            correction.max_chroma_shift,
        )
    )
    shift_b = float(
        np.clip(
            (128.0 - mean_b) * correction.strength,
            -correction.max_chroma_shift,
            correction.max_chroma_shift,
        )
    )
    lab[:, :, 1] = np.clip(lab[:, :, 1] + shift_a, 0, 255)
    lab[:, :, 2] = np.clip(lab[:, :, 2] + shift_b, 0, 255)
    corrected = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    metadata.update(
        {
            "status": "applied",
            "reason": "applied",
            "neutral_mean_a": mean_a,
            "neutral_mean_b": mean_b,
            "shift_a": shift_a,
            "shift_b": shift_b,
        }
    )
    return validate_rgb_uint8(corrected), metadata


def apply_safety_postprocessing(
    original_rgb: np.ndarray,
    candidate_rgb: np.ndarray,
    config: ColorRestorationConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    original = validate_rgb_uint8(original_rgb)
    candidate = validate_rgb_uint8(candidate_rgb).copy()
    if candidate.shape != original.shape:
        raise ValueError(f"candidate shape {candidate.shape} must match original shape {original.shape}")

    before = _lab_change_metrics(original, candidate)
    limits = {
        "mean_pixel_change": config.safety.max_mean_pixel_change,
        "mean_luminance_change": config.safety.max_mean_luminance_change,
        "mean_chroma_change": config.safety.max_mean_chroma_change,
    }
    if not config.safety.enabled:
        return candidate, {
            "name": "safety_postprocessing",
            "status": "skipped",
            "reason": "disabled",
            "limits": limits,
            "change_before_safety": before,
            "change_after_safety": before,
            "warnings": [],
        }
    required_factor = min(
        [1.0]
        + [
            limit / before[name]
            for name, limit in limits.items()
            if before[name] > limit and before[name] > 0
        ]
    )
    applied_factor = min(1.0, required_factor)
    correction_rejected = applied_factor < config.safety.min_blend_factor
    if correction_rejected:
        applied_factor = 0.0
    def blend_candidate(factor: float) -> np.ndarray:
        if factor >= 1.0:
            return candidate.copy()
        blended = (
            original.astype(np.float32)
            + factor * (candidate.astype(np.float32) - original.astype(np.float32))
        )
        return np.clip(np.rint(blended), 0, 255).astype(np.uint8)

    output = blend_candidate(applied_factor)
    after = _lab_change_metrics(original, output)
    for _ in range(3):
        exceeded = [
            limits[name] / after[name]
            for name in limits
            if after[name] > limits[name] and after[name] > 0
        ]
        if not exceeded or applied_factor == 0.0:
            break
        applied_factor *= min(exceeded) * 0.98
        if applied_factor < config.safety.min_blend_factor:
            correction_rejected = True
            applied_factor = 0.0
        output = blend_candidate(applied_factor)
        after = _lab_change_metrics(original, output)

    limited = applied_factor < 1.0
    warnings = []
    if limited:
        warnings.append("correction_limited_by_safety_policy")
    if correction_rejected:
        warnings.append("correction_rejected_below_min_blend_factor")
    metadata = {
        "name": "safety_postprocessing",
        "status": "limited" if limited else "applied",
        "requested_blend_factor": float(required_factor),
        "applied_blend_factor": float(applied_factor),
        "limits": limits,
        "change_before_safety": before,
        "change_after_safety": after,
        "warnings": warnings,
    }
    return validate_rgb_uint8(output), metadata
