from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .config import InferenceControlConfig
from .processing import validate_rgb_uint8


def mean_lab_chroma(image_rgb: np.ndarray) -> float:
    image = validate_rgb_uint8(image_rgb)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    centered_ab = lab[:, :, 1:] - 128.0
    return float(np.sqrt(np.sum(centered_ab**2, axis=2)).mean())


def chroma_gate_strength(mean_chroma: float, config: InferenceControlConfig) -> float:
    for threshold, strength in zip(config.chroma_thresholds, config.chroma_strengths):
        if mean_chroma < threshold:
            return float(strength)
    return float(config.chroma_strengths[-1])


def rgb_blend(reference_rgb: np.ndarray, model_rgb: np.ndarray, strength: float) -> np.ndarray:
    reference = validate_rgb_uint8(reference_rgb)
    model = validate_rgb_uint8(model_rgb)
    if reference.shape != model.shape:
        raise ValueError(f"model shape {model.shape} must match reference shape {reference.shape}")
    strength = float(np.clip(strength, 0.0, 1.0))
    blended = reference.astype(np.float32) * (1.0 - strength) + model.astype(np.float32) * strength
    return validate_rgb_uint8(np.clip(blended, 0, 255).astype(np.uint8))


def lab_protected_blend(
    reference_rgb: np.ndarray,
    model_rgb: np.ndarray,
    *,
    strength_l: float,
    strength_ab: float,
) -> np.ndarray:
    reference = validate_rgb_uint8(reference_rgb)
    model = validate_rgb_uint8(model_rgb)
    if reference.shape != model.shape:
        raise ValueError(f"model shape {model.shape} must match reference shape {reference.shape}")
    reference_lab = cv2.cvtColor(reference, cv2.COLOR_RGB2LAB).astype(np.float32)
    model_lab = cv2.cvtColor(model, cv2.COLOR_RGB2LAB).astype(np.float32)
    output_lab = reference_lab.copy()
    output_lab[:, :, 0] += float(strength_l) * (
        model_lab[:, :, 0] - reference_lab[:, :, 0]
    )
    output_lab[:, :, 1:] += float(strength_ab) * (
        model_lab[:, :, 1:] - reference_lab[:, :, 1:]
    )
    output = cv2.cvtColor(np.clip(output_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    return validate_rgb_uint8(output)


def apply_inference_control(
    quality_rgb: np.ndarray,
    model_rgb: np.ndarray,
    config: InferenceControlConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    quality = validate_rgb_uint8(quality_rgb)
    model = validate_rgb_uint8(model_rgb)
    if not config.enabled:
        return model.copy(), {
            "name": "inference_control",
            "status": "skipped",
            "backend": "none",
            "reason": "disabled",
        }

    mean_chroma = mean_lab_chroma(quality)
    metadata: dict[str, Any] = {
        "name": "inference_control",
        "status": "applied",
        "backend": config.mode,
        "reference": "quality_restored",
        "mean_quality_chroma": mean_chroma,
    }
    if config.mode == "rgb_blend":
        output = rgb_blend(quality, model, config.rgb_strength)
        metadata["rgb_strength"] = config.rgb_strength
        return output, metadata

    if config.mode == "lab_protected_gated":
        ab_strength = chroma_gate_strength(mean_chroma, config)
        l_strength = min(config.l_strength, ab_strength)
        metadata["chroma_gate_applied"] = True
    else:
        l_strength = config.l_strength
        ab_strength = config.ab_strength
        metadata["chroma_gate_applied"] = False
    output = lab_protected_blend(
        quality,
        model,
        strength_l=l_strength,
        strength_ab=ab_strength,
    )
    metadata["l_strength"] = float(l_strength)
    metadata["ab_strength"] = float(ab_strength)
    return output, metadata
