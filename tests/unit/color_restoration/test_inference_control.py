from __future__ import annotations

import cv2
import numpy as np

from old_photo_restoration.color_restoration.config import InferenceControlConfig
from old_photo_restoration.color_restoration.inference_control import (
    apply_inference_control,
    chroma_gate_strength,
    lab_protected_blend,
)


def test_lab_protected_blend_matches_original_inference_control() -> None:
    reference = np.full((24, 32, 3), [150, 135, 110], dtype=np.uint8)
    model = np.full((24, 32, 3), [175, 120, 95], dtype=np.uint8)
    reference_lab = cv2.cvtColor(reference, cv2.COLOR_RGB2LAB).astype(np.float32)
    model_lab = cv2.cvtColor(model, cv2.COLOR_RGB2LAB).astype(np.float32)
    expected_lab = reference_lab.copy()
    expected_lab[:, :, 0] += 0.35 * (model_lab[:, :, 0] - reference_lab[:, :, 0])
    expected_lab[:, :, 1:] += 0.65 * (model_lab[:, :, 1:] - reference_lab[:, :, 1:])
    expected = cv2.cvtColor(np.clip(expected_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)

    output = lab_protected_blend(reference, model, strength_l=0.35, strength_ab=0.65)

    assert np.array_equal(output, expected)


def test_original_chroma_gate_values_are_preserved() -> None:
    config = InferenceControlConfig()

    assert chroma_gate_strength(3.9, config) == 0.2
    assert chroma_gate_strength(7.9, config) == 0.4
    assert chroma_gate_strength(11.9, config) == 0.6
    assert chroma_gate_strength(12.0, config) == 0.7


def test_gated_control_preserves_contract() -> None:
    quality = np.full((20, 28, 3), 128, dtype=np.uint8)
    model = np.full((20, 28, 3), [180, 100, 100], dtype=np.uint8)

    output, metadata = apply_inference_control(
        quality,
        model,
        InferenceControlConfig(mode="lab_protected_gated"),
    )

    assert output.shape == quality.shape
    assert output.dtype == np.uint8
    assert metadata["chroma_gate_applied"] is True
    assert metadata["ab_strength"] == 0.2
