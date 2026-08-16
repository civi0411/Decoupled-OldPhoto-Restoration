from __future__ import annotations

import cv2
import numpy as np
import pytest

from old_photo_restoration.color_restoration.config import (
    ColorCorrectionConfig,
    ColorRestorationConfig,
    ContrastConfig,
    DenoiseConfig,
    SafetyConfig,
    SharpenConfig,
)
from old_photo_restoration.color_restoration.processing import (
    apply_conservative_color_correction,
    apply_safety_postprocessing,
    preprocess_after_lama,
    validate_rgb_uint8,
)


def make_gradient(height: int = 48, width: int = 64) -> np.ndarray:
    y = np.linspace(50, 200, height, dtype=np.float32)[:, None]
    x = np.linspace(40, 180, width, dtype=np.float32)[None, :]
    return np.clip(
        np.stack(
            [
                np.broadcast_to(x, (height, width)),
                np.broadcast_to(y, (height, width)),
                np.broadcast_to((x + y) / 2, (height, width)),
            ],
            axis=-1,
        ),
        0,
        255,
    ).astype(np.uint8)


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((8, 8), dtype=np.uint8),
        np.zeros((8, 8, 4), dtype=np.uint8),
        np.zeros((8, 8, 3), dtype=np.float32),
    ],
)
def test_invalid_input_contract_is_rejected(image: np.ndarray) -> None:
    with pytest.raises((TypeError, ValueError)):
        validate_rgb_uint8(image)


def test_preprocessing_preserves_shape_dtype_and_range() -> None:
    image = make_gradient()
    output, metadata = preprocess_after_lama(image, ColorRestorationConfig())

    assert output.shape == image.shape
    assert output.dtype == np.uint8
    assert int(output.min()) >= 0
    assert int(output.max()) <= 255
    assert metadata["contrast_applied"] is True


def test_default_quality_preprocessing_matches_original_contract() -> None:
    image = make_gradient()
    config = ColorRestorationConfig()
    denoised = cv2.fastNlMeansDenoisingColored(image, None, 3, 3, 7, 21)
    lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    enhanced_l = cv2.createCLAHE(clipLimit=1.35, tileGridSize=(8, 8)).apply(l_channel)
    contrast_restored = cv2.cvtColor(
        cv2.merge([enhanced_l, a_channel, b_channel]),
        cv2.COLOR_LAB2RGB,
    )
    blurred = cv2.GaussianBlur(contrast_restored, (0, 0), sigmaX=1.0)
    expected = cv2.addWeighted(contrast_restored, 1.12, blurred, -0.12, 0)

    output, metadata = preprocess_after_lama(image, config)

    assert np.array_equal(output, expected)
    assert metadata["backend"] == "opencv_conservative"
    assert metadata["sharpen_amount"] == 0.12


def test_color_correction_reduces_global_warm_cast() -> None:
    image = np.full((48, 64, 3), [175, 155, 125], dtype=np.uint8)
    config = ColorRestorationConfig(
        denoise=DenoiseConfig(enabled=False),
        contrast=ContrastConfig(enabled=False),
        color_correction=ColorCorrectionConfig(
            enabled=True,
            strength=0.5,
            neutral_saturation_max=100,
            min_neutral_pixels=64,
            max_chroma_shift=8.0,
        ),
        sharpen=SharpenConfig(enabled=False),
    )
    before_b = float(cv2.cvtColor(image, cv2.COLOR_RGB2LAB)[:, :, 2].mean())
    output, metadata = apply_conservative_color_correction(image, config)
    after_b = float(cv2.cvtColor(output, cv2.COLOR_RGB2LAB)[:, :, 2].mean())

    assert metadata["status"] == "applied"
    assert abs(after_b - 128.0) < abs(before_b - 128.0)


def test_safety_limits_large_candidate_change() -> None:
    original = np.zeros((32, 40, 3), dtype=np.uint8)
    candidate = np.full_like(original, 255)
    config = ColorRestorationConfig(
        sharpen=SharpenConfig(enabled=False),
        safety=SafetyConfig(
            enabled=True,
            max_mean_pixel_change=20.0,
            max_mean_luminance_change=20.0,
            max_mean_chroma_change=20.0,
            min_blend_factor=0.01,
        ),
    )
    output, metadata = apply_safety_postprocessing(original, candidate, config)

    assert metadata["status"] == "limited"
    assert metadata["applied_blend_factor"] < 1.0
    assert float(output.mean()) <= 21.0
    assert metadata["change_after_safety"]["mean_pixel_change"] <= 20.0
    assert metadata["change_after_safety"]["mean_luminance_change"] <= 20.0
