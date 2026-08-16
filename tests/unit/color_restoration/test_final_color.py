from __future__ import annotations

import cv2
import numpy as np

from old_photo_restoration.color_restoration.config import FinalColorConfig
from old_photo_restoration.color_restoration.final_color import (
    apply_final_chroma_matching,
    apply_final_color_correction,
    estimate_conservative_ccm,
)


def test_conservative_ccm_is_diagonal_and_bounded() -> None:
    image = np.full((32, 40, 3), [170, 145, 110], dtype=np.uint8)
    config = FinalColorConfig(ccm_strength=0.45, ccm_max_gain_delta=0.15)

    matrix, metadata = estimate_conservative_ccm(image, config)

    assert matrix.shape == (3, 3)
    assert np.count_nonzero(matrix - np.diag(np.diag(matrix))) == 0
    assert np.all(np.diag(matrix) >= 1.0 - 0.15 * 0.45)
    assert np.all(np.diag(matrix) <= 1.0 + 0.15 * 0.45)
    assert metadata["method"] == "robust_gray_world_diagonal_ccm"


def test_final_ccm_preserves_contract_and_changes_cast() -> None:
    image = np.full((32, 40, 3), [170, 145, 110], dtype=np.uint8)

    output, metadata = apply_final_color_correction(image, image, FinalColorConfig(method="ccm"))

    assert output.shape == image.shape
    assert output.dtype == np.uint8
    assert not np.array_equal(output, image)
    assert metadata["backend"] == "ccm"
    assert len(metadata["ccm"]["matrix"]) == 3


def test_final_chroma_matching_preserves_luminance_approximately() -> None:
    image = np.full((30, 36, 3), [130, 130, 160], dtype=np.uint8)
    reference = np.full((30, 36, 3), [165, 135, 105], dtype=np.uint8)
    before_l = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)[:, :, 0].astype(np.float32)

    output, metadata = apply_final_chroma_matching(
        image,
        reference,
        strength=0.35,
        max_shift=8.0,
    )
    after_l = cv2.cvtColor(output, cv2.COLOR_RGB2LAB)[:, :, 0].astype(np.float32)

    assert float(np.abs(after_l - before_l).mean()) <= 2.0
    assert abs(metadata["shift_a"]) <= 8.0
    assert abs(metadata["shift_b"]) <= 8.0
