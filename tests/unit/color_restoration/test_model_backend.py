from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from old_photo_restoration.color_restoration.config import ModelConfig
from old_photo_restoration.color_restoration.model import ColorRestorationUNet
from old_photo_restoration.color_restoration.model_backend import (
    reconstruct_lab_residual,
    run_model_restoration,
)


def test_lab_residual_zero_prediction_preserves_input() -> None:
    pytest.importorskip("kornia")
    inputs = torch.rand(1, 3, 16, 16) * 1.6 - 0.8

    output = reconstruct_lab_residual(
        inputs,
        torch.zeros_like(inputs),
        max_l_shift=15.0,
        max_ab_shift=40.0,
    )

    assert output.shape == inputs.shape
    assert torch.allclose(output, inputs, atol=2e-3)


def test_model_forward_contract() -> None:
    model = ColorRestorationUNet(mode="lab_residual", base_channels=2).eval()
    output = model(torch.zeros(1, 3, 32, 32))

    assert output.shape == (1, 3, 32, 32)
    assert torch.isfinite(output).all()


def test_tiny_checkpoint_model_inference(tmp_path: Path) -> None:
    pytest.importorskip("kornia")
    model = ColorRestorationUNet(mode="lab_residual", base_channels=2).eval()
    checkpoint_path = tmp_path / "color.pth"
    torch.save(
        {
            "checkpoint_format_version": 2,
            "model_state_dict": model.state_dict(),
            "model_config": model.get_config(),
            "dataset_config": {"quality_mode": "opencv_conservative"},
            "dataset_id": "synthetic-test",
        },
        checkpoint_path,
    )
    image = np.full((37, 45, 3), 128, dtype=np.uint8)

    output, metadata = run_model_restoration(
        image,
        ModelConfig(
            checkpoint_path=str(checkpoint_path),
            device="cpu",
            tile_size=32,
            overlap=8,
            tile_batch_size=2,
        ),
        runtime_quality_mode="opencv_conservative",
    )

    assert output.shape == image.shape
    assert output.dtype == np.uint8
    assert metadata["backend"] == "color_restoration_unet"
    assert metadata["model_mode"] == "lab_residual"


def test_model_rejects_quality_mode_mismatch(tmp_path: Path) -> None:
    model = ColorRestorationUNet(mode="lab_residual", base_channels=2).eval()
    checkpoint_path = tmp_path / "color.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model.get_config(),
            "dataset_config": {"quality_mode": "opencv_conservative"},
        },
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="expects quality_mode"):
        run_model_restoration(
            np.full((32, 32, 3), 128, dtype=np.uint8),
            ModelConfig(checkpoint_path=str(checkpoint_path), device="cpu", tile_size=32),
            runtime_quality_mode="custom",
        )
