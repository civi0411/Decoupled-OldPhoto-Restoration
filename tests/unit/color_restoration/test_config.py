from __future__ import annotations

from pathlib import Path

import pytest

from old_photo_restoration.color_restoration.config import (
    ColorRestorationConfig,
    color_restoration_config_from_dict,
    load_color_restoration_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_package_imports_without_loading_opencv_backend() -> None:
    import old_photo_restoration.color_restoration as package

    assert package.ColorRestorationConfig is ColorRestorationConfig


def test_default_staging_config_loads() -> None:
    config = load_color_restoration_config(PROJECT_ROOT / "configs" / "color_restoration.yaml")

    assert isinstance(config, ColorRestorationConfig)
    assert config.method == "model"
    assert config.model.checkpoint_path
    assert config.model.expected_sha256
    assert config.inference_control.mode == "lab_protected_fixed"
    assert config.final_color.method == "ccm"
    assert config.safety.enabled is False
    assert config.output.image_filename == "color_restored.png"


def test_invalid_strength_is_rejected() -> None:
    with pytest.raises(ValueError, match="strength"):
        color_restoration_config_from_dict({"color_correction": {"strength": 2.0}})
