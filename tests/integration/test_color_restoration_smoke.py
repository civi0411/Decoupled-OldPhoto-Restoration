from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from old_photo_restoration.color_restoration import (
    ColorRestorationContext,
    load_color_restoration_config,
    restore_color_after_lama,
)
from old_photo_restoration.color_restoration.io import (
    load_rgb_image,
    save_color_restoration_result,
)


STAGING_ROOT = Path(__file__).resolve().parents[2]


def test_end_to_end_smoke_and_artifact_io(tmp_path: Path) -> None:
    image = np.full((36, 52, 3), [155, 140, 115], dtype=np.uint8)
    config = load_color_restoration_config(STAGING_ROOT / "configs" / "color_restoration.yaml")
    config = replace(config, method="opencv_conservative")
    result = restore_color_after_lama(
        image,
        config,
        context=ColorRestorationContext(source_id="synthetic-smoke", inpainting_backend="test"),
    )

    assert result.image_rgb.shape == image.shape
    assert result.image_rgb.dtype == np.uint8
    json.dumps(result.metadata)
    assert result.metadata["pipeline_order"] == [
        "quality_restoration",
        "color_restoration_model",
        "inference_control",
        "ccm_color_correction",
        "safety_postprocessing",
    ]
    assert [stage["name"] for stage in result.metadata["stages"]] == [
        "quality_restoration",
        "color_restoration",
        "inference_control",
        "ccm_color_correction",
        "safety_postprocessing",
    ]
    assert result.metadata["stages"][3]["status"] == "applied"
    assert set(result.intermediates) == {
        "quality_restored",
        "model_restored",
        "inference_controlled",
        "ccm_corrected",
    }

    artifacts = save_color_restoration_result(result, tmp_path, config)
    assert artifacts["image"].name == "color_restored.png"
    assert artifacts["metadata"].name == "color_restoration_metadata.json"
    assert artifacts["image"].exists()
    assert artifacts["metadata"].exists()
    assert load_rgb_image(artifacts["image"]).shape == image.shape
    payload = json.loads(artifacts["metadata"].read_text(encoding="utf-8"))
    assert payload["feature"] == "color_restoration"
    assert payload["output_contract"]["resolution_preserved"] is True
    assert cv2.imread(str(artifacts["image"]), cv2.IMREAD_COLOR) is not None
