from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .config import ColorRestorationConfig
from .contracts import ColorRestorationContext, ColorRestorationResult
from .processing import (
    apply_conservative_color_correction,
    apply_safety_postprocessing,
    preprocess_after_lama,
    validate_rgb_uint8,
)


SCHEMA_VERSION = 4
FEATURE_NAME = "color_restoration"


def _runtime_quality_mode(config: ColorRestorationConfig) -> str:
    if config.denoise.enabled and config.contrast.enabled and config.sharpen.enabled:
        return "opencv_conservative"
    return "custom"


def _build_metadata(
    *,
    config: ColorRestorationConfig,
    image_shape: tuple[int, ...],
    stages: list[dict[str, Any]],
    context: ColorRestorationContext | None,
    status: str,
) -> dict[str, Any]:
    warnings = [
        warning
        for stage in stages
        for warning in stage.get("warnings", [])
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "feature": FEATURE_NAME,
        "status": status,
        "method": config.method,
        "pipeline_order": [
            "quality_restoration",
            "color_restoration_model",
            "inference_control",
            "ccm_color_correction",
            "safety_postprocessing",
        ],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "input_contract": {
            "color_space": "RGB",
            "shape": list(image_shape),
            "dtype": "uint8",
            "range": [0, 255],
        },
        "output_contract": {
            "color_space": "RGB",
            "shape": list(image_shape),
            "dtype": "uint8",
            "range": [0, 255],
            "resolution_preserved": True,
        },
        "context": None if context is None else context.to_dict(),
        "stages": stages,
        "warnings": warnings,
    }


def restore_color_after_lama(
    image_rgb: np.ndarray,
    config: ColorRestorationConfig,
    context: ColorRestorationContext | None = None,
    runtime_dir: str | Path | None = None,
) -> ColorRestorationResult:
    """Restore color after inpainting and finish with CCM before face restoration."""
    config.validate()
    original = validate_rgb_uint8(image_rgb).copy()
    if not config.enabled:
        metadata = _build_metadata(
            config=config,
            image_shape=original.shape,
            stages=[{"name": "color_restoration", "status": "skipped", "reason": "disabled"}],
            context=context,
            status="skipped",
        )
        return ColorRestorationResult(image_rgb=original, metadata=metadata)

    quality_restored, quality_metadata = preprocess_after_lama(original, config)
    if config.method == "model":
        from .model_backend import run_model_restoration

        model_restored, restoration_metadata = run_model_restoration(
            quality_restored,
            config.model,
            runtime_quality_mode=_runtime_quality_mode(config),
        )
    elif config.method == "opencv_neutral_fallback":
        model_restored, restoration_metadata = apply_conservative_color_correction(
            quality_restored,
            config,
        )
    else:
        model_restored = quality_restored
        restoration_metadata = {
            "name": "color_restoration",
            "status": "skipped",
            "backend": "none",
            "reason": "opencv_quality_only",
        }

    if config.method == "model":
        from .inference_control import apply_inference_control

        inference_controlled, control_metadata = apply_inference_control(
            quality_restored,
            model_restored,
            config.inference_control,
        )
    else:
        inference_controlled = model_restored
        control_metadata = {
            "name": "inference_control",
            "status": "skipped",
            "backend": "none",
            "reason": "model_backend_not_used",
        }

    from .final_color import apply_final_color_correction

    final_color_restored, final_color_metadata = apply_final_color_correction(
        inference_controlled,
        quality_restored,
        config.final_color,
    )
    output, safety_metadata = apply_safety_postprocessing(original, final_color_restored, config)
    stages = [
        quality_metadata,
        restoration_metadata,
        control_metadata,
        final_color_metadata,
        safety_metadata,
    ]
    metadata = _build_metadata(
        config=config,
        image_shape=original.shape,
        stages=stages,
        context=context,
        status="applied",
    )
    intermediates = {}
    if config.output.save_intermediates:
        intermediates = {
            "quality_restored": quality_restored,
            "model_restored": model_restored,
            "inference_controlled": inference_controlled,
            "ccm_corrected": final_color_restored,
        }
    return ColorRestorationResult(
        image_rgb=output,
        metadata=metadata,
        intermediates=intermediates,
    )
