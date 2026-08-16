from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import ColorRestorationConfig
from .contracts import ColorRestorationResult
from .processing import validate_rgb_uint8


def load_rgb_image(path: str | Path) -> np.ndarray:
    input_path = Path(path)
    image_bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read RGB image: {input_path}")
    return validate_rgb_uint8(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))


def save_rgb_image(path: str | Path, image_rgb: np.ndarray) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = validate_rgb_uint8(image_rgb)
    ok = cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError(f"Cannot write RGB image: {output_path}")
    return output_path


def save_color_restoration_result(
    result: ColorRestorationResult,
    output_dir: str | Path,
    config: ColorRestorationConfig,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    image_path = save_rgb_image(output_path / config.output.image_filename, result.image_rgb)

    intermediate_paths: dict[str, str] = {}
    if config.output.save_intermediates:
        for name, image in result.intermediates.items():
            saved = save_rgb_image(output_path / f"{name}.png", image)
            intermediate_paths[name] = str(saved)

    metadata_path = output_path / config.output.metadata_filename
    metadata_payload = {
        **result.metadata,
        "artifacts": {
            "color_restored": str(image_path),
            "metadata": str(metadata_path),
            "intermediates": intermediate_paths,
        },
    }
    metadata_path.write_text(
        json.dumps(metadata_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "image": image_path,
        "metadata": metadata_path,
        "intermediates": {name: Path(path) for name, path in intermediate_paths.items()},
    }
