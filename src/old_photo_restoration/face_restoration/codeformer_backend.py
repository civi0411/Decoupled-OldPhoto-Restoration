from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from old_photo_restoration.utils.image_io import read_image_rgb, write_image_rgb

from .config import FaceRestorationConfig


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _validate_rgb_uint8(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy array")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must have shape HxWx3, got {image.shape}")
    if image.dtype != np.uint8:
        raise TypeError(f"image dtype must be uint8, got {image.dtype}")
    return image


def _tail(value: str, limit: int = 2000) -> str:
    return value[-limit:] if value else ""


def _select_output(output_dir: Path, input_stem: str) -> Path | None:
    final_dir = output_dir / "final_results"
    search_root = final_dir if final_dir.is_dir() else output_dir
    images = [
        path
        for path in search_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    exact = [path for path in images if path.stem == input_stem]
    candidates = exact or images
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def _pass_through(
    image: np.ndarray,
    reason: str,
    warning: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    return image.copy(), {
        "name": "codeformer",
        "status": "skipped",
        "backend": "codeformer",
        "reason": reason,
        "warnings": [warning] if warning else [],
    }


def run_codeformer_restoration(
    image_rgb: np.ndarray,
    config: FaceRestorationConfig,
    work_dir: str | Path | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    config.validate()
    image = _validate_rgb_uint8(image_rgb)
    if not config.enabled:
        return _pass_through(image, "disabled")
    if not config.repo_path:
        if config.required:
            raise ValueError("repo_path is required when CodeFormer is required")
        return _pass_through(image, "repo_not_configured", "codeformer_repo_not_configured")

    repo_path = Path(config.repo_path).expanduser().resolve()
    inference_script = repo_path / "inference_codeformer.py"
    checkpoint_path = repo_path / "weights" / "CodeFormer" / "codeformer.pth"
    if not inference_script.is_file():
        message = f"CodeFormer inference script not found: {inference_script}"
        if config.required:
            raise FileNotFoundError(message)
        return _pass_through(image, "inference_script_missing", message)

    temporary: tempfile.TemporaryDirectory[str] | None = None
    if work_dir is None:
        temporary = tempfile.TemporaryDirectory(prefix="face-restoration-codeformer-")
        runtime_dir = Path(temporary.name)
    else:
        runtime_dir = Path(work_dir)
        runtime_dir.mkdir(parents=True, exist_ok=True)
    input_path = runtime_dir / "codeformer_input.png"
    write_image_rgb(input_path, image)
    raw_output_dir = runtime_dir / "codeformer_raw"
    raw_output_dir.mkdir(parents=True, exist_ok=True)

    python_command = (
        ["conda", "run", "-n", config.env_name, "python"]
        if config.env_name
        else [sys.executable]
    )
    command = [
        *python_command,
        str(inference_script),
        "--input_path",
        str(input_path.resolve()),
        "--output_path",
        str(raw_output_dir.resolve()),
        "--fidelity_weight",
        str(config.fidelity_weight),
        "--upscale",
        str(config.upscale),
    ]
    if config.face_upsample:
        command.append("--face_upsample")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        item for item in [str(repo_path), environment.get("PYTHONPATH", "")] if item
    )
    try:
        result = subprocess.run(
            command,
            cwd=repo_path,
            env=environment,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=config.timeout_sec,
            check=False,
        )
        selected = _select_output(raw_output_dir, input_path.stem) if result.returncode == 0 else None
        if selected is None:
            message = _tail(result.stderr) or "CodeFormer output not found"
            if config.required:
                raise RuntimeError(message)
            return _pass_through(image, "subprocess_failed", message)
        restored = read_image_rgb(selected)
        resized = False
        if restored.shape[:2] != image.shape[:2]:
            restored = cv2.resize(
                restored,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_LANCZOS4,
            )
            resized = True
        return _validate_rgb_uint8(restored), {
            "name": "codeformer",
            "status": "applied",
            "backend": "codeformer",
            "reason": "applied",
            "repo_path": str(repo_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_exists": checkpoint_path.is_file(),
            "env_name": config.env_name,
            "fidelity_weight": config.fidelity_weight,
            "upscale": config.upscale,
            "face_upsample": config.face_upsample,
            "resolution_restored_to_input": resized,
            "output": str(selected) if work_dir is not None else None,
            "output_persisted": work_dir is not None,
            "returncode": result.returncode,
            "stdout_tail": _tail(result.stdout),
            "stderr_tail": _tail(result.stderr),
            "warnings": [],
        }
    finally:
        if temporary is not None:
            temporary.cleanup()
