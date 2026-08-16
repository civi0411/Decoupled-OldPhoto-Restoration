from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .config import ModelConfig
from .model import ColorRestorationUNet, normalize_color_model_config
from .processing import validate_rgb_uint8


def resolve_torch_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(device)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_color_checkpoint(
    model_config: ModelConfig,
) -> tuple[ColorRestorationUNet, dict[str, Any], torch.device, dict[str, Any]]:
    if not model_config.checkpoint_path:
        raise ValueError("model.checkpoint_path is required when method='model'")
    path = Path(model_config.checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Color checkpoint not found: {path}")

    actual_sha256 = sha256_file(path)
    if model_config.expected_sha256 and actual_sha256 != model_config.expected_sha256.lower():
        raise ValueError(
            f"Color checkpoint SHA256 mismatch: expected {model_config.expected_sha256}, "
            f"got {actual_sha256}"
        )

    device = resolve_torch_device(model_config.device)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if "model_state_dict" not in checkpoint or "model_config" not in checkpoint:
        raise ValueError(f"Invalid color checkpoint contract: {path}")
    architecture = normalize_color_model_config(checkpoint["model_config"])
    model = ColorRestorationUNet(**architecture).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    metadata = {
        "checkpoint": str(path),
        "checkpoint_sha256": actual_sha256,
        "checkpoint_format_version": checkpoint.get("checkpoint_format_version"),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "dataset_id": checkpoint.get("dataset_id"),
        "input_profile": checkpoint.get("input_profile"),
    }
    return model, checkpoint, device, metadata


def _normalized_to_rgb01(image: torch.Tensor) -> torch.Tensor:
    return torch.clamp((image + 1.0) * 0.5, 0.0, 1.0)


def _rgb01_to_lab_float32(image: torch.Tensor) -> torch.Tensor:
    try:
        import kornia
    except ImportError as exc:
        raise ImportError("Model color restoration requires kornia") from exc
    with torch.autocast(device_type=image.device.type, enabled=False):
        return kornia.color.rgb_to_lab(image.float())


def reconstruct_lab_ab(
    input_rgb_normalized: torch.Tensor,
    predicted_ab_normalized: torch.Tensor,
) -> torch.Tensor:
    import kornia

    with torch.autocast(device_type=input_rgb_normalized.device.type, enabled=False):
        input_lab = _rgb01_to_lab_float32(_normalized_to_rgb01(input_rgb_normalized))
        predicted_ab = torch.clamp(predicted_ab_normalized.float(), -1.0, 1.0) * 128.0
        output_lab = torch.cat([input_lab[:, :1], predicted_ab], dim=1)
        output_rgb = torch.clamp(kornia.color.lab_to_rgb(output_lab), 0.0, 1.0)
        return output_rgb * 2.0 - 1.0


def reconstruct_lab_residual(
    input_rgb_normalized: torch.Tensor,
    predicted_lab_residual: torch.Tensor,
    *,
    max_l_shift: float,
    max_ab_shift: float,
) -> torch.Tensor:
    if predicted_lab_residual.shape[1] != 3:
        raise ValueError(
            f"Lab residual prediction must have 3 channels, got {predicted_lab_residual.shape}"
        )
    import kornia

    with torch.autocast(device_type=input_rgb_normalized.device.type, enabled=False):
        input_lab = _rgb01_to_lab_float32(_normalized_to_rgb01(input_rgb_normalized))
        residual = torch.clamp(predicted_lab_residual.float(), -1.0, 1.0)
        output_l = torch.clamp(input_lab[:, :1] + residual[:, :1] * max_l_shift, 0.0, 100.0)
        output_ab = torch.clamp(
            input_lab[:, 1:] + residual[:, 1:] * max_ab_shift,
            -128.0,
            127.0,
        )
        output_rgb = torch.clamp(
            kornia.color.lab_to_rgb(torch.cat([output_l, output_ab], dim=1)),
            0.0,
            1.0,
        )
        return output_rgb * 2.0 - 1.0


def model_rgb_output(model: ColorRestorationUNet, input_tensor: torch.Tensor) -> torch.Tensor:
    prediction = model(input_tensor)
    if model.mode == "lab_ab":
        return reconstruct_lab_ab(input_tensor, prediction)
    if model.mode == "lab_residual":
        return reconstruct_lab_residual(
            input_tensor,
            prediction,
            max_l_shift=model.lab_l_shift,
            max_ab_shift=model.lab_ab_shift,
        )
    return prediction


def _padding_mode(height: int, width: int, pad_bottom: int, pad_right: int) -> str:
    if min(height, width) <= 1 or pad_bottom >= height or pad_right >= width:
        return "replicate"
    return "reflect"


def _coverage_size(length: int, tile_size: int, stride: int) -> int:
    if length <= tile_size:
        return tile_size
    steps = (length - tile_size + stride - 1) // stride
    return tile_size + steps * stride


def tiled_color_inference(
    image_rgb: np.ndarray,
    model: ColorRestorationUNet,
    device: torch.device,
    *,
    tile_size: int,
    overlap: int,
    tile_batch_size: int,
) -> np.ndarray:
    image = validate_rgb_uint8(image_rgb)
    height, width = image.shape[:2]
    stride = tile_size - overlap
    padded_height = _coverage_size(height, tile_size, stride)
    padded_width = _coverage_size(width, tile_size, stride)
    pad_bottom = padded_height - height
    pad_right = padded_width - width

    input_tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()
    input_tensor = (input_tensor / 127.5 - 1.0).unsqueeze(0)
    input_tensor = F.pad(
        input_tensor,
        (0, pad_right, 0, pad_bottom),
        mode=_padding_mode(height, width, pad_bottom, pad_right),
    )

    window_1d = torch.hann_window(tile_size, periodic=False).clamp_min(0.05)
    window = (window_1d[:, None] * window_1d[None, :]).unsqueeze(0).unsqueeze(0)
    accumulator = torch.zeros((1, 3, padded_height, padded_width), dtype=torch.float32)
    weights = torch.zeros((1, 1, padded_height, padded_width), dtype=torch.float32)
    positions = [
        (top, left)
        for top in range(0, padded_height - tile_size + 1, stride)
        for left in range(0, padded_width - tile_size + 1, stride)
    ]

    with torch.inference_mode():
        for start in range(0, len(positions), tile_batch_size):
            batch_positions = positions[start : start + tile_batch_size]
            patches = torch.cat(
                [
                    input_tensor[:, :, top : top + tile_size, left : left + tile_size]
                    for top, left in batch_positions
                ],
                dim=0,
            ).to(device)
            predictions = model_rgb_output(model, patches).detach().cpu()
            for prediction, (top, left) in zip(predictions, batch_positions):
                accumulator[:, :, top : top + tile_size, left : left + tile_size] += (
                    prediction.unsqueeze(0) * window
                )
                weights[:, :, top : top + tile_size, left : left + tile_size] += window

    output = accumulator / weights.clamp_min(1e-6)
    output = output[:, :, :height, :width].squeeze(0)
    output = ((output.clamp(-1.0, 1.0) + 1.0) * 127.5).permute(1, 2, 0).numpy()
    return validate_rgb_uint8(np.clip(output, 0, 255).astype(np.uint8))


def run_model_restoration(
    image_rgb: np.ndarray,
    config: ModelConfig,
    *,
    runtime_quality_mode: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    model, checkpoint, device, checkpoint_metadata = load_color_checkpoint(config)
    trained_quality_mode = checkpoint.get("dataset_config", {}).get("quality_mode")
    if trained_quality_mode and trained_quality_mode != config.required_quality_mode:
        raise ValueError(
            f"Configured required_quality_mode={config.required_quality_mode!r} does not match "
            f"checkpoint quality_mode={trained_quality_mode!r}"
        )
    if trained_quality_mode and runtime_quality_mode != trained_quality_mode:
        raise ValueError(
            f"Color checkpoint expects quality_mode={trained_quality_mode!r}, "
            f"but runtime uses {runtime_quality_mode!r}"
        )
    restored = tiled_color_inference(
        image_rgb,
        model,
        device,
        tile_size=config.tile_size,
        overlap=config.overlap,
        tile_batch_size=config.tile_batch_size,
    )
    return restored, {
        "name": "color_restoration_model",
        "status": "applied",
        "backend": "color_restoration_unet",
        **checkpoint_metadata,
        "model_mode": model.mode,
        "model_config": model.get_config(),
        "device": str(device),
        "tile_size": config.tile_size,
        "overlap": config.overlap,
        "tile_batch_size": config.tile_batch_size,
        "trained_quality_mode": trained_quality_mode,
        "runtime_quality_mode": runtime_quality_mode,
    }
