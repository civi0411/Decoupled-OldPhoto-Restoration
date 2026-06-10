from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from old_photo_restoration.config import load_config
from old_photo_restoration.evaluation.metrics import compute_iou_binary, compute_mae_image, compute_psnr
from old_photo_restoration.pipeline import RestorationPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chạy pipeline submission với mask-bypass hoặc auto-mask r013.")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--mask", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/inference.yaml"))
    parser.add_argument("--checkpoint-config", type=Path, default=Path("configs/checkpoints.yaml"))
    parser.add_argument("--external-config", type=Path, default=Path("configs/external_paths.yaml"))
    parser.add_argument("--face-mode", choices=["off", "auto"], default="off")
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--reference-mask", type=Path, default=None)
    return parser


def resolve_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def read_rgb(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def compare_images(output_path: Path, reference_path: Path) -> dict[str, Any]:
    output = read_rgb(output_path)
    reference = read_rgb(reference_path)
    same_size = output.shape == reference.shape
    report: dict[str, Any] = {
        "same_size": bool(same_size),
        "output_shape": list(output.shape),
        "reference_shape": list(reference.shape),
        "mae": None,
        "max_absolute_error": None,
        "psnr": None,
    }
    if not same_size:
        return report

    diff = np.abs(output.astype(np.float32) - reference.astype(np.float32))
    report["mae"] = compute_mae_image(output, reference)
    report["max_absolute_error"] = int(diff.max())
    report["psnr"] = compute_psnr(output, reference)
    return report


def read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Không đọc được mask: {path}")
    if mask.ndim == 3:
        if mask.shape[2] == 4:
            mask = mask[:, :, :3]
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return np.where(mask > 127, 255, 0).astype(np.uint8)


def compare_masks(output_path: Path, reference_path: Path) -> dict[str, Any]:
    output = read_mask(output_path)
    reference = read_mask(reference_path)
    same_size = output.shape == reference.shape
    report: dict[str, Any] = {
        "same_size": bool(same_size),
        "output_shape": list(output.shape),
        "reference_shape": list(reference.shape),
        "iou": None,
        "mae": None,
        "max_absolute_error": None,
    }
    if not same_size:
        return report

    diff = np.abs(output.astype(np.float32) - reference.astype(np.float32))
    report["iou"] = compute_iou_binary(output, reference)
    report["mae"] = float(diff.mean())
    report["max_absolute_error"] = int(diff.max())
    return report


def main() -> int:
    args = build_parser().parse_args()
    image_path = resolve_path(args.image)
    mask_path = resolve_path(args.mask)
    output_dir = resolve_path(args.output_dir)
    reference_path = resolve_path(args.reference)
    reference_mask_path = resolve_path(args.reference_mask)

    config = load_config(
        inference_path=resolve_path(args.config),
        checkpoint_path=resolve_path(args.checkpoint_config),
        external_path=resolve_path(args.external_config),
    )
    pipeline = RestorationPipeline(config)

    result = pipeline.run(
        image_path=image_path,
        output_dir=output_dir,
        mask_path=mask_path,
        face_mode=args.face_mode,
        golden_reference=reference_path,
    )

    metadata_path = result.output_dir / "metadata.json"
    metadata = dict(result.metadata)
    comparison: dict[str, Any] | None = None
    mask_comparison: dict[str, Any] | None = None
    if reference_path is not None:
        comparison = compare_images(result.restored_path, reference_path)
        metadata["comparison"] = comparison
    if reference_mask_path is not None:
        mask_comparison = compare_masks(result.mask_path, reference_mask_path)
        metadata["mask_comparison"] = mask_comparison
    if comparison is not None or mask_comparison is not None:
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"restored_before_face: {result.restored_path}")
    print(f"final_mask: {result.mask_path}")
    print(f"metadata: {metadata_path}")
    if comparison is not None:
        print(f"same_size: {comparison['same_size']}")
        print(f"mae: {comparison['mae']}")
        print(f"max_absolute_error: {comparison['max_absolute_error']}")
        print(f"psnr: {comparison['psnr']}")
    if mask_comparison is not None:
        print(f"mask_same_size: {mask_comparison['same_size']}")
        print(f"mask_iou: {mask_comparison['iou']}")
        print(f"mask_mae: {mask_comparison['mae']}")
        print(f"mask_max_absolute_error: {mask_comparison['max_absolute_error']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
