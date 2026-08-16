from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone
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
from old_photo_restoration.utils.batch_output import BatchOutput, unique_item_ids
from old_photo_restoration.utils.metadata import save_metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one or more images into a batch output with one item directory per input."
    )
    parser.add_argument("--image", required=True, type=Path, nargs="+")
    parser.add_argument("--mask", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/inference.yaml"))
    parser.add_argument("--checkpoint-config", type=Path, default=Path("configs/checkpoints.yaml"))
    parser.add_argument("--external-config", type=Path, default=Path("configs/external_paths.yaml"))
    parser.add_argument("--face-mode", choices=["off", "auto"], default="off")
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--reference-mask", type=Path, default=None)
    parser.add_argument("--segmenter-arch", choices=["r013_custom_attnunet", "r014_resnet34"], default="r013_custom_attnunet")
    parser.add_argument("--segmenter-checkpoint", type=Path, default=None)
    parser.add_argument("--segmenter-threshold", type=float, default=None)
    parser.add_argument("--segmenter-dilation", type=int, default=None)
    parser.add_argument(
        "--post-inpainting",
        action="store_true",
        help="Run color restoration with CCM, then optional face restoration, after LaMa.",
    )
    parser.add_argument(
        "--color-restoration-config",
        type=Path,
        default=Path("configs/color_restoration.yaml"),
    )
    return parser


def resolve_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def read_rgb(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
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
        raise FileNotFoundError(f"Could not read mask: {path}")
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
    image_paths = [resolve_path(path) for path in args.image]
    if any(path is None for path in image_paths):
        raise ValueError("All image paths must be provided.")
    resolved_images = [path for path in image_paths if path is not None]
    if len(resolved_images) > 1 and any(
        value is not None for value in (args.mask, args.reference, args.reference_mask)
    ):
        raise ValueError("--mask, --reference, and --reference-mask only support a single image.")
    mask_path = resolve_path(args.mask)
    batch_dir = resolve_path(args.output_dir)
    assert batch_dir is not None
    reference_path = resolve_path(args.reference)
    reference_mask_path = resolve_path(args.reference_mask)

    config = load_config(
        inference_path=resolve_path(args.config),
        checkpoint_path=resolve_path(args.checkpoint_config),
        external_path=resolve_path(args.external_config),
    )
    pipeline = RestorationPipeline(config)
    batch = BatchOutput.create(batch_dir)
    item_ids = unique_item_ids(resolved_images, batch.items_dir)
    batch_items: list[dict[str, Any]] = []

    for image_path, item_id in zip(resolved_images, item_ids):
        item_dir = batch.item_dir(item_id)
        input_dir = item_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        copied_input = input_dir / f"original{image_path.suffix.lower()}"
        shutil.copy2(image_path, copied_input)
        started_at = datetime.now(timezone.utc).isoformat()
        try:
            result = pipeline.run(
                image_path=copied_input,
                output_dir=item_dir / "artifacts",
                mask_path=mask_path,
                face_mode=args.face_mode,
                golden_reference=reference_path,
                segmenter_arch=args.segmenter_arch,
                segmenter_checkpoint=args.segmenter_checkpoint,
                segmenter_threshold=args.segmenter_threshold,
                segmenter_dilation=args.segmenter_dilation,
                post_inpainting_enabled=args.post_inpainting,
                color_restoration_config_path=resolve_path(args.color_restoration_config),
            )
            metadata = dict(result.metadata)
            comparison = compare_images(result.restored_path, reference_path) if reference_path else None
            mask_comparison = compare_masks(result.mask_path, reference_mask_path) if reference_mask_path else None
            if comparison is not None:
                metadata["comparison"] = comparison
            if mask_comparison is not None:
                metadata["mask_comparison"] = mask_comparison
            if comparison is not None or mask_comparison is not None:
                save_metadata(result.output_dir / "metadata.json", metadata)

            final_path = item_dir / "final.png"
            shutil.copy2(result.restored_path, final_path)
            item_manifest = {
                "schema_version": 1,
                "batch_id": batch.batch_id,
                "item_id": item_id,
                "status": "completed",
                "started_at": started_at,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "source_input": str(image_path),
                "input": str(copied_input),
                "artifacts_dir": str(result.output_dir),
                "final_output": str(final_path),
                "final_mask": str(result.mask_path),
                "pipeline_metadata": str(result.output_dir / "metadata.json"),
            }
            save_metadata(item_dir / "manifest.json", item_manifest)
            batch_items.append(item_manifest)
            print(f"[completed] {item_id}: {final_path}")
        except Exception as exc:
            item_manifest = {
                "schema_version": 1,
                "batch_id": batch.batch_id,
                "item_id": item_id,
                "status": "failed",
                "started_at": started_at,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "source_input": str(image_path),
                "input": str(copied_input),
                "error": str(exc),
            }
            save_metadata(item_dir / "manifest.json", item_manifest)
            batch_items.append(item_manifest)
            print(f"[failed] {item_id}: {exc}")

    batch.write_manifest(batch_items)
    print(f"batch_manifest: {batch.manifest_path}")
    return 1 if any(item["status"] == "failed" for item in batch_items) else 0


if __name__ == "__main__":
    raise SystemExit(main())
