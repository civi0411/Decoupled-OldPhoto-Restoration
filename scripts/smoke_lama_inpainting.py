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
from old_photo_restoration.inpainting.lama_wrapper import LamaInpainter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke test official LaMa bằng golden mask.")
    parser.add_argument("--image", type=Path, default=Path("examples/inputs/demo3.png"))
    parser.add_argument("--mask", type=Path, default=Path("examples/golden/demo3_r013_repair_wide/final_mask.png"))
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("examples/golden/demo3_r013_repair_wide/restored_before_face.png"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("examples/outputs/lama_smoke_demo3"))
    parser.add_argument("--config", type=Path, default=Path("configs/inference.yaml"))
    parser.add_argument("--external-config", type=Path, default=Path("configs/external_paths.yaml"))
    parser.add_argument("--checkpoint-config", type=Path, default=Path("configs/checkpoints.yaml"))
    return parser


def resolve_path(path: Path) -> Path:
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
    mae = float(diff.mean())
    mse = float(np.mean(np.square(output.astype(np.float32) - reference.astype(np.float32))))
    psnr = float("inf") if mse == 0.0 else float(20.0 * np.log10(255.0) - 10.0 * np.log10(mse))
    report["mae"] = mae
    report["max_absolute_error"] = int(diff.max())
    report["psnr"] = psnr
    return report


def main() -> int:
    args = build_parser().parse_args()
    image_path = resolve_path(args.image)
    mask_path = resolve_path(args.mask)
    reference_path = resolve_path(args.reference)
    output_dir = resolve_path(args.output_dir)

    config = load_config(
        inference_path=resolve_path(args.config),
        checkpoint_path=resolve_path(args.checkpoint_config),
        external_path=resolve_path(args.external_config),
    )
    inpainter = LamaInpainter(config.lama)
    readiness = inpainter.readiness()
    print(json.dumps(readiness, ensure_ascii=False, indent=2))

    output_dir.mkdir(parents=True, exist_ok=True)
    restored_path = inpainter.inpaint(image_path=image_path, mask_path=mask_path, output_dir=output_dir)
    if not restored_path.exists():
        raise FileNotFoundError(f"LaMa không tạo output mong đợi: {restored_path}")

    comparison: dict[str, Any] | None = None
    if reference_path.exists():
        comparison = compare_images(restored_path, reference_path)

    report = {
        "image": str(image_path),
        "mask": str(mask_path),
        "reference": str(reference_path) if reference_path.exists() else None,
        "restored_output": str(restored_path),
        "readiness": readiness,
        "inpaint_result": inpainter.last_result,
        "comparison": comparison,
        "config_warnings": config.warnings,
    }
    report_path = output_dir / "lama_smoke_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"restored_output: {restored_path}")
    print(f"report: {report_path}")
    if comparison is not None:
        print(f"same_size: {comparison['same_size']}")
        print(f"mae: {comparison['mae']}")
        print(f"max_absolute_error: {comparison['max_absolute_error']}")
        print(f"psnr: {comparison['psnr']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
