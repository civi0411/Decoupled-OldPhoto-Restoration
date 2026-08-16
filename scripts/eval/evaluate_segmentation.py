from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
STEM_SUFFIXES = ("_mask", "-mask", "_pred", "_prediction", "_gt", "_label")


def configure_utf8_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


@dataclass(slots=True)
class PairItem:
    sample_id: str
    pred_path: Path
    gt_path: Path


@dataclass(slots=True)
class PixelMetrics:
    tp: int
    fp: int
    fn: int
    tn: int
    iou: float
    f1: float
    precision: float
    recall: float
    fpr: float
    pred_positive_ratio: float
    gt_positive_ratio: float


@dataclass(slots=True)
class DuplicateSampleIdReport:
    sample_id: str
    chosen_path: Path
    ignored_paths: list[Path]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Đánh giá segmentation masks có sẵn cho Module 1 bằng subcommand `masks`. "
            "Script này là mask-level evaluator, không phải full pipeline evaluator, "
            "không phải LPIPS/FID evaluator, và chưa hỗ trợ checkpoint inference."
        )
    )
    subparsers = parser.add_subparsers(dest="command")

    masks_parser = subparsers.add_parser(
        "masks",
        help="Đánh giá predicted masks có sẵn so với ground-truth masks.",
        description=(
            "Đánh giá pair predicted/ground-truth masks theo stem filename. "
            "Subcommand này yêu cầu --pred-mask-dir, --gt-mask-dir, --output-dir; "
            "hỗ trợ threshold cố định hoặc threshold sweep; "
            "không chạy full pipeline, không chạy checkpoint inference."
        ),
    )
    masks_parser.add_argument(
        "--pred-mask-dir",
        required=True,
        type=Path,
        help="Thư mục predicted masks đầu vào.",
    )
    masks_parser.add_argument(
        "--gt-mask-dir",
        required=True,
        type=Path,
        help="Thư mục ground-truth masks đầu vào.",
    )
    masks_parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Thư mục output bắt buộc để ghi metrics summary và CSV.",
    )
    masks_parser.add_argument(
        "--split-file",
        type=Path,
        default=None,
        help="File split tùy chọn; nếu truyền sẽ chỉ evaluate các sample có trong file này.",
    )
    masks_parser.add_argument(
        "--split-dir",
        type=Path,
        default=None,
        help="Thư mục chứa split file; dùng cùng --split-name nếu không truyền --split-file.",
    )
    masks_parser.add_argument(
        "--split-name",
        default="test.txt",
        help="Tên split file bên trong --split-dir. Mặc định: test.txt.",
    )
    masks_parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold nhị phân hóa predicted mask. Mặc định: 0.5.",
    )
    masks_parser.add_argument(
        "--sweep-thresholds",
        default="",
        help="Danh sách threshold cách nhau bởi dấu phẩy để sweep, ví dụ: 0.3,0.5,0.7.",
    )
    masks_parser.add_argument(
        "--pred-suffix",
        default="",
        help="Lọc predicted masks theo suffix của stem nếu cần.",
    )
    masks_parser.add_argument(
        "--gt-suffix",
        default="",
        help="Lọc ground-truth masks theo suffix của stem nếu cần.",
    )
    masks_parser.add_argument(
        "--allow-duplicate-sample-ids",
        action="store_true",
        help=(
            "Cho phép duplicate sample_id sau khi normalize stem. "
            "Script sẽ chọn file đầu tiên theo thứ tự sắp xếp và ghi warning vào summary."
        ),
    )
    masks_parser.set_defaults(handler=run_masks_evaluation)
    parser.epilog = (
        "Dùng `evaluate_segmentation.py masks --help` để xem đầy đủ các cờ bắt buộc và tùy chọn của subcommand."
    )
    return parser


def ensure_dir_exists(path: Path, flag_name: str) -> Path:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Thiếu đường dẫn cho {flag_name}: {resolved}")
    if not resolved.is_dir():
        raise NotADirectoryError(f"{flag_name} phải là thư mục: {resolved}")
    return resolved


def ensure_output_dir(path: Path) -> Path:
    resolved = path.resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def validate_threshold(threshold: float, flag_name: str) -> float:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"{flag_name} phải nằm trong [0, 1], nhận được {threshold}")
    return threshold


def parse_thresholds(threshold: float, sweep_thresholds: str) -> list[float]:
    values = [validate_threshold(threshold, "--threshold")]
    if sweep_thresholds.strip():
        parsed = []
        for raw_value in sweep_thresholds.split(","):
            raw_value = raw_value.strip()
            if not raw_value:
                continue
            parsed.append(validate_threshold(float(raw_value), "--sweep-thresholds"))
        if not parsed:
            raise ValueError("--sweep-thresholds không được rỗng nếu đã truyền tham số này.")
        values = sorted(set(parsed))
    return values


def normalize_sample_id(stem: str) -> str:
    normalized = stem
    for suffix in STEM_SUFFIXES:
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    return normalized


def collect_mask_files(
    mask_dir: Path,
    allow_duplicate_sample_ids: bool,
    label: str,
) -> tuple[dict[str, Path], list[DuplicateSampleIdReport]]:
    files: dict[str, Path] = {}
    grouped_paths: dict[str, list[Path]] = {}
    for path in sorted(mask_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        sample_id = normalize_sample_id(path.stem)
        grouped_paths.setdefault(sample_id, []).append(path)

    duplicate_reports: list[DuplicateSampleIdReport] = []
    duplicate_messages: list[str] = []
    for sample_id, candidates in grouped_paths.items():
        chosen_path = candidates[0]
        files[sample_id] = chosen_path
        if len(candidates) == 1:
            continue
        ignored_paths = candidates[1:]
        duplicate_reports.append(
            DuplicateSampleIdReport(
                sample_id=sample_id,
                chosen_path=chosen_path,
                ignored_paths=ignored_paths,
            )
        )
        duplicate_messages.append(
            f"{label} duplicate sample_id='{sample_id}': chosen={chosen_path}, ignored={ignored_paths}"
        )

    if duplicate_messages and not allow_duplicate_sample_ids:
        raise ValueError(
            "Phát hiện duplicate sample_id sau khi normalize stem. "
            "Dùng --allow-duplicate-sample-ids nếu muốn tiếp tục theo deterministic order. "
            f"Chi tiết: {'; '.join(duplicate_messages[:10])}"
        )
    return files, duplicate_reports


def require_mask_dependencies() -> tuple[Any, Any]:
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "This command requires numpy and opencv-python. Install dependencies or use the project environment."
        ) from exc
    return cv2, np


def read_split_ids(split_file: Path | None, split_dir: Path | None, split_name: str) -> tuple[set[str] | None, str | None]:
    source_path: Path | None = None
    if split_file is not None:
        source_path = split_file.resolve()
    elif split_dir is not None:
        source_path = split_dir.resolve() / split_name

    if source_path is None:
        return None, None
    if not source_path.exists():
        raise FileNotFoundError(f"Không tìm thấy split source: {source_path}")

    values = {
        normalize_sample_id(line.strip().lstrip("\ufeff"))
        for line in source_path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    }
    if not values:
        raise ValueError(f"Split source rỗng: {source_path}")
    return values, str(source_path)


def filter_by_suffix(paths: dict[str, Path], explicit_suffix: str) -> dict[str, Path]:
    if not explicit_suffix:
        return paths
    suffix = explicit_suffix.lower()
    filtered = {
        sample_id: path
        for sample_id, path in paths.items()
        if path.stem.lower().endswith(suffix)
    }
    return filtered


def build_pairs(
    pred_mask_dir: Path,
    gt_mask_dir: Path,
    split_ids: set[str] | None,
    pred_suffix: str,
    gt_suffix: str,
    allow_duplicate_sample_ids: bool,
) -> tuple[list[PairItem], list[str], list[str], list[str], list[DuplicateSampleIdReport], list[DuplicateSampleIdReport]]:
    pred_files_all, pred_duplicates = collect_mask_files(
        pred_mask_dir,
        allow_duplicate_sample_ids=allow_duplicate_sample_ids,
        label="pred",
    )
    gt_files_all, gt_duplicates = collect_mask_files(
        gt_mask_dir,
        allow_duplicate_sample_ids=allow_duplicate_sample_ids,
        label="gt",
    )
    pred_files = filter_by_suffix(pred_files_all, pred_suffix)
    gt_files = filter_by_suffix(gt_files_all, gt_suffix)

    if split_ids is not None:
        pred_files = {sample_id: path for sample_id, path in pred_files.items() if sample_id in split_ids}
        gt_files = {sample_id: path for sample_id, path in gt_files.items() if sample_id in split_ids}

    pred_ids = set(pred_files)
    gt_ids = set(gt_files)
    common_ids = sorted(pred_ids & gt_ids)
    missing_pred = sorted(gt_ids - pred_ids)
    missing_gt = sorted(pred_ids - gt_ids)

    skipped_by_split: list[str] = []
    if split_ids is not None:
        all_ids = sorted((pred_ids | gt_ids) ^ set(common_ids))
        skipped_by_split = sorted(sample_id for sample_id in split_ids if sample_id not in pred_ids and sample_id not in gt_ids)
        _ = all_ids

    pairs = [
        PairItem(sample_id=sample_id, pred_path=pred_files[sample_id], gt_path=gt_files[sample_id])
        for sample_id in common_ids
    ]
    return pairs, missing_pred, missing_gt, skipped_by_split, pred_duplicates, gt_duplicates


def load_grayscale_mask(path: Path) -> Any:
    cv2, _ = require_mask_dependencies()
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Không đọc được mask: {path}")
    if image.ndim == 3:
        if image.shape[2] == 4:
            image = image[:, :, :3]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, np = require_mask_dependencies()
    return image.astype(np.float32)


def to_probability_mask(mask: Any) -> Any:
    _, np = require_mask_dependencies()
    max_value = float(mask.max()) if mask.size else 0.0
    min_value = float(mask.min()) if mask.size else 0.0
    if max_value <= 1.0 and min_value >= 0.0:
        return np.clip(mask, 0.0, 1.0).astype(np.float32)
    return np.clip(mask / 255.0, 0.0, 1.0).astype(np.float32)


def to_binary_mask(mask: Any, threshold: float) -> Any:
    probability = to_probability_mask(mask)
    _, np = require_mask_dependencies()
    return (probability >= threshold).astype(np.uint8)


def compute_pixel_metrics(pred_binary: Any, gt_binary: Any) -> PixelMetrics:
    _, np = require_mask_dependencies()
    if pred_binary.shape != gt_binary.shape:
        raise ValueError(f"Mask shape không khớp: pred={pred_binary.shape}, gt={gt_binary.shape}")

    pred_bool = pred_binary.astype(bool)
    gt_bool = gt_binary.astype(bool)

    tp = int(np.logical_and(pred_bool, gt_bool).sum())
    fp = int(np.logical_and(pred_bool, ~gt_bool).sum())
    fn = int(np.logical_and(~pred_bool, gt_bool).sum())
    tn = int(np.logical_and(~pred_bool, ~gt_bool).sum())

    eps = 1e-6
    iou = float((tp + eps) / (tp + fp + fn + eps))
    f1 = float((2 * tp + eps) / (2 * tp + fp + fn + eps))
    precision = float((tp + eps) / (tp + fp + eps))
    recall = float((tp + eps) / (tp + fn + eps))
    fpr = float((fp + eps) / (fp + tn + eps))
    pred_positive_ratio = float(pred_bool.mean())
    gt_positive_ratio = float(gt_bool.mean())

    return PixelMetrics(
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        iou=iou,
        f1=f1,
        precision=precision,
        recall=recall,
        fpr=fpr,
        pred_positive_ratio=pred_positive_ratio,
        gt_positive_ratio=gt_positive_ratio,
    )


def aggregate_metrics(metrics: Iterable[PixelMetrics]) -> dict[str, float]:
    metrics_list = list(metrics)
    if not metrics_list:
        raise ValueError("Không có metrics để aggregate.")

    total_tp = sum(item.tp for item in metrics_list)
    total_fp = sum(item.fp for item in metrics_list)
    total_fn = sum(item.fn for item in metrics_list)
    total_tn = sum(item.tn for item in metrics_list)
    eps = 1e-6

    return {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "tn": total_tn,
        "iou": float((total_tp + eps) / (total_tp + total_fp + total_fn + eps)),
        "f1": float((2 * total_tp + eps) / (2 * total_tp + total_fp + total_fn + eps)),
        "precision": float((total_tp + eps) / (total_tp + total_fp + eps)),
        "recall": float((total_tp + eps) / (total_tp + total_fn + eps)),
        "fpr": float((total_fp + eps) / (total_fp + total_tn + eps)),
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_masks_evaluation(args: argparse.Namespace) -> int:
    pred_mask_dir = ensure_dir_exists(args.pred_mask_dir, "--pred-mask-dir")
    gt_mask_dir = ensure_dir_exists(args.gt_mask_dir, "--gt-mask-dir")
    output_dir = ensure_output_dir(args.output_dir)
    thresholds = parse_thresholds(args.threshold, args.sweep_thresholds)
    split_ids, split_source = read_split_ids(args.split_file, args.split_dir, args.split_name)
    pairs, missing_pred, missing_gt, skipped_by_split, pred_duplicates, gt_duplicates = build_pairs(
        pred_mask_dir=pred_mask_dir,
        gt_mask_dir=gt_mask_dir,
        split_ids=split_ids,
        pred_suffix=args.pred_suffix,
        gt_suffix=args.gt_suffix,
        allow_duplicate_sample_ids=args.allow_duplicate_sample_ids,
    )

    if not pairs:
        raise RuntimeError("Không có valid pair nào để evaluate sau khi ghép pred/gt và filter split.")

    per_image_rows: list[dict[str, Any]] = []
    sweep_rows: list[dict[str, Any]] = []
    aggregate_by_threshold: dict[float, dict[str, Any]] = {}

    loaded_masks: dict[str, tuple[Any, Any]] = {}
    for item in pairs:
        pred_mask = load_grayscale_mask(item.pred_path)
        gt_mask = load_grayscale_mask(item.gt_path)
        if pred_mask.shape != gt_mask.shape:
            raise ValueError(
                "Pred/GT shape không khớp cho "
                f"{item.sample_id}: pred={pred_mask.shape}, gt={gt_mask.shape}"
            )
        loaded_masks[item.sample_id] = (pred_mask, gt_mask)

    for threshold in thresholds:
        per_threshold_metrics: list[PixelMetrics] = []
        for item in pairs:
            pred_mask, gt_mask = loaded_masks[item.sample_id]
            pred_binary = to_binary_mask(pred_mask, threshold)
            gt_binary = to_binary_mask(gt_mask, 0.5)
            metrics = compute_pixel_metrics(pred_binary, gt_binary)
            per_threshold_metrics.append(metrics)
            per_image_rows.append(
                {
                    "sample_id": item.sample_id,
                    "threshold": f"{threshold:.4f}",
                    "pred_path": str(item.pred_path),
                    "gt_path": str(item.gt_path),
                    "tp": metrics.tp,
                    "fp": metrics.fp,
                    "fn": metrics.fn,
                    "tn": metrics.tn,
                    "iou": f"{metrics.iou:.6f}",
                    "f1": f"{metrics.f1:.6f}",
                    "precision": f"{metrics.precision:.6f}",
                    "recall": f"{metrics.recall:.6f}",
                    "fpr": f"{metrics.fpr:.6f}",
                    "pred_positive_ratio": f"{metrics.pred_positive_ratio:.6f}",
                    "gt_positive_ratio": f"{metrics.gt_positive_ratio:.6f}",
                }
            )

        aggregate = aggregate_metrics(per_threshold_metrics)
        aggregate_by_threshold[threshold] = aggregate
        sweep_rows.append(
            {
                "threshold": f"{threshold:.4f}",
                "sample_count": len(per_threshold_metrics),
                "tp": aggregate["tp"],
                "fp": aggregate["fp"],
                "fn": aggregate["fn"],
                "tn": aggregate["tn"],
                "iou": f"{aggregate['iou']:.6f}",
                "f1": f"{aggregate['f1']:.6f}",
                "precision": f"{aggregate['precision']:.6f}",
                "recall": f"{aggregate['recall']:.6f}",
                "fpr": f"{aggregate['fpr']:.6f}",
            }
        )

    best_threshold = max(aggregate_by_threshold, key=lambda value: aggregate_by_threshold[value]["iou"])
    best_metrics = aggregate_by_threshold[best_threshold]
    warnings_list: list[str] = []
    if missing_pred:
        warnings_list.append(f"Thiếu pred mask cho {len(missing_pred)} sample: {missing_pred[:10]}")
    if missing_gt:
        warnings_list.append(f"Thiếu gt mask cho {len(missing_gt)} sample: {missing_gt[:10]}")
    if skipped_by_split:
        warnings_list.append(f"Split source có sample không tìm thấy ở cả pred/gt: {skipped_by_split[:10]}")
    if pred_duplicates:
        warnings_list.append(
            "Pred duplicate sample_id được giữ theo deterministic order: "
            f"{[item.sample_id for item in pred_duplicates[:10]]}"
        )
    if gt_duplicates:
        warnings_list.append(
            "GT duplicate sample_id được giữ theo deterministic order: "
            f"{[item.sample_id for item in gt_duplicates[:10]]}"
        )

    summary = {
        "mode": "masks",
        "pred_mask_dir": str(pred_mask_dir),
        "gt_mask_dir": str(gt_mask_dir),
        "output_dir": str(output_dir),
        "split_source": split_source,
        "pair_count": len(pairs),
        "missing_pred_count": len(missing_pred),
        "missing_gt_count": len(missing_gt),
        "thresholds_evaluated": thresholds,
        "best_threshold_by_iou": best_threshold,
        "best_metrics": best_metrics,
        "warnings": warnings_list,
        "duplicate_sample_id_policy": (
            "fail_by_default"
            if not args.allow_duplicate_sample_ids
            else "allow_with_warning_choose_first_sorted_path"
        ),
        "duplicate_sample_id_reports": {
            "pred": [
                {
                    "sample_id": item.sample_id,
                    "chosen_path": str(item.chosen_path),
                    "ignored_paths": [str(path) for path in item.ignored_paths],
                }
                for item in pred_duplicates
            ],
            "gt": [
                {
                    "sample_id": item.sample_id,
                    "chosen_path": str(item.chosen_path),
                    "ignored_paths": [str(path) for path in item.ignored_paths],
                }
                for item in gt_duplicates
            ],
        },
        "notes": [
            "Script này hiện đánh giá predicted masks có sẵn cho Module 1.",
            "Checkpoint inference mode chưa được bật trong Phase 1B này.",
            "Script này không phải full pipeline evaluator và không phải LPIPS/FID evaluator.",
        ],
    }

    write_csv(
        output_dir / "per_image_metrics.csv",
        [
            "sample_id",
            "threshold",
            "pred_path",
            "gt_path",
            "tp",
            "fp",
            "fn",
            "tn",
            "iou",
            "f1",
            "precision",
            "recall",
            "fpr",
            "pred_positive_ratio",
            "gt_positive_ratio",
        ],
        per_image_rows,
    )
    write_csv(
        output_dir / "threshold_sweep.csv",
        ["threshold", "sample_count", "tp", "fp", "fn", "tn", "iou", "f1", "precision", "recall", "fpr"],
        sweep_rows,
    )
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"pair_count: {len(pairs)}")
    print(f"best_threshold_by_iou: {best_threshold:.4f}")
    print(f"metrics_summary: {output_dir / 'metrics_summary.json'}")
    print(f"per_image_metrics: {output_dir / 'per_image_metrics.csv'}")
    print(f"threshold_sweep: {output_dir / 'threshold_sweep.csv'}")
    return 0


def main() -> int:
    configure_utf8_stdio()
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "handler"):
        parser.error("Thiếu subcommand. Dùng `masks` hoặc chạy `evaluate_segmentation.py masks --help` để xem chi tiết.")
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
