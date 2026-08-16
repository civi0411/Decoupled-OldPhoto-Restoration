from __future__ import annotations

import argparse
import platform
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from old_photo_restoration.config import ProjectConfig, load_config
from old_photo_restoration.utils.checkpoints import sha256_file


@dataclass(slots=True)
class CheckItem:
    level: str
    message: str
    core_ok: bool = True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect artifact configuration and local runtime readiness for this project."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/inference.yaml"))
    parser.add_argument("--checkpoint-config", type=Path, default=Path("configs/checkpoints.yaml"))
    parser.add_argument("--external-config", type=Path, default=Path("configs/external_paths.yaml"))
    parser.add_argument("--post-inpainting", action="store_true")
    parser.add_argument(
        "--color-restoration-config",
        type=Path,
        default=Path("configs/color_restoration.yaml"),
    )
    parser.add_argument("--strict", action="store_true")
    return parser


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def add_result(results: list[CheckItem], level: str, message: str, core_ok: bool = True) -> None:
    results.append(CheckItem(level=level, message=message, core_ok=core_ok))


def check_package_import(results: list[CheckItem]) -> None:
    try:
        import old_photo_restoration  # noqa: F401
    except Exception as exc:
        add_result(results, "FAIL", f"Could not import old_photo_restoration: {exc}", core_ok=False)
        return
    add_result(results, "OK", "Imported old_photo_restoration successfully")


def check_python_and_torch(results: list[CheckItem]) -> None:
    add_result(results, "OK", f"Python version: {platform.python_version()}")
    try:
        import torch
    except Exception as exc:
        add_result(results, "FAIL", f"Could not import torch: {exc}", core_ok=False)
        return

    add_result(results, "OK", f"Imported torch successfully: {torch.__version__}")
    add_result(results, "OK", f"torch.cuda.is_available(): {torch.cuda.is_available()}")


def check_config(args: argparse.Namespace, results: list[CheckItem]) -> ProjectConfig | None:
    external_config_path = resolve_path(args.external_config)
    if external_config_path.exists():
        add_result(results, "OK", f"Found local external config: {external_config_path}")
    else:
        add_result(results, "MISSING", f"Missing local external config: {external_config_path}", core_ok=False)

    try:
        config = load_config(
            inference_path=resolve_path(args.config),
            checkpoint_path=resolve_path(args.checkpoint_config),
            external_path=external_config_path,
        )
    except Exception as exc:
        add_result(results, "FAIL", f"load_config() failed: {exc}", core_ok=False)
        return None

    add_result(results, "OK", "load_config() succeeded")
    return config


def check_segmentation_checkpoint(config: ProjectConfig, results: list[CheckItem]) -> None:
    checkpoint_path = config.checkpoint.expected_path
    if not checkpoint_path.exists():
        add_result(results, "MISSING", f"Missing R013 checkpoint: {checkpoint_path}", core_ok=False)
        return

    add_result(results, "OK", f"Found R013 checkpoint: {checkpoint_path}")
    actual_sha256 = sha256_file(checkpoint_path).lower()
    expected_sha256 = config.checkpoint.sha256.lower()
    if actual_sha256 == expected_sha256:
        add_result(results, "OK", f"R013 checkpoint SHA256 matches: {actual_sha256}")
    else:
        add_result(
            results,
            "FAIL",
            f"R013 checkpoint SHA256 mismatch. expected={expected_sha256}, actual={actual_sha256}",
            core_ok=False,
        )


def check_lama(config: ProjectConfig, results: list[CheckItem]) -> None:
    lama = config.lama
    if lama.repo_root.exists():
        add_result(results, "OK", f"LaMa repo_root exists: {lama.repo_root}")
    else:
        add_result(results, "MISSING", f"Missing LaMa repo_root: {lama.repo_root}", core_ok=False)

    if lama.predict_script.exists():
        add_result(results, "OK", f"LaMa predict.py exists: {lama.predict_script}")
    else:
        add_result(results, "MISSING", f"Missing LaMa predict.py: {lama.predict_script}", core_ok=False)

    if lama.checkpoint.exists():
        add_result(results, "OK", f"LaMa checkpoint exists: {lama.checkpoint}")
    else:
        add_result(results, "MISSING", f"Missing LaMa checkpoint: {lama.checkpoint}", core_ok=False)


def check_codeformer(config: ProjectConfig, results: list[CheckItem]) -> None:
    codeformer = config.codeformer
    repo_exists = codeformer.repo_root.exists()
    ckpt_exists = codeformer.checkpoint.exists()
    if repo_exists and ckpt_exists:
        add_result(results, "OPTIONAL", f"Optional CodeFormer dependency is available: repo={codeformer.repo_root}")
        return
    if not repo_exists:
        add_result(results, "OPTIONAL", f"CodeFormer repository is not available: {codeformer.repo_root}")
    if not ckpt_exists:
        add_result(results, "OPTIONAL", f"CodeFormer weights are not available: {codeformer.checkpoint}")


def check_color_restoration(args: argparse.Namespace, results: list[CheckItem]) -> None:
    try:
        from old_photo_restoration.color_restoration import load_color_restoration_config

        color_config = load_color_restoration_config(resolve_path(args.color_restoration_config))
    except Exception as exc:
        add_result(results, "FAIL", f"Color-restoration config failed: {exc}", core_ok=False)
        return

    add_result(results, "OK", f"Loaded color-restoration config: {resolve_path(args.color_restoration_config)}")
    if color_config.method != "model":
        add_result(results, "OK", f"Color-restoration method does not require a checkpoint: {color_config.method}")
        return

    try:
        import kornia  # noqa: F401
    except Exception as exc:
        add_result(results, "FAIL", f"Could not import kornia: {exc}", core_ok=False)
        return
    add_result(results, "OK", "Imported kornia successfully")

    checkpoint_value = color_config.model.checkpoint_path
    if not checkpoint_value:
        add_result(results, "MISSING", "Color-restoration checkpoint path is not configured", core_ok=False)
        return
    checkpoint_path = Path(checkpoint_value)
    checkpoint_path = checkpoint_path if checkpoint_path.is_absolute() else resolve_path(checkpoint_path)
    if not checkpoint_path.is_file():
        add_result(results, "MISSING", f"Missing color-restoration checkpoint: {checkpoint_path}", core_ok=False)
        return
    actual_sha256 = sha256_file(checkpoint_path).lower()
    expected_sha256 = (color_config.model.expected_sha256 or "").lower()
    if expected_sha256 and actual_sha256 != expected_sha256:
        add_result(
            results,
            "FAIL",
            f"Color-restoration checkpoint SHA256 mismatch. expected={expected_sha256}, actual={actual_sha256}",
            core_ok=False,
        )
        return
    add_result(results, "OK", f"Color-restoration checkpoint is ready: {checkpoint_path}")


def main() -> int:
    args = build_parser().parse_args()
    results: list[CheckItem] = []

    check_package_import(results)
    check_python_and_torch(results)
    config = check_config(args, results)
    if config is not None:
        check_segmentation_checkpoint(config, results)
        check_lama(config, results)
        check_codeformer(config, results)
    if args.post_inpainting:
        check_color_restoration(args, results)

    for item in results:
        print(f"[{item.level}] {item.message}")

    core_missing = any(not item.core_ok for item in results)
    if args.strict and core_missing:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
