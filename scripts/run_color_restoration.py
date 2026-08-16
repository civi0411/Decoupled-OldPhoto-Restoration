from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path


STAGING_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = STAGING_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run standalone conservative color restoration after LaMa."
    )
    parser.add_argument("--input", required=True, type=Path, help="RGB image produced after LaMa.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=STAGING_ROOT / "configs" / "color_restoration.yaml",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument(
        "--method",
        choices=["model", "opencv_conservative", "opencv_neutral_fallback"],
        default=None,
    )
    parser.add_argument(
        "--final-color-method",
        choices=["ccm", "lab_chroma_match", "ccm_then_chroma_match"],
        default=None,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from old_photo_restoration.color_restoration.config import (
        load_color_restoration_config,
    )
    from old_photo_restoration.color_restoration.core import restore_color_after_lama
    from old_photo_restoration.color_restoration.io import (
        load_rgb_image,
        save_color_restoration_result,
    )

    config = load_color_restoration_config(args.config)
    if args.method is not None:
        config = replace(config, method=args.method)
    if config.model.checkpoint_path or args.checkpoint is not None or args.device is not None:
        checkpoint_value = args.checkpoint or config.model.checkpoint_path
        checkpoint_path = None if checkpoint_value is None else Path(checkpoint_value)
        if checkpoint_path is not None and not checkpoint_path.is_absolute():
            checkpoint_path = (STAGING_ROOT / checkpoint_path).resolve()
        model_config = replace(
            config.model,
            checkpoint_path=None if checkpoint_path is None else str(checkpoint_path),
            device=args.device or config.model.device,
        )
        config = replace(config, model=model_config)
    if args.final_color_method is not None:
        config = replace(
            config,
            final_color=replace(config.final_color, method=args.final_color_method),
        )
    result = restore_color_after_lama(
        load_rgb_image(args.input),
        config,
        runtime_dir=args.output_dir / "_runtime",
    )
    artifacts = save_color_restoration_result(result, args.output_dir, config)
    print(f"restored_color: {artifacts['image']}")
    print(f"metadata: {artifacts['metadata']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
