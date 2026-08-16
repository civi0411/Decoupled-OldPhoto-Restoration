"""Command-line utility for generating synthetic crack datasets (ds-crack3d).

Combines clean high-resolution reference images (such as DIV2K) with crack mask annotations
(such as CrackForest) using geometric augmentations and realistic alpha blending.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import cv2
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from old_photo_restoration.data.synthesis_engine import SynthesisConfig, SyntheticCrackGenerator
from old_photo_restoration.utils.image_io import read_image_rgb, save_image_rgb


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synthesize paired old photo crack restoration samples.")
    parser.add_argument("--clean-dir", type=Path, required=True, help="Directory containing clean RGB images.")
    parser.add_argument("--crack-dir", type=Path, required=True, help="Directory containing binary crack masks.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Destination directory for synthesized triples.")
    parser.add_argument("--count", type=int, default=100, help="Number of synthetic triples to generate.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "synthesis.yaml", help="Path to synthesis YAML configuration.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser


def load_config_from_yaml(path: Path) -> SynthesisConfig:
    if not path.is_file():
        return SynthesisConfig()
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        synth_dict = data.get("synthesis", {})
        target_size = tuple(synth_dict.get("target_size", [512, 512]))
        return SynthesisConfig(
            target_size=(int(target_size[0]), int(target_size[1])),
            min_crack_width_dilation=int(synth_dict.get("min_crack_width_dilation", 0)),
            max_crack_width_dilation=int(synth_dict.get("max_crack_width_dilation", 2)),
            blend_edge_blur_sigma=float(synth_dict.get("blend_edge_blur_sigma", 1.2)),
            dark_crack_probability=float(synth_dict.get("dark_crack_probability", 0.85)),
            min_opacity=float(synth_dict.get("min_opacity", 0.65)),
            max_opacity=float(synth_dict.get("max_opacity", 0.95)),
            add_subtle_noise_probability=float(synth_dict.get("add_subtle_noise_probability", 0.30)),
            noise_sigma=float(synth_dict.get("noise_sigma", 5.0)),
        )
    except Exception as exc:
        logging.warning(f"Failed to load config {path}: {exc}. Using default configuration.")
        return SynthesisConfig()


def find_image_files(directory: Path) -> list[Path]:
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    return sorted([
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    ])


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = build_parser().parse_args()

    random.seed(args.seed)
    clean_files = find_image_files(args.clean_dir)
    crack_files = find_image_files(args.crack_dir)

    if not clean_files:
        logging.error(f"No clean images found inside {args.clean_dir}")
        return 1
    if not crack_files:
        logging.error(f"No crack masks found inside {args.crack_dir}")
        return 1

    config = load_config_from_yaml(args.config)
    generator = SyntheticCrackGenerator(config)

    out_damaged = args.output_dir / "damaged"
    out_clean = args.output_dir / "clean"
    out_mask = args.output_dir / "mask"
    out_damaged.mkdir(parents=True, exist_ok=True)
    out_clean.mkdir(parents=True, exist_ok=True)
    out_mask.mkdir(parents=True, exist_ok=True)

    logging.info(f"Synthesizing {args.count} samples into {args.output_dir} using {len(clean_files)} clean images and {len(crack_files)} crack patterns.")

    metadata_records: list[dict[str, Any]] = []

    for idx in range(1, args.count + 1):
        clean_path = random.choice(clean_files)
        crack_path = random.choice(crack_files)

        clean_rgb = read_image_rgb(clean_path)
        crack_raw = cv2.imread(str(crack_path), cv2.IMREAD_UNCHANGED)
        if crack_raw is None:
            continue
        if crack_raw.ndim == 3:
            crack_raw = cv2.cvtColor(crack_raw[:, :, :3], cv2.COLOR_BGR2GRAY)

        result = generator.synthesize(clean_rgb, crack_raw)
        sample_id = f"synth_{idx:05d}"

        save_image_rgb(out_damaged / f"{sample_id}.png", result["damaged_rgb"])
        save_image_rgb(out_clean / f"{sample_id}.png", result["clean_rgb"])
        cv2.imwrite(str(out_mask / f"{sample_id}.png"), result["mask"])

        metadata_records.append({
            "sample_id": sample_id,
            "clean_source": clean_path.name,
            "crack_source": crack_path.name,
            **result["metadata"],
        })

        if idx % max(1, args.count // 10) == 0 or idx == args.count:
            logging.info(f"Progress: [{idx}/{args.count}] samples generated.")

    summary_path = args.output_dir / "synthesis_summary.json"
    summary_path.write_text(json.dumps({
        "total_count": len(metadata_records),
        "target_size": list(config.target_size),
        "samples": metadata_records,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    logging.info(f"Synthesis completed successfully. Summary saved to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
