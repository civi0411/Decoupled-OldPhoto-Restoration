"""Command-line utility for training Attention U-Net crack segmentation models from scratch.

Supports both custom Attention U-Net (R013) and ResNet-34 Attention U-Net (R014) architectures
using paired (damaged, mask) training samples.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from old_photo_restoration.data.datasets import PairedCrackDataset
from old_photo_restoration.segmentation.model import build_segmentation_model
from old_photo_restoration.training.trainer import SegmentationTrainer, TrainerConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Attention U-Net crack segmentation models.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Path to dataset directory containing damaged/ and mask/ subdirectories.")
    parser.add_argument("--arch", type=str, default="r013_custom_attnunet", choices=["r013_custom_attnunet", "r014_resnet34"], help="Model architecture.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument("--val-split", type=float, default=0.15, help="Fraction of data allocated for validation monitoring.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "training_runs" / "scratch_run", help="Destination directory for logs and checkpoints.")
    parser.add_argument("--device", type=str, default="auto", help="Execution device (auto, cuda, cpu).")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker count.")
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = build_parser().parse_args()

    logging.info(f"Loading dataset from {args.data_dir}")
    full_dataset = PairedCrackDataset(root_dir=args.data_dir, split="train")

    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size

    if val_size > 0 and train_size > 0:
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )
    else:
        train_dataset = full_dataset
        val_dataset = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        if val_dataset is not None
        else None
    )

    logging.info(f"Instantiating model architecture={args.arch}")
    model = build_segmentation_model(arch=args.arch, checkpoint_path=None)

    config = TrainerConfig(
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    trainer = SegmentationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        output_dir=args.output_dir,
        device=args.device,
    )

    summary = trainer.fit()
    logging.info(f"Training finished. Best validation IoU: {summary['best_val_iou']:.4f} at epoch {summary['best_epoch']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
