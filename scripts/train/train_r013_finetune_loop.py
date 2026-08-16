"""Command-line utility for fine-tuning Attention U-Net R013 on real historical photo pairs.

Loads pretrained synthetic initialization weights (such as R009 or R011 checkpoints) and applies
controlled domain adaptation using lower learning rates on real-world photo damage pairs.
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

from old_photo_restoration.data.datasets import RealPhotoPairDataset
from old_photo_restoration.segmentation.model import build_segmentation_model
from old_photo_restoration.training.trainer import SegmentationTrainer, TrainerConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune Attention U-Net R013 on real historical photo pairs.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Path to real photo dataset containing damaged/ and mask/ subdirectories.")
    parser.add_argument("--pretrained-checkpoint", type=Path, default=None, help="Path to pretrained checkpoint (e.g., synthetic pretraining R009/R011).")
    parser.add_argument("--epochs", type=int, default=39, help="Number of fine-tuning epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Mini-batch size for fine-tuning.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Fine-tuning learning rate (typically lower than scratch training).")
    parser.add_argument("--val-split", type=float, default=0.20, help="Validation split ratio.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "training_runs" / "r013_finetune_run", help="Destination directory.")
    parser.add_argument("--device", type=str, default="auto", help="Execution device (auto, cuda, cpu).")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker count.")
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = build_parser().parse_args()

    logging.info(f"Loading real photo pairs from {args.data_dir}")
    full_dataset = RealPhotoPairDataset(root_dir=args.data_dir, split="train")

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

    logging.info(f"Building R013 model (pretrained_checkpoint={args.pretrained_checkpoint})")
    model = build_segmentation_model(arch="r013_custom_attnunet", checkpoint_path=args.pretrained_checkpoint)

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
    logging.info(f"Fine-tuning finished. Best validation IoU: {summary['best_val_iou']:.4f} at epoch {summary['best_epoch']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
