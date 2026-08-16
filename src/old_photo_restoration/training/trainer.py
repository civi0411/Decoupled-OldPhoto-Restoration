"""PyTorch training loop and validation evaluation engine for crack segmentation."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from old_photo_restoration.evaluation.metrics import compute_iou_binary, dice_score
from old_photo_restoration.training.losses import BCEDiceLoss


@dataclass
class TrainerConfig:
    """Configuration settings for training old photo crack segmentation models."""

    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 8
    num_workers: int = 4
    threshold: float = 0.5
    save_top_k: int = 1
    gradient_clip_val: float = 1.0


class SegmentationTrainer:
    """Orchestrates model training, validation metrics monitoring, and checkpoint persistence."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader[dict[str, Any]],
        val_loader: DataLoader[dict[str, Any]] | None,
        config: TrainerConfig,
        output_dir: str | Path,
        device: torch.device | str = "auto",
        criterion: nn.Module | None = None,
    ) -> None:
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = criterion or BCEDiceLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=0.5,
            patience=5,
        )

        self.logger = logging.getLogger("old_photo_restoration.training")
        self._history: list[dict[str, Any]] = []
        self.best_val_iou = 0.0
        self.best_epoch = -1

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        """Execute one complete training epoch across all batches."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            damaged = batch["damaged"].to(self.device)
            targets = batch["mask"].to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(damaged)
            loss = self.criterion(logits, targets)
            loss.backward()

            if self.config.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_val)

            self.optimizer.step()
            total_loss += float(loss.item())
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        return {"loss": avg_loss}

    @torch.inference_mode()
    def validate(self) -> dict[str, float]:
        """Evaluate validation metrics (IoU, Dice/F1, and loss)."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        iou_scores: list[float] = []
        dice_scores: list[float] = []
        num_batches = 0

        for batch in self.val_loader:
            damaged = batch["damaged"].to(self.device)
            targets = batch["mask"].to(self.device)

            logits = self.model(damaged)
            loss = self.criterion(logits, targets)
            total_loss += float(loss.item())
            num_batches += 1

            probs = torch.sigmoid(logits)
            pred_masks = (probs >= self.config.threshold).cpu().numpy()
            target_masks = targets.cpu().numpy()

            for pred, target in zip(pred_masks, target_masks):
                pred_bin = (pred[0] * 255).astype(np.uint8)
                target_bin = (target[0] * 255).astype(np.uint8)
                iou_scores.append(compute_iou_binary(pred_bin, target_bin))
                dice_scores.append(dice_score(target_bin, pred_bin))

        return {
            "val_loss": total_loss / max(1, num_batches),
            "val_iou": float(np.mean(iou_scores)) if iou_scores else 0.0,
            "val_f1": float(np.mean(dice_scores)) if dice_scores else 0.0,
        }

    def _save_checkpoint(self, epoch: int, filename: str, metrics: dict[str, Any]) -> Path:
        path = self.checkpoints_dir / filename
        payload = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": asdict(self.config),
            "metrics": metrics,
            "architecture_config": getattr(self.model, "get_config", lambda: {})(),
        }
        torch.save(payload, path)
        return path

    def fit(self) -> dict[str, Any]:
        """Run full training loop across all configured epochs."""
        self.logger.info(f"Starting training on device={self.device} for {self.config.epochs} epochs")
        start_time = time.time()

        for epoch in range(1, self.config.epochs + 1):
            epoch_start = time.time()
            train_metrics = self.train_one_epoch(epoch)
            val_metrics = self.validate()
            elapsed = time.time() - epoch_start

            current_iou = val_metrics.get("val_iou", 0.0)
            if self.val_loader is not None:
                self.scheduler.step(current_iou)

            epoch_record = {
                "epoch": epoch,
                "elapsed_sec": round(elapsed, 2),
                **train_metrics,
                **val_metrics,
            }
            self._history.append(epoch_record)

            self.logger.info(
                f"Epoch [{epoch}/{self.config.epochs}] "
                f"loss={train_metrics['loss']:.4f} | "
                f"val_iou={val_metrics.get('val_iou', 0.0):.4f} | "
                f"val_f1={val_metrics.get('val_f1', 0.0):.4f} "
                f"({elapsed:.1f}s)"
            )

            # Save last checkpoint
            self._save_checkpoint(epoch, "last.ckpt", epoch_record)

            # Save best checkpoint
            if current_iou > self.best_val_iou or epoch == 1:
                self.best_val_iou = current_iou
                self.best_epoch = epoch
                self._save_checkpoint(epoch, "best_iou.ckpt", epoch_record)

        total_elapsed = time.time() - start_time
        summary = {
            "total_epochs": self.config.epochs,
            "best_epoch": self.best_epoch,
            "best_val_iou": self.best_val_iou,
            "total_elapsed_sec": round(total_elapsed, 2),
            "history": self._history,
        }

        # Save training summary JSON
        summary_path = self.output_dir / "training_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return summary
