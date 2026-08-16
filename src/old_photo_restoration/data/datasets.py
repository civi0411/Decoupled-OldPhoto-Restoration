"""PyTorch Dataset classes for paired crack segmentation and old photo restoration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from old_photo_restoration.utils.image_io import read_image_rgb, validate_rgb_uint8


class PairedCrackDataset(Dataset[dict[str, Any]]):
    """Dataset for paired crack segmentation and restoration.

    Expects a root directory structured with subdirectories or matched filenames across:
        - damaged/ (or inputs/)
        - mask/ (or masks/)
        - clean/ (or targets/, optional for segmentation-only training)
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        target_size: tuple[int, int] = (512, 512),
        require_clean_targets: bool = False,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.split = split
        self.target_size = target_size
        self.require_clean_targets = require_clean_targets

        self.damaged_dir = self._find_subdir(["damaged", "inputs", "image", "images"])
        self.mask_dir = self._find_subdir(["mask", "masks", "ground_truth"])
        self.clean_dir = self._find_subdir(["clean", "targets", "gt_clean"]) if require_clean_targets else None

        if self.damaged_dir is None:
            raise FileNotFoundError(f"Could not find damaged/inputs image directory inside {self.root_dir}")
        if self.mask_dir is None:
            raise FileNotFoundError(f"Could not find mask/ground_truth directory inside {self.root_dir}")

        image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        self.damaged_files = sorted([
            path for path in self.damaged_dir.iterdir()
            if path.is_file() and path.suffix.lower() in image_extensions
        ])
        if not self.damaged_files:
            raise RuntimeError(f"No image files found in {self.damaged_dir}")

    def _find_subdir(self, candidates: list[str]) -> Path | None:
        for candidate in candidates:
            path = self.root_dir / candidate
            if path.is_dir():
                return path
        return None

    def __len__(self) -> int:
        return len(self.damaged_files)

    def _resolve_sibling_path(self, base_path: Path, target_dir: Path) -> Path | None:
        """Find matching file in target_dir regardless of extension (.png vs .jpg)."""
        for ext in [base_path.suffix, ".png", ".jpg", ".jpeg", ".bmp"]:
            candidate = target_dir / f"{base_path.stem}{ext}"
            if candidate.is_file():
                return candidate
            # Also check for _mask or _clean suffixes
            candidate_suffix = target_dir / f"{base_path.stem}_{target_dir.name}{ext}"
            if candidate_suffix.is_file():
                return candidate_suffix
        return None

    def __getitem__(self, index: int) -> dict[str, Any]:
        damaged_path = self.damaged_files[index]
        damaged_rgb = read_image_rgb(damaged_path)

        mask_path = self._resolve_sibling_path(damaged_path, self.mask_dir)
        if mask_path is None:
            raise FileNotFoundError(f"Missing mask for {damaged_path.name} inside {self.mask_dir}")

        mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask_raw is None:
            raise RuntimeError(f"Failed to read mask file: {mask_path}")
        if mask_raw.ndim == 3:
            mask_raw = cv2.cvtColor(mask_raw[:, :, :3], cv2.COLOR_BGR2GRAY)
        binary_mask = (mask_raw > 127).astype(np.uint8) * 255

        # Resize to standardized target size
        target_h, target_w = self.target_size
        if damaged_rgb.shape[:2] != (target_h, target_w):
            damaged_rgb = cv2.resize(damaged_rgb, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        if binary_mask.shape[:2] != (target_h, target_w):
            binary_mask = cv2.resize(binary_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        damaged_tensor = torch.from_numpy(
            np.ascontiguousarray(damaged_rgb.transpose(2, 0, 1))
        ).float() / 255.0

        mask_tensor = torch.from_numpy(
            np.ascontiguousarray(binary_mask)
        ).float().unsqueeze(0) / 255.0

        sample: dict[str, Any] = {
            "damaged": damaged_tensor,
            "mask": mask_tensor,
            "filename": damaged_path.name,
            "stem": damaged_path.stem,
        }

        if self.clean_dir is not None:
            clean_path = self._resolve_sibling_path(damaged_path, self.clean_dir)
            if clean_path is None and self.require_clean_targets:
                raise FileNotFoundError(f"Missing clean target for {damaged_path.name} in {self.clean_dir}")
            if clean_path is not None:
                clean_rgb = read_image_rgb(clean_path)
                if clean_rgb.shape[:2] != (target_h, target_w):
                    clean_rgb = cv2.resize(clean_rgb, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
                sample["clean"] = torch.from_numpy(
                    np.ascontiguousarray(clean_rgb.transpose(2, 0, 1))
                ).float() / 255.0

        return sample


class RealPhotoPairDataset(PairedCrackDataset):
    """Dataset specialized for real historical photo evaluation pairs (e.g., old_photo_pairs_10_hq)."""

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "val",
        target_size: tuple[int, int] = (512, 512),
    ) -> None:
        super().__init__(
            root_dir=root_dir,
            split=split,
            target_size=target_size,
            require_clean_targets=False,
        )
