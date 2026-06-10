# src/old_photo_restoration/postprocessing/colorization.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class Colorizer:
    """
    Wrapper colorization.
    Backend: DDColor hoặc DeOldify (subprocess pattern)
    """

    def __init__(self, repo_root: Path, checkpoint: Path) -> None:
        self.repo_root = repo_root
        self.checkpoint = checkpoint

    def readiness(self) -> dict[str, Any]:
        return {
            "repo_exists": self.repo_root.exists(),
            "checkpoint_exists": self.checkpoint.exists(),
            "available": self.repo_root.exists() and self.checkpoint.exists(),
        }

    def colorize(self, image: np.ndarray) -> np.ndarray:
        """
        Input: grayscale hoặc sepia RGB
        Output: colorized RGB
        """
        raise NotImplementedError(
            "Cần implement DDColor/DeOldify subprocess call."
        )