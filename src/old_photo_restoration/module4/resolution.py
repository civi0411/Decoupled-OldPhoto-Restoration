# src/old_photo_restoration/postprocessing/super_resolution.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class TiledSuperResolution:
    """
    Wrapper SR với tiling để tránh OOM trên ảnh lớn.
    Backend: Real-ESRGAN x4 (subprocess, giống pattern LaMa)
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

    def upscale(
        self,
        image: np.ndarray,
        tile_size: int = 256,
        overlap: int = 16,
        scale: int = 4,
    ) -> np.ndarray:
        """
        Chia ảnh thành tiles, upscale từng tile, stitch lại.
        Tránh OOM khi ảnh lớn.
        """
        h, w = image.shape[:2]
        # tính số tile theo chiều x và y
        tiles_x = _compute_tiles(w, tile_size, overlap)
        tiles_y = _compute_tiles(h, tile_size, overlap)

        output_h = h * scale
        output_w = w * scale
        output = np.zeros((output_h, output_w, 3), dtype=np.uint8)
        weight = np.zeros((output_h, output_w, 1), dtype=np.float32)

        for ty, tx in [(ty, tx) for ty in range(tiles_y) for tx in range(tiles_x)]:
            # crop tile từ ảnh gốc
            x0, x1, y0, y1 = _tile_coords(tx, ty, w, h, tile_size, overlap)
            tile = image[y0:y1, x0:x1]

            # upscale tile — gọi Real-ESRGAN
            upscaled_tile = self._upscale_single_tile(tile)

            # paste vào output với blending ở overlap
            ox0, ox1 = x0 * scale, x1 * scale
            oy0, oy1 = y0 * scale, y1 * scale
            blend = _blend_mask(upscaled_tile.shape[:2], overlap * scale)
            output[oy0:oy1, ox0:ox1] += (upscaled_tile * blend).astype(np.uint8)
            weight[oy0:oy1, ox0:ox1] += blend

        # normalize theo weight để smooth overlap seams
        weight = np.maximum(weight, 1e-6)
        output = np.clip(output.astype(np.float32) / weight, 0, 255).astype(np.uint8)
        return output

    def _upscale_single_tile(self, tile: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "Cần implement Real-ESRGAN subprocess call. "
            "Pattern giống LamaInpainter._run_predict()"
        )


def _compute_tiles(length: int, tile_size: int, overlap: int) -> int:
    step = tile_size - overlap
    return max(1, int(np.ceil((length - overlap) / step)))


def _tile_coords(
    tx: int, ty: int,
    w: int, h: int,
    tile_size: int,
    overlap: int,
) -> tuple[int, int, int, int]:
    step = tile_size - overlap
    x0 = min(tx * step, w - tile_size)
    y0 = min(ty * step, h - tile_size)
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(x0 + tile_size, w)
    y1 = min(y0 + tile_size, h)
    return x0, x1, y0, y1


def _blend_mask(shape: tuple[int, int], overlap: int) -> np.ndarray:
    """Tạo weight mask để blend smooth tại vùng overlap."""
    h, w = shape
    mask = np.ones((h, w, 1), dtype=np.float32)
    fade = np.linspace(0, 1, overlap)
    # fade in 4 cạnh
    mask[:overlap, :, 0] *= fade[:, None]
    mask[-overlap:, :, 0] *= fade[::-1, None]
    mask[:, :overlap, 0] *= fade[None, :]
    mask[:, -overlap:, 0] *= fade[None, ::-1]
    return mask