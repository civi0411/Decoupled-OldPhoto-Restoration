"""Synthetic crack and damage generation engine for old photo restoration.

This engine synthesizes realistic old photo damage (cracks, scratches, and aging artifacts)
by combining high-resolution clean target images (such as DIV2K) with crack mask patterns
(such as CrackForest or synthetic crack asset banks).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from old_photo_restoration.utils.image_io import validate_rgb_uint8


@dataclass(frozen=True)
class SynthesisConfig:
    """Configuration options for synthetic crack generation."""

    target_size: tuple[int, int] = (512, 512)
    min_crack_width_dilation: int = 0
    max_crack_width_dilation: int = 2
    blend_edge_blur_sigma: float = 1.2
    dark_crack_probability: float = 0.85
    min_opacity: float = 0.65
    max_opacity: float = 0.95
    add_subtle_noise_probability: float = 0.30
    noise_sigma: float = 5.0


class SyntheticCrackGenerator:
    """Synthesizes paired (damaged, clean, binary_mask) samples from clean images and crack patterns."""

    def __init__(self, config: SynthesisConfig | None = None) -> None:
        self.config = config or SynthesisConfig()

    @staticmethod
    def _ensure_binary_mask(mask: np.ndarray) -> np.ndarray:
        array = np.asarray(mask)
        if array.ndim == 3 and array.shape[2] == 3:
            array = cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        elif array.ndim == 3 and array.shape[2] == 1:
            array = array[:, :, 0]
        binary = (array > 127).astype(np.uint8) * 255
        return np.ascontiguousarray(binary)

    def _apply_geometric_transforms(
        self,
        clean: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Resize and apply random flips/rotations to augment diversity."""
        target_h, target_w = self.config.target_size
        clean_resized = cv2.resize(clean, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        mask_resized = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        # Random horizontal flip
        if random.random() < 0.5:
            clean_resized = cv2.flip(clean_resized, 1)
            mask_resized = cv2.flip(mask_resized, 1)

        # Random vertical flip
        if random.random() < 0.5:
            clean_resized = cv2.flip(clean_resized, 0)
            mask_resized = cv2.flip(mask_resized, 0)

        # Random 90/180/270 degree rotations for crack masks
        rotations = random.choice([0, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE])
        if rotations != 0:
            mask_resized = cv2.rotate(mask_resized, rotations)

        return validate_rgb_uint8(clean_resized), self._ensure_binary_mask(mask_resized)

    def _morphological_jitter(self, mask: np.ndarray) -> np.ndarray:
        """Randomly dilate or erode mask slightly to vary crack thickness."""
        min_d = self.config.min_crack_width_dilation
        max_d = self.config.max_crack_width_dilation
        if max_d <= 0 or min_d > max_d:
            return mask

        dilation_radius = random.randint(min_d, max_d)
        if dilation_radius == 0:
            return mask

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * dilation_radius + 1, 2 * dilation_radius + 1),
        )
        return cv2.dilate(mask, kernel, iterations=1)

    def _blend_cracks(
        self,
        clean: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Blend realistic crack colors onto the clean image using alpha feathering."""
        # Random opacity between bounds
        opacity = random.uniform(self.config.min_opacity, self.config.max_opacity)

        # Soften mask edges for realistic boundary transition
        blurred_mask = cv2.GaussianBlur(
            mask.astype(np.float32) / 255.0,
            (0, 0),
            sigmaX=self.config.blend_edge_blur_sigma,
        )
        alpha = (blurred_mask * opacity)[:, :, None]

        # Determine crack color: mostly dark/brownish (ink/dust/cracks) or whitish (scratches)
        if random.random() < self.config.dark_crack_probability:
            # Dark crack with slight tonal jitter
            base_color = np.array([
                random.randint(10, 45),
                random.randint(10, 40),
                random.randint(10, 35),
            ], dtype=np.float32)
        else:
            # Whitish / light scratch
            base_color = np.array([
                random.randint(210, 245),
                random.randint(210, 245),
                random.randint(205, 240),
            ], dtype=np.float32)

        damaged = clean.astype(np.float32) * (1.0 - alpha) + base_color * alpha
        return validate_rgb_uint8(np.clip(damaged, 0, 255).astype(np.uint8)), mask

    def _add_aging_noise(self, image: np.ndarray) -> np.ndarray:
        """Add subtle Gaussian grain or sensor noise to simulate aging."""
        if random.random() > self.config.add_subtle_noise_probability:
            return image

        noise = np.random.normal(0.0, self.config.noise_sigma, image.shape).astype(np.float32)
        noisy = image.astype(np.float32) + noise
        return validate_rgb_uint8(np.clip(noisy, 0, 255).astype(np.uint8))

    def synthesize(
        self,
        clean_rgb: np.ndarray,
        crack_mask: np.ndarray,
    ) -> dict[str, Any]:
        """Synthesize a complete old photo restoration training sample.

        Args:
            clean_rgb: Clean reference RGB image (HxWx3, uint8).
            crack_mask: Binary or grayscale crack annotation mask (HxW, uint8).

        Returns:
            Dictionary containing:
                - 'damaged_rgb': Synthetically damaged RGB image (HxWx3, uint8).
                - 'clean_rgb': Ground truth clean RGB image (HxWx3, uint8).
                - 'mask': Ground truth binary crack mask (HxW, uint8).
                - 'metadata': Synthesis parameters and statistics.
        """
        clean = validate_rgb_uint8(clean_rgb)
        mask = self._ensure_binary_mask(crack_mask)

        clean_aug, mask_aug = self._apply_geometric_transforms(clean, mask)
        mask_jittered = self._morphological_jitter(mask_aug)
        damaged, final_mask = self._blend_cracks(clean_aug, mask_jittered)
        damaged = self._add_aging_noise(damaged)

        crack_ratio = float((final_mask > 0).mean())
        return {
            "damaged_rgb": damaged,
            "clean_rgb": clean_aug,
            "mask": final_mask,
            "metadata": {
                "crack_ratio": crack_ratio,
                "target_size": list(self.config.target_size),
                "dark_crack": bool(damaged[final_mask > 0].mean() < 128) if crack_ratio > 0 else True,
            },
        }
