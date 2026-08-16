from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class DenoiseConfig:
    enabled: bool = True
    strength: int = 3


@dataclass(frozen=True)
class ContrastConfig:
    enabled: bool = True
    clip_limit: float = 1.35
    tile_grid_size: tuple[int, int] = (8, 8)


@dataclass(frozen=True)
class ColorCorrectionConfig:
    enabled: bool = False
    strength: float = 0.30
    neutral_saturation_max: int = 80
    min_neutral_pixels: int = 64
    max_chroma_shift: float = 6.0


@dataclass(frozen=True)
class SharpenConfig:
    enabled: bool = True
    amount: float = 0.12
    sigma: float = 1.0


@dataclass(frozen=True)
class SafetyConfig:
    enabled: bool = False
    max_mean_pixel_change: float = 24.0
    max_mean_luminance_change: float = 10.0
    max_mean_chroma_change: float = 10.0
    min_blend_factor: float = 0.10


@dataclass(frozen=True)
class OutputConfig:
    image_filename: str = "restored_color.png"
    metadata_filename: str = "color_restoration_metadata.json"
    save_intermediates: bool = True


@dataclass(frozen=True)
class ModelConfig:
    checkpoint_path: str | None = None
    expected_sha256: str | None = None
    device: str = "auto"
    tile_size: int = 256
    overlap: int = 64
    tile_batch_size: int = 4
    required_quality_mode: str = "opencv_conservative"


@dataclass(frozen=True)
class InferenceControlConfig:
    enabled: bool = True
    mode: str = "lab_protected_fixed"
    l_strength: float = 0.35
    ab_strength: float = 0.65
    rgb_strength: float = 0.60
    chroma_thresholds: tuple[float, float, float] = (4.0, 8.0, 12.0)
    chroma_strengths: tuple[float, float, float, float] = (0.20, 0.40, 0.60, 0.70)


@dataclass(frozen=True)
class FinalColorConfig:
    enabled: bool = True
    method: str = "ccm"
    ccm_strength: float = 0.45
    ccm_max_gain_delta: float = 0.15
    ccm_saturation_max: float = 0.55
    ccm_luminance_percentiles: tuple[float, float] = (35.0, 95.0)
    ccm_min_sample_pixels: int = 256
    chroma_match_strength: float = 0.35
    chroma_match_max_shift: float = 8.0


@dataclass(frozen=True)
class ColorRestorationConfig:
    enabled: bool = True
    method: str = "model"
    denoise: DenoiseConfig = field(default_factory=DenoiseConfig)
    contrast: ContrastConfig = field(default_factory=ContrastConfig)
    color_correction: ColorCorrectionConfig = field(default_factory=ColorCorrectionConfig)
    sharpen: SharpenConfig = field(default_factory=SharpenConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    inference_control: InferenceControlConfig = field(default_factory=InferenceControlConfig)
    final_color: FinalColorConfig = field(default_factory=FinalColorConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def validate(self) -> None:
        if self.method not in {"model", "opencv_conservative", "opencv_neutral_fallback"}:
            raise ValueError(
                "method must be one of: model, opencv_conservative, opencv_neutral_fallback"
            )
        if not 0 <= self.denoise.strength <= 20:
            raise ValueError("denoise.strength must be in [0, 20]")
        if not 0 < self.contrast.clip_limit <= 4:
            raise ValueError("contrast.clip_limit must be in (0, 4]")
        if len(self.contrast.tile_grid_size) != 2 or min(self.contrast.tile_grid_size) <= 0:
            raise ValueError("contrast.tile_grid_size must contain two positive integers")
        if not 0 <= self.color_correction.strength <= 1:
            raise ValueError("color_correction.strength must be in [0, 1]")
        if not 0 <= self.color_correction.neutral_saturation_max <= 255:
            raise ValueError("color_correction.neutral_saturation_max must be in [0, 255]")
        if self.color_correction.min_neutral_pixels <= 0:
            raise ValueError("color_correction.min_neutral_pixels must be > 0")
        if not 0 <= self.color_correction.max_chroma_shift <= 32:
            raise ValueError("color_correction.max_chroma_shift must be in [0, 32]")
        if not 0 <= self.sharpen.amount <= 0.5:
            raise ValueError("sharpen.amount must be in [0, 0.5]")
        if self.sharpen.sigma <= 0:
            raise ValueError("sharpen.sigma must be > 0")
        for name in (
            "max_mean_pixel_change",
            "max_mean_luminance_change",
            "max_mean_chroma_change",
        ):
            if getattr(self.safety, name) <= 0:
                raise ValueError(f"safety.{name} must be > 0")
        if not 0 < self.safety.min_blend_factor <= 1:
            raise ValueError("safety.min_blend_factor must be in (0, 1]")
        if self.model.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("model.device must be one of: auto, cpu, cuda")
        if self.model.tile_size <= 0 or self.model.tile_size % 16 != 0:
            raise ValueError("model.tile_size must be positive and divisible by 16")
        if not 0 <= self.model.overlap < self.model.tile_size:
            raise ValueError("model.overlap must satisfy 0 <= overlap < tile_size")
        if self.model.tile_batch_size <= 0:
            raise ValueError("model.tile_batch_size must be > 0")
        if self.model.expected_sha256 is not None:
            digest = self.model.expected_sha256.lower()
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise ValueError("model.expected_sha256 must be a 64-character hexadecimal digest")
        if self.inference_control.mode not in {
            "lab_protected_fixed",
            "lab_protected_gated",
            "rgb_blend",
        }:
            raise ValueError(
                "inference_control.mode must be one of: "
                "lab_protected_fixed, lab_protected_gated, rgb_blend"
            )
        for name in ("l_strength", "ab_strength", "rgb_strength"):
            if not 0 <= getattr(self.inference_control, name) <= 1:
                raise ValueError(f"inference_control.{name} must be in [0, 1]")
        if (
            len(self.inference_control.chroma_thresholds) != 3
            or tuple(sorted(self.inference_control.chroma_thresholds))
            != self.inference_control.chroma_thresholds
        ):
            raise ValueError("inference_control.chroma_thresholds must contain three sorted values")
        if len(self.inference_control.chroma_strengths) != 4 or any(
            not 0 <= value <= 1 for value in self.inference_control.chroma_strengths
        ):
            raise ValueError("inference_control.chroma_strengths must contain four values in [0, 1]")
        if self.final_color.method not in {"ccm", "lab_chroma_match", "ccm_then_chroma_match"}:
            raise ValueError(
                "final_color.method must be one of: ccm, lab_chroma_match, ccm_then_chroma_match"
            )
        if not 0 <= self.final_color.ccm_strength <= 1:
            raise ValueError("final_color.ccm_strength must be in [0, 1]")
        if not 0 <= self.final_color.ccm_max_gain_delta <= 1:
            raise ValueError("final_color.ccm_max_gain_delta must be in [0, 1]")
        if not 0 <= self.final_color.ccm_saturation_max <= 1:
            raise ValueError("final_color.ccm_saturation_max must be in [0, 1]")
        low, high = self.final_color.ccm_luminance_percentiles
        if not 0 <= low < high <= 100:
            raise ValueError("final_color.ccm_luminance_percentiles must satisfy 0 <= low < high <= 100")
        if self.final_color.ccm_min_sample_pixels <= 0:
            raise ValueError("final_color.ccm_min_sample_pixels must be > 0")
        if not 0 <= self.final_color.chroma_match_strength <= 1:
            raise ValueError("final_color.chroma_match_strength must be in [0, 1]")
        if self.final_color.chroma_match_max_shift < 0:
            raise ValueError("final_color.chroma_match_max_shift must be >= 0")
        _validate_filename(self.output.image_filename, ".png")
        _validate_filename(self.output.metadata_filename, ".json")


def _validate_filename(value: str, suffix: str) -> None:
    path = Path(value)
    if not value or path.name != value or path.suffix.lower() != suffix:
        raise ValueError(f"Output filename must be a basename ending in {suffix}: {value!r}")


def _section(data: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"Config section {name!r} must be a mapping")
    return value


def color_restoration_config_from_dict(data: Mapping[str, Any]) -> ColorRestorationConfig:
    contrast = dict(_section(data, "contrast"))
    if "tile_grid_size" in contrast:
        contrast["tile_grid_size"] = tuple(int(item) for item in contrast["tile_grid_size"])
    inference_control = dict(_section(data, "inference_control"))
    if "chroma_thresholds" in inference_control:
        inference_control["chroma_thresholds"] = tuple(
            float(item) for item in inference_control["chroma_thresholds"]
        )
    if "chroma_strengths" in inference_control:
        inference_control["chroma_strengths"] = tuple(
            float(item) for item in inference_control["chroma_strengths"]
        )
    final_color = dict(_section(data, "final_color"))
    if "ccm_luminance_percentiles" in final_color:
        final_color["ccm_luminance_percentiles"] = tuple(
            float(item) for item in final_color["ccm_luminance_percentiles"]
        )
    config = ColorRestorationConfig(
        enabled=bool(data.get("enabled", True)),
        method=str(data.get("method", "model")),
        denoise=DenoiseConfig(**dict(_section(data, "denoise"))),
        contrast=ContrastConfig(**contrast),
        color_correction=ColorCorrectionConfig(**dict(_section(data, "color_correction"))),
        sharpen=SharpenConfig(**dict(_section(data, "sharpen"))),
        safety=SafetyConfig(**dict(_section(data, "safety"))),
        model=ModelConfig(**dict(_section(data, "model"))),
        inference_control=InferenceControlConfig(**inference_control),
        final_color=FinalColorConfig(**final_color),
        output=OutputConfig(**dict(_section(data, "output"))),
    )
    config.validate()
    return config


def load_color_restoration_config(path: str | Path) -> ColorRestorationConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Color restoration config not found: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError("Color restoration config root must be a mapping")
    return color_restoration_config_from_dict(payload)
