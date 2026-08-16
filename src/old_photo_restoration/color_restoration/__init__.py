"""Standalone color restoration for RGB images after inpainting."""

from __future__ import annotations

from .config import ColorRestorationConfig, load_color_restoration_config
from .contracts import ColorRestorationContext, ColorRestorationResult


def restore_color_after_lama(
    image_rgb,
    config: ColorRestorationConfig,
    context: ColorRestorationContext | None = None,
    runtime_dir=None,
) -> ColorRestorationResult:
    """Lazy public entry point so config/package imports do not require OpenCV."""
    from .core import restore_color_after_lama as _restore_color_after_lama

    return _restore_color_after_lama(image_rgb, config, context, runtime_dir)


__all__ = [
    "ColorRestorationConfig",
    "ColorRestorationContext",
    "ColorRestorationResult",
    "load_color_restoration_config",
    "restore_color_after_lama",
]
