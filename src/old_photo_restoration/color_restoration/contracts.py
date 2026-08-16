from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ColorRestorationContext:
    """Optional lineage supplied by an upstream caller."""

    source_id: str | None = None
    upstream_run_id: str | None = None
    inpainting_backend: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "source_id": self.source_id,
            "upstream_run_id": self.upstream_run_id,
            "inpainting_backend": self.inpainting_backend,
        }


@dataclass
class ColorRestorationResult:
    """In-memory output contract. File IO is handled by a separate adapter."""

    image_rgb: np.ndarray
    metadata: dict[str, Any]
    intermediates: dict[str, np.ndarray] = field(default_factory=dict)

