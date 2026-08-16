from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class FaceRestorationResult:
    image_rgb: np.ndarray
    metadata: dict[str, Any]
    intermediates: dict[str, np.ndarray] = field(default_factory=dict)
