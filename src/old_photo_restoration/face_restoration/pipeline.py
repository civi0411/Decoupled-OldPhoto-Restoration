from __future__ import annotations

from pathlib import Path

import numpy as np

from .codeformer_backend import run_codeformer_restoration
from .config import FaceRestorationConfig
from .contracts import FaceRestorationResult


def restore_faces(
    image_rgb: np.ndarray,
    config: FaceRestorationConfig,
    runtime_dir: str | Path | None = None,
) -> FaceRestorationResult:
    restored, stage_metadata = run_codeformer_restoration(
        image_rgb,
        config,
        work_dir=runtime_dir,
    )
    return FaceRestorationResult(
        image_rgb=restored,
        metadata={
            "feature": "face_restoration",
            "status": stage_metadata["status"],
            "backend": config.backend,
            "stages": [stage_metadata],
            "warnings": stage_metadata.get("warnings", []),
        },
        intermediates={"codeformer_output": restored},
    )
