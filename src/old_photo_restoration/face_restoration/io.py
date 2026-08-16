from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from old_photo_restoration.utils.image_io import write_image_rgb

from .contracts import FaceRestorationResult


def save_face_restoration_result(
    result: FaceRestorationResult,
    output_dir: str | Path,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    image_path = output_path / "codeformer_output.png"
    write_image_rgb(image_path, result.image_rgb)
    metadata_path = output_path / "face_restoration_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                **result.metadata,
                "artifacts": {
                    "codeformer_output": str(image_path),
                    "metadata": str(metadata_path),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"image": image_path, "metadata": metadata_path}
