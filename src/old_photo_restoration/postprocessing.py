from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from old_photo_restoration.color_restoration import (
    ColorRestorationContext,
    load_color_restoration_config,
    restore_color_after_lama,
)
from old_photo_restoration.color_restoration.io import (
    load_rgb_image,
    save_color_restoration_result,
)
from old_photo_restoration.face_restoration import FaceRestorationConfig, restore_faces
from old_photo_restoration.face_restoration.io import save_face_restoration_result
from old_photo_restoration.utils.image_io import write_image_rgb


@dataclass(slots=True)
class PostProcessingResult:
    final_path: Path
    color_restored_path: Path
    face_restored_path: Path
    metadata: dict[str, Any]


class PostInpaintingProcessor:
    def __init__(self, project_config: Any, logger: logging.Logger) -> None:
        self.project_config = project_config
        self.logger = logger

    def run(
        self,
        image_path: Path,
        output_dir: Path,
        *,
        color_restoration_config_path: Path | None,
        face_mode: str,
    ) -> PostProcessingResult:
        color_output_dir = output_dir / "color_restoration"
        face_output_dir = output_dir / "face_restoration"
        final_output_dir = output_dir / "final"
        self.logger.info("stage=color_restoration status=started")

        color_config_path = color_restoration_config_path or (
            self.project_config.project_root / "configs" / "color_restoration.yaml"
        )
        color_config = load_color_restoration_config(color_config_path)
        checkpoint_path = color_config.model.checkpoint_path
        if checkpoint_path and not Path(checkpoint_path).is_absolute():
            checkpoint_path = str((self.project_config.project_root / checkpoint_path).resolve())
        color_config = replace(
            color_config,
            model=replace(color_config.model, checkpoint_path=checkpoint_path),
        )
        color_result = restore_color_after_lama(
            load_rgb_image(image_path),
            color_config,
            context=ColorRestorationContext(
                source_id=str(image_path),
                inpainting_backend="official_lama",
            ),
            runtime_dir=color_output_dir / "_runtime",
        )
        color_artifacts = save_color_restoration_result(
            color_result,
            color_output_dir,
            color_config,
        )
        color_restored_path = Path(color_artifacts["image"])
        self.logger.info(
            "stage=color_restoration status=%s output=%s",
            color_result.metadata.get("status", "applied"),
            color_restored_path,
        )

        face_config = FaceRestorationConfig(
            enabled=face_mode != "off",
            repo_path=str(self.project_config.codeformer.repo_root),
            env_name=self.project_config.codeformer.conda_env,
        )
        self.logger.info("stage=face_restoration status=started enabled=%s", face_config.enabled)
        face_result = restore_faces(
            color_result.image_rgb,
            face_config,
            runtime_dir=face_output_dir / "_runtime",
        )
        face_artifacts = save_face_restoration_result(face_result, face_output_dir)
        face_restored_path = Path(face_artifacts["image"])
        self.logger.info(
            "stage=face_restoration status=%s output=%s",
            face_result.metadata.get("status", "skipped"),
            face_restored_path,
        )

        final_output_dir.mkdir(parents=True, exist_ok=True)
        final_path = final_output_dir / "restored.png"
        write_image_rgb(final_path, face_result.image_rgb)

        # Compatibility aliases used by existing CLI, Gradio, and golden-reference tooling.
        shutil.copy2(color_restored_path, output_dir / "restored_before_face.png")
        shutil.copy2(face_restored_path, output_dir / "face_restored.png")
        shutil.copy2(final_path, output_dir / "restored_color.png")
        return PostProcessingResult(
            final_path=final_path,
            color_restored_path=color_restored_path,
            face_restored_path=face_restored_path,
            metadata={
                "enabled": True,
                "status": "applied",
                "input_stage": "lama_restored",
                "output_stage": "final",
                "face_mode": face_mode,
                "modules": {
                    "color_restoration": color_result.metadata,
                    "face_restoration": face_result.metadata,
                },
                "artifacts": {
                    "color_restoration_dir": str(color_output_dir),
                    "color_restored": str(color_restored_path),
                    "face_restoration_dir": str(face_output_dir),
                    "face_restored": str(face_restored_path),
                    "final": str(final_path),
                    "color_metadata": str(color_artifacts["metadata"]),
                    "face_metadata": str(face_artifacts["metadata"]),
                    "pipeline_log": str(output_dir / "logs" / "pipeline.log"),
                },
            },
        )
