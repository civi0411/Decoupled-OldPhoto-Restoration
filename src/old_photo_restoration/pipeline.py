from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from old_photo_restoration.config import ProjectConfig
from old_photo_restoration.module2.lama_wrapper import LamaInpainter
from old_photo_restoration.module1.mask_refinement import build_hybrid_mask
from old_photo_restoration.module1.predictor import SegmentationPredictor


def _read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Không đọc được mask: {path}")
    if mask.ndim == 3:
        if mask.shape[2] == 4:
            mask = mask[:, :, :3]
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return np.where(mask > 127, 255, 0).astype(np.uint8)


def _write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), mask):
        raise RuntimeError(f"Không ghi được mask: {path}")


def _mask_ratio(mask: np.ndarray) -> float:
    return float((mask > 0).mean())


@dataclass(slots=True)
class PipelineResult:
    input_path: Path
    mask_path: Path
    restored_path: Path
    output_dir: Path
    metadata: dict[str, Any]


class RestorationPipeline:
    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.inpainter = LamaInpainter(config.lama)
        self.segmenter = SegmentationPredictor(config)

    def run(
        self,
        image_path: Path,
        output_dir: Path,
        mask_path: Path | None = None,
        face_mode: str = "off",
        golden_reference: Path | None = None,
    ) -> PipelineResult:
        resolved_image = image_path.resolve()
        resolved_output_dir = output_dir.resolve()

        if not resolved_image.exists():
            raise FileNotFoundError(f"Không tìm thấy ảnh đầu vào: {resolved_image}")
        if face_mode != "off":
            raise NotImplementedError("CodeFormer is not implemented in Phase 1C. Use --face-mode off.")
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
        final_mask_path = resolved_output_dir / "final_mask.png"
        auto_metadata: dict[str, Any] = {}

        if mask_path is None:
            hybrid = build_hybrid_mask(
                image_path=resolved_image,
                predictor=self.segmenter,
                threshold=float(self.config.checkpoint.threshold_balanced),
            )
            dl_mask_path = resolved_output_dir / "dl_mask.png"
            cv_mask_path = resolved_output_dir / "cv_mask.png"
            union_mask_path = resolved_output_dir / "union_before_refine.png"
            _write_mask(dl_mask_path, hybrid["dl_mask"])
            _write_mask(cv_mask_path, hybrid["cv_mask"])
            _write_mask(union_mask_path, hybrid["union_mask"])
            final_mask = hybrid["final_mask"]
            _write_mask(final_mask_path, final_mask)
            resolved_mask = final_mask_path
            auto_metadata = {
                "segmentation_enabled": True,
                "segmentation_model_version": "r013",
                "segmentation_checkpoint": str(self.config.checkpoint.expected_path),
                "segmentation_threshold": float(self.config.checkpoint.threshold_balanced),
                "mask_source": "union",
                "mask_refine": "repair_wide_v1",
                "dl_mask_ratio": hybrid["stats"]["dl_mask_ratio"],
                "cv_mask_ratio": hybrid["stats"]["cv_mask_ratio"],
                "union_before_refine_ratio": hybrid["stats"]["union_before_refine_ratio"],
                "final_mask_ratio": hybrid["stats"]["final_mask_ratio"],
                "rejected_cv_over_cv_ratio": hybrid["stats"]["rejected_cv_over_cv_ratio"],
                "cv_profile": "notebook_v7_candidate",
                "checkpoint_sha256": self.segmenter.checkpoint_sha256,
            }
        else:
            resolved_mask = mask_path.resolve()
            if not resolved_mask.exists():
                raise FileNotFoundError(f"Không tìm thấy mask đầu vào: {resolved_mask}")
            final_mask = _read_mask(resolved_mask)
            _write_mask(final_mask_path, final_mask)

        lama_output = self.inpainter.inpaint(
            image_path=resolved_image,
            mask_path=final_mask_path,
            output_dir=resolved_output_dir,
        )
        restored_path = resolved_output_dir / "restored_before_face.png"
        shutil.copy2(lama_output, restored_path)

        last_result = self.inpainter.last_result or {}
        metadata: dict[str, Any] = {
            "pipeline_phase": "2B_auto_mask" if mask_path is None else "1C_mask_bypass",
            "image_path": str(resolved_image),
            "mask_path": str(resolved_mask),
            "restored_path": str(restored_path),
            "output_dir": str(resolved_output_dir),
            "mask_source": "external_mask" if mask_path is not None else "union",
            "segmentation_enabled": mask_path is None,
            "inpainting_backend": "official_lama",
            "face_restoration_enabled": False,
            "face_mode": face_mode,
            "config_mode": self.config.inference.mode,
            "config_mask_source": self.config.inference.mask_source,
            "config_mask_refine": self.config.inference.mask_refine,
            "config_segmentation_threshold": self.config.inference.segmentation_threshold,
            "config_inpainting_backend": self.config.inference.inpainting_backend,
            "lama_env_actual": last_result.get("selected_env"),
            "lama_device_actual": last_result.get("selected_device"),
            "lama_backend_actual": "official_lama",
            "lama_result": last_result,
            "final_mask_ratio": _mask_ratio(final_mask),
            "golden_reference": None if golden_reference is None else str(golden_reference.resolve()),
        }
        metadata.update(auto_metadata)
        metadata_path = resolved_output_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

        return PipelineResult(
            input_path=resolved_image,
            mask_path=final_mask_path,
            restored_path=restored_path,
            output_dir=resolved_output_dir,
            metadata=metadata,
        )
