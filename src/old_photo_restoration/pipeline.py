from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from old_photo_restoration.config import ProjectConfig
from old_photo_restoration.inpainting.lama_wrapper import LamaInpainter
from old_photo_restoration.postprocessing import PostInpaintingProcessor
from old_photo_restoration.segmentation.mask_refinement import build_hybrid_mask
from old_photo_restoration.segmentation.predictor import SegmentationPredictor
from old_photo_restoration.utils.run_logging import configure_run_logger


def _read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    if mask.ndim == 3:
        if mask.shape[2] == 4:
            mask = mask[:, :, :3]
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return np.where(mask > 127, 255, 0).astype(np.uint8)


def _write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), mask):
        raise RuntimeError(f"Could not write mask: {path}")


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
        self.segmenter = None # will be lazy initialized if needed

    def run(
        self,
        image_path: Path,
        output_dir: Path,
        mask_path: Path | None = None,
        face_mode: str = "off",
        golden_reference: Path | None = None,
        segmenter_arch: str = "r013_custom_attnunet",
        segmenter_checkpoint: Path | None = None,
        segmenter_threshold: float | None = None,
        segmenter_dilation: int | None = None,
        post_inpainting_enabled: bool = False,
        color_restoration_config_path: Path | None = None,
    ) -> PipelineResult:
        resolved_checkpoint = segmenter_checkpoint
        if segmenter_arch == "r014_resnet34" and not resolved_checkpoint:
            import os
            env_ckpt = os.environ.get("R014_SEGMENTER_CHECKPOINT")
            default_ckpt = (Path(__file__).resolve().parents[2] / "checkpoints/segmenter/seg-unet-resnet34-r014-s42/best_val_iou.pth").resolve()
            if env_ckpt:
                resolved_checkpoint = Path(env_ckpt)
            elif default_ckpt.exists():
                resolved_checkpoint = default_ckpt
            else:
                raise FileNotFoundError("R014 checkpoint not found. Set R014_SEGMENTER_CHECKPOINT or place the checkpoint at checkpoints/segmenter/seg-unet-resnet34-r014-s42/best_val_iou.pth.")
        
        if self.segmenter is None or self.segmenter.arch != segmenter_arch or (resolved_checkpoint and self.segmenter.checkpoint_path != Path(resolved_checkpoint)):
            self.segmenter = SegmentationPredictor(self.config, arch=segmenter_arch, checkpoint_override=resolved_checkpoint)
        resolved_image = image_path.resolve()
        resolved_output_dir = output_dir.resolve()

        if not resolved_image.exists():
            raise FileNotFoundError(f"Input image was not found: {resolved_image}")
        if face_mode != "off" and not post_inpainting_enabled:
            raise ValueError("face_mode requires post_inpainting_enabled=True.")
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
        logger = configure_run_logger(resolved_output_dir)
        final_mask_path = resolved_output_dir / "final_mask.png"
        auto_metadata: dict[str, Any] = {}

        if mask_path is None:
            if segmenter_threshold is not None:
                th = segmenter_threshold
            elif segmenter_arch == "r014_resnet34":
                th = 0.30
            else:
                th = float(self.config.checkpoint.threshold_balanced)
                
            if segmenter_dilation is not None:
                dil = segmenter_dilation
            elif segmenter_arch == "r014_resnet34":
                dil = 1
            else:
                dil = 0
                
            hybrid = build_hybrid_mask(
                image_path=resolved_image,
                predictor=self.segmenter,
                threshold=th,
                dilation_radius=dil,
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
                "segmentation_arch": segmenter_arch,
                "segmentation_dilation": dil,
            }
        else:
            resolved_mask = mask_path.resolve()
            if not resolved_mask.exists():
                raise FileNotFoundError(f"Input mask was not found: {resolved_mask}")
            final_mask = _read_mask(resolved_mask)
            _write_mask(final_mask_path, final_mask)

        inpainting_output_dir = resolved_output_dir / "inpainting"
        inpainting_output_dir.mkdir(parents=True, exist_ok=True)
        lama_output = self.inpainter.inpaint(
            image_path=resolved_image,
            mask_path=final_mask_path,
            output_dir=inpainting_output_dir,
        )
        inpainting_lama_path = inpainting_output_dir / "lama_restored.png"
        if Path(lama_output).resolve() != inpainting_lama_path.resolve():
            shutil.copy2(lama_output, inpainting_lama_path)
        lama_restored_path = resolved_output_dir / "lama_restored.png"
        shutil.copy2(inpainting_lama_path, lama_restored_path)
        restored_path = resolved_output_dir / "restored_before_face.png"
        shutil.copy2(lama_restored_path, restored_path)

        post_inpainting_metadata: dict[str, Any] = {
            "enabled": False,
            "status": "skipped",
            "reason": "disabled",
            "input_stage": "lama_restored",
            "output_stage": "restored_before_face",
            "artifacts": {
                "inpainting_dir": str(inpainting_output_dir),
                "lama_restored": str(inpainting_lama_path),
                "lama_restored_alias": str(lama_restored_path),
                "restored_before_face": str(restored_path),
            },
        }
        face_restoration_applied = False
        if post_inpainting_enabled:
            post_result = PostInpaintingProcessor(self.config, logger).run(
                inpainting_lama_path,
                resolved_output_dir,
                color_restoration_config_path=color_restoration_config_path,
                face_mode=face_mode,
            )
            restored_path = post_result.final_path
            face_restoration_applied = (
                post_result.metadata["modules"]["face_restoration"].get("status") == "applied"
            )
            post_inpainting_metadata = {
                **post_result.metadata,
                "artifacts": {
                    "inpainting_dir": str(inpainting_output_dir),
                    "lama_restored": str(inpainting_lama_path),
                    "lama_restored_alias": str(lama_restored_path),
                    **post_result.metadata["artifacts"],
                },
            }

        last_result = self.inpainter.last_result or {}
        metadata: dict[str, Any] = {
            "pipeline_mode_label": "auto_mask" if mask_path is None else "mask_bypass",
            "image_path": str(resolved_image),
            "mask_path": str(resolved_mask),
            "restored_path": str(restored_path),
            "output_dir": str(resolved_output_dir),
            "mask_source": "external_mask" if mask_path is not None else "union",
            "segmentation_enabled": mask_path is None,
            "inpainting_backend": "official_lama",
            "face_restoration_enabled": face_restoration_applied,
            "face_mode": face_mode,
            "post_inpainting": post_inpainting_metadata,
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
