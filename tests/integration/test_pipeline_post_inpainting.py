from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np

from old_photo_restoration.pipeline import RestorationPipeline


class FakeInpainter:
    def __init__(self) -> None:
        self.last_result = {"selected_env": "test", "selected_device": "cpu"}

    def inpaint(self, image_path, mask_path, output_dir):
        return image_path


def _write_image(path, image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)


def _project_config(project_root):
    return SimpleNamespace(
        project_root=project_root,
        lama=SimpleNamespace(),
        codeformer=SimpleNamespace(repo_root=project_root / "codeformer", conda_env="codeformer"),
        checkpoint=SimpleNamespace(
            expected_path=project_root / "segmenter.pth",
            sha256="0" * 64,
            threshold_balanced=0.5,
        ),
        inference=SimpleNamespace(
            mode="test",
            mask_source="external_mask",
            mask_refine="none",
            segmentation_threshold=0.5,
            inpainting_backend="official_lama",
        ),
    )


def test_pipeline_post_inpainting_is_optional_and_preserves_stage_lineage(tmp_path) -> None:
    image_path = tmp_path / "input.png"
    mask_path = tmp_path / "mask.png"
    _write_image(image_path, np.full((32, 40, 3), [100, 120, 140], dtype=np.uint8))
    _write_image(mask_path, np.zeros((32, 40), dtype=np.uint8))

    pipeline = RestorationPipeline(_project_config(tmp_path))
    pipeline.inpainter = FakeInpainter()

    disabled = pipeline.run(
        image_path=image_path,
        mask_path=mask_path,
        output_dir=tmp_path / "disabled",
    )
    assert disabled.restored_path.name == "restored_before_face.png"
    assert disabled.metadata["post_inpainting"]["status"] == "skipped"
    assert (disabled.output_dir / "lama_restored.png").is_file()
    assert (disabled.output_dir / "inpainting" / "lama_restored.png").is_file()

    color_config_path = tmp_path / "color_restoration.yaml"
    color_config_path.write_text(
        "\n".join(
            [
                "enabled: true",
                "method: opencv_conservative",
                "final_color:",
                "  enabled: true",
                "  method: ccm",
                "output:",
                "  save_intermediates: true",
            ]
        ),
        encoding="utf-8",
    )
    enabled = pipeline.run(
        image_path=image_path,
        mask_path=mask_path,
        output_dir=tmp_path / "enabled",
        post_inpainting_enabled=True,
        color_restoration_config_path=color_config_path,
    )
    assert enabled.restored_path.name == "restored.png"
    assert enabled.metadata["post_inpainting"]["status"] == "applied"
    assert enabled.metadata["post_inpainting"]["input_stage"] == "lama_restored"
    assert enabled.metadata["post_inpainting"]["output_stage"] == "final"
    assert (
        enabled.metadata["post_inpainting"]["modules"]["color_restoration"]["pipeline_order"]
        == [
            "quality_restoration",
            "color_restoration_model",
            "inference_control",
            "ccm_color_correction",
            "safety_postprocessing",
        ]
    )
    assert enabled.metadata["post_inpainting"]["modules"]["face_restoration"]["status"] == "skipped"
    assert (enabled.output_dir / "restored_before_face.png").is_file()
    assert (enabled.output_dir / "face_restored.png").is_file()
    assert (enabled.output_dir / "color_restoration" / "color_restoration_metadata.json").is_file()
    assert (enabled.output_dir / "color_restoration" / "quality_restored.png").is_file()
    assert (enabled.output_dir / "color_restoration" / "inference_controlled.png").is_file()
    assert (enabled.output_dir / "color_restoration" / "ccm_corrected.png").is_file()
    assert (enabled.output_dir / "face_restoration" / "face_restoration_metadata.json").is_file()
    assert (enabled.output_dir / "face_restoration" / "codeformer_output.png").is_file()
    assert (enabled.output_dir / "final" / "restored.png").is_file()
    assert (enabled.output_dir / "logs" / "pipeline.log").is_file()
