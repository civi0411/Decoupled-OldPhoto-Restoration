from __future__ import annotations

import json
from types import SimpleNamespace

import cv2
import numpy as np

from scripts import run_pipeline


class FakePipeline:
    def __init__(self, config) -> None:
        self.config = config

    def run(self, image_path, output_dir, **kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        restored_path = output_dir / "restored.png"
        mask_path = output_dir / "final_mask.png"
        assert cv2.imwrite(str(restored_path), image)
        assert cv2.imwrite(str(mask_path), np.zeros(image.shape[:2], dtype=np.uint8))
        metadata = {"status": "completed"}
        (output_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        return SimpleNamespace(
            restored_path=restored_path,
            mask_path=mask_path,
            output_dir=output_dir,
            metadata=metadata,
        )


def test_cli_creates_item_named_after_input_stem(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "old_photo_001.jpg"
    assert cv2.imwrite(str(image_path), np.full((12, 16, 3), 128, dtype=np.uint8))
    batch_dir = tmp_path / "batch_001"
    args = SimpleNamespace(
        image=[image_path],
        mask=None,
        output_dir=batch_dir,
        config=tmp_path / "inference.yaml",
        checkpoint_config=tmp_path / "checkpoints.yaml",
        external_config=tmp_path / "external.yaml",
        face_mode="off",
        reference=None,
        reference_mask=None,
        segmenter_arch="r013_custom_attnunet",
        segmenter_checkpoint=None,
        segmenter_threshold=None,
        segmenter_dilation=None,
        post_inpainting=False,
        color_restoration_config=tmp_path / "color.yaml",
    )
    monkeypatch.setattr(run_pipeline, "build_parser", lambda: SimpleNamespace(parse_args=lambda: args))
    monkeypatch.setattr(run_pipeline, "load_config", lambda **kwargs: SimpleNamespace())
    monkeypatch.setattr(run_pipeline, "RestorationPipeline", FakePipeline)

    assert run_pipeline.main() == 0

    item_dir = batch_dir / "items" / "old_photo_001"
    assert (item_dir / "input" / "original.jpg").is_file()
    assert (item_dir / "artifacts" / "restored.png").is_file()
    assert (item_dir / "final.png").is_file()
    assert (item_dir / "manifest.json").is_file()
    batch_manifest = json.loads((batch_dir / "batch_manifest.json").read_text(encoding="utf-8"))
    assert batch_manifest["summary"] == {"total": 1, "completed": 1, "failed": 0}
