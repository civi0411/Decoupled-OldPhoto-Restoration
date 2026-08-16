from __future__ import annotations

from pathlib import Path

import numpy as np

from old_photo_restoration.face_restoration.codeformer_backend import (
    run_codeformer_restoration,
)
from old_photo_restoration.face_restoration.config import FaceRestorationConfig


def test_disabled_codeformer_is_explicit_pass_through() -> None:
    image = np.full((20, 24, 3), 128, dtype=np.uint8)

    output, metadata = run_codeformer_restoration(image, FaceRestorationConfig(enabled=False))

    assert np.array_equal(output, image)
    assert metadata["status"] == "skipped"
    assert metadata["reason"] == "disabled"


def test_codeformer_subprocess_adapter_with_fake_repo(tmp_path: Path) -> None:
    repo = tmp_path / "fake_codeformer"
    repo.mkdir()
    script = repo / "inference_codeformer.py"
    script.write_text(
        "\n".join(
            [
                "import argparse",
                "import shutil",
                "from pathlib import Path",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--input_path')",
                "parser.add_argument('--output_path')",
                "parser.add_argument('--fidelity_weight')",
                "parser.add_argument('--upscale')",
                "parser.add_argument('--face_upsample', action='store_true')",
                "args = parser.parse_args()",
                "output = Path(args.output_path) / 'final_results'",
                "output.mkdir(parents=True, exist_ok=True)",
                "shutil.copy2(args.input_path, output / Path(args.input_path).name)",
            ]
        ),
        encoding="utf-8",
    )
    image = np.full((21, 29, 3), [120, 130, 140], dtype=np.uint8)

    output, metadata = run_codeformer_restoration(
        image,
        FaceRestorationConfig(enabled=True, required=True, repo_path=str(repo)),
        work_dir=tmp_path / "runtime",
    )

    assert np.array_equal(output, image)
    assert metadata["status"] == "applied"
    assert metadata["fidelity_weight"] == 0.7
    assert metadata["output_persisted"] is True
