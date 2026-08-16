from __future__ import annotations

import json
from pathlib import Path

from old_photo_restoration.utils.batch_output import (
    BatchOutput,
    item_id_from_path,
    unique_item_ids,
)


def test_item_id_defaults_to_filename_stem() -> None:
    assert item_id_from_path(Path("old_photo_001.jpg")) == "old_photo_001"
    assert item_id_from_path(Path("scan 2024 (final).png")) == "scan_2024_final"


def test_duplicate_stems_receive_stable_path_hash(tmp_path: Path) -> None:
    first = tmp_path / "a" / "old_photo_001.jpg"
    second = tmp_path / "b" / "old_photo_001.png"

    item_ids = unique_item_ids([first, second])

    assert item_ids[0] == "old_photo_001"
    assert item_ids[1].startswith("old_photo_001_")
    assert item_ids == unique_item_ids([first, second])


def test_batch_manifest_summarizes_items(tmp_path: Path) -> None:
    batch = BatchOutput.create(tmp_path / "batch_001")
    batch.write_manifest(
        [
            {"item_id": "old_photo_001", "status": "completed"},
            {"item_id": "old_photo_002", "status": "failed"},
        ]
    )

    payload = json.loads(batch.manifest_path.read_text(encoding="utf-8"))
    assert payload["batch_id"] == "batch_001"
    assert payload["status"] == "completed_with_errors"
    assert payload["summary"] == {"total": 2, "completed": 1, "failed": 1}
