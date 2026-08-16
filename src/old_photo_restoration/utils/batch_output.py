from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .metadata import save_metadata


_INVALID_ITEM_ID = re.compile(r"[^A-Za-z0-9_-]+")
_MULTIPLE_UNDERSCORES = re.compile(r"_+")


def item_id_from_path(image_path: Path) -> str:
    normalized = _INVALID_ITEM_ID.sub("_", image_path.stem.strip())
    normalized = _MULTIPLE_UNDERSCORES.sub("_", normalized).strip("_-")
    return normalized or "image"


def unique_item_ids(image_paths: Iterable[Path], existing_items_dir: Path | None = None) -> list[str]:
    used = {
        path.name
        for path in existing_items_dir.iterdir()
        if path.is_dir()
    } if existing_items_dir is not None and existing_items_dir.is_dir() else set()
    item_ids: list[str] = []
    for image_path in image_paths:
        base_id = item_id_from_path(image_path)
        item_id = base_id
        if item_id in used:
            digest = hashlib.sha256(str(image_path.resolve()).encode("utf-8")).hexdigest()[:8]
            item_id = f"{base_id}_{digest}"
            counter = 2
            while item_id in used:
                item_id = f"{base_id}_{digest}_{counter}"
                counter += 1
        used.add(item_id)
        item_ids.append(item_id)
    return item_ids


@dataclass(slots=True)
class BatchOutput:
    batch_dir: Path
    items_dir: Path
    manifest_path: Path
    batch_id: str

    @classmethod
    def create(cls, batch_dir: Path) -> BatchOutput:
        resolved = batch_dir.resolve()
        items_dir = resolved / "items"
        items_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            batch_dir=resolved,
            items_dir=items_dir,
            manifest_path=resolved / "batch_manifest.json",
            batch_id=resolved.name,
        )

    def item_dir(self, item_id: str) -> Path:
        path = self.items_dir / item_id
        path.mkdir(parents=True, exist_ok=False)
        return path

    def write_manifest(self, items: list[dict[str, Any]]) -> None:
        completed = sum(item.get("status") == "completed" for item in items)
        failed = sum(item.get("status") == "failed" for item in items)
        if failed == 0:
            status = "completed"
        elif completed == 0:
            status = "failed"
        else:
            status = "completed_with_errors"
        save_metadata(
            self.manifest_path,
            {
                "schema_version": 1,
                "batch_id": self.batch_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "status": status,
                "summary": {
                    "total": len(items),
                    "completed": completed,
                    "failed": failed,
                },
                "items": items,
            },
        )
