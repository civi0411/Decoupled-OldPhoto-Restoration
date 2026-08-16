from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_RUNS_MANIFEST = Path("artifacts/manifests/reproduction_runs_manifest.csv")


def configure_utf8_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Report available reproduction metadata for Module 1 training runs. "
            "Training datasets and checkpoint binaries are external artifacts described by manifests."
        )
    )
    subparsers = parser.add_subparsers(dest="command")

    status_parser = subparsers.add_parser(
        "status",
        help="Print reproduction metadata for the configured training runs.",
        description=(
            "Inspect reproduced runs, required datasets, and artifact lineage "
            "for the documented Module 1 training sequence."
        ),
    )
    status_parser.add_argument(
        "--runs-manifest",
        type=Path,
        default=DEFAULT_RUNS_MANIFEST,
        help="Path to the reproduction runs manifest CSV.",
    )
    status_parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root used to resolve relative paths.",
    )
    status_parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path for writing a JSON summary.",
    )
    status_parser.set_defaults(handler=run_status)
    return parser


def read_manifest(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    warnings: list[str] = []
    if not path.exists():
        warnings.append(f"Missing reproduction runs manifest: {path}")
        return [], warnings
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        warnings.append(f"Reproduction runs manifest is empty: {path}")
    return rows, warnings


def select_repro_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    keep = {"R010_REPRO", "R011_REPRO", "R012_REPRO", "R013_REPRO"}
    return [row for row in rows if row.get("run_id") in keep]


def write_json_output(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_status(args: argparse.Namespace) -> int:
    repo_root = args.repo_root.resolve()
    manifest_path = (repo_root / args.runs_manifest).resolve()
    rows, warnings = read_manifest(manifest_path)
    selected = select_repro_rows(rows)

    print("training_reproduction_metadata")
    print("training and fine-tuning artifacts are described via the reproduction manifest")
    print("manifest_path:", manifest_path)
    print("see_also: scripts/train/")
    for row in selected:
        print(
            f"- {row.get('run_id', '')}: dataset={row.get('dataset', '')} | "
            f"purpose={row.get('purpose', '')} | report_status={row.get('report_status', '')}"
        )
    if warnings:
        for item in warnings:
            print(f"[WARNING] {item}")

    payload = {
        "entrypoint": "training_evidence_status",
        "manifest_path": str(manifest_path),
        "warnings": warnings,
        "runs": selected,
        "notes": [
            "This command reports configured training runs and required artifacts.",
            "Use manifests and external paths to reproduce training runs.",
            "Training entrypoints are documented under scripts/train/.",
        ],
    }
    if args.json_output is not None:
        write_json_output(args.json_output.resolve(), payload)
    return 0


def main() -> int:
    configure_utf8_stdio()
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "handler"):
        parser.error("Missing subcommand. Use `status` to inspect reproduction metadata for training runs.")
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
