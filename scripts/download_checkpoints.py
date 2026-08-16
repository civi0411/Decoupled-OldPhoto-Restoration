from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_CHECKPOINTS_MANIFEST = Path("artifacts/manifests/checkpoints_manifest.csv")


def configure_utf8_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect checkpoint requirements and local artifact policy from manifests. "
            "Checkpoint binaries are intentionally excluded from Git."
        )
    )
    subparsers = parser.add_subparsers(dest="command")

    status_parser = subparsers.add_parser(
        "status",
        help="Print checkpoint requirements and artifact policy from the configured manifest.",
        description=(
            "Inspect checkpoint policy and use verify_artifacts.py to validate local artifacts "
            "instead of downloading or copying checkpoint binaries."
        ),
    )
    status_parser.add_argument(
        "--checkpoints-manifest",
        type=Path,
        default=DEFAULT_CHECKPOINTS_MANIFEST,
        help="Path to the checkpoints manifest CSV.",
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
        warnings.append(f"Missing checkpoints manifest: {path}")
        return [], warnings
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        warnings.append(f"Checkpoints manifest is empty: {path}")
    return rows, warnings


def find_r013_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    for row in rows:
        if row.get("run_id") == "R013_REPRO" and "local operational checkpoint" in row.get("role", ""):
            return row
    return None


def write_json_output(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_status(args: argparse.Namespace) -> int:
    repo_root = args.repo_root.resolve()
    manifest_path = (repo_root / args.checkpoints_manifest).resolve()
    rows, warnings = read_manifest(manifest_path)
    r013_row = find_r013_row(rows)
    runtime_warnings = list(warnings)
    if r013_row is None:
        runtime_warnings.append("R013_REPRO row is missing from the checkpoints manifest.")

    print("checkpoint_artifact_inspection")
    print("checkpoint binaries are local ignored artifacts and should not be committed")
    print("manifest_path:", manifest_path)
    print("use_verify_command:")
    print("- python -B scripts/verify_artifacts.py check-checkpoints --repo-root .")

    if r013_row is not None:
        print("R013_REPRO is the current final Module 1 segmenter checkpoint according to the checkpoints manifest.")
        print("r013_role:", r013_row.get("role", ""))
        print("r013_status:", r013_row.get("status", ""))
        print("r013_local_path:", r013_row.get("repo_relative_path", ""))
        print("r013_policy:", r013_row.get("git_policy", ""))
        if r013_row.get("notes", "").strip():
            print("r013_notes:", r013_row.get("notes", ""))
    if runtime_warnings:
        for item in runtime_warnings:
            print(f"[WARNING] {item}")

    payload = {
        "entrypoint": "checkpoint_status",
        "manifest_path": str(manifest_path),
        "warnings": runtime_warnings,
        "r013_row": r013_row,
        "notes": [
            "Checkpoint binaries are local ignored artifacts.",
            "Do not commit checkpoint binaries.",
            "Use scripts/verify_artifacts.py check-checkpoints for verification.",
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
        parser.error("Missing subcommand. Use `status` to inspect checkpoint requirements and artifact policy.")
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
