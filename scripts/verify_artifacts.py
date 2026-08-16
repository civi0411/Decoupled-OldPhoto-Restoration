from __future__ import annotations

import argparse
import csv
import json
import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_CHECKPOINTS_MANIFEST = Path("artifacts/manifests/checkpoints_manifest.csv")
DEFAULT_DATASETS_MANIFEST = Path("artifacts/manifests/datasets_manifest.csv")


def configure_utf8_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify local artifacts against manifest registries without downloading, copying, "
            "or modifying the repository."
        )
    )
    subparsers = parser.add_subparsers(dest="command")

    for command_name in ("check-checkpoints", "check-datasets", "check-all"):
        subparser = subparsers.add_parser(
            command_name,
            help=f"Run verification mode: {command_name}.",
        )
        subparser.add_argument(
            "--checkpoints-manifest",
            type=Path,
            default=DEFAULT_CHECKPOINTS_MANIFEST,
            help="Path tới checkpoints manifest CSV.",
        )
        subparser.add_argument(
            "--datasets-manifest",
            type=Path,
            default=DEFAULT_DATASETS_MANIFEST,
            help="Path tới datasets manifest CSV.",
        )
        subparser.add_argument(
            "--repo-root",
            type=Path,
            default=Path("."),
            help="Repo root để resolve repo_relative_path và gọi Git read-only.",
        )
        subparser.add_argument(
            "--strict",
            action="store_true",
            help="Trả mã lỗi nếu có ERROR hoặc WARNING bắt buộc.",
        )
        subparser.add_argument(
            "--json-output",
            type=Path,
            default=None,
            help="Optional path to write a JSON summary.",
        )
        subparser.set_defaults(handler=run_command)

    parser.epilog = "Valid commands: check-checkpoints, check-datasets, check-all."
    return parser


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV manifest not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def resolve_repo_path(repo_root: Path, repo_relative_path: str) -> Path | None:
    normalized = repo_relative_path.strip()
    if not normalized:
        return None
    return (repo_root / Path(normalized)).resolve()


def is_truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def check_git_tracked(repo_root: Path, path: Path | None) -> bool:
    if path is None:
        return False
    try:
        relative = path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return False
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(relative).replace("\\", "/")],
        cwd=repo_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return result.returncode == 0


def add_result(
    results: list[dict[str, Any]],
    category: str,
    artifact_id: str,
    severity: str,
    message: str,
    **extra: Any,
) -> None:
    entry: dict[str, Any] = {
        "category": category,
        "artifact_id": artifact_id,
        "severity": severity,
        "message": message,
    }
    entry.update(extra)
    results.append(entry)


def evaluate_checkpoint_rows(repo_root: Path, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for row in rows:
        artifact_id = row["artifact_id"]
        local_path = resolve_repo_path(repo_root, row.get("repo_relative_path", ""))
        local_exists = local_path.exists() if local_path is not None else False
        tracked = check_git_tracked(repo_root, local_path)
        expected_local = is_truthy(row.get("exists_in_local_workspace", ""))
        policy = row.get("git_policy", "").strip()
        status = row.get("status", "").strip()
        expected_sha = row.get("expected_sha256", "").strip().lower()
        required_for_demo = is_truthy(row.get("required_for_demo", ""))

        if policy == "local_ignored_do_not_commit" and tracked:
            add_result(
                results,
                "checkpoint",
                artifact_id,
                "ERROR",
                "Checkpoint marked local_ignored_do_not_commit is currently tracked by Git.",
                path=str(local_path) if local_path is not None else "",
            )
            continue

        if local_exists:
            actual_sha = sha256_file(local_path)
            if expected_sha and actual_sha != expected_sha:
                add_result(
                    results,
                    "checkpoint",
                    artifact_id,
                    "ERROR",
                    "SHA256 mismatch against manifest.",
                    path=str(local_path),
                    expected_sha256=expected_sha,
                    actual_sha256=actual_sha,
                )
                continue
            add_result(
                results,
                "checkpoint",
                artifact_id,
                "OK",
                "Local artifact exists and matches manifest.",
                path=str(local_path),
                tracked_by_git=tracked,
                sha256=actual_sha,
            )
            continue

        if expected_local or required_for_demo:
            severity = "ERROR"
            message = "Required local artifact is missing."
        elif status == "not_found_or_historical_only":
            severity = "INFO"
            message = "Historical artifact only; missing locally is acceptable."
        else:
            severity = "INFO"
            message = "Artifact has no local copy; external source or optional path is acceptable."

        add_result(
            results,
            "checkpoint",
            artifact_id,
            severity,
            message,
            path=str(local_path) if local_path is not None else "",
            tracked_by_git=tracked,
        )
    return results


def evaluate_dataset_rows(repo_root: Path, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for row in rows:
        dataset_id = row["dataset_id"]
        local_path = resolve_repo_path(repo_root, row.get("repo_relative_path", ""))
        local_exists = local_path.exists() if local_path is not None else False
        tracked = check_git_tracked(repo_root, local_path)
        status = row.get("status", "").strip()
        git_policy = row.get("git_policy", "").strip()
        expected_tracked = row.get("tracked_by_git", "").strip().lower() == "yes"

        if local_path is None:
            add_result(
                results,
                "dataset",
                dataset_id,
                "INFO",
                "Dataset/artifact is documented as external-only or summary-only.",
                tracked_by_git=False,
            )
            continue

        if expected_tracked and not tracked:
            add_result(
                results,
                "dataset",
                dataset_id,
                "WARNING",
                "Manifest expects path to be tracked by Git, but Git does not confirm tracking.",
                path=str(local_path),
            )
            continue

        if not expected_tracked and tracked and "do_not_commit" in git_policy:
            add_result(
                results,
                "dataset",
                dataset_id,
                "ERROR",
                "Manifest marked do_not_commit, but path is currently tracked by Git.",
                path=str(local_path),
            )
            continue

        if local_exists:
            add_result(
                results,
                "dataset",
                dataset_id,
                "OK",
                "Local path exists matching dataset policy.",
                path=str(local_path),
                tracked_by_git=tracked,
            )
            continue

        severity = "INFO" if status in {"available_external_only", "available_local_only"} else "WARNING"
        add_result(
            results,
            "dataset",
            dataset_id,
            severity,
            "Local path does not exist; verify policy and external source if needed.",
            path=str(local_path),
            tracked_by_git=tracked,
        )
    return results


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {"OK": 0, "INFO": 0, "WARNING": 0, "ERROR": 0}
    for item in results:
        severity = item["severity"]
        counts[severity] = counts.get(severity, 0) + 1
    return {
        "counts": counts,
        "results": results,
    }


def should_fail_strict(results: list[dict[str, Any]]) -> bool:
    for item in results:
        if item["severity"] in {"ERROR", "WARNING"}:
            return True
    return False


def print_summary(command: str, summary: dict[str, Any]) -> None:
    counts = summary["counts"]
    print(f"command: {command}")
    print(f"ok: {counts.get('OK', 0)}")
    print(f"info: {counts.get('INFO', 0)}")
    print(f"warning: {counts.get('WARNING', 0)}")
    print(f"error: {counts.get('ERROR', 0)}")
    for item in summary["results"]:
        print(f"[{item['severity']}] {item['category']}::{item['artifact_id']} - {item['message']}")


def write_json_output(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_command(args: argparse.Namespace) -> int:
    repo_root = args.repo_root.resolve()
    payload: dict[str, Any] = {
        "command": args.command,
        "repo_root": str(repo_root),
    }
    all_results: list[dict[str, Any]] = []

    if args.command in {"check-checkpoints", "check-all"}:
        checkpoint_rows = read_csv_manifest((repo_root / args.checkpoints_manifest).resolve())
        checkpoint_results = evaluate_checkpoint_rows(repo_root, checkpoint_rows)
        payload["checkpoints"] = summarize_results(checkpoint_results)
        all_results.extend(checkpoint_results)

    if args.command in {"check-datasets", "check-all"}:
        dataset_rows = read_csv_manifest((repo_root / args.datasets_manifest).resolve())
        dataset_results = evaluate_dataset_rows(repo_root, dataset_rows)
        payload["datasets"] = summarize_results(dataset_results)
        all_results.extend(dataset_results)

    overall = summarize_results(all_results)
    payload["overall"] = overall
    print_summary(args.command, overall)

    if args.json_output is not None:
        write_json_output(args.json_output.resolve(), payload)

    if args.strict and should_fail_strict(all_results):
        return 1
    return 0


def main() -> int:
    configure_utf8_stdio()
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "handler"):
        parser.error("Missing subcommand. Use check-checkpoints, check-datasets, or check-all.")
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
