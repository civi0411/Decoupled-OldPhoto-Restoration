from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def configure_utf8_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize the documented ablation protocol and available artifact coverage. "
            "This command does not execute the restoration pipeline."
        )
    )
    parser.add_argument(
        "--mode",
        default="protocol_status",
        choices=["protocol_status", "smoke_manifest"],
        help="Describe protocol scope and artifact coverage only; no pipeline execution is performed.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for writing the protocol summary files.",
    )
    parser.add_argument(
        "--write-protocol-status",
        action="store_true",
        help="Write JSON/Markdown protocol summaries when --output-dir is provided.",
    )
    parser.add_argument(
        "--available-artifacts-root",
        type=Path,
        default=None,
        help="Optional root used to check whether documented artifacts exist. No pipeline execution is performed.",
    )
    return parser


def resolve_optional_path(path: Path | None) -> Path | None:
    return None if path is None else path.resolve()


def build_protocol_definition() -> list[dict[str, str]]:
    return [
        {
            "case_id": "opencv_baseline",
            "description": "OpenCV baseline if the repository has the required backend and matching artifacts.",
            "status": "protocol_only",
        },
        {
            "case_id": "unet_mask_plus_classical_inpaint",
            "description": "Segmentation mask combined with a classical inpainting backend when a real implementation exists.",
            "status": "protocol_only",
        },
        {
            "case_id": "ground_truth_mask_plus_pretrained_lama",
            "description": "Oracle or ground-truth mask with official/pretrained LaMa.",
            "status": "partially evidenced by smoke artifacts",
        },
        {
            "case_id": "predicted_mask_plus_pretrained_lama",
            "description": "Predicted or hybrid mask with official/pretrained LaMa.",
            "status": "partially evidenced by smoke artifacts",
        },
        {
            "case_id": "full_pipeline_plus_optional_face_module",
            "description": "Full pipeline with the optional face module if stronger evidence is added later.",
            "status": "future_work_or_optional",
        },
    ]


def build_known_limitations() -> list[str]:
    return [
        "Full quantitative ablation is not yet packaged as portable artifacts in the repository.",
        "LPIPS, FID, and masked-region LPIPS do not yet have enough artifacts to support completed claims.",
        "LaMa is currently used as an official/pretrained wrapper, not as a fine-tuned model.",
        "Module 3 and CodeFormer should not be described as identity-preserving.",
        "This command summarizes protocol scope and artifact coverage; it is not a full experimental orchestrator.",
    ]


def inspect_artifacts(root: Path | None) -> list[dict[str, str]]:
    if root is None:
        return []
    targets = [
        "examples/outputs/seg_smoke_demo3/metadata.json",
        "examples/outputs/pipeline_smoke_demo3/metadata.json",
        "examples/outputs/gradio_smoke_demo3/metadata.json",
        "examples/outputs/readiness_mask_bypass_regression/metadata.json",
    ]
    rows: list[dict[str, str]] = []
    for target in targets:
        candidate = root / target
        rows.append(
            {
                "path": str(candidate),
                "exists": "yes" if candidate.exists() else "no",
            }
        )
    return rows


def build_status_payload(mode: str, artifacts_root: Path | None) -> dict[str, object]:
    return {
        "mode": mode,
        "runner_scope": "protocol_status_only",
        "protocol_cases": build_protocol_definition(),
        "known_limitations": build_known_limitations(),
        "artifacts_root_checked": str(artifacts_root) if artifacts_root is not None else None,
        "artifact_checks": inspect_artifacts(artifacts_root),
        "runner_note": (
            "This command documents the ablation protocol and current artifact coverage. "
            "Additional evaluation runners should be added only when matching evidence is available."
        ),
    }


def render_markdown(payload: dict[str, object]) -> str:
    protocol_cases = payload["protocol_cases"]
    artifact_checks = payload["artifact_checks"]
    lines = [
        "# Ablation Protocol Status",
        "",
        f"- mode: `{payload['mode']}`",
        f"- runner_scope: `{payload['runner_scope']}`",
        "",
        "## Protocol Cases",
    ]
    for item in protocol_cases:
        assert isinstance(item, dict)
        lines.append(f"- `{item['case_id']}`: {item['description']} | status={item['status']}")
    lines.extend(["", "## Known Limitations"])
    for item in payload["known_limitations"]:
        lines.append(f"- {item}")
    if artifact_checks:
        lines.extend(["", "## Artifact Checks"])
        for item in artifact_checks:
            assert isinstance(item, dict)
            lines.append(f"- `{item['path']}` -> exists={item['exists']}")
    return "\n".join(lines)


def write_protocol_status(output_dir: Path, payload: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ablation_protocol_status.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "ablation_protocol_status.md").write_text(
        render_markdown(payload),
        encoding="utf-8",
    )


def main() -> int:
    configure_utf8_stdio()
    args = build_parser().parse_args()
    output_dir = resolve_optional_path(args.output_dir)
    artifacts_root = resolve_optional_path(args.available_artifacts_root)

    if args.write_protocol_status and output_dir is None:
        raise SystemExit("Missing --output-dir while using --write-protocol-status.")

    payload = build_status_payload(args.mode, artifacts_root)
    should_write = args.write_protocol_status or output_dir is not None
    if should_write:
        assert output_dir is not None
        write_protocol_status(output_dir, payload)
        print(f"ablation_protocol_status_json: {output_dir / 'ablation_protocol_status.json'}")
        print(f"ablation_protocol_status_md: {output_dir / 'ablation_protocol_status.md'}")
    else:
        print("protocol_status_only")
        print("This command does not execute a full ablation run.")
        print("Use --output-dir to write the protocol summary to files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
