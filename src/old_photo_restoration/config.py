from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class LamaConfig:
    repo_root: Path
    checkpoint: Path
    conda_env_preferred: str = "lama_gpu"
    conda_env_fallback: str = "lama"

    @property
    def model_root(self) -> Path:
        return self.checkpoint.parents[1]

    @property
    def predict_script(self) -> Path:
        return self.repo_root / "bin" / "predict.py"


@dataclass(slots=True)
class CodeFormerConfig:
    repo_root: Path
    checkpoint: Path
    conda_env: str = "codeformer"


@dataclass(slots=True)
class CheckpointConfig:
    identifier: str
    name: str
    type: str
    expected_path: Path
    sha256: str
    threshold_balanced: float
    threshold_sensitive: float
    note: str = ""


@dataclass(slots=True)
class InferenceConfig:
    mode: str
    face_restoration: bool
    inpainting_backend: str
    mask_source: str
    mask_refine: str
    segmentation_threshold: float


@dataclass(slots=True)
class ProjectConfig:
    project_root: Path
    inference: InferenceConfig
    checkpoint: CheckpointConfig
    lama: LamaConfig
    codeformer: CodeFormerConfig
    warnings: list[str] = field(default_factory=list)
    external_config_path: Path | None = None


def _resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file YAML: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Nội dung YAML phải là object ở mức gốc: {path}")
    return payload


def load_config(
    inference_path: str | Path = "configs/inference.yaml",
    checkpoint_path: str | Path = "configs/checkpoints.yaml",
    external_path: str | Path = "configs/external_paths.yaml",
) -> ProjectConfig:
    project_root = Path(__file__).resolve().parents[2]
    inference_file = _resolve_path(project_root, inference_path)
    checkpoint_file = _resolve_path(project_root, checkpoint_path)
    external_file = _resolve_path(project_root, external_path)

    warnings_list: list[str] = []
    if external_file.exists():
        external_source = external_file
    else:
        external_source = _resolve_path(project_root, "configs/external_paths.example.yaml")
        warning_message = (
            "Không tìm thấy configs/external_paths.yaml; đang fallback sang "
            "configs/external_paths.example.yaml. Hãy tạo file local trước khi chạy trên máy khác."
        )
        warnings.warn(warning_message, stacklevel=2)
        warnings_list.append(warning_message)

    inference_payload = load_yaml(inference_file)
    checkpoints_payload = load_yaml(checkpoint_file)
    external_payload = load_yaml(external_source)

    if not checkpoints_payload:
        raise ValueError("configs/checkpoints.yaml không có checkpoint nào.")
    checkpoint_identifier, checkpoint_data = next(iter(checkpoints_payload.items()))
    if not isinstance(checkpoint_data, dict):
        raise ValueError(f"Cấu hình checkpoint không hợp lệ cho key {checkpoint_identifier}.")

    lama_payload = external_payload.get("lama")
    codeformer_payload = external_payload.get("codeformer")
    if not isinstance(lama_payload, dict):
        raise ValueError("Thiếu cấu hình `lama` trong external paths.")
    if not isinstance(codeformer_payload, dict):
        raise ValueError("Thiếu cấu hình `codeformer` trong external paths.")

    inference = InferenceConfig(
        mode=str(inference_payload["mode"]),
        face_restoration=bool(inference_payload["face_restoration"]),
        inpainting_backend=str(inference_payload["inpainting_backend"]),
        mask_source=str(inference_payload["mask_source"]),
        mask_refine=str(inference_payload["mask_refine"]),
        segmentation_threshold=float(inference_payload["segmentation_threshold"]),
    )
    checkpoint = CheckpointConfig(
        identifier=str(checkpoint_identifier),
        name=str(checkpoint_data["name"]),
        type=str(checkpoint_data["type"]),
        expected_path=_resolve_path(project_root, checkpoint_data["expected_path"]),
        sha256=str(checkpoint_data["sha256"]),
        threshold_balanced=float(checkpoint_data["threshold_balanced"]),
        threshold_sensitive=float(checkpoint_data["threshold_sensitive"]),
        note=str(checkpoint_data.get("note", "")),
    )
    lama = LamaConfig(
        repo_root=_resolve_path(project_root, lama_payload["repo_root"]),
        checkpoint=_resolve_path(project_root, lama_payload["checkpoint"]),
        conda_env_preferred=str(lama_payload.get("conda_env_preferred", "lama_gpu")),
        conda_env_fallback=str(lama_payload.get("conda_env_fallback", "lama")),
    )
    codeformer = CodeFormerConfig(
        repo_root=_resolve_path(project_root, codeformer_payload["repo_root"]),
        checkpoint=_resolve_path(project_root, codeformer_payload["checkpoint"]),
        conda_env=str(codeformer_payload.get("conda_env", "codeformer")),
    )
    return ProjectConfig(
        project_root=project_root,
        inference=inference,
        checkpoint=checkpoint,
        lama=lama,
        codeformer=codeformer,
        warnings=warnings_list,
        external_config_path=external_source,
    )
