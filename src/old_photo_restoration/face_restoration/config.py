from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FaceRestorationConfig:
    enabled: bool = False
    required: bool = False
    backend: str = "codeformer"
    repo_path: str | None = None
    env_name: str | None = None
    fidelity_weight: float = 0.70
    upscale: int = 1
    face_upsample: bool = False
    timeout_sec: int = 300

    def validate(self) -> None:
        if self.backend != "codeformer":
            raise ValueError("face restoration backend must be codeformer")
        if not 0 <= self.fidelity_weight <= 1:
            raise ValueError("fidelity_weight must be in [0, 1]")
        if self.upscale <= 0:
            raise ValueError("upscale must be > 0")
        if self.timeout_sec <= 0:
            raise ValueError("timeout_sec must be > 0")
