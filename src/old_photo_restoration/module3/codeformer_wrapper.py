from __future__ import annotations

from pathlib import Path


class CodeFormerWrapper:
    def __init__(self, repo_root: Path, checkpoint: Path, conda_env: str = "codeformer") -> None:
        self.repo_root = repo_root
        self.checkpoint = checkpoint
        self.conda_env = conda_env

    def run(self, image_path: Path, output_dir: Path, fidelity: float = 0.7) -> Path:
        raise NotImplementedError("CodeFormer wrapper mới chưa được migrate trong phase này.")
