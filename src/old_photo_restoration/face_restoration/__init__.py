"""Module face restoration."""
from .config import FaceRestorationConfig
from .contracts import FaceRestorationResult
from .pipeline import restore_faces

__all__ = [
    "FaceRestorationConfig",
    "FaceRestorationResult",
    "restore_faces",
]
