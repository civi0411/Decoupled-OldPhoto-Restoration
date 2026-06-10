from __future__ import annotations

from collections.abc import Sequence


def get_best_device(prefer: Sequence[str] = ("cuda", "mps", "cpu")) -> str:
    try:
        import torch
    except Exception:
        return "cpu"

    for device in prefer:
        if device == "cuda" and torch.cuda.is_available():
            return "cuda"
        if device == "mps":
            mps_backend = getattr(torch.backends, "mps", None)
            if mps_backend is not None and mps_backend.is_available():
                return "mps"
        if device == "cpu":
            return "cpu"
    return "cpu"
