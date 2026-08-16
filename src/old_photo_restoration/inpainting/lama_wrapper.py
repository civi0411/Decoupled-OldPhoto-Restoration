from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from old_photo_restoration.config import LamaConfig


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _tail(text: str, limit: int = 2500) -> str:
    return text[-limit:] if text else ""


def _extract_json_from_stdout(stdout: str) -> dict[str, Any] | None:
    for line in reversed((stdout or "").splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def _read_color(path: Path) -> np.ndarray | None:
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def _read_mask(path: Path) -> np.ndarray | None:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        return None
    if mask.ndim == 3:
        if mask.shape[2] == 4:
            mask = mask[:, :, :3]
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return np.where(mask > 127, 255, 0).astype(np.uint8)


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Không ghi được file: {path}")


def _list_output_images(output_dir: Path) -> list[Path]:
    if not output_dir.exists():
        return []
    return sorted(
        [path for path in output_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def _select_output(raw_output_dir: Path, input_stem: str) -> Path | None:
    preferred_names = [
        raw_output_dir / f"{input_stem}_mask.png",
        raw_output_dir / f"{input_stem}_mask.jpg",
        raw_output_dir / f"{input_stem}.png",
        raw_output_dir / f"{input_stem}.jpg",
    ]
    for candidate in preferred_names:
        if candidate.exists():
            return candidate
    images = _list_output_images(raw_output_dir)
    return images[0] if images else None


class LamaInpainter:
    def __init__(self, config: LamaConfig) -> None:
        self.config = config
        self.last_result: dict[str, Any] | None = None

    def _probe_env(self, env_name: str, timeout_sec: int = 30) -> dict[str, Any]:
        script = (
            "import json, torch; "
            "print(json.dumps({"
            "'available': True, "
            "'reason': 'available', "
            "'torch_version': torch.__version__, "
            "'cuda_build': torch.version.cuda, "
            "'cuda_available': bool(torch.cuda.is_available()), "
            "'device_count': int(torch.cuda.device_count()), "
            "'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None"
            "}))"
        )
        command = ["conda", "run", "-n", env_name, "python", "-c", script]
        env = os.environ.copy()
        env["CONDA_NO_PLUGINS"] = "true"
        try:
            completed = subprocess.run(
                command,
                env=env,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired:
            return {"env": env_name, "available": False, "reason": "probe_timeout"}
        except Exception as exc:
            return {"env": env_name, "available": False, "reason": f"probe_exception: {exc}"}

        payload = _extract_json_from_stdout(completed.stdout) or {}
        payload["env"] = env_name
        payload["returncode"] = completed.returncode
        payload["stdout_tail"] = _tail(completed.stdout, 800)
        payload["stderr_tail"] = _tail(completed.stderr, 800)
        if completed.returncode != 0:
            payload["available"] = False
            payload["reason"] = payload.get("reason") or "conda_env_unavailable"
        return payload

    def readiness(self) -> dict[str, Any]:
        repo_exists = self.config.repo_root.exists()
        predict_exists = self.config.predict_script.exists()
        checkpoint_exists = self.config.checkpoint.exists()
        preferred_probe = self._probe_env(self.config.conda_env_preferred)
        fallback_probe = self._probe_env(self.config.conda_env_fallback)
        if preferred_probe.get("available") and preferred_probe.get("cuda_available"):
            selected_env = self.config.conda_env_preferred
            selected_device = "cuda"
            status = "gpu"
            available = True
            reason = "preferred_env_gpu_available"
        elif fallback_probe.get("available"):
            selected_env = self.config.conda_env_fallback
            selected_device = "cpu"
            status = "cpu-only"
            available = True
            reason = "fallback_env_available"
        else:
            selected_env = None
            selected_device = None
            status = "unavailable"
            available = False
            reason = "no_usable_lama_env"
        return {
            "available": available and repo_exists and predict_exists and checkpoint_exists,
            "status": status,
            "reason": reason,
            "repo_root": str(self.config.repo_root),
            "predict_script": str(self.config.predict_script),
            "checkpoint": str(self.config.checkpoint),
            "model_root": str(self.config.model_root),
            "repo_exists": repo_exists,
            "predict_script_exists": predict_exists,
            "checkpoint_exists": checkpoint_exists,
            "preferred_env": self.config.conda_env_preferred,
            "fallback_env": self.config.conda_env_fallback,
            "selected_env": selected_env,
            "selected_device": selected_device,
            "preferred_probe": preferred_probe,
            "fallback_probe": fallback_probe,
        }

    def _build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["CONDA_NO_PLUGINS"] = "true"
        existing_pythonpath = env.get("PYTHONPATH", "")
        repo_text = str(self.config.repo_root)
        env["PYTHONPATH"] = repo_text if not existing_pythonpath else f"{repo_text}{os.pathsep}{existing_pythonpath}"
        return env

    def _prepare_input(self, image_path: Path, mask_path: Path, output_dir: Path) -> tuple[Path, Path, Path]:
        image = _read_color(image_path)
        mask = _read_mask(mask_path)
        if image is None:
            raise FileNotFoundError(f"Không đọc được input image: {image_path}")
        if mask is None:
            raise FileNotFoundError(f"Không đọc được mask: {mask_path}")
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(f"Size mismatch: image H/W={image.shape[:2]}, mask H/W={mask.shape[:2]}")

        run_id = f"{image_path.stem}_{int(time.time() * 1000)}"
        prepared_dir = output_dir / "_official_lama_input" / run_id
        prepared_dir.mkdir(parents=True, exist_ok=True)
        prepared_image = prepared_dir / f"{image_path.stem}.png"
        prepared_mask = prepared_dir / f"{image_path.stem}_mask.png"
        _write_png(prepared_image, image)
        _write_png(prepared_mask, mask)
        return prepared_dir, prepared_image, prepared_mask

    def _run_predict(
        self,
        *,
        prepared_dir: Path,
        output_dir: Path,
        input_stem: str,
        env_name: str,
        device: str,
        timeout_sec: int,
    ) -> dict[str, Any]:
        raw_output_dir = output_dir / "official_lama_raw" / f"{prepared_dir.name}_{env_name}_{device}"
        raw_output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            "conda",
            "run",
            "-n",
            env_name,
            "python",
            str(self.config.predict_script),
            f"model.path={self.config.model_root}",
            f"indir={prepared_dir}",
            f"outdir={raw_output_dir}",
            f"device={device}",
        ]
        attempt: dict[str, Any] = {
            "env": env_name,
            "device": device,
            "command": command,
            "raw_output_dir": str(raw_output_dir),
            "ok": False,
            "returncode": None,
            "reason": "",
            "stdout_tail": "",
            "stderr_tail": "",
            "selected_raw_output": None,
            "elapsed_sec": None,
        }
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                cwd=self.config.repo_root,
                env=self._build_env(),
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            attempt["elapsed_sec"] = time.perf_counter() - started
            attempt["reason"] = "timeout"
            attempt["stdout_tail"] = _tail(exc.stdout if isinstance(exc.stdout, str) else "")
            attempt["stderr_tail"] = _tail(exc.stderr if isinstance(exc.stderr, str) else "")
            return attempt
        except Exception as exc:
            attempt["elapsed_sec"] = time.perf_counter() - started
            attempt["reason"] = f"subprocess_exception: {exc}"
            return attempt

        attempt["elapsed_sec"] = time.perf_counter() - started
        attempt["returncode"] = completed.returncode
        attempt["stdout_tail"] = _tail(completed.stdout)
        attempt["stderr_tail"] = _tail(completed.stderr)
        if completed.returncode != 0:
            attempt["reason"] = "subprocess_failed"
            return attempt

        selected_output = _select_output(raw_output_dir, input_stem)
        if selected_output is None:
            attempt["reason"] = "output_missing"
            return attempt
        attempt["ok"] = True
        attempt["reason"] = "applied"
        attempt["selected_raw_output"] = str(selected_output)
        return attempt

    def inpaint(self, image_path: Path, mask_path: Path, output_dir: Path, timeout_sec: int = 900) -> Path:
        readiness = self.readiness()
        if not readiness["repo_exists"]:
            raise RuntimeError(f"Không tìm thấy repo LaMa: {self.config.repo_root}")
        if not readiness["predict_script_exists"]:
            raise RuntimeError(f"Không tìm thấy predict.py: {self.config.predict_script}")
        if not readiness["checkpoint_exists"]:
            raise RuntimeError(f"Không tìm thấy checkpoint LaMa: {self.config.checkpoint}")

        output_dir.mkdir(parents=True, exist_ok=True)
        prepared_dir, prepared_image, prepared_mask = self._prepare_input(image_path, mask_path, output_dir)
        attempts: list[dict[str, Any]] = []

        env_order: list[tuple[str, str]] = []
        preferred_probe = readiness["preferred_probe"]
        fallback_probe = readiness["fallback_probe"]
        if preferred_probe.get("available") and preferred_probe.get("cuda_available"):
            env_order.append((self.config.conda_env_preferred, "cuda"))
        if fallback_probe.get("available"):
            env_order.append((self.config.conda_env_fallback, "cpu"))
        if not env_order:
            raise RuntimeError(
                "Không tìm được LaMa conda env khả dụng. "
                f"preferred={self.config.conda_env_preferred}, fallback={self.config.conda_env_fallback}"
            )

        for env_name, device in env_order:
            attempt = self._run_predict(
                prepared_dir=prepared_dir,
                output_dir=output_dir,
                input_stem=image_path.stem,
                env_name=env_name,
                device=device,
                timeout_sec=timeout_sec,
            )
            attempts.append(attempt)
            if attempt.get("ok"):
                final_output = output_dir / "official_lama_restored.png"
                shutil.copy2(Path(str(attempt["selected_raw_output"])), final_output)
                self.last_result = {
                    "ok": True,
                    "output": str(final_output),
                    "selected_env": env_name,
                    "selected_device": device,
                    "attempts": attempts,
                    "prepared_input_dir": str(prepared_dir),
                    "prepared_image": str(prepared_image),
                    "prepared_mask": str(prepared_mask),
                    "readiness": readiness,
                }
                return final_output

        self.last_result = {
            "ok": False,
            "attempts": attempts,
            "prepared_input_dir": str(prepared_dir),
            "prepared_image": str(prepared_image),
            "prepared_mask": str(prepared_mask),
            "readiness": readiness,
        }
        reasons = ", ".join(f"{item['env']}:{item['device']}={item['reason']}" for item in attempts)
        raise RuntimeError(f"LaMa subprocess thất bại ở mọi env đã thử: {reasons}")
