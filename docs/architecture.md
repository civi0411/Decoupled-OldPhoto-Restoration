# Architecture

## Pipeline Overview

```text
Input RGB Image
       │
       ▼
┌─────────────────────────────────────────────────┐
│ Module 1 — Damage Segmentation                  │
│  R013 Attention U-Net (default)                 │
│  R014 ResNet-34 U-Net   (--segmenter-arch r014) │
│  Output: dl_mask.png                            │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│ Module 1.5 — Hybrid Mask Refinement             │
│  Classical CV: CLAHE → Blackhat → Canny         │
│  Union + repair_wide_v1 morphology              │
│  Output: final_mask.png                         │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│ Module 2 — Inpainting (LaMa, subprocess)        │
│  Output: inpainting/lama_restored.png           │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│ Module 2.5 — Color & Quality Restoration [opt]  │
│  quality_restoration → Color U-Net → CCM        │
│  Output: color_restoration/color_restored.png   │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│ Module 3 — Face Restoration [opt]               │
│  RetinaFace + CodeFormer (dependency-gated)     │
│  Output: face_restoration/codeformer_output.png │
└─────────────────────────────────────────────────┘
```

---

## Source Package Structure (`src/old_photo_restoration/`)

| Subpackage | Key Files | Purpose |
|---|---|---|
| `data/` | `synthesis_engine.py`, `datasets.py` | Synthetic data generation, PyTorch datasets |
| `segmentation/` | `models/`, `predictor.py`, `mask_refinement.py` | U-Net variants, tiling inference, hybrid mask |
| `training/` | `trainer.py`, `losses.py` | Training loop, BCEDice/Focal/Tversky losses |
| `inpainting/` | `lama_inpainter.py` | Subprocess wrapper around official LaMa |
| `color_restoration/` | `processor.py` | Color U-Net + CCM post-processing |
| `face_restoration/` | `wrapper.py` | Optional CodeFormer + RetinaFace |
| `pipeline/` | `pipeline.py` | `RestorationPipeline` orchestrator |
| `config/` | `config.py` | Pydantic config loaders |
| `utils/` | `sha256.py`, … | SHA256 verification, misc utilities |

---

## Module Details

### Module 1 — Segmentation Models

| Model | Arch | Default Threshold | Test IoU | Test F1 | Status |
|---|---|---|---|---|---|
| R013 | Custom Attention U-Net | 0.50 | 0.3456 | 0.5502 | **Default** |
| R014 | ResNet-34 + dilated bottleneck | 0.30 (dilation=1) | 0.4506 | 0.6212 | Optional |

- Both use **Attention Gates** on skip connections to suppress background noise.
- R014 has higher raw segmentation scores but lower downstream PSNR → kept optional.

### Module 1.5 — Hybrid Mask Refinement

`build_hybrid_mask` combines:
1. DL mask (`dl_mask.png`)
2. Classical CV mask (`cv_mask.png`): CLAHE → Blackhat → Canny
3. Union → `repair_wide_v1` morphological closing + dilation (3×3 kernel)

> **Design intent**: Prioritize inpainting coverage over pixel-perfect boundary accuracy.

### Module 2 — LaMa Inpainting

- Official pretrained Big-LaMa, zero local fine-tuning.
- Runs as an **isolated subprocess** (`LamaInpainter`) via `configs/external_paths.yaml`.
- Isolation prevents generative instability from affecting segmentation training.

### Module 2.5 — Post-Inpainting Color Restoration

Sequential stages (enabled with `--post-inpainting`):
1. `quality_restoration` — denoising, contrast recovery
2. `color_restoration_model` — Color Restoration U-Net (Lab residual v2)
3. `inference_control` — conservative blending (`model` vs `opencv_conservative`)
4. `ccm_color_correction` — Color Correction Matrix channel rebalance
5. `safety_postprocessing` — pixel clip, structural consistency enforcement

### Module 3 — Face Restoration (Optional)

- Activated with `--post-inpainting --face-mode auto`.
- Dependency-gated: gracefully skips if environment is unavailable (`status: skipped`).
- **No identity-preservation guarantee.**

---

## Output Layout

```
outputs/<run_id>/items/<image_id>/
├── artifacts/
│   ├── dl_mask.png
│   ├── cv_mask.png
│   ├── union_before_refine.png
│   ├── final_mask.png
│   ├── inpainting/lama_restored.png
│   ├── color_restoration/color_restored.png
│   └── face_restoration/codeformer_output.png
├── final.png
└── metadata.json
```

`metadata.json` records: checkpoint SHA256, thresholds, dilation radius, latency, mask ratios, environment selectors, per-stage status.

---

## Artifact & Checkpoint Policy

- **Checkpoint binaries** (`.pth`, `.ckpt`) → local only, not committed.
- **Datasets** → external, not committed.
- **Manifests** (`artifacts/manifests/`, `data/manifests/`) → committed, provide full provenance.
- Verify with:

```bash
python scripts/verify_artifacts.py check-all --repo-root .
python scripts/check_readiness.py --strict
```
