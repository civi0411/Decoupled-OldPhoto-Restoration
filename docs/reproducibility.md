# Reproducibility

## Scope

Minimal reproducibility is provided for:
- **Module 1**: segmentation via `R013_REPRO` checkpoint
- **Core pipeline**: segmentation → hybrid mask (`repair_wide_v1`) → pretrained LaMa
- **Regression testing**: `demo3` golden reference case

---

## R013 Audited Facts (must remain exact)

| Fact | Value |
|---|---|
| Raw dataset | 120 images |
| Valid pairs (`masks_fixed`) | **118** (missing `real_0099`, `real_0112`) |
| Split | **83 / 18 / 17** (train / val / test) |
| Initialization | From `R011_REPRO` |
| Reporting threshold | **0.50** |
| Checkpoint SHA256 | `5f3b340e38eba8290d2b8ca030bb51126308169f4e42087f46ddac0334e74203` |

---

## Required External Artifacts

Configure `configs/external_paths.yaml` (copy from `configs/external_paths.example.yaml`):

1. **Segmentation checkpoint** (R013):
   ```
   <LOCAL_ARTIFACT_ROOT>/module1_retrain_sequence/R013_REPRO/best_iou.ckpt
   ```
2. **Official LaMa** — pretrained Big-LaMa weights + conda env (`lama_gpu` or `lama`)
3. **CodeFormer** (optional) — source tree + `codeformer.pth`

---

## Demo3 Replay

### 1. Readiness check
```bash
python scripts/check_readiness.py --strict
```

### 2. Auto-mask pipeline (full segmentation → inpainting)
```bash
python scripts/run_pipeline.py \
  --image examples/inputs/demo3.png \
  --output-dir examples/outputs/seg_smoke_demo3 \
  --face-mode off \
  --reference examples/golden/demo3_r013_repair_wide/restored_before_face.png \
  --reference-mask examples/golden/demo3_r013_repair_wide/final_mask.png
```

Expected outputs: `artifacts/final_mask.png`, `artifacts/inpainting/lama_restored.png`, `metadata.json`

### 3. Mask-bypass pipeline (verify LaMa determinism)
```bash
python scripts/run_pipeline.py \
  --image examples/inputs/demo3.png \
  --mask examples/golden/demo3_r013_repair_wide/final_mask.png \
  --output-dir examples/outputs/pipeline_smoke_demo3 \
  --face-mode off \
  --reference examples/golden/demo3_r013_repair_wide/restored_before_face.png
```

On identical GPU: `MAE = 0`, `PSNR = inf` (deterministic).

---

## Directory Layout

| Path | Purpose |
|---|---|
| `examples/golden/` | Frozen reference outputs for regression tests |
| `examples/outputs/` | Local replay scratch (git-ignored) |
| `outputs/` | New inference runs (git-ignored) |
| `artifacts/manifests/` | Provenance manifests (committed) |
| `configs/external_paths.yaml` | Local machine paths (git-ignored) |
