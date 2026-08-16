# Experiment Log & Lineage

## Module 1 — Segmentation Training Sequence (R006–R014)

| Run | Dataset / Stage | Val IoU | Val F1 | Key Decision |
|---|---|---|---|---|
| R006 | Synthetic `ds-crack3d`, 50 ep | 0.3852 | 0.5249 | Baseline; weak recall → add heavy augmentation |
| R007 | Synthetic + augmentation | 0.3912 | 0.5257 | Precision improved, recall plateaued → restructure loss |
| R008 | Synthetic + Tversky loss | 0.4064 | 0.5492 | Tversky α=0.3 β=0.7 enforces recall |
| R009 | Synthetic, 60 ep | 0.4171 | — | **Real Test IoU: 0.0022** — severe domain gap; use as pretrain init only |
| R010 | Real fine-tune (thin masks) | — | — | Real Test IoU: 0.2927; masks too narrow for LaMa |
| R011 | Real, repair masks, β=0.8 | — | — | Test IoU: 0.4478, F1: 0.6186 @ 0.55; occasional thin-scratch miss |
| R012 | Manual subset (15 samples) | — | — | Test IoU: 0.2846 — small-sample overfit; **discarded** |
| **R013** | **Fixed 118 pairs** | — | — | **Test IoU: 0.3456, F1: 0.5502 @ 0.50 → Selected default** |
| R014 | ResNet-34, repair masks | — | — | Test IoU: 0.4506, F1: 0.6212 @ 0.30 — higher seg but lower PSNR; kept optional |

### R013 Audited Facts

- Dataset: 120 raw images → **118 valid pairs** (`masks_fixed`); missing `real_0099`, `real_0112`.
- Fixed split: **83 / 18 / 17** (train / val / test).
- Initialized from `R011_REPRO` (not R012).
- Reporting threshold: **0.50**.
- Fair comparison vs R011_REPRO @ 0.50: **+0.0911 IoU** (0.3380 → 0.3456), **+0.1065 F1**.

---

## Key Design Decisions (Failure-Driven)

1. **Modular pipeline over end-to-end**: Early end-to-end experiments caused regression instability and hallucination. Separating Modules 1 / 1.5 / 2 isolates evaluation and prevents error propagation.

2. **LaMa as external subprocess**: Fine-tuning LaMa locally introduced training divergence. Official pretrained LaMa wrapped via subprocess eliminates this risk.

3. **Tversky loss evolution**: Scratches occupy < 3% of pixels. BCE+Dice → Tversky (β=0.7 → β=0.8) progressively enforced recall on thin, faint scratches.

4. **R014 not promoted to default**: Higher raw IoU (0.4506 vs 0.3456) but downstream pipeline PSNR regressed. Segmentation accuracy ≠ restoration quality.

---

## Quantitative Restoration Results (Synthetic `ds-crack3d`)

### Auto/Hybrid Pipeline (n=30)

| Metric | Degraded | Restored | Δ | Improved |
|---|---|---|---|---|
| Full-image PSNR | 19.127 | 17.355 | −1.772 | 0/30 |
| Full-image MAE | 24.984 | 27.989 | +3.005 | 0/30 |
| **Masked-region MAE** | 35.666 | **34.151** | **−1.515** | **15/30** |

> Full-image metrics degrade due to `repair_wide_v1` widening (affects adjacent non-degraded pixels). Masked-region MAE confirms positive local repair signal.

### Oracle-Mask Ablation (n=15)

| Metric | Degraded | Auto/Hybrid | Oracle-Mask | Interpretation |
|---|---|---|---|---|
| Full-image PSNR | 17.974 | 16.698 | **17.991** | Oracle surpasses degraded baseline |
| Full-image MAE | 27.739 | 30.316 | **27.712** | Oracle matches degraded level |
| Masked-region MAE | 35.554 | 33.983 | **31.318** | −4.236 vs baseline |
| PSNR improved | — | 0/15 | **10/15 (66.7%)** | |

> **Conclusion**: Module 1 segmentation accuracy is the primary bottleneck. Oracle masks unlock LaMa's latent capability — improving the segmenter directly improves end-to-end restoration.

---

## Evaluation Boundaries

| Metric | Status |
|---|---|
| IoU, F1, Precision, Recall (Module 1) | ✅ Claimed (audited) |
| Full-image PSNR, MAE (synthetic) | ✅ Claimed (audited) |
| Masked-region MAE (synthetic) | ✅ Claimed (audited) |
| LPIPS, FID, masked-region LPIPS | ⏳ Deferred / future work |
| End-to-end real-photo benchmark | ⏳ Deferred / future work |

`demo3` = golden regression smoke test only, not a generalization benchmark.
