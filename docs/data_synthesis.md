# Data Synthesis Guide

## Strategy

Two-stage domain adaptation requires synthetic pre-training data:

1. **Synthetic pretraining (R006–R009)**: 1,000 pairs generated from clean images + crack annotations.
2. **Real fine-tuning (R010–R013)**: 118 real historical photo pairs (`r013_finetune_set`).

---

## Synthesis Engine (`src/old_photo_restoration/data/synthesis_engine.py`)

**Inputs:**
- Clean references: `DIV2K` or pristine archival scans
- Crack annotations: `CrackForest` dataset or `crack_bank_processed_rgba`

**Blending protocol (per pair):**

| Step | Operation |
|---|---|
| 1 | Random flips + 90° rotations |
| 2 | Morphological dilation jitter (0–2 px elliptical kernel) |
| 3 | Edge feathering (Gaussian blur σ=1.2) |
| 4 | Dark crack (85% prob): RGB ~[15–45, 15–40, 15–35], opacity ∈ [0.65, 0.95] |
| 4 | White scratch (15% prob): RGB ~[210–245, 210–245, 205–240] |
| 5 | Subtle Gaussian noise (σ=5, 30% prob) |

---

## Configuration (`configs/synthesis.yaml`)

```yaml
synthesis:
  target_size: [512, 512]
  min_crack_width_dilation: 0
  max_crack_width_dilation: 2
  blend_edge_blur_sigma: 1.2
  dark_crack_probability: 0.85
  min_opacity: 0.65
  max_opacity: 0.95
  add_subtle_noise_probability: 0.30
  noise_sigma: 5.0
```

---

## Generate Dataset

```bash
python scripts/data/generate_synthetic_crack3d.py \
    --clean-dir data/clean_source \
    --crack-dir data/crack_source \
    --output-dir data/synthetic_crack3d \
    --count 1000 \
    --seed 42
```

**Output layout:**
```
data/synthetic_crack3d/
├── damaged/   # synth_00001.png …
├── clean/
├── mask/
└── synthesis_summary.json
```

---

## Verify Lineage

```bash
python scripts/data/inspect_dataset_status.py status
```
