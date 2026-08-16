# Training Guide

## Two-Stage Domain Adaptation

| Stage | Runs | Dataset | LR | Epochs |
|---|---|---|---|---|
| Synthetic pretraining | R006–R009 | `ds-crack3d-512-n1000-v001` (1,000 pairs) | 1e-3 | 50–60 |
| Real fine-tuning | R010–R013 | `r013_finetune_set` (118 valid pairs) | 1e-4 | ~39 |

Loss: `BCEDiceLoss` (0.5×BCE + 0.5×Dice) → `Tversky` (α=0.3, β=0.7→0.8 for thin scratch recall)  
Optimizer: `AdamW` + `ReduceLROnPlateau`

---

## Model Architectures

| Key | Arch | Notes |
|---|---|---|
| `r013_custom_attnunet` | 4-level encoder-decoder + Attention Gates | Default Module 1 |
| `r014_resnet34` | ResNet-34 encoder + dilated bottleneck (dilation=3) | Experimental |

---

## Train from Scratch (Synthetic)

```bash
python scripts/train/train_segmentation_loop.py \
    --data-dir data/synthetic_crack3d \
    --arch r013_custom_attnunet \
    --epochs 50 \
    --batch-size 8 \
    --lr 1e-3 \
    --val-split 0.15 \
    --output-dir outputs/training_runs/scratch_r013 \
    --num-workers 4
```

Checkpoints saved: `best_iou.ckpt`, `last.ckpt`, `training_summary.json`.

---

## Fine-Tune on Real Photos

```bash
python scripts/train/train_r013_finetune_loop.py \
    --data-dir data/old_photo_pairs \
    --pretrained-checkpoint outputs/training_runs/scratch_r013/checkpoints/best_iou.ckpt \
    --epochs 39 \
    --batch-size 4 \
    --lr 1e-4 \
    --val-split 0.20 \
    --output-dir outputs/training_runs/finetune_r013 \
    --num-workers 2
```

---

## Inspect Historical Checkpoint Status

```bash
python scripts/train/inspect_training_status.py status       # R006–R013 metrics + SHA256
python scripts/train/inspect_r013_finetune_status.py status  # R013 config facts
```
