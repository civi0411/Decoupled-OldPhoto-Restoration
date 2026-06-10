# Reproducibility

## Demo3 Golden Case

Golden regression case hiện tại là `examples/inputs/demo3.png`.

Các golden artifact đi kèm:
- `examples/golden/demo3_r013_repair_wide/final_mask.png`
- `examples/golden/demo3_r013_repair_wide/restored_before_face.png`
- `examples/golden/demo3_r013_repair_wide/metadata.json`

## Local Artifacts Required

Repo Git không chứa các artifact runtime sau:
- `configs/external_paths.yaml`
- checkpoint r013 segmentation
- official LaMa repo và weights
- CodeFormer repo và weights

Checkpoint r013 cần đặt tại:

```text
checkpoints/segmenter/seg-unet-attn-r013-gen120-fixed118-local/best_val_iou.pth
```

SHA256 expected:

```text
a63381ade991cb936e2262e80fa6001c3a1fe9d10b1075be0d3c7f617c0a5725
```

## Readiness Check

Chạy trước khi smoke test:

```bash
python scripts/check_readiness.py
python scripts/check_readiness.py --strict
```

## Replay Demo3 Mask-Bypass

```bash
python scripts/run_pipeline.py --image examples/inputs/demo3.png --mask examples/golden/demo3_r013_repair_wide/final_mask.png --output-dir examples/outputs/pipeline_smoke_demo3 --face-mode off --reference examples/golden/demo3_r013_repair_wide/restored_before_face.png
```

Expected:
- same_size = `True`
- MAE = `0`
- max_diff = `0`
- PSNR = `inf`

## Replay Demo3 Auto-Mask

```bash
python scripts/run_pipeline.py --image examples/inputs/demo3.png --output-dir examples/outputs/seg_smoke_demo3 --face-mode off --reference examples/golden/demo3_r013_repair_wide/restored_before_face.png --reference-mask examples/golden/demo3_r013_repair_wide/final_mask.png
```

Expected metrics gần đúng:
- `final_mask_ratio` `0.0979878066233506`
- `final_mask IoU vs golden` `0.9997728216844948`
- restored `PSNR` `66.63675683358014`

## Vì Sao Checkpoint Không Nằm Trong Git

- Checkpoint lớn và không phù hợp để commit vào submission repo.
- Repo submission chỉ giữ code, config template và golden artifact cần cho tái lập.
- Việc tách checkpoint khỏi Git giúp repo nhẹ hơn và rõ ràng hơn cho người chấm.
