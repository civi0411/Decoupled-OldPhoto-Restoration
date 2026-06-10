# Experiment Summary

Repo submission không mang toàn bộ lịch sử thí nghiệm của repo research. Tài liệu này chỉ giữ các kết quả cần cho reproducibility của submission.

## Core Metrics

- `final_mask_ratio`: `0.0979878066233506`
- `final_mask IoU vs golden`: `0.9997728216844948`
- restored `MAE vs golden`: `0.008635014295578003`
- restored `PSNR vs golden`: `66.63675683358014`

## Phase Evidence

| Phase | Feature | Evidence | Metrics |
| --- | --- | --- | --- |
| 1C | Mask-bypass regression | `scripts/run_pipeline.py --mask ...` | `MAE = 0`, `PSNR = inf` |
| 2 | Auto-mask pipeline | golden demo3 comparison | `IoU = 0.99977`, `PSNR = 66.64` |
| 3B | Gradio local demo | import/launch smoke pass | UI import pass, launch pass |
| 3C | Docker skeleton | `docker compose config` pass | compose parse/config pass |

## Ghi Chú

- Checkpoint r013 không nằm trong Git, chỉ tồn tại local.
- `demo3` là golden case phục vụ hồi quy và trình diễn, không phải benchmark toàn bộ dataset.
