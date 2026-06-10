# Demo Script

## Kịch Bản 3-5 Phút

1. Chạy readiness check:
   `python scripts/check_readiness.py`
2. Chạy CLI auto-mask với demo3:
   `python scripts/run_pipeline.py --image examples/inputs/demo3.png --output-dir examples/outputs/seg_smoke_demo3 --face-mode off --reference examples/golden/demo3_r013_repair_wide/restored_before_face.png --reference-mask examples/golden/demo3_r013_repair_wide/final_mask.png`
3. Chạy Gradio demo:
   `python scripts/run_gradio_demo.py`
4. Chỉ vào ba thành phần chính:
   input image, final mask, restored image
5. Giải thích vì sao checkpoint và external dependency không nằm trong Git

## Câu Nói Gọn Khi Bảo Vệ

- Vì sao dùng modular pipeline:
  Tách segmentation, mask refinement và inpainting giúp dễ kiểm soát từng bước và dễ debug hơn so với đóng gói tất cả vào một khối đen.
- Vì sao dùng hybrid mask DL + CV:
  DL mask giúp bắt cấu trúc hỏng chính, còn CV mask hỗ trợ giữ thêm crack mảnh; union giúp tăng recall trước bước refine.
- Vì sao Docker không chứa weights:
  Submission repo cần nhẹ, rõ, và an toàn; weights và external runtime được mount từ host thay vì commit vào image.
- Vì sao CodeFormer/Colorization/SR là optional hoặc future:
  Các phần đó chưa phải implementation hiện tại, nên không được claim là đã tích hợp vào pipeline chính.
