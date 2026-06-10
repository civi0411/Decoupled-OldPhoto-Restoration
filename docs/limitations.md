# Limitations

## Current Implementation

Pipeline hiện tại gồm:
- r013 segmentation
- CV crack mask builder
- union mask
- `repair_wide_v1`
- official LaMa inpainting

## Not Implemented Yet

- CodeFormer face restoration
- Colorization
- Real-ESRGAN / super-resolution
- ONNX / TensorRT
- full dynamic offloading

## Deployment Limitation

Docker skeleton hiện tại không self-contained vì không chứa external weights hoặc external runtime.

## Evaluation Limitation

`demo3` là golden regression case để kiểm tra tái lập, không phải benchmark toàn bộ dữ liệu.

## Future Work

- CodeFormer integration
- FP16 inference
- ONNX export
- tiling super-resolution
- model offloading
