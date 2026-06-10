# Deployment

## Scope

Đây là Docker skeleton cho demo/deploy local. Image không chứa:
- checkpoint segmentation
- external LaMa
- external CodeFormer
- model weights khác

Mục tiêu là cung cấp khung build/chạy cho local Gradio demo và readiness check, không phải image production self-contained.

## Local Run

```bash
python scripts/check_readiness.py
python scripts/run_gradio_demo.py
```

## Docker Build

```bash
docker build -t old-photo-restoration .
```

## Docker Compose

1. Copy template config:

```bash
copy configs\external_paths.example.yaml configs\external_paths.yaml
```

2. Đặt checkpoint vào:

```text
checkpoints/segmenter/seg-unet-attn-r013-gen120-fixed118-local/best_val_iou.pth
```

3. Chạy:

```bash
docker compose up --build
```

## Volume Policy

Các thành phần sau phải mount từ host:
- `configs/external_paths.yaml`
- `checkpoints/`
- `examples/outputs/`
- external model folders nếu cần

## GPU Note

Skeleton mặc định CPU-friendly vì dùng `python:3.10-slim`. Nếu muốn GPU/LaMa chạy hoàn toàn trong container, cần custom image riêng với CUDA và dependency LaMa phù hợp.

## Stability Note

- Gradio chạy với `concurrency limit = 1`
- Mỗi lần run tạo output folder riêng

## Future Optimization

Roadmap có thể gồm:
- dynamic offloading
- FP16
- ONNX / TensorRT
- tiling super-resolution

Đây chưa phải phần của implementation hiện tại.
