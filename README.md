<div align="center">

# Phục Hồi Ảnh Cũ với Deep Learning

### Old Photo Restoration — Modular Deep Learning Pipeline

**Crack Segmentation · Attention Gate U-Net · Hybrid Mask Construction · LaMa Inpainting**

<br>

[![Python](https://img.shields.io/badge/Python-3.10-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-76b900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Gradio](https://img.shields.io/badge/Demo-Gradio-f97316?style=for-the-badge)](http://127.0.0.1:7860)
[![Docker](https://img.shields.io/badge/Deploy-Docker%20Skeleton-2496ed?style=for-the-badge&logo=docker&logoColor=white)](#docker--deployment)

<br>

<img src="docs/assets/demo3/contact_sheet.png" alt="Old Photo Restoration Demo" width="930"/>

`<sub><b>`Input photo`</b>` &nbsp;→&nbsp; `<b>`Predicted repair mask`</b>` &nbsp;→&nbsp; `<b>`LaMa restored output`</b></sub>`

</div>

---

## Giới thiệu

Ảnh cũ xuống cấp theo hai kiểu riêng biệt về bản chất:

- **Structured damage** — vết nứt giấy, xước, rách nhỏ: có cấu trúc không gian, cần localisation chính xác trước khi inpaint.
- **Unstructured degradation** — nhiễu hạt, mờ, phai màu: ảnh hưởng toàn cục, không có biên rõ ràng.

Một mạng end-to-end duy nhất xử lý cả hai thường học nghiệm trung bình: ảnh trông mượt hơn nhưng vết nứt vẫn còn hoặc texture bị làm nhòe. Project này chọn hướng **modular pipeline**: mỗi module xử lý đúng loại degradation của nó, có intermediate artifact riêng, có thể đánh giá độc lập, và có thể thay thế backend mà không phá toàn bộ hệ thống.

---

## Pipeline tổng quan

```
      Input old photo (RGB)
               │
               ▼
┌─────────────────────────────┐
│  r013 U-Net + Attention Gate│  ← Deep Learning segmentation
│  CrackSegmenter             │    phát hiện vùng cần sửa
└──────────────┬──────────────┘
               │  dl_mask.png
               ▼
┌─────────────────────────────┐
│  Classical CV Crack Detector│  ← CLAHE + morphology + component filter
│  bắt crack mảnh tương phản  │    bổ sung cho DL mask
└──────────────┬──────────────┘
               │  cv_mask.png
               ▼
┌─────────────────────────────┐
│  Hybrid Union Mask          │  union_mask = max(dl_mask, cv_mask)
└──────────────┬──────────────┘
               │  union_before_refine.png
               ▼
┌─────────────────────────────┐
│  repair_wide_v1 Refinement  │  gap bridging + dilation
│  làm mask phù hợp cho LaMa  │
└──────────────┬──────────────┘
               │  final_mask.png
               ▼
┌─────────────────────────────┐
│  Official LaMa Inpainting   │  FFC-based global context restoration
└──────────────┬──────────────┘
               │
               ▼
        Restored image (RGB)
        metadata.json
```

Pipeline ưu tiên **tính quan sát được**: tất cả intermediate artifact được lưu lại để dễ debug, giải thích và đánh giá từng bước.

---

## Deep Learning Architecture

### CrackSegmenter — U-Net + Attention Gate

Model segmentation trung tâm của pipeline là `CrackSegmenter`, kiến trúc U-Net style được tăng cường bằng **Attention Gate** tại mỗi skip connection của decoder.

#### Vì sao dùng Attention Gate?

U-Net chuẩn ghép feature map encoder vào decoder qua skip connection mà không phân biệt vùng quan trọng. Đối với crack segmentation, phần lớn ảnh là **background không liên quan** — plain paper, skin tone, background objects. Attention Gate học được cơ chế attention để **suppression background**, chỉ cho phép feature của vùng crack/defect đi qua skip connection. Kết quả là decoder tập trung tốt hơn vào vùng hư hại, giảm false positive trên vùng texture phức tạp.

#### Kiến trúc chi tiết

```
Input: RGB image, resize 512×512, normalize [0, 1]

Encoder
├── ConvBlock    3  →  8   (base_channels = 8)
├── DownBlock    8  → 16
├── DownBlock   16  → 32
└── DownBlock   32  → 64

Bottleneck: 64 channels

Decoder
├── UpBlock(64, 32 → 32)  + AttentionGate(g=64, x=32)
├── UpBlock(32, 16 → 16)  + AttentionGate(g=32, x=16)
├── UpBlock(16,  8 →  8)  + AttentionGate(g=16, x=8)
└── Conv 1×1 → 1 channel

Output: probability map 1 kênh, Sigmoid
        → resize về kích thước ảnh gốc
        → threshold 0.5 → binary mask 0/255
```

**AttentionGate** (mỗi skip connection):

```
gating signal g  → Conv 1×1 → BN
skip feature x   → Conv 1×1 → BN
                     ↓ Add → ReLU → Conv 1×1 → Sigmoid
                     ↓
attention coefficient α  (bilinear resize nếu khác spatial size)
output = α ⊙ x
```

#### Checkpoint r013

| Thông tin | Chi tiết                                                            |
| ---------- | -------------------------------------------------------------------- |
| File       | `best_val_iou.pth`                                                 |
| Load key   | `model_state_dict`                                                 |
| SHA256     | `a63381ade991cb936e2262e80fa6001c3a1fe9d10b1075be0d3c7f617c0a5725` |

#### Training summary

r013 được fine-tune từ checkpoint r011 trên tập mở rộng với **118 cặp ảnh hợp lệ**, chia theo tỉ lệ `83 / 18 / 17` (train / val / test). Best checkpoint được chọn theo **validation IoU**.

| Metric     | r011 (threshold 0.55) | r013 (threshold 0.50) |
| ---------- | --------------------- | --------------------- |
| IoU        | 0.2527                | **0.3457**      |
| F1         | 0.4025                | **0.5097**      |
| Precision  | 0.4112                | **0.5887**      |
| Recall     | 0.4083                | 0.4670                |
| Best epoch | —                    | 39                    |
| Val IoU    | —                    | 0.3812                |
| Val F1     | —                    | 0.5503                |

> **Lưu ý:** Bảng so sánh trên được đánh giá trên fixed r013 evaluation set. r013 cải thiện rõ về IoU và Precision so với r011 trên tập này. Đây không phải benchmark đại diện toàn bộ ảnh cũ thật.

---

## Hybrid Mask Construction

### Tại sao cần hybrid mask thay vì chỉ dùng DL mask?

Model học sâu capture được **semantically meaningful** damage region nhưng có thể bỏ sót các crack mảnh, tương phản cao vì chúng chiếm quá ít pixel để ảnh hưởng đến loss. Classical CV detector ngược lại — nó rất nhạy với cạnh tương phản cao nhưng tạo nhiều false positive trên texture phức tạp.

**Hybrid union** giữ điểm mạnh của cả hai:

```
union_mask = np.maximum(dl_mask, cv_mask)
```

Precision của DL mask + coverage của CV detector = mask đầy đủ hơn mà không cần hạ threshold segmentation (vốn sẽ tăng false positive).

### Classical CV Crack Detector

Pipeline CV gồm các bước: grayscale conversion → CLAHE equalization → blackhat/tophat morphology → Canny edge detection → percentile thresholding → connected component filtering theo aspect ratio và area.

### repair_wide_v1 Refinement

Mask sau union được refinement qua `repair_wide_v1` để phù hợp hơn với yêu cầu của LaMa inpainting: giữ component dài, bridge gap nhỏ giữa các đoạn crack, morphology close, và dilation nhẹ để đảm bảo LaMa có đủ context xung quanh vết nứt.

### Official LaMa Inpainting

LaMa dùng **Fast Fourier Convolution (FFC)** layers — xử lý global context qua frequency domain thay vì chỉ local receptive field. Điều này đặc biệt phù hợp cho long thin cracks kéo dài qua nhiều vùng của ảnh: standard conv inpainting với receptive field nhỏ thường để lại visible seam, trong khi FFC có thể tham chiếu toàn ảnh khi điền vùng bị mask.

LaMa được gọi qua **subprocess** từ external runtime — cách này giữ repo submission nhẹ, không nhúng external model vào source tree.

---

## Kết quả

### Regression metrics — demo3

Repo đi kèm golden case `demo3` để kiểm tra pipeline sau mỗi thay đổi.

| Stage                   | Metric                      | Kết quả    |
| ----------------------- | --------------------------- | ------------ |
| Mask-bypass pipeline    | MAE vs golden restored      | `0.0`      |
| Mask-bypass pipeline    | Max diff vs golden restored | `0`        |
| Auto-mask — final mask | IoU vs golden mask          | `0.9998`   |
| Auto-mask — final mask | Mask area ratio             | `0.0980`   |
| Auto-mask — restored   | PSNR vs golden restored     | `66.64 dB` |

> Đây là **regression metrics trên demo3**, không phải benchmark đầy đủ trên dataset ảnh cũ thật. Mục đích là đảm bảo pipeline không bị regression sau các thay đổi code.

### Demo gallery

<p align="center">
  <img src="docs/assets/demo3/input.png" width="280" alt="Input old photo"/>
   
  <img src="docs/assets/demo3/final_mask.png" width="280" alt="Final repair mask"/>
   
  <img src="docs/assets/demo3/restored_before_face.png" width="280" alt="Restored output"/>
</p>
<p align="center">
  <sub>Input image  ·  Final repair mask  ·  Restored output</sub>
</p>

<p align="center">
  <img src="docs/assets/demo3/overlay_mask.png" width="620" alt="Mask overlay"/>
</p>

---

## Cài đặt nhanh

### 1. Clone repo

```bash
git clone https://github.com/doraIaIa/deep-learning-old-photo-restoration.git
cd deep-learning-old-photo-restoration
```

### 2. Tạo môi trường Python

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Cấu hình external dependencies

```bash
# Windows
copy configs\external_paths.example.yaml configs\external_paths.yaml

# Linux / macOS
cp configs/external_paths.example.yaml configs/external_paths.yaml
```

Chỉnh `configs/external_paths.yaml` để trỏ tới LaMa và (optional) CodeFormer trên máy local. File này bị `.gitignore` vì chứa path riêng của từng máy.

### 4. Đặt checkpoint r013

```
checkpoints/segmenter/seg-unet-attn-r013-gen120-fixed118-local/best_val_iou.pth
```

SHA256 expected:

```
a63381ade991cb936e2262e80fa6001c3a1fe9d10b1075be0d3c7f617c0a5725
```

Checkpoint không được commit vào Git và cần được cung cấp riêng.

### 5. Kiểm tra readiness

```bash
# Basic check
python scripts/check_readiness.py

# Strict mode — fail nếu thiếu bất kỳ dependency nào
python scripts/check_readiness.py --strict
```

Readiness checker kiểm tra: Python imports, PyTorch/CUDA, config files, checkpoint path và SHA256, LaMa runtime, CodeFormer (optional).

---

## Chạy pipeline

### Auto-mask mode (chế độ chính)

Pipeline tự sinh mask rồi inpaint:

```bash
# Windows
python scripts\run_pipeline.py ^
  --image examples\inputs\demo3.png ^
  --output-dir examples\outputs\demo3_auto ^
  --face-mode off ^
  --reference examples\golden\demo3_r013_repair_wide\restored_before_face.png ^
  --reference-mask examples\golden\demo3_r013_repair_wide\final_mask.png

# Linux / macOS
python scripts/run_pipeline.py \
  --image examples/inputs/demo3.png \
  --output-dir examples/outputs/demo3_auto \
  --face-mode off \
  --reference examples/golden/demo3_r013_repair_wide/restored_before_face.png \
  --reference-mask examples/golden/demo3_r013_repair_wide/final_mask.png
```

Output:

```
examples/outputs/demo3_auto/
├── dl_mask.png               ← r013 segmentation output
├── cv_mask.png               ← classical CV detector output
├── union_before_refine.png   ← union trước repair_wide_v1
├── final_mask.png            ← refined mask truyền vào LaMa
├── restored_before_face.png  ← LaMa inpainting output
└── metadata.json             ← full run metadata
```

### Mask-bypass mode

Dùng mask có sẵn, bỏ qua segmentation — hữu ích cho oracle evaluation hoặc kiểm tra LaMa độc lập:

```bash
python scripts/run_pipeline.py \
  --image examples/inputs/demo3.png \
  --mask examples/golden/demo3_r013_repair_wide/final_mask.png \
  --output-dir examples/outputs/demo3_bypass \
  --face-mode off \
  --reference examples/golden/demo3_r013_repair_wide/restored_before_face.png
```

### Gradio demo

```bash
python scripts/run_gradio_demo.py
```

Mở `http://127.0.0.1:7860` — upload ảnh, xem final mask và restored output trực tiếp trong browser. Concurrency được giới hạn bằng `1` để tránh đơ máy khi demo.

---

## Cấu trúc repository

```
deep-learning-old-photo-restoration/
├── app/
│   └── gradio_demo.py
├── configs/
│   ├── checkpoints.yaml               ← checkpoint paths (relative)
│   ├── external_paths.example.yaml    ← template, gitignored khi copy
│   └── inference.yaml                 ← pipeline mode config
├── docs/
│   ├── assets/demo3/
│   ├── deployment.md
│   ├── demo_script.md
│   ├── experiment_summary.md
│   ├── external_dependencies.md
│   ├── limitations.md
│   └── reproducibility.md
├── examples/
│   ├── golden/demo3_r013_repair_wide/ ← frozen reference outputs
│   └── inputs/demo3.png
├── scripts/
│   ├── build_demo_assets.py
│   ├── check_readiness.py
│   ├── run_gradio_demo.py
│   ├── run_pipeline.py
│   └── smoke_lama_inpainting.py
├── src/old_photo_restoration/
│   ├── config.py
│   ├── pipeline.py
│   ├── evaluation/
│   ├── face_restoration/              ← stub, chưa bật
│   ├── inpainting/
│   ├── segmentation/
│   └── utils/
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Docker / Deployment

Repo có Docker skeleton để đóng gói demo. Image không tự chứa checkpoint và external model weights.

```bash
docker compose up --build
```

Chính sách volume:

| Path                            | Mount từ | Lý do                                   |
| ------------------------------- | --------- | ---------------------------------------- |
| `configs/external_paths.yaml` | host      | path riêng từng máy                   |
| `checkpoints/`                | host      | không commit weights vào image         |
| `examples/outputs/`           | host      | lưu kết quả ra ngoài container       |
| LaMa / CodeFormer               | host      | external runtime, không bake vào image |

GPU/CUDA runtime cần cấu hình riêng nếu muốn chạy full GPU trong container. Xem chi tiết tại [`docs/deployment.md`](docs/deployment.md).

---

## External Dependencies

| Dependency                                      | Vai trò           | Bắt buộc |
| ----------------------------------------------- | ------------------ | ---------- |
| [LaMa](https://github.com/advimman/lama)           | Inpainting backend | ✅ Có     |
| [CodeFormer](https://github.com/sczhou/CodeFormer) | Face restoration   | Optional   |

LaMa được gọi qua subprocess với conda environment `lama_gpu` (fallback về `lama` cho CPU). Xem hướng dẫn setup đầy đủ tại [`docs/external_dependencies.md`](docs/external_dependencies.md).

---

## Reproducibility

Golden regression case:

```
examples/golden/demo3_r013_repair_wide/
├── final_mask.png
├── metadata.json
└── restored_before_face.png
```

Tái tạo demo assets cho README và báo cáo:

```bash
python scripts/build_demo_assets.py
```

SHA256 của ba artifact golden được document tại [`docs/reproducibility.md`](docs/reproducibility.md).

---

## Trạng thái hiện tại

| Thành phần                             | Trạng thái          |
| ---------------------------------------- | --------------------- |
| r013 U-Net + Attention Gate segmentation | ✅ Implemented        |
| Classical CV crack mask builder          | ✅ Implemented        |
| Hybrid union mask                        | ✅ Implemented        |
| repair_wide_v1 mask refinement           | ✅ Implemented        |
| Official LaMa inpainting wrapper         | ✅ Implemented        |
| CLI pipeline (`run_pipeline.py`)       | ✅ Implemented        |
| Gradio local demo                        | ✅ Implemented        |
| Readiness checker                        | ✅ Implemented        |
| Docker deployment skeleton               | ✅ Implemented        |
| Demo assets + golden regression case     | ✅ Implemented        |
| CodeFormer face restoration              | Optional / chưa bật |
| Colorization                             | Not implemented       |
| Super-resolution / Real-ESRGAN           | Not implemented       |
| ONNX / TensorRT export                   | Not implemented       |

---

## Giới hạn hiện tại

- **Domain gap**: r013 được fine-tune trên tập synthetic crack data. Các crack mảnh phân nhánh trên nền sepia/paper tone thật vẫn có thể bị underdetect.
- **Chưa bật CodeFormer**: Face restoration hiện là stub, chưa active trong pipeline.
- **Chưa có colorization và super-resolution**: Future work.
- **Chưa có no-reference metrics**: Regression hiện dựa trên golden reference pair (PSNR/IoU). BRISQUE/NIQE cho real image chưa tích hợp.
- **Chưa có tiling**: Ảnh được resize về `512 × 512` trước khi inference, ảnh lớn sẽ mất detail.
- **Docker là skeleton**: External weights và LaMa runtime được volume-mount, không tự chứa trong image.
- **demo3 không phải full benchmark**: Regression metrics chỉ đại diện cho một golden case.

Chi tiết đầy đủ: [`docs/limitations.md`](docs/limitations.md)

---

## Roadmap

- Bật CodeFormer qua subprocess wrapper với `--face-mode auto`
- Tích hợp no-reference metrics (BRISQUE, NIQE) cho qualitative evaluation trên ảnh thật
- Cải thiện synthetic training data: sepia/paper texture augmentation, real paper crack crops
- ONNX export cho r013 segmentation model
- Tiling strategy cho ảnh độ phân giải cao
- Mở rộng test set và ablation study documentation

---

## Tài liệu liên quan

| Tài liệu                                                      | Nội dung                                              |
| --------------------------------------------------------------- | ------------------------------------------------------ |
| [`docs/reproducibility.md`](docs/reproducibility.md)             | Cách chạy lại demo3 và kiểm tra golden regression |
| [`docs/experiment_summary.md`](docs/experiment_summary.md)       | Tóm tắt các phase thực nghiệm và kết quả       |
| [`docs/external_dependencies.md`](docs/external_dependencies.md) | Hướng dẫn cài LaMa / CodeFormer                    |
| [`docs/deployment.md`](docs/deployment.md)                       | Docker skeleton và chính sách volume                |
| [`docs/demo_script.md`](docs/demo_script.md)                     | Kịch bản demo khi thuyết trình                     |
| [`docs/limitations.md`](docs/limitations.md)                     | Giới hạn hiện tại và future work                  |

---

## Repository Policy

Không commit:

```
configs/external_paths.yaml
checkpoints/
examples/outputs/
external_models/
*.pth  *.pt  *.ckpt  *.onnx  *.engine
```

Lý do: tránh lộ path local, tránh repo nặng, tách source code khỏi model weights, dễ clone và review.

---

## Acknowledgements

Project sử dụng [LaMa](https://github.com/advimman/lama) làm inpainting backend và thiết kế mở để tích hợp [CodeFormer](https://github.com/sczhou/CodeFormer) cho face restoration. Khi dùng hoặc mở rộng các external repository / pretrained model, cần tuân thủ license và citation của từng dự án gốc.

Repo này được phát triển cho mục đích học thuật trong môn **Deep Learning**.

---

## License

Academic course project. Vui lòng kiểm tra license của các external dependencies và pretrained weights trước khi tái phân phối hoặc deploy ngoài phạm vi học thuật.
