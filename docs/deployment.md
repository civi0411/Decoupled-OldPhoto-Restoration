# Deployment

## Scope

Docker/local deployment skeleton for demo purposes. Does **not** bundle:
- Segmentation checkpoint
- LaMa source or weights
- CodeFormer source or weights
- Datasets, research logs, or large outputs

---

## External Dependencies

| Dependency | Type | Configuration |
|---|---|---|
| LaMa (Big-LaMa) | Pretrained subprocess | `configs/external_paths.yaml` |
| R013 checkpoint | Local artifact | `<LOCAL_ARTIFACT_ROOT>/module1_retrain_sequence/R013_REPRO/best_iou.ckpt` |
| CodeFormer | Optional, dependency-gated | `configs/external_paths.yaml` |

R013 SHA256: `5f3b340e38eba8290d2b8ca030bb51126308169f4e42087f46ddac0334e74203`

> `configs/external_paths.yaml` is machine-specific → git-ignored. Copy from `configs/external_paths.example.yaml`.

---

## Run Locally

```bash
# 1. Verify environment and checkpoint bindings
python scripts/check_readiness.py

# 2. Launch Gradio demo
python scripts/run_gradio_demo.py
```

---

## Run with Docker

```bash
docker compose up --build
```

> Docker is a deployment skeleton, not a self-contained production image.  
> GPU/CUDA setup is machine-specific. LaMa and checkpoints must be mounted externally.
