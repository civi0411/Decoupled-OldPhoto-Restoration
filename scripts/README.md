# CLI & Utility Scripts

This directory contains command-line scripts for running end-to-end restoration, checking environment readiness, verifying artifact contracts, generating synthetic datasets, training segmentation models, and performing quantitative evaluation. Execute all scripts from the repository root directory.

---

## 1. Core & Demo Pipelines

- `run_pipeline.py`: Executes the modular old photo restoration pipeline (segmentation, hybrid mask refinement, LaMa inpainting, and optional post-inpainting color/face restoration) on single images or batch directories.
- `run_color_restoration.py`: Runs the standalone post-inpainting color and quality restoration module on pre-inpainted images.
- `run_gradio_demo.py`: Launches an interactive Gradio web application (`app/gradio_demo.py`) for visual demonstration and inspection.
- `check_readiness.py`: Verifies local configuration files (`external_paths.yaml`), Python dependencies, and checkpoint bindings (`--strict` mode available).
- `verify_artifacts.py`: Audits local checkpoints and dataset manifests against canonical SHA256 hashes and packaging policies (`CẤM SAI LỆCH nội dung nhất`).
- `download_checkpoints.py`: Helper utility providing instructions and validation for retrieving external model binaries.
- `build_demo_assets.py`: Generates standardized visual assets (`contact_sheet.png`, overlays) used in documentation and demonstrations.
- `smoke_lama_inpainting.py` / `smoke_r014_segmenter.py`: Lightweight smoke tests for external LaMa runtime wrapper and R014 segmenter.

---

## 2. Data Synthesis & Inspection (`scripts/data/`)

- `data/generate_synthetic_crack3d.py`: **[NEW]** Synthesizes paired training triples `(damaged, clean, mask)` by blending clean reference photos (`DIV2K`) with crack annotations (`CrackForest`) using geometric transforms and alpha blending.
- `data/inspect_dataset_status.py`: Inspects historical dataset lineage and expected local layout according to `artifacts/manifests/datasets_manifest.csv`.

---

## 3. Training Loops & Status (`scripts/train/`)

- `train/train_segmentation_loop.py`: **[NEW]** Runs actual PyTorch training from scratch for Attention U-Net (`R013` custom Attention U-Net or `R014` ResNet-34) with `BCEDiceLoss` and validation monitoring.
- `train/train_r013_finetune_loop.py`: **[NEW]** Runs actual domain fine-tuning loop for `R013` on real historical photo pairs.
- `train/inspect_training_status.py`: Displays historical training progression stats from `reproduction_runs_manifest.csv`.
- `train/inspect_r013_finetune_status.py`: Entry point for inspecting or verifying the `R013` fine-tuning configuration record.

---

## 4. Quantitative Evaluation (`scripts/eval/`)

- `eval/evaluate_segmentation.py`: Computes segmentation metrics (`IoU`, `F1`, `Precision`, `Recall`) comparing predicted masks against ground truth across benchmark splits.
- `eval/eval_pipeline_paired.py`: Runs quantitative restoration evaluations (`PSNR`, `MAE`, `Masked-region MAE`) over paired benchmark datasets.
- `eval/run_ablation.py`: Executes and inspects systematic ablation studies for hybrid mask refinement configurations.

---

## Checkpoint & Binary Policy
In accordance with our strict repository hygiene guidelines (`CẤM SAI LỆCH nội dung nhất`), large model checkpoints (`.pth`, `.ckpt`), datasets, and intermediate runtime outputs are **never committed to Git**. This repository tracks clean source code, configurations, declarative manifests, and documentation required for reproducible execution.
