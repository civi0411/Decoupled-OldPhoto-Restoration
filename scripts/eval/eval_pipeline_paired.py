import os
import sys
import pandas as pd
from pathlib import Path
import json

sys.path.insert(0, "F:/deeplearning/old_photo_restoration_blueprint21_submission/src")
from old_photo_restoration.config import load_config
from old_photo_restoration.pipeline import RestorationPipeline
from old_photo_restoration.evaluation.metrics import compute_psnr, compute_mae_image

def main():
    img_dir = Path("F:/deeplearning/experiment_value/ds_temp/ds-crack3d-512-n1000-v001/val/images")
    gt_dir = Path("F:/deeplearning/experiment_value/ds_temp/ds-crack3d-512-n1000-v001/val/gt")
    out_dir = Path("F:/deeplearning/experiment_value/r014_resnet34_safe_experiment/pipeline_check/restoration_temp")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    config = load_config(
        inference_path=Path("F:/deeplearning/old_photo_restoration_blueprint21_submission/configs/inference.yaml"),
        checkpoint_path=Path("F:/deeplearning/old_photo_restoration_blueprint21_submission/configs/checkpoints.yaml"),
        external_path=Path("F:/deeplearning/old_photo_restoration_blueprint21_submission/configs/external_paths.yaml"),
    )
    
    pipeline = RestorationPipeline(config)
    
    samples = [f"val_{str(i).zfill(6)}.png" for i in range(1, 16)]
    
    r014_ckpt = Path("F:/deeplearning/experiment_value/r014_resnet34_safe_experiment/kaggle_outputs/r014_resnet34_outputs_download/checkpoints/stage2_real_best_val_iou.pth")
    
    results = []
    
    for sample in samples:
        print(f"Processing {sample}...")
        img_path = img_dir / sample
        gt_path = gt_dir / sample
        
        # Load degraded and reference to calculate baseline metrics
        import cv2
        import numpy as np
        img_bgr = cv2.imread(str(img_path))
        gt_bgr = cv2.imread(str(gt_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gt_rgb = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB)
        
        degraded_psnr = compute_psnr(img_rgb, gt_rgb)
        degraded_mae = compute_mae_image(img_rgb, gt_rgb)
        
        # We need a mask to compute masked_MAE, let's use the degraded diff
        diff = np.abs(img_rgb.astype(np.float32) - gt_rgb.astype(np.float32)).mean(axis=-1)
        gt_mask = (diff > 20).astype(np.uint8) * 255 # heuristic rough mask
        
        masked_mae_degraded = np.sum(np.abs(img_rgb.astype(np.float32) - gt_rgb.astype(np.float32)) * (gt_mask[..., None]>0)) / (gt_mask.sum() * 3 + 1e-8)

        # Run R013
        res_013 = pipeline.run(
            image_path=img_path,
            output_dir=out_dir / "r013" / sample,
            segmenter_arch="r013_custom_attnunet"
        )
        out_013 = cv2.cvtColor(cv2.imread(str(res_013.restored_path)), cv2.COLOR_BGR2RGB)
        r013_psnr = compute_psnr(out_013, gt_rgb)
        r013_mae = compute_mae_image(out_013, gt_rgb)
        masked_mae_r013 = np.sum(np.abs(out_013.astype(np.float32) - gt_rgb.astype(np.float32)) * (gt_mask[..., None]>0)) / (gt_mask.sum() * 3 + 1e-8)

        # Run R014
        res_014 = pipeline.run(
            image_path=img_path,
            output_dir=out_dir / "r014" / sample,
            segmenter_arch="r014_resnet34",
            segmenter_checkpoint=r014_ckpt,
            segmenter_threshold=0.3
        )
        out_014 = cv2.cvtColor(cv2.imread(str(res_014.restored_path)), cv2.COLOR_BGR2RGB)
        r014_psnr = compute_psnr(out_014, gt_rgb)
        r014_mae = compute_mae_image(out_014, gt_rgb)
        masked_mae_r014 = np.sum(np.abs(out_014.astype(np.float32) - gt_rgb.astype(np.float32)) * (gt_mask[..., None]>0)) / (gt_mask.sum() * 3 + 1e-8)

        results.append({
            "sample": sample,
            "degraded_PSNR": degraded_psnr,
            "R013_PSNR": r013_psnr,
            "R014_PSNR": r014_psnr,
            "degraded_MAE": degraded_mae,
            "R013_MAE": r013_mae,
            "R014_MAE": r014_mae,
            "masked_MAE_degraded": masked_mae_degraded,
            "masked_MAE_R013": masked_mae_r013,
            "masked_MAE_R014": masked_mae_r014,
        })
        
    df = pd.DataFrame(results)
    df.to_csv("F:/deeplearning/experiment_value/r014_resnet34_safe_experiment/pipeline_check/PIPE_07_restoration_metrics.csv", index=False)
    
    # improved counts
    r014_better_psnr = (df['R014_PSNR'] > df['R013_PSNR']).sum()
    r014_better_mae = (df['R014_MAE'] < df['R013_MAE']).sum()
    
    with open("F:/deeplearning/experiment_value/r014_resnet34_safe_experiment/pipeline_check/PIPE_07_restoration_check.md", "w") as f:
        f.write("# Restoration Check\n")
        f.write(f"- Mean R013 PSNR: {df['R013_PSNR'].mean():.4f}\n")
        f.write(f"- Mean R014 PSNR: {df['R014_PSNR'].mean():.4f}\n")
        f.write(f"- Mean R013 MAE: {df['R013_MAE'].mean():.4f}\n")
        f.write(f"- Mean R014 MAE: {df['R014_MAE'].mean():.4f}\n")
        f.write(f"- Mean R013 Masked MAE: {df['masked_MAE_R013'].mean():.4f}\n")
        f.write(f"- Mean R014 Masked MAE: {df['masked_MAE_R014'].mean():.4f}\n")
        f.write(f"- R014 improved PSNR over R013 in {r014_better_psnr}/15 samples\n")
        f.write(f"- R014 improved MAE over R013 in {r014_better_mae}/15 samples\n")

if __name__ == "__main__":
    main()
