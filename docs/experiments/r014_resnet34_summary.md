# Experiment R014: ResNet-34 Encoder Segmentation

This document details the findings from experiment R014, an isolated iteration aiming to assess the impact of using an ImageNet-pretrained ResNet-34 encoder compared to the baseline lightweight custom Attention U-Net (R013).

## Objective
Evaluate whether substituting the encoder block with a standard `resnet34(pretrained=True)` provides significant segmentation gains, and define an appropriate morphological dilation policy to ensure compatibility with downstream LaMa inpainting.

## Metrics (Held-out Test Split)

| Model | Encoder | Pretrained | Test IoU | Test F1 |
|-------|---------|------------|----------|---------|
| R013 (Baseline) | Custom ConvBlocks | No | 0.3664 | 0.5363 |
| R014 (Experimental) | ResNet-34 | Yes | 0.4506 | 0.6213 |

*Note: Both models utilize deep supervision during training and Attention Gates in the decoder. Test split metrics show a measurable improvement in raw segmentation precision.*

## Post-processing & Morphological Sweep
While R014 exhibited stronger segmentation IoU, initial visual evaluations indicated that its predicted masks were occasionally too narrow/conservative compared to R013, likely due to prompt compliance in the underlying datasets.

To align R014's masks with the wide-mask assumption of LaMa inpainting, a 15-sample paired morphological sweep was conducted:
- **Best Threshold**: `0.30`
- **Best Dilation Radius**: `1` (yielding a 3x3 kernel)
- **Outcome**: The optimized R014 policy improved the downstream "Masked Mean Absolute Error (MAE)" by approximately `-1.04` compared to R013 on the evaluation subset.

## Conclusion & Integration
- **R014 + 3x3 dilation** is integrated into the UI and CLI as a **recommended experimental/demo option**.
- **R013** remains the **stable default fallback**, as it has proven reliable across a wider variety of in-the-wild historical photos and does not require an external heavy checkpoint.
- Users wishing to test R014 must explicitly supply the checkpoint path or configure the `R014_SEGMENTER_CHECKPOINT` environment variable.
