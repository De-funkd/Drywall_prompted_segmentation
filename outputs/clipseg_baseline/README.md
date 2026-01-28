# Baseline Model Visualization

This directory contains visualizations for the baseline CLIPSeg model results. The visualizations compare ground truth masks with predictions from the baseline model for both cracks and taping datasets.

## Directory Structure

```
outputs/clipseg_baseline/
├── cracks/
│   ├── crack_on_drywall_surface/
│   ├── drywall_crack/
│   │   ├── visualizations/     # Side-by-side comparison images
│   │   ├── samples/            # Sample visualizations for quick review
│   │   └── *.png               # Prediction masks
│   ├── drywall_joint_seam/
│   └── drywall_joint_tape/
├── taping/
│   ├── crack_on_drywall_surface/
│   ├── drywall_crack/
│   ├── drywall_joint_seam/
│   └── drywall_joint_tape/
│       ├── visualizations/     # Side-by-side comparison images  
│       ├── samples/            # Sample visualizations for quick review
│       └── *.png               # Prediction masks
└── README.md                   # This file
```

## Visualization Format

Each visualization image shows:
- Left: Original image
- Middle: Ground truth mask (with prompt used)
- Right: Predicted mask (with IoU and Dice scores)

## Special Handling for Taping Dataset

The taping dataset has multiple ground truth masks per image. For visualization purposes, the script identifies the ground truth mask with the highest IoU to the prediction and uses that for the comparison visualization. This approach provides a representative comparison while acknowledging that the model produces a single prediction for images that may have multiple targets.

## Generated Files

- `*_vis.png` - Visualizations for the cracks dataset
- `*_best_match_vis.png` - Visualizations for the taping dataset (showing best GT match)
- `*_pred_mask.png` - Prediction masks for easy access

## Metrics Included

Each visualization includes:
- IoU (Intersection over Union) score
- Dice coefficient
- Prompt used for segmentation

## Visualization Scripts

Visualization scripts are located in the `utils/` directory:
- `visualize_finetuned_results.py` - For generating finetuned model visualizations
- `visualize_ensemble_results.py` - For generating ensemble model visualizations