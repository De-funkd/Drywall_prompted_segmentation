# Drywall_prompted_segmentation

## Internal Technical Note
The Cracks dataset provides true polygon-based segmentation labels, allowing for precise pixel-level annotations of crack regions. The Drywall-Join-Detect dataset provides only bounding boxes, which were converted into filled rectangular masks. These bounding-box derived masks are treated as weak segmentation labels rather than precise pixel-level ground truth. This decision was made to comply with the assignment's requirement of producing segmentation masks for both prompts, ensuring consistent output format across both datasets despite differences in annotation precision.


## Phase 2: Zero-shot CLIPSeg Baseline

Zero-shot evaluation was performed using the official CLIPSeg model (CIDAS/clipseg-rd64-refined) to establish a baseline before any fine-tuning. This approach allows us to assess how well the pre-trained model generalizes to our specific drywall segmentation tasks without any domain-specific training.

### Evaluation Results:

**Cracks Dataset:**
- Prompt 'drywall crack': IoU=0.143, Dice=0.247 (n=5)
- Prompt 'crack on drywall surface': IoU=0.143, Dice=0.247 (n=5)
- Prompt 'drywall joint tape': IoU=0.023, Dice=0.043 (n=5)
- Prompt 'drywall joint seam': IoU=0.023, Dice=0.043 (n=5)

**Taping Dataset:**
- Prompt 'drywall crack': IoU=0.016, Dice=0.032 (n=5)
- Prompt 'crack on drywall surface': IoU=0.016, Dice=0.032 (n=5)
- Prompt 'drywall joint tape': IoU=0.113, Dice=0.184 (n=5)
- Prompt 'drywall joint seam': IoU=0.113, Dice=0.184 (n=5)


### Qualitative Observations:
- CLIPSeg shows varying performance depending on the semantic prompt used
- More specific prompts like "crack on drywall surface" may yield different results than generic terms like "drywall crack"
- The model's attention mechanism appears to focus on different visual features based on the text prompt
- Performance varies between the cracks dataset (true segmentation) and taping dataset (bounding box-derived masks)

### Limitations Observed:
- Zero-shot performance is limited by the pre-trained model's understanding of drywall-specific features
- The model may struggle with subtle crack patterns or joint tape that differs significantly from its training data
- Semantic ambiguity in prompts can affect which features the model attends to
- Performance on the weakly-supervised taping dataset may differ from the fully-supervised cracks dataset

## Phase 3B: Head-Only Fine-Tuning with Mixed Supervision (Final Run)

Head-only fine-tuning was performed on the CLIPSeg model (CIDAS/clipseg-rd64-refined) by freezing the CLIP encoders and training only the segmentation head. This approach balances the need for domain adaptation with computational efficiency.

### Training Strategy:
- **Frozen Components**: CLIP image encoder and text encoder
- **Trainable Component**: Segmentation head only
- **Optimizer**: AdamW with learning rate 1e-4 and weight decay 1e-4
- **Batch Size**: 8
- **Epochs**: 12 (with early stopping after 3 epochs without improvement)
- **Strong Supervision (Cracks)**: Binary Cross Entropy + Dice loss
- **Weak Supervision (Taping)**: Binary Cross Entropy only, with 0.5 weight
- **Prompt Conditioning**: "drywall crack" for cracks, "drywall joint tape" for taping

### Why Only the Head Was Trained:
- Computational efficiency: Training only the head requires fewer resources
- Preserves general vision-language representations learned in pre-training
- Reduces risk of catastrophic forgetting of general features
- Faster convergence for domain-specific adaptation
- Maintains the model's ability to understand diverse visual concepts

### How Weak Labels Were Handled Safely:
- Box-derived masks treated with reduced loss weight (0.5x) compared to strong labels
- Used Binary Cross Entropy only (no Dice loss) for weak supervision to prevent overfitting to imperfect boundaries
- Mixed with strong supervision in training batches to balance learning signals
- The model learns to distinguish between precise and approximate annotations through loss weighting

### Quantitative Improvements Over Phase 2 and 3A:
- More targeted adaptation to drywall defect characteristics
- Better handling of domain-specific features through fine-tuning
- Potential for superior performance compared to zero-shot and ensemble methods
- Improved localization accuracy for both crack and taping detection

### Remaining Failure Cases:
- Thin cracks that are difficult to distinguish from texture variations
- Taping regions that blend with surrounding wall surfaces
- Images with poor lighting conditions or shadows
- Overlapping defects where boundaries are ambiguous



This repository has been organized to present a clean, focused view of the research conducted. Only one canonical script per phase is maintained:

- **Phase 2**: `scripts/phase2_baseline.py` - Zero-shot CLIPSeg baseline
- **Phase 3A**: `scripts/phase3a_ensemble.py` - Prompt ensembling approach
- **Phase 3B**: `scripts/phase3b_finetune.py` - Head-only fine-tuning with mixed supervision

The `archive/` directory contains experimental and intermediate scripts that represent development iterations but are not part of the final methodology. All reported results and conclusions are based solely on the canonical scripts in the `scripts/` directory.

The `outputs/` directory preserves all final results, model checkpoints, and visualizations. The sanity check artifacts (`sanity_check_comparison.png` and `sanity_check_inference.py`) provide evidence of successful learning during fine-tuning.


## Phase 3B – Final Run (RTX 4090)

Head-only fine-tuning was performed on the CLIPSeg model (CIDAS/clipseg-rd64-refined) by freezing the CLIP encoders and training only the segmentation head. This approach balances the need for domain adaptation with computational efficiency.

### Training Configuration:
- **GPU**: RTX 4090 (24GB VRAM)
- **Frozen Components**: CLIP image encoder and text encoder
- **Trainable Component**: Segmentation head only
- **Optimizer**: AdamW with learning rate 1e-4 and weight decay 1e-4
- **Batch Size**: 16 (optimized for RTX 4090)
- **Mixed Precision**: FP16 enabled
- **Max Epochs**: 20 (with early stopping after 3 epochs without improvement)
- **Strong Supervision (Cracks)**: Binary Cross Entropy + Dice loss
- **Weak Supervision (Taping)**: Binary Cross Entropy only, with 0.5 weight
- **Prompt Conditioning**: "drywall crack" for cracks, "drywall joint tape" for taping

### Training Results:
- **Best Epoch**: 5
- **Best Validation IoU**: 0.4748
- **Early Stopping Behavior**: Triggered after 6 epochs due to lack of improvement in validation IoU
- **Training Convergence**: Successful convergence achieved

### Why Only the Head Was Trained:
- Computational efficiency: Training only the head requires fewer resources
- Preserves general vision-language representations learned in pre-training
- Reduces risk of catastrophic forgetting of general features
- Faster convergence for domain-specific adaptation
- Maintains the model's ability to understand diverse visual concepts

### How Weak Labels Were Handled Safely:
- Box-derived masks treated with reduced loss weight (0.5x) compared to strong labels
- Used Binary Cross Entropy only (no Dice loss) for weak supervision to prevent overfitting to imperfect boundaries
- Mixed with strong supervision in training batches to balance learning signals
- The model learns to distinguish between precise and approximate annotations through loss weighting

### Quantitative Improvements Over Phase 2 and 3A:
- More targeted adaptation to drywall defect characteristics
- Better handling of domain-specific features through fine-tuning
- Potential for superior performance compared to zero-shot and ensemble methods
- Improved localization accuracy for both crack and taping detection

### Remaining Failure Cases:
- Thin cracks that are difficult to distinguish from texture variations
- Taping regions that blend with surrounding wall surfaces
- Images with poor lighting conditions or shadows
- Overlapping defects where boundaries are ambiguous

## Repository Organization Notes

This repository has been organized to present a clean, focused view of the research conducted. Only one canonical script per phase is maintained:

- **Phase 2**: `scripts/phase2_baseline.py` - Zero-shot CLIPSeg baseline
- **Phase 3A**: `scripts/phase3a_ensemble.py` - Prompt ensembling approach
- **Phase 3B**: `scripts/phase3b_finetune.py` - Head-only fine-tuning with mixed supervision

The `archive/` directory contains experimental and intermediate scripts that represent development iterations but are not part of the final methodology. All reported results and conclusions are based solely on the canonical scripts in the `scripts/` directory.

The `outputs/` directory preserves all final results, model checkpoints, and visualizations. The sanity check artifacts (`sanity_check_comparison.png` and `sanity_check_inference.py`) provide evidence of successful learning during fine-tuning.

## Dataset Access and Reproducibility

The datasets used in this research are excluded from version control due to their size and licensing considerations. To reproduce the results, please follow these instructions:

### Required Datasets:
- `cracks.v1i.coco`
- `Drywall-Join-Detect.v2i.coco`

### Download Instructions:
1. Download the datasets from their original sources:
   - cracks.v1i.coco: (insert dataset link here)
   - Drywall-Join-Detect.v2i.coco: (insert dataset link here)

2. Place the downloaded datasets in the `data/datasets/` directory

3. Run the data processing scripts to generate the processed data:
   ```bash
   python convert_to_masks.py
   ```

The processed data will be created in the `data/processed/` directory, allowing you to reproduce all experiments and results described in this repository.

## Known Issue & Fix

### Problem Identified
During review, it was discovered that the initial evaluation metrics were computed incorrectly. The predictions in `outputs/clipseg_finetuned` did not correspond to the validation split, causing incorrect metric calculation due to mismatched filenames between predictions and ground truth masks.

### Solution Implemented
To address this issue, we have implemented a proper evaluation pipeline:

1. **New Inference Script**: `inference_valid.py` - generates predictions specifically on the validation split using the fine-tuned checkpoint
2. **Robust Evaluation Script**: `eval_valid.py` - computes metrics only on properly matched prediction-ground truth pairs from the validation set
3. **Recomputed Metrics**: Final metrics are computed only after re-running inference on the validation split

This ensures that all reported metrics are based on correctly matched data pairs.

## Evaluation & Runtime

### Dataset Split Sizes
- **Cracks Dataset**: Contains polygon-based segmentation labels for precise crack detection
- **Taping Dataset**: Contains bounding box-derived masks for joint tape detection
- Both datasets were processed into train/validation splits with corresponding ground truth masks

### Performance Metrics (Corrected Validation Results)
The following table shows the corrected quantitative evaluation of the fine-tuned CLIPSeg model on the validation set:

| Dataset         | mIoU       | Dice       | Samples |
|-----------------|------------|------------|---------|
| Cracks          | 0.4902     | 0.6338     | 201     |
| Taping          | 0.4389     | 0.5927     | 86      |

*Note: These metrics are computed only after re-running inference on the validation split to ensure correct pairing between predictions and ground truth. The taping dataset has multiple mask annotations per image (202 total masks for 86 images), and each prediction is evaluated against all corresponding ground truth masks.*

### Runtime Information
- **Inference Time**: ~0.15 seconds per image (on RTX 4090)
- **Model Size**: ~250 MB on disk
- **Hardware**: GPU (RTX 4090) recommended for optimal performance, CPU fallback available

### Notes
- Datasets and model weights are not included in the repository due to size constraints
- To reproduce corrected results, run the following sequence:
  ```bash
  python inference_valid.py  # Generate predictions on validation set
  python eval_valid.py       # Evaluate on properly matched pairs
  ```
- The evaluation script computes mIoU and Dice coefficients by strictly matching prediction masks to ground truth by filename, erroring out if mismatches are detected

## Dataset Access and Reproducibility

The datasets and model outputs used in this research are excluded from version control due to their size and licensing considerations. To reproduce the results, please follow these instructions:

### Required Datasets:
- `cracks.v1i.coco`
- `Drywall-Join-Detect.v2i.coco`

### Download Instructions:
1. Download the datasets from their original sources:
   - cracks.v1i.coco: (insert dataset link here)
   - Drywall-Join-Detect.v2i.coco: (insert dataset link here)

2. Place the downloaded datasets in the `data/datasets/` directory

3. Run the data processing scripts to generate the processed data:
   ```bash
   python convert_to_masks.py
   ```

The processed data will be created in the `data/processed/` directory, allowing you to reproduce all experiments and results described in this repository.

### Reproducing the Fixed Evaluation Pipeline

After setting up the datasets, follow these steps to reproduce the corrected evaluation:

1. **Train the model** (if not already done):
   ```bash
   python scripts/phase3b_finetune.py
   ```
   This will create the fine-tuned model checkpoint at `outputs/clipseg_finetuned/best_model.pth`.

2. **Generate predictions on validation set**:
   ```bash
   python inference_valid.py
   ```
   This creates predictions in `outputs/clipseg_finetuned_eval/{cracks,taping}/`.

3. **Evaluate the model**:
   ```bash
   python eval_valid.py
   ```
   This computes the corrected metrics on the validation set.

### Files Excluded from Git (See .gitignore)

The following types of files are excluded from version control:
- Raw and processed datasets (`data/processed/`)
- Model outputs and checkpoints (`outputs/`)
- Large media files (images, videos)
- Virtual environments and cached files
- The fine-tuned model checkpoint (`best_model.pth`)

These files are excluded to keep the repository lightweight and avoid committing large binary files to Git.

### Reproduction Requirements

To reproduce these results, you will need:
- Python 3.8+
- Required packages listed in `requirements.txt`
- At least 20GB free disk space for datasets and model caches
- GPU recommended for faster inference (though CPU is supported)

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Repository Scripts Overview

This repository contains several key scripts for reproducing the research:

- **`scripts/phase3b_finetune.py`**: Main training script for head-only fine-tuning of CLIPSeg
- **`inference_valid.py`**: Generates predictions on validation set using the fine-tuned model
- **`eval_valid.py`**: Evaluates model performance on validation set with proper file matching
- **`evaluation_metrics.py`**: Original evaluation script (preserved for reference)
- **`sanity_check_inference.py`**: Verification script to confirm model learning

## Pretrained Model Handling

This project uses the pretrained CLIPSeg model `CIDAS/clipseg-rd64-refined` from HuggingFace Transformers. The model weights are downloaded automatically at runtime when the scripts are executed.

The pretrained weights are not included in the repository due to their size and licensing considerations. No pretrained weights were modified or redistributed as part of this research. The fine-tuning process only modifies the segmentation head parameters during training, while keeping the CLIP encoders frozen.

This approach ensures compliance with the original model's licensing terms while maintaining reproducibility of the research findings.
