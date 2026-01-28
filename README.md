# Drywall Prompted Segmentation

Binary segmentation project fine-tuned CLIPSeg on drywall and taping dataset for crack and joint tape detection.

## Repository Structure

```
├── archive/                    # Experimental and intermediate scripts
├── data/                       # Dataset files and processed data
├── utils                       # Evaluation, inference and visualisation generation scripts
├── outputs/                    # Model outputs and checkpoints
├── scripts/                    # Training and baseline scripts
├── .gitignore                  # Git ignore rules
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Overarching Goal

Fine-tune CLIPSeg model for binary segmentation of drywall defects (cracks and joint tape) using a mixed supervision approach with both strong (polygon-based) and weak (bounding box-derived) labels.

## How to Run/Replicate the Work

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Prepare datasets:
   - Download `cracks.v1i.coco` and `Drywall-Join-Detect.v2i.coco` datasets
   - Place in `data/datasets/` directory
   - Run data processing scripts to generate processed data

3. Train the model:
   ```bash
   python scripts/phase3b_finetune.py
   ```

4. Generate predictions on validation set:
   ```bash
   python evals_&_inf/inference_valid.py
   ```

5. Evaluate the model:
   ```bash
   python evals_&_inf/eval_valid.py
   ```

6. Perform sanity check:
   ```bash
   python evals_&_inf/sanity_check_inference.py
   ```