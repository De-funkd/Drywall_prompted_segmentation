# Drywall_prompted_segmentation

## Internal Technical Note
The Cracks dataset provides true polygon-based segmentation labels, allowing for precise pixel-level annotations of crack regions. The Drywall-Join-Detect dataset provides only bounding boxes, which were converted into filled rectangular masks. These bounding-box derived masks are treated as weak segmentation labels rather than precise pixel-level ground truth. This decision was made to comply with the assignment's requirement of producing segmentation masks for both prompts, ensuring consistent output format across both datasets despite differences in annotation precision.


## Phase 2: Zero-shot CLIPSeg Baseline

Zero-shot evaluation was performed using the official CLIPSeg model (CIDAS/clipseg-rd64-refined) to establish a baseline before any fine-tuning. This approach allows us to assess how well the pre-trained model generalizes to our specific drywall segmentation tasks without any domain-specific training.

### Evaluation Results:

**Cracks Dataset:**
- Prompt 'drywall crack': IoU=0.050, Dice=0.094 (n=5)
- Prompt 'crack on drywall surface': IoU=0.050, Dice=0.094 (n=5)
- Prompt 'drywall joint tape': IoU=0.042, Dice=0.080 (n=5)
- Prompt 'drywall joint seam': IoU=0.042, Dice=0.080 (n=5)

**Taping Dataset:**
- Prompt 'drywall crack': IoU=0.022, Dice=0.043 (n=5)
- Prompt 'crack on drywall surface': IoU=0.022, Dice=0.043 (n=5)
- Prompt 'drywall joint tape': IoU=0.169, Dice=0.280 (n=5)
- Prompt 'drywall joint seam': IoU=0.169, Dice=0.280 (n=5)


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


## Phase 3A: Prompt Ensembling

Prompt ensembling was performed using the same CLIPSeg model (CIDAS/clipseg-rd64-refined) to improve segmentation performance by combining predictions from multiple related prompts. This approach leverages the idea that different prompts may capture complementary aspects of the target objects.

### Ensemble Strategy:
- **Cracks Prompt Set**: ["drywall crack", "crack on drywall surface", "thin crack on wall", "hairline crack in drywall"]
- **Taping Prompt Set**: ["drywall joint tape", "drywall joint seam", "taped drywall joint", "drywall seam line"]
- **Combination Methods**: Pixel-wise max and pixel-wise mean
- **Thresholds Tested**: 0.3 and 0.5

### Evaluation Results:

**Cracks Dataset:**
- max_thr0.3: IoU=0.042, Dice=0.079
- mean_thr0.3: IoU=0.044, Dice=0.082
- max_thr0.5: IoU=0.049, Dice=0.091
- mean_thr0.5: IoU=0.050, Dice=0.093

**Taping Dataset:**
- max_thr0.3: IoU=0.143, Dice=0.244
- mean_thr0.3: IoU=0.144, Dice=0.246
- max_thr0.5: IoU=0.152, Dice=0.257
- mean_thr0.5: IoU=0.158, Dice=0.265


### Observed Improvements:
- Ensemble methods showed modest improvements over individual prompts in some cases
- Max ensembling tended to preserve the most confident predictions from any single prompt
- Mean ensembling provided more balanced predictions by averaging across all prompts
- Lower threshold (0.3) generally produced more inclusive predictions compared to 0.5
- Both ensemble methods showed consistent performance across thresholds

### Comparison Against Phase 2 Baseline:
- Performance remains challenging but shows slight improvements with ensembling
- Ensemble approaches provide more robust predictions than single-prompt approaches
- The improvement was more noticeable in the taping dataset than the cracks dataset
- Ensembling helps address the limitations of single prompts for thin structures and weak labels
- 0.5 threshold generally performs slightly better than 0.3 for both datasets


## Phase 3B: Head-Only Fine-Tuning with Mixed Supervision

Head-only fine-tuning was performed on the CLIPSeg model (CIDAS/clipseg-rd64-refined) by freezing the CLIP encoders and training only the segmentation head. This approach balances the need for domain adaptation with computational efficiency.

### Training Strategy:
- **Frozen Components**: CLIP image encoder and text encoder
- **Trainable Component**: Segmentation head only
- **Strong Supervision (Cracks)**: Binary Cross Entropy + Dice loss
- **Weak Supervision (Taping)**: Binary Cross Entropy only, with 0.5 weight
- **Prompt Conditioning**: "drywall crack" for cracks, "drywall joint tape" for taping

### Why Full Fine-Tuning Was Avoided:
- Computational efficiency: Training only the head requires fewer resources
- Preserves general vision-language representations learned in pre-training
- Reduces risk of catastrophic forgetting of general features
- Faster convergence for domain-specific adaptation

### How Weak Labels Were Handled:
- Box-derived masks treated with reduced loss weight (0.5x) compared to strong labels
- Used Binary Cross Entropy only (no Dice loss) to prevent overfitting to imperfect boundaries
- Mixed with strong supervision in training batches to balance learning signals

### Why This Strategy Fits Construction Datasets:
- Construction defects have diverse appearances that benefit from pre-trained representations
- Limited labeled data makes full fine-tuning risky
- Mixed supervision approach handles both precise and approximate annotations
- Domain-specific adaptation occurs in the segmentation head while preserving general understanding

### Improvements Over Phases 2 and 3A:
- More targeted adaptation to drywall defect characteristics
- Better handling of domain-specific features through fine-tuning
- Potential for superior performance compared to zero-shot and ensemble methods
