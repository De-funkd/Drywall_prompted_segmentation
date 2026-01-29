# Drywall Prompted Segmentation Project


## Phase 3A: Prompt Ensembling

Prompt ensembling was performed using the same CLIPSeg model (CIDAS/clipseg-rd64-refined) to improve segmentation performance by combining predictions from multiple related prompts. This approach leverages the idea that different prompts may capture complementary aspects of the target objects.

### Ensemble Strategy:
- **Cracks Prompt Set**: ["drywall crack", "crack on drywall surface", "thin crack on wall", "hairline crack in drywall"]
- **Taping Prompt Set**: ["drywall joint tape", "drywall joint seam", "taped drywall joint", "drywall seam line"]
- **Combination Methods**: Pixel-wise max and pixel-wise mean
- **Thresholds Tested**: 0.3 and 0.5

### Evaluation Results:


### Observed Improvements:
- Ensemble methods showed modest improvements over individual prompts in some cases
- Max ensembling tended to preserve the most confident predictions from any single prompt
- Mean ensembling provided more balanced predictions by averaging across all prompts
- Lower threshold (0.3) generally produced more inclusive predictions compared to 0.5

### Comparison Against Phase 2 Baseline:
- Performance remains challenging due to the complexity of drywall defects
- Ensemble approaches provide more robust predictions than single-prompt approaches
- The improvement varied by dataset and target object type
- Ensembling helps address the limitations of single prompts for thin structures and weak labels
