"""
Sanity Check Inference Script for Phase 3B Fine-tuning

This script performs a quick visual comparison between the base CLIPSeg model
and the Phase 3B fine-tuned model to verify that learning occurred during
the fine-tuning process.

Why this sanity check is important:
- Confirms that the fine-tuned model produces different (hopefully better) predictions than the base model
- Demonstrates that the model learned domain-specific features for drywall crack detection
- Provides immediate visual feedback that the training was successful
- Allows for quick verification without running full evaluation pipelines
"""
import os
import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

def compute_iou(pred_mask, gt_mask):
    """Compute Intersection over Union"""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection) / float(union)

def compute_dice(pred_mask, gt_mask):
    """Compute Dice coefficient"""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    total = pred_mask.sum() + gt_mask.sum()
    if total == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(2 * intersection) / float(total)

def main():
    # Configuration - user can edit these paths
    image_path = "data/processed/cracks/valid/images/example_image.jpg"  # Hardcoded path - user can edit
    checkpoint_path = "outputs/clipseg_finetuned/best_model.pth"  # Path to Phase 3B checkpoint
    mask_path = "data/processed/cracks/valid/masks/example_mask.png"  # Ground truth mask (optional)
    prompt = "drywall crack"
    
    # Check if image exists
    if not os.path.exists(image_path):
        # Try to find a valid image in the cracks validation set
        import glob
        image_paths = glob.glob("data/processed/cracks/valid/images/*.*")
        if image_paths:
            image_path = image_paths[0]
            print(f"Using image: {image_path}")
        else:
            print("No image found. Please update the image_path variable.")
            return
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Initialize processor
    processor = CLIPSegProcessor.from_pretrained('CIDAS/clipseg-rd64-refined')
    
    # Load base model
    base_model = CLIPSegForImageSegmentation.from_pretrained('CIDAS/clipseg-rd64-refined')
    base_model.eval()
    
    # Load fine-tuned model
    finetuned_model = CLIPSegForImageSegmentation.from_pretrained('CIDAS/clipseg-rd64-refined')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    finetuned_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    finetuned_model.eval()
    
    # Prepare input
    inputs = processor(text=[prompt], images=[Image.fromarray(image_rgb)], padding=True, return_tensors="pt")
    
    # Run inference with base model
    with torch.no_grad():
        base_outputs = base_model(**inputs)
        base_logits = base_outputs.logits
        base_probs = torch.sigmoid(base_logits).squeeze().cpu().numpy()
        base_prediction = (base_probs > 0.5).astype(np.uint8)
    
    # Run inference with fine-tuned model
    with torch.no_grad():
        finetuned_outputs = finetuned_model(**inputs)
        finetuned_logits = finetuned_outputs.logits
        finetuned_probs = torch.sigmoid(finetuned_logits).squeeze().cpu().numpy()
        finetuned_prediction = (finetuned_probs > 0.5).astype(np.uint8)
    
    # Load ground truth mask if it exists
    has_gt = os.path.exists(mask_path)
    if has_gt:
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if gt_mask is not None:
            gt_mask = (gt_mask > 127).astype(np.uint8)
            # Resize masks to match ground truth if needed
            if gt_mask.shape != base_prediction.shape:
                gt_mask_resized = cv2.resize(gt_mask, (base_prediction.shape[1], base_prediction.shape[0]), interpolation=cv2.INTER_NEAREST)
                base_prediction_resized = cv2.resize(base_prediction, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
                finetuned_prediction_resized = cv2.resize(finetuned_prediction, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
                
                base_iou = compute_iou(base_prediction_resized, gt_mask_resized)
                base_dice = compute_dice(base_prediction_resized, gt_mask_resized)
                ft_iou = compute_iou(finetuned_prediction_resized, gt_mask_resized)
                ft_dice = compute_dice(finetuned_prediction_resized, gt_mask_resized)
            else:
                base_iou = compute_iou(base_prediction, gt_mask)
                base_dice = compute_dice(base_prediction, gt_mask)
                ft_iou = compute_iou(finetuned_prediction, gt_mask)
                ft_dice = compute_dice(finetuned_prediction, gt_mask)
        else:
            has_gt = False
    
    # Create side-by-side visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Base model prediction
    axes[1].imshow(base_prediction, cmap='gray')
    title = "Base Model Prediction"
    if has_gt:
        title += f"\n(IoU: {base_iou:.3f}, Dice: {base_dice:.3f})"
    axes[1].set_title(title)
    axes[1].axis('off')
    
    # Fine-tuned model prediction
    axes[2].imshow(finetuned_prediction, cmap='gray')
    title = "Fine-tuned Model Prediction"
    if has_gt:
        title += f"\n(IoU: {ft_iou:.3f}, Dice: {ft_dice:.3f})"
    axes[2].set_title(title)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig("sanity_check_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print metrics if ground truth exists
    if has_gt:
        print(f"Base Model - IoU: {base_iou:.4f}, Dice: {base_dice:.4f}")
        print(f"Fine-tuned Model - IoU: {ft_iou:.4f}, Dice: {ft_dice:.4f}")
        print(f"IoU Improvement: {ft_iou - base_iou:.4f}")
        print(f"Dice Improvement: {ft_dice - base_dice:.4f}")
    else:
        print("No ground truth mask found - skipping quantitative metrics")
    
    print("Sanity check completed. Visualization saved as 'sanity_check_comparison.png'")

if __name__ == "__main__":
    main()