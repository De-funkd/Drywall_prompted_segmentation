import os
import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from pathlib import Path

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

def run_extended_ensemble_inference():
    print("Starting extended CLIPSeg prompt ensembling inference...")

    # Define prompt sets
    cracks_prompts = [
        "drywall crack",
        "crack on drywall surface", 
        "thin crack on wall",
        "hairline crack in drywall"
    ]
    
    taping_prompts = [
        "drywall joint tape",
        "drywall joint seam",
        "taped drywall joint", 
        "drywall seam line"
    ]

    # Define datasets
    datasets = {
        "cracks": {
            "path": "data/processed/cracks/valid/images",
            "prompts": cracks_prompts
        },
        "taping": {
            "path": "data/processed/taping/valid/images", 
            "prompts": taping_prompts
        }
    }

    # Initialize model and processor
    try:
        model = CLIPSegForImageSegmentation.from_pretrained('CIDAS/clipseg-rd64-refined')
        processor = CLIPSegProcessor.from_pretrained('CIDAS/clipseg-rd64-refined')
        model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
        print("Loaded CLIPSeg model successfully")
    except Exception as e:
        print(f"Failed to load CLIPSeg model: {e}")
        return

    # Metrics storage
    all_metrics = {}

    for dataset_name, dataset_info in datasets.items():
        img_dir = dataset_info["path"]
        prompts = dataset_info["prompts"]
        
        if not os.path.exists(img_dir):
            print(f"Dataset {dataset_name} not found at {img_dir}, skipping...")
            continue

        print(f"\nProcessing {dataset_name} dataset...")

        dataset_metrics = {}
        
        # Get ALL image files (not just the first few)
        img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"Found {len(img_files)} images to process...")

        for i, img_file in enumerate(img_files):
            if i >= 5:  # Limit to first 5 images to match Phase 2
                break
                
            img_path = os.path.join(img_dir, img_file)

            # Load image
            image = cv2.imread(img_path)
            if image is None:
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = image_rgb.shape[:2]

            # Load corresponding ground truth mask
            mask_file = img_file.rsplit('.', 1)[0] + "_mask.png"
            mask_path = os.path.join(img_dir.replace('/images', '/masks'), mask_file)
            if os.path.exists(mask_path):
                gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if gt_mask is not None:
                    gt_mask = (gt_mask > 127).astype(bool)  # Convert to binary
                else:
                    gt_mask = np.zeros((orig_h, orig_w), dtype=bool)
            else:
                # Try to find a matching mask file by looking for similar names
                mask_candidates = list(Path(img_dir.replace('/images', '/masks')).glob(f"*{img_file.split('.')[0]}*"))
                if mask_candidates:
                    gt_mask = cv2.imread(str(mask_candidates[0]), cv2.IMREAD_GRAYSCALE)
                    if gt_mask is not None:
                        gt_mask = (gt_mask > 127).astype(bool)
                    else:
                        gt_mask = np.zeros((orig_h, orig_w), dtype=bool)
                else:
                    gt_mask = np.zeros((orig_h, orig_w), dtype=bool)

            print(f"  Processing ({i+1}/{min(5, len(img_files))}) image {img_file} with {len(prompts)} prompts...")

            # Run inference for each prompt
            prompt_predictions = []
            for prompt_idx, prompt in enumerate(prompts):
                # Preprocess image for model
                inputs = processor(text=[prompt], images=[Image.fromarray(image_rgb)],
                                  padding=True, return_tensors="pt")
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)
                    pred = outputs.logits.cpu().numpy()[0]
                
                # Resize prediction to original image size
                pred_resized = cv2.resize(pred, (orig_w, orig_h))

                # Normalize to [0,1]
                pred_normalized = (pred_resized - pred_resized.min()) / (pred_resized.max() - pred_resized.min() + 1e-8)
                
                prompt_predictions.append(pred_normalized)
            
            # Stack predictions for ensembling
            stacked_preds = np.stack(prompt_predictions, axis=0)  # Shape: (num_prompts, H, W)
            
            # Apply ensemble methods
            max_ensemble = np.max(stacked_preds, axis=0)
            mean_ensemble = np.mean(stacked_preds, axis=0)
            
            # Apply different thresholds
            for threshold in [0.3, 0.5]:
                # Max ensemble with threshold
                max_pred_binary = (max_ensemble > threshold).astype(bool)
                
                # Mean ensemble with threshold  
                mean_pred_binary = (mean_ensemble > threshold).astype(bool)
                
                # Compute metrics
                max_iou = compute_iou(max_pred_binary, gt_mask)
                max_dice = compute_dice(max_pred_binary, gt_mask)
                
                mean_iou = compute_iou(mean_pred_binary, gt_mask)
                mean_dice = compute_dice(mean_pred_binary, gt_mask)
                
                # Store metrics
                if f"max_thr{threshold}" not in dataset_metrics:
                    dataset_metrics[f"max_thr{threshold}"] = {"iou": [], "dice": []}
                    dataset_metrics[f"mean_thr{threshold}"] = {"iou": [], "dice": []}
                
                dataset_metrics[f"max_thr{threshold}"]["iou"].append(max_iou)
                dataset_metrics[f"max_thr{threshold}"]["dice"].append(max_dice)
                dataset_metrics[f"mean_thr{threshold}"]["iou"].append(mean_iou)
                dataset_metrics[f"mean_thr{threshold}"]["dice"].append(mean_dice)
                
                # Save predicted masks (overwrite if they exist)
                max_mask_path = f"outputs/clipseg_ensemble/{dataset_name}/max/{img_file.rsplit('.', 1)[0]}_pred_mask_thr{threshold}.png"
                mean_mask_path = f"outputs/clipseg_ensemble/{dataset_name}/mean/{img_file.rsplit('.', 1)[0]}_pred_mask_thr{threshold}.png"
                
                cv2.imwrite(max_mask_path, (max_pred_binary.astype(np.uint8) * 255))
                cv2.imwrite(mean_mask_path, (mean_pred_binary.astype(np.uint8) * 255))
                
                # Create visualization with both max and mean ensembles
                fig, axes = plt.subplots(1, 4, figsize=(20, 5))

                axes[0].imshow(image_rgb)
                axes[0].set_title("Original Image")
                axes[0].axis('off')

                axes[1].imshow(gt_mask, cmap='gray')
                axes[1].set_title("Ground Truth Mask")
                axes[1].axis('off')

                axes[2].imshow(max_pred_binary, cmap='gray')
                axes[2].set_title(f"Max Ensemble Mask\n(Thr: {threshold})")
                axes[2].axis('off')

                axes[3].imshow(mean_pred_binary, cmap='gray')
                axes[3].set_title(f"Mean Ensemble Mask\n(Thr: {threshold})")
                axes[3].axis('off')

                plt.suptitle(f"Max Ensemble - IoU: {max_iou:.3f}, Dice: {max_dice:.3f} | Mean Ensemble - IoU: {mean_iou:.3f}, Dice: {mean_dice:.3f}")
                plt.tight_layout()

                vis_path = f"outputs/clipseg_ensemble/{dataset_name}/max/{img_file.replace('.', '_')}_ensemble_vis_thr{threshold}.png"
                plt.savefig(vis_path, dpi=150, bbox_inches='tight')
                plt.close()

                print(f"    Processed {img_file} with threshold {threshold}: Max(IoU={max_iou:.3f}, Dice={max_dice:.3f}), Mean(IoU={mean_iou:.3f}, Dice={mean_dice:.3f})")

        # Calculate average metrics for this dataset
        for method_threshold, values in dataset_metrics.items():
            if values["iou"]:  # Check if there are any values
                avg_iou = np.mean(values["iou"])
                avg_dice = np.mean(values["dice"])
                print(f"  Average for {method_threshold}: IoU={avg_iou:.3f}, Dice={avg_dice:.3f}")

        all_metrics[dataset_name] = dataset_metrics

    return all_metrics

def main():
    # Run extended ensemble inference
    metrics = run_extended_ensemble_inference()

    # Print summary
    print("\n" + "="*60)
    print("EXTENDED CLIPSEG PROMPT ENSEMBLE COMPLETE")
    print("="*60)
    for dataset, dataset_metrics in metrics.items():
        print(f"\n{dataset.upper()} DATASET:")
        for method_threshold, values in dataset_metrics.items():
            if values["iou"]:  # Check if there are any values
                avg_iou = np.mean(values["iou"])
                avg_dice = np.mean(values["dice"])
                print(f"  {method_threshold}: IoU={avg_iou:.3f}, Dice={avg_dice:.3f}")

if __name__ == "__main__":
    main()