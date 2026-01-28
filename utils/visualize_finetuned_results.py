"""
Visualization script for finetuned CLIPSeg model results.

This script creates visualizations comparing ground truth masks with predictions
from the finetuned model for both cracks and taping datasets.

For the taping dataset, which has multiple ground truth masks per image,
this script finds the ground truth mask with the highest IoU to the prediction
and uses that for visualization. This approach provides a representative
comparison while acknowledging that the model produces a single prediction
for images that may have multiple targets.

The script generates organized output directories mirroring the baseline
model structure with visualizations and prediction copies.
"""

import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def compute_iou(pred_mask, gt_mask):
    """
    Compute Intersection over Union between prediction and ground truth masks.
    
    Args:
        pred_mask: Predicted mask (binary, values {0, 255})
        gt_mask: Ground truth mask (binary, values {0, 255})
        
    Returns:
        IoU score (float)
    """
    # Convert to boolean arrays
    pred_bool = pred_mask > 127  # threshold at 127 to handle {0, 255}
    gt_bool = gt_mask > 127

    # Calculate intersection and union
    intersection = np.logical_and(pred_bool, gt_bool).sum()
    union = np.logical_or(pred_bool, gt_bool).sum()

    # Handle edge case where both masks are empty
    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return intersection / union


def compute_dice(pred_mask, gt_mask):
    """
    Compute Dice coefficient between prediction and ground truth masks.
    
    Args:
        pred_mask: Predicted mask (binary, values {0, 255})
        gt_mask: Ground truth mask (binary, values {0, 255})
        
    Returns:
        Dice coefficient (float)
    """
    # Convert to boolean arrays
    pred_bool = pred_mask > 127  # threshold at 127 to handle {0, 255}
    gt_bool = gt_mask > 127

    # Calculate intersection and cardinalities
    intersection = np.logical_and(pred_bool, gt_bool).sum()
    pred_sum = pred_bool.sum()
    gt_sum = gt_bool.sum()

    # Handle edge case where both masks are empty
    if pred_sum == 0 and gt_sum == 0:
        return 1.0

    # Calculate Dice coefficient
    dice = (2 * intersection) / (pred_sum + gt_sum)

    return dice


def find_best_gt_for_prediction(pred_mask, gt_masks):
    """
    For taping dataset with multiple GT masks per image, find the GT mask
    with the highest IoU to the prediction.
    
    Args:
        pred_mask: Predicted mask (binary, values {0, 255})
        gt_masks: List of ground truth masks for the same image
        
    Returns:
        Tuple of (best_gt_mask, best_iou_score, index_of_best_mask)
    """
    best_iou = -1
    best_gt = None
    best_idx = -1
    
    for idx, gt_mask in enumerate(gt_masks):
        iou = compute_iou(pred_mask, gt_mask)
        if iou > best_iou:
            best_iou = iou
            best_gt = gt_mask
            best_idx = idx
            
    return best_gt, best_iou, best_idx


def get_matching_files(predictions_dir, ground_truth_dir, images_dir):
    """
    Match prediction files to ground truth files and images by filename.
    
    Args:
        predictions_dir: Directory containing prediction masks
        ground_truth_dir: Directory containing ground truth masks
        images_dir: Directory containing original images
        
    Returns:
        List of tuples (prediction_path, list_of_gt_paths, image_path)
    """
    # Get all prediction files
    pred_files = [f for f in os.listdir(predictions_dir) if f.endswith('_pred.png')]
    
    # Get all ground truth files
    gt_files = [f for f in os.listdir(ground_truth_dir) if f.endswith('_mask.png')]
    
    # Get all image files
    img_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Create mapping from base name to prediction file
    pred_map = {}
    for pred_file in pred_files:
        # Extract base name (remove '_pred.png')
        base_name = pred_file.replace('_pred.png', '')
        pred_path = os.path.join(predictions_dir, pred_file)
        pred_map[base_name] = pred_path
    
    # Create mapping from base name to ground truth files (one image can have multiple GT masks)
    gt_map = {}
    for gt_file in gt_files:
        # Extract base name (remove '_mask.png')
        gt_base_name = gt_file.replace('_mask.png', '')
        
        if gt_base_name not in gt_map:
            gt_map[gt_base_name] = []
        gt_path = os.path.join(ground_truth_dir, gt_file)
        gt_map[gt_base_name].append(gt_path)
    
    # Create mapping from base name to image file
    img_map = {}
    for img_file in img_files:
        # Extract base name (remove extension)
        img_base_name = Path(img_file).stem
        img_path = os.path.join(images_dir, img_file)
        img_map[img_base_name] = img_path
    
    # Find matching triplets
    matched_triplets = []
    unmatched_items = []
    
    for base_name, pred_path in pred_map.items():
        # Find corresponding ground truth masks and image
        gt_paths = gt_map.get(base_name, [])
        img_path = img_map.get(base_name)
        
        if not gt_paths:
            unmatched_items.append(f"No ground truth masks found for prediction: {pred_path}")
        elif not img_path:
            unmatched_items.append(f"No image found for prediction: {pred_path}")
        else:
            matched_triplets.append((pred_path, gt_paths, img_path))
    
    return matched_triplets, unmatched_items


def create_visualization(image_path, gt_mask_path, pred_mask_path, output_path, prompt, iou, dice):
    """
    Create a side-by-side visualization showing original image, GT mask, and prediction.
    
    Args:
        image_path: Path to original image
        gt_mask_path: Path to ground truth mask
        pred_mask_path: Path to predicted mask
        output_path: Path to save visualization
        prompt: Text prompt used for segmentation
        iou: IoU score
        dice: Dice coefficient
    """
    # Load original image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Load GT mask
    gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Load predicted mask
    pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Create subplot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Ground truth mask
    axes[1].imshow(gt_mask, cmap='gray')
    axes[1].set_title(f"Ground Truth\n(Prompt: '{prompt}')")
    axes[1].axis('off')
    
    # Predicted mask
    axes[2].imshow(pred_mask, cmap='gray')
    axes[2].set_title(f"Prediction\n(IoU: {iou:.3f}, Dice: {dice:.3f})")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize finetuned CLIPSeg model results')
    parser.add_argument('--predictions-dir', type=str,
                        default='outputs/clipseg_finetuned_eval',
                        help='Directory containing prediction masks')
    parser.add_argument('--ground-truth-base', type=str,
                        default='data/processed',
                        help='Base directory containing ground truth masks')
    parser.add_argument('--images-base', type=str,
                        default='data/processed',
                        help='Base directory containing original images')
    parser.add_argument('--output-base', type=str,
                        default='outputs/clipseg_finetuned',
                        help='Base directory to save visualizations')
    
    args = parser.parse_args()
    
    print("Starting visualization of finetuned CLIPSeg model results...")
    print("="*80)
    
    # Define datasets with their prompts
    datasets = [
        {
            "name": "cracks",
            "prompt": "drywall crack",
            "prediction_dir": os.path.join(args.predictions_dir, "cracks"),
            "gt_dir": os.path.join(args.ground_truth_base, "cracks", "valid", "masks"),
            "image_dir": os.path.join(args.images_base, "cracks", "valid", "images")
        },
        {
            "name": "taping", 
            "prompt": "drywall joint tape",
            "prediction_dir": os.path.join(args.predictions_dir, "taping"),
            "gt_dir": os.path.join(args.ground_truth_base, "taping", "valid", "masks"),
            "image_dir": os.path.join(args.images_base, "taping", "valid", "images")
        }
    ]
    
    total_processed = 0
    
    for dataset in datasets:
        print(f"\nProcessing {dataset['name']} dataset...")
        
        prediction_dir = dataset['prediction_dir']
        gt_dir = dataset['gt_dir']
        image_dir = dataset['image_dir']
        prompt = dataset['prompt']
        
        # Check if directories exist
        if not os.path.exists(prediction_dir):
            print(f"ERROR: Predictions directory does not exist: {prediction_dir}")
            continue
        if not os.path.exists(gt_dir):
            print(f"ERROR: Ground truth directory does not exist: {gt_dir}")
            continue
        if not os.path.exists(image_dir):
            print(f"ERROR: Images directory does not exist: {image_dir}")
            continue
            
        # Create output directories
        prompt_folder_name = prompt.replace(" ", "_")  # Replace spaces with underscores
        vis_output_dir = os.path.join(args.output_base, dataset['name'], prompt_folder_name, "visualizations")
        pred_output_dir = os.path.join(args.output_base, dataset['name'], prompt_folder_name, "predictions")
        
        os.makedirs(vis_output_dir, exist_ok=True)
        os.makedirs(pred_output_dir, exist_ok=True)
        
        # Get matching files
        matched_triplets, unmatched_items = get_matching_files(
            prediction_dir, gt_dir, image_dir
        )
        
        if unmatched_items:
            print(f"WARNING: {len(unmatched_items)} issues found:")
            for item in unmatched_items[:10]:  # Show first 10 warnings
                print(f"  - {item}")
            if len(unmatched_items) > 10:
                print(f"  ... and {len(unmatched_items) - 10} more")
        
        if not matched_triplets:
            print(f"ERROR: No matching prediction-ground truth-image triplets found in {prediction_dir}")
            continue
        else:
            print(f"Found {len(matched_triplets)} matching triplets in {dataset['name']}")
        
        # Process each triplet
        for pred_path, gt_paths, img_path in matched_triplets:
            # Extract base name from prediction path
            pred_filename = os.path.basename(pred_path)
            base_name = pred_filename.replace('_pred.png', '')
            
            # Load prediction mask
            pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            
            # Handle different dataset types
            if dataset['name'] == 'taping':
                # For taping, find the GT mask with highest IoU to the prediction
                gt_masks = [cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) for gt_path in gt_paths]
                best_gt_mask, best_iou, best_idx = find_best_gt_for_prediction(pred_mask, gt_masks)
                
                # Use the best GT mask for visualization
                best_gt_path = gt_paths[best_idx]
                
                # Compute metrics with the best GT mask
                iou = compute_iou(pred_mask, best_gt_mask)
                dice = compute_dice(pred_mask, best_gt_mask)
                
                # Create visualization with best match
                vis_filename = f"{base_name}_best_match_vis.png"
                vis_path = os.path.join(vis_output_dir, vis_filename)
                create_visualization(img_path, best_gt_path, pred_path, vis_path, prompt, iou, dice)
                
                print(f"  Processed {base_name}: Best IoU={best_iou:.3f} with GT mask {os.path.basename(best_gt_path)}")
            else:
                # For cracks, there's typically one GT mask per image
                gt_path = gt_paths[0]  # Take the first (and usually only) GT mask
                
                # Load GT mask and compute metrics
                gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                iou = compute_iou(pred_mask, gt_mask)
                dice = compute_dice(pred_mask, gt_mask)
                
                # Create visualization
                vis_filename = f"{base_name}_vis.png"
                vis_path = os.path.join(vis_output_dir, vis_filename)
                create_visualization(img_path, gt_path, pred_path, vis_path, prompt, iou, dice)
                
                print(f"  Processed {base_name}: IoU={iou:.3f}, Dice={dice:.3f}")
            
            # Copy prediction to predictions folder
            pred_output_path = os.path.join(pred_output_dir, pred_filename)
            pred_img = cv2.imread(pred_path)
            cv2.imwrite(pred_output_path, pred_img)
        
        dataset_count = len(matched_triplets)
        total_processed += dataset_count
        print(f"Completed {dataset['name']} dataset. Visualizations saved to {vis_output_dir}")
    
    print(f"\nVisualization completed! Total images processed: {total_processed}")
    print("Output structure created:")
    print("outputs/clipseg_finetuned/")
    print("├── cracks/")
    print("│   └── drywall_crack/")
    print("│       └── visualizations/")
    print("└── taping/")
    print("    └── drywall_joint_tape/")
    print("        └── visualizations/")


if __name__ == "__main__":
    main()