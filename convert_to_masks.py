import json
import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import shutil

def convert_coco_to_binary_masks(coco_json_path, images_base_dir, output_masks_dir, output_images_dir):
    """
    Convert COCO format annotations to binary segmentation masks
    """
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create mappings
    img_id_to_info = {img['id']: img for img in coco_data['images']}
    
    # Process each annotation
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        img_info = img_id_to_info[img_id]
        img_filename = img_info['file_name']
        
        # Find the corresponding image file in the source directories
        img_path = None
        for root, dirs, files in os.walk(images_base_dir):
            for file in files:
                if file == img_filename or file.startswith(img_filename.split('.')[0]):
                    img_path = os.path.join(root, file)
                    break
            if img_path:
                break
        
        if img_path is None:
            print(f"Warning: Could not find image {img_filename} in {images_base_dir}")
            continue
        
        # Load image to get dimensions
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not load image {img_path}")
            continue
        
        h, w = img.shape[:2]
        
        # Create or load the binary mask for this image
        mask_filename = img_filename.rsplit('.', 1)[0] + "_mask.png"
        mask_path = os.path.join(output_masks_dir, mask_filename)
        
        # Initialize mask as zeros (background)
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((h, w), dtype=np.uint8)
        
        # Handle different annotation types
        if 'segmentation' in ann and ann['segmentation']:
            segmentation = ann['segmentation']
            
            # Handle different segmentation formats
            if isinstance(segmentation[0], list):
                # Polygon format: [[x1,y1,x2,y2,...]]
                for seg in segmentation:
                    if len(seg) >= 6:  # At least 3 points (6 coordinates)
                        points = [(int(seg[i]), int(seg[i+1])) for i in range(0, len(seg), 2)]
                        points = np.array(points, dtype=np.int32)
                        
                        # Fill the polygon area with 255 (foreground)
                        cv2.fillPoly(mask, [points], 255)
            elif isinstance(segmentation, dict):
                # RLE format - decode it
                print(f"RLE format detected for annotation {ann['id']}, skipping for now")
                continue
        elif 'bbox' in ann:
            # Bounding box format: [x, y, width, height]
            x, y, width, height = ann['bbox']
            x, y, width, height = int(x), int(y), int(width), int(height)
            cv2.rectangle(mask, (x, y), (x + width, y + height), 255, thickness=cv2.FILLED)
        
        # Save the updated mask
        cv2.imwrite(mask_path, mask)
        
        # Copy the image to the output images directory if not already there
        output_img_path = os.path.join(output_images_dir, os.path.basename(img_path))
        if not os.path.exists(output_img_path):
            shutil.copy2(img_path, output_img_path)

def process_dataset(dataset_name, dataset_path, output_images_dir, output_masks_dir):
    """
    Process an entire dataset folder structure
    """
    print(f"\nProcessing {dataset_name} dataset...")
    
    # Create output directories
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_masks_dir, exist_ok=True)
    
    # Look for COCO annotation files in train, valid, test splits
    splits = ['train', 'valid', 'test']
    
    for split in splits:
        split_path = os.path.join(dataset_path, split)
        if os.path.exists(split_path):
            # Look for annotation file in this split
            annotation_file = os.path.join(split_path, '_annotations.coco.json')
            if os.path.exists(annotation_file):
                print(f"Processing {split} split...")
                convert_coco_to_binary_masks(
                    annotation_file, 
                    split_path,  # images base directory
                    output_masks_dir, 
                    output_images_dir
                )

def main():
    print("Converting COCO annotations to binary segmentation masks...")
    
    # Define dataset paths and output paths
    datasets = {
        "cracks": {
            "input_path": "data/datasets/cracks.v1i.coco",
            "output_images": "data/cracks/images",
            "output_masks": "data/cracks/masks"
        },
        "taping": {  # This is the drywall join detect dataset
            "input_path": "data/datasets/Drywall-Join-Detect.v2i.coco",
            "output_images": "data/taping/images", 
            "output_masks": "data/taping/masks"
        }
    }
    
    for dataset_name, paths in datasets.items():
        process_dataset(
            dataset_name,
            paths["input_path"],
            paths["output_images"],
            paths["output_masks"]
        )
    
    # Print final statistics
    print("\nConversion completed! Final dataset statistics:")
    
    for dataset_name, paths in datasets.items():
        img_count = len([f for f in os.listdir(paths["output_images"]) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        mask_count = len([f for f in os.listdir(paths["output_masks"]) 
                          if f.lower().endswith('_mask.png')])
        
        print(f"\n{dataset_name.upper()} DATASET:")
        print(f"- Images: {img_count}")
        print(f"- Masks: {mask_count}")
        print(f"- Images directory: {paths['output_images']}")
        print(f"- Masks directory: {paths['output_masks']}")

if __name__ == "__main__":
    main()