import json
import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import shutil
from tqdm import tqdm

def convert_coco_to_binary_masks_optimized(coco_json_path, images_base_dir, output_masks_dir, output_images_dir, dataset_type="cracks"):
    """
    Optimized conversion of COCO format annotations to binary segmentation masks
    """
    print(f"Processing annotations from {coco_json_path}")
    
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create mappings
    img_id_to_info = {img['id']: img for img in coco_data['images']}
    
    # Group annotations by image_id for efficiency
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    # Track processed items
    processed_count = 0
    skipped_count = 0
    skipped_list = []
    
    # Process each unique image once with progress bar
    for img_id, annotations in tqdm(annotations_by_image.items(), desc=f"Processing {Path(coco_json_path).parent.name}"):
        img_info = img_id_to_info[img_id]
        img_filename = img_info['file_name']
        
        # Find the corresponding image file in the source directories (excluding Zone.Identifier files)
        img_path = None
        for root, dirs, files in os.walk(images_base_dir):
            # Filter out Zone.Identifier files
            image_files = [f for f in files if not f.endswith(':Zone.Identifier') and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for file in image_files:
                if file == img_filename or file.startswith(img_filename.split('.')[0]):
                    img_path = os.path.join(root, file)
                    break
            if img_path:
                break
        
        if img_path is None:
            print(f"Warning: Could not find image {img_filename} in {images_base_dir}")
            skipped_list.append((img_filename, "File not found"))
            skipped_count += 1
            continue
        
        # Load image to get dimensions
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not load image {img_path}")
            skipped_list.append((img_filename, "Could not load image"))
            skipped_count += 1
            continue
        
        h, w = img.shape[:2]
        
        # Create the binary mask for this image
        mask_filename = img_filename.rsplit('.', 1)[0] + "_mask.png"
        mask_path = os.path.join(output_masks_dir, mask_filename)
        
        # Initialize mask as zeros (background)
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Process all annotations for this image
        for ann in annotations:
            # Handle different annotation types based on dataset type
            if dataset_type == "cracks":
                # Cracks dataset has true segmentation polygons
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
            else:  # drywall dataset - bounding boxes only
                # Drywall dataset has only bounding boxes, convert to filled rectangles
                if 'bbox' in ann:
                    # Bounding box format: [x, y, width, height]
                    x, y, width, height = ann['bbox']
                    x, y, width, height = int(x), int(y), int(width), int(height)
                    cv2.rectangle(mask, (x, y), (x + width, y + height), 255, thickness=cv2.FILLED)
        
        # Save the mask
        cv2.imwrite(mask_path, mask)
        
        # Copy the image to the output images directory if not already there
        output_img_path = os.path.join(output_images_dir, os.path.basename(img_path))
        if not os.path.exists(output_img_path):
            shutil.copy2(img_path, output_img_path)
        
        processed_count += 1
    
    return processed_count, skipped_count, skipped_list

def process_dataset_optimized(dataset_name, dataset_path, output_base_dir, dataset_type="cracks"):
    """
    Process an entire dataset folder structure efficiently
    """
    print(f"\nProcessing {dataset_name} dataset...")
    
    # Look for COCO annotation files in train, valid, test splits
    splits = ['train', 'valid', 'test']
    
    total_processed = 0
    total_skipped = 0
    
    for split in splits:
        split_path = os.path.join(dataset_path, split)
        if os.path.exists(split_path):
            # Create output directories for this split
            output_split_dir = os.path.join(output_base_dir, split)
            output_images_dir = os.path.join(output_split_dir, "images")
            output_masks_dir = os.path.join(output_split_dir, "masks")
            
            os.makedirs(output_images_dir, exist_ok=True)
            os.makedirs(output_masks_dir, exist_ok=True)
            
            # Look for annotation file in this split
            annotation_file = os.path.join(split_path, '_annotations.coco.json')
            if os.path.exists(annotation_file):
                print(f"Processing {split} split...")
                processed_count, skipped_count, skipped_list = convert_coco_to_binary_masks_optimized(
                    annotation_file, 
                    split_path,  # images base directory
                    output_masks_dir, 
                    output_images_dir,
                    dataset_type
                )
                
                print(f"  - Images processed: {processed_count}")
                print(f"  - Masks generated: {processed_count}")
                print(f"  - Images skipped: {skipped_count}")
                
                if skipped_list:
                    for img_name, reason in skipped_list[:5]:  # Show first 5 skipped
                        print(f"    - {img_name}: {reason}")
                    if len(skipped_list) > 5:
                        print(f"    ... and {len(skipped_list) - 5} more")
                
                total_processed += processed_count
                total_skipped += skipped_count
            else:
                print(f"  - No annotation file found in {split_path}")
    
    return total_processed, total_skipped

def update_readme():
    """
    Append the required technical note to README.md
    """
    readme_path = "README.md"
    
    tech_note = (
        "\n\n## Internal Technical Note\n"
        "The Cracks dataset provides true polygon-based segmentation labels, allowing for precise "
        "pixel-level annotations of crack regions. The Drywall-Join-Detect dataset provides only "
        "bounding boxes, which were converted into filled rectangular masks. These bounding-box "
        "derived masks are treated as weak segmentation labels rather than precise pixel-level "
        "ground truth. This decision was made to comply with the assignment's requirement of "
        "producing segmentation masks for both prompts, ensuring consistent output format across "
        "both datasets despite differences in annotation precision.\n"
    )
    
    if os.path.exists(readme_path):
        with open(readme_path, 'a') as f:
            f.write(tech_note)
    else:
        with open(readme_path, 'w') as f:
            f.write("# Drywall Prompted Segmentation Project\n")
            f.write(tech_note)
    
    print("Updated README.md with internal technical note")

def main():
    print("Converting COCO annotations to binary segmentation masks...")
    
    # Create the processed data directory
    os.makedirs("data/processed", exist_ok=True)
    
    # Process the cracks dataset
    cracks_output_dir = "data/processed/cracks"
    cracks_input_dir = "data/datasets/cracks.v1i.coco"
    
    print(f"\nProcessing Cracks dataset...")
    cracks_processed, cracks_skipped = process_dataset_optimized(
        "cracks",
        cracks_input_dir,
        cracks_output_dir,
        "cracks"
    )
    
    # Process the drywall-join-detect dataset (taping)
    taping_output_dir = "data/processed/taping"
    taping_input_dir = "data/datasets/Drywall-Join-Detect.v2i.coco"
    
    print(f"\nProcessing Drywall-Join-Detect (taping) dataset...")
    taping_processed, taping_skipped = process_dataset_optimized(
        "drywall-join-detect",
        taping_input_dir,
        taping_output_dir,
        "drywall"  # This will trigger bbox -> mask conversion
    )
    
    # Print final statistics
    print("\n" + "="*60)
    print("CONVERSION COMPLETED! Final statistics:")
    print("="*60)
    
    print(f"\nCracks Dataset:")
    print(f"- Images processed: {cracks_processed}")
    print(f"- Images skipped: {cracks_skipped}")
    print(f"- Output directory: {cracks_output_dir}")
    
    print(f"\nTaping (Drywall-Join-Detect) Dataset:")
    print(f"- Images processed: {taping_processed}")
    print(f"- Images skipped: {taping_skipped}")
    print(f"- Output directory: {taping_output_dir}")
    
    print(f"\nTotal processed: {cracks_processed + taping_processed} images")
    print(f"Total skipped: {cracks_skipped + taping_skipped} images")
    
    # Update README with technical note
    update_readme()
    
    print("\nBinary segmentation masks have been generated successfully!")

if __name__ == "__main__":
    main()