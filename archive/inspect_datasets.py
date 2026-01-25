import json
import os
from pathlib import Path
import cv2
import numpy as np
from collections import Counter
import random

def inspect_directory_structure(base_path, dataset_name):
    """Print the directory tree for the dataset"""
    print(f"\n=== DIRECTORY STRUCTURE: {dataset_name} ===")
    print(f"Base path: {base_path}")
    
    for root, dirs, files in os.walk(base_path):
        level = root.replace(base_path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        
        # Print only first 10 files to avoid clutter
        for file in files[:10]:
            print(f"{subindent}{file}")
        if len(files) > 10:
            print(f"{subindent}... and {len(files)-10} more files")

def find_image_dirs_and_annotations(base_path):
    """Identify image directories and annotation files"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    annotation_extensions = ['.json', '.txt', '.xml']
    
    image_dirs = []
    annotation_files = []
    
    for root, dirs, files in os.walk(base_path):
        # Check if this directory contains image files
        image_files = [f for f in files if Path(f).suffix.lower() in image_extensions]
        if image_files:
            image_dirs.append(root)
        
        # Check if this directory contains annotation files
        annot_files = [f for f in files if Path(f).suffix.lower() in annotation_extensions]
        for annot_file in annot_files:
            annotation_files.append(os.path.join(root, annot_file))
    
    return image_dirs, annotation_files

def inspect_coco_annotations(annotation_file):
    """Inspect COCO format annotations"""
    print(f"\n=== COCO ANNOTATION INSPECTION: {annotation_file} ===")
    
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)
    
    # Print basic stats
    num_images = len(coco_data.get('images', []))
    num_annotations = len(coco_data.get('annotations', []))
    categories = coco_data.get('categories', [])
    
    print(f"Number of images: {num_images}")
    print(f"Number of annotations: {num_annotations}")
    print(f"Categories: {[(cat['id'], cat['name']) for cat in categories]}")
    
    # Inspect a sample annotation
    if coco_data.get('annotations'):
        sample_ann = coco_data['annotations'][0]
        print(f"\nSample annotation (ID: {sample_ann.get('id', 'N/A')}):")
        print(f"  Image ID: {sample_ann.get('image_id', 'N/A')}")
        print(f"  Category ID: {sample_ann.get('category_id', 'N/A')}")
        
        if 'bbox' in sample_ann:
            print(f"  BBox: {sample_ann['bbox']}")
        
        if 'segmentation' in sample_ann:
            seg = sample_ann['segmentation']
            print(f"  Segmentation exists: YES")
            print(f"  Segmentation type: {'Polygon-based' if isinstance(seg, list) and len(seg) > 0 else 'Other'}")
            if isinstance(seg, list) and len(seg) > 0:
                print(f"  Segmentation sample: {seg[0][:6]}..." if len(seg[0]) > 6 else f"  Segmentation: {seg[0]}")
        else:
            print(f"  Segmentation exists: NO")
    
    return num_images, num_annotations, categories

def check_image_properties(image_paths, dataset_name, num_samples=5):
    """Check properties of random sample of images"""
    print(f"\n=== IMAGE SANITY CHECKS: {dataset_name} ===")
    
    if len(image_paths) < num_samples:
        selected_images = image_paths
    else:
        selected_images = random.sample(image_paths, num_samples)
    
    for img_path in selected_images:
        try:
            img = cv2.imread(img_path)
            if img is not None:
                h, w, channels = img.shape if len(img.shape) == 3 else (img.shape[0], img.shape[1], 1)
                print(f"  {os.path.basename(img_path)}: {w}x{h}, {channels} channels")
            else:
                print(f"  {os.path.basename(img_path)}: FAILED TO LOAD")
        except Exception as e:
            print(f"  {os.path.basename(img_path)}: ERROR - {str(e)}")

def calculate_avg_resolution(image_paths):
    """Calculate average resolution of images"""
    total_w, total_h = 0, 0
    count = 0
    
    for img_path in image_paths:
        try:
            img = cv2.imread(img_path)
            if img is not None:
                h, w = img.shape[:2]
                total_w += w
                total_h += h
                count += 1
        except:
            continue
    
    if count > 0:
        avg_w = total_w // count
        avg_h = total_h // count
        return f"{avg_w}x{avg_h}"
    else:
        return "N/A"

def get_all_image_paths(base_path):
    """Get all image paths in the dataset"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.JPG', '.JPEG', '.PNG'}
    image_paths = []
    
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if Path(file).suffix in image_extensions:
                image_paths.append(os.path.join(root, file))
    
    return image_paths

def main():
    print("Roboflow Dataset Inspector")
    print("="*50)
    
    # Define dataset paths
    datasets = {
        "cracks": "data/datasets/cracks.v1i.coco",
        "drywall_join_detect": "data/datasets/Drywall-Join-Detect.v2i.coco"
    }
    
    dataset_summaries = {}
    
    for dataset_name, base_path in datasets.items():
        if not os.path.exists(base_path):
            print(f"Dataset {dataset_name} not found at {base_path}")
            continue
        
        # 1. Directory inspection
        inspect_directory_structure(base_path, dataset_name)
        
        # Find image directories and annotation files
        image_dirs, annotation_files = find_image_dirs_and_annotations(base_path)
        print(f"\nImage directories found: {image_dirs}")
        print(f"Annotation files found: {annotation_files}")
        
        # 2. COCO annotation inspection
        num_images_total = 0
        num_annotations_total = 0
        all_categories = []
        
        for annot_file in annotation_files:
            if '_annotations.coco.json' in annot_file:
                num_imgs, num_anns, cats = inspect_coco_annotations(annot_file)
                num_images_total += num_imgs
                num_annotations_total += num_anns
                all_categories.extend(cats)
        
        # 3. Image sanity checks
        all_image_paths = get_all_image_paths(base_path)
        check_image_properties(all_image_paths, dataset_name)
        
        # Calculate average resolution
        avg_res = calculate_avg_resolution(all_image_paths)
        
        # Check if segmentation exists in any annotation
        has_segmentation = False
        for annot_file in annotation_files:
            if '_annotations.coco.json' in annot_file:
                with open(annot_file, 'r') as f:
                    coco_data = json.load(f)
                    for ann in coco_data.get('annotations', []):
                        if 'segmentation' in ann and ann['segmentation']:
                            has_segmentation = True
                            break
                    if has_segmentation:
                        break
        
        # Store summary
        dataset_summaries[dataset_name] = {
            'images': num_images_total,
            'annotations': num_annotations_total,
            'classes': list(set([cat['name'] for cat in all_categories])),
            'segmentation_available': "YES" if has_segmentation else "NO",
            'avg_resolution': avg_res
        }
    
    # 4. Dataset summary
    print("\n" + "="*50)
    print("DATASET SUMMARY")
    print("="*50)
    
    for dataset_name, summary in dataset_summaries.items():
        print(f"DATASET: {dataset_name}")
        print(f"- Images: {summary['images']}")
        print(f"- Annotations: {summary['annotations']}")
        print(f"- Classes: {summary['classes']}")
        print(f"- Segmentation available: {summary['segmentation_available']}")
        print(f"- Avg resolution: {summary['avg_resolution']}")
        print()

if __name__ == "__main__":
    main()