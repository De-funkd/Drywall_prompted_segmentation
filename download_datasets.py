import os
import requests
from pathlib import Path
import zipfile
import shutil
from roboflow import Roboflow

def download_with_api_key():
    """Attempt to download using Roboflow API with API key"""
    # Check if ROBOFLOW_API_KEY environment variable exists
    import os
    api_key = os.environ.get("ROBOFLOW_API_KEY")

    if api_key is None:
        print("No Roboflow API key found. Please set ROBOFLOW_API_KEY environment variable.")
        print("For public datasets, you can get a free API key from https://roboflow.com")
        return None, None

    try:
        rf = Roboflow(api_key)

        # Try to download the taping area dataset
        print("Attempting to download taping area dataset...")
        taping_project = rf.workspace("objectdetect-pu6rn").project("drywall-join-detect")
        taping_version = taping_project.versions()[-1]  # Get the latest version
        taping_dir = taping_project.version(taping_version["version"]).download("folder", local_dir="data/taping/raw")

        # Try to download the cracks dataset
        print("Attempting to download cracks dataset...")
        cracks_project = rf.workspace("fyp-ny1jt").project("cracks-3ii36")
        cracks_version = cracks_project.versions()[-1]  # Get the latest version
        cracks_dir = cracks_project.version(cracks_version["version"]).download("folder", local_dir="data/cracks/raw")

        return taping_dir.location, cracks_dir.location

    except Exception as e:
        print(f"Error downloading datasets: {str(e)}")
        return None, None

def manual_download_instructions():
    """Provide instructions for manual download"""
    print("\nManual download instructions:")
    print("1. Visit https://universe.roboflow.com/objectdetect-pu6rn/drywall-join-detect")
    print("2. Click 'Download Dataset'")
    print("3. Select format: YOLOv5 PyTorch")
    print("4. Extract to data/taping/raw/")
    print("\n5. Visit https://universe.roboflow.com/fyp-ny1jt/cracks-3ii36")
    print("6. Click 'Download Dataset'")
    print("7. Select format: YOLOv5 PyTorch")
    print("8. Extract to data/cracks/raw/")
    print("\nAfter manual download, run the annotation conversion script.")

def inspect_and_convert_annotations():
    """Function to inspect and convert annotations once downloaded"""
    import cv2
    import numpy as np
    from PIL import Image
    import json
    import yaml

    def convert_to_binary_masks(dataset_name, raw_dir, output_img_dir, output_mask_dir):
        """Convert annotations to binary masks"""
        print(f"\nProcessing {dataset_name} dataset...")

        # Create output directories if they don't exist
        os.makedirs(output_img_dir, exist_ok=True)
        os.makedirs(output_mask_dir, exist_ok=True)

        # Look for different annotation formats
        annotation_formats = []

        # Check for YOLO format (txt files)
        yolo_annotations = list(Path(raw_dir).rglob("*.txt"))
        if yolo_annotations:
            annotation_formats.append("YOLO")
            print(f"Found YOLO annotations: {len(yolo_annotations)} files")

        # Check for COCO format (json files)
        coco_annotations = list(Path(raw_dir).rglob("*.json"))
        if coco_annotations:
            annotation_formats.append("COCO")
            print(f"Found COCO annotations: {len(coco_annotations)} files")

        # Check for other formats
        other_annotations = list(Path(raw_dir).rglob("*annotation*")) + list(Path(raw_dir).rglob("*label*"))
        if other_annotations:
            print(f"Found other annotation formats: {len(other_annotations)} files")

        # Process based on format found
        if "YOLO" in annotation_formats:
            process_yolo_format(raw_dir, output_img_dir, output_mask_dir)
        elif "COCO" in annotation_formats:
            process_coco_format(raw_dir, output_img_dir, output_mask_dir)
        else:
            print(f"No recognized annotation format found in {raw_dir}")
            return 0

        # Count processed images
        img_count = len(list(Path(output_img_dir).glob("*")))
        print(f"Processed {img_count} images for {dataset_name}")
        return img_count

    def process_yolo_format(raw_dir, output_img_dir, output_mask_dir):
        """Process YOLO format annotations"""
        # Find the train/val/test splits
        splits = ['train', 'valid', 'test']
        for split in splits:
            split_dir = Path(raw_dir) / split
            if split_dir.exists():
                # Copy images
                img_dir = split_dir / "images"
                if img_dir.exists():
                    for img_file in img_dir.glob("*"):
                        shutil.copy2(img_file, output_img_dir / img_file.name)

                # Convert annotations to masks
                label_dir = split_dir / "labels"
                if label_dir.exists():
                    for txt_file in label_dir.glob("*.txt"):
                        # Get corresponding image
                        img_name = txt_file.stem + ".jpg"  # Assuming jpg, could be png too
                        img_path = Path(output_img_dir) / img_name

                        # If image doesn't exist with .jpg extension, try other common extensions
                        if not img_path.exists():
                            for ext in [".png", ".jpeg", ".JPG", ".PNG"]:
                                alt_img_path = Path(output_img_dir) / (txt_file.stem + ext)
                                if alt_img_path.exists():
                                    img_path = alt_img_path
                                    break

                        if img_path.exists():
                            # Load image to get dimensions
                            img = cv2.imread(str(img_path))
                            h, w = img.shape[:2]

                            # Create binary mask
                            mask = np.zeros((h, w), dtype=np.uint8)

                            # Read YOLO annotations
                            with open(txt_file, 'r') as f:
                                for line in f:
                                    values = line.strip().split()
                                    if len(values) >= 5:  # class_id x_center y_center width height
                                        # Convert normalized coordinates to pixel coordinates
                                        x_center = float(values[1]) * w
                                        y_center = float(values[2]) * h
                                        width = float(values[3]) * w
                                        height = float(values[4]) * h

                                        # Calculate bounding box coordinates
                                        x1 = int(x_center - width / 2)
                                        y1 = int(y_center - height / 2)
                                        x2 = int(x_center + width / 2)
                                        y2 = int(y_center + height / 2)

                                        # Draw filled rectangle on mask (value 255 for object)
                                        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=cv2.FILLED)

                            # Save mask
                            mask_path = Path(output_mask_dir) / f"{txt_file.stem}_mask.png"
                            cv2.imwrite(str(mask_path), mask)

    def process_coco_format(raw_dir, output_img_dir, output_mask_dir):
        """Process COCO format annotations"""
        # Find COCO annotation files
        coco_files = list(Path(raw_dir).rglob("*.json"))

        for coco_file in coco_files:
            with open(coco_file, 'r') as f:
                coco_data = json.load(f)

            # Create mapping from image ID to image info
            img_info = {img['id']: img for img in coco_data['images']}

            # Process each annotation
            for ann in coco_data['annotations']:
                img_id = ann['image_id']
                img_filename = img_info[img_id]['file_name']

                # Copy image if it exists in the raw directory
                img_path = Path(raw_dir) / img_filename
                if img_path.exists():
                    shutil.copy2(img_path, Path(output_img_dir) / img_filename)

                # Create binary mask
                img_width = img_info[img_id]['width']
                img_height = img_info[img_id]['height']
                mask = np.zeros((img_height, img_width), dtype=np.uint8)

                # Handle different annotation types
                if 'segmentation' in ann:
                    # Polygon segmentation
                    segmentation = ann['segmentation']
                    for seg in segmentation:
                        # Convert to integer points
                        points = [(int(seg[i]), int(seg[i+1])) for i in range(0, len(seg), 2)]
                        points = np.array(points, dtype=np.int32)

                        # Fill polygon on mask
                        cv2.fillPoly(mask, [points], 255)
                elif 'bbox' in ann:
                    # Bounding box annotation
                    x, y, width, height = ann['bbox']
                    x, y, width, height = int(x), int(y), int(width), int(height)
                    cv2.rectangle(mask, (x, y), (x + width, y + height), 255, thickness=cv2.FILLED)

                # Save mask
                mask_filename = img_filename.rsplit('.', 1)[0] + "_mask.png"
                mask_path = Path(output_mask_dir) / mask_filename
                cv2.imwrite(str(mask_path), mask)

    # Process both datasets
    taping_count = convert_to_binary_masks(
        "taping",
        "data/taping/raw",
        "data/taping/images",
        "data/taping/masks"
    )

    cracks_count = convert_to_binary_masks(
        "cracks",
        "data/cracks/raw",
        "data/cracks/images",
        "data/cracks/masks"
    )

    print(f"\nDataset Statistics:")
    print(f"Taping area dataset: {taping_count} images")
    print(f"Cracks dataset: {cracks_count} images")

    return taping_count, cracks_count

# Main execution
if __name__ == "__main__":
    print("Attempting to download datasets using Roboflow API...")
    taping_dir, cracks_dir = download_with_api_key()

    if taping_dir is None or cracks_dir is None:
        manual_download_instructions()

        # Check if raw data already exists (from manual download)
        taping_raw_exists = os.path.exists("data/taping/raw") and any(Path("data/taping/raw").iterdir())
        cracks_raw_exists = os.path.exists("data/cracks/raw") and any(Path("data/cracks/raw").iterdir())

        if taping_raw_exists or cracks_raw_exists:
            print("\nDetected manually downloaded data. Converting annotations to binary masks...")
            taping_count, cracks_count = inspect_and_convert_annotations()
        else:
            print("\nNo raw data found. Please download datasets manually and rerun this script.")
    else:
        print("Datasets downloaded successfully via API!")
        taping_count, cracks_count = inspect_and_convert_annotations()