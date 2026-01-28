#!/usr/bin/env python3
"""
Inference script to generate predictions on validation set for CLIPSeg finetuned model.

This script:
- Loads the fine-tuned checkpoint from outputs/clipseg_finetuned/best_model.pth
- Iterates only over validation images
- Uses the correct prompt per dataset: cracks → "drywall crack", taping → "drywall joint tape"
- Saves predictions to outputs/clipseg_finetuned_eval/{cracks,taping}/
- Uses EXACT base filenames: image: XYZ.jpg, GT: XYZ_mask.png, pred: XYZ_pred.png
- Applies sigmoid + threshold 0.5
- Saves binary masks with values {0,255}
- Logs number of images processed and output directory
"""

import os
import numpy as np
import torch
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import cv2
from pathlib import Path


def run_validation_inference():
    """Run inference on validation sets using the fine-tuned model."""
    
    # Load the fine-tuned model
    print("Loading fine-tuned CLIPSeg model...")
    model = CLIPSegForImageSegmentation.from_pretrained('CIDAS/clipseg-rd64-refined')
    
    # Load the checkpoint
    checkpoint_path = "outputs/clipseg_finetuned/best_model.pth"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load processor
    processor = CLIPSegProcessor.from_pretrained('CIDAS/clipseg-rd64-refined')
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Define output directories
    output_base = "outputs/clipseg_finetuned_eval"
    os.makedirs(os.path.join(output_base, "cracks"), exist_ok=True)
    os.makedirs(os.path.join(output_base, "taping"), exist_ok=True)
    
    # Define validation datasets
    datasets = [
        {
            "name": "cracks",
            "image_dir": "data/processed/cracks/valid/images",  # Note: directory name might be "cracks"
            "mask_dir": "data/processed/cracks/valid/masks",
            "prompt": "drywall crack"
        },
        {
            "name": "taping", 
            "image_dir": "data/processed/taping/valid/images",
            "mask_dir": "data/processed/taping/valid/masks",
            "prompt": "drywall joint tape"
        }
    ]
    
    total_processed = 0
    
    for dataset in datasets:
        print(f"\nProcessing {dataset['name']} validation set...")
        
        image_dir = Path(dataset['image_dir'])
        output_dir = Path(output_base) / dataset['name']
        prompt = dataset['prompt']
        
        # Get all image files
        image_files = [f for f in os.listdir(image_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"Found {len(image_files)} images to process")
        
        for i, img_file in enumerate(image_files):
            # Extract base filename without extension
            base_name = Path(img_file).stem
            
            # Load image
            img_path = image_dir / img_file
            image = Image.open(img_path).convert('RGB')
            
            # Process with CLIPSeg
            inputs = processor(text=[prompt], images=[image], padding=True, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            
            with torch.no_grad():
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                
                # Apply sigmoid to get probabilities
                probs = torch.sigmoid(logits)
                
                # Resize to match original image size
                original_size = (image.height, image.width)
                if probs.shape[-2:] != original_size:
                    probs = torch.nn.functional.interpolate(
                        probs.unsqueeze(0),
                        size=original_size,
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)
                
                # Apply threshold of 0.5
                pred_mask = (probs > 0.5).float().cpu().numpy()
                
                # Convert to binary mask with values {0, 255}
                pred_mask_uint8 = (pred_mask * 255).astype(np.uint8)

                # Handle different tensor shapes - squeeze if needed
                if len(pred_mask_uint8.shape) == 3:
                    pred_mask_uint8 = pred_mask_uint8.squeeze(0)  # Remove channel dimension if present
                
                # Save prediction with exact naming convention
                pred_filename = f"{base_name}_pred.png"
                pred_path = output_dir / pred_filename
                cv2.imwrite(str(pred_path), pred_mask_uint8)
                
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(image_files)} images for {dataset['name']}")
        
        total_processed += len(image_files)
        print(f"Completed {dataset['name']} dataset. Predictions saved to {output_dir}")
    
    print(f"\nInference completed! Total images processed: {total_processed}")
    print(f"All predictions saved to {output_base}")
    

if __name__ == "__main__":
    run_validation_inference()