import torch
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

# Test if the model can be loaded
try:
    print("Testing CLIPSeg model loading...")
    model = CLIPSegForImageSegmentation.from_pretrained('CIDAS/clipseg-rd64-refined')
    processor = CLIPSegProcessor.from_pretrained('CIDAS/clipseg-rd64-refined')
    print("Model loaded successfully!")
    print(f"Model: {type(model)}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
except Exception as e:
    print(f"Error loading model: {e}")