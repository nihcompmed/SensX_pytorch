import torch
from transformers import ViTForImageClassification, ViTImageProcessor
import torchvision.transforms as T

# 1. Define the QOI Wrapper Class
class ViTQOI:
    def __init__(self, model, processor, device):
        self.model = model
        # Pre-calculate normalization constants on GPU
        # These are usually standard ImageNet Mean/Std
        self.mean = torch.tensor(processor.image_mean).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor(processor.image_std).view(1, 3, 1, 1).to(device)
    
    def __call__(self, raw_inputs, model):
        """
        Wrapper that handles Preprocessing -> Inference -> QOI Extraction
        Args:
            raw_inputs: Batch of raw images (0-1 range). Shape (B, 3, H, W)
            model: The ViT model
        Returns:
            Probability of 'Smiling' class (Tensor of shape [B])
        """
        # Normalize (x - mean) / std
        normalized_inputs = (raw_inputs - self.mean) / self.std
        
        # Inference
        outputs = model(normalized_inputs)
        
        # Extract Probability of Class 1 (Smiling)
        probs = torch.softmax(outputs.logits, dim=1)
        return probs[:, 1] 

# 2. Function to Initialize Model & QOI
def initialize_model_and_qoi(model_path, device):
    print(f"Loading ViT from {model_path}...")
    
    # Load Model
    model = ViTForImageClassification.from_pretrained(model_path).to(device)
    model.eval()
    
    # Load Processor (for mean/std stats and size)
    processor = ViTImageProcessor.from_pretrained(model_path)
    
    # Create QOI Instance
    qoi_func = ViTQOI(model, processor, device)
    
    # Define Input Preprocessing (Raw Resize -> Tensor)
    # This transforms the raw PIL image to the 0-1 tensor the analysis expects
    height, width = processor.size['height'], processor.size['width']
    preprocessing_transform = T.Compose([
        T.Resize((height, width)),
        T.ToTensor()
    ])
    
    return model, qoi_func, preprocessing_transform, (height, width)
