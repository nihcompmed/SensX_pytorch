import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from transformers import ViTForImageClassification, ViTImageProcessor
import os

class HuggingFaceModelWrapper(nn.Module):
    def __init__(self, model):
        super(HuggingFaceModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).logits

def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :]
    result = result.reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.permute(0, 3, 1, 2)
    return result

def run_gradcam(model_path, image_path, save_path="gradcam_result.png"):
    print(f"Loading model from {model_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    original_model = ViTForImageClassification.from_pretrained(model_path).to(device)
    original_model.eval()
    
    # CRITICAL: Ensure gradients are enabled for the model parameters
    for param in original_model.parameters():
        param.requires_grad = True
        
    model_wrapper = HuggingFaceModelWrapper(original_model)
    processor = ViTImageProcessor.from_pretrained(model_path)

    # TARGET LAYER: Last encoder layer's first LayerNorm
    target_layers = [original_model.vit.encoder.layer[-1].layernorm_before]

    # Prepare Image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    rgb_img = np.array(image).astype(np.float32) / 255.0
    rgb_img = cv2.resize(rgb_img, (224, 224)) 

    # --- USE GRADCAM++ ---
    cam = GradCAMPlusPlus(
        model=model_wrapper, 
        target_layers=target_layers, 
        reshape_transform=reshape_transform
    )

    grayscale_cam = cam(input_tensor=inputs.pixel_values, targets=None)
    grayscale_cam = grayscale_cam[0, :]
    
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    original_uint8 = (rgb_img * 255).astype(np.uint8)
    comparison = np.hstack((original_uint8, visualization))
    
    Image.fromarray(comparison).save(save_path)
    print(f"GradCAM++ saved to {save_path}")

if __name__ == "__main__":

    raw_input_dir = "data/"
    output_dir = 'gradcam_plus_plus/'

    for category in ['smiling', 'eyeglasses']:

        if category == 'smiling':
            model_path = "./vit-Smiling-model-final"
        elif category == 'eyeglasses':
            model_path = "./vit-Eyeglasses-model-final"

        for img_name in ['000276', '000375']:

            image_path = os.path.join(raw_input_dir, f'{img_name}.jpg')
            save_path = os.path.join(output_dir, f'{img_name}.jpg')


            run_gradcam(model_path, image_path, save_path=save_path)




