import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor
import os

raw_input_dir = "data/"

def visualize_attention(image_path, save_path):

    # Load Image
    image = Image.open(image_path).convert("RGB")
    
    # Preprocess
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    # Forward pass with output_attentions=True
    # This is critical: we must request the attention scores
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)



    # outputs.attentions is a tuple of shape (num_layers, batch_size, num_heads, seq_len, seq_len)
    # The classification token (CLS) is usually at index 0.
    # We want to see which patches the CLS token attended to.

    # Get attentions from the LAST layer (usually the most semantically relevant)
    # Shape: (1, num_heads, seq_len, seq_len)
    last_layer_attention = outputs.attentions[-1]

    # Average across all attention heads
    # Shape: (1, seq_len, seq_len)
    attentions = torch.mean(last_layer_attention, dim=1)


    # 4. Extract Attention Map for CLS Token
    # The attention matrix represents [target_token, source_token]
    # We want row 0 (CLS token) and all columns 1: (image patches)
    # Remove the first token (self-attention to CLS)
    # Shape: (seq_len - 1)
    cls_attention = attentions[0, 0, 1:]


    # 5. Reshape to Coarse Grid (e.g., 14x14)
    grid_size = int(np.sqrt(cls_attention.shape[0]))
    attention_grid = cls_attention.reshape(grid_size, grid_size).cpu().numpy()

    # 6. Resize to Image Size (NEAREST NEIGHBOR)
    # We strictly use INTER_NEAREST to preserve the blocky, coarse grid structure.
    img_np = np.array(image)
    h, w = img_np.shape[:2]

    attention_map = cv2.resize(attention_grid, (w, h), interpolation=cv2.INTER_NEAREST)

    # Normalize to [0, 1]
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())

    # 7. Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # A. Original
    ax1.imshow(image)
    ax1.set_title("Original Image")
    ax1.axis('off')

    # B. Raw Coarse Heatmap
    # We use interpolation='nearest' to ensure matplotlib doesn't smooth it either
    ax2.imshow(attention_map, cmap='jet', interpolation='nearest')
    ax2.set_title(f"Raw Attention Grid ({grid_size}x{grid_size})")
    ax2.axis('off')

    # C. Blocky Overlay
    heatmap = cv2.applyColorMap(np.uint8(255 * attention_map), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

    ax3.imshow(overlay, interpolation='nearest')
    ax3.set_title("Overlay (Nearest Neighbor)")
    ax3.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Visualization saved to {save_path}")
    plt.close()

for category in ['smiling', 'eyeglasses']:

    for img_name in ['000276', '000375']:

        #########################################################
        # 1. LOAD YOUR FINE-TUNED MODEL
        
        if category == 'smiling':
            model_path = "./vit-Smiling-model-final"
        elif category == 'eyeglasses':
            model_path = "./vit-Eyeglasses-model-final"
        
        model = ViTForImageClassification.from_pretrained(model_path, attn_implementation="eager")
        processor = ViTImageProcessor.from_pretrained(model_path)
        
        # Move model to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        # Run usage
        img_path = os.path.join(raw_input_dir, f'{img_name}.jpg')
        save_fname = f'raw_attention_maps/attention_map_{category}_{img_name}.jpg'
        visualize_attention(img_path, save_fname)
        
        
