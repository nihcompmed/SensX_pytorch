import yaml
import torch
import os
import argparse
import pickle
from PIL import Image
from define_qoi import initialize_model_and_qoi
import sys
sys.path.append('../') # Uncomment if needed
from sensx_pytorch_32bit import SensitivityAnalyzer

def main():
    # --- ARGUMENT PARSING ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()

    # 1. Load Config
    print(f"Loading configuration from: {args.config}")
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # 2. Setup Device
    device_name = config['experiment']['device']
    if device_name == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Using CPU.")
        device_name = "cpu"
    device = torch.device(device_name)
    
    # 3. Initialize Model & QOI
    model, qoi_func, transform, img_size = initialize_model_and_qoi(
        config['experiment']['model_path'], 
        device
    )

    # 4. Initialize Engine
    constraints = config['data_constraints']
    
    analyzer = SensitivityAnalyzer(
        model=model,
        qoi_func=qoi_func,
        global_lower_path=constraints['global_lower_path'],
        global_upper_path=constraints['global_upper_path'],
        perturb_features_path=constraints.get('perturb_features_path'),
        device=device
    )

    # 5. Load ALL Images
    input_dir = config['experiment']['input_dir']
    output_dir = config['experiment']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    total_files = len(image_files)
    
    if total_files == 0:
        print("No images found in input directory.")
        return

    print(f"Loading ALL {total_files} images into memory...")

    batch_tensors = []
    valid_filenames = []

    for img_name in image_files:
        img_path = os.path.join(input_dir, img_name)
        # No try-except: Fail fast if image is corrupt
        raw_image = Image.open(img_path).convert("RGB")
        t_img = transform(raw_image) # (C, H, W)
        batch_tensors.append(t_img)
        valid_filenames.append(img_name)

    # 6. Stack into One Giant Tensor
    # Shape: (Total_Files, C, H, W)
    x_batch = torch.stack(batch_tensors).to(device)
    
    print(f"Input Tensor Shape: {x_batch.shape}")
    print("Starting Full Analysis...")

    # 7. Run Analysis
    results = analyzer.run_full_analysis(
        x=x_batch, 
        config=config, 
        save_path=None 
    )

    if results is None:
        print("Analysis failed (Optimal Delta not found).")
        return

    # 8. Save Results (Named after Config)
    # Inject filenames for mapping back to inputs
    results['filenames'] = valid_filenames
    
    # Logic: /path/to/exp_vit_patch16.yaml -> results_exp_vit_patch16.pkl
    config_basename = os.path.splitext(os.path.basename(args.config))[0]
    save_name = f"results_{config_basename}.pkl"
    save_path = os.path.join(output_dir, save_name)
    
    print(f"Saving results to {save_path}...")
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)

    print("Job Complete.")

if __name__ == "__main__":
    main()
