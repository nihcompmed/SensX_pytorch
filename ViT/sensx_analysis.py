import yaml
import torch
import os
import argparse
from PIL import Image
from define_qoi import initialize_model_and_qoi
import sys
sys.path.append('../')
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

    # 5. Run Loop
    input_dir = config['experiment']['input_dir']
    output_dir = config['experiment']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    print(f"Starting Analysis on {len(image_files)} images...")

    for img_name in image_files:
        print(f"\nProcessing {img_name}...")
        try:
            img_path = os.path.join(input_dir, img_name)
            raw_image = Image.open(img_path).convert("RGB")
            x_input = transform(raw_image).to(device)

            save_path = os.path.join(output_dir, f"result_{img_name}.pkl")
            
            analyzer.run_full_analysis(
                x=x_input, 
                config=config, 
                save_path=save_path
            )
        except Exception as e:
            print(f"Failed to process {img_name}: {e}")

    print("\nJob Complete.")

if __name__ == "__main__":
    main()
