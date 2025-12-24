import torch
import torch.nn as nn
import numpy as np
import argparse
import yaml
import os
import model_library as ml

# --- ENABLE A100 TENSOR CORES (TF32) ---
# This gives a massive speedup on A100 GPUs for Float32 operations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Import your engine
import sys
sys.path.append('../') # Uncomment if needed
from sensx_pytorch_32bit import SensitivityAnalyzer


def main():
    # 1. Parse Arguments
    parser = argparse.ArgumentParser(description="Run Sensitivity Analysis Worker")
    parser.add_argument("config_path", type=str, help="Path to the YAML config file")
    args = parser.parse_args()

    # 2. Load Config
    if not os.path.exists(args.config_path):
        print(f"Error: Config file not found at {args.config_path}")
        sys.exit(1)

    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    exp_cfg = config['experiment']
    device = torch.device(exp_cfg['device'] if torch.cuda.is_available() else "cpu")
    print(f"Starting Job: {exp_cfg['job_id']} on {device}")

    # Check if output exists to avoid re-running
    output_path = exp_cfg['output_path']
    if os.path.exists(output_path):
        print(f"Output file {output_path} already exists. Skipping.")
        return

    # 3. Load Data
    data_path = exp_cfg['input_data_path']
    print(f"Loading data from {data_path}...")
    X_input = np.load(data_path)
    X_tensor = torch.from_numpy(X_input).to(dtype=torch.float32, device=device)
    num_genes = X_tensor.shape[1]

    # 4. Load Model
    model_path = exp_cfg['model_path']
    print(f"Loading model from {model_path}...")

    # Initialize model structure
    model = ml.BinaryClassifier(num_genes).to(device)

    # Load weights (weights_only=False required for your checkpoint format)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 5. Initialize Engine
    analyzer = SensitivityAnalyzer(
        model=model,
        qoi_func=ml.probability_qoi,
        global_lower_path=config['data_constraints']['global_lower_path'],
        global_upper_path=config['data_constraints']['global_upper_path'],
        perturb_features_path=config['data_constraints']['perturb_features_path'],
        device=device
    )

    # 6. Run Analysis
    print("Running analysis...")
    analyzer.run_full_analysis(
        x=X_tensor,
        config=config,
        save_path=output_path
    )

    print(f"Job {exp_cfg['job_id']} Completed.")

if __name__ == "__main__":
    main()






