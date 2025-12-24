import os
import yaml
import re
import numpy as np
import copy
import stat

# --- Configuration Constants ---
BASE_CONFIG_PATH = 'config_base.yaml'    
SAMPLES_DIR = 'high_confidence_samples'  
MODELS_DIR = 'saved_models'              
BOUNDS_DIR = 'global_bounds'             
CONFIG_DIR = 'configs'                   
RESULTS_DIR = 'sensitivity_results'      
RUN_SCRIPT_NAME = 'run_all_jobs.sh'      # Name of the bash script to generate

NUM_RUNS = 10                            
TRAJECTORIES_PER_RUN = 50                

def main():
    if not os.path.exists(BASE_CONFIG_PATH):
        print(f"Error: Base config file '{BASE_CONFIG_PATH}' not found.")
        return

    print(f"Loading base configuration from {BASE_CONFIG_PATH}...")
    with open(BASE_CONFIG_PATH, 'r') as f:
        base_config = yaml.safe_load(f)

    # --- 1. MEMORY SAFETY & STABILITY SETTINGS ---
    print(f"  > Enforcing Stability Batch Size = 32.")
    base_config['stability_analysis']['batch_size'] = 32

    raw_deltas = np.logspace(-4, 0, num=50)
    base_config['stability_analysis']['deltas'] = [float(d) for d in raw_deltas]
    print(f"  > Injected 50 log-spaced deltas.")

    # --- 2. SENSITIVITY SPEED SETTINGS ---
    print(f"  > Switching method to 'feature_batching' with Batch Size 2048.")
    base_config['sensitivity_analysis']['method'] = "feature_batching"
    base_config['sensitivity_analysis']['batch_size'] = 2048
    base_config['sensitivity_analysis']['n_trajectories'] = TRAJECTORIES_PER_RUN

    # --- 3. GENERATE ---
    os.makedirs(CONFIG_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    sample_files = [f for f in os.listdir(SAMPLES_DIR) if f.endswith('_high_conf.npy')]
    if not sample_files:
        print("No sample files found.")
        return

    print(f"Found {len(sample_files)} sample files. Generating configs...")

    # List to store all command lines
    all_commands = []
    
    # Add shebang and basic setup
    all_commands.append("#!/bin/bash")
    all_commands.append("# Auto-generated job script")
    all_commands.append("echo 'Starting Sensitivity Analysis batch...'")

    for sample_file in sample_files:
        cell_name = sample_file.replace('_high_conf.npy', '')
        
        # Locate Model
        model_filename = f"model_{cell_name}.pth"
        model_path = os.path.join(MODELS_DIR, model_filename)
        
        if not os.path.exists(model_path):
            alt_name = cell_name.replace(',', '_')
            alt_path = os.path.join(MODELS_DIR, f"model_{alt_name}.pth")
            
            if os.path.exists(alt_path):
                print(f"  Fixed mapping: '{cell_name}' -> '{alt_name}'")
                model_path = alt_path
            else:
                print(f"Warning: Model not found for {cell_name}. Skipping.")
                continue

        # Generate Configs & Commands
        for run_id in range(NUM_RUNS):
            run_config = copy.deepcopy(base_config)
            
            safe_job_name = cell_name.replace(',', '_')
            job_id = f"{safe_job_name}_run{run_id}"
            
            run_config['experiment'].update({
                'job_id': job_id,
                'cell_type': cell_name,
                'run_id': run_id,
                'input_data_path': os.path.join(SAMPLES_DIR, sample_file),
                'model_path': model_path,
                'output_path': os.path.join(RESULTS_DIR, f"results_{job_id}.pkl"),
                'device': run_config['experiment'].get('device', 'cuda')
            })
            
            run_config['data_constraints'].update({
                'global_lower_path': os.path.join(BOUNDS_DIR, 'global_lower.npy'),
                'global_upper_path': os.path.join(BOUNDS_DIR, 'global_upper.npy'),
                'perturb_features_path': None
            })

            # Save YAML
            yaml_filename = f"config_{job_id}.yaml"
            yaml_path = os.path.join(CONFIG_DIR, yaml_filename)
            with open(yaml_path, 'w') as f:
                yaml.dump(run_config, f, default_flow_style=False)
            
            # Add command to the list
            # We use the relative path 'configs/...' assuming user runs from root
            cmd = f"python sensx_analysis.py {os.path.join(CONFIG_DIR, yaml_filename)}"
            all_commands.append(cmd)
    
    # --- 4. WRITE BASH SCRIPT ---
    with open(RUN_SCRIPT_NAME, 'w') as f:
        f.write('\n'.join(all_commands))
    
    # Make executable
    st = os.stat(RUN_SCRIPT_NAME)
    os.chmod(RUN_SCRIPT_NAME, st.st_mode | stat.S_IEXEC)
    
    print(f"Successfully generated {len(sample_files) * NUM_RUNS} config files.")
    print(f"Generated execution script: ./{RUN_SCRIPT_NAME}")
    print(f"Run it using: ./{RUN_SCRIPT_NAME}")

if __name__ == "__main__":
    main()
