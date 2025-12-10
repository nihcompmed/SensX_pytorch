import yaml
import os
import copy
import argparse

def main():
    parser = argparse.ArgumentParser(description="Generate config files for robustness analysis")
    parser.add_argument("--base_config", type=str, required=True, help="Path to base config (e.g., config_base.yaml)")
    parser.add_argument("--job_name", type=str, required=True, help="Unique name for this job (e.g., 'smiling', 'eyeglasses')")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model for this specific job")
    parser.add_argument("--output_root", type=str, required=True, help="Root folder for results (e.g., 'smiling_analysis_results')")
    parser.add_argument("--runs", type=int, default=40, help="Number of repetitions")
    parser.add_argument("--trajectories", type=int, default=5, help="Trajectories per run")
    args = parser.parse_args()

    # Paths
    config_out_dir = f"configs_{args.job_name}"
    os.makedirs(config_out_dir, exist_ok=True)
    
    # Load Base Config
    with open(args.base_config, 'r') as f:
        base_config = yaml.safe_load(f)
    
    script_filename = f"run_{args.job_name}_experiments.sh"
    #shell_script_lines = ["#!/bin/bash", f"echo 'Starting {args.job_name} Analysis...'"]
    shell_script_lines = []

    print(f"Generating {args.runs} configs for '{args.job_name}'...")
    print(f"  > Model: {args.model_path}")
    print(f"  > Output: {args.output_root}")

    for i in range(args.runs):
        run_config = copy.deepcopy(base_config)
        
        # 1. Update Critical Paths
        run_config['experiment']['model_path'] = args.model_path
        run_config['sensitivity_analysis']['n_trajectories'] = args.trajectories
        
        # 2. Update Output Directory
        # Structure: smiling_analysis_results/run_00/
        run_output_dir = os.path.join(args.output_root, f"run_{i:02d}")
        run_config['experiment']['output_dir'] = run_output_dir
        
        # 3. Save Config
        config_filename = f"config_{args.job_name}_run_{i:02d}.yaml"
        config_path = os.path.join(config_out_dir, config_filename)
        
        with open(config_path, 'w') as f:
            yaml.dump(run_config, f, sort_keys=False)
            
        # Add to shell script
        #shell_script_lines.append(f"echo 'Running Run {i}/{args.runs}...'")
        shell_script_lines.append(f"python3 sensx_analysis.py --config {config_path}")

    # Save Shell Script
    with open(script_filename, "w") as f:
        f.write("\n".join(shell_script_lines))
    
    print(f"Done. Execution script saved to: {script_filename}")

if __name__ == "__main__":
    main()
