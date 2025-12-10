python3 generate_experiments.py \
  --base_config config_base.yaml \
  --job_name smiling \
  --model_path "./vit-Smiling-model-final" \
  --output_root "smiling_analysis_results" \
  --runs 10 \
  --trajectories 50
python3 generate_experiments.py \
  --base_config config_base.yaml \
  --job_name eyeglasses \
  --model_path "./vit-Eyeglasses-model-final" \
  --output_root "eyeglasses_analysis_results" \
  --runs 100 \
  --trajectories 50
