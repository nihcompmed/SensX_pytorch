Scalable model-agnostic local XAI based on global sensitivity analysis

Synthetic:

1. Run make_data_main.py to generate synthetic data for all 4 models (orange_skin, XOR, nonlinear additive, switch).
2. Run train_models.py to train FFN binary classifiers for synthetic data.
3. 

ViT:

1. download.py to download pretrained ViT model from HuggingFace.
2. finetune_vit.sh to train two binary classifiers, one to identify Smiling faces and other to identify Eyeglasses. Make sure IMAGES_DIR is the location of the CelebA aligned images.
3. Edit config_base.yaml to change SensX hyperparameters, as required.
4. Run sh bash_final_script.sh to generate swarm job bash files run_smiling_experiments.sh and run_eyeglasses_experiments.sh.
5. We specifically ran, swarm -f run_smiling_experiments.sh -g 10 -t 12 --time=10:00:00 --gres=gpu:a100:1 --partition=gpu and swarm -f run_eyeglasses_experiments.sh -g 10 -t 12 --time=10:00:00 --gres=gpu:a100:1 --partition=gpu.
6. Run plot_sensx_saliency_map.py to plot saliency/heatmapts of SensX values.
7. Run plot_sensx_rank_masks.py to plot masks of top SensX features.
8. Run visualize_attention.py to plot raw attention maps.
9. Run visualize_gradcam_plusplus.py to plot saliency maps from GradCam++.
10. Run pert_landscapes_sensx_ranks.py to get SensX landscapes.
11. 

Single-cell data from human lung cell atlas (HLCA):

1. Download HLCA core atlas (4cb45d80-499a-48ae-a056-c71ac3552c94.h5ad) from https://data.humancellatlas.org/hca-bio-networks/lung/atlases/lung-v1-0.
2. Run train.py to train the FFN models for each cell type. The training script will require around 55 GB memory.
3. Run get_global_domain.py to get the global domain for non-constant genes.
4. Run shortlist_cells.py to shortlist cells for local XAI.
5. Run generate_configs.py to generate config files for SensX analysis and bash script run_all_jobs.sh to process all config files.
6. We ran swarm -f run_all_jobs.sh -g 20 -t 12 -b 10 --time=4:00:00 --gres=gpu:a100:1 --partition=gpu
8. Run aggregate_sensitivity_results.py to aggregate SensX results.
9. Run python3 umap_sensx_vectors.py to get the UMAP based on our custom ranking distance.
10. Run deep_shap.py to get DeepSHAP results of the FFN models.
8. We specifically ran swarm -f bash_sensx_landscapes.sh -g 20 -t 12 --time=10:00:00 --gres=gpu:a100:1 --partition=gpu to get the landscape.
9. Run pert_landscapes_sensx_ranks.py and pert_landscapes_shap_ranks.py to get perturbation landscapes.
10. Run plot_pert_landscapes.py to plot example perturbation landscapes comparing SensX with deepSHAP.
11. 

