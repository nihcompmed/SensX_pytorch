#!/bin/bash

# Define common paths
PICKLE_FILE="./CelebA_img_labels.p"
IMAGES_DIR="/data/aggarwalm4/ViT_Jax/CelebA/img_align_celeba"
BASE_MODEL="./pretrained_vit_base"

## Activate environment
#source activate sensx_env 
## OR depending on your cluster setup: 
## conda activate sensx_env

echo "=========================================="
echo "STARTING JOB 1: SMILING"
echo "=========================================="

python train_vit.py \
    --attribute "Smiling" \
    --pickle_path "$PICKLE_FILE" \
    --images_dir "$IMAGES_DIR" \
    --base_model "$BASE_MODEL"

echo "=========================================="
echo "STARTING JOB 2: EYEGLASSES"
echo "=========================================="

python train_vit.py \
    --attribute "Eyeglasses" \
    --pickle_path "$PICKLE_FILE" \
    --images_dir "$IMAGES_DIR" \
    --base_model "$BASE_MODEL"

echo "All jobs complete."
