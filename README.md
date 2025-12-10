Scalable model-agnostic local XAI based on global sensitivity analysis

ViT:

1. download.py to download pretrained ViT model from HuggingFace.
2. finetune_vit.sh to train two binary classifiers, one to identify Smiling faces and other to identify Eyeglasses. Make sure IMAGES_DIR is the location of the CelebA aligned images.
3. Run sh bash_final_script.sh to generate swarm job bash files run_smiling_experiments.sh and run_eyeglasses_experiments.sh.
4. We ran specifically, swarm -f run_smiling_experiments.sh -g 10 -t 12 --time=6:00:00 --gres=gpu:a100:1 --partition=gpu and swarm -f run_eyeglasses_experiments.sh -g 10 -t 12 --time=6:00:00 --gres=gpu:a100:1 --partition=gpu
5. config_base.yaml has the hyperparameters.

Single-cell data (HLCA):

1. 


