from transformers import ViTForImageClassification, ViTImageProcessor

# The model repository name on Hugging Face
repo_name = 'google/vit-base-patch16-224-in21k'

# The local folder where you want to save it
local_model_path = "./pretrained_vit_base" 

print(f"Downloading model from {repo_name}...")

# 1. Download the Feature Extractor (Processor)
processor = ViTImageProcessor.from_pretrained(repo_name)
processor.save_pretrained(local_model_path)

# 2. Download the Model Weights
# We just load the base model here to save the raw weights
model = ViTForImageClassification.from_pretrained(repo_name)
model.save_pretrained(local_model_path)

print(f"Model successfully saved to: {local_model_path}")
