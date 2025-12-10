import torch
import torch.nn as nn
import os
import pickle
import argparse
import numpy as np
from transformers import ViTImageProcessor, ViTForImageClassification, TrainingArguments, Trainer
from datasets import Dataset, Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# --- 0. ARGUMENT PARSING ---
parser = argparse.ArgumentParser(description="Train ViT on a specific attribute with weighted loss")
parser.add_argument("--attribute", type=str, required=True, help="The dictionary key to predict (e.g., 'Smiling', 'Eyeglasses')")
parser.add_argument("--pickle_path", type=str, required=True, help="Path to the .pkl file containing labels")
parser.add_argument("--images_dir", type=str, required=True, help="Path to the folder containing images")
parser.add_argument("--base_model", type=str, default="./pretrained_vit_base", help="Path to the downloaded ViT model")
parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size per device")
args = parser.parse_args()

print(f"--- Starting Training for Attribute: {args.attribute} ---")

# --- 1. LOAD DATA ---
print(f"Loading labels from {args.pickle_path}...")
with open(args.pickle_path, 'rb') as f:
    label_dict = pickle.load(f)

# --- 2. PREPARE DATASET ---
def create_dataset_from_dict(image_dir, full_dict, target_attribute):
    image_paths = []
    labels = []
    missing_count = 0
    
    # Validation: Check if attribute exists in the first entry
    first_key = next(iter(full_dict))
    if target_attribute not in full_dict[first_key]:
        raise ValueError(f"Attribute '{target_attribute}' not found in dictionary keys.")

    for filename, attributes in full_dict.items():
        full_path = os.path.join(image_dir, filename)
        
        # Only add if file exists to avoid errors during training
        if os.path.exists(full_path):
            image_paths.append(full_path)
            labels.append(attributes[target_attribute])
        else:
            missing_count += 1

    if missing_count > 0:
        print(f"Warning: {missing_count} images were missing from disk and skipped.")

    dataset = Dataset.from_dict({"image": image_paths, "label": labels})
    dataset = dataset.cast_column("image", Image())
    return dataset

full_ds = create_dataset_from_dict(args.images_dir, label_dict, args.attribute)

# Split 85% Train / 15% Test
ds = full_ds.train_test_split(test_size=0.15, seed=42)

# --- 3. CALCULATE CLASS WEIGHTS (For Imbalance) ---
# Count labels in training set
train_labels = np.array(ds["train"]['label'])
class_counts = np.bincount(train_labels)
total_samples = len(train_labels)

print(f"Class distribution: Class 0 (Not {args.attribute}): {class_counts[0]}, Class 1 ({args.attribute}): {class_counts[1]}")

# Calculate weights: Total / (Num_Classes * Count)
# This ensures each class contributes equally to the loss
weights = total_samples / (2.0 * class_counts)

# Move to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_weights = torch.tensor(weights, dtype=torch.float).to(device)

print(f"Computed Class Weights: {class_weights}")

# --- 4. TRANSFORM & METRICS ---
processor = ViTImageProcessor.from_pretrained(args.base_model)

def transform(example_batch):
    inputs = processor([x.convert("RGB") for x in example_batch['image']], return_tensors='pt')
    inputs['labels'] = example_batch['label']
    return inputs

prepared_ds = ds.with_transform(transform)

def compute_metrics(p):
    predictions, labels = p
    preds = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# --- 5. MODEL SETUP ---
output_model_dir = f"./vit-{args.attribute}-model"
id2label = {"0": f"Not {args.attribute}", "1": args.attribute}
label2id = {f"Not {args.attribute}": "0", args.attribute: "1"}

model = ViTForImageClassification.from_pretrained(
    args.base_model,
    num_labels=2,
    id2label=id2label,
    label2id=label2id
)
model.to(device)

# --- 6. CUSTOM TRAINER FOR WEIGHTED LOSS ---
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Weighted Cross Entropy
        loss_fct = nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

# --- 7. TRAINING ARGUMENTS ---
training_args = TrainingArguments(
    output_dir=output_model_dir,
    per_device_train_batch_size=args.batch_size,
    eval_strategy="steps",
    num_train_epochs=args.epochs,
    fp16=torch.cuda.is_available(),
    save_steps=1000,
    eval_steps=1000,
    logging_steps=100,
    learning_rate=2e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to='none',
    load_best_model_at_end=True,
    metric_for_best_model="f1" # Save model based on F1 score, not just loss
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    data_collator=lambda x: {
        'pixel_values': torch.stack([i['pixel_values'] for i in x]), # <--- FIXED
        'labels': torch.tensor([i['labels'] for i in x])
    },
    train_dataset=prepared_ds["train"],
    eval_dataset=prepared_ds["test"],
    compute_metrics=compute_metrics,
)

# --- 8. RUN TRAINING ---
trainer.train()

# --- 9. SAVE FINAL MODEL ---
final_save_path = f"{output_model_dir}-final"
trainer.save_model(final_save_path)
processor.save_pretrained(final_save_path)

print(f"\nTraining Complete. Best model saved to: {final_save_path}")
