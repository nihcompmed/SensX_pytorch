import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from model import FeedForwardNet

# Configuration
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001  # 1e-3
WEIGHT_DECAY = 0.0001  # 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_FILE = 'synthetic_data.p'

def get_dataloaders(X, y, train_idxs, initial_test_idxs):
    """
    Constructs dataloaders using specific indices:
    - Train: Uses train_idxs exactly.
    - Val/Test: Splits initial_test_idxs 50/50.
    """
    
    # 1. Create Training Set
    X_train = X[train_idxs]
    y_train = y[train_idxs]

    # 2. Split the 'test_idxs' into Validation (50%) and Final Test (50%)
    # We use y[initial_test_idxs] for stratification to ensure class balance in val/test
    val_sub_idxs, test_sub_idxs = train_test_split(
        np.arange(len(initial_test_idxs)), 
        test_size=0.5, 
        random_state=42, 
        stratify=y[initial_test_idxs]
    )
    
    # Map back to original indices
    val_idxs = initial_test_idxs[val_sub_idxs]
    test_idxs = initial_test_idxs[test_sub_idxs]

    X_val = X[val_idxs]
    y_val = y[val_idxs]
    
    X_test = X[test_idxs]
    y_test = y[test_idxs]

    # 3. Convert to Tensors
    tensor_x_train = torch.Tensor(X_train) 
    tensor_y_train = torch.Tensor(y_train).float() 
    
    tensor_x_val = torch.Tensor(X_val)
    tensor_y_val = torch.Tensor(y_val).float()

    tensor_x_test = torch.Tensor(X_test)
    tensor_y_test = torch.Tensor(y_test).float()

    # 4. Create Loaders
    train_loader = DataLoader(TensorDataset(tensor_x_train, tensor_y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(tensor_x_val, tensor_y_val), batch_size=BATCH_SIZE)
    test_loader = DataLoader(TensorDataset(tensor_x_test, tensor_y_test), batch_size=BATCH_SIZE)

    print(f"    Data Split -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    return train_loader, val_loader, test_loader, X_train.shape[1]

def evaluate(model, loader, device):
    """
    Helper function to calculate F1 score and Loss over a dataloader
    """
    model.eval()
    all_preds = []
    all_targets = []
    running_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)
            running_loss += loss.item()
            
            # Convert logits to probabilities and then to binary predictions
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    avg_loss = running_loss / len(loader)
    f1 = f1_score(all_targets, all_preds)
    return avg_loss, f1

def train_one_dataset(datatype, data_dict):
    print(f"\n--- Processing {datatype} ---")
    
    # Extract Data
    X = data_dict[datatype]['X']
    
    # Extract Indices
    train_idxs = data_dict[datatype]['train_idxs']
    test_idxs = data_dict[datatype]['test_idxs']
    
    # Handle label shape
    y_raw = data_dict[datatype]['y']
    if len(y_raw.shape) > 1 and y_raw.shape[1] > 1:
        y = np.argmax(y_raw, axis=1).reshape(-1, 1)
    else:
        y = y_raw.reshape(-1, 1)

    # Get Dataloaders using the explicit indices
    train_loader, val_loader, test_loader, input_dim = get_dataloaders(X, y, train_idxs, test_idxs)

    model = FeedForwardNet(input_dim).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()

    best_val_f1 = -1.0
    best_model_path = f"saved_models/best_model_{datatype}.pth"

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation phase (Calculate F1)
        val_loss, val_f1 = evaluate(model, val_loader, DEVICE)
        avg_train_loss = train_loss / len(train_loader)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f} | Val F1: {val_f1:.4f}")

        # Save best model based on F1 Score
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
    
    print(f"Finished training {datatype}. Best Val F1: {best_val_f1:.4f}.")
    
    # --- Final Test Set Evaluation ---
    print(f"Loading best model from {best_model_path} for testing...")
    best_model = FeedForwardNet(input_dim).to(DEVICE)
    best_model.load_state_dict(torch.load(best_model_path))
    
    test_loss, test_f1 = evaluate(best_model, test_loader, DEVICE)
    print(f"RESULTS for {datatype}: Test F1 Score: {test_f1:.4f}")

def main():
    print(f"Loading data from {DATA_FILE}...")
    with open(DATA_FILE, 'rb') as f:
        data_dict = pickle.load(f)

    for datatype in data_dict.keys():
        train_one_dataset(datatype, data_dict)

if __name__ == "__main__":
    main()
