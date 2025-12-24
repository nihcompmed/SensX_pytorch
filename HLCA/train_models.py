import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import anndata
import numpy as np
import scipy.sparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm
import os
import re
from model_library import BinaryClassifier

# --- Configuration ---
H5AD_PATH = '4cb45d80-499a-48ae-a056-c71ac3552c94.h5ad' 
SAVE_DIR = 'saved_models'
MIN_CELLS_PER_TYPE = 10000 
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 20
CELL_TYPE_COL = 'cell_type' 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sanitize_filename(name):
    return re.sub(r'[^\w\-_\. ]', '_', name).replace(' ', '_')

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Running on device: {DEVICE}")
    
    print(f"Loading data from {H5AD_PATH}...")
    try:
        adata = anndata.read_h5ad(H5AD_PATH)
    except NameError:
        print("Error: 'anndata' missing. Run: pip install anndata")
        return

    # 1. Ensure Sparse CSR Format
    if not scipy.sparse.issparse(adata.X):
        print("Data is dense.")
    else:
        print("Data is sparse. Converting to CSR...")
        adata.X = adata.X.tocsr()

    # 2. Filter Constant Genes
    print("Filtering constant genes...")
    if scipy.sparse.issparse(adata.X):
        mean_sq = np.array(adata.X.power(2).mean(axis=0)).flatten()
        mean = np.array(adata.X.mean(axis=0)).flatten()
        var = mean_sq - mean**2
    else:
        var = np.var(adata.X, axis=0)
    
    non_constant_mask = (var > 1e-6)
    adata = adata[:, non_constant_mask].copy()
    valid_gene_names = list(adata.var_names)
    num_genes = adata.n_vars
    print(f"Genes remaining: {num_genes}")

    # 3. Filter Cell Types
    print(f"Filtering cell types with < {MIN_CELLS_PER_TYPE} cells...")
    counts = adata.obs[CELL_TYPE_COL].value_counts()
    valid_types = counts[counts >= MIN_CELLS_PER_TYPE].index.tolist()
    
    # Subset adata to only include valid types
    adata = adata[adata.obs[CELL_TYPE_COL].isin(valid_types)].copy()
    print(f"Remaining data shape: {adata.shape}")
    print(f"Cell types to train: {valid_types}")

    # 4. GLOBAL SPLIT (Stratified by Cell Type)
    print("\nPerforming Global Stratified Split...")
    
    X_sparse = adata.X
    all_indices = np.arange(adata.n_obs)
    all_labels_strings = adata.obs[CELL_TYPE_COL].values
    
    # Split 1: Train (70%) vs Temp (30%)
    train_idx, temp_idx, train_labels_str, temp_labels_str = train_test_split(
        all_indices, all_labels_strings, 
        test_size=0.3, 
        stratify=all_labels_strings, 
        random_state=42
    )
    
    # Split 2: Val (15%) vs Test (15%)
    val_idx, test_idx, val_labels_str, test_labels_str = train_test_split(
        temp_idx, temp_labels_str, 
        test_size=0.5, 
        stratify=temp_labels_str,    
        random_state=42
    )
    
    print(f"Global Split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    
    # 5. Training Loop
    results = {}
    
    for target in valid_types:
        print(f"\n{'='*10} Processing: {target} {'='*10}")
        
        # --- NEW: SKIP LOGIC ---
        safe_name = sanitize_filename(target)
        save_path = os.path.join(SAVE_DIR, f"model_{safe_name}.pth")
        
        if os.path.exists(save_path):
            print(f"Model already exists at: {save_path}")
            print("Skipping training for this cell type.")
            continue
        # -----------------------
        
        y_train = (train_labels_str == target).astype(np.float32)
        y_val   = (val_labels_str == target).astype(np.float32)
        y_test  = (test_labels_str == target).astype(np.float32)
        
        n_pos = np.sum(y_train == 1)
        n_neg = np.sum(y_train == 0)
        
        if n_pos == 0: 
            print("Skipping: No positive samples in training set.")
            continue
            
        pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(DEVICE)
        print(f"Train Balance: {n_pos} Pos, {n_neg} Neg (Weight: {pos_weight.item():.2f})")

        train_ds = TensorDataset(torch.tensor(train_idx), torch.tensor(y_train).unsqueeze(1))
        val_ds = TensorDataset(torch.tensor(val_idx), torch.tensor(y_val).unsqueeze(1))
        test_ds = TensorDataset(torch.tensor(test_idx), torch.tensor(y_test).unsqueeze(1))
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
        
        model = ml.BinaryClassifier(num_genes).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        best_val_f1 = -1.0
        best_epoch = -1
        
        # --- TRAINING ---
        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
            
            for idx_batch, y_batch in pbar:
                y_batch = y_batch.to(DEVICE)
                
                idx_numpy = idx_batch.numpy()
                x_batch_dense = X_sparse[idx_numpy].toarray().astype(np.float32)
                x_batch_tensor = torch.tensor(x_batch_dense).to(DEVICE)

                optimizer.zero_grad()
                out = model(x_batch_tensor)
                loss = criterion(out, y_batch)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            avg_train_loss = running_loss / len(train_loader)
            
            # --- VALIDATION F1 ---
            model.eval()
            val_preds_list = []
            val_targets_list = []
            
            with torch.no_grad():
                for idx_batch, y_batch in val_loader:
                    y_batch = y_batch.to(DEVICE)
                    idx_numpy = idx_batch.numpy()
                    x_batch_dense = X_sparse[idx_numpy].toarray().astype(np.float32)
                    x_batch_tensor = torch.tensor(x_batch_dense).to(DEVICE)
                    
                    out = model(x_batch_tensor)
                    preds = (torch.sigmoid(out) > 0.5).float()
                    val_preds_list.append(preds.cpu().numpy())
                    val_targets_list.append(y_batch.cpu().numpy())

            val_preds_all = np.concatenate(val_preds_list)
            val_targets_all = np.concatenate(val_targets_list)
            val_f1 = f1_score(val_targets_all, val_preds_all, average='binary', zero_division=0)
            
            print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} | Val F1: {val_f1:.4f}")

            # Checkpoint
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch + 1
                
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'gene_mask': non_constant_mask,
                    'gene_names': valid_gene_names,
                    'cell_type': target,
                    'best_val_f1': best_val_f1
                }
                torch.save(checkpoint, save_path)
        
        print(f"Training Complete. Best Val F1: {best_val_f1:.4f} (Epoch {best_epoch})")

        # --- TEST EVALUATION ---
        if os.path.exists(save_path):
            checkpoint = torch.load(save_path, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
        
        model.eval()
        test_preds_list = []
        test_targets_list = []
        
        with torch.no_grad():
            for idx_batch, y_batch in test_loader:
                y_batch = y_batch.to(DEVICE)
                idx_numpy = idx_batch.numpy()
                x_batch_dense = X_sparse[idx_numpy].toarray().astype(np.float32)
                x_batch_tensor = torch.tensor(x_batch_dense).to(DEVICE)

                out = model(x_batch_tensor)
                preds = (torch.sigmoid(out) > 0.5).float()
                test_preds_list.append(preds.cpu().numpy())
                test_targets_list.append(y_batch.cpu().numpy())

        test_preds_all = np.concatenate(test_preds_list)
        test_targets_all = np.concatenate(test_targets_list)
        test_f1 = f1_score(test_targets_all, test_preds_all, average='binary', zero_division=0)
        
        results[target] = {'Val_F1': best_val_f1, 'Test_F1': test_f1}
        print(f"TEST RESULTS -> F1 Score: {test_f1:.4f}")

    print("\n--- Final Summary ---")
    for cell, res in results.items():
        print(f"{cell}: Val F1={res['Val_F1']:.3f}, Test F1={res['Test_F1']:.3f}")

if __name__ == "__main__":
    main()
