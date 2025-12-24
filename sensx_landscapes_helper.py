import torch
from tqdm import tqdm
import numpy as np
import math
from torch.distributions.uniform import Uniform
import os

def landscapes(qoi_function, model, ranks, model_input, 
               global_lower, global_upper, 
               save_path=None,  
               delta_arr=None, rank_arr=None, 
               n_samples=1000, ptiles=None, 
               batch_size=128, device=None):

    if save_path is None:
        print(f'Save path not given! Exiting.')
        exit()

    # --- 1. Setup & Defaults ---
    if delta_arr is None:
        step = 0.05
        delta_arr = np.arange(step, 1, step)

    if ptiles is None:
        ptiles = np.array([1, 50, 99])
    
    # Move static tensors to device
    global_lower = torch.as_tensor(global_lower, device=device)
    global_upper = torch.as_tensor(global_upper, device=device)
    model_input = model_input.to(device)
    
    ranks_tensor = torch.as_tensor(ranks, device=device)
    max_rank = torch.max(ranks_tensor).item()

    if rank_arr is None:
        rank_arr = np.unique(np.logspace(0, math.log10(max_rank), num=25).astype(int))

    global_range = global_upper - global_lower

    # Pre-allocate output arrays to avoid growing lists (Faster + Less Memory)
    # Shape: (n_deltas, n_ranks, n_ptiles)
    # We use float16 immediately to save RAM during the loops
    pos_perts = np.zeros((len(delta_arr), len(rank_arr), len(ptiles)), dtype=np.float16)
    neg_perts = np.zeros((len(delta_arr), len(rank_arr), len(ptiles)), dtype=np.float16)

    # --- 2. Main Loop ---
    # Enumerate helps us index into the pre-allocated arrays
    for d_idx, delta in enumerate(delta_arr):
        delta_range = delta * global_range 
        
        local_lower = torch.max(global_lower, model_input - delta_range)
        local_upper = torch.min(global_upper, model_input + delta_range)
        dist = Uniform(low=local_lower, high=local_upper)

        with torch.no_grad():
            for r_idx, pert_ranks in enumerate(rank_arr):
                
                top_mask = ranks_tensor <= pert_ranks
                bottom_mask = ranks_tensor > (max_rank - pert_ranks)

                batch_results_top = []
                batch_results_bot = []

                # --- 3. Batching + Sampling ---
                for i in range(0, n_samples, batch_size):
                    curr_batch_size = min(batch_size, n_samples - i)
                    
                    # Fresh samples per batch/rank
                    batch_samples = dist.sample((curr_batch_size,))

                    # Apply masks
                    batch_top = torch.where(top_mask, batch_samples, model_input)
                    batch_bot = torch.where(bottom_mask, batch_samples, model_input)

                    out_top = qoi_function(batch_top, model)
                    out_bot = qoi_function(batch_bot, model)

                    # Store as float32 temporarily for accuracy, convert later
                    batch_results_top.append(out_top.cpu().numpy())
                    batch_results_bot.append(out_bot.cpu().numpy())

                full_top = np.concatenate(batch_results_top, axis=0)
                full_bot = np.concatenate(batch_results_bot, axis=0)


                # --- 4. Store directly into pre-allocated float16 array ---
                # This keeps memory footprint flat (doesn't grow with loop)
                pos_perts[d_idx, r_idx, :] = np.percentile(full_top, ptiles, axis=0).astype(np.float16)
                neg_perts[d_idx, r_idx, :] = np.percentile(full_bot, ptiles, axis=0).astype(np.float16)

    # --- 5. Save Compressed ---
    #print(f"Saving compressed results to {save_path}...")
    np.savez_compressed(
        save_path, 
        pos_perts=pos_perts, 
        neg_perts=neg_perts,
        delta_arr=delta_arr,
        rank_arr=rank_arr,
        ptiles=ptiles
    )

    #print("Done.")
    return save_path
