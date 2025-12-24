import torch
import numpy as np
import pickle
import os
from tqdm import tqdm

## --- ENABLE A100 TENSOR CORES (TF32) ---
#torch.backends.cuda.matmul.allow_tf32 = True
#torch.backends.cudnn.allow_tf32 = True

class SensitivityAnalyzer:
    def __init__(self, model, qoi_func, global_lower_path, global_upper_path, perturb_features_path=None, device=None):
        """
        Generic Sensitivity Analysis Engine (PyTorch).
        Supports Arbitrary Input Shapes (N, *S).
        
        OPTIMIZATION: Mixed Precision
        - Model & Data: Float32 (Speed on A100)
        - Sensitivity Math: Float64 (Precision for dy/dx and accumulation)
        """
        self.model = model
        self.qoi_func = qoi_func
        self.device = device if device else torch.device("cpu")
        
        # --- LOAD CONSTRAINTS ---
        print(f"  [Engine] Loading constraints...")
        
        if not os.path.exists(global_lower_path):
            raise FileNotFoundError(f"Global Lower bounds file not found: {global_lower_path}")
        lower_np = np.load(global_lower_path)
        # Flatten constraints immediately to D
        self.global_lower = torch.from_numpy(lower_np).flatten().to(dtype=torch.float32, device=self.device)
        
        if not os.path.exists(global_upper_path):
            raise FileNotFoundError(f"Global Upper bounds file not found: {global_upper_path}")
        upper_np = np.load(global_upper_path)
        self.global_upper = torch.from_numpy(upper_np).flatten().to(dtype=torch.float32, device=self.device)
        
        self.global_range = self.global_upper - self.global_lower

        self.perturb_features = None
        if perturb_features_path:
            if os.path.exists(perturb_features_path):
                print(f"  [Engine] Loading perturbation mask from {perturb_features_path}")
                self.perturb_features = np.load(perturb_features_path).flatten()
            else:
                print(f"  [Engine] Warning: perturb path provided but file not found. Using ALL features.")

    # ==========================================
    # STEP 1: STABILITY PROFILE (Float32)
    # ==========================================
    def compute_stability_profile(self, x, deltas, n_s, eval_batch_size):
        # Step 1 stays in Float32 because it's just a rough scan for stability.
        x = x.to(dtype=torch.float32, device=self.device)
        N = x.shape[0]
        original_shape = x.shape[1:] 
        D = x[0].numel()             
        
        x_flat = x.reshape(N, D)
        profile = {}
        print(f"  [Step 1] Computing Stability Profile (N={N}, Shape={original_shape})...")
        
        for delta in tqdm(deltas, desc="  Scanning Deltas", leave=True):
            delta_range = delta * self.global_range 
            
            local_lower = torch.max(self.global_lower, x_flat - delta_range)
            local_upper = torch.min(self.global_upper, x_flat + delta_range)
            
            qoi_accumulator = []
            num_batches = int(np.ceil(n_s / eval_batch_size))
            
            for b in range(num_batches):
                current_batch_size = min(eval_batch_size, n_s - b * eval_batch_size)
                
                rand_noise = torch.rand((current_batch_size, N, D), dtype=torch.float32, device=self.device)
                batch_samples_flat = local_lower.unsqueeze(0) + rand_noise * (local_upper.unsqueeze(0) - local_lower.unsqueeze(0))
                model_input = batch_samples_flat.view(-1, *original_shape)
                
                with torch.no_grad():
                    flat_qois = self.qoi_func(model_input, self.model).view(-1)
                
                qoi_accumulator.append(flat_qois.view(current_batch_size, N).cpu())

            all_qois = torch.cat(qoi_accumulator, dim=0).numpy()
            
            # --- CALCULATE PERCENTILES (25th and 75th) ---
            q1, q99 = np.percentile(all_qois, [1, 99], axis=0)
            
            profile[delta] = {
                "median": np.median(all_qois, axis=0),
                "min": np.min(all_qois, axis=0),
                "max": np.max(all_qois, axis=0),
                "std": np.std(all_qois, axis=0),
                "q1": q1, # Save Q1
                "q99": q99  # Save Q99
            }
        return profile

    # ==========================================
    # STEP 2: OPTIMAL DELTA SEARCH (Float32)
    # ==========================================
    def find_optimal_delta(self, stability_profile, tau_a):
        """
        Identify the optimal delta for each input.
        
        Logic:
        1. Look for the first delta where the ENTIRE remaining tail of the curve 
           stays within a variation of `tau_a`.
        2. Variation is defined as (Global Max of Tail Q99 - Global Min of Tail Q1).
           (Interquartile Range Stability).
        
        If no stable tail is found, defaults to the LARGEST delta (1.0).
        """
        sorted_deltas = sorted(stability_profile.keys())
        deltas = np.array(sorted_deltas)
        
        # Stack Percentiles for vectorized operations
        q1_stack = np.stack([stability_profile[d]['q1'] for d in sorted_deltas], axis=0)       # (T, N)
        q99_stack = np.stack([stability_profile[d]['q99'] for d in sorted_deltas], axis=0)       # (T, N)
        
        N = q1_stack.shape[1]
        
        # Default to the LARGEST delta (deltas[-1]). 
        optimal_deltas = np.full(N, deltas[-1], dtype=np.float32) 
        
        found_mask = np.zeros(N, dtype=bool)

        print(f"  [Step 2] Finding Optimal Delta per input (IQR Logic: Q99 - Q1)...")
        
        valid_matrix = np.zeros((len(deltas), N), dtype=bool)
        
        for j in range(len(deltas) - 1):
            # Look at the tail (future deltas)
            # We want the "Envelope" of the IQR for the rest of the curve
            tail_q99s = q99_stack[j+1:, :]
            tail_q1s = q1_stack[j+1:, :]
            
            # --- LOGIC CHANGE ---
            # Upper Bound of Envelope: Max of all future Q99s
            # Lower Bound of Envelope: Min of all future Q1s
            global_tail_q99 = np.max(tail_q99s, axis=0)
            global_tail_q1 = np.min(tail_q1s, axis=0)
            
            variation = global_tail_q99 - global_tail_q1
            
            # Strict amplitude check on the IQR envelope
            is_stable = (variation <= tau_a)
            valid_matrix[j, :] = is_stable

        for i in range(N):
            col_valid = valid_matrix[:, i]
            if np.any(col_valid):
                # Pick the FIRST index where stability starts
                first_stable_idx = np.argmax(col_valid)
                optimal_deltas[i] = deltas[first_stable_idx]
                found_mask[i] = True
            
        print(f"  [Step 2] Found stable delta for {np.sum(found_mask)}/{N} inputs.")
        return torch.from_numpy(optimal_deltas).to(dtype=torch.float32, device=self.device)

    # ==========================================
    # STEP 3A: FEATURE BATCHING (Mixed Precision)
    # ==========================================
    def compute_sensitivity(self, x, delta_star, n_w, batch_size):
        """
        Mixed Precision Feature Batching:
        - Input/Model: Float32
        - Accumulation/Division: Float64
        """
        x = x.to(dtype=torch.float32, device=self.device) 
        N = x.shape[0]
        original_shape = x.shape[1:]
        D = x[0].numel()
        
        # Bounds in Float32 for perturbation generation
        g_lower = self.global_lower.to(dtype=torch.float32)
        g_upper = self.global_upper.to(dtype=torch.float32)
        g_range = g_upper - g_lower
        
        if self.perturb_features is not None:
            active_indices = torch.as_tensor(self.perturb_features, device=self.device, dtype=torch.long)
            num_active = len(active_indices)
        else:
            active_indices = torch.arange(D, device=self.device, dtype=torch.long)
            num_active = D

        # Delta in Float32
        if isinstance(delta_star, torch.Tensor):
            delta_full = delta_star.to(dtype=torch.float32, device=self.device).view(N)
        elif isinstance(delta_star, np.ndarray):
            delta_full = torch.from_numpy(delta_star).to(dtype=torch.float32, device=self.device).view(N)
        else:
            delta_full = torch.full((N,), delta_star, dtype=torch.float32, device=self.device)
            
        print(f"  [Step 3] Sensitivity (Feature Batching, Mixed Precision): N={N}, D={D}, Batch={batch_size}")
        
        results_list = []

        for i in tqdm(range(N), desc="  Processing Inputs"):
            x_i_flat = x[i : i+1].reshape(1, D)
            delta_i = delta_full[i]
            
            # --- CRITICAL: Accumulator is Float64 ---
            # RENAME: Generic naming
            feature_sensitivity = torch.zeros(D, device=self.device, dtype=torch.float64)

            delta_range = delta_i * g_range.unsqueeze(0) 
            local_lower = torch.max(g_lower.unsqueeze(0), x_i_flat - delta_range)
            local_upper = torch.min(g_upper.unsqueeze(0), x_i_flat + delta_range)

            for w in range(n_w):
                rand_noise = torch.rand_like(x_i_flat)
                z_i = local_lower + rand_noise * (local_upper - local_lower)
                diff_i = z_i - x_i_flat 
                
                permuted_active_order = active_indices[torch.randperm(num_active, device=self.device)]
                current_x = x_i_flat.clone()
                
                # Model Inference in Float32
                with torch.no_grad():
                    model_in = current_x.view(1, *original_shape)
                    # Convert output immediately to Float64
                    last_qoi = self.qoi_func(model_in, self.model).to(dtype=torch.float64).item()
                
                num_chunks = int(np.ceil(num_active / batch_size))
                
                for k in range(num_chunks):
                    start_idx = k * batch_size
                    end_idx = min((k + 1) * batch_size, num_active)
                    actual_bs = end_idx - start_idx
                    
                    chunk_feat_indices = permuted_active_order[start_idx:end_idx]
                    chunk_deltas = diff_i[:, chunk_feat_indices] # Float32
                    
                    # Perturbation Updates (Float32)
                    updates = torch.zeros((actual_bs, 1, D), dtype=torch.float32, device=self.device)
                    rows = torch.arange(actual_bs, device=self.device)
                    updates[rows, 0, chunk_feat_indices] = chunk_deltas.squeeze(0)
                    
                    cumulative_changes = torch.cumsum(updates, dim=0)
                    batch_inputs_flat = current_x + cumulative_changes
                    model_inputs = batch_inputs_flat.view(actual_bs, *original_shape)
                    
                    # Model Inference (Float32) -> Cast to Float64
                    with torch.no_grad():
                        flat_outputs = self.qoi_func(model_inputs, self.model).to(dtype=torch.float64).view(-1)
                    
                    # --- CRITICAL: EE MATH IN FLOAT64 ---
                    prev_qois = torch.cat([torch.tensor([last_qoi], device=self.device, dtype=torch.float64), flat_outputs[:-1]])
                    
                    dy = flat_outputs - prev_qois
                    dx = chunk_deltas.view(-1).to(dtype=torch.float64) # Cast dx to 64
                    
                    valid_mask = (dx != 0)
                    ee = torch.zeros_like(dx) # Float64
                    if valid_mask.any():
                        ee[valid_mask] = torch.abs(dy[valid_mask] / dx[valid_mask])
                    
                    # RENAME: Generic update
                    feature_sensitivity.scatter_add_(0, chunk_feat_indices, ee)
                    
                    last_qoi = flat_outputs[-1].item()
                    current_x = batch_inputs_flat[-1]

            feature_sensitivity /= n_w
            results_list.append(feature_sensitivity.view(1, *original_shape))

        return torch.cat(results_list, dim=0)

    # ==========================================
    # STEP 3B: INPUT BATCHING (Mixed Precision)
    # ==========================================
    def compute_sensitivity_wide(self, x_batch, delta_star, n_w, input_batch_size):
        """
        Mixed Precision Input Batching.
        WARNING: Only efficient for Low-D data (Tabular/Genomics).
        Inefficient for Images (High-D).
        """
        N = x_batch.shape[0]
        original_shape = x_batch.shape[1:]
        D = x_batch[0].numel()
        
        g_lower = self.global_lower.to(dtype=torch.float32)
        g_upper = self.global_upper.to(dtype=torch.float32)
        g_range = g_upper - g_lower
        
        if self.perturb_features is not None:
            active_indices = torch.as_tensor(self.perturb_features, device=self.device, dtype=torch.long)
        else:
            active_indices = torch.arange(D, device=self.device, dtype=torch.long)

        if isinstance(delta_star, torch.Tensor):
            delta_full = delta_star.to(dtype=torch.float32, device=self.device).view(N)
        else:
            delta_full = torch.full((N,), delta_star, dtype=torch.float32, device=self.device)

        print(f"  [Step 3] Sensitivity (Input Batching, Mixed Precision): N={N}, D={D}, ChunkSize={input_batch_size}")
        
        results_list = []
        
        for i in range(0, N, input_batch_size):
            x_chunk = x_batch[i : i + input_batch_size] 
            delta_chunk = delta_full[i : i + input_batch_size].view(-1, 1) 
            chunk_bs = x_chunk.shape[0]
            
            # Working memory in Float32
            x_flat = x_chunk.reshape(chunk_bs, D).to(dtype=torch.float32, device=self.device)
            
            # Accumulator in Float64
            # RENAME: Generic
            feature_sensitivity = torch.zeros_like(x_flat, dtype=torch.float64)

            for w in tqdm(range(n_w), desc=f"    Batch {i}-{i+chunk_bs}", leave=False):
                delta_range = delta_chunk * g_range.unsqueeze(0)
                local_lower = torch.max(g_lower.unsqueeze(0), x_flat - delta_range)
                local_upper = torch.min(g_upper.unsqueeze(0), x_flat + delta_range)
                
                rand_noise = torch.rand_like(x_flat)
                z_batch = local_lower + rand_noise * (local_upper - local_lower)
                diff_batch = z_batch - x_flat
                
                permuted_active_order = active_indices[torch.randperm(len(active_indices), device=self.device)]
                current_x = x_flat.clone()
                
                with torch.no_grad():
                    model_input = current_x.view(chunk_bs, *original_shape)
                    # Output -> Float64
                    last_qois = self.qoi_func(model_input, self.model).to(dtype=torch.float64).view(-1)

                # WARNING: This loop length is D. Slow for images.
                for feat_idx in permuted_active_order:
                    current_x[:, feat_idx] = z_batch[:, feat_idx]
                    
                    model_input = current_x.view(chunk_bs, *original_shape)
                    with torch.no_grad():
                        new_qois = self.qoi_func(model_input, self.model).to(dtype=torch.float64).view(-1)
                    
                    dy = new_qois - last_qois
                    dx = diff_batch[:, feat_idx].to(dtype=torch.float64) # Float64
                    
                    valid_mask = (dx != 0)
                    ee = torch.zeros_like(dx)
                    if valid_mask.any():
                        ee[valid_mask] = torch.abs(dy[valid_mask] / dx[valid_mask])
                    
                    feature_sensitivity[:, feat_idx] += ee
                    last_qois = new_qois

            feature_sensitivity /= n_w
            results_list.append(feature_sensitivity.view(chunk_bs, *original_shape))
        
        return torch.cat(results_list, dim=0)

    # ==========================================
    # ORCHESTRATOR
    # ==========================================
    def run_full_analysis(self, x, config, save_path=None):
        # [NEW] Safety Check for Dimensions
        D_input = x[0].numel()
        D_bounds = self.global_lower.numel()
        if D_input != D_bounds:
            raise ValueError(f"Shape Mismatch: Input x has {D_input} features/pixels, "
                             f"but bounds file has {D_bounds}. Check your bounds files.")

        profile = self.compute_stability_profile(
            x, 
            deltas=config['stability_analysis']['deltas'],
            n_s=config['stability_analysis']['n_samples'],
            eval_batch_size=config['stability_analysis']['batch_size']
        )
        
        delta_star = self.find_optimal_delta(
            profile, 
            tau_a=config['optimal_delta_search']['tau_a']
        )
        
        if delta_star is None:
            return None

        print(delta_star)

        # [NEW] Auto-Guardrail for High Dimensionality
        # If input has > 10k features, FORCE Feature Batching to prevent hanging/OOM.
        if D_input > 10000:
            print(f"  [Auto-Switch] High dimensionality detected (D={D_input}). Forcing 'feature_batching'.")
            print(f"  [Reason] Input Batching is inefficient for D > 10,000 and may crash memory.")
            method = "feature_batching"
        else:
            method = config['sensitivity_analysis'].get('method', 'feature_batching')
        
        if method == "input_batching":
            sensitivity_map = self.compute_sensitivity_wide(
                x, 
                delta_star, 
                n_w=config['sensitivity_analysis']['n_trajectories'],
                input_batch_size=config['sensitivity_analysis']['batch_size'] 
            )
        else:
            sensitivity_map = self.compute_sensitivity(
                x, 
                delta_star, 
                n_w=config['sensitivity_analysis']['n_trajectories'],
                batch_size=config['sensitivity_analysis']['batch_size'] 
            )
        
        results = {
            'input': x.cpu(),
            'profile': profile,
            'delta_star': delta_star.cpu(),
            'sensitivity_map': sensitivity_map.cpu()
        }
        
        if save_path:
            with open(save_path, 'wb') as f:
                pickle.dump(results, f)
            print(f"Results saved to {save_path}")
            
        return results
